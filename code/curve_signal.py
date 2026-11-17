"""Learn when the curve appears in an acquisition.

The network reads every channel in the same arrival band the heatmaps are
zoomed to. It keeps the channels separate, puts more weight on the channels
where a stripe grows, and predicts the signal where that stripe starts. It
also predicts the time, in seconds, on the first Yes signal.

The curve is the signal marked on the heatmap. Training uses that mark:
signals well before it are No, the short run just before it is Maybe, and
the marked signal onward is Yes. The time target is the marked time.

These files do not store a clock. On a logger that saves one signal per
minute, the reported signal number is the minute.

Run from this folder:

    python curve_signal.py
"""

from __future__ import annotations

import csv
import importlib.util
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent
DATA_DIR = HERE / "data"
LABELS_PATH = DATA_ROOT / "wave _start_analysis" / "poi_labels.csv"
# Google Drive download folders end with a stamp such as -20261107T212556Z-1-001.
_DOWNLOAD_STAMP = re.compile(r"-\d{8}T\d{6}Z-\d+-\d+$")
CACHE_PATH = HERE / "curve_waves.npz"
MODEL_PATH = HERE / "curve_model.pt"
OUT_CSV = HERE / "curve_onset.csv"

WAVE_VERSION = 9
WAVE_LENGTH = 512
BAND_HZ = (23_000.0, 73_000.0)
CLASSES = ("No", "Maybe", "Yes")
MAYBE_RUN = 12
EPOCHS = 60

# Header that starts a 1-by-N single-precision row inside a MAT workspace.
_ROW_SIG = (
    b"\x07\x00\x00\x00\x00\x00\x00\x00"
    b"\x05\x00\x00\x00\x08\x00\x00\x00"
    b"\x01\x00\x00\x00"
)


def writable_path(path: Path) -> Path:
    """Use path, or a sibling file if another program has the file open."""
    try:
        with path.open("a", encoding="utf-8"):
            return path
    except PermissionError:
        alternate = path.with_name(f"{path.stem}_new{path.suffix}")
        print(f"{path.name} is open. Writing {alternate.name} instead.")
        return alternate


def _extended(path: Path) -> str:
    """Windows path that can be opened when the name is longer than 260 characters."""
    text = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        return "\\\\?\\" + text
    return text


def _is_file(path: Path) -> bool:
    return os.path.isfile(_extended(path))


def _read_named(path: Path, wanted: set[str]) -> dict:
    found: dict = {}
    with open(_extended(path), "rb") as handle:
        reader = MatFile5Reader(handle, struct_as_record=False, squeeze_me=True)
        reader.mat_stream.seek(0)
        reader.initialize_read()
        reader.read_file_header()
        while not reader.end_of_stream() and wanted - found.keys():
            header, nxt = reader.read_var_header()
            name = "" if header.name is None else header.name.decode("latin1")
            if name == "":
                break
            if name in wanted:
                found[name] = reader.read_var_array(header, process=True)
            reader.mat_stream.seek(nxt)
    return found


def _workspace(path: Path) -> bytes:
    with open(_extended(path), "rb") as handle:
        reader = MatFile5Reader(handle)
        reader.mat_stream.seek(0)
        reader.initialize_read()
        reader.read_file_header()
        while not reader.end_of_stream():
            header, nxt = reader.read_var_header()
            name = "" if header.name is None else header.name.decode("latin1")
            if name == "":
                array = np.asarray(reader.read_var_array(header, process=False))
                return array.ravel().tobytes()
            reader.mat_stream.seek(nxt)
    raise ValueError(f"No waveform workspace in {path.name}")


def _channel_names(header: dict, n_channels: int) -> list[str]:
    for key in ("Tch", "Tch_light"):
        names = getattr(header.get(key), "_fieldnames", None)
        if names and len(names) == n_channels:
            return list(names)
    return [f"ai{i}" for i in range(1, n_channels + 1)]


def load_waveforms(path: Path) -> tuple[float, dict[str, np.ndarray]]:
    """Return the sample rate and each channel as a matrix of signals."""
    header = _read_named(path, {"fs", "Tch", "Tch_light"})
    if "fs" not in header:
        raise ValueError(f"'fs' missing in {path.name}")
    fs = float(np.asarray(header["fs"]).reshape(-1)[0])
    raw = _workspace(path)
    rows: list[tuple[int, int]] = []
    pos = 0
    while True:
        hit = raw.find(_ROW_SIG, pos)
        if hit < 0:
            break
        if hit + 40 <= len(raw):
            length = struct.unpack_from("<I", raw, hit + 20)[0]
            data_type, nbytes = struct.unpack_from("<II", raw, hit + 32)
            if (
                data_type == 7
                and nbytes == length * 4
                and 256 <= length <= 200_000
                and hit + 40 + nbytes <= len(raw)
            ):
                rows.append((hit + 40, length))
        pos = hit + 1
    if not rows:
        raise ValueError(f"No waveforms found in {path.name}")
    groups: list[list[tuple[int, int]]] = [[rows[0]]]
    for prev, nxt in zip(rows, rows[1:]):
        gap = nxt[0] - (prev[0] + prev[1] * 4)
        if gap > 200 or nxt[1] != prev[1]:
            groups.append([nxt])
        else:
            groups[-1].append(nxt)
    names = _channel_names(header, len(groups))
    channels: dict[str, np.ndarray] = {}
    for name, group in zip(names, groups):
        length = group[0][1]
        stacked = np.empty((len(group), length), dtype=np.float32)
        for index, (offset, _length) in enumerate(group):
            stacked[index] = np.frombuffer(raw, dtype="<f4", count=length, offset=offset)
        channels[name] = stacked
    return fs, channels


def _zoom_bounds(arrays: list[np.ndarray], fs: float) -> tuple[int, int]:
    """Samples covering the arrival band used on the heatmaps.

    Every saved click falls in this band, and the large electrical pulse
    after it stays outside. A new recording uses the same band.
    """
    n_samples = min(array.shape[1] for array in arrays)
    gate0 = min(n_samples - 8, max(0, int(round(0.18e-3 * fs))))
    gate1 = min(n_samples, max(gate0 + 8, int(round(1.75e-3 * fs))))
    return gate0, gate1


def _resample(filtered: np.ndarray, gate0: int, gate1: int) -> np.ndarray:
    piece = filtered[:, gate0:gate1]
    source = np.linspace(0.0, 1.0, piece.shape[1])
    grid = np.linspace(0.0, 1.0, WAVE_LENGTH)
    waves = np.empty((piece.shape[0], WAVE_LENGTH), dtype=np.float32)
    for index, row in enumerate(piece):
        waves[index] = np.interp(grid, source, row).astype(np.float32)
    return waves


def prepare_recording(channel_map: dict[str, np.ndarray], fs: float) -> tuple[np.ndarray, float, float]:
    """Every channel as one heatmap, shape (channels, signals, samples).

    Channels share the arrival band the heatmaps zoom into. Each channel is
    scaled by its own formed wave so a quiet channel can still show a stripe.
    """
    names = list(channel_map)
    raw = [np.asarray(channel_map[name], dtype=np.float64) for name in names]
    n_signals = min(array.shape[0] for array in raw)
    n_samples = min(array.shape[1] for array in raw)
    raw = [array[:n_signals, :n_samples] for array in raw]
    sos = butter(4, list(BAND_HZ), btype="band", fs=fs, output="sos")
    filtered = [sosfiltfilt(sos, array, axis=1) for array in raw]
    gate0, gate1 = _zoom_bounds(filtered, fs)
    stack = np.empty((len(filtered), n_signals, WAVE_LENGTH), dtype=np.float32)
    units = []
    for array in filtered:
        tail = array[int(n_signals * 0.9) :, gate0:gate1]
        units.append(float(np.median(np.max(np.abs(tail), axis=1))))
    floor = max(float(np.median(units)) * 0.05, 1e-6)
    for index, array in enumerate(filtered):
        stack[index] = _resample(array, gate0, gate1)
        stack[index] /= max(units[index], floor)
    np.clip(stack, -4.0, 4.0, out=stack)
    t0 = gate0 / fs
    t1 = (gate1 - 1) / fs
    return stack, t0, t1


def _signal_count(waves: np.ndarray | torch.Tensor) -> int:
    """Number of signals in a (signals, samples) or (channels, signals, samples) array."""
    return int(waves.shape[0] if waves.ndim == 2 else waves.shape[1])


def _experiment_key(name: str) -> str:
    """Folder name with a download stamp removed."""
    return _DOWNLOAD_STAMP.sub("", name)


def _walk_files(root: Path):
    """Files under root, including names longer than the normal Windows limit."""
    pending = [root]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(_extended(current)) as scan:
                entries = list(scan)
        except OSError:
            continue
        for entry in entries:
            child = current / entry.name
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                pending.append(child)
            else:
                yield child


def _mat_path(experiment: str) -> Path | None:
    """Find the waveform file under data/, at any folder depth."""
    filename = f"{experiment}.mat"
    matches = []
    if DATA_DIR.is_dir():
        matches = [path for path in _walk_files(DATA_DIR) if path.name == filename]
        matches.sort(key=lambda path: str(path).lower())
        if matches:
            return matches[0]
    for root in (DATA_ROOT, HERE):
        path = root / experiment / filename
        if _is_file(path):
            return path
    return None


def _marked_rows() -> list[dict]:
    """Every heatmap mark that names a signal. The channel cell may be empty."""
    rows = []
    with LABELS_PATH.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["signal_id"] and row["time_s"]:
                rows.append(row)
    return rows


def _point_of_interest(n_signals: int, signal_id: int) -> int:
    """Row index of the heatmap mark. Signal numbers start at 1."""
    return min(max(int(signal_id) - 1, 0), n_signals - 1)


def class_targets(n_signals: int, mark: int) -> np.ndarray:
    """No before the curve, Maybe on the short run just before it, then Yes."""
    labels = np.full(n_signals, 2, dtype=np.int64)
    labels[:mark] = 0
    labels[max(0, mark - MAYBE_RUN) : mark] = 1
    return labels


def _time_fraction(time_s: float, t0: float, t1: float) -> float:
    span = max(t1 - t0, 1e-9)
    return float(np.clip((time_s - t0) / span, 0.0, 1.0))


class CurveModel(torch.nn.Module):
    """Predict the target signal from the stripe on the channel heatmaps.

    The early rows of each channel are removed. A convolution reads the
    stripe across neighboring signals and time. Channels where that stripe
    grows get more weight. A second convolution scores the signal where the
    weighted stripe starts. The start time is read from that signal.
    """

    def __init__(self) -> None:
        super().__init__()
        self.local = torch.nn.Sequential(
            torch.nn.Conv2d(2, 8, (5, 9), padding=(2, 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), padding=(2, 2)),
            torch.nn.ReLU(),
        )
        self.time_pool = torch.nn.Conv2d(8, 1, (1, 7), padding=(0, 3))
        self.channel = torch.nn.Linear(8, 1)
        self.trend = torch.nn.Sequential(
            torch.nn.Conv1d(16, 16, 31, padding=15),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 1, 31, padding=15),
        )
        self.time_head = torch.nn.Linear(8, 1)

    def _stack(self, waves: torch.Tensor) -> torch.Tensor:
        """(signals, samples) or (channels, signals, samples) -> (channels, signals, samples)."""
        if waves.ndim == 2:
            return waves.unsqueeze(0)
        return waves

    def _maps(self, stack: torch.Tensor) -> torch.Tensor:
        """Stripe features, shape (channels, features, signals, time)."""
        _channels, n_signals, _length = stack.shape
        early_n = max(1, min(20, n_signals // 10))
        changed = stack - stack[:, :early_n].mean(dim=1, keepdim=True)
        image = torch.stack([changed, changed.abs()], dim=1)
        return self.local(image)

    def _channel_weights(self, mapped: torch.Tensor) -> torch.Tensor:
        """More weight on channels whose stripe is stronger at the end."""
        _channels, _features, n_signals, _length = mapped.shape
        early_n = max(1, min(20, n_signals // 10))
        pooled = mapped.mean(dim=3)
        early = pooled[:, :, :early_n].mean(dim=2)
        late = pooled[:, :, int(n_signals * 0.85) :].mean(dim=2)
        scores = self.channel(late - early).reshape(-1)
        return torch.softmax(scores, dim=0)

    def read(self, waves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Onset fraction, and the time fraction of every signal."""
        stack = self._stack(waves)
        mapped = self._maps(stack)
        channel_w = self._channel_weights(mapped)
        time_weight = torch.softmax(self.time_pool(mapped), dim=3)
        pooled = (mapped * time_weight).sum(dim=3)
        mixed = (pooled * channel_w[:, None, None]).sum(dim=0).transpose(0, 1)
        delta = torch.zeros_like(mixed)
        delta[1:] = mixed[1:] - mixed[:-1]
        features = torch.cat([mixed, delta], dim=1)
        center = features.mean(dim=0, keepdim=True)
        scale = features.std(dim=0, keepdim=True).clamp_min(1e-3)
        features = (features - center) / scale
        scores = self.trend(features.transpose(0, 1).unsqueeze(0)).reshape(-1)
        weights = torch.softmax(scores, dim=0)
        n_signals = scores.shape[0]
        slots = torch.arange(n_signals, dtype=waves.dtype) / max(n_signals - 1, 1)
        onset = (weights * slots).sum()
        logits = self.time_head(mapped.permute(0, 2, 3, 1)).squeeze(-1)
        mixed_time = (logits * channel_w[:, None, None]).sum(dim=0)
        time_weights = torch.softmax(mixed_time, dim=1)
        time_slots = torch.arange(mixed_time.shape[1], dtype=waves.dtype) / max(mixed_time.shape[1] - 1, 1)
        when = (time_weights * time_slots).sum(dim=1)
        return onset, when

    def onset_fraction(self, waves: torch.Tensor) -> torch.Tensor:
        """Where the curve starts, as a fraction of the acquisition."""
        onset, _when = self.read(waves)
        return onset

    def time_fraction(self, waves: torch.Tensor, index: int) -> torch.Tensor:
        """Where the wave sits inside one signal, as a fraction of its window."""
        _onset, when = self.read(waves)
        return when[index]


def _record_loss(model: CurveModel, record: dict) -> torch.Tensor:
    waves = torch.tensor(record["waves"])
    fraction, when = model.read(waves)
    target = float(record["mark"]) / max(_signal_count(waves) - 1, 1)
    onset_loss = (fraction - target).square()
    time_loss = (when[record["mark"]] - record["time_fraction"]).square()
    return onset_loss + 0.25 * time_loss


def _errors(model: CurveModel, records: list[dict]) -> tuple[float, float]:
    """Farthest signal miss, and the mean absolute time miss in seconds."""
    farthest = 0.0
    time_misses = []
    with torch.no_grad():
        for record in records:
            _labels, appeared, _probabilities, start_s = predict_acquisition(
                model, record["waves"], record["t0"], record["t1"]
            )
            if appeared is None:
                farthest = max(farthest, float(_signal_count(record["waves"])))
            else:
                farthest = max(farthest, abs(appeared - record["signal_id"]))
            if start_s is not None:
                time_misses.append(abs(start_s - record["time_s"]))
    mean_time = float(np.mean(time_misses)) if time_misses else float("nan")
    return farthest, mean_time


def train_model(records: list[dict], seed: int, epochs: int = EPOCHS) -> tuple[CurveModel, float]:
    torch.manual_seed(seed)
    model = CurveModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    model.train()
    last = float("nan")
    best_far = float("inf")
    best_time = float("inf")
    best_state = None
    for epoch in range(epochs):
        losses = []
        order = list(records)
        rng.shuffle(order)
        for record in order:
            optimizer.zero_grad()
            loss = _record_loss(model, record)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        last = float(np.mean(losses))
        farthest, mean_time = _errors(model, records)
        model.train()
        better = farthest < best_far or (farthest == best_far and mean_time < best_time)
        if better:
            best_far = farthest
            best_time = mean_time
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"    epoch {epoch + 1:3d}  loss {last:.5f}  "
                f"farthest signal {farthest:.0f}  mean time miss {mean_time:.6f} s",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, last


@torch.no_grad()
def predict_acquisition(
    model: CurveModel, waves: np.ndarray, t0: float, t1: float
) -> tuple[np.ndarray, int | None, np.ndarray, float | None]:
    """Label every signal and report the learned onset signal and time."""
    model.eval()
    stored = torch.tensor(np.asarray(waves, dtype=np.float32))
    n = _signal_count(stored)
    with torch.no_grad():
        share_tensor, when = model.read(stored)
    share = float(share_tensor)
    index = int(np.clip(np.floor(share * (n - 1) + 0.5), 0, n - 1))
    appeared = index + 1
    numbered = class_targets(n, index)
    labels = np.array([CLASSES[int(item)] for item in numbered], dtype=object)
    probabilities = np.eye(3, dtype=np.float32)[numbered]
    time_share = float(when[index])
    start_s = float(t0) + time_share * (float(t1) - float(t0))
    return labels, appeared, probabilities, start_s


def signal_partitions(labels: np.ndarray) -> dict[str, str]:
    """Signal numbers in each class, as ranges. Signal 1 is the first row."""
    grouped = {name: [] for name in CLASSES}
    start = 0
    names = [str(label) for label in labels]
    for index in range(1, len(names) + 1):
        if index == len(names) or names[index] != names[start]:
            first = start + 1
            last = index
            grouped[names[start]].append(str(first) if first == last else f"{first}-{last}")
            start = index
    return {name: ", ".join(grouped[name]) for name in CLASSES}


def _finish_record(record: dict) -> dict:
    n_signals = _signal_count(record["waves"])
    mark = _point_of_interest(n_signals, record["signal_id"])
    record["mark"] = mark
    record["labels"] = class_targets(n_signals, mark)
    record["time_fraction"] = _time_fraction(record["time_s"], record["t0"], record["t1"])
    return record


def _load_cache() -> list[dict] | None:
    if not CACHE_PATH.is_file():
        return None
    blob = np.load(CACHE_PATH, allow_pickle=True)
    if int(np.asarray(blob["version"]).reshape(-1)[0]) != WAVE_VERSION:
        return None
    records = []
    for experiment, channel, signal_id, fs, waves, t0, t1, time_s in zip(
        blob["experiment"],
        blob["channel"],
        blob["signal_id"],
        blob["fs"],
        blob["waves"],
        blob["t0"],
        blob["t1"],
        blob["time_s"],
    ):
        records.append(
            _finish_record(
                {
                    "experiment": str(experiment),
                    "channel": str(channel),
                    "signal_id": int(signal_id),
                    "fs": float(fs),
                    "waves": np.asarray(waves, dtype=np.float32),
                    "t0": float(t0),
                    "t1": float(t1),
                    "time_s": float(time_s),
                }
            )
        )
    return records


def _save_cache(records: list[dict]) -> None:
    np.savez_compressed(
        CACHE_PATH,
        version=np.array([WAVE_VERSION]),
        experiment=np.array([record["experiment"] for record in records], dtype=object),
        channel=np.array([record["channel"] for record in records], dtype=object),
        signal_id=np.array([record["signal_id"] for record in records]),
        fs=np.array([record["fs"] for record in records]),
        waves=np.array([record["waves"] for record in records], dtype=object),
        t0=np.array([record["t0"] for record in records]),
        t1=np.array([record["t1"] for record in records]),
        time_s=np.array([record["time_s"] for record in records]),
    )


def extract_records() -> list[dict]:
    cached = _load_cache()
    if cached:
        print(f"Using waveforms in {CACHE_PATH.name}")
        return cached
    records = []
    for row in _marked_rows():
        experiment = row["experiment"]
        mat_path = _mat_path(experiment)
        if mat_path is None:
            print(f"No waveform file for {experiment}")
            continue
        print(f"Reading {experiment}")
        fs, channels = load_waveforms(mat_path)
        names = list(channels)
        waves, t0, t1 = prepare_recording(channels, fs)
        del channels
        signal_id = int(row["signal_id"])
        records.append(
            _finish_record(
                {
                    "experiment": experiment,
                    "channel": ", ".join(names),
                    "signal_id": signal_id,
                    "fs": fs,
                    "waves": waves,
                    "t0": t0,
                    "t1": t1,
                    "time_s": float(row["time_s"]),
                }
            )
        )
        print(
            f"  {len(names)} channels  {_signal_count(waves)} signals  "
            f"marked signal {signal_id}  {t0:.6f}-{t1:.6f} s"
        )
    if len(records) < 4:
        raise SystemExit(f"Only {len(records)} acquisitions could be read. Not enough to train.")
    _save_cache(records)
    print(f"Wrote {CACHE_PATH}")
    return records


def measure_accuracy(model: CurveModel, records: list[dict]) -> None:
    """Signal match, per-signal labels, and time error against the heatmap marks."""
    hits = 0
    row_hits = 0
    row_total = 0
    time_misses = []
    for record in records:
        labels, appeared, _probabilities, start_s = predict_acquisition(
            model, record["waves"], record["t0"], record["t1"]
        )
        hits += int(appeared == record["signal_id"])
        predicted = np.array([CLASSES.index(str(name)) for name in labels])
        row_hits += int((predicted == record["labels"]).sum())
        row_total += int(record["labels"].size)
        if start_s is not None:
            time_misses.append(abs(start_s - record["time_s"]))
    n = len(records)
    print(f"Curve-signal accuracy {hits}/{n} = {hits / n:.0%}")
    print(f"Per-signal accuracy {row_hits}/{row_total} = {row_hits / row_total:.0%}")
    if time_misses:
        print(f"Mean time miss {float(np.mean(time_misses)):.6f} s")


def _report(
    name: str,
    channel: str,
    target: int | None,
    labels: np.ndarray,
    appeared: int | None,
    start_s: float | None,
) -> None:
    parts = signal_partitions(labels)
    when = appeared if appeared else "not yet"
    print(name)
    if channel:
        count = channel.count(",") + 1 if "," in channel else 1
        print(f"  channels {count}: {channel}")
    if target is not None:
        print(f"  target {target}")
    print(f"  curve at signal {when}")
    if start_s is None:
        print("  Yes starts at an unknown time")
    else:
        print(f"  Yes starts at {start_s:.6f} s")
    print(f"  No    {parts['No']}")
    print(f"  Maybe {parts['Maybe']}")
    print(f"  Yes   {parts['Yes']}")


def save_table(rows: list[dict]) -> None:
    path = writable_path(OUT_CSV)
    fields = ["experiment", "target_signal", "curve_signal", "yes_start_s", "no_signals", "maybe_signals", "yes_signals"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def judge_acquisition(channels: dict[str, np.ndarray] | np.ndarray, fs: float, blob: dict | None = None) -> dict:
    """Label each signal of a recording and report when the curve starts.

    channels is every channel in the recording. A single voltage matrix is
    treated as one channel.
    """
    if isinstance(channels, np.ndarray):
        channels = {"ai1": channels}
    if blob is None:
        blob = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = CurveModel()
    model.load_state_dict(blob["state_dict"])
    waves, t0, t1 = prepare_recording(channels, fs)
    labels, appeared, probabilities, start_s = predict_acquisition(model, waves, t0, t1)
    return {
        "labels": labels,
        "appearance_signal": appeared,
        "yes_start_s": start_s,
        "probability": probabilities,
    }


def _recording_for_image(path: Path) -> Path | None:
    """Find the voltage file that belongs with a heatmap or acquisition picture."""
    text = str(path)
    for row in _marked_rows():
        experiment = row["experiment"]
        if experiment in text:
            mat = _mat_path(experiment)
            if mat is not None:
                return mat
    folder = path.parent
    mat = folder / f"{_experiment_key(folder.name)}.mat"
    if _is_file(mat):
        return mat
    named = _mat_path(_experiment_key(folder.name))
    if named is not None:
        return named
    return None


def _known_experiment(path: Path) -> str | None:
    """Experiment name when the path contains one of the saved marks."""
    text = str(path)
    for row in _marked_rows():
        if row["experiment"] in text:
            return row["experiment"]
    key = _experiment_key(path.stem)
    for row in _marked_rows():
        if row["experiment"] == key:
            return key
    return None


def _setup_onset(experiment: str, channels: dict[str, np.ndarray], fs: float) -> dict | None:
    """One-channel model trained on the other files of this setup, when one exists."""
    script = HERE / "tested_models" / "2D-CNN_1D-CNN_one_channel" / "channel_onset_model.py"
    if not script.is_file():
        return None
    module = sys.modules.get("channel_onset_setup")
    if module is None:
        spec = importlib.util.spec_from_file_location("channel_onset_setup", script)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules["channel_onset_setup"] = module
        spec.loader.exec_module(module)
    return module.predict_if_setup(experiment, channels, fs)


def _test_recording(path: Path) -> None:
    fs, channels = load_waveforms(path)
    experiment = _known_experiment(path)
    setup = _setup_onset(experiment, channels, fs) if experiment else None
    if setup is not None:
        index = int(setup["signal"]) - 1
        numbered = class_targets(int(setup["n_signals"]), index)
        labels = np.array([CLASSES[int(item)] for item in numbered], dtype=object)
        print(
            f"2D-CNN + 1D-CNN trained on the other {setup['setup']} recordings, channel {setup['channel']}"
        )
        _report(path.name, setup["channel"], None, labels, int(setup["signal"]), float(setup["start_s"]))
        return
    if experiment:
        print("No same-setup channel model for this recording. Using the all-channel model.")
    names = ", ".join(channels)
    result = judge_acquisition(channels, fs)
    _report(
        path.name,
        names,
        None,
        result["labels"],
        result["appearance_signal"],
        result["yes_start_s"],
    )


def _test_path(path: Path) -> None:
    path = path.expanduser()
    if not os.path.exists(_extended(path)):
        print(f"Nothing found at {path}")
        return
    if os.path.isdir(_extended(path)):
        mats = sorted(path for path in _walk_files(path) if path.suffix.lower() == ".mat")
        if len(mats) == 1:
            path = mats[0]
        else:
            named = _mat_path(_experiment_key(path.name))
            if named is None:
                print("Paste the path of the .mat recording or the acquisition image.")
                return
            path = named
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        found = _recording_for_image(path)
        if found is None:
            print("That image has no recording next to it. Paste the .mat path.")
            return
        _test_recording(found)
        return
    if suffix == ".mat":
        _test_recording(path)
        return
    print("Paste a .mat recording or an acquisition image.")


def test_pasted_path() -> None:
    """Ask for a new recording or acquisition image and name its signals."""
    if not sys.stdin.isatty():
        return
    print()
    print("Paste a path to a new .mat recording or an acquisition image.")
    print("Press Enter with nothing on the line to stop.")
    while True:
        try:
            pasted = input("Path: ").strip().strip('"').strip("'")
        except EOFError:
            return
        if not pasted:
            return
        _test_path(Path(pasted))


def main() -> None:
    records = extract_records()
    n_weights = sum(parameter.numel() for parameter in CurveModel().parameters())
    print(f"Network size {n_weights} weights")
    print("Training on every acquisition")
    model, loss = train_model(records, seed=1)
    torch.save(
        {"state_dict": model.state_dict(), "wave_length": WAVE_LENGTH, "classes": CLASSES},
        MODEL_PATH,
    )
    print(f"Wrote {MODEL_PATH.name}   final loss {loss:.3f}")

    table = []
    for record in records:
        labels, appeared, _probabilities, start_s = predict_acquisition(
            model, record["waves"], record["t0"], record["t1"]
        )
        parts = signal_partitions(labels)
        _report(record["experiment"], record["channel"], record["signal_id"], labels, appeared, start_s)
        table.append(
            {
                "experiment": record["experiment"],
                "target_signal": record["signal_id"],
                "curve_signal": "" if appeared is None else appeared,
                "yes_start_s": "" if start_s is None else f"{start_s:.6f}",
                "no_signals": parts["No"],
                "maybe_signals": parts["Maybe"],
                "yes_signals": parts["Yes"],
            }
        )
    save_table(table)
    measure_accuracy(model, records)
    test_pasted_path()


if __name__ == "__main__":
    main()
