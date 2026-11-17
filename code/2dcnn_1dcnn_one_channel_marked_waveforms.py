"""Single-channel onset model.

The input is the one channel named on the heatmap mark, cropped to that
heatmap's own time window. A convolution reads the stripe. The onset is the
signal where a rising score turns on and stays on. Maybe is the 12 signals
before that signal, and Yes runs from it to the end.

Recordings with a blank channel cell are not used. The current all-channel
model remains curve_signal.py.

Run from the acquisition_progress folder:

    python tested_models/2D-CNN_1D-CNN_one_channel/channel_onset_model.py

That trains one model per setup, using only the other recordings of that
setup that share the clicked channel and heatmap window. Pass compare to
repeat the mixed-channel hidden test.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import curve_signal as curves

WAVE_LENGTH = 384
MODEL_PATH = curves.HERE / "channel_onset_model.pt"
SETUP_DIR = Path(__file__).resolve().parent / "setups"
MARK_WINDOWS = curves.DATA_ROOT / "wave _start_analysis" / "mark_windows.csv"
SETUP_EPOCHS = 30

# A setup is recordings of the same material and channel count. Two files are
# trained together only when they also name the same channel.
_SETUP_PREFIXES = (
    ("FA paste", "FA paste"),
    ("Senscomp_LC3", "Senscomp LC3"),
    ("Senscomp_OPC1L", "Senscomp OPC"),
    ("OPC1L 11 degree", "OPC 16ch"),
)

# Time window of each saved heatmap, in seconds. Measured from the figure axis.
WINDOWS = {
    "FA paste_WB0.3_20 degree_06.12.2025_ML": (2.157412e-04, 1.245434e-03),
    "FA paste_WB0.4_20 degree_06.16.2025_ML": (2.055556e-04, 1.442163e-03),
    "FA_WB0.3_Paste_25 degrees_8 channels_06.12_ML": (4.546861e-04, 1.135214e-03),
    "LC3 paste_WB0.3_20 degree_06.11.2025_ML": (4.154377e-04, 8.665573e-04),
    "OPC1L 11 degree_0.39_07.04.25_ML": (3.349757e-04, 6.382603e-04),
    "OPC1L 11 degree_0.5_06.27.25_ML": (2.312241e-04, 1.101571e-03),
    "Senscomp_LC3 paste_WB0.4_20 degree_06.08.2025_ML": (5.970532e-04, 1.235874e-03),
    "Senscomp_LC3 paste_WB0.5_20 degree_06.07.2025_ML": (3.313452e-04, 1.401073e-03),
    "Senscomp_OPC1L paste_WB0.4_20 degree_06.05.2025_ML": (2.044369e-04, 8.589394e-04),
    "Senscomp_OPC1L paste_WB0.5_20 degree_06.04.2025_ML": (5.549988e-04, 1.176540e-03),
}

# Same hidden recordings as the earlier tests, limited to those with a channel and a window.
HELD_OUT = {
    "FA paste_WB0.3_20 degree_06.12.2025_ML",
    "LC3 paste_WB0.3_20 degree_06.11.2025_ML",
    "OPC1L 11 degree_0.5_06.27.25_ML",
    "Senscomp_LC3 paste_WB0.5_20 degree_06.07.2025_ML",
}


def _resample(filtered: np.ndarray, gate0: int, gate1: int) -> np.ndarray:
    piece = filtered[:, gate0:gate1]
    source = np.linspace(0.0, 1.0, piece.shape[1])
    grid = np.linspace(0.0, 1.0, WAVE_LENGTH)
    waves = np.empty((piece.shape[0], WAVE_LENGTH), dtype=np.float32)
    for index, row in enumerate(piece):
        waves[index] = np.interp(grid, source, row).astype(np.float32)
    return waves


def prepare_channel(samples: np.ndarray, fs: float, window: tuple[float, float]) -> tuple[np.ndarray, float, float]:
    """One channel heatmap, shape (1, signals, samples), on the figure's time window."""
    array = np.asarray(samples, dtype=np.float64)
    sos = butter(4, list(curves.BAND_HZ), btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, array, axis=1)
    n_samples = filtered.shape[1]
    gate0 = min(n_samples - 8, max(0, int(round(window[0] * fs))))
    gate1 = min(n_samples, max(gate0 + 8, int(round(window[1] * fs))))
    waves = _resample(filtered, gate0, gate1)
    tail = waves[int(waves.shape[0] * 0.9) :]
    unit = float(np.median(np.max(np.abs(tail), axis=1)))
    waves /= max(unit, 1e-6)
    np.clip(waves, -4.0, 4.0, out=waves)
    return waves[None, :, :], gate0 / fs, (gate1 - 1) / fs


def load_marked_channels() -> list[dict]:
    """Recordings whose mark names a channel and whose heatmap window is known."""
    records = []
    for row in curves._marked_rows():
        experiment = row["experiment"]
        channel = row["channel"]
        if not channel or experiment not in WINDOWS:
            continue
        path = curves._mat_path(experiment)
        if path is None:
            print(f"No waveform file for {experiment}", flush=True)
            continue
        fs, channels = curves.load_waveforms(path)
        if channel not in channels:
            print(f"{experiment} has no channel {channel}", flush=True)
            continue
        waves, t0, t1 = prepare_channel(channels[channel], fs, WINDOWS[experiment])
        signal_id = int(row["signal_id"])
        n_signals = curves._signal_count(waves)
        mark = curves._point_of_interest(n_signals, signal_id)
        records.append(
            {
                "experiment": experiment,
                "channel": channel,
                "signal_id": signal_id,
                "waves": waves,
                "t0": t0,
                "t1": t1,
                "mark": mark,
                "time_fraction": curves._time_fraction(float(row["time_s"]), t0, t1),
                "time_s": float(row["time_s"]),
            }
        )
        print(
            f"  {experiment[:42]:42}  {channel:4}  {n_signals} signals  "
            f"mark {signal_id}  {t0:.6f}-{t1:.6f} s",
            flush=True,
        )
    return records


class ChannelOnsetModel(torch.nn.Module):
    """Find the signal where the stripe on one channel turns on."""

    def __init__(self) -> None:
        super().__init__()
        self.local = torch.nn.Sequential(
            torch.nn.Conv2d(2, 8, (7, 9), padding=(3, 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (7, 5), padding=(3, 2)),
            torch.nn.ReLU(),
        )
        self.score = torch.nn.Conv1d(8, 1, 31, padding=15)
        self.time_head = torch.nn.Linear(8, 1)

    def read(self, waves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Onset fraction, and the time fraction inside every signal."""
        stack = waves if waves.ndim == 3 else waves.unsqueeze(0)
        _channels, n_signals, _length = stack.shape
        early_n = max(1, min(20, n_signals // 10))
        changed = stack - stack[:, :early_n].mean(dim=1, keepdim=True)
        image = torch.stack([changed, changed.abs()], dim=1)
        mapped = self.local(image)
        pooled = mapped.mean(dim=3)
        raw = self.score(pooled).reshape(-1)
        level = torch.cumsum(torch.nn.functional.softplus(raw), dim=0)
        turned = torch.sigmoid(level - level.mean())
        previous = torch.cat([turned.new_zeros(1), turned[:-1]])
        mass = (turned - previous).clamp_min(0)
        mass = mass / mass.sum().clamp_min(1e-6)
        slots = torch.arange(n_signals, dtype=waves.dtype) / max(n_signals - 1, 1)
        onset = (mass * slots).sum()
        time_logits = self.time_head(mapped.squeeze(0).permute(1, 2, 0)).squeeze(-1)
        time_weights = torch.softmax(time_logits, dim=1)
        time_slots = torch.arange(time_logits.shape[1], dtype=waves.dtype) / max(time_logits.shape[1] - 1, 1)
        when = (time_weights * time_slots).sum(dim=1)
        return onset, when


def _loss(model: ChannelOnsetModel, record: dict) -> torch.Tensor:
    waves = torch.tensor(record["waves"])
    fraction, when = model.read(waves)
    target = float(record["mark"]) / max(curves._signal_count(waves) - 1, 1)
    onset_loss = (fraction - target).square()
    time_loss = (when[int(record["mark"])] - float(record["time_fraction"])).square()
    return onset_loss + 0.25 * time_loss


@torch.no_grad()
def predict(model: ChannelOnsetModel, record: dict) -> tuple[int, float]:
    """Return the 1-based curve signal and its start time in seconds."""
    model.eval()
    waves = torch.tensor(np.asarray(record["waves"], dtype=np.float32))
    n_signals = curves._signal_count(waves)
    fraction, when = model.read(waves)
    index = int(np.clip(np.floor(float(fraction) * (n_signals - 1) + 0.5), 0, n_signals - 1))
    start_s = float(record["t0"]) + float(when[index]) * (float(record["t1"]) - float(record["t0"]))
    return index + 1, start_s


def _errors(model: ChannelOnsetModel, records: list[dict]) -> tuple[float, float]:
    farthest = 0.0
    time_misses = []
    for record in records:
        appeared, start_s = predict(model, record)
        farthest = max(farthest, abs(appeared - int(record["signal_id"])))
        time_misses.append(abs(start_s - float(record["time_s"])))
    return farthest, float(np.mean(time_misses))


def train_model(records: list[dict], seed: int, epochs: int = 40) -> ChannelOnsetModel:
    torch.manual_seed(seed)
    model = ChannelOnsetModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    model.train()
    best_far = float("inf")
    best_time = float("inf")
    best_state = None
    for epoch in range(epochs):
        losses = []
        order = list(records)
        rng.shuffle(order)
        for record in order:
            optimizer.zero_grad()
            loss = _loss(model, record)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        farthest, mean_time = _errors(model, records)
        model.train()
        if farthest < best_far or (farthest == best_far and mean_time < best_time):
            best_far = farthest
            best_time = mean_time
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"    epoch {epoch + 1:3d}  loss {float(np.mean(losses)):.5f}  "
                f"farthest signal {farthest:.0f}  mean time miss {mean_time:.6f} s",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _report(title: str, rows: list[tuple[str, int, int, float, float]]) -> None:
    misses = [abs(predicted - target) for _name, target, predicted, _got, _true in rows]
    time_misses = [abs(got - true) for _name, _target, _predicted, got, true in rows]
    print(title, flush=True)
    for name, target, predicted, got, true in rows:
        print(
            f"  {name[:42]:42}  target {target:4d}  predicted {predicted:4d}  "
            f"miss {predicted - target:+4d}  time {got:.6f} vs {true:.6f}",
            flush=True,
        )
    ordered = sorted(misses)
    print(
        f"  mean abs miss {sum(misses) / len(misses):.1f}  "
        f"median {ordered[len(ordered) // 2]}  farthest {max(misses)}  "
        f"mean time miss {sum(time_misses) / len(time_misses):.6f} s",
        flush=True,
    )


def setup_name(experiment: str) -> str | None:
    """Material and channel-count family of a recording name."""
    for prefix, name in _SETUP_PREFIXES:
        if experiment.startswith(prefix):
            return name
    return None


def _usable_setups(records: list[dict]) -> dict[str, list[dict]]:
    """Setups with at least two recordings that share one channel."""
    grouped: dict[str, list[dict]] = {}
    for record in records:
        name = setup_name(record["experiment"])
        if name is not None:
            grouped.setdefault(name, []).append(record)
    usable = {}
    for name, members in grouped.items():
        channels = {item["channel"] for item in members}
        if len(members) >= 2 and len(channels) == 1:
            usable[name] = members
        elif len(channels) > 1:
            listed = ", ".join(sorted(channels))
            print(f"{name} names {listed}, so those files are not trained as one setup.", flush=True)
    return usable


def write_mark_windows(records: list[dict], usable: dict[str, list[dict]]) -> None:
    """Save the clicked channel and heatmap window beside each mark."""
    trained = {item["experiment"] for members in usable.values() for item in members}
    by_experiment = {item["experiment"]: item for item in records}
    rows = []
    for row in curves._marked_rows():
        experiment = row["experiment"]
        known = by_experiment.get(experiment)
        window = WINDOWS.get(experiment)
        rows.append(
            {
                "experiment": experiment,
                "channel": row["channel"],
                "signal_id": row["signal_id"],
                "time_s": row["time_s"],
                "window_start_s": "" if window is None else f"{window[0]:.9f}",
                "window_end_s": "" if window is None else f"{window[1]:.9f}",
                "setup": setup_name(experiment) or "",
                "same_setup_model": "yes" if experiment in trained else "",
            }
        )
        if known is None and not row["channel"]:
            print(f"  {experiment[:42]:42}  channel still blank", flush=True)
    MARK_WINDOWS.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "channel",
        "signal_id",
        "time_s",
        "window_start_s",
        "window_end_s",
        "setup",
        "same_setup_model",
    ]
    with MARK_WINDOWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {MARK_WINDOWS}", flush=True)


def _manifest() -> dict[str, dict[str, str]]:
    path = SETUP_DIR / "manifest.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["experiment"]: row for row in csv.DictReader(handle)}


def train_setups() -> None:
    """Train one onset model on the other files of each matching setup."""
    print("Channel and heatmap window", flush=True)
    records = load_marked_channels()
    usable = _usable_setups(records)
    write_mark_windows(records, usable)
    SETUP_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for name, members in usable.items():
        print(f"Setup {name}, channel {members[0]['channel']}", flush=True)
        for held in members:
            partners = [item for item in members if item["experiment"] != held["experiment"]]
            print(f"  hold out {held['experiment'][:48]}", flush=True)
            model = train_model(partners, seed=1, epochs=SETUP_EPOCHS)
            weights = f"{name.lower().replace(' ', '_')}_{len(manifest_rows)}.pt"
            torch.save({"state_dict": model.state_dict()}, SETUP_DIR / weights)
            window = WINDOWS[held["experiment"]]
            appeared, start_s = predict(model, held)
            print(
                f"    target {held['signal_id']}  predicted {appeared}  "
                f"miss {appeared - held['signal_id']:+d}  time {start_s:.6f} s",
                flush=True,
            )
            manifest_rows.append(
                {
                    "experiment": held["experiment"],
                    "setup": name,
                    "channel": held["channel"],
                    "window_start_s": f"{window[0]:.9f}",
                    "window_end_s": f"{window[1]:.9f}",
                    "weights": weights,
                }
            )
    fields = ["experiment", "setup", "channel", "window_start_s", "window_end_s", "weights"]
    with (SETUP_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {SETUP_DIR / 'manifest.csv'}", flush=True)


def predict_if_setup(
    experiment: str, channels: dict[str, np.ndarray], fs: float
) -> dict | None:
    """Predict with the model trained on the other recordings of this setup."""
    row = _manifest().get(experiment)
    if row is None or not row.get("weights"):
        return None
    channel = row["channel"]
    if channel not in channels:
        return None
    model = ChannelOnsetModel()
    blob = torch.load(SETUP_DIR / row["weights"], map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    window = (float(row["window_start_s"]), float(row["window_end_s"]))
    waves, t0, t1 = prepare_channel(channels[channel], fs, window)
    appeared, start_s = predict(model, {"waves": waves, "t0": t0, "t1": t1})
    return {
        "signal": appeared,
        "start_s": start_s,
        "channel": channel,
        "setup": row["setup"],
        "n_signals": curves._signal_count(waves),
    }


def compare() -> None:
    """Train this model and the current model on the same channel-labeled split."""
    print("Single-channel recordings", flush=True)
    marked = load_marked_channels()
    train = [record for record in marked if record["experiment"] not in HELD_OUT]
    held = [record for record in marked if record["experiment"] in HELD_OUT]
    print(f"Train {len(train)}  hold out {len(held)}", flush=True)
    if len(train) < 4 or len(held) < 2:
        raise SystemExit("Not enough channel-labeled recordings to compare.")

    print("Current heatmap model", flush=True)
    cached = curves.extract_records()
    train_names = {item["experiment"] for item in train}
    current, _loss = curves.train_model(
        [record for record in cached if record["experiment"] in train_names],
        seed=1,
        epochs=40,
    )
    current_rows = []
    cache_held = [record for record in cached if record["experiment"] in HELD_OUT]
    for record in cache_held:
        _labels, appeared, _probabilities, start_s = curves.predict_acquisition(
            current, record["waves"], record["t0"], record["t1"]
        )
        current_rows.append(
            (
                record["experiment"],
                int(record["signal_id"]),
                int(appeared),
                float(start_s),
                float(record["time_s"]),
            )
        )

    print("Single-channel onset model", flush=True)
    model = train_model(train, seed=1, epochs=40)
    torch.save({"state_dict": model.state_dict(), "windows": WINDOWS}, MODEL_PATH)
    single_rows = []
    for record in held:
        appeared, start_s = predict(model, record)
        single_rows.append(
            (record["experiment"], int(record["signal_id"]), appeared, start_s, float(record["time_s"]))
        )

    print()
    _report("Current heatmap model on hidden recordings", current_rows)
    _report("Single-channel onset model on hidden recordings", single_rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare()
    else:
        train_setups()
