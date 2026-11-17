"""1D-CNN on the first-arrival curve of one channel.

Each channel is reduced to two series: when the wave first rises inside the
heatmap window, and how strong that signal is. A convolution reads those
series and names the signal where the arrival leaves its early plateau.

Trained on the marked training channels. The held-out channels are scored
after training.

Run from the acquisition_progress folder:

    python tested_models/1D-CNN_first_arrival/arrival_model.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import curve_signal as curves

HERE = Path(__file__).resolve().parent
DATA = curves.HERE / "heatmaps" / "ChannelHeatmaps" / "data"
LABELS = DATA / "lr_wave_onsets.csv"
SPLIT = DATA / "split.csv"
MODEL_PATH = HERE / "arrival_model.pt"
TRAIN_CSV = HERE / "train_predictions.csv"
TEST_CSV = HERE / "test_predictions.csv"
EPOCHS = 40
SEED = 1

# Heatmap windows measured from the figure axis. Recordings without a saved
# window use the same default the channel figures were drawn with.
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
DEFAULT_WINDOW = (0.0, 2.0e-3)


def splits() -> dict[str, str]:
    found = {}
    with SPLIT.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "marked":
                found[row["experiment"]] = row["split"]
    return found


def _smooth_rows(values: np.ndarray, width: int) -> np.ndarray:
    pad = width // 2
    padded = np.pad(values, ((0, 0), (pad, width - 1 - pad)), mode="edge")
    total = np.cumsum(padded, axis=1)
    total = np.pad(total, ((0, 0), (1, 0)))
    return (total[:, width:] - total[:, :-width]) / width


def arrival_series(samples: np.ndarray, fs: float, window: tuple[float, float]) -> np.ndarray:
    """Shape (2, signals): arrival shift from the early plateau, and relative strength."""
    array = np.asarray(samples, dtype=np.float64)
    sos = butter(4, [23_000.0, 73_000.0], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, array, axis=1)
    n_samples = filtered.shape[1]
    gate0 = min(n_samples - 8, max(0, int(round(window[0] * fs))))
    gate1 = min(n_samples, max(gate0 + 8, int(round(window[1] * fs))))
    piece = filtered[:, gate0:gate1]
    envelope = _smooth_rows(np.abs(hilbert(piece, axis=1)), 9)
    late = envelope[int(envelope.shape[0] * 0.75) :]
    level = 0.25 * float(np.median(np.max(late, axis=1))) + 1e-9
    above = envelope > level
    first = above.argmax(axis=1).astype(np.float64)
    first[~above.any(axis=1)] = piece.shape[1]
    first /= piece.shape[1]
    trend = np.convolve(first, np.ones(15) / 15, mode="same")
    early_n = max(12, trend.size // 8)
    baseline = float(np.percentile(trend[: max(early_n, trend.size // 3)], 80))
    shift = (trend - baseline) / (float(np.percentile(np.abs(trend - baseline), 90)) + 1e-3)
    strength = np.sqrt(np.mean(piece ** 2, axis=1))
    strength = strength / (float(np.median(strength[int(strength.size * 0.7) :])) + 1e-9)
    strength = np.convolve(strength, np.ones(11) / 11, mode="same")
    return np.stack([shift, strength]).astype(np.float32)


def load_records(which: str) -> list[dict]:
    group = splits()
    rows = []
    with LABELS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if group.get(row["dataset"]) == which and row["status"] == "yes" and row["signal_id"]:
                rows.append(row)
    by_experiment: dict[str, list[dict]] = {}
    for row in rows:
        by_experiment.setdefault(row["dataset"], []).append(row)
    records = []
    for experiment, items in by_experiment.items():
        mat_path = DATA / "marked" / which / experiment / f"{experiment}.mat"
        if not curves._is_file(mat_path):
            mat_path = curves._mat_path(experiment)
        if mat_path is None:
            print(f"No waveform file for {experiment}", flush=True)
            continue
        print(f"Reading {which:5}  {experiment}", flush=True)
        fs, channels = curves.load_waveforms(mat_path)
        window = WINDOWS.get(experiment, DEFAULT_WINDOW)
        for row in items:
            channel = row["channel"]
            if channel not in channels:
                print(f"  missing channel {channel}", flush=True)
                continue
            series = arrival_series(channels[channel], fs, window)
            signal_id = int(row["signal_id"])
            n_signals = int(series.shape[1])
            records.append(
                {
                    "experiment": experiment,
                    "channel": channel,
                    "signal_id": signal_id,
                    "mark": min(max(signal_id - 1, 0), n_signals - 1),
                    "series": series,
                    "n_signals": n_signals,
                }
            )
        del channels
    print(f"{which}: {len(records)} marked channels", flush=True)
    return records


class ArrivalModel(torch.nn.Module):
    """Read the arrival curve and name the signal where it turns."""

    def __init__(self) -> None:
        super().__init__()
        self.score = torch.nn.Sequential(
            torch.nn.Conv1d(2, 16, 21, padding=10),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 16, 15, padding=7),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 1, 31, padding=15),
        )

    def fraction(self, series: torch.Tensor) -> torch.Tensor:
        raw = self.score(series.unsqueeze(0)).reshape(-1)
        level = torch.cumsum(torch.nn.functional.softplus(raw), dim=0)
        turned = torch.sigmoid(level - level.mean())
        previous = torch.cat([turned.new_zeros(1), turned[:-1]])
        mass = (turned - previous).clamp_min(0)
        mass = mass / mass.sum().clamp_min(1e-6)
        slots = torch.arange(series.shape[1], dtype=series.dtype) / max(series.shape[1] - 1, 1)
        return (mass * slots).sum()


def predict_signal(model: ArrivalModel, record: dict) -> int:
    model.eval()
    with torch.no_grad():
        share = float(model.fraction(torch.tensor(record["series"])))
    n_signals = record["n_signals"]
    index = int(np.clip(np.floor(share * (n_signals - 1) + 0.5), 0, n_signals - 1))
    return index + 1


def mean_miss(model: ArrivalModel, records: list[dict]) -> float:
    misses = [abs(predict_signal(model, record) - record["signal_id"]) for record in records]
    return float(np.mean(misses))


def train_model(records: list[dict]) -> ArrivalModel:
    torch.manual_seed(SEED)
    model = ArrivalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(SEED)
    best_mean = float("inf")
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        order = list(records)
        rng.shuffle(order)
        losses = []
        for record in order:
            optimizer.zero_grad()
            share = model.fraction(torch.tensor(record["series"]))
            target = float(record["mark"]) / max(record["n_signals"] - 1, 1)
            loss = (share - target).square()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        current = mean_miss(model, records)
        if current < best_mean:
            best_mean = current
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"    epoch {epoch + 1:3d}  loss {float(np.mean(losses)):.5f}  "
                f"train mean miss {current:.1f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    return model


def score(model: ArrivalModel, records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        appeared = predict_signal(model, record)
        rows.append(
            {
                "experiment": record["experiment"],
                "channel": record["channel"],
                "target_signal": record["signal_id"],
                "predicted_signal": appeared,
                "miss": appeared - record["signal_id"],
            }
        )
    return rows


def summarize(title: str, rows: list[dict]) -> None:
    misses = [abs(int(row["miss"])) for row in rows]
    ordered = sorted(misses)
    signed = [int(row["miss"]) for row in rows]
    print(title, flush=True)
    print(
        f"  channels {len(rows)}  mean miss {sum(misses) / len(misses):.1f}  "
        f"median {ordered[len(ordered) // 2]}  farthest {max(misses)}  "
        f"signed mean {sum(signed) / len(signed):+.1f}",
        flush=True,
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(int(row["miss"]))
    for name, values in grouped.items():
        absolute = [abs(value) for value in values]
        print(
            f"  {name[:52]:52} mean {sum(absolute) / len(absolute):5.1f}  "
            f"signed {sum(values) / len(values):+6.1f}",
            flush=True,
        )


def write_table(path: Path, rows: list[dict]) -> None:
    fields = ["experiment", "channel", "target_signal", "predicted_signal", "miss"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


def main() -> None:
    train_records = load_records("train")
    test_records = load_records("test")
    print("Training 1D-CNN on the first-arrival curve", flush=True)
    model = train_model(train_records)
    torch.save({"state_dict": model.state_dict(), "seed": SEED, "epochs": EPOCHS}, MODEL_PATH)
    print(f"Wrote {MODEL_PATH}", flush=True)
    train_rows = score(model, train_records)
    test_rows = score(model, test_records)
    summarize("Training channels", train_rows)
    summarize("New recordings, held out", test_rows)
    write_table(TRAIN_CSV, train_rows)
    write_table(TEST_CSV, test_rows)


if __name__ == "__main__":
    main()
