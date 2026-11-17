"""Train on the unmarked heatmap pictures.

The marked pictures have the click drawn on them, so this model reads the
unmarked picture of the same channel. The click in lr_wave_onsets.csv is the
target. Whole recordings stay in the existing train or test split.

Run from the acquisition_progress folder:

    python tested_models/heatmap_image/heatmap_image_model.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
DATA = ROOT / "heatmaps" / "ChannelHeatmaps" / "data"
LABELS = DATA / "lr_wave_onsets.csv"
SPLIT = DATA / "split.csv"
MODEL_PATH = HERE / "heatmap_image_model.pt"
TRAIN_CSV = HERE / "train_predictions.csv"
TEST_CSV = HERE / "test_predictions.csv"
HEIGHT = 224
WIDTH = 160
EPOCHS = 30
SEED = 1


def splits() -> dict[str, str]:
    found = {}
    with SPLIT.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "original":
                found[row["experiment"]] = row["split"]
    return found


def crop_heatmap(image: np.ndarray) -> np.ndarray:
    """Keep the colored panel and drop the title, tick labels, and colorbar."""
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    hot = (
        ((red > 0.45) & (red > blue + 0.12))
        | ((red > 0.65) & (green > 0.55) & (blue < 0.55))
        | ((red > 0.12) & (red < 0.5) & (green < 0.25) & (blue < 0.25) & (red > green + 0.05))
    )
    rows = np.flatnonzero(hot.any(axis=1))
    cols = np.flatnonzero(hot.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("no heatmap colors found")
    panel = image[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]
    white = panel.mean(axis=2) > 0.92
    fraction = white.mean(axis=0)
    cut = panel.shape[1]
    for column in range(panel.shape[1] - 1, int(panel.shape[1] * 0.7), -1):
        if fraction[column] > 0.55:
            edge = column
            while edge > 0 and fraction[edge] > 0.45:
                edge -= 1
            if column - edge > 4 and (panel.shape[1] - column) < panel.shape[1] * 0.12:
                cut = edge
                break
    panel = panel[:, :cut]
    white = panel.mean(axis=2) > 0.92
    keep_rows = np.flatnonzero(~white.all(axis=1))
    keep_cols = np.flatnonzero(~white.all(axis=0))
    return panel[int(keep_rows[0]) : int(keep_rows[-1]) + 1, int(keep_cols[0]) : int(keep_cols[-1]) + 1]


def load_picture(path: Path) -> np.ndarray:
    image = plt.imread(path)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = image[..., :3]
    if float(image.max()) > 1.5:
        image = image / 255.0
    panel = crop_heatmap(image)
    # axis xy draws signal 1 at the bottom. Flip so row 0 is signal 1.
    panel = np.flipud(panel)
    row_index = np.linspace(0, panel.shape[0] - 1, HEIGHT).astype(int)
    col_index = np.linspace(0, panel.shape[1] - 1, WIDTH).astype(int)
    fitted = panel[row_index][:, col_index]
    return np.moveaxis(fitted, 2, 0).astype(np.float32)


def load_records(which: str) -> list[dict]:
    group = splits()
    records = []
    with LABELS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            experiment = row["dataset"]
            if group.get(experiment) != which or row["status"] != "yes" or not row["signal_id"]:
                continue
            path = DATA / "original" / which / experiment / f"{row['channel']}.png"
            if not path.is_file():
                print(f"No heatmap picture for {experiment} {row['channel']}", flush=True)
                continue
            picture = load_picture(path)
            signal_id = int(float(row["signal_id"]))
            n_signals = int(float(row["n_signals"]))
            records.append(
                {
                    "experiment": experiment,
                    "channel": row["channel"],
                    "signal_id": signal_id,
                    "n_signals": n_signals,
                    "mark": min(max(signal_id - 1, 0), n_signals - 1),
                    "picture": picture,
                }
            )
    print(f"{which}: {len(records)} heatmap pictures", flush=True)
    return records


class HeatmapImageModel(torch.nn.Module):
    """Read the heatmap picture from early signals toward later ones."""

    def __init__(self, spatial_layers: int = 2, signal_layers: int = 1) -> None:
        super().__init__()
        layers = [
            torch.nn.Conv2d(3, 8, (7, 9), padding=(3, 4)),
            torch.nn.ReLU(),
        ]
        for _ in range(spatial_layers - 1):
            layers.extend(
                [
                    torch.nn.Conv2d(8, 8, (7, 5), padding=(3, 2)),
                    torch.nn.ReLU(),
                ]
            )
        self.local = torch.nn.Sequential(*layers)
        if signal_layers == 1:
            self.score = torch.nn.Conv1d(8, 1, 31, padding=15)
        else:
            signal = []
            for _ in range(signal_layers - 1):
                signal.extend(
                    [
                        torch.nn.Conv1d(8, 8, 15, padding=7),
                        torch.nn.ReLU(),
                    ]
                )
            signal.append(torch.nn.Conv1d(8, 1, 31, padding=15))
            self.score = torch.nn.Sequential(*signal)

    def fraction(self, picture: torch.Tensor) -> torch.Tensor:
        early_n = max(1, min(20, picture.shape[1] // 10))
        changed = picture - picture[:, :early_n].mean(dim=1, keepdim=True)
        mapped = self.local(changed.unsqueeze(0)).mean(dim=3)
        raw = self.score(mapped).reshape(-1)
        level = torch.cumsum(torch.nn.functional.softplus(raw), dim=0)
        turned = torch.sigmoid(level - level.mean())
        previous = torch.cat([turned.new_zeros(1), turned[:-1]])
        mass = (turned - previous).clamp_min(0)
        mass = mass / mass.sum().clamp_min(1e-6)
        slots = torch.arange(raw.shape[0], dtype=picture.dtype) / max(raw.shape[0] - 1, 1)
        return (mass * slots).sum()


def predict_signal(model: HeatmapImageModel, record: dict) -> int:
    model.eval()
    with torch.no_grad():
        share = float(model.fraction(torch.tensor(record["picture"])))
    n_signals = record["n_signals"]
    index = int(np.clip(np.floor(share * (n_signals - 1) + 0.5), 0, n_signals - 1))
    return index + 1


def mean_miss(model: HeatmapImageModel, records: list[dict]) -> float:
    misses = [abs(predict_signal(model, record) - record["signal_id"]) for record in records]
    return float(np.mean(misses))


def load_model(path: Path = MODEL_PATH) -> HeatmapImageModel:
    """Rebuild the saved picture model, including a searched depth."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = HeatmapImageModel(
        int(blob.get("spatial_layers", 2)),
        int(blob.get("signal_layers", 1)),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def train_model(records: list[dict]) -> HeatmapImageModel:
    torch.manual_seed(SEED)
    model = HeatmapImageModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
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
            share = model.fraction(torch.tensor(record["picture"]))
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
                f"  epoch {epoch + 1:3d}  loss {float(np.mean(losses)):.5f}  "
                f"train mean miss {current:.1f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    return model


def score(model: HeatmapImageModel, records: list[dict]) -> list[dict]:
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
    signed = [int(row["miss"]) for row in rows]
    ordered = sorted(misses)
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
    print("Training on the unmarked heatmap pictures", flush=True)
    model = train_model(train_records)
    torch.save({"state_dict": model.state_dict(), "seed": SEED, "epochs": EPOCHS}, MODEL_PATH)
    print(f"Wrote {MODEL_PATH}", flush=True)
    train_rows = score(model, train_records)
    test_rows = score(model, test_records)
    summarize("Training pictures", train_rows)
    summarize("New recordings, held out", test_rows)
    write_table(TRAIN_CSV, train_rows)
    write_table(TEST_CSV, test_rows)


if __name__ == "__main__":
    main()
