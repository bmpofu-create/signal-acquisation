"""Tall-kernel heatmap model.

The kernel covers many signals at once, so it can see the bend in the picture.
The curve signal is the center of a softmax over rows. This is a different
readout from the saved model, which turns a score on and leaves it on.

The unmarked pictures and the same train/test recordings are used. The saved
heatmap weights are replaced only when both means fall.

Run from the acquisition_progress folder:

    python tested_models/heatmap_image/tall_heatmap_model.py
"""

from __future__ import annotations

import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "heatmap_tall_model.pt"
TRAIN_CSV = HERE / "tall_train_predictions.csv"
TEST_CSV = HERE / "tall_test_predictions.csv"
EPOCHS = 30
SEED = 1
LEARNING_RATE = 3e-3


def load_images():
    spec = importlib.util.spec_from_file_location(
        "heatmap_image_model", HERE / "2dcnn_4spatial_unmarked_heatmaps.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TallHeatmapModel(torch.nn.Module):
    """Name the row where the stripe bends, from a tall view of the picture."""

    def __init__(self) -> None:
        super().__init__()
        self.local = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, (31, 9), padding=(15, 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (15, 5), padding=(7, 2)),
            torch.nn.ReLU(),
        )
        self.score = torch.nn.Conv1d(8, 1, 1)

    def fraction(self, picture: torch.Tensor) -> torch.Tensor:
        early_n = max(1, min(20, picture.shape[1] // 10))
        changed = picture - picture[:, :early_n].mean(dim=1, keepdim=True)
        mapped = self.local(changed.unsqueeze(0)).mean(dim=3)
        raw = self.score(mapped).reshape(-1)
        mass = torch.softmax(raw, dim=0)
        slots = torch.arange(raw.shape[0], dtype=picture.dtype) / max(raw.shape[0] - 1, 1)
        return (mass * slots).sum()


def predict_signal(model: TallHeatmapModel, record: dict) -> int:
    model.eval()
    with torch.no_grad():
        share = float(model.fraction(torch.tensor(record["picture"])))
    n_signals = int(record["n_signals"])
    index = int(np.clip(np.floor(share * (n_signals - 1) + 0.5), 0, n_signals - 1))
    return index + 1


def mean_miss(model: TallHeatmapModel, records: list[dict]) -> float:
    misses = [abs(predict_signal(model, record) - int(record["signal_id"])) for record in records]
    return float(np.mean(misses))


def train_model(records: list[dict]) -> tuple[TallHeatmapModel, int]:
    torch.manual_seed(SEED)
    model = TallHeatmapModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(SEED)
    best_mean = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        order = list(records)
        rng.shuffle(order)
        losses = []
        for record in order:
            optimizer.zero_grad()
            share = model.fraction(torch.tensor(record["picture"]))
            target = float(record["mark"]) / max(int(record["n_signals"]) - 1, 1)
            loss = (share - target).square()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        current = mean_miss(model, records)
        if current < best_mean:
            best_mean = current
            best_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"  epoch {epoch + 1:3d}  loss {float(np.mean(losses)):.5f}  "
                f"train mean miss {current:.1f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def score(model: TallHeatmapModel, records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        appeared = predict_signal(model, record)
        rows.append(
            {
                "experiment": record["experiment"],
                "channel": record["channel"],
                "target_signal": record["signal_id"],
                "predicted_signal": appeared,
                "miss": appeared - int(record["signal_id"]),
            }
        )
    return rows


def summarize(title: str, rows: list[dict]) -> tuple[float, int, int, float]:
    misses = [abs(int(row["miss"])) for row in rows]
    signed = [int(row["miss"]) for row in rows]
    ordered = sorted(misses)
    mean = sum(misses) / len(misses)
    median = ordered[len(ordered) // 2]
    farthest = max(misses)
    signed_mean = sum(signed) / len(signed)
    print(title, flush=True)
    print(
        f"  channels {len(rows)}  mean miss {mean:.1f}  median {median}  "
        f"farthest {farthest}  signed mean {signed_mean:+.1f}",
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
    return mean, median, farthest, signed_mean


def write_table(path: Path, rows: list[dict]) -> None:
    fields = ["experiment", "channel", "target_signal", "predicted_signal", "miss"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


def main() -> None:
    images = load_images()
    train_records = images.load_records("train")
    test_records = images.load_records("test")
    print("Training tall-kernel softmax on the unmarked heatmap pictures", flush=True)
    model, best_epoch = train_model(train_records)
    torch.save(
        {"state_dict": model.state_dict(), "seed": SEED, "epochs": best_epoch, "learning_rate": LEARNING_RATE},
        MODEL_PATH,
    )
    print(f"Wrote {MODEL_PATH} at epoch {best_epoch}", flush=True)
    train_rows = score(model, train_records)
    test_rows = score(model, test_records)
    train_mean, _, _, _ = summarize("Training pictures", train_rows)
    held_mean, _, _, _ = summarize("New recordings, held out", test_rows)
    write_table(TRAIN_CSV, train_rows)
    write_table(TEST_CSV, test_rows)

    saved = images.load_model()
    base_train = images.mean_miss(saved, train_records)
    base_held = images.mean_miss(saved, test_records)
    print(f"Saved heatmap model  train {base_train:.2f}  held-out {base_held:.2f}", flush=True)
    if train_mean < base_train and held_mean < base_held:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": SEED,
                "epochs": best_epoch,
                "learning_rate": LEARNING_RATE,
                "structure": "tall softmax",
            },
            images.MODEL_PATH,
        )
        print(f"Replaced {images.MODEL_PATH}", flush=True)
    else:
        print("Saved heatmap weights kept.", flush=True)


if __name__ == "__main__":
    main()
