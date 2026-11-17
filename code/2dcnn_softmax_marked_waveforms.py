"""2D-CNN with a tall kernel and a softmax over signals.

The kernel spans many signals, so it can see the bend rather than one
waveform at a time. The curve signal is the center of the softmax, not a
cumulative turn-on. Trained on the marked training channels.

Run from the acquisition_progress folder:

    python tested_models/2D-CNN_softmax/bend_model.py
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "bend_model.pt"
TRAIN_CSV = HERE / "train_predictions.csv"
TEST_CSV = HERE / "test_predictions.csv"
EPOCHS = 40
SEED = 1


def _trainer():
    path = Path(__file__).resolve().parents[2] / "train_marked_channels.py"
    spec = importlib.util.spec_from_file_location("train_marked_channels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BendModel(torch.nn.Module):
    """Find the signal where the stripe bends."""

    def __init__(self) -> None:
        super().__init__()
        self.local = torch.nn.Sequential(
            torch.nn.Conv2d(2, 8, (31, 9), padding=(15, 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (15, 5), padding=(7, 2)),
            torch.nn.ReLU(),
        )
        self.score = torch.nn.Conv1d(8, 1, 1)

    def fraction(self, waves: torch.Tensor) -> torch.Tensor:
        stack = waves if waves.ndim == 3 else waves.unsqueeze(0)
        n_signals = stack.shape[1]
        early_n = max(1, min(20, n_signals // 10))
        changed = stack - stack[:, :early_n].mean(dim=1, keepdim=True)
        image = torch.stack([changed, changed.abs()], dim=1)
        pooled = self.local(image).mean(dim=3)
        raw = self.score(pooled).reshape(-1)
        mass = torch.softmax(raw, dim=0)
        slots = torch.arange(n_signals, dtype=waves.dtype) / max(n_signals - 1, 1)
        return (mass * slots).sum()


def predict_signal(model: BendModel, record: dict) -> int:
    model.eval()
    with torch.no_grad():
        share = float(model.fraction(torch.tensor(record["waves"])))
    n_signals = record["n_signals"]
    index = int(np.clip(np.floor(share * (n_signals - 1) + 0.5), 0, n_signals - 1))
    return index + 1


def mean_miss(model: BendModel, records: list[dict]) -> float:
    misses = [abs(predict_signal(model, record) - int(record["signal_id"])) for record in records]
    return float(np.mean(misses))


def train_model(records: list[dict]) -> BendModel:
    torch.manual_seed(SEED)
    model = BendModel()
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
            share = model.fraction(torch.tensor(record["waves"]))
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


def score(model: BendModel, records: list[dict]) -> list[dict]:
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
    trainer = _trainer()
    onset = trainer._onset()
    train_records = trainer.build_records(onset, "train")
    test_records = trainer.build_records(onset, "test")
    print("Training 2D-CNN + softmax", flush=True)
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
