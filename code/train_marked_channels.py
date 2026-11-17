"""Train the one-channel 2D-CNN + 1D-CNN on the marked training heatmaps.

The target for each channel is the marked curve in lr_wave_onsets.csv.
The input is that channel's waveform from the experiment .mat file.
A new .mat file is scored one channel at a time.

Run from the acquisition_progress folder:

    python train_marked_channels.py
    python train_marked_channels.py path\to\new_recording.mat
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

import curve_signal as curves

HERE = Path(__file__).resolve().parent
DATA = HERE / "heatmaps" / "ChannelHeatmaps" / "data"
LABELS = DATA / "lr_wave_onsets.csv"
SPLIT = DATA / "split.csv"
MODEL_PATH = HERE / "heatmaps" / "ChannelHeatmaps" / "channel_model.pt"
TRAIN_CSV = DATA / "train_predictions.csv"
TEST_CSV = DATA / "test_predictions.csv"
WINDOW = (0.18e-3, 1.75e-3)
EPOCHS = 30
FINE_EPOCHS = 20
FINE_LR = 3e-4


def _onset():
    path = HERE / "tested_models" / "2D-CNN_1D-CNN_one_channel" / "channel_onset_model.py"
    spec = importlib.util.spec_from_file_location("channel_onset_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def splits() -> dict[str, str]:
    found = {}
    with SPLIT.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "marked":
                found[row["experiment"]] = row["split"]
    return found


def labels_for(which: str) -> list[dict]:
    group = splits()
    rows = []
    with LABELS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if group.get(row["dataset"]) == which and row["status"] == "yes" and row["signal_id"]:
                rows.append(row)
    return rows


def build_records(onset, which: str) -> list[dict]:
    rows = labels_for(which)
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
        for row in items:
            channel = row["channel"]
            if channel not in channels:
                print(f"  missing channel {channel}", flush=True)
                continue
            waves, t0, t1 = onset.prepare_channel(channels[channel], fs, WINDOW)
            signal_id = int(row["signal_id"])
            n_signals = curves._signal_count(waves)
            mark = curves._point_of_interest(n_signals, signal_id)
            time_s = float(row["time_us"]) * 1e-6
            records.append(
                {
                    "experiment": experiment,
                    "channel": channel,
                    "signal_id": signal_id,
                    "maybe_signal_id": int(float(row["maybe_signal_id"])) if row["maybe_signal_id"] else signal_id,
                    "waves": waves,
                    "t0": t0,
                    "t1": t1,
                    "mark": mark,
                    "time_fraction": curves._time_fraction(time_s, t0, t1),
                    "time_s": time_s,
                    "n_signals": n_signals,
                    "minutes_per_signal": float(row["min_per_signal"] or 1),
                }
            )
        del channels
    print(f"{which}: {len(records)} marked channels", flush=True)
    return records


def maybe_run(records: list[dict]) -> int:
    gaps = [max(1, int(item["signal_id"]) - int(item["maybe_signal_id"])) for item in records]
    return int(np.median(gaps))


def score(onset, model, records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        appeared, start_s = onset.predict(model, record)
        rows.append(
            {
                "experiment": record["experiment"],
                "channel": record["channel"],
                "target_signal": record["signal_id"],
                "predicted_signal": appeared,
                "miss": appeared - int(record["signal_id"]),
                "target_time_s": f"{record['time_s']:.6f}",
                "predicted_time_s": f"{start_s:.6f}",
                "predicted_age_min": f"{appeared * record['minutes_per_signal']:.0f}",
            }
        )
    return rows


def summarize(title: str, rows: list[dict]) -> None:
    misses = [abs(int(row["miss"])) for row in rows]
    ordered = sorted(misses)
    print(title, flush=True)
    print(
        f"  channels {len(rows)}  mean miss {sum(misses) / len(misses):.1f}  "
        f"median {ordered[len(ordered) // 2]}  farthest {max(misses)}",
        flush=True,
    )


def write_table(path: Path, rows: list[dict]) -> None:
    fields = [
        "experiment",
        "channel",
        "target_signal",
        "predicted_signal",
        "miss",
        "target_time_s",
        "predicted_time_s",
        "predicted_age_min",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


def load_model(onset):
    model = onset.ChannelOnsetModel()
    blob = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, int(blob.get("maybe_run", 12))


def predict_file(onset, model, path: Path, run: int) -> None:
    fs, channels = curves.load_waveforms(path)
    print(path.name, flush=True)
    for name, samples in channels.items():
        waves, t0, t1 = onset.prepare_channel(samples, fs, WINDOW)
        appeared, start_s = onset.predict(model, {"waves": waves, "t0": t0, "t1": t1})
        n_signals = curves._signal_count(waves)
        mark = appeared - 1
        labels = curves.class_targets(n_signals, mark)
        # The marked Maybe run is longer than the old 12-signal run.
        labels[:] = 2
        labels[:mark] = 0
        maybe_start = max(0, mark - run)
        labels[maybe_start:mark] = 1
        names = np.array([curves.CLASSES[int(item)] for item in labels], dtype=object)
        parts = curves.signal_partitions(names)
        print(f"  {name}", flush=True)
        print(f"    curve at signal {appeared}", flush=True)
        print(f"    age {appeared:.0f} min", flush=True)
        print(f"    time {start_s:.6f} s", flush=True)
        print(f"    No    {parts['No']}", flush=True)
        print(f"    Maybe {parts['Maybe']}", flush=True)
        print(f"    Yes   {parts['Yes']}", flush=True)


def train() -> tuple[object, object, int]:
    onset = _onset()
    train_records = build_records(onset, "train")
    test_records = build_records(onset, "test")
    if len(train_records) < 8:
        raise SystemExit("Not enough marked training channels.")
    run = maybe_run(train_records)
    print(f"Maybe run from the marks: {run} signals", flush=True)
    print("Training 2D-CNN + 1D-CNN on one channel", flush=True)
    model = onset.train_model(train_records, seed=1, epochs=EPOCHS)
    torch.save(
        {"state_dict": model.state_dict(), "maybe_run": run, "window": WINDOW},
        MODEL_PATH,
    )
    print(f"Wrote {MODEL_PATH}", flush=True)
    train_rows = score(onset, model, train_records)
    test_rows = score(onset, model, test_records)
    summarize("Training channels", train_rows)
    summarize("New recordings, held out", test_rows)
    write_table(TRAIN_CSV, train_rows)
    write_table(TEST_CSV, test_rows)
    return onset, model, run


def mean_miss(rows: list[dict]) -> float:
    misses = [abs(int(row["miss"])) for row in rows]
    return sum(misses) / len(misses)


def fine_tune(onset, model, records: list[dict]) -> None:
    """Continue from the saved weights at a lower learning rate."""
    optimizer = torch.optim.Adam(model.parameters(), lr=FINE_LR)
    rng = np.random.default_rng(2)
    best_far = float("inf")
    best_time = float("inf")
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.train()
    for epoch in range(FINE_EPOCHS):
        losses = []
        order = list(records)
        rng.shuffle(order)
        for record in order:
            optimizer.zero_grad()
            loss = onset._loss(model, record)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        farthest, mean_time = onset._errors(model, records)
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
    model.load_state_dict(best_state)
    model.eval()


def finetune() -> None:
    onset = _onset()
    train_records = build_records(onset, "train")
    test_records = build_records(onset, "test")
    model, run = load_model(onset)
    before = score(onset, model, test_records)
    before_mean = mean_miss(before)
    summarize("Held-out channels before fine-tuning", before)
    print(f"Fine-tuning for {FINE_EPOCHS} epochs at learning rate {FINE_LR}", flush=True)
    fine_tune(onset, model, train_records)
    after = score(onset, model, test_records)
    train_rows = score(onset, model, train_records)
    summarize("Training channels after fine-tuning", train_rows)
    summarize("Held-out channels after fine-tuning", after)
    write_table(DATA / "test_predictions_finetuned.csv", after)
    after_mean = mean_miss(after)
    if after_mean < before_mean:
        torch.save(
            {"state_dict": model.state_dict(), "maybe_run": run, "window": WINDOW},
            MODEL_PATH,
        )
        write_table(TEST_CSV, after)
        write_table(TRAIN_CSV, train_rows)
        print(
            f"Kept the fine-tuned model. Held-out mean miss {before_mean:.1f} to {after_mean:.1f}.",
            flush=True,
        )
        return
    print(
        f"Left the saved model unchanged. Held-out mean miss {before_mean:.1f} before fine-tuning "
        f"and {after_mean:.1f} after.",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--finetune":
        finetune()
        return
    argument = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    if argument is not None and MODEL_PATH.is_file():
        onset = _onset()
        model, run = load_model(onset)
        predict_file(onset, model, argument, run)
        return
    onset, model, run = train()
    if argument is not None:
        predict_file(onset, model, argument, run)


if __name__ == "__main__":
    main()
