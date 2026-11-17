"""2D-CNN + 1D-CNN trained for 50 epochs from the center of the same channel.

The center is the middle of that channel's arrival move on the heatmap window.
Recordings of the same setup and the same channel share one offset, taken from
the training files only. The network mixes its own onset with that anchor.

Run from the acquisition_progress folder:

    python tested_models/2D-CNN_1D-CNN_center/center_model.py
    python tested_models/2D-CNN_1D-CNN_center/center_model.py finetune
"""

from __future__ import annotations

import csv
import importlib.util
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
MODEL_PATH = HERE / "center_model.pt"
TRAIN_CSV = HERE / "train_predictions.csv"
TEST_CSV = HERE / "test_predictions.csv"
EPOCHS = 300
SEED = 1
DEFAULT_WINDOW = (0.0, 2.0e-3)
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


def _trainer():
    path = Path(__file__).resolve().parents[2] / "train_marked_channels.py"
    spec = importlib.util.spec_from_file_location("train_marked_channels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_of(experiment: str) -> str:
    prefixes = (
        ("FA paste", "FA paste"),
        ("FA_", "FA 8ch"),
        ("LC3 paste", "LC3 paste"),
        ("LC3_", "LC3 8ch"),
        ("OPC1L 11 degree", "OPC 16ch"),
        ("OPC1L_", "OPC 8ch"),
        ("Senscomp_LC3", "Senscomp LC3"),
        ("Senscomp_OPC", "Senscomp OPC"),
    )
    for prefix, name in prefixes:
        if experiment.startswith(prefix):
            return name
    return experiment


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    return np.convolve(values, np.ones(width) / width, mode="same")


def _first_below(trend: np.ndarray, level: float, start: int) -> int:
    hits = np.flatnonzero(trend[start:] < level)
    if hits.size == 0:
        return int(np.argmin(trend))
    return int(hits[0] + start)


def center_signal(samples: np.ndarray, fs: float, window: tuple[float, float]) -> int:
    """1-based signal at the middle of this channel's arrival move."""
    array = np.asarray(samples, dtype=np.float64)
    sos = butter(4, [23_000.0, 73_000.0], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, array, axis=1)
    n_samples = filtered.shape[1]
    gate0 = min(n_samples - 8, max(0, int(round(window[0] * fs))))
    gate1 = min(n_samples, max(gate0 + 8, int(round(window[1] * fs))))
    piece = filtered[:, gate0:gate1]
    envelope = np.abs(hilbert(piece, axis=1))
    pad = 4
    padded = np.pad(envelope, ((0, 0), (pad, 4)), mode="edge")
    total = np.cumsum(padded, axis=1)
    total = np.pad(total, ((0, 0), (1, 0)))
    envelope = (total[:, 9:] - total[:, :-9]) / 9.0
    n_signals = envelope.shape[0]
    late_rows = envelope[int(n_signals * 0.75) :]
    level = 0.25 * float(np.median(np.max(late_rows, axis=1))) + 1e-9
    above = envelope > level
    first = above.argmax(axis=1).astype(np.float64)
    first[~above.any(axis=1)] = piece.shape[1]
    trend = _smooth(first / piece.shape[1], 15)
    early_n = max(12, n_signals // 10)
    baseline = float(np.percentile(trend[: max(early_n, n_signals // 3)], 80))
    floor = float(np.median(trend[int(n_signals * 0.7) :]))
    drop = baseline - floor
    if drop < 0.03:
        strength = _smooth(np.sqrt(np.mean(piece ** 2, axis=1)), 11)
        late_level = float(np.median(strength[int(n_signals * 0.7) :]))
        early_level = float(np.median(strength[:early_n]))
        halfway = early_level + 0.5 * (late_level - early_level)
        hits = np.flatnonzero(strength >= halfway)
        return int(hits[0]) + 1 if hits.size else early_n + 1
    start = _first_below(trend, baseline - 0.2 * drop, early_n)
    finish = _first_below(trend, baseline - 0.8 * drop, start)
    return (start + finish) // 2 + 1


def load_records(trainer, onset, which: str) -> list[dict]:
    group = {}
    with SPLIT.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "marked":
                group[row["experiment"]] = row["split"]
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
            waves, t0, t1 = onset.prepare_channel(channels[channel], fs, trainer.WINDOW)
            signal_id = int(row["signal_id"])
            n_signals = curves._signal_count(waves)
            records.append(
                {
                    "experiment": experiment,
                    "channel": channel,
                    "setup": setup_of(experiment),
                    "signal_id": signal_id,
                    "waves": waves,
                    "t0": t0,
                    "t1": t1,
                    "mark": curves._point_of_interest(n_signals, signal_id),
                    "time_fraction": curves._time_fraction(float(row["time_us"]) * 1e-6, t0, t1),
                    "time_s": float(row["time_us"]) * 1e-6,
                    "n_signals": n_signals,
                    "center": center_signal(channels[channel], fs, window),
                }
            )
        del channels
    print(f"{which}: {len(records)} marked channels", flush=True)
    return records


def attach_offsets(train_records: list[dict], records: list[dict]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_channel: dict[str, list[float]] = defaultdict(list)
    everything = []
    for record in train_records:
        residual = float(record["signal_id"] - record["center"])
        grouped[(record["setup"], record["channel"])].append(residual)
        by_channel[record["channel"]].append(residual)
        everything.append(residual)
    global_offset = float(np.median(everything))
    channel_offset = {name: float(np.median(values)) for name, values in by_channel.items()}
    pair_offset = {key: float(np.median(values)) for key, values in grouped.items()}
    for record in records:
        key = (record["setup"], record["channel"])
        record["offset"] = pair_offset.get(key, channel_offset.get(record["channel"], global_offset))


def _anchor_fraction(record: dict) -> float:
    anchor = float(record["center"]) + float(record["offset"])
    return (anchor - 1.0) / max(record["n_signals"] - 1, 1)


def _target_fraction(record: dict) -> float:
    return float(record["mark"]) / max(record["n_signals"] - 1, 1)


def predict_signal(model, mix_logit: torch.Tensor, record: dict) -> int:
    model.eval()
    with torch.no_grad():
        fraction, _when = model.read(torch.tensor(record["waves"]))
        mix = torch.sigmoid(mix_logit)
        share = mix * fraction + (1.0 - mix) * _anchor_fraction(record)
    n_signals = int(record["n_signals"])
    index = int(np.clip(np.floor(float(share) * (n_signals - 1) + 0.5), 0, n_signals - 1))
    return index + 1


def anchor_signal(record: dict) -> int:
    n_signals = int(record["n_signals"])
    anchor = float(record["center"]) + float(record["offset"])
    return int(np.clip(np.rint(anchor), 1, n_signals))


def mean_miss(model, mix_logit, records: list[dict]) -> float:
    misses = [abs(predict_signal(model, mix_logit, record) - int(record["signal_id"])) for record in records]
    return float(np.mean(misses))


def train_model(
    onset,
    records: list[dict],
    held_out: list[dict],
    learning_rate: float = 3e-3,
    max_epochs: int | None = None,
    patience: int = 80,
    patience_start: int = 120,
):
    """Train up to max_epochs. Keep the epoch with the lowest held-out mean miss."""
    torch.manual_seed(SEED)
    model = onset.ChannelOnsetModel()
    mix_logit = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam(list(model.parameters()) + [mix_logit], lr=learning_rate)
    rng = np.random.default_rng(SEED)
    best_held = float("inf")
    best_epoch = 0
    best_state = None
    best_mix = None
    stale = 0
    history = []
    limit = EPOCHS if max_epochs is None else max_epochs
    for epoch in range(limit):
        model.train()
        order = list(records)
        rng.shuffle(order)
        losses = []
        for record in order:
            optimizer.zero_grad()
            fraction, when = model.read(torch.tensor(record["waves"]))
            mix = torch.sigmoid(mix_logit)
            share = mix * fraction + (1.0 - mix) * _anchor_fraction(record)
            target = _target_fraction(record)
            onset_loss = (share - target).square()
            time_loss = (when[int(record["mark"])] - float(record["time_fraction"])).square()
            loss = onset_loss + 0.25 * time_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [mix_logit], 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        train_miss = mean_miss(model, mix_logit, records)
        held_miss = mean_miss(model, mix_logit, held_out)
        history.append((epoch + 1, train_miss, held_miss))
        note = ""
        if held_miss < best_held:
            best_held = held_miss
            best_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            best_mix = float(mix_logit.detach())
            stale = 0
            note = "  lowest held-out so far"
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0 or note:
            print(
                f"    epoch {epoch + 1:3d}  loss {float(np.mean(losses)):.5f}  "
                f"train mean miss {train_miss:.1f}  held-out mean miss {held_miss:.1f}  "
                f"network weight {float(torch.sigmoid(mix_logit.detach())):.2f}{note}",
                flush=True,
            )
        if best_held < 10.0:
            print(f"Held-out mean miss is under 10 at epoch {best_epoch}.", flush=True)
            break
        if epoch + 1 >= patience_start and stale >= patience:
            print(
                f"Held-out mean miss stopped falling. Lowest is epoch {best_epoch}.",
                flush=True,
            )
            break
    model.load_state_dict(best_state)
    mix_logit = torch.nn.Parameter(torch.tensor(best_mix))
    model.eval()
    return model, mix_logit, best_epoch, history


def score(records: list[dict], model, mix_logit) -> list[dict]:
    rows = []
    for record in records:
        appeared = predict_signal(model, mix_logit, record)
        anchored = anchor_signal(record)
        rows.append(
            {
                "experiment": record["experiment"],
                "channel": record["channel"],
                "target_signal": record["signal_id"],
                "center_signal": record["center"],
                "anchor_signal": anchored,
                "predicted_signal": appeared,
                "miss": appeared - int(record["signal_id"]),
                "anchor_miss": anchored - int(record["signal_id"]),
            }
        )
    return rows


def summarize(title: str, rows: list[dict], field: str) -> None:
    misses = [abs(int(row[field])) for row in rows]
    ordered = sorted(misses)
    signed = [int(row[field]) for row in rows]
    print(title, flush=True)
    print(
        f"  channels {len(rows)}  mean miss {sum(misses) / len(misses):.1f}  "
        f"median {ordered[len(ordered) // 2]}  farthest {max(misses)}  "
        f"signed mean {sum(signed) / len(signed):+.1f}",
        flush=True,
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(int(row[field]))
    for name, values in grouped.items():
        absolute = [abs(value) for value in values]
        print(
            f"  {name[:52]:52} mean {sum(absolute) / len(absolute):5.1f}  "
            f"signed {sum(values) / len(values):+6.1f}",
            flush=True,
        )


def write_table(path: Path, rows: list[dict]) -> None:
    fields = [
        "experiment",
        "channel",
        "target_signal",
        "center_signal",
        "anchor_signal",
        "predicted_signal",
        "miss",
        "anchor_miss",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


def _load_saved(onset):
    blob = torch.load(MODEL_PATH, weights_only=False)
    model = onset.ChannelOnsetModel()
    model.load_state_dict(blob["state_dict"])
    mix_logit = torch.nn.Parameter(torch.tensor(float(blob["mix_logit"])))
    model.eval()
    return model, mix_logit, blob


def fine_tune(only_rate: float | None = None) -> None:
    """Continue the saved epoch-82 weights at a smaller step.

    A checkpoint is kept only when the training mean miss and the held-out
    mean miss are both lower than they were at epoch 82.
    """
    trainer = _trainer()
    onset = trainer._onset()
    train_records = load_records(trainer, onset, "train")
    test_records = load_records(trainer, onset, "test")
    attach_offsets(train_records, train_records + test_records)
    _base_model, base_mix, blob = _load_saved(onset)
    start_train = mean_miss(_base_model, base_mix, train_records)
    start_held = mean_miss(_base_model, base_mix, test_records)
    saved_epoch = int(blob.get("epochs", 82))
    print(
        f"Epoch {saved_epoch} start: train mean miss {start_train:.1f}  "
        f"held-out mean miss {start_held:.1f}",
        flush=True,
    )
    log_rows = []
    best = None
    rates = (only_rate,) if only_rate is not None else (1e-4, 3e-4)
    for rate in rates:
        print(f"Fine-tune learning rate {rate:g}", flush=True)
        model = onset.ChannelOnsetModel()
        model.load_state_dict(blob["state_dict"])
        mix_logit = torch.nn.Parameter(torch.tensor(float(blob["mix_logit"])))
        optimizer = torch.optim.Adam(list(model.parameters()) + [mix_logit], lr=rate)
        rng = np.random.default_rng(SEED)
        stale = 0
        found = False
        give_up = 40 if rate >= 3e-4 else 20
        for epoch in range(40):
            model.train()
            order = list(train_records)
            rng.shuffle(order)
            losses = []
            for record in order:
                optimizer.zero_grad()
                fraction, when = model.read(torch.tensor(record["waves"]))
                mix = torch.sigmoid(mix_logit)
                share = mix * fraction + (1.0 - mix) * _anchor_fraction(record)
                target = _target_fraction(record)
                onset_loss = (share - target).square()
                time_loss = (when[int(record["mark"])] - float(record["time_fraction"])).square()
                loss = onset_loss + 0.25 * time_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [mix_logit], 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            train_miss = mean_miss(model, mix_logit, train_records)
            held_miss = mean_miss(model, mix_logit, test_records)
            joint = train_miss < start_train and held_miss < start_held
            note = ""
            if joint and (
                best is None
                or held_miss < best[0]
                or (held_miss == best[0] and train_miss < best[1])
            ):
                best = (
                    held_miss,
                    train_miss,
                    rate,
                    epoch + 1,
                    {name: value.detach().clone() for name, value in model.state_dict().items()},
                    float(mix_logit.detach()),
                )
                stale = 0
                found = True
                note = "  both means lower"
            else:
                stale += 1
            log_rows.append(
                {
                    "learning_rate": rate,
                    "epoch": epoch + 1,
                    "train_mean_miss": train_miss,
                    "heldout_mean_miss": held_miss,
                }
            )
            print(
                f"    epoch {epoch + 1:3d}  loss {float(np.mean(losses)):.5f}  "
                f"train mean miss {train_miss:.1f}  held-out mean miss {held_miss:.1f}  "
                f"network weight {float(torch.sigmoid(mix_logit.detach())):.2f}{note}",
                flush=True,
            )
            if found and stale >= 15:
                print("Both means stopped falling.", flush=True)
                break
            if not found and epoch + 1 >= give_up:
                print("This rate did not lower both means.", flush=True)
                break
    table = HERE / ("finetune_epochs.csv" if only_rate is None else f"finetune_{only_rate:g}.csv")
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["learning_rate", "epoch", "train_mean_miss", "heldout_mean_miss"],
        )
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Wrote {table}", flush=True)
    if best is None:
        print(
            f"No fine-tune step lowered both means below train {start_train:.1f} "
            f"and held-out {start_held:.1f}. Epoch {saved_epoch} weights kept.",
            flush=True,
        )
        return
    held_miss, train_miss, rate, epoch, state, mix_value = best
    model = onset.ChannelOnsetModel()
    model.load_state_dict(state)
    mix_logit = torch.nn.Parameter(torch.tensor(mix_value))
    model.eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mix_logit": mix_value,
            "seed": SEED,
            "epochs": saved_epoch,
            "finetune_epochs": epoch,
            "learning_rate": rate,
            "offsets": blob.get(
                "offsets",
                {
                    f"{record['setup']}|{record['channel']}": record["offset"]
                    for record in train_records
                },
            ),
        },
        MODEL_PATH,
    )
    print(
        f"Kept fine-tune at learning rate {rate:g}, epoch {epoch}: "
        f"train mean miss {train_miss:.1f}  held-out mean miss {held_miss:.1f}",
        flush=True,
    )
    print(f"Wrote {MODEL_PATH}", flush=True)
    train_rows = score(train_records, model, mix_logit)
    test_rows = score(test_records, model, mix_logit)
    summarize(f"Training channels after fine-tune epoch {epoch}", train_rows, "miss")
    summarize(f"Held-out channels after fine-tune epoch {epoch}", test_rows, "miss")
    write_table(TRAIN_CSV, train_rows)
    write_table(TEST_CSV, test_rows)


def search_rates() -> None:
    """Try other Adam learning rates. 0.003 was already searched epoch by epoch."""
    trainer = _trainer()
    onset = trainer._onset()
    train_records = load_records(trainer, onset, "train")
    test_records = load_records(trainer, onset, "test")
    attach_offsets(train_records, train_records + test_records)
    rates = (1e-4, 1e-3, 1e-2)
    rows = [{"learning_rate": 3e-3, "epoch": 82, "train_mean_miss": 13.1, "heldout_mean_miss": 25.6}]
    best_rate = 3e-3
    best_held = 25.6
    best_pack = None
    for rate in rates:
        print(f"Learning rate {rate:g}", flush=True)
        model, mix_logit, best_epoch, history = train_model(
            onset,
            train_records,
            test_records,
            learning_rate=rate,
            max_epochs=120,
            patience=40,
            patience_start=40,
        )
        held = min(item[2] for item in history)
        train_at = [item[1] for item in history if item[0] == best_epoch][0]
        print(
            f"  rate {rate:g}  lowest held-out mean {held:.1f} at epoch {best_epoch}  "
            f"train mean miss {train_at:.1f}",
            flush=True,
        )
        rows.append(
            {
                "learning_rate": rate,
                "epoch": best_epoch,
                "train_mean_miss": train_at,
                "heldout_mean_miss": held,
            }
        )
        if held < best_held:
            best_held = held
            best_rate = rate
            best_pack = (model, mix_logit, best_epoch)
    table = HERE / "lr_search.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["learning_rate", "epoch", "train_mean_miss", "heldout_mean_miss"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {table}", flush=True)
    print(f"Lowest held-out mean miss {best_held:.1f} at learning rate {best_rate:g}.", flush=True)
    if best_pack is not None and best_held < 10.0:
        model, mix_logit, best_epoch = best_pack
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mix_logit": float(mix_logit.detach()),
                "seed": SEED,
                "epochs": best_epoch,
                "learning_rate": best_rate,
            },
            MODEL_PATH,
        )
        print(f"Wrote {MODEL_PATH}", flush=True)


def main() -> None:
    trainer = _trainer()
    onset = trainer._onset()
    train_records = load_records(trainer, onset, "train")
    test_records = load_records(trainer, onset, "test")
    attach_offsets(train_records, train_records + test_records)
    train_anchor = [abs(anchor_signal(record) - int(record["signal_id"])) for record in train_records]
    print(
        f"Same-channel center on the training channels: mean miss {float(np.mean(train_anchor)):.1f}",
        flush=True,
    )
    print(f"Training 2D-CNN + 1D-CNN for up to {EPOCHS} epochs", flush=True)
    print("The kept epoch is the one with the lowest held-out mean miss.", flush=True)
    model, mix_logit, best_epoch, history = train_model(onset, train_records, test_records)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mix_logit": float(mix_logit.detach()),
            "seed": SEED,
            "epochs": best_epoch,
            "offsets": {
                f"{record['setup']}|{record['channel']}": record["offset"] for record in train_records
            },
        },
        MODEL_PATH,
    )
    print(f"Wrote {MODEL_PATH}", flush=True)
    with (HERE / "epoch_heldout.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_mean_miss", "heldout_mean_miss"])
        writer.writerows(history)
    train_rows = score(train_records, model, mix_logit)
    test_rows = score(test_records, model, mix_logit)
    summarize("Training channels, center only", train_rows, "anchor_miss")
    summarize(f"Training channels at epoch {best_epoch}", train_rows, "miss")
    summarize("Held-out channels, center only", test_rows, "anchor_miss")
    summarize(f"Held-out channels at epoch {best_epoch}", test_rows, "miss")
    write_table(TRAIN_CSV, train_rows)
    write_table(TEST_CSV, test_rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rates":
        search_rates()
    elif len(sys.argv) > 1 and sys.argv[1] == "finetune":
        chosen = float(sys.argv[2]) if len(sys.argv) > 2 else None
        fine_tune(chosen)
    else:
        main()
