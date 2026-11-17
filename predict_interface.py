"""Local page for a new recording.

Upload a .mat file and, when the chosen model reads a picture, a heatmap.
The page names the curve signal and the No, Maybe, and Yes ranges.

Run from the final_models folder:

    python predict_interface.py

Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import csv
import importlib.util
import secrets
import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, render_template_string, request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import curve_signal as curves

CODE = HERE / "code"
FINAL = HERE / "models"
DATA = HERE / "data"
UPLOADS = HERE / "uploads"
WAVE_WINDOW = (0.18e-3, 1.75e-3)
MAYBE_FALLBACK = 27
PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Find the curve signal</title>
  <style>
    :root {
      --ink: #1c2430;
      --muted: #5c6775;
      --line: #e3ddd2;
      --paper: #f6f3ec;
      --card: #fffdf8;
      --green: #1f6b4a;
      --green-soft: #e7f3ec;
      --amber: #8a5a12;
      --amber-soft: #fbf1dc;
      --no: #4d5968;
      --no-soft: #eef1f4;
      --danger: #8d2d2d;
      --danger-soft: #fdecec;
    }
    * { box-sizing: border-box; }
    html { background: var(--paper); color-scheme: light; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font: 16px/1.5 "Segoe UI", sans-serif;
    }
    .wrap { max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }
    header h1 { margin: 0 0 8px; font-size: 34px; letter-spacing: -0.03em; }
    header p { margin: 0; max-width: 68ch; color: var(--muted); }
    .card {
      margin-top: 22px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 22px;
    }
    .steps { display: grid; gap: 18px; }
    .step { display: grid; grid-template-columns: 36px 1fr; gap: 12px; }
    .num {
      width: 36px; height: 36px; border-radius: 50%;
      background: var(--green); color: white;
      display: grid; place-items: center; font-weight: 700;
    }
    h2 { margin: 4px 0 4px; font-size: 18px; }
    .help { margin: 0 0 10px; color: var(--muted); }
    label { display: block; margin: 12px 0 6px; font-weight: 650; }
    select, button {
      font: inherit;
      color: var(--ink);
      background: white;
    }
    select {
      width: min(420px, 100%);
      padding: 10px 12px;
      border: 1px solid #cfc6b8;
      border-radius: 10px;
    }
    .note {
      margin: 12px 0 0;
      padding: 12px 14px;
      background: var(--green-soft);
      border-radius: 12px;
    }
    .files { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .drop {
      display: block;
      padding: 16px;
      border: 1.5px dashed #c9bfb0;
      border-radius: 14px;
      background: #fff;
      cursor: pointer;
    }
    .drop strong { display: block; }
    .drop span { color: var(--muted); font-size: 14px; }
    .drop input { margin-top: 10px; max-width: 100%; }
    .chosen { display: block; margin: 8px 0 0; font-size: 14px; color: var(--green); min-height: 1.2em; }
    button.primary {
      margin-top: 18px;
      background: var(--green);
      color: white;
      border: 0;
      border-radius: 12px;
      padding: 12px 18px;
      font-weight: 700;
      cursor: pointer;
    }
    button.primary:hover { background: #18573c; }
    .error {
      margin-top: 16px;
      padding: 12px 14px;
      background: var(--danger-soft);
      color: var(--danger);
      border-radius: 12px;
    }
    .model-name { margin: 0; font-size: 24px; }
    table { width: 100%; border-collapse: collapse; margin-top: 14px; }
    th { text-align: left; font-size: 13px; color: var(--muted); font-weight: 650; padding: 8px; }
    td { padding: 10px 8px; border-top: 1px solid var(--line); vertical-align: top; }
    .pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-variant-numeric: tabular-nums;
    }
    .pill.no { background: var(--no-soft); color: var(--no); }
    .pill.maybe { background: var(--amber-soft); color: var(--amber); }
    .pill.yes { background: var(--green-soft); color: var(--green); }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 0; }
    .legend span { font-size: 14px; }
    .scores { display: flex; gap: 12px; margin-top: 14px; }
    .score { flex: 1; background: #f7f4ee; border-radius: 14px; padding: 14px 16px; }
    .score b { display: block; font-size: 28px; letter-spacing: -0.03em; }
    .score span { color: var(--muted); font-size: 14px; }
    @media (max-width: 720px) {
      .files { grid-template-columns: 1fr; }
      header h1 { font-size: 28px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Find the curve signal</h1>
      <p>Upload one recording. The page names the signal where the curve starts, then lists the No, Maybe, and Yes ranges. Age is that signal number, in minutes.</p>
    </header>
    <form class="card" method="post" enctype="multipart/form-data">
      <div class="steps">
        <div class="step">
          <div class="num">1</div>
          <div>
            <h2>Which experiment is this?</h2>
            <p class="help">Automatic uses the model that did best for that setup.</p>
            <label for="setup">Experiment</label>
            <select id="setup" name="setup">
              {% for value, label in setups %}
              <option value="{{ value }}" {% if value == chosen_setup %}selected{% endif %}>{{ label }}</option>
              {% endfor %}
            </select>
          </div>
        </div>
        <div class="step">
          <div class="num">2</div>
          <div>
            <h2>Which model should run?</h2>
            <p class="help">Leave this on Automatic unless you want a specific model.</p>
            <label for="model">Model</label>
            <select id="model" name="model">
              {% for value, label in model_choices %}
              <option value="{{ value }}" {% if value == chosen_model %}selected{% endif %}>{{ label }}</option>
              {% endfor %}
            </select>
            <p class="note" id="auto-note"></p>
          </div>
        </div>
        <div class="step">
          <div class="num">3</div>
          <div>
            <h2>Add the files</h2>
            <p class="help">Pick a saved recording, or upload one. After the first prediction, change the model and predict again. The same recording stays selected. Automatic is the best model for that setup unless you choose another one.</p>
            <label for="saved">Saved recording</label>
            <select id="saved" name="saved">
              <option value="">Use the uploaded recording</option>
              {% for value, label in recordings %}
              <option value="{{ value }}" {% if value == chosen_saved %}selected{% endif %}>{{ label }}</option>
              {% endfor %}
            </select>
            {% if kept_name %}
            <p class="note" id="kept-note">Using {{ kept_name }}. Change the model and predict again. You do not need to upload this file again.</p>
            {% endif %}
            <input type="hidden" name="kept" value="{{ kept }}">
            <div class="files">
              <label class="drop" for="recording">
                <strong>MATLAB recording</strong>
                <span>.mat file, one experiment</span>
                <input id="recording" name="recording" type="file" accept=".mat">
                <span class="chosen" id="recording-name">No file yet</span>
              </label>
              <label class="drop" for="heatmap">
                <strong>Heatmap picture</strong>
                <span>Optional .png, one channel</span>
                <input id="heatmap" name="heatmap" type="file" accept=".png,.jpg,.jpeg">
                <span class="chosen" id="heatmap-name">No picture yet</span>
              </label>
            </div>
          </div>
        </div>
      </div>
      <button class="primary" type="submit">Predict the curve</button>
    </form>
    {% if error %}
    <p class="error">{{ error }}</p>
    {% endif %}
    {% if result %}
    <section class="card">
      <h2 class="model-name">{{ result.model_name }}</h2>
      <p class="note">{{ result.reason }}</p>
      {% if result.scores %}
      <div class="scores">
        <div class="score"><b>{{ result.scores.mean_miss }}</b><span>Mean miss, in signals</span></div>
        <div class="score"><b>{{ result.scores.accuracy }}%</b><span>Accuracy</span></div>
      </div>
      <p class="help">Accuracy is the share of signals whose No, Maybe, or Yes label matches the saved click. {{ result.scores.exact }}% of the compared channels land on the exact click ({{ result.scores.exact_count }} of {{ result.scores.channels }}).</p>
      {% elif result.unscored %}
      <p class="help">This recording is not in the saved clicks, so mean miss and accuracy are not available. The ranges below are still the prediction.</p>
      {% endif %}
      <div class="legend">
        <span class="pill no">No</span> <span>curve has not started</span>
        <span class="pill maybe">Maybe</span> <span>the 27 signals just before the curve</span>
        <span class="pill yes">Yes</span> <span>the curve and every signal after it</span>
        <span>Signal emerges is the time inside that waveform when the signal starts.</span>
      </div>
      <table>
        <thead>
          <tr><th>Channel</th><th>Signal</th><th>Miss</th><th>Age</th><th>Signal emerges</th><th>No</th><th>Maybe</th><th>Yes</th></tr>
        </thead>
        <tbody>
          {% for row in result.rows %}
          <tr>
            <td>{{ row.channel }}</td>
            <td>{{ row.signal }}</td>
            <td>{{ row.miss }}</td>
            <td>{{ row.age }}</td>
            <td>{{ row.time_s }}</td>
            <td><span class="pill no">{{ row.no }}</span></td>
            <td><span class="pill maybe">{{ row.maybe }}</span></td>
            <td><span class="pill yes">{{ row.yes }}</span></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>
    {% endif %}
  </div>
  <script>
    const note = document.getElementById("auto-note");
    const setup = document.getElementById("setup");
    const model = document.getElementById("model");
    const heatmap = document.getElementById("heatmap");
    const recording = document.getElementById("recording");
    const saved = document.getElementById("saved");
    function fileName(input, empty) {
      const target = document.getElementById(input.id + "-name");
      target.textContent = input.files[0] ? input.files[0].name : empty;
    }
    function activeSetup() {
      if (setup.value !== "other") return setup.value;
      const name = saved.value.toLowerCase();
      if (name.includes("senscomp") && name.includes("lc3")) return "Senscomp LC3";
      if (name.includes("senscomp") && name.includes("opc")) return "Senscomp OPC";
      if (name.includes("fa paste")) return "FA paste";
      if (name.includes("fa_") || name.includes("fa ")) return "FA 8ch";
      if (name.includes("lc3 paste")) return "LC3 paste";
      if (name.includes("lc3")) return "LC3 8ch";
      if (name.includes("11 degree")) return "OPC 16ch";
      if (name.includes("opc")) return "OPC 8ch";
      return "other";
    }
    function explain() {
      if (model.value !== "auto") {
        note.textContent = "You chose a model. Predict again to run it on the same recording.";
        return;
      }
      const which = activeSetup();
      const picture = heatmap.files.length > 0 || saved.value.includes("FA paste");
      if (which === "FA paste" && picture) {
        note.textContent = "Automatic will use the 2D-CNN tall kernel trained on unmarked heatmaps.";
      } else if (which === "FA paste") {
        note.textContent = "Automatic will use the 1D-CNN first arrival trained on marked waveforms. Add the heatmap picture, or pick the saved FA paste recording, to use the tall-kernel heatmap model.";
      } else if (which === "LC3 8ch") {
        note.textContent = "Automatic will use the 1D-CNN first arrival trained on marked waveforms.";
      } else if (which === "OPC 8ch") {
        note.textContent = "Automatic will use the 2D-CNN + 1D-CNN trained on marked waveforms.";
      } else {
        note.textContent = "Automatic will use the 2D-CNN + 1D-CNN trained on marked waveforms. This setup has no held-out score of its own.";
      }
    }
    [setup, model, heatmap, saved].forEach((item) => item.addEventListener("change", explain));
    recording.addEventListener("change", () => fileName(recording, "No file yet"));
    heatmap.addEventListener("change", () => { fileName(heatmap, "No picture yet"); explain(); });
    explain();
  </script>
</body>
</html>
"""

SETUPS = (
    ("other", "Other or not sure"),
    ("FA paste", "FA paste"),
    ("FA 8ch", "FA 8-channel"),
    ("LC3 paste", "LC3 paste"),
    ("LC3 8ch", "LC3 8-channel"),
    ("OPC 16ch", "OPC 16-channel"),
    ("OPC 8ch", "OPC 8-channel"),
    ("Senscomp LC3", "Senscomp LC3"),
    ("Senscomp OPC", "Senscomp OPC"),
)
MODEL_CHOICES = (
    ("auto", "Automatic"),
    ("waveform", "2D-CNN + 1D-CNN, one channel, marked waveforms"),
    ("first-arrival", "1D-CNN first arrival, marked waveforms"),
    ("tall-waveform", "2D-CNN softmax, marked waveforms"),
    ("center", "2D-CNN + 1D-CNN center, marked waveforms"),
    ("heatmap-4", "2D-CNN, 4 spatial layers, unmarked heatmaps"),
    ("heatmap-tall", "2D-CNN tall kernel, unmarked heatmaps"),
)
MODEL_NAMES = dict(MODEL_CHOICES)
_CACHE = {}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _maybe_run() -> int:
    if "maybe_run" not in _CACHE:
        blob = torch.load(FINAL / "2dcnn_1dcnn_one_channel__marked_waveforms.pt", map_location="cpu", weights_only=False)
        _CACHE["maybe_run"] = int(blob.get("maybe_run", MAYBE_FALLBACK))
    return _CACHE["maybe_run"]


def _ranges(n_signals: int, signal: int) -> dict[str, str]:
    mark = int(signal) - 1
    labels = np.full(n_signals, "Yes", dtype=object)
    labels[:mark] = "No"
    maybe_start = max(0, mark - _maybe_run())
    labels[maybe_start:mark] = "Maybe"
    return curves.signal_partitions(labels)


def _row(channel: str, signal: int, n_signals: int, time_s: str) -> dict:
    parts = _ranges(n_signals, signal)
    return {
        "channel": channel,
        "signal": signal,
        "age": f"{signal} min",
        "time_s": time_s,
        "n_signals": n_signals,
        "miss": "—",
        "no": parts["No"] or "—",
        "maybe": parts["Maybe"] or "—",
        "yes": parts["Yes"] or "—",
    }


def _ordered(channels: dict) -> list[tuple[str, np.ndarray]]:
    def key(item):
        digits = "".join(character for character in item[0] if character.isdigit())
        return (int(digits) if digits else 0, item[0])

    return sorted(channels.items(), key=key)


def choose_model(setup: str, model: str, has_heatmap: bool) -> tuple[str, str]:
    if model != "auto":
        return model, "You chose this model."
    # Assignments use the saved models on the shared held-out recordings.
    # Leave-one-setup weights were not saved, so they are not called here.
    if setup == "FA paste" and has_heatmap:
        return (
            "heatmap-tall",
            "FA paste uses the 2D-CNN tall kernel trained on unmarked heatmaps. On the held-out FA paste 0.4 recording its mean miss was 21.2 signals, the lowest of the saved models. It reads the uploaded picture.",
        )
    if setup == "FA paste":
        return (
            "first-arrival",
            "FA paste uses the 1D-CNN first arrival trained on marked waveforms when no picture is uploaded. On the held-out FA paste 0.4 recording its mean miss was 26.6 signals. Upload the heatmap picture to use the tall-kernel heatmap model instead.",
        )
    if setup == "LC3 8ch":
        return (
            "first-arrival",
            "LC3 8-channel uses the 1D-CNN first arrival trained on marked waveforms. On the held-out LC3 8-channel 0.4 recording its mean miss was 12.5 signals, the lowest of the saved models.",
        )
    if setup == "OPC 8ch":
        return (
            "waveform",
            "OPC 8-channel uses the 2D-CNN + 1D-CNN trained on one marked waveform channel. Across the two held-out recordings its mean miss was 14.4 signals (20.9 at water 0.3 and 7.9 at water 0.4). No other saved model stayed low on both.",
        )
    return (
        "waveform",
        "This setup has no held-out recording scored with the saved models, so Automatic uses the 2D-CNN + 1D-CNN trained on marked waveforms. On the four held-out recordings its mean miss was 26.0 signals.",
    )


def _onset():
    if "onset" not in _CACHE:
        _CACHE["onset"] = _module(CODE / "2dcnn_1dcnn_one_channel_marked_waveforms.py", "channel_onset_final")
    return _CACHE["onset"]


def _waveform():
    if "waveform" not in _CACHE:
        onset = _onset()
        model = onset.ChannelOnsetModel()
        blob = torch.load(FINAL / "2dcnn_1dcnn_one_channel__marked_waveforms.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        _CACHE["maybe_run"] = int(blob.get("maybe_run", MAYBE_FALLBACK))
        _CACHE["waveform"] = (onset, model)
    return _CACHE["waveform"]


def _format_emergence(start_s: float) -> str:
    return f"{start_s * 1e6:.0f} µs ({start_s:.6f} s)"


@torch.no_grad()
def _emergence_s(samples: np.ndarray, fs: float, signal: int) -> float:
    """Time inside the predicted signal, from the waveform model's time head."""
    onset, model = _waveform()
    waves, t0, t1 = onset.prepare_channel(samples, fs, WAVE_WINDOW)
    prepared = torch.tensor(np.asarray(waves, dtype=np.float32))
    _fraction, when = model.read(prepared)
    n_signals = curves._signal_count(waves)
    index = int(np.clip(int(signal) - 1, 0, n_signals - 1))
    return float(t0) + float(when[index]) * (float(t1) - float(t0))


def predict_waveform(mat_path: Path) -> list[dict]:
    onset, model = _waveform()
    fs, channels = curves.load_waveforms(mat_path)
    rows = []
    for name, samples in _ordered(channels):
        waves, t0, t1 = onset.prepare_channel(samples, fs, WAVE_WINDOW)
        signal, start_s = onset.predict(model, {"waves": waves, "t0": t0, "t1": t1})
        n_signals = curves._signal_count(waves)
        rows.append(_row(name, signal, n_signals, _format_emergence(start_s)))
    return rows


def _arrival():
    if "arrival" not in _CACHE:
        module = _module(CODE / "1dcnn_first_arrival_marked_waveforms.py", "arrival_interface")
        model = module.ArrivalModel()
        blob = torch.load(FINAL / "1dcnn_first_arrival__marked_waveforms.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        _CACHE["arrival"] = (module, model)
    return _CACHE["arrival"]


def predict_arrival(mat_path: Path) -> list[dict]:
    module, model = _arrival()
    fs, channels = curves.load_waveforms(mat_path)
    window = module.WINDOWS.get(mat_path.stem, module.DEFAULT_WINDOW)
    rows = []
    for name, samples in _ordered(channels):
        series = module.arrival_series(samples, fs, window)
        signal = module.predict_signal(model, {"series": series, "n_signals": int(series.shape[1])})
        emerged = _emergence_s(samples, fs, signal)
        rows.append(_row(name, signal, int(series.shape[1]), _format_emergence(emerged)))
    return rows


def _bend():
    if "bend" not in _CACHE:
        module = _module(CODE / "2dcnn_softmax_marked_waveforms.py", "bend_interface")
        model = module.BendModel()
        blob = torch.load(FINAL / "2dcnn_softmax__marked_waveforms.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        _CACHE["bend"] = (module, model)
    return _CACHE["bend"]


def predict_bend(mat_path: Path) -> list[dict]:
    module, model = _bend()
    onset = _onset()
    fs, channels = curves.load_waveforms(mat_path)
    rows = []
    for name, samples in _ordered(channels):
        waves, _t0, _t1 = onset.prepare_channel(samples, fs, WAVE_WINDOW)
        n_signals = curves._signal_count(waves)
        signal = module.predict_signal(model, {"waves": waves, "n_signals": n_signals})
        emerged = _emergence_s(samples, fs, signal)
        rows.append(_row(name, signal, n_signals, _format_emergence(emerged)))
    return rows


def _center():
    if "center" not in _CACHE:
        module = _module(CODE / "2dcnn_1dcnn_center_marked_waveforms.py", "center_interface")
        onset = _onset()
        module.MODEL_PATH = FINAL / "2dcnn_1dcnn_center__marked_waveforms.pt"
        model, mix_logit, blob = module._load_saved(onset)
        offsets = {str(key): float(value) for key, value in blob.get("offsets", {}).items()}
        _CACHE["center"] = (module, onset, model, mix_logit, offsets)
    return _CACHE["center"]


def _offset_for(offsets: dict[str, float], setup: str, channel: str) -> float:
    key = f"{setup}|{channel}"
    if key in offsets:
        return offsets[key]
    same_setup = [value for name, value in offsets.items() if name.startswith(setup + "|")]
    if same_setup:
        return float(np.median(same_setup))
    if offsets:
        return float(np.median(list(offsets.values())))
    return 0.0


def predict_center(mat_path: Path, setup: str) -> list[dict]:
    module, onset, model, mix_logit, offsets = _center()
    fs, channels = curves.load_waveforms(mat_path)
    window = module.WINDOWS.get(mat_path.stem, module.DEFAULT_WINDOW)
    rows = []
    for name, samples in _ordered(channels):
        waves, _t0, _t1 = onset.prepare_channel(samples, fs, WAVE_WINDOW)
        n_signals = curves._signal_count(waves)
        center = module.center_signal(samples, fs, window)
        record = {
            "waves": waves,
            "n_signals": n_signals,
            "center": center,
            "offset": _offset_for(offsets, setup, name),
        }
        signal = module.predict_signal(model, mix_logit, record)
        emerged = _emergence_s(samples, fs, signal)
        rows.append(_row(name, signal, n_signals, _format_emergence(emerged)))
    return rows


def _heatmap_images():
    if "heatmap_images" not in _CACHE:
        _CACHE["heatmap_images"] = _module(CODE / "2dcnn_4spatial_unmarked_heatmaps.py", "heatmap_interface")
    return _CACHE["heatmap_images"]


def _heatmap_four():
    if "heatmap_four" not in _CACHE:
        images = _heatmap_images()
        _CACHE["heatmap_four"] = images.load_model(FINAL / "2dcnn_4spatial__unmarked_heatmaps.pt")
    return _CACHE["heatmap_four"]


def _heatmap_tall():
    if "heatmap_tall" not in _CACHE:
        images = _heatmap_images()
        tall = _module(CODE / "2dcnn_tall_unmarked_heatmaps.py", "tall_heatmap_interface")
        model = tall.TallHeatmapModel()
        blob = torch.load(FINAL / "2dcnn_tall__unmarked_heatmaps.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        _CACHE["heatmap_tall"] = (tall, model)
    return _CACHE["heatmap_tall"]


def _channel_from_name(filename: str, channels: dict) -> str:
    stem = Path(filename).stem.lower().replace("_marked", "")
    for name in channels:
        if name.lower() == stem or stem.endswith(name.lower()):
            return name
    return Path(filename).stem


def _heatmap_signal(kind: str, picture: np.ndarray, n_signals: int) -> int:
    record = {"picture": picture, "n_signals": n_signals}
    if kind == "heatmap-4":
        return _heatmap_images().predict_signal(_heatmap_four(), record)
    tall, model = _heatmap_tall()
    return tall.predict_signal(model, record)


def predict_heatmap(kind: str, mat_path: Path, image_path: Path, image_name: str) -> list[dict]:
    images = _heatmap_images()
    fs, channels = curves.load_waveforms(mat_path)
    n_signals = int(next(iter(channels.values())).shape[0])
    signal = _heatmap_signal(kind, images.load_picture(image_path), n_signals)
    channel = _channel_from_name(image_name, channels)
    samples = channels.get(channel)
    emerged = _emergence_s(samples, fs, signal) if samples is not None else None
    shown = _format_emergence(emerged) if emerged is not None else "—"
    return [_row(channel, signal, n_signals, shown)]


def _unmarked_dir(folder: Path) -> Path:
    nested = folder / "unmarked"
    return nested if nested.is_dir() else folder


def predict_heatmap_folder(kind: str, mat_path: Path, folder: Path) -> list[dict]:
    images = _heatmap_images()
    fs, channels = curves.load_waveforms(mat_path)
    n_signals = int(next(iter(channels.values())).shape[0])
    rows = []
    for picture_path in sorted(_unmarked_dir(folder).glob("*.png")):
        channel = _channel_from_name(picture_path.name, channels)
        if channel not in channels:
            continue
        signal = _heatmap_signal(kind, images.load_picture(picture_path), n_signals)
        emerged = _emergence_s(channels[channel], fs, signal)
        rows.append(_row(channel, signal, n_signals, _format_emergence(emerged)))
    if not rows:
        raise ValueError("No heatmap pictures were found next to this recording.")
    return rows


def _labels() -> dict[tuple[str, str], int]:
    if "labels" not in _CACHE:
        table = {}
        with (DATA / "lr_wave_onsets.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["status"] == "yes" and row["signal_id"]:
                    table[(row["dataset"], row["channel"])] = int(float(row["signal_id"]))
        _CACHE["labels"] = table
    return _CACHE["labels"]


def _class_ids(n_signals: int, signal: int) -> np.ndarray:
    mark = int(signal) - 1
    labels = np.full(n_signals, 2, dtype=np.int16)
    if mark > 0:
        labels[:mark] = 0
    start = max(0, mark - _maybe_run())
    if mark > start:
        labels[start:mark] = 1
    return labels


def attach_scores(experiment: str, rows: list[dict]) -> dict | None:
    known = _labels()
    misses = []
    correct = 0
    total = 0
    exact = 0
    for row in rows:
        target = known.get((experiment, row["channel"]))
        if target is None:
            continue
        miss = abs(int(row["signal"]) - target)
        row["miss"] = str(miss)
        misses.append(miss)
        if int(row["signal"]) == target:
            exact += 1
        predicted = _class_ids(int(row["n_signals"]), int(row["signal"]))
        truth = _class_ids(int(row["n_signals"]), target)
        correct += int(np.count_nonzero(predicted == truth))
        total += int(row["n_signals"])
    if not misses:
        return None
    return {
        "mean_miss": f"{sum(misses) / len(misses):.1f}",
        "accuracy": f"{100.0 * correct / total:.1f}",
        "exact": f"{100.0 * exact / len(misses):.1f}",
        "exact_count": exact,
        "channels": len(misses),
    }


def infer_setup(name: str) -> str:
    text = name.lower()
    if "senscomp" in text and "lc3" in text:
        return "Senscomp LC3"
    if "senscomp" in text and "opc" in text:
        return "Senscomp OPC"
    if "fa paste" in text:
        return "FA paste"
    if "fa_" in text or text.startswith("fa "):
        return "FA 8ch"
    if "lc3 paste" in text:
        return "LC3 paste"
    if "lc3" in text:
        return "LC3 8ch"
    if "11 degree" in text:
        return "OPC 16ch"
    if "opc" in text:
        return "OPC 8ch"
    return "other"


def saved_recordings() -> list[tuple[str, str]]:
    found = []
    for split in ("train", "test"):
        folder = DATA / split
        if not folder.is_dir():
            continue
        for experiment in sorted(path.name for path in folder.iterdir() if path.is_dir()):
            if (folder / experiment / f"{experiment}.mat").is_file():
                found.append((f"{split}/{experiment}", f"{split}: {experiment}"))
    return found


def _saved_mat(value: str) -> Path | None:
    if value.count("/") != 1 or ".." in value:
        return None
    split, experiment = value.split("/", 1)
    if split not in {"train", "test"}:
        return None
    mat = DATA / split / experiment / f"{experiment}.mat"
    return mat if mat.is_file() else None


def _keep_upload(recording, heatmap) -> tuple[str, str]:
    token = secrets.token_hex(8)
    folder = UPLOADS / token
    folder.mkdir(parents=True)
    recording.save(folder / Path(recording.filename).name)
    if heatmap is not None and heatmap.filename:
        heatmap.save(folder / Path(heatmap.filename).name)
    return token, Path(recording.filename).name


def _recall(token: str) -> tuple[Path, Path | None] | None:
    if len(token) != 16 or any(character not in "0123456789abcdef" for character in token):
        return None
    folder = UPLOADS / token
    mats = list(folder.glob("*.mat")) if folder.is_dir() else []
    if not mats:
        return None
    pictures = [path for path in folder.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return mats[0], pictures[0] if pictures else None


def predict(
    mat_path: Path,
    image_path: Path | None,
    image_name: str,
    setup: str,
    model: str,
    picture_dir: Path | None = None,
) -> dict:
    has_picture = image_path is not None or (
        picture_dir is not None and any(_unmarked_dir(picture_dir).glob("*.png"))
    )
    kind, reason = choose_model(setup, model, has_picture)
    if kind in {"heatmap-4", "heatmap-tall"}:
        if image_path is not None:
            rows = predict_heatmap(kind, mat_path, image_path, image_name)
        elif picture_dir is not None:
            rows = predict_heatmap_folder(kind, mat_path, picture_dir)
        else:
            raise ValueError("This model reads a heatmap picture. Upload the .png, or choose the waveform model.")
    elif kind == "first-arrival":
        rows = predict_arrival(mat_path)
    elif kind == "tall-waveform":
        rows = predict_bend(mat_path)
    elif kind == "center":
        rows = predict_center(mat_path, setup)
    else:
        kind = "waveform"
        rows = predict_waveform(mat_path)
    scores = attach_scores(mat_path.stem, rows)
    return {
        "model_name": MODEL_NAMES.get(kind, kind),
        "reason": reason,
        "rows": rows,
        "scores": scores,
        "unscored": scores is None,
    }


def _page(**kwargs):
    values = {
        "setups": SETUPS,
        "model_choices": MODEL_CHOICES,
        "recordings": saved_recordings(),
        "chosen_setup": "other",
        "chosen_model": "auto",
        "chosen_saved": "",
        "kept": "",
        "kept_name": "",
        "error": None,
        "result": None,
    }
    values.update(kwargs)
    return render_template_string(PAGE, **values)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 800 * 1024 * 1024


@app.get("/")
def form():
    return _page()


@app.post("/")
def submit():
    setup = request.form.get("setup", "other")
    model = request.form.get("model", "auto")
    saved = request.form.get("saved", "")
    kept = request.form.get("kept", "")
    recording = request.files.get("recording")
    heatmap = request.files.get("heatmap")
    error = None
    result = None
    kept_name = ""
    try:
        if recording is not None and recording.filename:
            kept, kept_name = _keep_upload(recording, heatmap)
            saved = ""
            recalled = _recall(kept)
        elif saved:
            mat_path = _saved_mat(saved)
            if mat_path is None:
                raise ValueError("That saved recording is not in the final_models data folder.")
            recalled = (mat_path, None)
            kept_name = mat_path.name
        else:
            recalled = _recall(kept)
            if recalled is None:
                raise ValueError("Choose a saved recording, or upload a .mat file.")
            kept_name = recalled[0].name
        mat_path, image_path = recalled
        if setup == "other":
            setup = infer_setup(mat_path.stem)
        picture_dir = None if image_path is not None else mat_path.parent
        image_name = image_path.name if image_path is not None else ""
        result = predict(mat_path, image_path, image_name, setup, model, picture_dir)
    except Exception as exc:
        error = str(exc)
    return _page(
        chosen_setup=setup,
        chosen_model=model,
        chosen_saved=saved,
        kept=kept,
        kept_name=kept_name,
        error=error,
        result=result,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False)
