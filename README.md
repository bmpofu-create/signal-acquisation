# Final models

This folder holds the models the prediction page runs, the code for those models, and the train and test recordings they were scored on.

Start the page from this folder:

```
python predict_interface.py
```

Then open http://127.0.0.1:8765

## How to use the page

1. Choose the experiment, or leave it on “Other or not sure.” If the recording name matches a known setup, the page picks that setup.
2. Leave the model on Automatic. That is the best saved model for the setup. Choose another model only when you want to compare.
3. Pick a saved train or test recording, or upload a `.mat` file. A heatmap picture is needed only when you upload a new file and the chosen model reads a picture. Saved recordings already include their pictures.
4. Click “Predict the curve.”

The recording stays selected. Change the model and click Predict again. You do not upload the file a second time.

Automatic choices:

| Setup | Model |
|---|---|
| FA paste, with a heatmap picture | 2D-CNN tall kernel, trained on unmarked heatmaps |
| FA paste, recording only | 1D-CNN first arrival, trained on marked waveforms |
| LC3 8-channel | 1D-CNN first arrival, trained on marked waveforms |
| OPC 8-channel | 2D-CNN + 1D-CNN, one channel, trained on marked waveforms |
| Every other setup | 2D-CNN + 1D-CNN, one channel, trained on marked waveforms |

## What the numbers mean

When the recording is in `data/lr_wave_onsets.csv`, the page reports two scores.

**Mean miss** is the average, over channels, of how many signals the prediction is away from the saved click. Lower is better.

**Accuracy** is the share of signals whose No, Maybe, or Yes label matches the label made from that click. Maybe is the 27 signals immediately before the click. Yes starts at the click. No is everything earlier. This percentage stays high because most signals are clearly No or clearly Yes. The page also reports the percentage of channels that land on the exact click.

A recording that is not in the click file still gets a prediction. Mean miss and accuracy are left blank because there is no click to compare with.

## What is in this folder

- `predict_interface.py` — the page
- `models/` — the six weight files. Each name is the network structure, then the data it was trained on.
- `code/` — the Python files for those models, named the same way
- `data/train` and `data/test` — each recording, with `unmarked/` and `marked/` heatmap pictures. The page predicts from the unmarked pictures. The marked pictures have the click drawn on them. The `.mat` recordings stay in those folders on this computer. They are not in the repository because each file is too large for git.
- `data/lr_wave_onsets.csv` — the click on each channel
- `data/split.csv` — which recording is train or test
- `scores/` — the train and test prediction tables from when the models were trained
- `models.csv` — which weight file is which model

The weight files are:

| File | Structure | Trained on |
|---|---|---|
| `2dcnn_1dcnn_one_channel__marked_waveforms.pt` | 2D-CNN then 1D-CNN, one channel | marked waveform recordings |
| `1dcnn_first_arrival__marked_waveforms.pt` | 1D-CNN on the first-arrival series | marked waveform recordings |
| `2dcnn_softmax__marked_waveforms.pt` | 2D-CNN with a softmax over signals | marked waveform recordings |
| `2dcnn_1dcnn_center__marked_waveforms.pt` | 2D-CNN + 1D-CNN mixed with a center anchor | marked waveform recordings |
| `2dcnn_4spatial__unmarked_heatmaps.pt` | 2D-CNN with 4 spatial layers | unmarked heatmap pictures |
| `2dcnn_tall__unmarked_heatmaps.pt` | tall-kernel 2D-CNN | unmarked heatmap pictures |

The page loads the scripts in `code/`. Those files are copies of the model code. Do not run them to train. Training from this folder would look for weights and data in the wrong place. Use the page to predict.
