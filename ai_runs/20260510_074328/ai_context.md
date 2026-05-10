# AI Iteration Context

## 1. Iteration ID

20260510_074328

## 2. Previous Iteration

ai_runs/20260510_052426/

## 3. Previous Metrics Summary

Precision: 0.143105
Recall: 0.0876145
F1: 0.108687
Localization error: 0.189045
False positives: 3952
Missed detections: 6873
Main failure: Unavailable

## 4. Observed Problem

Training and evaluation completed, but the detection result is weak: F1=0.0952381, precision=0.0681922, recall=0.157839, false positives=16247, missed detections=6344.

## 5. Improvement Hypothesis

Because confidence fusion produced many false positives,
we add an optional BEV point-extraction distance suppression setting,
expecting precision and F1 to improve because nearby duplicate peaks are less likely to count as separate detections.

## 6. Changes Made

Changed files:
- src/evaluate_main.py: adds det_min_distance and logs point-extraction settings.
- scripts/commit_ai_runs.py: records point-extraction settings in ai_context.md.
- docs/experiment_protocol.md: marks point-extraction parameters as comparison-critical.

## 7. Training Configuration

dataset: WildTrack
views: 0,1,2
epochs: 10
batch_size: 2
learning_rate: Unavailable
max_frames: Unavailable
device: Unavailable
seed: Unavailable
checkpoint_path: /content/BEV_Track-Predict/outputs/train_multicam_mvdet_style_v3/model_final.pth
fusion_mode: confidence
train_command: python scripts/train_main.py --data_root /content/BEV_Track-Predict/wildtrack --epochs 10 --batch 2 --fusion_mode confidence

## 8. Evaluation Configuration

model_path: /content/BEV_Track-Predict/outputs/train_multicam_mvdet_style_v3/model_final.pth
views: 0,1,2
threshold: 0.05
distance_threshold: 3
min_distance: 3
nms_ksize: 3
max_preds: 200
thresholds: 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50
metrics_output: metrics.json
device: Unavailable
max_frames: Unavailable

## 9. Current Metrics

Precision: 0.0681922
Recall: 0.157839
F1: 0.0952381
Localization error: 0.192811
False positives: 16247
Missed detections: 6344
Status: success

## 10. Result Interpretation

Inconclusive. This run is a baseline measurement until a same-configuration comparison exists.

## 11. Next Iteration Recommendation

Next action:
Compare confidence fusion with det_min_distance enabled against the previous confidence run under the same WildTrack views and evaluation sweep.

Reason:
The latest confidence run gained recall but produced too many false positives; distance suppression directly targets duplicate nearby peaks.

Expected validation:
A new ai_runs timestamp whose metrics.json reports lower false positives and higher F1 than ai_runs/20260510_050623.

## 12. Do Not Do Next

Do not implement tracking.
Do not implement ReID.
Do not implement trajectory prediction.
Do not introduce BEVFormer, PETR, LSS, DETR3D, or other large BEV frameworks.
Do not claim model improvement until metrics are compared under the same configuration.
