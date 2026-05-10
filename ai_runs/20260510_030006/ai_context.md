# AI Iteration Context

## 1. Iteration ID

20260510_030006

## 2. Previous Iteration

No previous formal iteration.

## 3. Previous Metrics Summary

Precision: Unavailable
Recall: Unavailable
F1: Unavailable
Localization error: Unavailable
False positives: Unavailable
Missed detections: Unavailable
Main failure: Unavailable

## 4. Observed Problem

Training and evaluation completed, but the detection result is weak: F1=0.109871, precision=0.102085, recall=0.118943, false positives=7881, missed detections=6637.

## 5. Improvement Hypothesis

Because the current run needs a controlled comparison before any model-level claim,
we preserve the training entrypoint and record the comparison-critical settings,
expecting the next iteration to compare metrics under the same dataset, views, max_frames, checkpoint rule, threshold sweep, and fusion_mode.

## 6. Changes Made

Changed files:
- scripts/run_colab_exp.py: records dataset, views, max_frames, fusion_mode, actual checkpoint_path, and train_command in metrics.json.
- scripts/commit_ai_runs.py: separates previous-run metrics from current-run metrics in ai_context.md.
- docs/iteration_records/ITERATION_002.md: records the diagnostic decision and change boundary.

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
fusion_mode: concat
train_command: python scripts/train_main.py --data_root /content/BEV_Track-Predict/wildtrack --epochs 10 --batch 2

## 8. Evaluation Configuration

model_path: /content/BEV_Track-Predict/outputs/train_multicam_mvdet_style_v3/model_final.pth
views: 0,1,2
threshold: 0.1
distance_threshold: Unavailable
metrics_output: metrics.json
device: Unavailable
max_frames: Unavailable

## 9. Current Metrics

Precision: 0.102085
Recall: 0.118943
F1: 0.109871
Localization error: 0.189854
False positives: 7881
Missed detections: 6637
Status: success

## 10. Result Interpretation

Inconclusive. This run is a baseline measurement until a same-configuration comparison exists.

## 11. Next Iteration Recommendation

Next action:
Run the next Colab training/evaluation after this logging change and verify that metrics.json contains dataset, views, max_frames, fusion_mode, checkpoint_path, and detection metrics.

Reason:
Without these fields, future model changes cannot be compared safely under the experiment protocol.

Expected validation:
A new ai_runs timestamp whose metrics.json has both detection metrics and comparison-critical configuration fields.

## 12. Do Not Do Next

Do not implement tracking.
Do not implement ReID.
Do not implement trajectory prediction.
Do not introduce BEVFormer, PETR, LSS, DETR3D, or other large BEV frameworks.
Do not claim model improvement until metrics are compared under the same configuration.
