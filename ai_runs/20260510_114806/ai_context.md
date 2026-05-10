# AI Iteration Context

## 1. Iteration ID

20260510_114806

## 2. Previous Iteration

ai_runs/20260510_074328/

## 3. Previous Metrics Summary

Precision: 0.0681922
Recall: 0.157839
F1: 0.0952381
Localization error: 0.192811
False positives: 16247
Missed detections: 6344
Main failure: Unavailable

## 4. Observed Problem

Training and evaluation completed, but the detection result is weak: F1=0.0911259, precision=0.0728182, recall=0.121731, false positives=11676, missed detections=6616.

## 5. Improvement Hypothesis

Because confidence fusion still produces too many false positives and point-extraction distance suppression did not improve F1,
we add optional BEV/image Gaussian MSE loss weights,
expecting precision and F1 to improve because background heatmap activations can be penalized more directly during training.

## 6. Changes Made

Changed files:
- src/train_main.py: adds switchable heatmap loss weight arguments.
- src/trainer.py: uses weighted Gaussian MSE only when loss weights differ from 1.0.
- scripts/run_colab_exp.py: passes loss weight overrides from Colab config or environment and records them in metrics.json.
- scripts/commit_ai_runs.py: records loss weights in ai_context.md.
- docs/experiment_protocol.md: marks training loss weights as comparison-critical.

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
bev_pos_weight: 1
bev_neg_weight: 2
img_pos_weight: 1
img_neg_weight: 1
train_command: python scripts/train_main.py --data_root /content/BEV_Track-Predict/wildtrack --epochs 10 --batch 2 --fusion_mode confidence --bev_neg_weight 2.0

## 8. Evaluation Configuration

model_path: /content/BEV_Track-Predict/outputs/train_multicam_mvdet_style_v3/model_final.pth
views: 0,1,2
threshold: 0.05
distance_threshold: 3
min_distance: 0
nms_ksize: 3
max_preds: 200
thresholds: 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50
metrics_output: metrics.json
device: Unavailable
max_frames: Unavailable

## 9. Current Metrics

Precision: 0.0728182
Recall: 0.121731
F1: 0.0911259
Localization error: 0.1942
False positives: 11676
Missed detections: 6616
Status: success

## 10. Result Interpretation

Inconclusive. This run is a baseline measurement until a same-configuration comparison exists.

## 11. Next Iteration Recommendation

Next action:
Compare confidence fusion with BEV negative loss weighting against the previous confidence runs under the same WildTrack views and evaluation sweep.

Reason:
The latest distance-suppressed run increased false positives, so the next controlled lever is training-time suppression of background heatmap activations.

Expected validation:
A new ai_runs timestamp whose metrics.json reports lower false positives and higher F1 than the prior confidence runs without distance suppression.

## 12. Do Not Do Next

Do not implement tracking.
Do not implement ReID.
Do not implement trajectory prediction.
Do not introduce BEVFormer, PETR, LSS, DETR3D, or other large BEV frameworks.
Do not claim model improvement until metrics are compared under the same configuration.
