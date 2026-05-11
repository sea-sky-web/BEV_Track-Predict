# AI Iteration Context

## 1. Iteration ID

20260511_041025

## 2. Previous Iteration

ai_runs/20260511_030352/

## 3. Previous Metrics Summary

Precision: 0.0943133
Recall: 0.143767
F1: 0.113904
Localization error: 0.190993
False positives: 10400
Missed detections: 6450
Main failure: ALPHA=2.0 improved the previous confidence-fusion result, but it was only a single Colab run and needed a controlled reproduction before another model change.

## 4. Observed Problem

The previous ALPHA=2.0 confidence-fusion result was promising but not yet reproducibility-tested. This run repeated the same WildTrack views, confidence fusion, auxiliary loss alpha, loss weights, threshold sweep, and point-extraction settings.

## 5. Improvement Hypothesis

Because ALPHA=2.0 previously improved F1 under confidence fusion,
we repeat ALPHA=2.0 with explicit comparison-critical settings fixed,
expecting F1 to remain near the previous run if the gain is reproducible rather than a single-run fluctuation.

## 6. Changes Made

Changed files:
- No source code was changed in this run.
- Colab runtime setting: ALPHA=2.0, FUSION_MODE=confidence, BEV_POS_WEIGHT=1.0, BEV_NEG_WEIGHT=1.0, IMG_POS_WEIGHT=1.0, IMG_NEG_WEIGHT=1.0.
- Evaluation setting: det_min_distance=0.0, det_nms_ksize=3, det_max_preds=200, det_dist_thr=3.0, same threshold sweep.
- ai_runs/20260511_041025: archived this controlled reproduction result.

Archive note:
- Colab training and evaluation completed with return code 0.
- The notebook output was read from Google Drive autosave and archived through the GitHub connector with private Drive paths omitted.

## 7. Training Configuration

dataset: WildTrack
views: 0,1,2
epochs: 10
batch_size: 2
learning_rate: Unavailable
max_frames: 300
device: A100 Colab runtime
seed: Unavailable
checkpoint_path: /content/BEV_Track-Predict/outputs/train_multicam_mvdet_style_v3/model_final.pth
fusion_mode: confidence
alpha: 2.0
bev_pos_weight: 1.0
bev_neg_weight: 1.0
img_pos_weight: 1.0
img_neg_weight: 1.0
train_command: ALPHA=2.0 FUSION_MODE=confidence BEV_POS_WEIGHT=1.0 BEV_NEG_WEIGHT=1.0 IMG_POS_WEIGHT=1.0 IMG_NEG_WEIGHT=1.0 python scripts/run_colab_exp.py

## 8. Evaluation Configuration

model_path: /content/BEV_Track-Predict/outputs/train_multicam_mvdet_style_v3/model_final.pth
views: 0,1,2
threshold: 0.1
distance_threshold: 3.0
min_distance: 0.0
nms_ksize: 3
max_preds: 200
thresholds: 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50
metrics_output: runs/wildtrack_baseline/eval_metrics.json
device: A100 Colab runtime
max_frames: 300

## 9. Current Metrics

Precision: 0.0933065
Recall: 0.110846
F1: 0.101323
Localization error: 0.194681
False positives: 8114
Missed detections: 6698
Status: success

## 10. Result Interpretation

Inconclusive / partially reproduced. Compared with ai_runs/20260511_030352, precision stayed nearly the same (0.0943133 -> 0.0933065) and false positives decreased (10400 -> 8114), but recall dropped (0.143767 -> 0.110846), missed detections increased (6450 -> 6698), localization error worsened slightly (0.190993 -> 0.194681), and F1 dropped from 0.113904 to 0.101323.

This still remains above the earlier pre-alpha confidence-fusion run ai_runs/20260510_114806 (F1=0.091126), but the previous ALPHA=2.0 magnitude was not reproduced. The evidence supports that ALPHA=2.0 may help over the older confidence baseline, but it is not stable enough to justify a new model change yet.

## 11. Next Iteration Recommendation

Next action:
Run one controlled ALPHA=1.0 confidence-fusion counterpart with the exact same explicit loss weights and evaluation settings used in this run.

Reason:
The current repeat changed only the stochastic training sample path, but without an ALPHA=1.0 same-control counterpart on current main we cannot isolate whether the remaining gain over ai_runs/20260510_114806 is due to alpha or run variance.

Expected validation:
A new ai_runs timestamp with alpha=1.0, fusion_mode=confidence, views 0,1,2, max_frames=300, det_min_distance=0.0, det_nms_ksize=3, det_max_preds=200, det_dist_thr=3.0, and the same threshold sweep, compared directly against ai_runs/20260511_041025.

## 12. Do Not Do Next

Do not implement tracking.
Do not implement ReID.
Do not implement trajectory prediction.
Do not introduce BEVFormer, PETR, LSS, DETR3D, or other large BEV frameworks.
Do not change the dataset away from WildTrack.
Do not add another model or fusion change before the ALPHA=1.0 same-control counterpart is available.
