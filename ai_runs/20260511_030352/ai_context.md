# AI Iteration Context

## 1. Iteration ID

20260511_030352

## 2. Previous Iteration

ai_runs/20260510_114806/

## 3. Previous Metrics Summary

Precision: 0.0728182
Recall: 0.121731
F1: 0.0911259
Localization error: 0.194200
False positives: 11676
Missed detections: 6616
Main failure: Low F1 with many false positives under confidence fusion and BEV negative loss weighting.

## 4. Observed Problem

The previous valid confidence-fusion run with BEV negative loss weighting remained weak: F1=0.0911259, precision=0.0728182, recall=0.121731, false positives=11676, missed detections=6616.

## 5. Improvement Hypothesis

Because the current training logic follows the MVDet pattern of BEV heatmap supervision plus per-view auxiliary image heatmap supervision,
we ran confidence fusion with ALPHA=2.0,
expecting stronger pre-BEV per-view supervision to reduce feature information loss before projection and improve F1.

## 6. Changes Made

Changed files:
- No source code was changed in this run.
- Colab runtime setting: ALPHA=2.0 and FUSION_MODE=confidence.
- ai_runs/20260511_030352: archived this training/evaluation result.

Archive note:
- Colab training and evaluation completed with return code 0.
- Colab Secret token access timed out during push, so the final ai_runs files were completed through the GitHub connector from the observed notebook summary.

## 7. Training Configuration

dataset: WildTrack
views: 0,1,2
epochs: 10
batch_size: 2
learning_rate: Unavailable
max_frames: Unavailable
device: A100 Colab runtime
seed: Unavailable
checkpoint_path: /content/BEV_Track-Predict/outputs/train_multicam_mvdet_style_v3/model_final.pth
fusion_mode: confidence
alpha: 2.0
bev_pos_weight: 1.0
bev_neg_weight: 1.0
img_pos_weight: 1.0
img_neg_weight: 1.0
train_command: ALPHA=2.0 FUSION_MODE=confidence python scripts/run_colab_exp.py

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

Precision: 0.0943133
Recall: 0.143767
F1: 0.113904
Localization error: 0.190993
False positives: 10400
Missed detections: 6450
Status: success

## 10. Result Interpretation

Yes. Compared with ai_runs/20260510_114806 under confidence fusion, F1 improved from 0.0911259 to 0.113904, precision improved from 0.0728182 to 0.0943133, recall improved from 0.121731 to 0.143767, false positives decreased from 11676 to 10400, missed detections decreased from 6616 to 6450, and localization error improved from 0.194200 to 0.190993.

The result also slightly exceeds the earlier concat baseline F1 of 0.109871, but the next iteration should verify that alpha=2.0 is reproducible before adding a new model change.

## 11. Next Iteration Recommendation

Next action:
Run a repeat or narrow controlled alpha comparison for confidence fusion, preferably ALPHA=1.0 versus ALPHA=2.0 under the same evaluation sweep, to confirm the improvement is attributable to auxiliary supervision strength.

Reason:
ALPHA=2.0 produced the first confidence-fusion result above the concat baseline, but a single Colab run should be confirmed before introducing another model or loss change.

Expected validation:
A new ai_runs timestamp that keeps WildTrack, views 0,1,2, fusion_mode=confidence, det_min_distance=0.0, and the same threshold sweep fixed while comparing alpha settings.

## 12. Do Not Do Next

Do not implement tracking.
Do not implement ReID.
Do not implement trajectory prediction.
Do not introduce BEVFormer, PETR, LSS, DETR3D, or other large BEV frameworks.
Do not change the dataset away from WildTrack.
Do not add another model change before confirming the ALPHA=2.0 result is reproducible.
