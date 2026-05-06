# Model Definition

## 1. Purpose

This repository implements a geometry-guided multi-view BEV pedestrian detector for the WildTrack dataset.

The model converts synchronized multi-view images into a BEV pedestrian occupancy heatmap and BEV pedestrian point detections.

The current model is the first-stage perception module for later tracking and prediction research, but tracking and prediction are not part of the current model definition.

---

## 2. Current Stage Boundary

The current stage is limited to BEV pedestrian detection.

The current stage includes:

- Multi-view image feature extraction.
- Geometry-based projection into BEV space.
- Multi-view BEV feature fusion.
- BEV pedestrian heatmap prediction.
- BEV pedestrian point extraction.
- BEV detection evaluation.

The current stage excludes:

- Multi-object tracking.
- Identity association.
- Trajectory prediction.
- Occupancy-flow prediction.
- Crowd risk prediction.
- Station operation decision-making.
- Large autonomous-driving BEV frameworks.

---

## 3. Dataset

The only dataset used in the current stage is WildTrack.

The model must use WildTrack synchronized multi-view images, camera calibration, and ground-plane pedestrian annotations.

No additional dataset is part of the current model definition.

---

## 4. Input Contract

For each frame `t`, the model input is:

```text
X_t = {I_t^v | v ∈ V_selected}
```

Where:

- `I_t^v` is the image from camera view `v`.
- `V_selected` is the selected set of WildTrack camera views.

The model also requires camera projection information:

```text
G = {G_v | v ∈ V_selected}
```

Where `G_v` maps image features between camera view `v` and the BEV coordinate system.

The model must use projection information for BEV feature construction.

---

## 5. Output Contract

The primary output is:

```text
H_t ∈ R^{1 × H_bev × W_bev}
```

Where `H_t` is the BEV pedestrian occupancy heatmap.

The required post-processed output is:

```text
P_t = {(x_i, y_i, score_i)}
```

Where each item represents one pedestrian detection in BEV coordinates.

No trajectory, identity, or future prediction output is part of the current model.

---

## 6. Architecture Contract

The current model architecture is:

```text
Multi-view images
→ shared image encoder
→ per-view image features
→ geometry-based BEV projection
→ spatial-aware multi-view confidence fusion
→ BEV decoder
→ BEV pedestrian occupancy heatmap
→ BEV pedestrian point extraction
```

This architecture must remain MVDet-style and geometry-guided.

The current model must not be replaced by BEVFormer, PETR, LSS, or other large autonomous-driving BEV frameworks.

---

## 7. Fixed Model Improvement

The only model-level improvement defined in the current stage is:

> Spatial-aware multi-view confidence fusion.

The baseline fusion is naive multi-view BEV fusion.

The improved fusion learns spatially varying view confidence weights in BEV space.

The fusion input is:

```text
B_t = {B_t^v | v ∈ V_selected}
```

Where `B_t^v` is the projected BEV feature map of view `v`.

The fusion output is:

```text
B_t^fused
```

The fusion rule is:

```text
B_t^fused(x, y) = Σ_v w_t^v(x, y) · B_t^v(x, y)
```

Where:

```text
Σ_v w_t^v(x, y) = 1
```

The confidence weights are learned from BEV features.

---

## 8. Loss and Evaluation Boundary

The model is trained to predict the BEV pedestrian occupancy heatmap from WildTrack annotations.

The exact loss function, training hyperparameters, and evaluation commands are defined in:

```text
docs/experiment_protocol.md
docs/evaluation_protocol.md
```

The model must be evaluated with BEV detection metrics, including:

- Precision.
- Recall.
- F1 score.
- Localization error.

Training loss alone is not a valid success criterion.

---

## 9. Explicit Non-goals

The following are not part of the current model definition:

- Full tracking.
- Re-identification.
- Trajectory forecasting.
- Occupancy-flow forecasting.
- Crowd density forecasting.
- Traffic control decision-making.
- Transformer-based large BEV perception frameworks.
- Synthetic data training.
- Real station deployment.

These items must not be implemented under the current model definition.

---

## 10. Success Criteria

The current model is considered valid only if it can:

1. Load WildTrack multi-view samples.
2. Use selected camera views.
3. Construct BEV features using camera projection.
4. Predict BEV pedestrian occupancy heatmaps.
5. Extract BEV pedestrian point detections.
6. Report Precision, Recall, F1 score, and localization error.
7. Compare naive fusion with spatial-aware confidence fusion.
8. Save metrics for experiment comparison.
