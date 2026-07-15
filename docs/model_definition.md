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
→ shared ResNet-18 image encoder by default
→ per-view image features
→ geometry-based BEV projection
→ spatial-aware multi-view confidence fusion
→ BEV decoder
→ BEV pedestrian occupancy heatmap
→ BEV pedestrian point extraction
```

This architecture must remain MVDet-style and geometry-guided.

The current model must not be replaced by BEVFormer, PETR, LSS, or other large autonomous-driving BEV frameworks.

The default backbone is stride-8 ResNet-18 with ImageNet weights. ResNet-50
remains available only as a legacy explicit option.

---

## 7. Fixed Model Improvement

The only model-level improvement defined in the current stage is:

> Spatial-aware multi-view confidence fusion.

The baseline fusion is naive multi-view BEV concatenation.

The improved default fusion is `confidence_v2`, which predicts per-view
BEV-space attention weights from the joint multi-view representation. The
legacy per-view confidence module remains available as `confidence_v1`.

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
- MODA.
- MODP.
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

These items must not be implemented under Module 1.

---

## 10. Success Criteria (Module 1)

The current model is considered valid only if it can:

1. Load WildTrack multi-view samples.
2. Use selected camera views.
3. Construct BEV features using camera projection.
4. Predict BEV pedestrian occupancy heatmaps.
5. Extract BEV pedestrian point detections.
6. Report Precision, Recall, F1 score, MODA, MODP, and localization error.
7. Compare naive fusion with spatial-aware confidence fusion.
8. Save metrics for experiment comparison.

Quantitative targets (MODA ≥ 0.30, SNR ≥ 1.0, etc.) and visual acceptance
criteria for the BEV heatmap are defined in `docs/training_goals.md`.

---

## 11. Module 2 Stage Boundary

Module 2 builds on the frozen Module 1 detector. The Module 1 checkpoint,
configuration, and evaluation pipeline are immutable — Module 2 code must not
modify any file under Sections 1–10.

### 11.1 Module 2 Scope

Module 2 includes:

- Temporal annotation reading with `personID` for GT trajectory construction.
- World-coordinate conversion: `positionID` → full grid → reduced grid → meters.
- Multi-object tracking by detection (Kalman + Hungarian, nearest-neighbor baseline).
- BEV occupancy field and velocity field construction from tracked positions.
- Non-learning prediction baselines (persistence, constant velocity, field advection, oracle).
- ConvLSTM-based spatiotemporal field prediction (future stage).
- End-to-end evaluation with detection → tracking → prediction error decomposition.

### 11.2 Module 2 Excludes

- Modification of Module 1 detector, backbone, fusion, or training pipeline.
- Cross-camera ReID (unless justified by IDSW analysis).
- BEVFormer, PETR, LSS, DETR3D, or large Transformer architectures.
- Datasets other than WildTrack.
- Real deployment, crowd control, or risk decision-making.

### 11.3 Module 2 Code Location

All Module 2 code resides in `src/temporal/`. Module 1 source files under
`src/` (dataset.py, models.py, trainer.py, train_main.py, evaluate_main.py,
loss.py, metrics.py, geometry.py, config.py, calibration.py, augmentation.py,
utils.py) must not be modified for Module 2 purposes.

### 11.4 Module 2 Input

Module 2 consumes:

- Frozen detector BEV point detections (JSONL with world coordinates).
- WildTrack GT annotations with `personID` and `positionID`.

Module 2 does not consume raw images or BEV heatmaps directly.

### 11.5 Module 2 Output

- Tracked pedestrian trajectories in world coordinates (JSONL).
- BEV occupancy field and velocity field (NPZ, shape `[5, 120, 360]`).
- Future field predictions at horizons 0.5s, 1.0s, 2.0s.
- Individual trajectory predictions derived from predicted velocity fields.
- Evaluation metrics: MOTA, IDF1, IDSW, occupancy AUPRC, velocity EPE, ADE, FDE.

### 11.6 Module 2 Coordinate Convention

The only canonical coordinate is `(world_x_m, world_y_m)` in meters.

```text
row_full = positionID mod 480
col_full = positionID div 480
world_x_m = ORIGINE_X_M + (row_full + 0.5) × 0.025
world_y_m = ORIGINE_Y_M + (col_full + 0.5) × 0.025
```

Reduced grid (120 × 360, 0.1 m/cell) is a derived representation.
All intermediate coordinates must carry explicit `_full` or `_reduced` suffixes.

### 11.7 Module 2 Time Convention

```text
frame_rate = 2 Hz
timestamp_s = frame_index / 2.0
dt = 0.5 s
```

Fixed time splits (no random shuffling):

| Split | Frames | Count |
|---|---:|---:|
| Train | 0–319 | 320 |
| Validation | 320–359 | 40 |
| Test | 360–399 | 40 |
