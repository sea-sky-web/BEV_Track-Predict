# Second Stage Innovation Review

> Purpose: define the post-release innovation direction with enough evidence,
> and explicitly record assumptions that may be wrong before any model change is
> treated as valid progress.
>
> Current decision: release baseline is treated as frozen. This document does
> not redefine the release result. It defines the next research direction and
> the pre-flight checks required before claiming improvement.

---

## 1. Current Release Context

The current documented release baseline is:

```text
backbone: resnet18
fusion_mode: concat
optimizer: SGD lr=0.1 momentum=0.5 wd=5e-4
scheduler: OneCycleLR max_lr=0.1
epochs: 10
train frames: 360
test frames: 40, frame_start=360
augment: false
NMS: det_min_distance=6.0 reduced BEV cells
threshold: 0.400
MODA: 0.857
Precision: 0.918
Recall: 0.889
F1: 0.903
TP / FP / FN: 869 / 53 / 83
```

This result comes from `docs/daily-log.md` and `docs/active_plan.md`.
The second stage should use this release as the performance baseline, not keep
re-tuning release post-processing.

Important: local `git tag --list` currently returns no release tag. Therefore
the release baseline must be identified by explicit commit, run ID, checkpoint,
and evaluation JSON before it is used as a formal comparison target. Do not rely
on the word "release" alone.

---

## 2. Literature Basis

### 2.1 WildTrack Scene Assumption

WildTrack is a seven-static-camera, calibrated, overlapping multi-camera
pedestrian dataset. Its central value is that high-precision camera calibration
allows algorithms to exploit cross-view geometry.

Reference:
- WILDTRACK: https://arxiv.org/abs/1707.09299

Implication for this project:
- It is reasonable to design an innovation that uses fixed camera geometry.
- It is not necessary to introduce tracking, ReID, trajectory prediction, or a
  large autonomous-driving BEV framework to exploit the scene.

### 2.2 MVDet Baseline

MVDet projects per-view CNN feature maps onto a common ground plane, concatenates
the projected features, adds coordinate channels, then applies ground-plane
convolutions and peak extraction. It reports 88.2% MODA on WildTrack.

Reference:
- MVDet: https://arxiv.org/abs/2007.07247

Implication for this project:
- The release `concat` baseline is not a weak placeholder; it is the core MVDet
  idea.
- A second-stage innovation must explain why it improves view aggregation or
  spatial ambiguity beyond this strong concat baseline.

### 2.3 MVDeTr Observation

MVDeTr argues that after feature maps are projected onto the ground plane,
standard convolution applies the same computation regardless of object location,
even though projection distortion varies by position and camera. It uses a
shadow transformer to attend differently across positions and cameras.

Reference:
- MVDeTr: https://arxiv.org/abs/2108.05888

Implication for this project:
- The research opportunity is position- and camera-dependent aggregation.
- Directly importing a transformer would violate the current lightweight
  MVDet-style boundary and increase risk.
- A smaller, project-compatible version is to inject geometry-derived
  reliability signals into the existing view-fusion path.

---

## 3. Proposed Innovation

### Name

Geometry-Reliability Prompted Fusion.

### Claim

For each projected BEV feature map, not all view/cell pairs are equally reliable.
A BEV location may be outside a camera's feature plane, close to the image border,
or strongly distorted by the homography. The current model has the information
needed to compute this, but the fusion module does not receive it.

### Current Code Gap

The current model already:

- builds per-view homographies in `src/train_main.py` and `src/evaluate_main.py`;
- warps each view's image feature map into BEV in `src/models.py`;
- computes validity and in-bounds information inside `warp_perspective_torch`;
- supports learned view weights through `confidence_v2`.

But the model currently does not:

- expose per-cell valid masks to fusion;
- expose border distance or projective distortion to fusion;
- distinguish a zero feature caused by out-of-view padding from a legitimate
  zero-valued feature;
- provide a geometry prior to `confidence_v2`.

### Proposed Rule

For each view `v` and BEV cell `(x, y)`, compute geometry metadata:

```text
M_v(x,y) = [
  valid_mask,
  source_border_margin,
  log_projective_scale,
  normalized_coverage_count
]
```

Then fuse projected features by:

```text
feature_score_v = learned_score(B_1, ..., B_V)_v
geometry_score_v = small_mlp(M_v)
score_v = feature_score_v + beta * geometry_score_v
w_v = softmax_v(score_v)
B_fused = sum_v w_v * B_v
```

The geometry score should be optional and switchable. `beta` should start at
`1.0`, with ablation values `0.0` and `learned`.

---

## 4. Why This Is Feasible in This Codebase

### 4.1 Low implementation surface

Expected files:

- `src/geometry.py`: add a function that returns per-view BEV sampling metadata
  using the same inverse-homography logic as `warp_perspective_torch`.
- `src/models.py`: add `GeometryReliabilityFusion` and a new fusion mode such as
  `geo_confidence_v1`.
- `src/train_main.py`: build and pass geometry metadata to the model.
- `src/evaluate_main.py`: build the same metadata at evaluation time.
- `scripts/visualize_fusion_weights.py`: optionally visualize learned weights
  against reliability maps.

No dataset format, label format, tracking output, ReID branch, trajectory target,
or large BEV architecture is required.

### 4.2 Memory and compute cost

With 7 views, 4 metadata channels, and reduced BEV size roughly `120 x 360`, the
metadata tensor is:

```text
7 * 4 * 120 * 360 = 1,209,600 float values
```

This is about 4.8 MB in float32 before batching. It is static per camera setup
and can be registered as a buffer. The cost is small compared with projected
512-channel BEV features.

### 4.3 Compatibility with the project boundary

This remains inside:

```text
multi-view image features
→ geometry-based BEV projection
→ multi-view BEV feature fusion
→ BEV heatmap
→ point detection
```

It does not introduce:

- tracking;
- ReID;
- trajectory prediction;
- occupancy flow;
- BEVFormer / PETR / LSS / DETR3D;
- extra datasets;
- synthetic data.

---

## 5. Red-Team Review: What May Be Wrong

This section is mandatory. The previous proposal assumed too much. The following
issues can invalidate the innovation conclusion if not checked first.

### P0. Current documentation and code are inconsistent

#### P0.1 `img_head mid_ch` may not match the claimed release architecture

`docs/active_plan.md` says:

```text
img_head mid_ch: 64
```

and records:

```text
img_head mid_ch 128→64 | #77 | 架构对齐
```

But current `src/models.py` still instantiates:

```python
self.img_head = ImgHeadFoot(in_ch=feat_ch, mid_ch=128)
```

This is not a small typo. If the release run used current main, then the
architecture may not be fully aligned with the documented MVDet configuration.
If the release run used a different remote state, current main is not the same as
the release.

Required check:

```bash
rg -n "ImgHeadFoot\\(in_ch=feat_ch" src/models.py
```

Then verify the exact commit and model state used by the release run.

Do not start a fusion innovation until this is resolved or explicitly accepted as
non-blocking.

#### P0.2 Protocol files referenced by model definition are missing

`docs/model_definition.md` references:

```text
docs/experiment_protocol.md
docs/evaluation_protocol.md
```

but those files are not present in the current checkout. The available protocol
file is `docs/experiment_iteration_protocol.md`.

This matters because the evaluation and experiment contract is fragmented.
Before claiming a new method improves the release, the exact evaluation command,
fixed threshold/NMS settings, and checkpoint source must be written in one
authoritative place.

#### P0.3 `docs/daily-log.md` references a missing methodology document

`docs/daily-log.md` says a `docs/research-methodology.md` was created, but that
file is absent.

This is a documentation integrity issue. It suggests that some research rules may
exist only in conversation or external state, not in the repository.

#### P0.4 README and config defaults are stale relative to release execution

`README.md` still describes older defaults such as:

```text
lr=0.05
OneCycleLR max_lr=0.05
augmentation enabled
NMS radius 20 cells
```

`configs/exp_colab.yaml` still describes:

```text
fusion_mode: confidence_v2
optimizer: adam
scheduler: cosine
augment: true
```

But the GitHub workflow currently runs through `scripts/colab_train.py`, which
hard-codes:

```text
fusion_mode: concat
augment: false
optimizer: sgd
scheduler: onecycle
lr_init: 0.1
max_frames: 360
eval frame_start: 360
eval max_frames: 40
det_min_distances: 3.0,4.0,5.0,6.0,7.0,8.0
```

Therefore, `configs/exp_colab.yaml` is not the truth for the current Colab
workflow. The execution command is the truth.

#### P0.5 Fusion comparison is confounded by BEV head changes

Current `src/models.py` uses:

```text
fusion_mode == concat       -> MVDetMapClassifier
fusion_mode != concat       -> BEVHeadDilated
```

So comparing `concat` against `confidence_v2` changes both:

- fusion strategy;
- BEV decoder/head architecture and parameterization.

This is a serious attribution problem. If `confidence_v2` improves or regresses,
we cannot say whether the cause is fusion or the head.

Required experimental design:

1. Performance comparison:
   - release `concat` vs proposed method.
2. Attribution comparison:
   - existing `confidence_v2` vs `geo_confidence_v1` under the same BEV head.

Do not claim "geometry reliability improves over concat" as a pure fusion claim
unless the head confound is controlled.

### P1. The geometry prior may be redundant

Out-of-bounds projections are already zero-padded by `grid_sample`. It is
possible that the BEV head already learns to ignore padded regions.

Counter-check:

- train `confidence_v2` and `geo_confidence_v1` with identical head and
  evaluation;
- inspect whether reliability maps correlate with learned weights;
- run `beta=0.0` as an ablation.

If `beta=0.0` and `beta=1.0` are indistinguishable, the geometry prior is not
contributing.

### P1. Projective distortion features may be noisy

A homography Jacobian or projective scale can become numerically unstable near
singular regions or image borders.

Mitigation:

- use log scale;
- clamp values;
- normalize per view;
- start with only `valid_mask` and `border_margin` before adding Jacobian terms.

### P1. Global MODA may not reveal the intended benefit

If the method only improves low-coverage or distorted areas, the global 40-frame
test split may hide the effect.

Required extra metrics:

- MODA / FP / FN by BEV coverage count;
- MODA / FP / FN by source border margin bins;
- errors in high-density vs low-density frames;
- reliability-weight visualization for several frames.

These metrics are diagnostic. The headline metric still must include global
MODA, Precision, Recall, F1, MODP, localization error, FP, and FN.

### P1. Fixed post-processing must be separated from diagnostic sweeps

The release result was obtained after sweeping NMS/threshold. If the innovation
also picks its best threshold from the test split, the comparison may overfit
the 40-frame evaluation set.

Required rule:

- headline comparison uses release-fixed `threshold=0.400` and
  `det_min_distance=6.0`;
- NMS/threshold sweep is allowed only as diagnostic analysis.

---

## 6. Required Pre-Flight Checks

Before implementing `geo_confidence_v1`, complete these checks:

### Check 1: Identify the frozen release artifact

Record:

```text
release commit:
release checkpoint path or artifact:
release run ID:
release eval JSON:
release command:
```

If no artifact exists, the release baseline is a documented result, not a
replayable artifact. That must be stated in any paper or report.

### Check 2: Resolve architecture mismatch

Decide one of:

```text
A. Fix src/models.py to instantiate ImgHeadFoot(..., mid_ch=64), then rerun.
B. Update docs to admit the release used mid_ch=128.
C. Prove from the release artifact that the run used a different code state.
```

### Check 3: Write one authoritative evaluation contract

Until missing protocol files are restored, use this explicit contract:

```text
test split: frame_start=360, max_frames=40
views: 0,1,2,3,4,5,6
threshold: 0.400 for headline
det_min_distance: 6.0 for headline
MODA matching distance: 0.5m
metrics: Precision, Recall, F1, MODA, MODP, loc_err_m, TP, FP, FN
```

### Check 4: Define fair baselines

Use two comparisons:

```text
Performance target:
release concat vs geo_confidence_v1

Attribution target:
confidence_v2 vs geo_confidence_v1
```

If possible, add a third control:

```text
geometry metadata passed but beta=0.0
```

---

## 7. Minimal Implementation Plan After Checks Pass

### Step 1: Geometry metadata extraction

Add a pure function in `src/geometry.py`:

```text
compute_bev_sampling_meta(proj_mat, src_hw, dst_hw) -> Tensor[C,H,W]
```

Start with:

- `valid_mask`;
- `border_margin`;
- `coverage_count` computed across views outside the function.

Postpone Jacobian scale until the first two are verified.

### Step 2: Fusion module

Add `GeometryReliabilityFusion` in `src/models.py`.

Inputs:

```text
feats_bev: (B,V,C,H,W)
geo_meta:  (1,V,G,H,W)
```

Output:

```text
fused: (B,C,H,W)
```

Rule:

```text
feature_score = existing confidence_v2 score path
geometry_score = small Conv2d/MLP on metadata
weights = softmax(feature_score + beta * geometry_score, dim=view)
```

### Step 3: CLI and mode

Add a new fusion mode:

```text
geo_confidence_v1
```

Do not replace `concat` or `confidence_v2`.

### Step 4: Visualization

Extend `scripts/visualize_fusion_weights.py` to save:

- learned per-view weights;
- valid masks;
- border margins;
- optionally `weight - normalized_reliability` residual maps.

### Step 5: Validation

Local smoke:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/bevtrack_pycache python -m compileall src scripts tests
PYTHONPATH=src pytest tests/test_smoke_forward.py tests/test_geometry.py -v
```

Formal Colab:

```bash
python scripts/colab_train.py --epochs 10 --max_frames 360 --bev_pos_weight 1.0
python src/evaluate_main.py ... --fusion_mode geo_confidence_v1 --report_detection ...
```

The exact Colab command must be copied into the run's `ai_context.md`.

---

## 8. Success and Failure Criteria

### Success

The proposal is considered promising only if:

- global MODA is not worse than release by more than 0.005;
- `geo_confidence_v1` improves over `confidence_v2` under the same BEV head;
- FP does not increase sharply from the release value of 53;
- reliability maps and learned weights are qualitatively consistent;
- all metrics are recorded in a timestamped `ai_runs` directory.

### Strong success

```text
MODA >= 0.882
or
MODA improves by >= 0.015 with no FP/FN tradeoff collapse
```

### Failure

The proposal should be rejected or reworked if:

- it only improves under threshold/NMS sweep but not fixed release settings;
- it improves over `concat` but not over `confidence_v2`;
- learned weights remain uniform or unrelated to reliability maps;
- FP increases enough to erase recall gains;
- results cannot be reproduced from a timestamped run.

---

## 9. Final Recommendation

The innovation direction is still reasonable, but the earlier proposal was too
optimistic. The largest hidden risk is not the geometry prior itself; it is that
the current documentation, config, and model code disagree about what the release
actually is.

Therefore the next action should not be "implement the new fusion module"
immediately. The next action should be:

```text
Perform a release-consistency audit and fix or explicitly document all P0
contradictions, especially img_head mid_ch and the fusion/head confound.
```

Only after that audit should `geo_confidence_v1` be implemented and evaluated.

