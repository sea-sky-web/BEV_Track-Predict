# [BUG][HIGH] Training hard-capped at 300 frames and 3/7 views — insufficient for convergence

## Labels
`bug`, `high-priority`, `training`, `data`

## Priority
**P1 — Directly limits the information available during training**

---

## Problem Statement

Two hard-coded limits in the current configuration artificially constrain training:

### 1. `max_frames = 300`

```python
# src/config.py
max_frames: int = 300
```

Wildtrack provides approximately **2,000 labeled frames** (train: ~1,440, val: ~560).
Capping at 300 means the model trains on only **~17% of available training data**.

With 300 frames × 10 epochs = 3,000 forward passes. MVDet trains on ~1,440 frames
× 10 epochs = 14,400 forward passes — **4.8× more gradient updates**.

### 2. Views 0, 1, 2 only (3 of 7 cameras)

```python
# scripts/run_colab_exp.py / configs/exp_colab.yaml
--views 0,1,2
```

Wildtrack has 7 synchronized cameras arranged around a pedestrian plaza. Using only
3 cameras means:
- ~57% of the scene volume has no second-view confirmation
- Pedestrians in camera-blind zones appear in ground truth but not in features
- The concat/confidence fusion module receives 3×512 instead of 7×512 channels

This creates **irreducible false negatives**: the GT heatmap has positive peaks where
the network has only zero-padded BEV features.

---

## Root Causes

### `max_frames` cap

The cap was introduced for speed during early prototyping on limited hardware.
It remains in place as a default, making every experiment since then data-starved.

### View selection

The `valid_ratio` filter in `train_main.py` was introduced to drop views where
the homography projects mostly outside the image. However the threshold is too
aggressive, dropping views that have partial but still useful coverage.

---

## Proposed Fix

### Part A — Remove the `max_frames` cap for full-data runs

```python
# src/config.py
max_frames: int = -1   # -1 means "use all frames"
```

```python
# src/dataset.py — WildtrackMVDetDataset.__init__
if cfg.max_frames > 0:
    frame_list = frame_list[:cfg.max_frames]
# else: use all frames
```

Keep `max_frames` as an option for smoke tests (`--max_frames 10`) but
never use it for a real experiment.

### Part B — Enable all 7 views; fix valid_ratio threshold

```python
# src/train_main.py — view selection logic
# Before: threshold drops views with valid_ratio < 0.3
# After: log valid_ratio per view and only drop if valid_ratio < 0.05
# This retains all 7 Wildtrack cameras since none are nearly fully occluded

valid_views = [v for v in range(7) if compute_valid_ratio(proj_mats[v]) >= 0.05]
# Expected result: all 7 views pass
```

### Part C — Staged view expansion (incremental verification)

To verify that more views monotonically improve performance:

```
Experiment A: views [0,1,2]          -- current baseline
Experiment B: views [0,1,2,3,4]      -- +2 views
Experiment C: views [0,1,2,3,4,5,6]  -- all 7 views
```

Each experiment uses **identical hyperparameters**, differing only in view count.
Results should be recorded as a view-ablation table in `ai_runs/`.

### Part D — Update the Colab YAML

```yaml
# configs/exp_colab.yaml
train_cmd: >
  python src/train_main.py
    --data_root ${DATA_ROOT}
    --views 0,1,2,3,4,5,6
    --max_frames -1
    --fusion_mode confidence
    --pretrained true
    --alpha 1.0
    --epochs 10
```

---

## Acceptance Criteria

- [ ] `max_frames = -1` is the default and maps to "use all frames"
- [ ] Default view set is all 7 Wildtrack cameras
- [ ] `valid_ratio` threshold documented and set to 0.05 (not 0.30)
- [ ] A 3-run view ablation experiment is recorded in `ai_runs/`
- [ ] Training with all frames does not OOM on Colab A100 (expected ~6 GB VRAM at batch=1)
- [ ] `docs/experiment_protocol.md` updated: all experiments must state frame count and view set

---

## Memory / Speed Notes

| Config | VRAM (est.) | Time/epoch (A100) |
|--------|-------------|-------------------|
| 3 views, 300 frames, batch=1 | ~4 GB | ~2 min |
| 7 views, 1440 frames, batch=1 | ~8 GB | ~15 min |
| 7 views, 1440 frames, batch=2 | ~14 GB | ~10 min |

Colab A100 has 40 GB VRAM — all configurations are feasible.
If running on local GPU (MX450, 4 GB), keep `--max_frames 300 --views 0,1,2`
for local smoke tests only.

---

## References

- Wildtrack dataset paper: 7 cameras, 400 frames annotated at 2fps = 2000 frames total
- MVDet experiments: trained on full Wildtrack train split (no frame cap)
