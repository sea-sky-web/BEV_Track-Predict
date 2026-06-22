# [IMPROVEMENT][MEDIUM] Evaluation metrics don't align with MVDet standard — add MODA/MODP

## Labels
`improvement`, `medium-priority`, `evaluation`, `metrics`

## Priority
**P2 — Blocks valid comparison with published results**

---

## Problem Statement

The current `src/evaluate_main.py` computes:
- Precision, Recall, F1 (sweep over threshold → take best F1)
- Localization error (mean distance at best F1 threshold)

MVDet, MVDeTr, and all published multi-view detection papers report:
- **MODA** (Multiple Object Detection Accuracy)
- **MODP** (Multiple Object Detection Precision)

These metrics are fundamentally different:

| Property | Current (F1 sweep) | MODA / MODP |
|----------|-------------------|-------------|
| Standard | Not standard | CLEAR MOT standard |
| FP penalty | None (precision is ratio) | FP explicitly penalized in MODA |
| Threshold dependency | Best F1 picked post-hoc | Single threshold at 0.5m distance |
| Comparable to MVDet? | No | Yes |
| Library | Custom | `py-motmetrics` |

**Consequence:** We cannot determine whether the model is improving or whether
we are selecting a more favorable threshold. Best F1 ≈ 0.10 is meaningless as
an absolute number without knowing the threshold and the FP/FN distribution.

---

## MODA / MODP Definition

Given a distance threshold `d_thresh = 0.5 m`:

```
MODA = 1 - (FP + FN) / N_gt           (range: -∞ to 1.0; negative means more FP+FN than GT)
MODP = Σ (1 - d_i / d_thresh) / TP    (range: 0 to 1.0; higher is better localization)
```

Where:
- `N_gt`: total ground truth pedestrian count
- `FP`: false positives (predictions with no matching GT within d_thresh)
- `FN`: false negatives (GT with no matching prediction within d_thresh)
- `TP`: true positives (matched pairs)
- `d_i`: distance between matched prediction and GT

---

## Proposed Implementation

### Step 1 — Install `py-motmetrics`

```
# requirements.txt
motmetrics>=1.4.0
scipy>=1.10
```

### Step 2 — Add `src/metrics.py`

```python
"""CLEAR MOT metrics for BEV pedestrian detection (single-frame, multi-person)."""
import numpy as np
from scipy.spatial.distance import cdist

def compute_moda_modp(
    pred_pts: np.ndarray,    # (N_pred, 2) in meters [x, y]
    gt_pts:   np.ndarray,    # (N_gt, 2) in meters [x, y]
    d_thresh: float = 0.5,   # 0.5 m standard threshold
) -> dict:
    """
    Compute MODA and MODP for a single frame.
    Returns dict with keys: moda, modp, tp, fp, fn, n_gt.
    """
    n_gt   = len(gt_pts)
    n_pred = len(pred_pts)

    if n_gt == 0 and n_pred == 0:
        return dict(moda=1.0, modp=1.0, tp=0, fp=0, fn=0, n_gt=0)
    if n_gt == 0:
        return dict(moda=-n_pred / max(n_gt, 1), modp=0.0, tp=0, fp=n_pred, fn=0, n_gt=0)
    if n_pred == 0:
        return dict(moda=1 - n_gt / n_gt, modp=0.0, tp=0, fp=0, fn=n_gt, n_gt=n_gt)

    # Hungarian matching within d_thresh
    dist_mat = cdist(pred_pts, gt_pts, metric="euclidean")   # (N_pred, N_gt)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(dist_mat)
    matched_dists = dist_mat[row_ind, col_ind]
    valid = matched_dists <= d_thresh

    tp = int(valid.sum())
    fp = n_pred - tp
    fn = n_gt  - tp

    moda = 1.0 - (fp + fn) / n_gt
    modp = float(np.mean(1.0 - matched_dists[valid] / d_thresh)) if tp > 0 else 0.0

    return dict(moda=moda, modp=modp, tp=tp, fp=fp, fn=fn, n_gt=n_gt)


def aggregate_metrics(per_frame: list[dict]) -> dict:
    """Average MODA/MODP across frames using count-weighted aggregation."""
    total_gt = sum(f["n_gt"] for f in per_frame)
    total_fp = sum(f["fp"]   for f in per_frame)
    total_fn = sum(f["fn"]   for f in per_frame)
    total_tp = sum(f["tp"]   for f in per_frame)

    moda = 1.0 - (total_fp + total_fn) / max(total_gt, 1)
    modp = (
        sum(f["modp"] * f["tp"] for f in per_frame) / max(total_tp, 1)
    )
    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    return dict(
        MODA=round(moda, 4),
        MODP=round(modp, 4),
        Precision=round(precision, 4),
        Recall=round(recall, 4),
        F1=round(f1, 4),
        TP=total_tp, FP=total_fp, FN=total_fn, N_GT=total_gt,
    )
```

### Step 3 — Integrate into `src/evaluate_main.py`

```python
from src.metrics import compute_moda_modp, aggregate_metrics

# In evaluate_detection():
per_frame_results = []
for frame_idx, (imgs, map_gt, _) in enumerate(val_loader):
    pred_heatmap = model(imgs)
    pred_pts = extract_detections(pred_heatmap, threshold=0.4)   # fixed threshold
    gt_pts   = heatmap_to_world_coords(map_gt, cfg)
    per_frame_results.append(compute_moda_modp(pred_pts, gt_pts, d_thresh=0.5))

metrics = aggregate_metrics(per_frame_results)
print(f"MODA: {metrics['MODA']:.4f}  MODP: {metrics['MODP']:.4f}  F1: {metrics['F1']:.4f}")
```

### Step 4 — Convert BEV grid coordinates to meters

```python
def grid_to_world_meters(ix: float, iy: float, cfg) -> tuple[float, float]:
    """Convert BEV grid index to world coordinates in meters."""
    x = cfg.origin_x + ix * cfg.STEP_M   # origin_x = -3.0 m
    y = cfg.origin_y + iy * cfg.STEP_M   # origin_y = -9.0 m
    return x, y
```

---

## Acceptance Criteria

- [ ] `src/metrics.py` implemented with `compute_moda_modp()` and `aggregate_metrics()`
- [ ] `src/evaluate_main.py` reports MODA and MODP in addition to existing metrics
- [ ] Unit test: `test_metrics.py` verifies MODA=1.0 for perfect predictions and MODA<0 for all FP
- [ ] `metrics.json` in `ai_runs/` includes `MODA` and `MODP` keys
- [ ] `docs/experiment_protocol.md` updated: MODA and MODP are required fields in every `ai_context.md`
- [ ] The target for this project is: **MODA ≥ 0.75, MODP ≥ 0.70** (matching MVDet within 15 points)

---

## References

- CLEAR MOT metrics: Bernardin & Stiefelhagen (2008)
- py-motmetrics: https://github.com/cheind/py-motmetrics
- MVDet paper Table 2: MODA 88.2%, MODP 75.7% on Wildtrack
- MVDeTr paper Table 1: MODA 93.2%, MODP 81.4% on Wildtrack
