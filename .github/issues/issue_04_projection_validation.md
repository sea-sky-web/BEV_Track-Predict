# [BUG][MEDIUM] Projection matrix correctness unverified — silent coordinate misalignment possible

## Labels
`bug`, `medium-priority`, `geometry`, `testing`

## Priority
**P2 — Silent error; may cause systematic spatial offset in all predictions**

---

## Problem Statement

The core geometric operation of this project is the homography-based projection of
image feature maps into BEV space. This is implemented in `src/geometry.py` via
`build_mvdet_proj_mat()` and applied in `src/models.py` via `warp_perspective_torch()`.

**No automated test or visualization validates that a known world-ground point
projects to the correct BEV grid cell.** The pipeline has multiple coordinate
system conventions that can silently conflict:

1. **OpenCV vs PyTorch coordinate order**: OpenCV uses `(row, col)` / `(y, x)`;
   PyTorch `grid_sample` uses normalized `(x, y)` in `[-1, 1]`. A transpose
   error here shifts every detection in a direction-dependent way.

2. **Wildtrack BEV grid origin convention**: The grid origin is at `(-3.0, -9.0)`
   meters. `build_mvdet_proj_mat()` must correctly chain:
   `worldgrid_indices → world_meters → camera_coords → image_pixels`.
   A sign error in the translation vector or a missing column removal (for Z=0
   ground plane) produces a plausible-looking projection that is offset by meters.

3. **Homography inverse direction**: `warp_perspective_torch()` applies the
   **inverse** homography to map BEV grid points back to image coordinates for
   `grid_sample`. If the matrix is inverted in the wrong direction (or not
   inverted at all), the warp appears to run but produces zeroed or scrambled features.

4. **Feature map scaling**: Intrinsics are scaled from original (1920×1080) to
   feature resolution (480×270) via `scale_intrinsics()`. An off-by-one in this
   scaling shifts all projections by up to 8 pixels (one stride-8 cell).

### Why this is hard to detect

All four failure modes produce non-zero, non-NaN feature maps in BEV space. The
model's loss decreases (it learns to predict near-zero everywhere, which minimizes
MSE on sparse GTs). Only a spatial visualization reveals that predictions are
systematically offset from ground truth.

---

## Proposed Fix

### Part A — Add a deterministic projection unit test

Create `tests/test_geometry.py`:

```python
import numpy as np
import torch
from src.geometry import build_mvdet_proj_mat, warp_perspective_torch
from src.calibration import CalibrationLoader
from src.config import Cfg

def test_foot_point_projects_to_correct_bev_cell():
    """
    For a known annotated pedestrian in frame 0000:
    - Load camera 0 calibration (K, R, t)
    - Build projection matrix
    - The pedestrian's world-grid position (ix, iy) should project from
      the image back to within ±1 cell of (ix, iy) in BEV space
    """
    cfg = Cfg()
    loader = CalibrationLoader(cfg.wildtrack_root)
    K, R, t = loader.load(cam_idx=0)
    proj_mat = build_mvdet_proj_mat(K, R, t, cfg)

    # Known annotation from frame 0000: positionID=22 → (ix=22%cfg.NB_WIDTH, iy=22//cfg.NB_WIDTH)
    # Verify by loading the JSON and computing manually
    known_ix, known_iy = 22 % cfg.NB_WIDTH, 22 // cfg.NB_WIDTH

    # The BEV cell (known_ix, known_iy) should warp to non-zero feature at that position
    # Create a 1-channel "marker" image feature at the foot point's image projection
    # Warp to BEV and check the peak is at (known_iy, known_ix)
    ...  # full implementation in PR

def test_bev_warp_is_invertible():
    """Warping with H then H^{-1} should recover original (up to bilinear error)."""
    ...
```

### Part B — Add a visual debugging script

Create `scripts/visualize_projection.py`:

```python
"""
Usage: python scripts/visualize_projection.py --data_root /path/to/wildtrack --frame 0 --cam 0

Saves two images:
  debug_cam0_frame0_image.png  — camera image with projected foot points overlaid
  debug_cam0_frame0_bev.png    — BEV heatmap with GT foot points overlaid

If the red dots (projected) align with green dots (GT), geometry is correct.
"""
```

This script must be run once per camera and the results committed to `docs/geometry_validation/`.

### Part C — Add an assertion in `warp_perspective_torch()`

```python
def warp_perspective_torch(feat, H, output_size):
    # Existing code
    ...
    # Add: check that at least 10% of BEV cells receive non-zero features
    with torch.no_grad():
        nonzero_ratio = (warped.abs().sum(dim=1) > 0).float().mean()
    assert nonzero_ratio > 0.1, (
        f"Homography produces near-empty BEV map ({nonzero_ratio:.2%} non-zero). "
        "Check projection matrix orientation."
    )
    return warped
```

---

## Acceptance Criteria

- [ ] `tests/test_geometry.py` exists with at least 2 passing tests
- [ ] `scripts/visualize_projection.py` exists and can be run independently
- [ ] Visualization images for all 7 cameras committed to `docs/geometry_validation/`
- [ ] Images confirm that projected foot points align with GT annotations (< 2 BEV cells error)
- [ ] `warp_perspective_torch()` includes the non-zero ratio assertion
- [ ] CI (`.github/workflows/python-smoke.yml`) runs `pytest tests/test_geometry.py`

---

## References

- `src/geometry.py`: `build_mvdet_proj_mat()`, `warp_perspective_torch()`
- `src/calibration.py`: `CalibrationLoader`, `decide_unit_scale()`
- Wildtrack annotation format: positionID → (ix, iy) per `src/dataset.py`
- PyTorch `grid_sample` coordinate convention: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
