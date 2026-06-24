from geometry import (
    compute_valid_ratio_from_homography,
    make_worldgrid2worldcoord_mat,
    warp_perspective_torch,
)
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_worldgrid_transform_uses_cell_centers():
    mat = make_worldgrid2worldcoord_mat(origin_x=-3.0, origin_y=-9.0, step=0.1)
    world = mat @ np.array([0.0, 0.0, 1.0])

    assert np.allclose(world[:2], [-2.95, -8.95])
    recovered = np.linalg.inv(mat) @ world
    assert np.allclose(recovered[:2], [0.0, 0.0])


def test_valid_ratio_identity_is_full_coverage():
    ratio = compute_valid_ratio_from_homography(
        np.eye(3), src_hw=(4, 4), dst_hw=(4, 4))
    assert ratio == 1.0


def test_warp_perspective_identity_matches_input():
    src = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    out = warp_perspective_torch(
        src, torch.eye(3), dsize=(
            4, 4), min_valid_ratio=1.0)
    assert torch.allclose(out, src, atol=1e-5)


def test_warp_perspective_min_valid_ratio_fails_for_bad_shift():
    src = torch.zeros(1, 1, 4, 4)
    shift = torch.tensor([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="valid ratio"):
        warp_perspective_torch(src, shift, dsize=(4, 4), min_valid_ratio=0.5)
