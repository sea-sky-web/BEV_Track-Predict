from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry import (
    compute_bev_geometry_metadata,
    compute_valid_ratio_from_homography,
    make_worldgrid2worldcoord_mat,
    warp_perspective_torch,
)


def test_worldgrid_transform_uses_cell_centers():
    mat = make_worldgrid2worldcoord_mat(origin_x=-3.0, origin_y=-9.0, step=0.1)
    world = mat @ np.array([0.0, 0.0, 1.0])

    assert np.allclose(world[:2], [-2.95, -8.95])
    recovered = np.linalg.inv(mat) @ world
    assert np.allclose(recovered[:2], [0.0, 0.0])


def test_valid_ratio_identity_is_full_coverage():
    ratio = compute_valid_ratio_from_homography(np.eye(3), src_hw=(4, 4), dst_hw=(4, 4))
    assert ratio == 1.0


def test_warp_perspective_identity_matches_input():
    src = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    out = warp_perspective_torch(src, torch.eye(3), dsize=(4, 4), min_valid_ratio=1.0)
    assert torch.allclose(out, src)


def test_warp_perspective_min_valid_ratio_fails_for_bad_shift():
    src = torch.zeros(1, 1, 4, 4)
    shift = torch.tensor([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="valid ratio"):
        warp_perspective_torch(src, shift, dsize=(4, 4), min_valid_ratio=0.5)


def test_bev_geometry_metadata_identity():
    V = 3
    proj_mats = torch.eye(3).unsqueeze(0).repeat(V, 1, 1)
    meta = compute_bev_geometry_metadata(proj_mats, src_hw=(8, 8), dst_hw=(8, 8))
    assert meta.shape == (V, 3, 8, 8)
    assert (meta[:, 0] == 1.0).all(), "identity proj should be fully valid"
    assert (meta[:, 1] >= 0.0).all() and (meta[:, 1] <= 1.0).all()
    assert torch.allclose(meta[:, 2], torch.ones(V, 8, 8)), "coverage should be 1.0"


def test_bev_geometry_metadata_shifted_view():
    shift = torch.tensor([[1.0, 0.0, 100.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    identity = torch.eye(3)
    proj_mats = torch.stack([identity, shift])  # V=2
    meta = compute_bev_geometry_metadata(proj_mats, src_hw=(8, 8), dst_hw=(8, 8))
    assert meta[0, 0].sum() > 0, "identity view should have valid cells"
    assert meta[1, 0].sum() == 0, "shifted view should have no valid cells"
    assert (meta[:, 2] <= 0.6).all(), "coverage should be at most 0.5 (1/2 views valid)"
