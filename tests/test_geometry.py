import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.device = object
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_nn_functional_stub = types.ModuleType("torch.nn.functional")
    torch_nn_stub.functional = torch_nn_functional_stub
    torch_stub.nn = torch_nn_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_nn_stub
    sys.modules["torch.nn.functional"] = torch_nn_functional_stub

calibration_stub = types.ModuleType("calibration")
calibration_stub.scale_intrinsics = lambda k, sx, sy: k
sys.modules.setdefault("calibration", calibration_stub)

from geometry import build_mvdet_proj_mat, make_worldgrid2worldcoord_mat


def _apply_homography(mat, xy):
    point = np.array([xy[0], xy[1], 1.0], dtype=np.float64)
    projected = mat @ point
    return projected[:2] / projected[2]


def test_build_mvdet_proj_mat_preserves_bev_xy_order():
    k_feat = np.eye(3, dtype=np.float64)
    r = np.eye(3, dtype=np.float64)
    t = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
    worldgrid2worldcoord = make_worldgrid2worldcoord_mat(
        origin_x=0.0,
        origin_y=0.0,
        step=1.0,
    )

    proj = build_mvdet_proj_mat(k_feat, r, t, worldgrid2worldcoord)

    image_xy = np.array([13.5, 27.5], dtype=np.float64)
    bev_xy = _apply_homography(proj, image_xy)

    np.testing.assert_allclose(bev_xy, [13.0, 27.0], atol=1e-9)
    assert not np.allclose(bev_xy, [27.0, 13.0], atol=1e-9)


def test_build_mvdet_proj_mat_inverse_maps_bev_xy_to_image_xy():
    k_feat = np.eye(3, dtype=np.float64)
    r = np.eye(3, dtype=np.float64)
    t = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
    worldgrid2worldcoord = make_worldgrid2worldcoord_mat(
        origin_x=0.0,
        origin_y=0.0,
        step=1.0,
    )

    proj = build_mvdet_proj_mat(k_feat, r, t, worldgrid2worldcoord)

    image_xy = _apply_homography(np.linalg.inv(proj), np.array([13.0, 27.0]))

    np.testing.assert_allclose(image_xy, [13.5, 27.5], atol=1e-9)
