"""Tests for field building, baselines, and field metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest


# ── Field Builder ──


def test_occupancy_empty():
    from temporal.field_builder import build_occupancy_field

    pos = np.empty((0, 2), dtype=np.float64)
    occ = build_occupancy_field(pos, sigma_m=0.2)
    assert occ.shape == (120, 360)
    assert occ.sum() == 0.0


def test_occupancy_single_person():
    from temporal.field_builder import build_occupancy_field
    from temporal.coordinates import reduced_to_world

    wx, wy = reduced_to_world(np.array([60.0]), np.array([180.0]))
    pos = np.array([[wx.item(), wy.item()]])
    occ = build_occupancy_field(pos, sigma_m=0.2)

    assert occ.shape == (120, 360)
    assert occ.max() > 0.0
    peak_r, peak_c = np.unravel_index(occ.argmax(), occ.shape)
    assert abs(peak_r - 60) <= 1
    assert abs(peak_c - 180) <= 1


def test_occupancy_range():
    from temporal.field_builder import build_occupancy_field
    from temporal.coordinates import reduced_to_world

    wx, wy = reduced_to_world(np.array([30.0]), np.array([90.0]))
    pos = np.array([[wx.item(), wy.item()]])
    occ = build_occupancy_field(pos, sigma_m=0.2)

    assert occ.min() >= 0.0
    assert occ.max() <= 1.0


def test_velocity_field_static():
    from temporal.field_builder import build_velocity_field
    from temporal.coordinates import reduced_to_world

    wx, wy = reduced_to_world(np.array([60.0]), np.array([180.0]))
    pos = np.array([[wx.item(), wy.item()]])
    vel = np.array([[0.0, 0.0]])
    vx, vy = build_velocity_field(pos, vel, sigma_m=0.2)

    assert vx.shape == (120, 360)
    np.testing.assert_allclose(vx.max(), 0.0, atol=1e-6)
    np.testing.assert_allclose(vy.max(), 0.0, atol=1e-6)


def test_velocity_field_moving():
    from temporal.field_builder import build_velocity_field
    from temporal.coordinates import reduced_to_world

    wx, wy = reduced_to_world(np.array([60.0]), np.array([180.0]))
    pos = np.array([[wx.item(), wy.item()]])
    vel = np.array([[1.0, 0.5]])  # m/s
    vx, vy = build_velocity_field(pos, vel, sigma_m=0.2)

    peak_idx = np.unravel_index(np.abs(vx).argmax(), vx.shape)
    assert abs(vx[peak_idx] - 1.0) < 0.5
    assert vy[peak_idx] > 0.0


def test_build_all_fields_shape():
    from temporal.field_builder import build_all_fields

    pos = np.array([[1.0, 2.0], [3.0, 4.0]])
    vel = np.array([[0.5, 0.0], [0.0, 0.5]])
    fields = build_all_fields(pos, vel, sigma_m=0.2)
    assert fields.shape == (5, 120, 360)


def test_valid_mask_all_ones():
    from temporal.field_builder import build_valid_mask

    mask = build_valid_mask()
    assert mask.shape == (120, 360)
    np.testing.assert_array_equal(mask, 1.0)


# ── Baselines ──


def test_persistence_identity():
    from temporal.baselines import predict_persistence

    occ = np.random.rand(120, 360).astype(np.float32)
    preds = predict_persistence(occ, n_future=4)
    assert len(preds) == 4
    for p in preds:
        np.testing.assert_array_equal(p, occ)


def test_constant_velocity_straight_line():
    """A single person moving in x at constant velocity should be accurately predicted."""
    from temporal.baselines import predict_constant_velocity
    from temporal.coordinates import reduced_to_world

    wx, wy = reduced_to_world(np.array([60.0]), np.array([180.0]))
    pos = np.array([[wx.item(), wy.item()]])
    vx_ms = 0.5  # m/s
    vel = np.array([[vx_ms, 0.0]])

    preds = predict_constant_velocity(pos, vel, n_future=2, dt=0.5, sigma_m=0.2)
    assert len(preds) == 2

    for step, p in enumerate(preds, 1):
        expected_row = 60.0 + (vx_ms * 0.5 * step) / 0.1
        peak_r, _ = np.unravel_index(p.argmax(), p.shape)
        assert abs(peak_r - expected_row) <= 2


def test_field_advection_conservation():
    """Total occupancy mass should be approximately conserved under advection."""
    from temporal.baselines import predict_field_advection

    h, w = 120, 360
    occ = np.zeros((h, w), dtype=np.float32)
    occ[50:70, 170:190] = 0.8

    vx = np.full((h, w), 0.1, dtype=np.float32)
    vy = np.zeros((h, w), dtype=np.float32)

    preds = predict_field_advection(occ, vx, vy, n_future=2, dt=0.5)
    assert len(preds) == 2

    mass_orig = occ.sum()
    for p in preds:
        mass_pred = p.sum()
        assert abs(mass_pred - mass_orig) / max(mass_orig, 1e-6) < 0.05


def test_oracle_matches_gt():
    from temporal.baselines import predict_oracle
    from temporal.field_builder import build_occupancy_field

    pos_future = [
        np.array([[1.0, 2.0]]),
        np.array([[1.5, 2.5]]),
    ]
    preds = predict_oracle(pos_future, sigma_m=0.2)
    assert len(preds) == 2

    direct = build_occupancy_field(pos_future[0], sigma_m=0.2)
    np.testing.assert_array_equal(preds[0], direct)


# ── Field Metrics ──


def test_occupancy_perfect_prediction():
    from temporal.field_metrics import compute_occupancy_at_threshold

    gt = np.zeros((120, 360), dtype=np.float32)
    gt[50:60, 170:180] = 1.0
    pred = gt.copy()

    m = compute_occupancy_at_threshold(pred, gt, threshold=0.5)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.iou == 1.0


def test_occupancy_empty_gt():
    from temporal.field_metrics import compute_occupancy_auprc

    gt = np.zeros((120, 360), dtype=np.float32)
    pred = np.random.rand(120, 360).astype(np.float32)
    auprc = compute_occupancy_auprc(pred, gt)
    assert auprc == 0.0


def test_velocity_epe_zero():
    from temporal.field_metrics import compute_velocity_epe

    vx = np.random.rand(120, 360).astype(np.float32)
    vy = np.random.rand(120, 360).astype(np.float32)
    m = compute_velocity_epe(vx, vy, vx, vy)
    assert m.epe == 0.0


def test_velocity_epe_with_mask():
    from temporal.field_metrics import compute_velocity_epe

    pred_vx = np.ones((120, 360), dtype=np.float32)
    pred_vy = np.zeros((120, 360), dtype=np.float32)
    gt_vx = np.zeros((120, 360), dtype=np.float32)
    gt_vy = np.zeros((120, 360), dtype=np.float32)

    mask = np.zeros((120, 360), dtype=np.float32)
    mask[50:60, 170:180] = 1.0

    m = compute_velocity_epe(pred_vx, pred_vy, gt_vx, gt_vy, occupancy_mask=mask)
    assert m.epe_occupied == 1.0


def test_trajectory_ade_fde():
    from temporal.field_metrics import compute_trajectory_ade_fde

    pred = np.array([[[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]])
    gt = np.array([[[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]])

    m = compute_trajectory_ade_fde(pred, gt, horizon_s=1.5)
    assert m.ade == 0.0
    assert m.fde == 0.0
    assert m.n_trajectories == 1


def test_trajectory_ade_fde_nonzero():
    from temporal.field_metrics import compute_trajectory_ade_fde

    pred = np.array([[[1.0, 2.0], [2.0, 3.0]]])
    gt = np.array([[[1.0, 2.0], [1.0, 2.0]]])

    m = compute_trajectory_ade_fde(pred, gt, horizon_s=1.0)
    assert m.ade > 0.0
    assert m.fde > 0.0


def test_trajectory_empty():
    from temporal.field_metrics import compute_trajectory_ade_fde

    pred = np.empty((0, 4, 2))
    gt = np.empty((0, 4, 2))
    m = compute_trajectory_ade_fde(pred, gt, horizon_s=2.0)
    assert m.n_trajectories == 0
    assert m.ade == 0.0
