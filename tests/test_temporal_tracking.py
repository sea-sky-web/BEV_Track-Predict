"""Tests for multi-object trackers and tracking metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest


# ── Nearest Neighbor Tracker ──


def test_nn_single_target():
    from temporal.tracker_nn import NearestNeighborTracker

    tracker = NearestNeighborTracker(dist_gate=1.0, max_age=2, min_hits=1)
    dets = np.array([[1.0, 2.0]])
    out = tracker.update(dets, frame_index=0)
    assert len(out.active_tracks) == 1
    assert abs(out.active_tracks[0].world_x_m - 1.0) < 1e-6


def test_nn_two_targets_no_swap():
    from temporal.tracker_nn import NearestNeighborTracker

    tracker = NearestNeighborTracker(dist_gate=2.0, max_age=2, min_hits=1)

    dets0 = np.array([[0.0, 0.0], [5.0, 5.0]])
    out0 = tracker.update(dets0, frame_index=0)
    ids0 = {t.track_id: (t.world_x_m, t.world_y_m) for t in out0.active_tracks}

    dets1 = np.array([[0.1, 0.1], [5.1, 5.1]])
    out1 = tracker.update(dets1, frame_index=1)
    ids1 = {t.track_id: (t.world_x_m, t.world_y_m) for t in out1.active_tracks}

    for tid in ids0:
        assert tid in ids1


def test_nn_track_death():
    from temporal.tracker_nn import NearestNeighborTracker

    tracker = NearestNeighborTracker(dist_gate=1.0, max_age=1, min_hits=1)
    tracker.update(np.array([[1.0, 2.0]]), frame_index=0)
    out = tracker.update(np.empty((0, 2)), frame_index=1)
    assert len(out.active_tracks) == 1  # still alive (age <= max_age)
    out = tracker.update(np.empty((0, 2)), frame_index=2)
    assert len(out.active_tracks) == 0  # dead


def test_nn_min_hits():
    from temporal.tracker_nn import NearestNeighborTracker

    tracker = NearestNeighborTracker(dist_gate=1.0, max_age=2, min_hits=3)
    for i in range(2):
        out = tracker.update(np.array([[1.0, 2.0]]), frame_index=i)
    assert len(out.active_tracks) == 0  # not confirmed yet

    out = tracker.update(np.array([[1.0, 2.0]]), frame_index=2)
    assert len(out.active_tracks) == 1  # confirmed


# ── Kalman + Hungarian Tracker ──


def test_kalman_single_target():
    from temporal.tracker_kalman import KalmanHungarianTracker

    tracker = KalmanHungarianTracker(dist_gate=2.0, max_age=2, min_hits=1)
    dets = np.array([[1.0, 2.0]])
    out = tracker.update(dets, frame_index=0)
    assert len(out.active_tracks) == 1


def test_kalman_velocity_estimation():
    """Constant velocity target should have correct velocity after a few frames."""
    from temporal.tracker_kalman import KalmanHungarianTracker

    tracker = KalmanHungarianTracker(dist_gate=2.0, max_age=2, min_hits=1, dt=0.5)
    vx_true = 1.0  # m/s
    for i in range(10):
        x = 0.0 + vx_true * 0.5 * i
        out = tracker.update(np.array([[x, 0.0]]), frame_index=i)

    t = out.active_tracks[0]
    assert abs(t.vx - vx_true) < 0.3
    assert abs(t.vy) < 0.3


def test_kalman_two_crossing_targets():
    """Two targets crossing paths should maintain ID."""
    from temporal.tracker_kalman import KalmanHungarianTracker

    tracker = KalmanHungarianTracker(dist_gate=3.0, max_age=2, min_hits=1, dt=0.5)

    for i in range(10):
        x_a = float(i) * 0.5
        y_a = 0.0
        x_b = 5.0 - float(i) * 0.5
        y_b = 0.0
        out = tracker.update(np.array([[x_a, y_a], [x_b, y_b]]), frame_index=i)

    assert len(out.active_tracks) == 2


def test_kalman_miss_and_recover():
    """Target disappears for 1 frame, then reappears at predicted position."""
    from temporal.tracker_kalman import KalmanHungarianTracker

    tracker = KalmanHungarianTracker(dist_gate=2.0, max_age=2, min_hits=1, dt=0.5)

    for i in range(5):
        tracker.update(np.array([[float(i) * 0.5, 0.0]]), frame_index=i)

    out_miss = tracker.update(np.empty((0, 2)), frame_index=5)
    assert len(out_miss.active_tracks) == 1

    out_recover = tracker.update(np.array([[3.0, 0.0]]), frame_index=6)
    assert len(out_recover.active_tracks) == 1


def test_kalman_false_detection():
    """Spurious detection should not create long-lived track if min_hits > 1."""
    from temporal.tracker_kalman import KalmanHungarianTracker

    tracker = KalmanHungarianTracker(dist_gate=1.0, max_age=1, min_hits=2)
    tracker.update(np.array([[100.0, 100.0]]), frame_index=0)
    out = tracker.update(np.empty((0, 2)), frame_index=1)
    confirmed = [t for t in out.active_tracks if t.confirmed]
    assert len(confirmed) == 0


# ── Tracking Metrics ──


def test_perfect_tracking():
    from temporal.tracking_metrics import evaluate_tracking

    gt_frames = [
        {"positions": np.array([[1.0, 2.0], [3.0, 4.0]]), "ids": np.array([0, 1])},
        {"positions": np.array([[1.1, 2.1], [3.1, 4.1]]), "ids": np.array([0, 1])},
    ]
    pred_frames = [
        {"positions": np.array([[1.0, 2.0], [3.0, 4.0]]), "ids": np.array([10, 11])},
        {"positions": np.array([[1.1, 2.1], [3.1, 4.1]]), "ids": np.array([10, 11])},
    ]

    m = evaluate_tracking(gt_frames, pred_frames, dist_thr=0.5)
    assert m.tp == 4
    assert m.fp == 0
    assert m.fn == 0
    assert m.id_switches == 0
    assert m.mota == 1.0
    assert m.idf1 > 0.99


def test_all_missed():
    from temporal.tracking_metrics import evaluate_tracking

    gt_frames = [
        {"positions": np.array([[1.0, 2.0]]), "ids": np.array([0])},
    ]
    pred_frames = [
        {"positions": np.empty((0, 2)), "ids": np.array([])},
    ]

    m = evaluate_tracking(gt_frames, pred_frames, dist_thr=0.5)
    assert m.tp == 0
    assert m.fn == 1
    assert m.mota == 0.0


def test_id_switch_detected():
    from temporal.tracking_metrics import evaluate_tracking

    gt_frames = [
        {"positions": np.array([[1.0, 0.0], [5.0, 0.0]]), "ids": np.array([0, 1])},
        {"positions": np.array([[1.1, 0.0], [5.1, 0.0]]), "ids": np.array([0, 1])},
    ]
    # frame 0: pred 10 → gt 0, pred 11 → gt 1
    # frame 1: pred 11 → gt 0, pred 10 → gt 1 (swapped IDs)
    pred_frames = [
        {"positions": np.array([[1.0, 0.0], [5.0, 0.0]]), "ids": np.array([10, 11])},
        {"positions": np.array([[1.1, 0.0], [5.1, 0.0]]), "ids": np.array([11, 10])},
    ]

    m = evaluate_tracking(gt_frames, pred_frames, dist_thr=0.5)
    assert m.id_switches == 2
