"""Tests for temporal.trajectory_predictor — constant-velocity baseline."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from temporal.trajectory_predictor import (
    predict_constant_velocity,
    predict_trajectories_constant_velocity,
    TrajectoryPrediction,
)
from temporal.annotation_reader import Detection, Trajectory


class TestPredictConstantVelocity:
    def test_stationary(self):
        pos = np.array([1.0, 2.0])
        vel = np.array([0.0, 0.0])
        pred = predict_constant_velocity(pos, vel, n_future=4, dt=0.5)
        assert pred.shape == (4, 2)
        for i in range(4):
            np.testing.assert_allclose(pred[i], [1.0, 2.0])

    def test_constant_motion(self):
        pos = np.array([0.0, 0.0])
        vel = np.array([2.0, 1.0])  # 2 m/s east, 1 m/s north
        pred = predict_constant_velocity(pos, vel, n_future=4, dt=0.5)
        assert pred.shape == (4, 2)
        np.testing.assert_allclose(pred[0], [1.0, 0.5])
        np.testing.assert_allclose(pred[1], [2.0, 1.0])
        np.testing.assert_allclose(pred[2], [3.0, 1.5])
        np.testing.assert_allclose(pred[3], [4.0, 2.0])

    def test_single_step(self):
        pos = np.array([5.0, 5.0])
        vel = np.array([1.0, -1.0])
        pred = predict_constant_velocity(pos, vel, n_future=1, dt=0.5)
        assert pred.shape == (1, 2)
        np.testing.assert_allclose(pred[0], [5.5, 4.5])

    def test_custom_dt(self):
        pos = np.array([0.0, 0.0])
        vel = np.array([1.0, 0.0])
        pred = predict_constant_velocity(pos, vel, n_future=2, dt=1.0)
        np.testing.assert_allclose(pred[0], [1.0, 0.0])
        np.testing.assert_allclose(pred[1], [2.0, 0.0])


def _make_trajectory(pid, frame_positions):
    """Helper: create Trajectory from [(frame_idx, x, y), ...]."""
    traj = Trajectory(person_id=pid)
    for fi, x, y in frame_positions:
        traj.detections.append(Detection(
            frame_index=fi, frame_stem=f"{fi:08d}",
            person_id=pid, position_id=0,
            world_x_m=x, world_y_m=y,
        ))
    return traj


class TestPredictTrajectoriesConstantVelocity:
    def test_moving_person(self):
        traj = _make_trajectory(1, [
            (0, 0.0, 0.0),
            (1, 0.5, 0.0),
            (2, 1.0, 0.0),
            (3, 1.5, 0.0),
        ])
        trajectories = {1: traj}
        preds = predict_trajectories_constant_velocity(
            trajectories, observe_until_frame=3, n_future=2, dt=0.5
        )
        assert len(preds) == 1
        assert preds[0].person_id == 1
        assert preds[0].predicted_positions.shape == (2, 2)

    def test_skips_short_trajectory(self):
        traj = _make_trajectory(1, [(5, 1.0, 2.0)])
        trajectories = {1: traj}
        preds = predict_trajectories_constant_velocity(
            trajectories, observe_until_frame=5, n_future=2
        )
        assert len(preds) == 0

    def test_skips_stale_trajectory(self):
        traj = _make_trajectory(1, [
            (0, 0.0, 0.0),
            (1, 0.5, 0.0),
        ])
        preds = predict_trajectories_constant_velocity(
            {1: traj}, observe_until_frame=5, n_future=2
        )
        assert len(preds) == 0

    def test_multiple_persons(self):
        t1 = _make_trajectory(1, [(0, 0.0, 0.0), (1, 1.0, 0.0)])
        t2 = _make_trajectory(2, [(0, 5.0, 5.0), (1, 5.0, 6.0)])
        preds = predict_trajectories_constant_velocity(
            {1: t1, 2: t2}, observe_until_frame=1, n_future=3
        )
        assert len(preds) == 2
        pids = {p.person_id for p in preds}
        assert pids == {1, 2}

    def test_prediction_direction(self):
        traj = _make_trajectory(1, [
            (0, 0.0, 0.0),
            (1, 0.0, 1.0),
            (2, 0.0, 2.0),
        ])
        preds = predict_trajectories_constant_velocity(
            {1: traj}, observe_until_frame=2, n_future=2, dt=0.5
        )
        assert len(preds) == 1
        assert preds[0].predicted_positions[0, 1] > 2.0
