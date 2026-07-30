"""Tests for MLP trajectory predictor."""

import numpy as np
import pytest

from temporal.mlp_predictor import (
    extract_trajectory_windows,
    train_mlp_predictor,
    predict_mlp,
    MLPTrajectoryConfig,
)
from temporal.annotation_reader import Detection, Trajectory


def _make_trajectory(pid, start_frame, n_frames, vx=0.5, vy=0.1):
    """Create a trajectory with constant velocity."""
    dets = []
    for i in range(n_frames):
        dets.append(Detection(
            frame_index=start_frame + i,
            frame_stem=f"{(start_frame + i) * 400:08d}",
            person_id=pid,
            position_id=0,
            world_x_m=1.0 + vx * i * 0.5,
            world_y_m=2.0 + vy * i * 0.5,
        ))
    return Trajectory(person_id=pid, detections=dets)


class TestExtractWindows:

    def test_basic_extraction(self):
        traj = _make_trajectory(1, start_frame=320, n_frames=12)
        trajectories = {1: traj}
        history, future = extract_trajectory_windows(trajectories, "val", n_history=4, n_future=4)
        assert history.shape[1] == 4
        assert future.shape[1] == 4
        assert history.shape[0] > 0

    def test_requires_consecutive_frames(self):
        dets = [
            Detection(320, "0", 1, 0, 0.0, 0.0),
            Detection(321, "1", 1, 0, 0.1, 0.0),
            Detection(323, "3", 1, 0, 0.3, 0.0),
            Detection(324, "4", 1, 0, 0.4, 0.0),
            Detection(325, "5", 1, 0, 0.5, 0.0),
            Detection(326, "6", 1, 0, 0.6, 0.0),
            Detection(327, "7", 1, 0, 0.7, 0.0),
            Detection(328, "8", 1, 0, 0.8, 0.0),
        ]
        traj = Trajectory(person_id=1, detections=dets)
        history, future = extract_trajectory_windows({1: traj}, "val", 4, 4)
        for i in range(history.shape[0]):
            assert history[i, 1, 0] - history[i, 0, 0] == pytest.approx(0.1, abs=1e-6)

    def test_empty_if_not_in_split(self):
        traj = _make_trajectory(1, start_frame=0, n_frames=10)
        trajectories = {1: traj}
        history, future = extract_trajectory_windows(trajectories, "val", 4, 4)
        assert history.shape[0] == 0


class TestTrainAndPredict:

    def test_train_reduces_loss(self):
        rng = np.random.default_rng(42)
        n = 200
        history = rng.standard_normal((n, 4, 2)) * 0.5
        future = history[:, -1:, :] + rng.standard_normal((n, 4, 2)) * 0.1

        config = MLPTrajectoryConfig(
            n_history=4, n_future=4, hidden_dim=32,
            max_epochs=50, patience=50, seed=0,
        )
        result = train_mlp_predictor(
            history[:150], future[:150],
            history[150:], future[150:],
            config,
        )
        assert result["train_losses"][-1] < result["train_losses"][0]
        assert result["best_val_ade"] > 0

    def test_predict_shape(self):
        rng = np.random.default_rng(0)
        weights = {
            "W1": rng.standard_normal((8, 64)),
            "b1": np.zeros(64),
            "W2": rng.standard_normal((64, 8)),
            "b2": np.zeros(8),
            "x_mean": np.zeros(8),
            "x_std": np.ones(8),
        }
        history = rng.standard_normal((10, 4, 2))
        pred = predict_mlp(history, weights, n_future=4)
        assert pred.shape == (10, 4, 2)

    def test_constant_velocity_data_learnable(self):
        """MLP should learn constant-velocity pattern perfectly."""
        rng = np.random.default_rng(7)
        n = 500
        v = rng.standard_normal((n, 1, 2)) * 0.3
        start = rng.standard_normal((n, 1, 2)) * 2.0
        t_hist = np.arange(4).reshape(1, 4, 1) * 0.5
        t_fut = np.arange(4, 8).reshape(1, 4, 1) * 0.5
        history = start + v * t_hist
        future = start + v * t_fut

        config = MLPTrajectoryConfig(
            n_history=4, n_future=4, hidden_dim=64,
            learning_rate=1e-3, max_epochs=500, patience=100, seed=0,
        )
        result = train_mlp_predictor(
            history[:400], future[:400],
            history[400:], future[400:],
            config,
        )
        assert result["best_val_ade"] < 0.15

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(0)
        n = 50
        h = rng.standard_normal((n, 4, 2))
        f = np.repeat(h[:, -1:, :], 4, axis=1) + rng.standard_normal((n, 4, 2)) * 0.1

        config = MLPTrajectoryConfig(max_epochs=10, seed=42)
        r1 = train_mlp_predictor(h[:40], f[:40], h[40:], f[40:], config)
        r2 = train_mlp_predictor(h[:40], f[:40], h[40:], f[40:], config)
        assert r1["best_val_ade"] == r2["best_val_ade"]
