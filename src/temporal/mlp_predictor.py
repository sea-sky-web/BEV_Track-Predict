"""MLP trajectory predictor — minimal learning baseline.

A 2-layer MLP that predicts future positions from observed positions.
Input: relative displacements over n_history frames (flattened).
Output: relative displacements for n_future frames (flattened).

This module is intentionally minimal: no social pooling, no attention,
no scene context. If this cannot beat constant-velocity, then WildTrack
motion is too simple for learning-based prediction to add value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from temporal.annotation_reader import Trajectory
from temporal.time_utils import DT, get_split_range, FRAME_RATE_HZ


@dataclass
class MLPTrajectoryConfig:
    n_history: int = 4
    n_future: int = 4
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    seed: int = 0


def extract_trajectory_windows(
    trajectories: dict[int, Trajectory],
    split: str,
    n_history: int = 4,
    n_future: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract all valid (history, future) trajectory windows from a split.

    Returns:
        history: (N, n_history, 2) absolute positions.
        future: (N, n_future, 2) absolute positions.
    """
    split_start, split_end = get_split_range(split)
    total_window = n_history + n_future

    history_list = []
    future_list = []

    for pid, traj in trajectories.items():
        frames = traj.frame_indices
        positions = traj.positions

        for wi in range(len(frames) - total_window + 1):
            window_frames = frames[wi:wi + total_window]
            if window_frames[0] < split_start or window_frames[-1] >= split_end:
                continue
            if window_frames != list(range(window_frames[0], window_frames[0] + total_window)):
                continue

            window_pos = positions[wi:wi + total_window]
            history_list.append(window_pos[:n_history])
            future_list.append(window_pos[n_history:])

    if not history_list:
        return np.empty((0, n_history, 2)), np.empty((0, n_future, 2))

    return np.array(history_list), np.array(future_list)


def positions_to_displacements(positions: np.ndarray) -> np.ndarray:
    """Convert absolute positions (N, T, 2) to displacements relative to last observed position.

    Returns (N, T, 2) where each position is offset by the last history position.
    """
    return positions - positions[:, -1:, :]


def train_mlp_predictor(
    train_history: np.ndarray,
    train_future: np.ndarray,
    val_history: np.ndarray,
    val_future: np.ndarray,
    config: MLPTrajectoryConfig,
) -> dict:
    """Train the MLP predictor using numpy-only gradient descent.

    Uses relative displacements as input/output for translation invariance.

    Returns dict with: weights, best_val_ade, train_losses, val_ades.
    """
    rng = np.random.default_rng(config.seed)

    n_train = train_history.shape[0]
    n_val = val_history.shape[0]

    last_obs_train = train_history[:, -1:, :]
    last_obs_val = val_history[:, -1:, :]

    x_train = (train_history - last_obs_train).reshape(n_train, -1)
    y_train = (train_future - last_obs_train).reshape(n_train, -1)

    x_val = (val_history - last_obs_val).reshape(n_val, -1)
    y_val = (val_future - last_obs_val).reshape(n_val, -1)

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0) + 1e-8
    x_train_norm = (x_train - x_mean) / x_std
    x_val_norm = (x_val - x_mean) / x_std

    input_dim = config.n_history * 2
    output_dim = config.n_future * 2
    h_dim = config.hidden_dim

    scale1 = np.sqrt(2.0 / input_dim)
    W1 = rng.standard_normal((input_dim, h_dim)) * scale1
    b1 = np.zeros(h_dim)
    scale2 = np.sqrt(2.0 / h_dim)
    W2 = rng.standard_normal((h_dim, output_dim)) * scale2
    b2 = np.zeros(output_dim)

    best_val_ade = float("inf")
    best_weights = None
    patience_counter = 0
    train_losses = []
    val_ades = []

    lr = config.learning_rate
    wd = config.weight_decay

    for epoch in range(config.max_epochs):
        h = x_train_norm @ W1 + b1
        h_relu = np.maximum(h, 0)
        y_pred = h_relu @ W2 + b2

        residual = y_pred - y_train
        loss = 0.5 * np.mean(residual ** 2)
        train_losses.append(float(loss))

        dout = residual / n_train
        dW2 = h_relu.T @ dout + wd * W2
        db2 = dout.sum(axis=0)
        dh_relu = dout @ W2.T
        dh = dh_relu * (h > 0).astype(np.float64)
        dW1 = x_train_norm.T @ dh + wd * W1
        db1 = dh.sum(axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        h_v = x_val_norm @ W1 + b1
        h_v_relu = np.maximum(h_v, 0)
        y_val_pred = h_v_relu @ W2 + b2

        val_errors = np.linalg.norm(
            y_val_pred.reshape(n_val, config.n_future, 2) - y_val.reshape(n_val, config.n_future, 2),
            axis=2,
        )
        val_ade = float(val_errors.mean())
        val_ades.append(val_ade)

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            best_weights = {
                "W1": W1.copy(), "b1": b1.copy(),
                "W2": W2.copy(), "b2": b2.copy(),
                "x_mean": x_mean, "x_std": x_std,
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    return {
        "weights": best_weights,
        "best_val_ade": best_val_ade,
        "train_losses": train_losses,
        "val_ades": val_ades,
        "epochs_trained": len(train_losses),
    }


def predict_mlp(
    history: np.ndarray,
    weights: dict,
    n_future: int = 4,
) -> np.ndarray:
    """Predict future positions given observed history.

    Args:
        history: (N, n_history, 2) absolute positions.
        weights: dict from train_mlp_predictor.

    Returns:
        (N, n_future, 2) predicted absolute positions.
    """
    n = history.shape[0]
    last_obs = history[:, -1:, :]
    x = (history - last_obs).reshape(n, -1)
    x_norm = (x - weights["x_mean"]) / weights["x_std"]

    h = x_norm @ weights["W1"] + weights["b1"]
    h_relu = np.maximum(h, 0)
    y_pred = h_relu @ weights["W2"] + weights["b2"]

    pred_rel = y_pred.reshape(n, n_future, 2)
    return pred_rel + last_obs


def evaluate_mlp_predictor(
    trajectories: dict[int, Trajectory],
    weights: dict,
    split: str = "val",
    n_history: int = 4,
    n_future: int = 4,
) -> dict:
    """Evaluate trained MLP predictor on a split.

    Returns dict with ade_mean, fde_mean, n_trajectories.
    """
    history, future_gt = extract_trajectory_windows(trajectories, split, n_history, n_future)

    if history.shape[0] == 0:
        return {"ade_mean": 0.0, "fde_mean": 0.0, "n_trajectories": 0}

    pred = predict_mlp(history, weights, n_future)
    errors = np.linalg.norm(pred - future_gt, axis=2)

    return {
        "ade_mean": float(errors.mean()),
        "ade_std": float(errors.mean(axis=1).std()),
        "fde_mean": float(errors[:, -1].mean()),
        "fde_std": float(errors[:, -1].std()),
        "n_trajectories": int(history.shape[0]),
    }
