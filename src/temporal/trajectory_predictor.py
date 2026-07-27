"""Constant-velocity trajectory prediction baseline.

Uses Kalman tracker velocity estimates to linearly extrapolate
individual trajectories into the future. Serves as the non-learning
baseline for ADE/FDE evaluation before training learned predictors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temporal.annotation_reader import Trajectory, compute_velocities
from temporal.time_utils import DT, get_split_range, FRAME_RATE_HZ


@dataclass
class TrajectoryPrediction:
    person_id: int
    last_observed_frame: int
    last_position: np.ndarray  # (2,)
    last_velocity: np.ndarray  # (2,)
    predicted_positions: np.ndarray  # (T_future, 2)


def predict_constant_velocity(
    position: np.ndarray,
    velocity: np.ndarray,
    n_future: int,
    dt: float = DT,
) -> np.ndarray:
    """Linearly extrapolate a single trajectory.

    Args:
        position: (2,) last observed world position in meters.
        velocity: (2,) estimated velocity in m/s.
        n_future: number of future steps to predict.
        dt: time step in seconds.

    Returns:
        (n_future, 2) predicted positions.
    """
    steps = np.arange(1, n_future + 1, dtype=np.float64)
    return position[None, :] + velocity[None, :] * (steps[:, None] * dt)


def predict_trajectories_constant_velocity(
    trajectories: dict[int, Trajectory],
    observe_until_frame: int,
    n_future: int = 4,
    dt: float = DT,
) -> list[TrajectoryPrediction]:
    """Predict future positions for all trajectories visible at observe_until_frame.

    Only trajectories with at least 2 observed frames up to observe_until_frame
    are predicted (need velocity estimate).

    Args:
        trajectories: {person_id: Trajectory} from build_trajectories().
        observe_until_frame: last frame to observe (inclusive).
        n_future: number of future steps.
        dt: time step in seconds.

    Returns:
        List of TrajectoryPrediction for each active trajectory.
    """
    predictions = []

    for pid, traj in trajectories.items():
        observed = [d for d in traj.detections if d.frame_index <= observe_until_frame]
        if len(observed) < 2:
            continue

        last_det = observed[-1]
        if last_det.frame_index < observe_until_frame - 1:
            continue

        vel = compute_velocities(traj, dt=dt)
        traj_frames = traj.frame_indices
        last_idx_in_traj = traj_frames.index(last_det.frame_index)
        last_vel = vel[last_idx_in_traj]

        last_pos = np.array([last_det.world_x_m, last_det.world_y_m])
        pred_pos = predict_constant_velocity(last_pos, last_vel, n_future, dt)

        predictions.append(TrajectoryPrediction(
            person_id=pid,
            last_observed_frame=last_det.frame_index,
            last_position=last_pos,
            last_velocity=last_vel,
            predicted_positions=pred_pos,
        ))

    return predictions


def evaluate_trajectory_baseline(
    trajectories: dict[int, Trajectory],
    split: str = "val",
    n_history: int = 4,
    n_future: int = 4,
    dt: float = DT,
) -> dict:
    """Run constant-velocity baseline over all windows in a split and compute ADE/FDE.

    Args:
        trajectories: all trajectories (must cover the split range).
        split: "val" or "test".
        n_history: minimum observation frames.
        n_future: prediction horizon.
        dt: time step.

    Returns:
        dict with ade_mean, ade_std, fde_mean, fde_std, n_trajectories, n_windows, horizon_s.
    """
    from temporal.field_metrics import compute_trajectory_ade_fde

    split_start, split_end = get_split_range(split)
    n_split = split_end - split_start
    total_window = n_history + n_future

    all_ade = []
    all_fde = []
    total_traj = 0

    for wi in range(n_split - total_window + 1):
        observe_until = split_start + n_history + wi - 1
        future_start = observe_until + 1

        preds = predict_trajectories_constant_velocity(
            trajectories, observe_until, n_future, dt
        )

        if not preds:
            continue

        pred_list = []
        gt_list = []
        for p in preds:
            gt_future = []
            valid = True
            for step in range(n_future):
                target_frame = future_start + step
                found = False
                for d in trajectories[p.person_id].detections:
                    if d.frame_index == target_frame:
                        gt_future.append([d.world_x_m, d.world_y_m])
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if valid and len(gt_future) == n_future:
                pred_list.append(p.predicted_positions)
                gt_list.append(np.array(gt_future))

        if not pred_list:
            continue

        pred_arr = np.array(pred_list)
        gt_arr = np.array(gt_list)
        horizon_s = n_future * dt

        metrics = compute_trajectory_ade_fde(pred_arr, gt_arr, horizon_s)
        all_ade.append(metrics.ade)
        all_fde.append(metrics.fde)
        total_traj += metrics.n_trajectories

    if not all_ade:
        return {
            "ade_mean": 0.0, "ade_std": 0.0,
            "fde_mean": 0.0, "fde_std": 0.0,
            "n_trajectories": 0, "n_windows": 0,
            "horizon_s": n_future * dt,
        }

    return {
        "ade_mean": float(np.mean(all_ade)),
        "ade_std": float(np.std(all_ade)),
        "fde_mean": float(np.mean(all_fde)),
        "fde_std": float(np.std(all_fde)),
        "n_trajectories": total_traj,
        "n_windows": len(all_ade),
        "horizon_s": n_future * dt,
    }
