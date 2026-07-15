"""Field prediction evaluation metrics.

- Occupancy: AUPRC, IoU, Precision, Recall, F1 (at a threshold).
- Velocity: EPE (endpoint error) in m/s.
- Trajectory: ADE, FDE in meters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OccupancyMetrics:
    auprc: float
    iou: float
    precision: float
    recall: float
    f1: float
    threshold: float


@dataclass
class VelocityMetrics:
    epe: float
    epe_occupied: float


@dataclass
class TrajectoryMetrics:
    ade: float
    fde: float
    horizon_s: float
    n_trajectories: int


def compute_occupancy_auprc(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray | None = None,
    n_thresholds: int = 100,
    gt_threshold: float = 1e-3,
) -> float:
    pred_f = pred.ravel().astype(np.float64)
    gt_f = (gt.ravel() > gt_threshold).astype(np.float64)

    if valid_mask is not None:
        mask = valid_mask.ravel().astype(bool)
        pred_f = pred_f[mask]
        gt_f = gt_f[mask]

    n_pos = gt_f.sum()
    if n_pos == 0:
        return 0.0

    pred_max = pred_f.max() if pred_f.size > 0 else 1.0
    thresholds = np.linspace(0.0, max(pred_max, 1e-6), n_thresholds + 1)
    precisions = []
    recalls = []

    for thr in thresholds:
        pred_pos = pred_f >= thr
        tp = float((pred_pos & (gt_f > 0.5)).sum())
        fp = float((pred_pos & (gt_f <= 0.5)).sum())
        fn = float((~pred_pos & (gt_f > 0.5)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    order = np.argsort(recalls)
    recalls = recalls[order]
    precisions = precisions[order]

    auprc = float(np.trapezoid(precisions, recalls))
    return auprc


def compute_occupancy_at_threshold(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.5,
    valid_mask: np.ndarray | None = None,
    gt_threshold: float = 1e-3,
) -> OccupancyMetrics:
    pred_f = pred.ravel().astype(np.float64)
    gt_f = (gt.ravel() > gt_threshold).astype(np.float64)

    if valid_mask is not None:
        mask = valid_mask.ravel().astype(bool)
        pred_f = pred_f[mask]
        gt_f = gt_f[mask]

    pred_bin = pred_f >= threshold
    gt_bin = gt_f > 0.5

    tp = float((pred_bin & gt_bin).sum())
    fp = float((pred_bin & ~gt_bin).sum())
    fn = float((~pred_bin & gt_bin).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0.0

    auprc = compute_occupancy_auprc(pred, gt, valid_mask, gt_threshold=gt_threshold)

    return OccupancyMetrics(
        auprc=auprc, iou=iou, precision=precision, recall=recall, f1=f1, threshold=threshold
    )


def compute_velocity_epe(
    pred_vx: np.ndarray,
    pred_vy: np.ndarray,
    gt_vx: np.ndarray,
    gt_vy: np.ndarray,
    occupancy_mask: np.ndarray | None = None,
) -> VelocityMetrics:
    """Endpoint error for velocity fields.

    Returns overall EPE and occupied-only EPE (m/s).
    """
    dvx = (pred_vx - gt_vx).astype(np.float64)
    dvy = (pred_vy - gt_vy).astype(np.float64)
    epe_map = np.sqrt(dvx ** 2 + dvy ** 2)

    epe = float(epe_map.mean())

    if occupancy_mask is not None:
        occ = occupancy_mask.ravel().astype(bool)
        epe_occ = float(epe_map.ravel()[occ].mean()) if occ.any() else 0.0
    else:
        epe_occ = epe

    return VelocityMetrics(epe=epe, epe_occupied=epe_occ)


def compute_trajectory_ade_fde(
    pred_trajectories: np.ndarray,
    gt_trajectories: np.ndarray,
    horizon_s: float,
) -> TrajectoryMetrics:
    """ADE and FDE for matched trajectory pairs.

    Args:
        pred_trajectories: (N, T, 2) predicted positions.
        gt_trajectories: (N, T, 2) ground-truth positions.
        horizon_s: prediction horizon in seconds.

    Returns:
        TrajectoryMetrics with ADE, FDE, horizon, and count.
    """
    if pred_trajectories.shape[0] == 0:
        return TrajectoryMetrics(ade=0.0, fde=0.0, horizon_s=horizon_s, n_trajectories=0)

    errors = np.linalg.norm(pred_trajectories - gt_trajectories, axis=2)
    ade = float(errors.mean())
    fde = float(errors[:, -1].mean())

    return TrajectoryMetrics(
        ade=ade,
        fde=fde,
        horizon_s=horizon_s,
        n_trajectories=int(pred_trajectories.shape[0]),
    )
