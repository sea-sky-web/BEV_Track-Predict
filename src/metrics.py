"""Detection metrics for WildTrack BEV point predictions."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except Exception:
        # Small deterministic fallback for environments without scipy.
        pairs: list[tuple[float, int, int]] = []
        for row in range(cost.shape[0]):
            for col in range(cost.shape[1]):
                pairs.append((float(cost[row, col]), row, col))
        pairs.sort(key=lambda x: x[0])

        used_rows: set[int] = set()
        used_cols: set[int] = set()
        rows: list[int] = []
        cols: list[int] = []
        for _, row, col in pairs:
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            rows.append(row)
            cols.append(col)
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def compute_moda_modp(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    d_thresh: float = 0.5,
) -> dict[str, float]:
    """
    Compute CLEAR/MVDet-style MODA and MODP for one frame.

    Points must be shaped (N, 2) and expressed in the same distance unit.
    WildTrack evaluation uses meters with a 0.5m matching threshold.
    """
    pred_pts = np.asarray(pred_pts, dtype=np.float64).reshape(-1, 2)
    gt_pts = np.asarray(gt_pts, dtype=np.float64).reshape(-1, 2)
    if d_thresh <= 0.0:
        raise ValueError(f"d_thresh must be positive, got {d_thresh}")

    n_pred = int(pred_pts.shape[0])
    n_gt = int(gt_pts.shape[0])
    if n_pred == 0 or n_gt == 0:
        tp = 0
        fp = n_pred
        fn = n_gt
        moda = 1.0 if n_gt == 0 and n_pred == 0 else (0.0 if n_gt == 0 else 1.0 - (fp + fn) / n_gt)
        return {
            "moda": float(moda),
            "modp": 0.0,
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "n_gt": float(n_gt),
            "modp_sum": 0.0,
        }

    dist = np.sqrt(((pred_pts[:, None, :] - gt_pts[None, :, :]) ** 2).sum(axis=2))
    row_ind, col_ind = _linear_sum_assignment(dist)
    matched = dist[row_ind, col_ind] <= d_thresh
    match_dists = dist[row_ind[matched], col_ind[matched]]

    tp = int(match_dists.shape[0])
    fp = n_pred - tp
    fn = n_gt - tp
    # MODP = mean(1 - d_i / d_thresh) over matched pairs, range [0, 1]
    modp_sum = float((1.0 - match_dists / d_thresh).sum())
    moda = 1.0 - float(fp + fn) / float(n_gt) if n_gt > 0 else (1.0 if fp == 0 else 0.0)
    modp = modp_sum / float(tp) if tp > 0 else 0.0

    return {
        "moda": float(moda),
        "modp": float(modp),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "n_gt": float(n_gt),
        "modp_sum": float(modp_sum),
    }


def aggregate_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-frame MODA/MODP rows and expose precision/recall/F1 too."""
    rows = list(rows)
    tp = float(sum(row.get("tp", 0.0) for row in rows))
    fp = float(sum(row.get("fp", 0.0) for row in rows))
    fn = float(sum(row.get("fn", 0.0) for row in rows))
    n_gt = float(sum(row.get("n_gt", 0.0) for row in rows))
    modp_sum = float(sum(row.get("modp_sum", 0.0) for row in rows))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    moda = 1.0 - (fp + fn) / n_gt if n_gt > 0 else (1.0 if fp == 0 else 0.0)
    modp = modp_sum / tp if tp > 0 else 0.0

    return {
        "moda": float(moda),
        "modp": float(modp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "n_gt": float(n_gt),
        "modp_sum": float(modp_sum),
    }
