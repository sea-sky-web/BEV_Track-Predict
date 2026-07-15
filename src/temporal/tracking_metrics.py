"""Multi-object tracking evaluation metrics.

Implements MOTA, IDF1, ID switches, and fragmentations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackingMetrics:
    mota: float
    idf1: float
    id_switches: int
    fragmentations: int
    tp: int
    fp: int
    fn: int
    n_gt: int
    n_pred: int


def _match_frame(
    gt_positions: np.ndarray,
    pred_positions: np.ndarray,
    gt_ids: np.ndarray,
    pred_ids: np.ndarray,
    dist_thr: float,
) -> tuple[list[tuple[int, int]], int, int, int]:
    """Match predictions to GT in one frame.

    Returns (matches, tp, fp, fn) where matches is list of (gt_id, pred_id).
    """
    n_gt = gt_positions.shape[0]
    n_pred = pred_positions.shape[0]

    if n_gt == 0 and n_pred == 0:
        return [], 0, 0, 0
    if n_gt == 0:
        return [], 0, n_pred, 0
    if n_pred == 0:
        return [], 0, 0, n_gt

    cost = np.linalg.norm(gt_positions[:, None, :] - pred_positions[None, :, :], axis=2)

    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    tp = 0
    for gi, pi in zip(row_ind, col_ind):
        if cost[gi, pi] <= dist_thr:
            matches.append((int(gt_ids[gi]), int(pred_ids[pi])))
            tp += 1

    fp = n_pred - tp
    fn = n_gt - tp
    return matches, tp, fp, fn


def evaluate_tracking(
    gt_frames: list[dict],
    pred_frames: list[dict],
    dist_thr: float = 0.5,
) -> TrackingMetrics:
    """Evaluate tracking across frames.

    Each frame is a dict with:
        positions: np.ndarray (N, 2)  -- world coordinates
        ids: np.ndarray (N,)         -- identity labels
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0
    id_switches = 0
    fragmentations = 0

    prev_mapping: dict[int, int] = {}

    idtp_gt: dict[int, int] = {}
    idtp_pred: dict[int, int] = {}
    idfn: dict[int, int] = {}
    idfp: dict[int, int] = {}

    for gt_frame, pred_frame in zip(gt_frames, pred_frames):
        gt_pos = gt_frame["positions"]
        pred_pos = pred_frame["positions"]
        gt_ids = gt_frame["ids"]
        pred_ids = pred_frame["ids"]

        n_gt = gt_pos.shape[0] if gt_pos.ndim == 2 else 0
        n_pred = pred_pos.shape[0] if pred_pos.ndim == 2 else 0
        total_gt += n_gt
        total_pred += n_pred

        if n_gt == 0 and n_pred == 0:
            continue

        if n_gt == 0:
            total_fp += n_pred
            for pid in pred_ids:
                idfp[int(pid)] = idfp.get(int(pid), 0) + 1
            continue

        if n_pred == 0:
            total_fn += n_gt
            for gid in gt_ids:
                idfn[int(gid)] = idfn.get(int(gid), 0) + 1
            continue

        matches, tp, fp, fn = _match_frame(gt_pos, pred_pos, gt_ids, pred_ids, dist_thr)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        curr_mapping: dict[int, int] = {}
        matched_gt_ids = set()
        matched_pred_ids = set()

        for gid, pid in matches:
            curr_mapping[gid] = pid
            matched_gt_ids.add(gid)
            matched_pred_ids.add(pid)

            idtp_gt[gid] = idtp_gt.get(gid, 0) + 1
            idtp_pred[pid] = idtp_pred.get(pid, 0) + 1

            if gid in prev_mapping and prev_mapping[gid] != pid:
                id_switches += 1

        for gid in gt_ids:
            gid_int = int(gid)
            if gid_int not in matched_gt_ids:
                idfn[gid_int] = idfn.get(gid_int, 0) + 1
                if gid_int in prev_mapping:
                    fragmentations += 1

        for pid in pred_ids:
            pid_int = int(pid)
            if pid_int not in matched_pred_ids:
                idfp[pid_int] = idfp.get(pid_int, 0) + 1

        prev_mapping = curr_mapping

    mota = 1.0 - (total_fp + total_fn + id_switches) / max(total_gt, 1)

    total_idtp = sum(idtp_gt.values())
    total_idfn = sum(idfn.values())
    total_idfp = sum(idfp.values())
    idf1_precision = total_idtp / max(total_idtp + total_idfp, 1)
    idf1_recall = total_idtp / max(total_idtp + total_idfn, 1)
    idf1 = 2.0 * idf1_precision * idf1_recall / max(idf1_precision + idf1_recall, 1e-12)

    return TrackingMetrics(
        mota=mota,
        idf1=idf1,
        id_switches=id_switches,
        fragmentations=fragmentations,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        n_gt=total_gt,
        n_pred=total_pred,
    )
