"""Load detector JSONL output and convert to the same format as GT annotations.

Bridges Module 1 detector output → Module 2 temporal pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from temporal.annotation_reader import Detection


def load_detections_jsonl(path: Path) -> dict[int, list[dict]]:
    """Load detector JSONL output grouped by frame_index.

    Returns:
        {frame_index: [{"world_x_m": float, "world_y_m": float, "score": float}, ...]}
    """
    by_frame: dict[int, list[dict]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fi = rec["frame_index"]
            if fi not in by_frame:
                by_frame[fi] = []
            by_frame[fi].append(rec)
    return by_frame


def detections_to_positions(
    det_by_frame: dict[int, list[dict]],
    frame_start: int,
    n_frames: int,
) -> list[np.ndarray]:
    """Convert detector JSONL to per-frame position arrays.

    Returns:
        list of (N_i, 2) arrays for each frame, same interface as GT.
    """
    result = []
    for fi in range(frame_start, frame_start + n_frames):
        dets = det_by_frame.get(fi, [])
        if dets:
            positions = np.array([[d["world_x_m"], d["world_y_m"]] for d in dets],
                                 dtype=np.float64)
        else:
            positions = np.empty((0, 2), dtype=np.float64)
        result.append(positions)
    return result


def detections_to_scores(
    det_by_frame: dict[int, list[dict]],
    frame_start: int,
    n_frames: int,
) -> list[np.ndarray]:
    """Extract per-frame score arrays from detector JSONL."""
    result = []
    for fi in range(frame_start, frame_start + n_frames):
        dets = det_by_frame.get(fi, [])
        if dets:
            scores = np.array([d.get("score", 1.0) for d in dets], dtype=np.float64)
        else:
            scores = np.empty((0,), dtype=np.float64)
        result.append(scores)
    return result


def match_detections_to_gt(
    det_positions: np.ndarray,
    gt_positions: np.ndarray,
    gt_ids: np.ndarray,
    dist_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian matching of detections to GT for GT-association evaluation.

    Returns:
        matched_positions: (M, 2) — detector positions matched to GT
        matched_ids: (M,) — GT person IDs assigned to matched detections
    """
    from scipy.optimize import linear_sum_assignment

    if det_positions.shape[0] == 0 or gt_positions.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int64)

    cost = np.linalg.norm(
        det_positions[:, None, :] - gt_positions[None, :, :], axis=2
    )
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pos = []
    matched_ids = []
    for di, gi in zip(row_ind, col_ind):
        if cost[di, gi] <= dist_thr:
            matched_pos.append(det_positions[di])
            matched_ids.append(int(gt_ids[gi]))

    if matched_pos:
        return np.array(matched_pos), np.array(matched_ids, dtype=np.int64)
    return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int64)
