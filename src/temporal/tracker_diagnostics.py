"""Tracker diagnostic: ID switch and FP analysis.

Identifies per-frame IDSW events, their root cause, and suggests
tracker parameter adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class IDSwitchEvent:
    frame_index: int
    gt_id: int
    old_track_id: int
    new_track_id: int
    gt_position: tuple[float, float]
    distance_to_old: float
    distance_to_new: float


@dataclass
class FalsePositiveEvent:
    frame_index: int
    track_id: int
    position: tuple[float, float]
    nearest_gt_distance: float


@dataclass
class TrackerDiagnostics:
    id_switch_events: list[IDSwitchEvent]
    fp_events: list[FalsePositiveEvent]
    idsw_by_gt_id: dict[int, int]
    fp_by_track_id: dict[int, int]
    total_idsw: int
    total_fp: int
    summary: str


def diagnose_tracker(
    gt_frames: list[dict],
    pred_frames: list[dict],
    dist_thr: float = 0.5,
    frame_offset: int = 0,
) -> TrackerDiagnostics:
    """Run tracking evaluation with detailed per-event diagnostics.

    Args:
        gt_frames: list of {"positions": (N,2), "ids": (N,)} per frame.
        pred_frames: list of {"positions": (N,2), "ids": (N,)} per frame.
        dist_thr: matching distance threshold in meters.
        frame_offset: absolute frame index of first frame (for reporting).

    Returns:
        TrackerDiagnostics with per-event details.
    """
    idsw_events: list[IDSwitchEvent] = []
    fp_events: list[FalsePositiveEvent] = []
    prev_mapping: dict[int, int] = {}
    prev_pred_positions: dict[int, np.ndarray] = {}

    for fi, (gt_frame, pred_frame) in enumerate(zip(gt_frames, pred_frames)):
        abs_frame = frame_offset + fi
        gt_pos = gt_frame["positions"]
        pred_pos = pred_frame["positions"]
        gt_ids = gt_frame["ids"]
        pred_ids = pred_frame["ids"]

        n_gt = gt_pos.shape[0] if gt_pos.ndim == 2 else 0
        n_pred = pred_pos.shape[0] if pred_pos.ndim == 2 else 0

        curr_pred_positions = {}
        if n_pred > 0:
            for j, pid in enumerate(pred_ids):
                curr_pred_positions[int(pid)] = pred_pos[j]

        if n_gt == 0 or n_pred == 0:
            if n_pred > 0:
                for j, pid in enumerate(pred_ids):
                    fp_events.append(FalsePositiveEvent(
                        frame_index=abs_frame,
                        track_id=int(pid),
                        position=(float(pred_pos[j, 0]), float(pred_pos[j, 1])),
                        nearest_gt_distance=float("inf"),
                    ))
            prev_mapping = {}
            prev_pred_positions = curr_pred_positions
            continue

        cost = np.linalg.norm(gt_pos[:, None, :] - pred_pos[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        matched_pred_idx = set()
        for gi, pi in zip(row_ind, col_ind):
            if cost[gi, pi] <= dist_thr:
                matches.append((int(gt_ids[gi]), int(pred_ids[pi]), gi, pi))
                matched_pred_idx.add(pi)

        curr_mapping: dict[int, int] = {}
        for gid, pid, gi, pi in matches:
            curr_mapping[gid] = pid

            if gid in prev_mapping and prev_mapping[gid] != pid:
                old_pid = prev_mapping[gid]
                dist_to_old = float("inf")
                if old_pid in curr_pred_positions:
                    dist_to_old = float(np.linalg.norm(
                        gt_pos[gi] - curr_pred_positions[old_pid]
                    ))
                elif old_pid in prev_pred_positions:
                    dist_to_old = float(np.linalg.norm(
                        gt_pos[gi] - prev_pred_positions[old_pid]
                    ))

                idsw_events.append(IDSwitchEvent(
                    frame_index=abs_frame,
                    gt_id=gid,
                    old_track_id=old_pid,
                    new_track_id=pid,
                    gt_position=(float(gt_pos[gi, 0]), float(gt_pos[gi, 1])),
                    distance_to_old=dist_to_old,
                    distance_to_new=float(cost[gi, pi]),
                ))

        for pi in range(n_pred):
            if pi not in matched_pred_idx:
                nearest_gt = float(cost[:, pi].min()) if n_gt > 0 else float("inf")
                fp_events.append(FalsePositiveEvent(
                    frame_index=abs_frame,
                    track_id=int(pred_ids[pi]),
                    position=(float(pred_pos[pi, 0]), float(pred_pos[pi, 1])),
                    nearest_gt_distance=nearest_gt,
                ))

        prev_mapping = curr_mapping
        prev_pred_positions = curr_pred_positions

    idsw_by_gt = {}
    for ev in idsw_events:
        idsw_by_gt[ev.gt_id] = idsw_by_gt.get(ev.gt_id, 0) + 1

    fp_by_track = {}
    for ev in fp_events:
        fp_by_track[ev.track_id] = fp_by_track.get(ev.track_id, 0) + 1

    fp_near_gate = [e for e in fp_events if e.nearest_gt_distance < dist_thr * 2]
    fp_far = [e for e in fp_events if e.nearest_gt_distance >= dist_thr * 2]

    summary_lines = [
        f"Total IDSW: {len(idsw_events)}, Total FP: {len(fp_events)}",
        f"IDSW by GT person: {idsw_by_gt}",
        f"FP near gate (<{dist_thr*2:.1f}m): {len(fp_near_gate)}, "
        f"FP far (>={dist_thr*2:.1f}m): {len(fp_far)}",
    ]

    if idsw_events:
        avg_dist_new = np.mean([e.distance_to_new for e in idsw_events])
        summary_lines.append(f"Avg IDSW match distance: {avg_dist_new:.3f}m")

    if fp_events:
        top_fp_tracks = sorted(fp_by_track.items(), key=lambda x: -x[1])[:5]
        summary_lines.append(f"Top FP tracks: {top_fp_tracks}")

    return TrackerDiagnostics(
        id_switch_events=idsw_events,
        fp_events=fp_events,
        idsw_by_gt_id=idsw_by_gt,
        fp_by_track_id=fp_by_track,
        total_idsw=len(idsw_events),
        total_fp=len(fp_events),
        summary="\n".join(summary_lines),
    )
