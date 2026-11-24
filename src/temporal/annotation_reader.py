"""Temporal annotation reader for WildTrack.

Reads annotations_positions/ JSON files with personID and positionID,
producing structured records with world coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from temporal.coordinates import position_id_to_world


FRAME_RATE_HZ = 2.0


@dataclass
class Detection:
    frame_index: int
    frame_stem: str
    person_id: int
    position_id: int
    world_x_m: float
    world_y_m: float


@dataclass
class Trajectory:
    person_id: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def frame_indices(self) -> list[int]:
        return [d.frame_index for d in self.detections]

    @property
    def positions(self) -> np.ndarray:
        return np.array([[d.world_x_m, d.world_y_m] for d in self.detections], dtype=np.float64)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([d.frame_index / FRAME_RATE_HZ for d in self.detections], dtype=np.float64)


def read_annotation_file(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_all_annotations(
    annotations_dir: Path,
    frame_start: int = 0,
    max_frames: int = -1,
) -> list[list[Detection]]:
    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON annotation files in {annotations_dir}")

    if frame_start > 0:
        json_files = json_files[frame_start:]
    if max_frames > 0:
        json_files = json_files[:max_frames]

    all_frames: list[list[Detection]] = []
    for frame_index_offset, jf in enumerate(json_files):
        frame_index = frame_start + frame_index_offset
        stem = jf.stem
        objects = read_annotation_file(jf)

        frame_dets: list[Detection] = []
        for obj in objects:
            pid = obj.get("personID", None)
            pos_id = obj.get("positionID", None)
            if pid is None or pos_id is None:
                continue
            pid = int(pid)
            pos_id = int(pos_id)

            wx, wy = position_id_to_world(pos_id)
            frame_dets.append(Detection(
                frame_index=frame_index,
                frame_stem=stem,
                person_id=pid,
                position_id=pos_id,
                world_x_m=float(wx),
                world_y_m=float(wy),
            ))
        all_frames.append(frame_dets)

    return all_frames


def build_trajectories(frames: list[list[Detection]]) -> dict[int, Trajectory]:
    trajectories: dict[int, Trajectory] = {}
    for frame_dets in frames:
        for det in frame_dets:
            if det.person_id not in trajectories:
                trajectories[det.person_id] = Trajectory(person_id=det.person_id)
            trajectories[det.person_id].detections.append(det)

    for traj in trajectories.values():
        traj.detections.sort(key=lambda d: d.frame_index)

    return trajectories


def compute_velocities(traj: Trajectory, dt: float = 1.0 / FRAME_RATE_HZ) -> np.ndarray:
    """Compute per-detection velocity via backward finite differences.

    Returns shape (N, 2) in m/s. Uses backward diff (pos[i] - pos[i-1]) / dt
    for all points with a valid predecessor. First point uses forward diff
    only if no backward neighbor. Does not interpolate across frame gaps > 1.
    """
    pos = traj.positions
    frames = np.array(traj.frame_indices, dtype=np.int64)
    n = len(frames)
    vel = np.zeros((n, 2), dtype=np.float64)

    if n < 2:
        return vel

    for i in range(n):
        if i == 0:
            if frames[1] - frames[0] == 1:
                vel[0] = (pos[1] - pos[0]) / dt
        else:
            gap_prev = frames[i] - frames[i - 1]
            if gap_prev == 1:
                vel[i] = (pos[i] - pos[i - 1]) / dt

    return vel
