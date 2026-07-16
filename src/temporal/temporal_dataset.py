"""Temporal field sequence dataset for ConvLSTM training.

Builds all-frame fields once at init, then returns sliding-window
(history, future) pairs via __getitem__.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from temporal.annotation_reader import (
    load_all_annotations,
    build_trajectories,
    compute_velocities,
)
from temporal.field_builder import build_all_fields
from temporal.time_utils import make_temporal_windows, get_split_range, DT


class FieldSequenceDataset(Dataset):

    def __init__(
        self,
        annotations_dir: Path | str,
        split: str = "train",
        history_len: int = 4,
        future_len: int = 4,
        sigma_m: float = 0.2,
        bev_down: int = 4,
    ):
        annotations_dir = Path(annotations_dir)
        start, end = get_split_range(split)
        n_frames = end - start

        frames = load_all_annotations(annotations_dir, frame_start=start, max_frames=n_frames)
        trajectories = build_trajectories(frames)

        person_velocities: dict[int, dict[int, np.ndarray]] = {}
        for pid, traj in trajectories.items():
            vel = compute_velocities(traj, dt=DT)
            for i, det in enumerate(traj.detections):
                if det.frame_index not in person_velocities:
                    person_velocities[det.frame_index] = {}
                person_velocities[det.frame_index][pid] = vel[i]

        from temporal.coordinates import grid_shape_reduced
        gh, gw = grid_shape_reduced(bev_down)
        all_fields = np.zeros((n_frames, 5, gh, gw), dtype=np.float32)
        for fi in range(n_frames):
            abs_fi = start + fi
            dets = frames[fi]

            if len(dets) == 0:
                continue

            positions = np.array([[d.world_x_m, d.world_y_m] for d in dets])
            velocities = np.zeros((len(dets), 2), dtype=np.float64)
            for j, d in enumerate(dets):
                vel_map = person_velocities.get(d.frame_index, {})
                if d.person_id in vel_map:
                    velocities[j] = vel_map[d.person_id]

            all_fields[fi] = build_all_fields(positions, velocities, sigma_m=sigma_m, bev_down=bev_down)

        self.fields = all_fields
        self.windows = make_temporal_windows(n_frames, history_len, future_len, frame_offset=0)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[idx]
        hist_idx = window["history_indices"]
        fut_idx = window["future_indices"]

        history = torch.from_numpy(self.fields[hist_idx])
        future = torch.from_numpy(self.fields[fut_idx])
        return history, future
