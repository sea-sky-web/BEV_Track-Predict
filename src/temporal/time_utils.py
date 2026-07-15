"""Time utilities for temporal module.

Frame rate: 2 Hz.
timestamp_s = frame_index / 2.0
dt = 0.5 s
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

FRAME_RATE_HZ = 2.0
DT = 1.0 / FRAME_RATE_HZ  # 0.5 s


def frame_index_to_timestamp(frame_index: int | np.ndarray) -> np.ndarray:
    return np.asarray(frame_index, dtype=np.float64) / FRAME_RATE_HZ


def timestamp_to_frame_index(timestamp_s: float | np.ndarray) -> np.ndarray:
    return np.round(np.asarray(timestamp_s, dtype=np.float64) * FRAME_RATE_HZ).astype(np.int64)


def make_temporal_windows(
    n_frames: int,
    history_len: int,
    future_len: int,
    frame_offset: int = 0,
) -> list[dict]:
    """Create non-overlapping temporal windows for train/val/test.

    Returns list of dicts with keys:
        history_indices: list of frame indices for input
        future_indices: list of frame indices for prediction target
    """
    total_len = history_len + future_len
    windows = []
    for start in range(n_frames - total_len + 1):
        abs_start = frame_offset + start
        windows.append({
            "history_indices": list(range(abs_start, abs_start + history_len)),
            "future_indices": list(range(abs_start + history_len, abs_start + total_len)),
        })
    return windows


SPLIT_RANGES = {
    "train": (0, 320),
    "val": (320, 360),
    "test": (360, 400),
}


def get_split_range(split: str) -> tuple[int, int]:
    if split not in SPLIT_RANGES:
        raise ValueError(f"Unknown split '{split}', must be one of {list(SPLIT_RANGES.keys())}")
    return SPLIT_RANGES[split]
