"""JSONL and NPZ schema utilities for temporal data I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# -- JSONL point/track records --

DETECTION_FIELDS = [
    "frame_index",
    "frame_stem",
    "world_x_m",
    "world_y_m",
    "score",
]

TRACK_FIELDS = [
    "frame_index",
    "frame_stem",
    "timestamp_s",
    "source",
    "person_id",
    "track_id",
    "world_x_m",
    "world_y_m",
    "score",
    "observed",
]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# -- NPZ field storage --

def save_fields_npz(
    path: Path,
    occupancy: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    confidence: np.ndarray,
    valid_mask: np.ndarray,
    frame_index: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "occupancy": occupancy.astype(np.float32),
        "vx": vx.astype(np.float32),
        "vy": vy.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "valid_mask": valid_mask.astype(np.float32),
        "frame_index": np.array(frame_index, dtype=np.int64),
    }
    if metadata:
        save_dict["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **save_dict)


def load_fields_npz(path: Path) -> dict[str, np.ndarray]:
    data = dict(np.load(path, allow_pickle=False))
    return data


def make_stacked_field(
    occupancy: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    confidence: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    return np.stack([occupancy, vx, vy, confidence, valid_mask], axis=0).astype(np.float32)
