"""Base class for multi-object trackers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackState:
    track_id: int
    world_x_m: float
    world_y_m: float
    vx: float = 0.0
    vy: float = 0.0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    confirmed: bool = False


@dataclass
class TrackerOutput:
    frame_index: int
    active_tracks: list[TrackState] = field(default_factory=list)


class BaseTracker(ABC):
    """Abstract multi-object tracker interface."""

    @abstractmethod
    def update(
        self,
        detections: np.ndarray,
        frame_index: int,
    ) -> TrackerOutput:
        """Process one frame of detections.

        Args:
            detections: (N, 2) or (N, 3) array of [world_x_m, world_y_m, (score)].
            frame_index: current frame index.

        Returns:
            TrackerOutput with active (confirmed) tracks.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        ...
