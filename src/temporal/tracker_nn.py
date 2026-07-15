"""Nearest-neighbor baseline tracker.

Greedy distance matching without motion model.
"""

from __future__ import annotations

import numpy as np

from temporal.tracker_base import BaseTracker, TrackerOutput, TrackState


class NearestNeighborTracker(BaseTracker):

    def __init__(self, dist_gate: float = 1.0, max_age: int = 2, min_hits: int = 2):
        self.dist_gate = dist_gate
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[TrackState] = []
        self._next_id = 0

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, detections: np.ndarray, frame_index: int) -> TrackerOutput:
        dets = np.atleast_2d(detections)
        if dets.ndim != 2 or (dets.shape[0] > 0 and dets.shape[1] < 2):
            dets = np.empty((0, 2), dtype=np.float64)

        n_dets = dets.shape[0]
        n_tracks = len(self._tracks)

        matched_det = set()
        matched_trk = set()

        if n_dets > 0 and n_tracks > 0:
            track_pos = np.array([[t.world_x_m, t.world_y_m] for t in self._tracks])
            cost = np.linalg.norm(track_pos[:, None, :] - dets[None, :, :2], axis=2)

            flat_order = np.argsort(cost, axis=None)
            for idx in flat_order:
                ti, di = divmod(int(idx), n_dets)
                if ti in matched_trk or di in matched_det:
                    continue
                if cost[ti, di] > self.dist_gate:
                    break
                self._tracks[ti].world_x_m = float(dets[di, 0])
                self._tracks[ti].world_y_m = float(dets[di, 1])
                self._tracks[ti].hits += 1
                self._tracks[ti].time_since_update = 0
                self._tracks[ti].age += 1
                if self._tracks[ti].hits >= self.min_hits:
                    self._tracks[ti].confirmed = True
                matched_det.add(di)
                matched_trk.add(ti)

        for di in range(n_dets):
            if di not in matched_det:
                new_track = TrackState(
                    track_id=self._new_id(),
                    world_x_m=float(dets[di, 0]),
                    world_y_m=float(dets[di, 1]),
                    hits=1,
                    age=1,
                )
                if new_track.hits >= self.min_hits:
                    new_track.confirmed = True
                self._tracks.append(new_track)

        for ti in range(n_tracks):
            if ti not in matched_trk:
                self._tracks[ti].time_since_update += 1
                self._tracks[ti].age += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        active = [t for t in self._tracks if t.confirmed]
        return TrackerOutput(frame_index=frame_index, active_tracks=active)
