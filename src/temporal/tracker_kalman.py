"""Kalman filter + Hungarian assignment tracker.

Constant-velocity state: [x, y, vx, vy].
World coordinates in meters, dt = 0.5 s.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from temporal.tracker_base import BaseTracker, TrackerOutput, TrackState
from temporal.time_utils import DT


class KalmanTrack:

    def __init__(self, track_id: int, x: float, y: float, dt: float = DT):
        self.track_id = track_id
        self.dt = dt

        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 10.0, 10.0])

        self.F = np.eye(4, dtype=np.float64)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        q_pos = 0.01
        q_vel = 1.0
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        self.R = np.diag([0.1, 0.1])

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confirmed = False

    @property
    def position(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.state[2]), float(self.state[3])

    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.state[:2]

    def update(self, z: np.ndarray) -> None:
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0


class KalmanHungarianTracker(BaseTracker):

    def __init__(
        self,
        dist_gate: float = 1.0,
        max_age: int = 2,
        min_hits: int = 2,
        dt: float = DT,
    ):
        self.dist_gate = dist_gate
        self.max_age = max_age
        self.min_hits = min_hits
        self.dt = dt
        self._tracks: list[KalmanTrack] = []
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

        for t in self._tracks:
            t.predict()

        n_dets = dets.shape[0]
        n_tracks = len(self._tracks)

        matched_det = set()
        matched_trk = set()

        if n_dets > 0 and n_tracks > 0:
            pred_pos = np.array([t.position for t in self._tracks])
            cost = np.linalg.norm(pred_pos[:, None, :] - dets[None, :, :2], axis=2)

            row_ind, col_ind = linear_sum_assignment(cost)

            for ti, di in zip(row_ind, col_ind):
                if cost[ti, di] <= self.dist_gate:
                    self._tracks[ti].update(dets[di, :2])
                    if self._tracks[ti].hits >= self.min_hits:
                        self._tracks[ti].confirmed = True
                    matched_det.add(di)
                    matched_trk.add(ti)

        for di in range(n_dets):
            if di not in matched_det:
                new_track = KalmanTrack(
                    track_id=self._new_id(),
                    x=float(dets[di, 0]),
                    y=float(dets[di, 1]),
                    dt=self.dt,
                )
                if new_track.hits >= self.min_hits:
                    new_track.confirmed = True
                self._tracks.append(new_track)

        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= self.max_age
        ]

        active = []
        for t in self._tracks:
            if t.confirmed:
                x, y = t.position
                vx, vy = t.velocity
                active.append(TrackState(
                    track_id=t.track_id,
                    world_x_m=x,
                    world_y_m=y,
                    vx=vx,
                    vy=vy,
                    age=t.age,
                    hits=t.hits,
                    time_since_update=t.time_since_update,
                    confirmed=True,
                ))

        return TrackerOutput(frame_index=frame_index, active_tracks=active)
