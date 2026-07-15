"""Non-learning prediction baselines for spatiotemporal fields.

1. Persistence: future = current occupancy.
2. Constant Velocity: per-track extrapolation then re-rasterize.
3. Field Advection: semi-Lagrangian advection of occupancy by velocity field.
4. Oracle: GT identity + GT velocity (upper bound).
"""

from __future__ import annotations

import numpy as np

from temporal.coordinates import (
    grid_shape_reduced,
    world_to_reduced,
    reduced_to_world,
    DEFAULT_BEV_DOWN,
)
from temporal.field_builder import build_occupancy_field, build_velocity_field
from temporal.time_utils import DT


def predict_persistence(
    current_occupancy: np.ndarray,
    n_future: int,
) -> list[np.ndarray]:
    return [current_occupancy.copy() for _ in range(n_future)]


def predict_constant_velocity(
    positions_world: np.ndarray,
    velocities: np.ndarray,
    n_future: int,
    dt: float = DT,
    sigma_m: float = 0.2,
    bev_down: int = DEFAULT_BEV_DOWN,
) -> list[np.ndarray]:
    """Extrapolate each tracked position by constant velocity, re-rasterize."""
    predictions = []
    for step in range(1, n_future + 1):
        future_pos = positions_world + velocities * (dt * step)
        occ = build_occupancy_field(future_pos, sigma_m=sigma_m, bev_down=bev_down)
        predictions.append(occ)
    return predictions


def predict_field_advection(
    current_occupancy: np.ndarray,
    vx_field: np.ndarray,
    vy_field: np.ndarray,
    n_future: int,
    dt: float = DT,
    bev_down: int = DEFAULT_BEV_DOWN,
) -> list[np.ndarray]:
    """Semi-Lagrangian advection of occupancy field by velocity field.

    For each grid cell (r, c), trace back by -v*dt to find source,
    then bilinearly interpolate the occupancy from the source location.
    """
    from config import STEP_M
    cell_m = STEP_M * bev_down
    h, w = current_occupancy.shape

    row_grid, col_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    predictions = []
    occ = current_occupancy.astype(np.float64)
    vx = vx_field.astype(np.float64)
    vy = vy_field.astype(np.float64)

    for _ in range(n_future):
        src_row = row_grid - vx * dt / cell_m
        src_col = col_grid - vy * dt / cell_m

        src_row = np.clip(src_row, 0, h - 1)
        src_col = np.clip(src_col, 0, w - 1)

        r0 = np.floor(src_row).astype(int)
        c0 = np.floor(src_col).astype(int)
        r1 = np.minimum(r0 + 1, h - 1)
        c1 = np.minimum(c0 + 1, w - 1)

        dr = src_row - r0
        dc = src_col - c0

        advected = (
            occ[r0, c0] * (1 - dr) * (1 - dc)
            + occ[r1, c0] * dr * (1 - dc)
            + occ[r0, c1] * (1 - dr) * dc
            + occ[r1, c1] * dr * dc
        )

        predictions.append(advected.astype(np.float32))
        occ = advected

    return predictions


def predict_oracle(
    future_positions_per_step: list[np.ndarray],
    sigma_m: float = 0.2,
    bev_down: int = DEFAULT_BEV_DOWN,
) -> list[np.ndarray]:
    """Oracle baseline: build occupancy from GT future positions.

    Args:
        future_positions_per_step: list of (N_t, 2) arrays for each future step.
    """
    predictions = []
    for positions in future_positions_per_step:
        occ = build_occupancy_field(positions, sigma_m=sigma_m, bev_down=bev_down)
        predictions.append(occ)
    return predictions
