"""Coordinate conversions between positionID, full grid, reduced grid, and world meters.

Canonical coordinate: (world_x_m, world_y_m) in meters.
Full grid: 480 x 1440, 0.025 m/cell.
Reduced grid (bev_down=4): 120 x 360, 0.1 m/cell.
"""

from __future__ import annotations

import numpy as np

from config import (
    DEFAULT_BEV_DOWN,
    NB_HEIGHT,
    NB_WIDTH,
    ORIGINE_X_M,
    ORIGINE_Y_M,
    STEP_M,
)


def position_id_to_full_grid(position_id: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pid = np.asarray(position_id, dtype=np.int64)
    row_full = pid % NB_HEIGHT
    col_full = pid // NB_HEIGHT
    return row_full, col_full


def full_grid_to_position_id(row_full: int | np.ndarray, col_full: int | np.ndarray) -> np.ndarray:
    return np.asarray(col_full, dtype=np.int64) * NB_HEIGHT + np.asarray(row_full, dtype=np.int64)


def full_grid_to_world(row_full: np.ndarray, col_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_f = np.asarray(row_full, dtype=np.float64)
    col_f = np.asarray(col_full, dtype=np.float64)
    world_x_m = ORIGINE_X_M + (row_f + 0.5) * STEP_M
    world_y_m = ORIGINE_Y_M + (col_f + 0.5) * STEP_M
    return world_x_m, world_y_m


def world_to_full_grid(world_x_m: np.ndarray, world_y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wx = np.asarray(world_x_m, dtype=np.float64)
    wy = np.asarray(world_y_m, dtype=np.float64)
    row_full = (wx - ORIGINE_X_M) / STEP_M - 0.5
    col_full = (wy - ORIGINE_Y_M) / STEP_M - 0.5
    return row_full, col_full


def position_id_to_world(position_id: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_full, col_full = position_id_to_full_grid(position_id)
    return full_grid_to_world(row_full, col_full)


def world_to_position_id(world_x_m: np.ndarray, world_y_m: np.ndarray) -> np.ndarray:
    row_full, col_full = world_to_full_grid(world_x_m, world_y_m)
    return full_grid_to_position_id(np.round(row_full).astype(np.int64), np.round(col_full).astype(np.int64))


def full_grid_to_reduced(
    row_full: np.ndarray, col_full: np.ndarray, bev_down: int = DEFAULT_BEV_DOWN
) -> tuple[np.ndarray, np.ndarray]:
    row_reduced = np.asarray(row_full, dtype=np.float64) / bev_down
    col_reduced = np.asarray(col_full, dtype=np.float64) / bev_down
    return row_reduced, col_reduced


def reduced_to_full_grid(
    row_reduced: np.ndarray, col_reduced: np.ndarray, bev_down: int = DEFAULT_BEV_DOWN
) -> tuple[np.ndarray, np.ndarray]:
    row_full = np.asarray(row_reduced, dtype=np.float64) * bev_down
    col_full = np.asarray(col_reduced, dtype=np.float64) * bev_down
    return row_full, col_full


def reduced_to_world(
    row_reduced: np.ndarray, col_reduced: np.ndarray, bev_down: int = DEFAULT_BEV_DOWN
) -> tuple[np.ndarray, np.ndarray]:
    cell_m = STEP_M * bev_down
    wx = ORIGINE_X_M + (np.asarray(row_reduced, dtype=np.float64) + 0.5) * cell_m
    wy = ORIGINE_Y_M + (np.asarray(col_reduced, dtype=np.float64) + 0.5) * cell_m
    return wx, wy


def world_to_reduced(
    world_x_m: np.ndarray, world_y_m: np.ndarray, bev_down: int = DEFAULT_BEV_DOWN
) -> tuple[np.ndarray, np.ndarray]:
    cell_m = STEP_M * bev_down
    row_reduced = (np.asarray(world_x_m, dtype=np.float64) - ORIGINE_X_M) / cell_m - 0.5
    col_reduced = (np.asarray(world_y_m, dtype=np.float64) - ORIGINE_Y_M) / cell_m - 0.5
    return row_reduced, col_reduced


def grid_shape_full() -> tuple[int, int]:
    return NB_HEIGHT, NB_WIDTH


def grid_shape_reduced(bev_down: int = DEFAULT_BEV_DOWN) -> tuple[int, int]:
    return NB_HEIGHT // bev_down, NB_WIDTH // bev_down
