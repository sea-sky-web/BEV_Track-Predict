"""BEV occupancy and velocity field construction from tracked positions.

Operates on the reduced grid (120 x 360, 0.1 m/cell).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from temporal.coordinates import (
    grid_shape_reduced,
    world_to_reduced,
    reduced_to_world,
    DEFAULT_BEV_DOWN,
)
from temporal.time_utils import DT


def _gaussian_kernel_sigma_to_cells(sigma_m: float, cell_m: float) -> float:
    return sigma_m / cell_m


def build_occupancy_field(
    positions_world: np.ndarray,
    sigma_m: float = 0.2,
    bev_down: int = DEFAULT_BEV_DOWN,
) -> np.ndarray:
    """Build occupancy field from world-coordinate positions.

    O(q) = 1 - exp(-sum_i K_sigma(q - p_i))

    Args:
        positions_world: (N, 2) array of [world_x_m, world_y_m].
        sigma_m: Gaussian kernel sigma in meters.
        bev_down: BEV downsampling factor.

    Returns:
        (H_reduced, W_reduced) occupancy field in [0, 1].
    """
    h, w = grid_shape_reduced(bev_down)
    accumulator = np.zeros((h, w), dtype=np.float64)

    if positions_world.shape[0] == 0:
        return accumulator.astype(np.float32)

    from config import STEP_M
    cell_m = STEP_M * bev_down
    sigma_cells = _gaussian_kernel_sigma_to_cells(sigma_m, cell_m)

    for i in range(positions_world.shape[0]):
        row_r, col_r = world_to_reduced(
            positions_world[i, 0], positions_world[i, 1], bev_down
        )
        row_r = float(row_r)
        col_r = float(col_r)

        ri = int(round(row_r))
        ci = int(round(col_r))
        if 0 <= ri < h and 0 <= ci < w:
            accumulator[ri, ci] += 1.0

    if sigma_cells > 0:
        accumulator = gaussian_filter(accumulator, sigma=sigma_cells, mode="constant", cval=0.0)

    occupancy = 1.0 - np.exp(-accumulator)
    return occupancy.astype(np.float32)


def build_velocity_field(
    positions_world: np.ndarray,
    velocities: np.ndarray,
    sigma_m: float = 0.2,
    bev_down: int = DEFAULT_BEV_DOWN,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build kernel-weighted velocity field.

    V(q) = sum_i K(q-p_i) * v_i / (eps + sum_i K(q-p_i))

    Args:
        positions_world: (N, 2) array of [world_x_m, world_y_m].
        velocities: (N, 2) array of [vx, vy] in m/s.
        sigma_m: Gaussian kernel sigma in meters.

    Returns:
        (vx_field, vy_field) each of shape (H_reduced, W_reduced).
    """
    h, w = grid_shape_reduced(bev_down)
    weight_sum = np.zeros((h, w), dtype=np.float64)
    vx_weighted = np.zeros((h, w), dtype=np.float64)
    vy_weighted = np.zeros((h, w), dtype=np.float64)

    if positions_world.shape[0] == 0:
        return vx_weighted.astype(np.float32), vy_weighted.astype(np.float32)

    from config import STEP_M
    cell_m = STEP_M * bev_down
    sigma_cells = _gaussian_kernel_sigma_to_cells(sigma_m, cell_m)

    for i in range(positions_world.shape[0]):
        row_r, col_r = world_to_reduced(
            positions_world[i, 0], positions_world[i, 1], bev_down
        )
        ri = int(round(float(row_r)))
        ci = int(round(float(col_r)))
        if 0 <= ri < h and 0 <= ci < w:
            weight_sum[ri, ci] += 1.0
            vx_weighted[ri, ci] += velocities[i, 0]
            vy_weighted[ri, ci] += velocities[i, 1]

    if sigma_cells > 0:
        weight_sum = gaussian_filter(weight_sum, sigma=sigma_cells, mode="constant", cval=0.0)
        vx_weighted = gaussian_filter(vx_weighted, sigma=sigma_cells, mode="constant", cval=0.0)
        vy_weighted = gaussian_filter(vy_weighted, sigma=sigma_cells, mode="constant", cval=0.0)

    denom = weight_sum + eps
    vx_field = vx_weighted / denom
    vy_field = vy_weighted / denom

    return vx_field.astype(np.float32), vy_field.astype(np.float32)


def build_confidence_field(
    positions_world: np.ndarray,
    scores: np.ndarray,
    sigma_m: float = 0.2,
    bev_down: int = DEFAULT_BEV_DOWN,
    eps: float = 1e-6,
) -> np.ndarray:
    """Build score-weighted confidence field (for detector input only).

    Returns shape (H_reduced, W_reduced).
    """
    h, w = grid_shape_reduced(bev_down)
    weight_sum = np.zeros((h, w), dtype=np.float64)
    score_weighted = np.zeros((h, w), dtype=np.float64)

    if positions_world.shape[0] == 0:
        return score_weighted.astype(np.float32)

    from config import STEP_M
    cell_m = STEP_M * bev_down
    sigma_cells = _gaussian_kernel_sigma_to_cells(sigma_m, cell_m)

    for i in range(positions_world.shape[0]):
        row_r, col_r = world_to_reduced(
            positions_world[i, 0], positions_world[i, 1], bev_down
        )
        ri = int(round(float(row_r)))
        ci = int(round(float(col_r)))
        if 0 <= ri < h and 0 <= ci < w:
            weight_sum[ri, ci] += 1.0
            score_weighted[ri, ci] += scores[i]

    if sigma_cells > 0:
        weight_sum = gaussian_filter(weight_sum, sigma=sigma_cells, mode="constant", cval=0.0)
        score_weighted = gaussian_filter(score_weighted, sigma=sigma_cells, mode="constant", cval=0.0)

    denom = weight_sum + eps
    return (score_weighted / denom).astype(np.float32)


def build_valid_mask(
    bev_down: int = DEFAULT_BEV_DOWN,
) -> np.ndarray:
    """Build a constant valid mask (all ones for now).

    Full valid_mask requires camera projection coverage computation,
    which depends on calibration data. When calibration is available,
    this should be replaced with the union of per-view valid regions.
    """
    h, w = grid_shape_reduced(bev_down)
    return np.ones((h, w), dtype=np.float32)


def build_all_fields(
    positions_world: np.ndarray,
    velocities: np.ndarray,
    sigma_m: float = 0.2,
    bev_down: int = DEFAULT_BEV_DOWN,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    """Build stacked field tensor [occupancy, vx, vy, confidence, valid_mask].

    Returns shape (5, H_reduced, W_reduced).
    """
    occ = build_occupancy_field(positions_world, sigma_m, bev_down)
    vx, vy = build_velocity_field(positions_world, velocities, sigma_m, bev_down)

    if scores is not None:
        conf = build_confidence_field(positions_world, scores, sigma_m, bev_down)
    else:
        conf = np.zeros_like(occ)

    valid = build_valid_mask(bev_down)

    return np.stack([occ, vx, vy, conf, valid], axis=0).astype(np.float32)
