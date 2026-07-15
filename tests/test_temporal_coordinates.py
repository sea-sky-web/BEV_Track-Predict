"""Tests for temporal coordinate conversions and annotation reader."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json
import tempfile

import numpy as np
import pytest


# ── Coordinate conversions ──


def test_position_id_roundtrip():
    from temporal.coordinates import position_id_to_full_grid, full_grid_to_position_id

    pids = np.array([0, 1, 479, 480, 481, 480 * 1440 - 1])
    row, col = position_id_to_full_grid(pids)
    recovered = full_grid_to_position_id(row, col)
    np.testing.assert_array_equal(recovered, pids)


def test_position_id_to_world_roundtrip():
    from temporal.coordinates import position_id_to_world, world_to_position_id

    pids = np.array([0, 240, 479, 480, 960, 480 * 720, 480 * 1440 - 1])
    wx, wy = position_id_to_world(pids)
    recovered = world_to_position_id(wx, wy)
    np.testing.assert_array_equal(recovered, pids)


def test_full_grid_to_world_cell_center():
    from temporal.coordinates import full_grid_to_world
    from config import ORIGINE_X_M, ORIGINE_Y_M, STEP_M

    wx, wy = full_grid_to_world(np.array([0]), np.array([0]))
    expected_x = ORIGINE_X_M + 0.5 * STEP_M
    expected_y = ORIGINE_Y_M + 0.5 * STEP_M
    np.testing.assert_allclose(wx, expected_x, atol=1e-10)
    np.testing.assert_allclose(wy, expected_y, atol=1e-10)


def test_full_vs_reduced_world_consistency():
    """Full grid center and reduced grid center for same cell should give close world coords."""
    from temporal.coordinates import full_grid_to_world, reduced_to_world

    wx_full, wy_full = full_grid_to_world(np.array([2]), np.array([4]))
    wx_red, wy_red = reduced_to_world(np.array([2.0 / 4]), np.array([4.0 / 4]))
    assert abs(wx_full.item() - wx_red.item()) < 0.05
    assert abs(wy_full.item() - wy_red.item()) < 0.05


def test_grid_shapes():
    from temporal.coordinates import grid_shape_full, grid_shape_reduced

    assert grid_shape_full() == (480, 1440)
    assert grid_shape_reduced(4) == (120, 360)


def test_boundary_position_ids():
    from temporal.coordinates import position_id_to_full_grid

    row, col = position_id_to_full_grid(0)
    assert int(row) == 0 and int(col) == 0

    max_pid = 480 * 1440 - 1
    row, col = position_id_to_full_grid(max_pid)
    assert int(row) == 479 and int(col) == 1439


def test_unit_sizes():
    """Full grid cell = 0.025m, reduced grid cell = 0.1m."""
    from temporal.coordinates import full_grid_to_world, reduced_to_world

    wx0, _ = full_grid_to_world(np.array([0]), np.array([0]))
    wx1, _ = full_grid_to_world(np.array([1]), np.array([0]))
    np.testing.assert_allclose(wx1 - wx0, 0.025, atol=1e-10)

    wx0r, _ = reduced_to_world(np.array([0.0]), np.array([0.0]))
    wx1r, _ = reduced_to_world(np.array([1.0]), np.array([0.0]))
    np.testing.assert_allclose(wx1r - wx0r, 0.1, atol=1e-10)


# ── Time utilities ──


def test_frame_to_timestamp():
    from temporal.time_utils import frame_index_to_timestamp, timestamp_to_frame_index

    ts = frame_index_to_timestamp(np.array([0, 1, 2, 100]))
    np.testing.assert_allclose(ts, [0.0, 0.5, 1.0, 50.0])

    fi = timestamp_to_frame_index(np.array([0.0, 0.5, 1.0, 50.0]))
    np.testing.assert_array_equal(fi, [0, 1, 2, 100])


def test_split_ranges():
    from temporal.time_utils import get_split_range

    assert get_split_range("train") == (0, 320)
    assert get_split_range("val") == (320, 360)
    assert get_split_range("test") == (360, 400)


def test_temporal_windows():
    from temporal.time_utils import make_temporal_windows

    windows = make_temporal_windows(n_frames=10, history_len=4, future_len=4, frame_offset=0)
    assert len(windows) == 3  # 10 - 8 + 1
    assert windows[0]["history_indices"] == [0, 1, 2, 3]
    assert windows[0]["future_indices"] == [4, 5, 6, 7]
    assert windows[2]["history_indices"] == [2, 3, 4, 5]
    assert windows[2]["future_indices"] == [6, 7, 8, 9]


# ── Annotation reader ──


def _make_annotations_dir(tmp_path, n_frames=5, n_people=3):
    ann_dir = tmp_path / "annotations_positions"
    ann_dir.mkdir()
    for fi in range(n_frames):
        objects = []
        for pid in range(n_people):
            pos_id = pid * 480 + fi
            objects.append({"personID": pid, "positionID": pos_id})
        with open(ann_dir / f"{fi:08d}.json", "w") as f:
            json.dump(objects, f)
    return ann_dir


def test_load_annotations(tmp_path):
    from temporal.annotation_reader import load_all_annotations

    ann_dir = _make_annotations_dir(tmp_path, n_frames=5, n_people=3)
    frames = load_all_annotations(ann_dir)
    assert len(frames) == 5
    assert len(frames[0]) == 3
    assert frames[0][0].person_id == 0
    assert frames[0][0].frame_index == 0


def test_load_annotations_with_offset(tmp_path):
    from temporal.annotation_reader import load_all_annotations

    ann_dir = _make_annotations_dir(tmp_path, n_frames=10, n_people=2)
    frames = load_all_annotations(ann_dir, frame_start=3, max_frames=4)
    assert len(frames) == 4
    assert frames[0][0].frame_index == 3


def test_build_trajectories(tmp_path):
    from temporal.annotation_reader import load_all_annotations, build_trajectories

    ann_dir = _make_annotations_dir(tmp_path, n_frames=5, n_people=3)
    frames = load_all_annotations(ann_dir)
    trajs = build_trajectories(frames)
    assert len(trajs) == 3
    for pid, traj in trajs.items():
        assert len(traj.detections) == 5
        assert traj.person_id == pid


def test_trajectory_velocity_static(tmp_path):
    """Person at fixed positionID → zero velocity."""
    from temporal.annotation_reader import load_all_annotations, build_trajectories, compute_velocities

    ann_dir = tmp_path / "annotations_positions"
    ann_dir.mkdir()
    for fi in range(5):
        objects = [{"personID": 0, "positionID": 100}]
        with open(ann_dir / f"{fi:08d}.json", "w") as f:
            json.dump(objects, f)

    frames = load_all_annotations(ann_dir)
    trajs = build_trajectories(frames)
    vel = compute_velocities(trajs[0])
    np.testing.assert_allclose(vel, 0.0, atol=1e-10)


def test_trajectory_velocity_linear(tmp_path):
    """Person moving +1 row per frame → velocity in x direction."""
    from temporal.annotation_reader import load_all_annotations, build_trajectories, compute_velocities
    from config import STEP_M

    ann_dir = tmp_path / "annotations_positions"
    ann_dir.mkdir()
    for fi in range(5):
        pos_id = fi  # row = fi, col = 0
        objects = [{"personID": 0, "positionID": pos_id}]
        with open(ann_dir / f"{fi:08d}.json", "w") as f:
            json.dump(objects, f)

    frames = load_all_annotations(ann_dir)
    trajs = build_trajectories(frames)
    vel = compute_velocities(trajs[0])

    expected_vx = STEP_M / 0.5  # 0.025 m / 0.5 s = 0.05 m/s
    for i in range(1, 4):
        np.testing.assert_allclose(vel[i, 0], expected_vx, atol=1e-10)
        np.testing.assert_allclose(vel[i, 1], 0.0, atol=1e-10)


def test_empty_frame(tmp_path):
    from temporal.annotation_reader import load_all_annotations

    ann_dir = tmp_path / "annotations_positions"
    ann_dir.mkdir()
    with open(ann_dir / "00000000.json", "w") as f:
        json.dump([], f)

    frames = load_all_annotations(ann_dir)
    assert len(frames) == 1
    assert len(frames[0]) == 0


def test_person_gap(tmp_path):
    """Person disappears for a frame then reappears."""
    from temporal.annotation_reader import load_all_annotations, build_trajectories, compute_velocities

    ann_dir = tmp_path / "annotations_positions"
    ann_dir.mkdir()
    for fi in range(5):
        objects = []
        if fi != 2:
            objects.append({"personID": 0, "positionID": 100})
        with open(ann_dir / f"{fi:08d}.json", "w") as f:
            json.dump(objects, f)

    frames = load_all_annotations(ann_dir)
    trajs = build_trajectories(frames)
    traj = trajs[0]
    assert len(traj.detections) == 4
    vel = compute_velocities(traj)
    np.testing.assert_allclose(vel, 0.0, atol=1e-10)


# ── JSONL schemas ──


def test_jsonl_roundtrip(tmp_path):
    from temporal.schemas import write_jsonl, read_jsonl

    records = [
        {"frame_index": 0, "world_x_m": 1.0, "world_y_m": 2.0},
        {"frame_index": 1, "world_x_m": 1.1, "world_y_m": 2.1},
    ]
    path = tmp_path / "test.jsonl"
    write_jsonl(records, path)
    loaded = read_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0]["world_x_m"] == 1.0


def test_npz_roundtrip(tmp_path):
    from temporal.schemas import save_fields_npz, load_fields_npz

    h, w = 120, 360
    occ = np.random.rand(h, w).astype(np.float32)
    vx = np.random.rand(h, w).astype(np.float32)
    vy = np.random.rand(h, w).astype(np.float32)
    conf = np.random.rand(h, w).astype(np.float32)
    valid = np.ones((h, w), dtype=np.float32)

    path = tmp_path / "fields.npz"
    save_fields_npz(path, occ, vx, vy, conf, valid, frame_index=42)
    loaded = load_fields_npz(path)

    np.testing.assert_array_equal(loaded["occupancy"], occ)
    np.testing.assert_array_equal(loaded["vx"], vx)
    np.testing.assert_array_equal(loaded["frame_index"], 42)
