"""Tests for temporal.detection_loader — JSONL loading, position/score extraction, Hungarian matching."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from temporal.detection_loader import (
    load_detections_jsonl,
    detections_to_positions,
    detections_to_scores,
    match_detections_to_gt,
)


@pytest.fixture
def sample_jsonl(tmp_path):
    lines = [
        {"frame_index": 0, "world_x_m": 1.0, "world_y_m": 2.0, "score": 0.9},
        {"frame_index": 0, "world_x_m": 3.0, "world_y_m": 4.0, "score": 0.8},
        {"frame_index": 1, "world_x_m": 1.1, "world_y_m": 2.1, "score": 0.85},
        {"frame_index": 2, "world_x_m": 5.0, "world_y_m": 6.0, "score": 0.7},
    ]
    path = tmp_path / "detections.jsonl"
    with open(path, "w") as f:
        for rec in lines:
            f.write(json.dumps(rec) + "\n")
    return path


class TestLoadDetectionsJsonl:
    def test_groups_by_frame(self, sample_jsonl):
        by_frame = load_detections_jsonl(sample_jsonl)
        assert set(by_frame.keys()) == {0, 1, 2}
        assert len(by_frame[0]) == 2
        assert len(by_frame[1]) == 1
        assert len(by_frame[2]) == 1

    def test_preserves_coordinates(self, sample_jsonl):
        by_frame = load_detections_jsonl(sample_jsonl)
        assert by_frame[0][0]["world_x_m"] == 1.0
        assert by_frame[0][0]["world_y_m"] == 2.0

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        by_frame = load_detections_jsonl(path)
        assert by_frame == {}

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        content = '\n{"frame_index": 0, "world_x_m": 1.0, "world_y_m": 2.0, "score": 0.9}\n\n'
        path.write_text(content)
        by_frame = load_detections_jsonl(path)
        assert len(by_frame) == 1
        assert len(by_frame[0]) == 1


class TestDetectionsToPositions:
    def test_basic(self, sample_jsonl):
        by_frame = load_detections_jsonl(sample_jsonl)
        positions = detections_to_positions(by_frame, frame_start=0, n_frames=3)
        assert len(positions) == 3
        assert positions[0].shape == (2, 2)
        assert positions[1].shape == (1, 2)
        assert positions[2].shape == (1, 2)

    def test_missing_frame_returns_empty(self, sample_jsonl):
        by_frame = load_detections_jsonl(sample_jsonl)
        positions = detections_to_positions(by_frame, frame_start=0, n_frames=5)
        assert positions[3].shape == (0, 2)
        assert positions[4].shape == (0, 2)

    def test_frame_offset(self, sample_jsonl):
        by_frame = load_detections_jsonl(sample_jsonl)
        positions = detections_to_positions(by_frame, frame_start=1, n_frames=2)
        assert len(positions) == 2
        assert positions[0].shape == (1, 2)
        np.testing.assert_allclose(positions[0][0], [1.1, 2.1])

    def test_dtype_float64(self, sample_jsonl):
        by_frame = load_detections_jsonl(sample_jsonl)
        positions = detections_to_positions(by_frame, frame_start=0, n_frames=1)
        assert positions[0].dtype == np.float64


class TestDetectionsToScores:
    def test_basic(self, sample_jsonl):
        by_frame = load_detections_jsonl(sample_jsonl)
        scores = detections_to_scores(by_frame, frame_start=0, n_frames=3)
        assert len(scores) == 3
        assert scores[0].shape == (2,)
        np.testing.assert_allclose(scores[0], [0.9, 0.8])

    def test_missing_frame_returns_empty(self, sample_jsonl):
        by_frame = load_detections_jsonl(sample_jsonl)
        scores = detections_to_scores(by_frame, frame_start=10, n_frames=2)
        assert scores[0].shape == (0,)

    def test_default_score(self, tmp_path):
        path = tmp_path / "no_score.jsonl"
        path.write_text('{"frame_index": 0, "world_x_m": 1.0, "world_y_m": 2.0}\n')
        by_frame = load_detections_jsonl(path)
        scores = detections_to_scores(by_frame, frame_start=0, n_frames=1)
        np.testing.assert_allclose(scores[0], [1.0])


class TestMatchDetectionsToGt:
    def test_perfect_match(self):
        det = np.array([[1.0, 2.0], [3.0, 4.0]])
        gt = np.array([[1.0, 2.0], [3.0, 4.0]])
        gt_ids = np.array([10, 20])
        pos, ids = match_detections_to_gt(det, gt, gt_ids, dist_thr=0.5)
        assert pos.shape == (2, 2)
        assert set(ids.tolist()) == {10, 20}

    def test_within_threshold(self):
        det = np.array([[1.1, 2.1]])
        gt = np.array([[1.0, 2.0]])
        gt_ids = np.array([5])
        pos, ids = match_detections_to_gt(det, gt, gt_ids, dist_thr=0.5)
        assert pos.shape == (1, 2)
        assert ids[0] == 5
        np.testing.assert_allclose(pos[0], [1.1, 2.1])

    def test_beyond_threshold(self):
        det = np.array([[10.0, 20.0]])
        gt = np.array([[1.0, 2.0]])
        gt_ids = np.array([5])
        pos, ids = match_detections_to_gt(det, gt, gt_ids, dist_thr=0.5)
        assert pos.shape == (0, 2)
        assert ids.shape == (0,)

    def test_empty_detections(self):
        det = np.empty((0, 2))
        gt = np.array([[1.0, 2.0]])
        gt_ids = np.array([5])
        pos, ids = match_detections_to_gt(det, gt, gt_ids, dist_thr=0.5)
        assert pos.shape == (0, 2)
        assert ids.shape == (0,)

    def test_empty_gt(self):
        det = np.array([[1.0, 2.0]])
        gt = np.empty((0, 2))
        gt_ids = np.array([], dtype=np.int64)
        pos, ids = match_detections_to_gt(det, gt, gt_ids, dist_thr=0.5)
        assert pos.shape == (0, 2)

    def test_many_to_many_optimal(self):
        det = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        gt = np.array([[0.05, 0.0], [1.05, 0.0]])
        gt_ids = np.array([100, 200])
        pos, ids = match_detections_to_gt(det, gt, gt_ids, dist_thr=0.5)
        assert pos.shape[0] == 2
        assert set(ids.tolist()) == {100, 200}

    def test_ids_dtype_int64(self):
        det = np.array([[1.0, 2.0]])
        gt = np.array([[1.0, 2.0]])
        gt_ids = np.array([42])
        _, ids = match_detections_to_gt(det, gt, gt_ids, dist_thr=0.5)
        assert ids.dtype == np.int64
