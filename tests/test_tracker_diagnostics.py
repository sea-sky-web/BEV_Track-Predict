"""Tests for tracker_diagnostics module."""

import numpy as np
import pytest

from temporal.tracker_diagnostics import diagnose_tracker, IDSwitchEvent, FalsePositiveEvent


def _frame(positions, ids):
    return {
        "positions": np.array(positions, dtype=np.float64).reshape(-1, 2),
        "ids": np.array(ids, dtype=np.int64),
    }


def _empty_frame():
    return {"positions": np.empty((0, 2)), "ids": np.array([], dtype=np.int64)}


class TestDiagnoseTracker:

    def test_perfect_tracking_no_events(self):
        gt = [_frame([[0, 0], [1, 1]], [1, 2])] * 3
        pred = [_frame([[0, 0], [1, 1]], [10, 20])] * 3
        diag = diagnose_tracker(gt, pred, dist_thr=0.5)
        assert diag.total_idsw == 0
        assert diag.total_fp == 0

    def test_id_switch_detected(self):
        gt = [
            _frame([[0, 0], [1, 0]], [1, 2]),
            _frame([[0.1, 0], [0.9, 0]], [1, 2]),
        ]
        pred = [
            _frame([[0, 0], [1, 0]], [10, 20]),
            _frame([[0.1, 0], [0.9, 0]], [20, 10]),
        ]
        diag = diagnose_tracker(gt, pred, dist_thr=0.5)
        assert diag.total_idsw == 2
        assert len(diag.id_switch_events) == 2

    def test_false_positive_detected(self):
        gt = [_frame([[0, 0]], [1])]
        pred = [_frame([[0, 0], [5, 5]], [10, 20])]
        diag = diagnose_tracker(gt, pred, dist_thr=0.5)
        assert diag.total_fp == 1
        assert diag.fp_events[0].track_id == 20
        assert diag.fp_events[0].nearest_gt_distance > 0.5

    def test_empty_frames(self):
        gt = [_empty_frame(), _frame([[0, 0]], [1])]
        pred = [_empty_frame(), _frame([[0, 0]], [10])]
        diag = diagnose_tracker(gt, pred, dist_thr=0.5)
        assert diag.total_idsw == 0
        assert diag.total_fp == 0

    def test_frame_offset(self):
        gt = [_frame([[0, 0]], [1]), _frame([[0, 0]], [1])]
        pred = [_frame([[0, 0]], [10]), _frame([[0, 0]], [20])]
        diag = diagnose_tracker(gt, pred, dist_thr=0.5, frame_offset=320)
        assert diag.total_idsw == 1
        assert diag.id_switch_events[0].frame_index == 321

    def test_fp_near_vs_far_classification(self):
        gt = [_frame([[0, 0]], [1])]
        pred = [_frame([[0, 0], [0.7, 0], [5, 5]], [10, 20, 30])]
        diag = diagnose_tracker(gt, pred, dist_thr=0.5)
        assert diag.total_fp == 2
        near = [e for e in diag.fp_events if e.nearest_gt_distance < 1.0]
        far = [e for e in diag.fp_events if e.nearest_gt_distance >= 1.0]
        assert len(near) == 1
        assert len(far) == 1

    def test_idsw_by_gt_id_aggregation(self):
        gt = [
            _frame([[0, 0]], [1]),
            _frame([[0.1, 0]], [1]),
            _frame([[0.2, 0]], [1]),
        ]
        pred = [
            _frame([[0, 0]], [10]),
            _frame([[0.1, 0]], [20]),
            _frame([[0.2, 0]], [30]),
        ]
        diag = diagnose_tracker(gt, pred, dist_thr=0.5)
        assert diag.total_idsw == 2
        assert diag.idsw_by_gt_id[1] == 2

    def test_summary_contains_key_info(self):
        gt = [_frame([[0, 0]], [1]), _frame([[0, 0]], [1])]
        pred = [_frame([[0, 0]], [10]), _frame([[0, 0], [3, 3]], [20, 30])]
        diag = diagnose_tracker(gt, pred, dist_thr=0.5)
        assert "Total IDSW:" in diag.summary
        assert "Total FP:" in diag.summary
