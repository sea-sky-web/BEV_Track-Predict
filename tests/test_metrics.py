import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import aggregate_metrics, compute_moda_modp


def test_moda_modp_perfect_match():
    row = compute_moda_modp(
        pred_pts=[[0.0, 0.0], [2.0, 2.0]],
        gt_pts=[[0.0, 0.0], [2.0, 2.0]],
        d_thresh=0.5,
    )

    assert row["tp"] == 2.0
    assert row["fp"] == 0.0
    assert row["fn"] == 0.0
    assert row["moda"] == 1.0
    assert row["modp"] == 0.0


def test_moda_penalizes_false_positive_and_miss():
    row = compute_moda_modp(
        pred_pts=[[0.0, 0.0], [10.0, 10.0]],
        gt_pts=[[0.0, 0.0], [1.0, 1.0]],
        d_thresh=0.5,
    )

    assert row["tp"] == 1.0
    assert row["fp"] == 1.0
    assert row["fn"] == 1.0
    assert row["moda"] == 0.0


def test_aggregate_metrics_keeps_counts_and_f1():
    rows = [
        compute_moda_modp([[0.0, 0.0]], [[0.0, 0.0]], d_thresh=0.5),
        compute_moda_modp([[10.0, 10.0]], [[0.0, 0.0]], d_thresh=0.5),
    ]
    agg = aggregate_metrics(rows)

    assert agg["tp"] == 1.0
    assert agg["fp"] == 1.0
    assert agg["fn"] == 1.0
    assert agg["n_gt"] == 2.0
    assert math.isclose(agg["precision"], 0.5)
    assert math.isclose(agg["recall"], 0.5)
    assert math.isclose(agg["f1"], 0.5)
