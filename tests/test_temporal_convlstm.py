"""Tests for ConvLSTM model, losses, dataset, and trainer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ── ConvLSTM Cell ──


def test_convlstm_cell_forward_shape():
    from temporal.convlstm import ConvLSTMCell

    cell = ConvLSTMCell(in_channels=5, hidden_channels=32, kernel_size=3)
    x = torch.randn(2, 5, 16, 16)
    h, c = cell(x)
    assert h.shape == (2, 32, 16, 16)
    assert c.shape == (2, 32, 16, 16)


def test_convlstm_cell_with_state():
    from temporal.convlstm import ConvLSTMCell

    cell = ConvLSTMCell(in_channels=5, hidden_channels=32, kernel_size=3)
    x = torch.randn(2, 5, 8, 8)
    state = (torch.randn(2, 32, 8, 8), torch.randn(2, 32, 8, 8))
    h, c = cell(x, state)
    assert h.shape == (2, 32, 8, 8)
    assert torch.isfinite(h).all()


# ── SpatioTemporalPredictor ──


def test_predictor_forward_shape():
    from temporal.convlstm import SpatioTemporalPredictor

    model = SpatioTemporalPredictor(
        in_channels=5, hidden_channels=16, kernel_size=3,
        n_encoder_layers=2, n_future=4, out_channels=3,
    )
    x = torch.randn(2, 4, 5, 16, 32)
    out = model(x)
    assert out.shape == (2, 4, 3, 16, 32)


def test_predictor_full_resolution():
    from temporal.convlstm import SpatioTemporalPredictor

    model = SpatioTemporalPredictor(
        in_channels=5, hidden_channels=8, kernel_size=3,
        n_encoder_layers=2, n_future=4, out_channels=3,
    )
    x = torch.randn(1, 4, 5, 120, 360)
    out = model(x)
    assert out.shape == (1, 4, 3, 120, 360)


def test_predictor_backward_finite():
    from temporal.convlstm import SpatioTemporalPredictor

    model = SpatioTemporalPredictor(
        in_channels=5, hidden_channels=8, kernel_size=3,
        n_encoder_layers=2, n_future=4, out_channels=3,
    )
    x = torch.randn(1, 4, 5, 16, 16, requires_grad=False)
    out = model(x)
    loss = out.sum()
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"


# ── Losses ──


def test_occupancy_loss_perfect():
    from temporal.temporal_loss import OccupancyLoss

    loss_fn = OccupancyLoss()
    gt = torch.zeros(2, 16, 16)
    gt[:, 5:10, 5:10] = 1.0
    logits = torch.where(gt > 0.5, torch.tensor(5.0), torch.tensor(-5.0))
    loss = loss_fn(logits, gt)
    assert loss.item() < 0.1


def test_occupancy_loss_with_mask():
    from temporal.temporal_loss import OccupancyLoss

    loss_fn = OccupancyLoss()
    gt = torch.ones(2, 16, 16)
    logits = torch.full((2, 16, 16), -5.0)
    mask = torch.zeros(2, 16, 16)
    loss = loss_fn(logits, gt, valid_mask=mask)
    assert torch.isfinite(loss)


def test_velocity_loss_zero_on_match():
    from temporal.temporal_loss import VelocityLoss

    loss_fn = VelocityLoss()
    vx = torch.randn(2, 16, 16)
    vy = torch.randn(2, 16, 16)
    occ = torch.ones(2, 16, 16)
    loss = loss_fn(vx, vy, vx, vy, occ)
    assert loss.item() < 1e-6


def test_velocity_loss_only_occupied():
    from temporal.temporal_loss import VelocityLoss

    loss_fn = VelocityLoss()
    pred_vx = torch.ones(2, 16, 16)
    pred_vy = torch.ones(2, 16, 16)
    gt_vx = torch.zeros(2, 16, 16)
    gt_vy = torch.zeros(2, 16, 16)
    occ = torch.zeros(2, 16, 16)
    loss = loss_fn(pred_vx, pred_vy, gt_vx, gt_vy, occ)
    assert loss.item() == 0.0


def test_trace_consistency_loss_finite():
    from temporal.temporal_loss import TraceConsistencyLoss

    loss_fn = TraceConsistencyLoss(dt=0.5, cell_m=0.1)
    occ_steps = [torch.rand(2, 16, 16) for _ in range(4)]
    vx_steps = [torch.randn(2, 16, 16) * 0.1 for _ in range(4)]
    vy_steps = [torch.randn(2, 16, 16) * 0.1 for _ in range(4)]
    loss = loss_fn(occ_steps, vx_steps, vy_steps)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_combined_loss_all_ablations():
    from temporal.temporal_loss import CombinedTemporalLoss

    pred = torch.randn(2, 4, 3, 16, 16)
    gt = torch.rand(2, 4, 5, 16, 16)

    for ablation in ["occ_only", "occ_vel", "full"]:
        loss_fn = CombinedTemporalLoss(ablation=ablation)
        result = loss_fn(pred, gt)
        assert "total" in result
        assert torch.isfinite(result["total"])

        if ablation == "occ_only":
            assert result["vel"].item() == 0.0
            assert result["trace"].item() == 0.0


def test_combined_loss_no_nan():
    from temporal.temporal_loss import CombinedTemporalLoss

    loss_fn = CombinedTemporalLoss(ablation="full")
    pred = torch.randn(2, 4, 3, 16, 16)
    gt = torch.zeros(2, 4, 5, 16, 16)
    result = loss_fn(pred, gt)
    for k, v in result.items():
        assert torch.isfinite(v), f"NaN/Inf in {k}"


# ── Dataset ──


def _make_temp_annotations(tmp_path, n_frames=12, n_people=3):
    ann_dir = tmp_path / "annotations_positions"
    ann_dir.mkdir()
    for fi in range(n_frames):
        objects = []
        for pid in range(n_people):
            pos_id = pid * 480 + fi * 2
            objects.append({"personID": pid, "positionID": pos_id})
        with open(ann_dir / f"{fi:08d}.json", "w") as f:
            json.dump(objects, f)
    return ann_dir


def test_field_sequence_dataset_shapes(tmp_path):
    ann_dir = _make_temp_annotations(tmp_path, n_frames=12, n_people=2)

    from temporal.time_utils import SPLIT_RANGES
    import temporal.time_utils as tu
    original = tu.SPLIT_RANGES.copy()
    tu.SPLIT_RANGES["train"] = (0, 12)
    try:
        from temporal.temporal_dataset import FieldSequenceDataset
        ds = FieldSequenceDataset(ann_dir, split="train", history_len=4, future_len=4)
        assert len(ds) == 5  # 12 - 8 + 1
        hist, fut = ds[0]
        assert hist.shape == (4, 5, 120, 360)
        assert fut.shape == (4, 5, 120, 360)
    finally:
        tu.SPLIT_RANGES.update(original)


def test_field_sequence_dataset_values(tmp_path):
    ann_dir = _make_temp_annotations(tmp_path, n_frames=10, n_people=1)

    import temporal.time_utils as tu
    original = tu.SPLIT_RANGES.copy()
    tu.SPLIT_RANGES["train"] = (0, 10)
    try:
        from temporal.temporal_dataset import FieldSequenceDataset
        ds = FieldSequenceDataset(ann_dir, split="train", history_len=3, future_len=3)
        hist, fut = ds[0]
        assert hist[:, 0].max() > 0  # occupancy has some signal
        assert torch.isfinite(hist).all()
        assert torch.isfinite(fut).all()
    finally:
        tu.SPLIT_RANGES.update(original)


# ── Trainer (smoke) ──


def test_trainer_one_step():
    from temporal.convlstm import SpatioTemporalPredictor
    from temporal.temporal_loss import CombinedTemporalLoss
    from temporal.temporal_trainer import TemporalTrainer, set_seed

    set_seed(42)
    model = SpatioTemporalPredictor(
        in_channels=5, hidden_channels=8, kernel_size=3,
        n_encoder_layers=1, n_future=2, out_channels=3,
    )
    criterion = CombinedTemporalLoss(ablation="full")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    history = torch.randn(2, 4, 5, 8, 8)
    future = torch.rand(2, 2, 5, 8, 8)

    model.train()
    optimizer.zero_grad()
    pred = model(history)
    loss_dict = criterion(pred, future)
    loss_dict["total"].backward()
    optimizer.step()

    assert torch.isfinite(loss_dict["total"])
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
