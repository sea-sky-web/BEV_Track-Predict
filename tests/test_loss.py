"""Unit tests for loss.py -- WeightedGaussianMSE and GaussianMSE."""
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loss import GaussianMSE, WeightedGaussianMSE, PenaltyReducedFocalLoss, create_loss_criterion
from utils import build_gaussian_kernel_2d


def _kernel(size=7, sigma=1.5):
    return build_gaussian_kernel_2d(size, sigma, torch.device("cpu"))


def test_gaussian_mse_zero_prediction_zero_target():
    loss = GaussianMSE()(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8), _kernel())
    assert loss.item() < 1e-8


def test_gaussian_mse_positive_when_prediction_off():
    loss = GaussianMSE()(torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8), _kernel())
    assert loss.item() > 0.0


def test_gaussian_mse_finite():
    pred = torch.randn(2, 1, 16, 16)
    tgt = torch.zeros(2, 1, 32, 32)
    tgt[:, :, 8, 8] = 1.0
    loss = GaussianMSE()(pred, tgt, _kernel())
    assert torch.isfinite(loss)


def test_weighted_gaussian_mse_higher_with_pos_weight():
    """pos_weight=10 should yield a larger loss than pos_weight=1 when GT has pedestrians."""
    kernel = _kernel()
    tgt = torch.zeros(1, 1, 8, 8)
    tgt[0, 0, 4, 4] = 1.0

    l1 = GaussianMSE()(torch.zeros(1, 1, 8, 8), tgt.clone(), kernel)
    l10 = WeightedGaussianMSE(pos_weight=10.0)(torch.zeros(1, 1, 8, 8), tgt.clone(), kernel)

    ratio = l10.item() / (l1.item() + 1e-12)
    assert ratio > 2.0, f"Expected weighted loss >> unweighted, got ratio={ratio:.2f}"


def test_weighted_gaussian_mse_gradient_amplified():
    """Gradient magnitude at pedestrian location should be larger under pos_weight=10."""
    kernel = _kernel()
    tgt = torch.zeros(1, 1, 8, 8)
    tgt[0, 0, 4, 4] = 1.0

    pred1 = torch.zeros(1, 1, 8, 8, requires_grad=True)
    GaussianMSE()(pred1, tgt.clone(), kernel).backward()
    g1 = pred1.grad.abs().max().item()

    pred10 = torch.zeros(1, 1, 8, 8, requires_grad=True)
    WeightedGaussianMSE(pos_weight=10.0)(pred10, tgt.clone(), kernel).backward()
    g10 = pred10.grad.abs().max().item()

    assert g10 > g1 * 2.0, f"Expected amplified gradient, got g1={g1:.4f} g10={g10:.4f}"


def test_create_loss_criterion_unweighted():
    c = create_loss_criterion(weighted=False)
    assert isinstance(c, GaussianMSE)


def test_create_loss_criterion_weighted():
    c = create_loss_criterion(weighted=True, pos_weight=5.0, neg_weight=0.5)
    assert isinstance(c, WeightedGaussianMSE)
    assert c.pos_weight == 5.0
    assert c.neg_weight == 0.5


def test_weighted_loss_accepts_multichannel():
    kernel = _kernel()
    pred = torch.randn(2, 2, 8, 8)
    tgt = torch.zeros(2, 2, 8, 8)
    tgt[:, :, 3, 3] = 1.0
    loss = WeightedGaussianMSE(pos_weight=10.0)(pred, tgt, kernel)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_focal_loss_finite_and_positive():
    kernel = _kernel()
    pred = torch.randn(2, 1, 16, 16)
    tgt = torch.zeros(2, 1, 32, 32)
    tgt[:, :, 8, 8] = 1.0
    loss = PenaltyReducedFocalLoss()(pred, tgt, kernel)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_focal_loss_decreases_as_prediction_approaches_target():
    kernel = _kernel(size=7, sigma=1.5)
    tgt = torch.zeros(1, 1, 8, 8)
    tgt[0, 0, 4, 4] = 1.0

    far_logits = torch.full((1, 1, 8, 8), -5.0)
    near_logits = torch.full((1, 1, 8, 8), -5.0)
    near_logits[0, 0, 4, 4] = 5.0

    loss_far = PenaltyReducedFocalLoss()(far_logits, tgt.clone(), kernel)
    loss_near = PenaltyReducedFocalLoss()(near_logits, tgt.clone(), kernel)
    assert loss_near.item() < loss_far.item()


def test_create_loss_criterion_focal():
    c = create_loss_criterion(loss_type="focal", focal_alpha=2.0, focal_beta=4.0)
    assert isinstance(c, PenaltyReducedFocalLoss)
    assert c.alpha == 2.0
    assert c.beta == 4.0
