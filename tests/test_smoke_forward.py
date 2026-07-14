from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import create_model


def _make_model(fusion_mode: str, backbone: str = "resnet18"):
    dev = torch.device("cpu")
    proj_mats = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    return create_model(
        num_views=2,
        proj_mats=proj_mats,
        reduced_hw=(8, 8),
        feat_hw=(8, 8),
        device=dev,
        pretrained=False,
        backbone=backbone,
        feat_ch=512,
        add_coord=True,
        fusion_mode=fusion_mode,
    )


@pytest.mark.parametrize("fusion_mode", ["concat", "confidence_v1", "confidence_v2", "geo_confidence_v1"])
def test_resnet18_forward_shapes_for_fusion_modes(fusion_mode):
    model = _make_model(fusion_mode)
    model.eval()
    x = torch.randn(1, 2, 3, 64, 64)

    with torch.no_grad():
        map_logits, offset_preds, imgs_logits = model(x)

    assert map_logits.shape == (1, 1, 8, 8)
    assert offset_preds.shape == (1, 2, 8, 8)
    assert imgs_logits.shape == (1, 2, 2, 8, 8)
    assert torch.isfinite(map_logits).all()
    assert torch.isfinite(offset_preds).all()
    assert torch.isfinite(imgs_logits).all()


@pytest.mark.parametrize("fusion_mode", ["concat", "confidence_v2"])
def test_mobilenet_v2_forward_shapes(fusion_mode):
    model = _make_model(fusion_mode, backbone="mobilenet_v2")
    model.eval()
    x = torch.randn(1, 2, 3, 64, 64)

    with torch.no_grad():
        map_logits, offset_preds, imgs_logits = model(x)

    assert map_logits.shape == (1, 1, 8, 8)
    assert offset_preds.shape == (1, 2, 8, 8)
    assert imgs_logits.shape == (1, 2, 2, 8, 8)
    assert torch.isfinite(map_logits).all()


def test_confidence_v2_backward_reaches_backbone():
    model = _make_model("confidence_v2")
    model.train()
    x = torch.randn(1, 2, 3, 64, 64)

    map_logits, offset_preds, imgs_logits = model(x)
    (map_logits.mean() + offset_preds.mean() + imgs_logits.mean()).backward()

    backbone_grads = [p.grad for p in model.backbone.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in backbone_grads)


def test_geo_confidence_beta_has_gradient():
    model = _make_model("geo_confidence_v1")
    model.train()
    x = torch.randn(1, 2, 3, 64, 64)
    map_logits, _, _ = model(x)
    map_logits.mean().backward()
    assert model.confidence_fusion.beta.grad is not None
