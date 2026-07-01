from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from augmentation import ViewCoherentAugment, parse_color_jitter


def test_parse_color_jitter_requires_four_nonnegative_values():
    assert parse_color_jitter("0.1,0.2,0.3,0.0") == (0.1, 0.2, 0.3, 0.0)
    with pytest.raises(ValueError):
        parse_color_jitter("0.1,0.2,0.3")
    with pytest.raises(ValueError):
        parse_color_jitter("0.1,-0.2,0.3,0.0")
    with pytest.raises(ValueError):
        parse_color_jitter("0.1,0.2,0.3,0.6")


def test_view_coherent_hflip_flips_images_bev_and_aux_labels():
    aug = ViewCoherentAugment(hflip_prob=1.0, color_jitter=(0.0, 0.0, 0.0, 0.0))
    imgs = torch.arange(2 * 3 * 2 * 3, dtype=torch.float32).reshape(2, 3, 2, 3) / 100.0
    map_gt = torch.arange(1 * 2 * 3, dtype=torch.float32).reshape(1, 2, 3)
    aux_gt = torch.arange(2 * 2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 2, 3)

    out_imgs, out_map, out_aux = aug(imgs, map_gt, aux_gt)

    assert torch.allclose(out_imgs, torch.flip(imgs, dims=(-1,)), atol=1e-5)
    assert torch.allclose(out_map, torch.flip(map_gt, dims=(-1,)), atol=1e-5)
    assert torch.allclose(out_aux, torch.flip(aux_gt, dims=(-1,)), atol=1e-5)


def test_view_coherent_augment_disabled_returns_inputs():
    aug = ViewCoherentAugment(enabled=False)
    imgs = torch.zeros(2, 3, 2, 2)
    map_gt = torch.zeros(1, 2, 2)
    aux_gt = torch.zeros(2, 2, 2, 2)
    out_imgs, out_map, out_aux = aug(imgs, map_gt, aux_gt)
    assert out_imgs is imgs
    assert out_map is map_gt
    assert out_aux is aux_gt
