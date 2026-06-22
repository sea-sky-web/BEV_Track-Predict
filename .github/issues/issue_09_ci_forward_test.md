# [IMPROVEMENT][LOW] CI smoke test only checks imports — extend to run a true end-to-end forward pass

## Labels
`improvement`, `low-priority`, `ci`, `testing`

## Priority
**P3 — Improves confidence in changes; prevents regressions**

---

## Problem Statement

`.github/workflows/python-smoke.yml` currently only validates that modules import
without error. It does not run any model forward pass, loss computation, or
evaluation step. This means:

- A broken `forward()` method in `models.py` would pass CI
- A projection matrix dimension bug would pass CI
- A dataset index-out-of-bounds would pass CI

All of these bugs have historically appeared during Colab runs (costing 15–30 min
per failed run), when they could have been caught in 60 seconds by CI.

---

## Current CI (`.github/workflows/python-smoke.yml`)

```yaml
steps:
  - run: pip install -r requirements.txt
  - run: python scripts/verify_modules.py
  # No actual model run
```

`scripts/verify_modules.py` checks:
- `from src.models import MVDetLikeNet` succeeds
- `from src.dataset import WildtrackMVDetDataset` succeeds
- etc.

---

## Proposed Extension

### Add a synthetic smoke test that requires no real data

```python
# tests/test_smoke_forward.py
"""End-to-end forward pass using synthetic (random) data — no Wildtrack files needed."""
import torch
import pytest
from src.models import create_model
from src.config import Cfg
from src.loss import GaussianMSE

def make_synthetic_batch(cfg, num_views=3, batch=1):
    """Create random inputs matching the model's expected shapes."""
    imgs    = torch.randn(batch, num_views, 3, *cfg.img_hw)
    map_gt  = torch.zeros(batch, 1, *cfg.bev_hw_full)
    map_gt[0, 0, 100, 50] = 1.0   # one synthetic pedestrian
    proj_mats = torch.eye(3).unsqueeze(0).repeat(num_views, 1, 1)   # identity (no warp)
    return imgs, map_gt, proj_mats

def test_forward_pass_runs():
    cfg = Cfg()
    cfg.num_views = 3
    imgs, map_gt, proj_mats = make_synthetic_batch(cfg)
    model = create_model(cfg, proj_mats=proj_mats, num_views=3, pretrained=False)
    model.eval()
    with torch.no_grad():
        bev_pred, aux_pred = model(imgs)
    assert bev_pred.shape == (1, 1, cfg.bev_hw[0], cfg.bev_hw[1]), \
        f"Unexpected BEV output shape: {bev_pred.shape}"
    assert not bev_pred.isnan().any(), "NaN in BEV prediction"

def test_loss_backward_runs():
    cfg = Cfg()
    imgs, map_gt, proj_mats = make_synthetic_batch(cfg)
    model = create_model(cfg, proj_mats=proj_mats, num_views=3, pretrained=False)
    loss_fn = GaussianMSE(cfg)
    bev_pred, _ = model(imgs)
    loss = loss_fn(bev_pred, map_gt)
    loss.backward()
    # Check that backbone received gradients
    for name, p in model.named_parameters():
        if "trunk" in name and p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            break

def test_concat_and_confidence_fusion():
    cfg = Cfg()
    imgs, map_gt, proj_mats = make_synthetic_batch(cfg)
    for mode in ["concat", "confidence"]:
        cfg.fusion_mode = mode
        model = create_model(cfg, proj_mats=proj_mats, num_views=3, pretrained=False)
        model.eval()
        with torch.no_grad():
            out, _ = model(imgs)
        assert out.shape[1] == 1, f"fusion_mode={mode}: wrong output channels"
```

### Update CI workflow

```yaml
# .github/workflows/python-smoke.yml
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: {python-version: "3.10"}
      - run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -r requirements.txt
      - run: python scripts/verify_modules.py
      - run: pytest tests/test_smoke_forward.py -v --tb=short
      - run: pytest tests/test_geometry.py -v --tb=short    # from issue #4
```

### Expected CI runtime

| Step | Time |
|------|------|
| pip install | ~90 s |
| verify_modules | ~5 s |
| test_smoke_forward | ~30 s (CPU, no real data) |
| test_geometry | ~10 s |
| **Total** | **~3 min** |

---

## Acceptance Criteria

- [ ] `tests/test_smoke_forward.py` with 3+ tests implemented
- [ ] All tests pass on CPU with `pretrained=False` (no internet required in CI)
- [ ] CI workflow updated to run `pytest tests/` after verify_modules
- [ ] CI passes on `main` branch after all preceding issues are merged
- [ ] README updated: "Run `pytest tests/` to verify installation"

---

## Non-goals

- Do not add tests that require Wildtrack data (not available in CI)
- Do not add benchmark/performance tests (flaky in shared runners)
- Do not mock the dataset — synthetic random tensors are sufficient for shape/gradient checks
