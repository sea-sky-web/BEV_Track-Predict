# [BUG][CRITICAL] Pretrained backbone disabled by default — primary cause of F1≈0.10

## Labels
`bug`, `critical`, `training`, `performance`

## Priority
**P0 — Blocks all meaningful experimentation**

---

## Problem Statement

`src/config.py` sets `pretrained = False` as the default for the ResNet-50 backbone.
All experiment runs to date have therefore trained the backbone **from random initialization**
on only 300 frames of Wildtrack. This is the single largest contributor to the current
F1 ≈ 0.10 performance, which is near-random-guess level.

### Why this is critical

MVDet and MVDeTr — both of which this project's architecture is based on — use
**ImageNet-pretrained** backbone weights as their starting point. The Wildtrack
training set (~1800 frames) is far too small to learn meaningful visual features
from scratch in a ResNet with 25 M parameters. When the backbone outputs random
noise, no amount of BEV projection, fusion tuning, or loss weight search can
produce valid detections.

### Evidence

| Run | pretrained | frames | epochs | F1 |
|-----|-----------|--------|--------|----|
| 20260504_023641 | False | 300 | 10 | ~0.09 |
| 20260511_041025 | False | 300 | 10 | 0.101 |
| MVDet (paper) | **True** | ~1800 | 10 | **MODA 88.2%** |

The metric has not improved across 10+ experiment iterations because the root cause
was never addressed.

---

## Root Cause

```python
# src/config.py  — line ~30
pretrained: bool = False      # ← should be True
```

```python
# src/models.py — ResNet50Stride8Trunk.__init__
resnet = torchvision.models.resnet50(pretrained=pretrained)
# When pretrained=False, all weights are random Xavier init
```

The flag propagates through `create_model()` → `MVDetLikeNet` → `ResNet50Stride8Trunk`
without any override in `train_main.py` or the experiment YAML.

---

## Proposed Fix

### Step 1 — Change the default in `src/config.py`

```python
# Before
pretrained: bool = False

# After
pretrained: bool = True
```

### Step 2 — Update `src/models.py` to use the non-deprecated API (torchvision ≥ 0.13)

```python
# Before
resnet = torchvision.models.resnet50(pretrained=pretrained)

# After
from torchvision.models import resnet50, ResNet50_Weights
weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
resnet = resnet50(weights=weights)
```

### Step 3 — Freeze backbone for first N epochs (optional but recommended)

Training the full network end-to-end from epoch 0 with a small dataset risks
catastrophic forgetting of ImageNet features. Add a `freeze_backbone_epochs`
option to `MVDetTrainer`:

```python
# src/trainer.py — train_epoch() preamble
if epoch < self.cfg.freeze_backbone_epochs:
    for param in self.model.trunk.parameters():
        param.requires_grad = False
else:
    for param in self.model.trunk.parameters():
        param.requires_grad = True
```

Default `freeze_backbone_epochs = 3`.

### Step 4 — Update `configs/exp_colab.yaml` and `src/config.py` docs

```yaml
# configs/exp_colab.yaml
train_cmd: >
  python src/train_main.py
    --data_root ${DATA_ROOT}
    --views 0,1,2,3,4,5,6
    --pretrained true          # ← add this flag
    --fusion_mode confidence
    --alpha 1.0
    --epochs 10
```

---

## Acceptance Criteria

- [ ] `pretrained` defaults to `True` in `src/config.py`
- [ ] `src/models.py` uses the non-deprecated `weights=` API
- [ ] Smoke test (`--max_frames 2 --device cpu`) runs without error
- [ ] A training run with pretrained=True on ≥300 frames achieves **F1 > 0.40** within 10 epochs
- [ ] `docs/experiment_protocol.md` updated to document that pretrained=True is required
- [ ] `AGENTS.md` updated: any new experiment must specify pretrained status explicitly in `ai_context.md`

---

## Expected Impact

Based on the MVDet paper and standard transfer learning practice, enabling pretrained
weights alone — with no other changes — is expected to raise F1 from ~0.10 to **0.50–0.70**
within the same 10-epoch budget on 300 frames.

---

## References

- MVDet paper (ECCV 2020): Table 2, ResNet-18 pretrained on ImageNet
- PyTorch transfer learning guide: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- torchvision weights API migration: https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/
