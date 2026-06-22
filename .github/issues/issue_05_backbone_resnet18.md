# [IMPROVEMENT][HIGH] Replace ResNet-50 backbone with ResNet-18 to reduce overfitting and training cost

## Labels
`improvement`, `high-priority`, `architecture`, `backbone`

## Priority
**P1 — Reduces overfitting risk on small dataset; cuts training time by ~4×**

---

## Problem Statement

The current backbone is **ResNet-50** with `replace_stride_with_dilation=[False, True, True]`,
reduced to 512 channels via a 1×1 conv. This gives:

- ~25 M backbone parameters
- Feature maps: `(B, 512, H/8, W/8)`
- Training cost: ~4× ResNet-18 per view

MVDet and MVDeTr both use **ResNet-18** for multi-view detection on Wildtrack.
The choice is intentional:

1. **Shared backbone processes V views sequentially.** With V=7, ResNet-50 is run
   7 times per frame. Each forward pass on a 270×480 feature map is ~4× slower
   than ResNet-18.

2. **Small dataset → high overfitting risk.** 1,440 training frames is insufficient
   to fully fine-tune 25 M backbone parameters. ResNet-18's 11 M parameters are
   already larger than what the dataset can regularize without aggressive dropout/augmentation.

3. **BEV fusion is the research contribution, not backbone capacity.** Improving
   the confidence fusion module does not require a stronger backbone — it requires
   a stable, well-trained one. ResNet-50's extra capacity is wasted on this task.

### Performance comparison (literature)

| Backbone | Params | Wildtrack MODA | Training time / epoch |
|----------|--------|----------------|----------------------|
| ResNet-18 (MVDet) | 11 M | 88.2% | ~5 min |
| ResNet-50 (ours) | 25 M | ~10% (F1) | ~20 min (est.) |
| ResNet-18 + Transformer (MVDeTr) | ~20 M total | 93.2% | ~15 min |

---

## Proposed Change

### `src/models.py` — Replace `ResNet50Stride8Trunk` with `ResNet18Stride8Trunk`

```python
# Before
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Stride8Trunk(nn.Module):
    def __init__(self, out_ch: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(weights=weights)
        # replace_stride_with_dilation for layer3, layer4
        # output: 2048 channels -> 1x1 conv -> out_ch

# After
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Stride8Trunk(nn.Module):
    """
    ResNet-18 backbone with stride-8 output via dilated conv in layer3/layer4.
    Output: (B, out_ch, H/8, W/8)
    Matches the backbone architecture used in MVDet (ECCV 2020).
    """
    def __init__(self, out_ch: int = 512, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)

        # Apply dilated convolutions to layer3 and layer4 for stride-8 output
        resnet.layer3[0].conv1.stride = (1, 1)
        resnet.layer3[0].downsample[0].stride = (1, 1)
        for m in resnet.layer3.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation = (2, 2)
                m.padding = (2, 2)

        resnet.layer4[0].conv1.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        for m in resnet.layer4.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation = (4, 4)
                m.padding = (4, 4)

        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        # ResNet-18 layer4 output: 512 channels (same as our target!)
        # No 1x1 reduction needed if out_ch=512
        self.proj = nn.Conv2d(512, out_ch, 1) if out_ch != 512 else nn.Identity()

    def forward(self, x):
        return self.proj(self.features(x))
```

### `src/config.py` — Update backbone name

```python
backbone: str = "resnet18"   # was "resnet50"
feat_ch: int  = 512          # unchanged — ResNet-18 layer4 also outputs 512
```

### Backward compatibility

Keep `ResNet50Stride8Trunk` in `models.py` but mark it `# legacy — use ResNet18Stride8Trunk`.
Update `create_model()` to dispatch on `cfg.backbone`.

---

## Migration of existing checkpoints

Existing checkpoints trained with ResNet-50 are **not compatible** with ResNet-18.
Since all current checkpoints have F1 ≈ 0.10 (effectively untrained), no migration
is needed — all runs should restart with the new backbone.

---

## Acceptance Criteria

- [ ] `ResNet18Stride8Trunk` implemented in `src/models.py`
- [ ] Stride-8 output verified: input `(1, 3, 720, 1280)` → output `(1, 512, 90, 160)` ✓
- [ ] `cfg.backbone = "resnet18"` is the new default
- [ ] `create_model()` dispatches correctly on `cfg.backbone`
- [ ] Smoke test passes with new backbone on CPU
- [ ] Training run with ResNet-18 + pretrained=True achieves lower wall-clock time per epoch
   than ResNet-50 (measure and record)
- [ ] `docs/model_definition.md` updated to specify backbone as ResNet-18

---

## References

- MVDet source (official): uses `resnet18(pretrained=True)` with dilated layer3/layer4
- "Deep Residual Learning for Image Recognition" (He et al., 2016)
- Dilated convolution for stride reduction: "Multi-Scale Context Aggregation by Dilated Convolutions" (Yu & Koltun, 2016)
