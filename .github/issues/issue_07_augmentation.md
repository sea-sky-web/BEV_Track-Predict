# [IMPROVEMENT][MEDIUM] Add view-coherent data augmentation — currently zero augmentation applied

## Labels
`improvement`, `medium-priority`, `training`, `augmentation`

## Priority
**P2 — Expected +3–8 MODA points; required before claiming fusion module is the bottleneck**

---

## Problem Statement

`src/dataset.py` applies **no data augmentation** beyond ImageNet normalization.
Every training frame is seen in exactly one orientation, scale, and lighting condition
across all epochs.

MVDet applies at minimum:
- Random horizontal flip (applied coherently across all views)
- Color jitter (per-view independently, since photometric conditions differ)

MVDeTr additionally applies:
- Random crop (with BEV GT crop)
- Random scale

Without augmentation, the model will overfit to specific textures and positions in
the 300-frame (or 1440-frame) training set. This is especially harmful because
Wildtrack has limited pedestrian pose and appearance diversity.

---

## Design Constraint: View-Coherent Augmentation

The critical requirement is that **geometric augmentations must be applied identically
across all V views for a given frame**. Each view is a different camera angle of the
same scene. If view 0 is flipped horizontally but view 3 is not, the BEV projection
becomes geometrically inconsistent and the fusion module receives contradictory features.

Photometric augmentations (brightness, contrast, saturation) can safely be applied
**independently per view**, as each camera has its own exposure and white balance.

---

## Proposed Implementation

### Step 1 — Add `src/augmentation.py`

```python
"""View-coherent augmentation for multi-view pedestrian detection."""
import random
import torch
import torchvision.transforms.functional as TF
from typing import List, Optional


class ViewCoherentAugment:
    """
    Applies the same geometric transform to all V views of a frame,
    and independent photometric transforms per view.

    Args:
        hflip_prob: probability of horizontal flip (default 0.5)
        color_jitter: (brightness, contrast, saturation, hue) magnitude
        enabled: set False during validation/evaluation
    """
    def __init__(
        self,
        hflip_prob:    float = 0.5,
        color_jitter:  tuple = (0.3, 0.3, 0.3, 0.05),
        enabled:       bool  = True,
    ):
        self.hflip_prob   = hflip_prob
        self.color_jitter = color_jitter
        self.enabled      = enabled

    def __call__(
        self,
        imgs:   List[torch.Tensor],   # V × (3, H, W), float, normalized
        map_gt: torch.Tensor,         # (1, H_bev, W_bev) BEV ground truth
        aux_gt: torch.Tensor,         # (V, 2, H_feat, W_feat) per-view aux GT
    ):
        if not self.enabled:
            return imgs, map_gt, aux_gt

        # --- Geometric: same decision for all views ---
        do_hflip = random.random() < self.hflip_prob

        aug_imgs = []
        for v, img in enumerate(imgs):
            img = self._photometric(img)
            if do_hflip:
                img = TF.hflip(img)
            aug_imgs.append(img)

        if do_hflip:
            map_gt = TF.hflip(map_gt)
            aux_gt = TF.hflip(aux_gt)

        return aug_imgs, map_gt, aux_gt

    def _photometric(self, img: torch.Tensor) -> torch.Tensor:
        """Independent per-view color jitter (applied before normalization ideally,
        but can be applied post-normalization as an approximation)."""
        b, c, s, h = self.color_jitter
        img = TF.adjust_brightness(img, 1 + random.uniform(-b, b))
        img = TF.adjust_contrast(img,   1 + random.uniform(-c, c))
        img = TF.adjust_saturation(img, 1 + random.uniform(-s, s))
        img = TF.adjust_hue(img,            random.uniform(-h, h))
        return img.clamp(0, 1)   # clamp after un-normalizing if needed
```

### Step 2 — Integrate into `src/dataset.py`

```python
# WildtrackMVDetDataset.__init__
self.augment = ViewCoherentAugment(
    hflip_prob=cfg.aug_hflip_prob,
    color_jitter=cfg.aug_color_jitter,
    enabled=(split == "train"),
)

# WildtrackMVDetDataset.__getitem__
imgs, map_gt, aux_gt = self.augment(imgs, map_gt, aux_gt)
```

### Step 3 — Add augmentation config fields

```python
# src/config.py
aug_hflip_prob:   float = 0.5
aug_color_jitter: tuple = (0.3, 0.3, 0.3, 0.05)
aug_enabled:      bool  = True
```

### Step 4 — Verify BEV GT is correctly flipped

After horizontal flip, a pedestrian at BEV grid `(ix, iy)` should appear at
`(NB_WIDTH - 1 - ix, iy)`. The flip along the width axis of the BEV map
must be verified with a visualization test before enabling in training.

---

## Augmentation Ablation Required

Before claiming the augmentation improves results, run a controlled A/B:

```
Run A: pretrained=True, all frames, all views, NO augmentation (baseline)
Run B: pretrained=True, all frames, all views, hflip only
Run C: pretrained=True, all frames, all views, hflip + color_jitter
```

All other hyperparameters identical. Record MODA/MODP for each.

---

## What NOT to Add

The following augmentations are **explicitly excluded** because they break geometric
consistency or are too expensive for this stage:

- Random crop (requires recomputing projection matrices for cropped image)
- Random rotation (non-planar BEV warp)
- Mixup / CutMix
- Any augmentation applied differently to different views

---

## Acceptance Criteria

- [ ] `src/augmentation.py` implemented with `ViewCoherentAugment`
- [ ] Augmentation is disabled during validation/evaluation (checked via `split` flag)
- [ ] Unit test: verify that hflip applied to (imgs, map_gt) produces consistent geometry
- [ ] A/B ablation (3 runs) recorded in `ai_runs/` showing augmentation effect on MODA
- [ ] `docs/experiment_protocol.md` updated: augmentation config must be stated in `ai_context.md`

---

## References

- MVDet source: horizontal flip applied coherently across views during training
- "AutoAugment for Object Detection" for augmentation magnitude choices
- Wildtrack dataset: limited appearance diversity, augmentation especially beneficial
