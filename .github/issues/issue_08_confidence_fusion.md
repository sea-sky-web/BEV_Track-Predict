# [IMPROVEMENT][MEDIUM] Confidence fusion architecture too shallow to learn meaningful view weights

## Labels
`improvement`, `medium-priority`, `architecture`, `fusion`

## Priority
**P2 — Can only be properly evaluated after issues #1, #2, #3 are fixed**

---

## Problem Statement

The current `SpatialAwareConfidenceFusion` in `src/models.py` is the project's
primary research contribution. Its implementation:

```python
class SpatialAwareConfidenceFusion(nn.Module):
    # Input: (B, V, C, H, W)
    # 2-layer conv: C -> C//4 -> 1  (per-view scalar score map)
    # Softmax across V dim
    # Weighted sum across V dim
    # Output: (B, C, H, W)
```

Issues with this design:

### Issue A: The weight network has no cross-view information

The confidence score for view `v` at location `(x, y)` is computed **solely from
view v's own features** at that location. But the whole purpose of confidence fusion
is to resolve conflicts between views — e.g., "view 3 is occluded here, trust view 1
instead." This requires the weight network to see **at least a summary of all views**
to make a meaningful decision.

The current softmax is computed across views, but the input to each view's score
network sees only that view's features. This means the weights are determined by
absolute feature magnitude, not relative view quality — equivalent to argmax over
feature norms, which is a poor proxy for visibility/occlusion.

### Issue B: Weights collapse to uniform at initialization

With random-initialized weight network (and pretrained=False backbone), the features
from all views are similarly distributed noise. The softmax outputs `1/V` for all
views everywhere. This is identical to average fusion. The network has no gradient
signal to differentiate between views because the BEV loss doesn't directly supervise
the weight maps.

### Issue C: No positional encoding for BEV location

The weight network needs to know "where in BEV space" each location is to reason
about which cameras cover that area. Currently no positional encoding is added to
the BEV features before the weight computation. A location at the BEV boundary
(covered by only 1-2 cameras) gets the same weight treatment as a central location
(covered by 4-5 cameras).

---

## Proposed Improvements (in order of complexity)

### Option A — Concat-then-weight (minimal change, cross-view awareness)

Instead of scoring each view independently, concatenate all view features first,
then predict weights:

```python
class ConcatAttentionFusion(nn.Module):
    """
    Fuses V BEV feature maps by first concatenating them, then predicting
    per-view attention weights from the joint representation.
    """
    def __init__(self, num_views: int, feat_ch: int):
        super().__init__()
        # Joint representation: V*feat_ch -> feat_ch (compress)
        self.joint_compress = nn.Sequential(
            nn.Conv2d(num_views * feat_ch, feat_ch, 1),
            nn.ReLU(inplace=True),
        )
        # Per-view weight: feat_ch -> num_views (one score per view per location)
        self.weight_head = nn.Conv2d(feat_ch, num_views, 1)

    def forward(self, bev_feats: torch.Tensor) -> torch.Tensor:
        # bev_feats: (B, V, C, H, W)
        B, V, C, H, W = bev_feats.shape
        stacked = bev_feats.view(B, V * C, H, W)            # (B, V*C, H, W)
        joint   = self.joint_compress(stacked)               # (B, C, H, W)
        weights = self.weight_head(joint)                    # (B, V, H, W)
        weights = torch.softmax(weights, dim=1).unsqueeze(2) # (B, V, 1, H, W)
        fused   = (bev_feats * weights).sum(dim=1)           # (B, C, H, W)
        return fused
```

This adds cross-view context at the cost of V×C input channels to `joint_compress`.

### Option B — Add positional encoding before weight computation

```python
class PositionalBEVFusion(nn.Module):
    def __init__(self, num_views: int, feat_ch: int, bev_h: int, bev_w: int):
        super().__init__()
        # Learnable 2D positional embedding added to each view's features
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 2, bev_h, bev_w))
        # 2 channels: normalized (x, y) BEV coordinates
        nn.init.normal_(self.pos_embed, std=0.02)
        # ... rest same as Option A
```

### Option C — Visibility mask from geometry (principled)

Compute a binary mask from the homography: locations where the projection falls
outside the image get weight = 0 (not learnable — hard-coded from geometry):

```python
def compute_visibility_mask(proj_mat, feat_hw, bev_hw):
    """
    Returns (H_bev, W_bev) binary tensor: 1 if BEV location maps to valid image region.
    Uses existing compute_valid_ratio_from_homography() logic but per-pixel.
    """
    ...
```

This provides a strong geometric prior that the weight network can then refine.

---

## Implementation Priority

Implement in this order:

1. **Fix issues #1, #2, #3 first** — confidence fusion cannot be fairly evaluated
   without a properly trained backbone and full data.

2. **Establish concat baseline MODA** — this is the ground truth to beat.

3. **Implement Option A** — cross-view aware weight network (3-day effort).

4. **Run A/B: concat vs Option A** — if Option A doesn't beat concat by ≥1 MODA point,
   implement Option C (geometry-guided mask).

---

## Acceptance Criteria

- [ ] `SpatialAwareConfidenceFusion` replaced with `ConcatAttentionFusion` (Option A)
- [ ] Cross-view weight visualization script: saves weight maps per view per frame
- [ ] A/B comparison: concat fusion vs `ConcatAttentionFusion` under identical training config
- [ ] If Option A doesn't improve: implement Option C and re-run A/B
- [ ] `docs/model_definition.md` Section 7 updated to reflect the new fusion rule
- [ ] All fusion modes (`concat`, `confidence_v1`, `confidence_v2`) selectable via `--fusion_mode`

---

## References

- MVDeTr (Hou et al., 2022): Deformable self-attention for view fusion (upper bound)
- "Attention Is All You Need" (Vaswani et al., 2017): scaled dot-product attention
- "DETR3D" (Wang et al., 2022): cross-attention from BEV queries to image features
- Original `SpatialAwareConfidenceFusion`: `src/models.py`
