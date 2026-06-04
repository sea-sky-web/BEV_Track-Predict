
## 2024-05-24 - Vectorizing View Dimension in MVDet Framework
**Learning:** In the multi-view BEV detection architecture, iterating over views with a Python `for`-loop in the `forward` pass (e.g., extracting backbone features and applying perspective warp individually) causes a measurable sequential bottleneck.
**Action:** Always flatten/merge the batch and view dimensions `(B, V, C, H, W) -> (B*V, C, H, W)` before passing tensors into model layers (like the backbone or `warp_perspective_torch`) to unlock full GPU parallelization. Reshape back to `(B, V, ...)` only when view-specific interactions (like fusion) are needed.
