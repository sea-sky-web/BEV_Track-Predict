## 2024-05-24 - [Avoid Python loops for per-view ops]
**Learning:** In the WildTrack BEV multi-view framework, using Python list comprehensions to map PyTorch operations over the view dimension (e.g., `torch.stack([_photometric(img) for img in imgs])`) is a significant CPU bottleneck.
**Action:** Always vectorize operations across the view dimension. Specifically for per-pixel operations like 1x1 hue convolutions, reshape `(V, C, H, W)` into `(C, V*H*W)` and apply `torch.matmul` with the transformation matrix.

## 2024-05-24 - [Perspective Warp Optimization Pattern]
**Learning:** The default naive geometry inverse-warping using `float64` precision and `nan_to_num` masking is extremely slow. `torch.bmm` is highly performant here, and pushing out-of-bounds coordinates explicitly via `torch.where(invalid, 2.0, x)` natively takes advantage of `grid_sample`'s `padding_mode="zeros"` without explicit masking logic.
**Action:** When implementing spatial mapping logic in PyTorch, use `float32` coordinate grids, `torch.bmm`, and let `grid_sample`'s out-of-bounds padding handle masking.
