
## 2024-05-18 - [Optimization] PyTorch warp_perspective_torch vectorization
**Learning:** For PyTorch grid sampling and perspective warping, using `float32` coordinate arithmetic with `torch.bmm` is highly performant. Out-of-bounds handling should leverage `grid_sample`'s native `padding_mode` by pushing invalid coordinates out-of-bounds (e.g., `invalid = z <= 1e-6; coords = torch.where(invalid, 2.0, coords)`) rather than using complex arithmetic penalties or slow mask-filling operations like `torch.nan_to_num`.
**Action:** Always prefer vectorized `torch.bmm` with native type precision (`float32`) instead of fallback types (`float64`) and leverage built-in behavior (like out-of-bounds mapping for `grid_sample`) instead of manual filtering.
