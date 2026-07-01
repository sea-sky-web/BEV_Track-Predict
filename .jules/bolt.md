## 2024-07-01 - Optimizing PyTorch Warp Perspective
**Learning:** Using `float64` coordinates with complex bounds checking (e.g. `clamp` and `nan_to_num`) is slow. Using `float32` coordinates with `torch.bmm` and explicitly mapping invalid coordinates to `2.0` (out-of-bounds for `grid_sample`) provides a significant performance boost.
**Action:** When implementing spatial transformations or grid sampling, prefer `float32` coordinate arithmetic, batched operations like `torch.bmm`, and rely on native `padding_mode` for out-of-bounds rather than manual bounds clamping.
