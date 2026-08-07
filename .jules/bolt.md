
## 2024-05-18 - [Optimize PyTorch Grid Sample]
**Learning:** When using PyTorch `grid_sample` for perspective warping, using `float32` coordinate arithmetic and `torch.bmm` is highly performant. Instead of complex mask logic or `clamp`/`nan_to_num` for out-of-bounds coordinates, leverage `grid_sample`'s native padding behavior by pushing invalid coordinates out-of-bounds (e.g., set to 2.0).
**Action:** Always favor `float32` arithmetic, `torch.bmm` for batch matrix mults, and native `padding_mode` when implementing coordinate warping manually.
