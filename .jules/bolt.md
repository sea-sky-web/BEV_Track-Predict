
## 2024-05-18 - [PyTorch Grid Sampling Optimization for Perspective Warping]
**Learning:** When performing large-scale PyTorch geometry projections (`warp_perspective_torch`), heavy coordinate masking operations (`torch.where`, `torch.nan_to_num`) over massive tensors slow down the calculation due to memory bandwidth constraints.
**Action:** Replace `torch.where` mask-filling with mathematically equivalent out-of-bounds coordinate shifting (`invalid = z <= 1e-6; torch.where(invalid, 2.0, coords)`), allowing `F.grid_sample` to use its highly optimized native C++ zero-padding. Combined with using `torch.bmm` over simple matrix multiplication, it provides significant speed improvements.
