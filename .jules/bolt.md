## 2025-06-05 - Vectorize Multiview Framework and Optimize Grid Sampler

**Learning:**
The MVDet-like architecture has a critical bottleneck in mapping multi-view features to BEV format when Python for-loops are used. Since all views undergo the exact same backbone and perspective warping operations, iterating over views linearly scales the time linearly. However, merging the Batch and View dimension (so the shape becomes `[B * V, ...]`) lets PyTorch utilize optimized batched linear algebra and C++ loops under the hood, significantly reducing python overhead and execution time.
Additionally, using `torch.bmm` for perspective matrix inverse projection improves speed significantly. Furthermore, out-of-bounds filling can natively happen at `grid_sample` just by letting the normalized grid be outside `[-1, 1]` (e.g. `2.0`), saving multiple heavy mask-filling calls like `torch.where` and `torch.nan_to_num`.

**Action:**
Always vectorize over parallel dimensions (like `num_views` or `time`) before feeding inputs through identical layers instead of looping in python. For grid sampling, use native `padding_mode` for out-of-bounds fallback rather than doing mask assignments manually via python/torch logic.
