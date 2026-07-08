
## 2026-07-08 - Batched Execution in MVDetLikeNet
**Learning:** In PyTorch, processing multiple views sequentially with a for-loop inside `MVDetLikeNet` incurs unnecessary overhead. Batching across the view dimension (`B*V`) for backbone processing and perspective warping is mathematically identical (diff is exactly 0) and improves CPU execution time by ~11% (from 55s to 49s) due to more efficient BLAS operation scheduling and avoiding Python loop overhead. This aligns with memory directives indicating that flattening batch and view dimensions is a key architectural pattern for multi-view frameworks.
**Action:** Replace the sequential view processing in `MVDetLikeNet.forward()` with batched operations over the flattened `B*V` dimension.
