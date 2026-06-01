## 2024-06-01 - Batch processing over Views dimension
**Learning:** In PyTorch, looping over dimensions like views using python for-loops introduces significant kernel launch overhead and prevents the GPU from parallelizing effectively. In `MVDetLikeNet.forward`, the views were processed sequentially in a python for loop.
**Action:** Always look for opportunities to merge dimensions (like Batch and View dimensions: `(B, V, C, H, W) -> (B * V, C, H, W)`) to vectorize operations and reduce Python loop overhead in model forward passes.
