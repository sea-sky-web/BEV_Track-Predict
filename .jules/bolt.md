
## 2024-05-18 - [Vectorize View Dimensions in MVDetLikeNet]
**Learning:** A key architectural pattern for maximizing GPU parallelization in this multi-view framework is vectorizing operations across the view dimension.
**Action:** Achieve this by flattening and merging the batch and view dimensions (e.g., `.reshape(B * V, ...)`) before passing tensors into model layers, rather than using Python for-loops.
