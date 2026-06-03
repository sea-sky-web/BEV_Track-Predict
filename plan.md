1. Refactor `MVDetLikeNet.forward` in `src/models.py` to remove the `for vi in range(V):` loop. The multi-view dimension should be flattened into the batch dimension before running through the backbone and interpolating, maximizing GPU parallelization.
2. Refactor `train_epoch` and `val_epoch` in `src/trainer.py` to remove `for vi in range(imgs_res.shape[1]):` loop and compute the image loss with vectorized operations.
3. Refactor `evaluate_detection` in `src/evaluate_main.py` to remove `for vi in range(imgs_res.shape[1]):` loop and compute the image loss with vectorized operations.
4. Record critical learnings to `.jules/bolt.md`.
5. Run pre-commit instructions.
6. Commit the changes.
