# AI Training Context

This directory contains the latest Colab training result.

Please read the following files first:

1. ai_runs/latest/metrics.json
2. ai_runs/latest/error.log
3. ai_runs/latest/train_tail.log

Your task:
- If the previous training failed, fix the code according to error.log.
- If the training succeeded but metrics are poor, make a small and testable improvement.
- Keep the Colab entry command unchanged:
  python scripts/run_colab_exp.py
- Do not delete ai_runs.
- Do not commit model checkpoints or large datasets.
- Keep changes minimal and explain why they help.
