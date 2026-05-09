# AI Training Context

Please follow this order:

1. Read `ai_runs/latest_run.txt` to get the latest timestamp.
2. Read `ai_runs/{timestamp}/metrics.json`.
3. Read `ai_runs/{timestamp}/error.log` and `ai_runs/{timestamp}/train_tail.log`.

Your task:
- If training failed, fix the error according to `error.log`.
- If training succeeded but metrics are poor, make a small, testable optimization.
- Keep the training entry command unchanged:
  `python scripts/run_colab_exp.py`
- Do not commit `runs/`, `wildtrack/`, model weights, or datasets.
- Keep changes minimal and explain why they help.
