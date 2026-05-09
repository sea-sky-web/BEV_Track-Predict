# ITERATION_002 Diagnostic Record

## Previous run

`ai_runs/latest_run.txt` points to `20260509_073835`.

The run completed successfully with `success=true`, `return_code=0`, and an empty archived `error.log`.

Key metrics:

- Precision: `0.1017489612522949`
- Recall: `0.13978494623655913`
- F1: `0.11777206129068336`
- False positives: `9296`
- Missed detections: `6480`
- Localization error: `0.1940380442516077 m`
- Best detection threshold: `0.1`
- Loss: `0.012742100482185681`

## Input evidence

Read evidence:

- `AGENTS.md`
- `docs/model_definition.md`
- `docs/experiment_protocol.md`
- `docs/experiment_iteration_protocol.md`
- `docs/current_iteration_plan.md`
- `ai_runs/latest_run.txt`
- `ai_runs/20260509_073835/metrics.json`
- `ai_runs/20260509_073835/error.log`
- `ai_runs/20260509_073835/train_tail.log`
- `src/train_main.py`
- `scripts/run_colab_exp.py`
- `scripts/commit_ai_runs.py`
- Review feedback on closed PR #21

The train/eval chain is complete enough to diagnose, but the archive format still needs correction before model optimization.

## Model understanding

The current training command runs `scripts/train_main.py`. That script writes the final checkpoint to `DEFAULT_OUTPUT_DIR/model_final.pth`, where `DEFAULT_OUTPUT_DIR` is `outputs/train_multicam_mvdet_style_v3` in `src/config.py`.

`run_colab_exp.py` writes launcher logs and metrics under the configured run output directory. This run output directory is not necessarily the model checkpoint directory.

`commit_ai_runs.py` archives a completed run into `ai_runs/YYYYMMDD_HHMMSS/`, writes `ai_context.md`, then updates `ai_runs/latest_run.txt`.

## Observed problem

Closed PR #21 correctly chose experiment traceability as the next smallest action, but its implementation had two flaws:

1. It could record `checkpoint_path` as `output_dir/model_final.pth`, even though training actually saves the checkpoint under `DEFAULT_OUTPUT_DIR`.
2. It filled `Previous Metrics Summary` from the current run metrics instead of reading the previous timestamped run before updating `latest_run.txt`.

Both flaws directly affect experiment comparability, so they must be fixed before any model-level change.

## Candidate causes

1. The run output directory and model checkpoint directory are different concepts.
   - Supporting evidence: `configs/exp_colab.yaml` uses `runs/wildtrack_baseline` for launcher metrics/logs, while `src/train_main.py` saves to `outputs/train_multicam_mvdet_style_v3/model_final.pth`.
   - Uncertainty: future training scripts may add an explicit `--output_dir` argument.
   - Minimal validation action: resolve checkpoint path from the training log `[OK] saved ...model_final.pth`, then fall back to explicit config or the known default.

2. Previous iteration must be captured before latest is overwritten.
   - Supporting evidence: `commit_ai_runs.py` writes the new `latest_run.txt` after archiving, so the previous value is available before that write.
   - Uncertainty: first formal runs may have no previous timestamp.
   - Minimal validation action: read old `ai_runs/latest_run.txt` first; if it differs from the current timestamp, load that run's metrics for the Previous Metrics Summary.

## Selected hypothesis

Because the previous logging PR would record misleading checkpoint and previous-run metadata, we make a narrower logging correction, expecting the next archived run to preserve an accurate checkpoint path and a clear previous/current metric separation.

## Rejected options

- Do not implement confidence fusion in this PR; this is still a logging correctness fix.
- Do not change evaluation thresholds or detection matching.
- Do not change train/eval entry commands.
- Do not rewrite existing timestamped `ai_runs` history.

## Change boundary

This iteration is limited to:

- `scripts/run_colab_exp.py`: record actual checkpoint path from config, train log, explicit output arg, or known default.
- `scripts/commit_ai_runs.py`: read previous latest run before writing the new latest pointer and use it only for previous metrics.
- `docs/iteration_records/ITERATION_002.md`: document this corrected diagnosis.

No model, dataset, geometry, loss, or metric definition is changed.

## Input output acceptance criteria

Input:

- A completed Colab run with `runs/<exp_name>/metrics.json`, `train.log`, and `error.log`.
- Existing `ai_runs/latest_run.txt` before the new archive is written.

Output:

- New `metrics.json` contains a checkpoint path that points to the path saved by the training log when available.
- New `ai_context.md` has Previous Metrics Summary from the previous timestamped ai_runs directory, not from the current run.
- Current Metrics remains populated from the current run.

Acceptance criteria:

- `python3 -m compileall src scripts` passes.
- The next Colab run still uses `python scripts/run_colab_exp.py` and `python scripts/commit_ai_runs.py`.
- PR only changes logging/traceability files.

## Expected metric impact

No direct model metric improvement is claimed. The expected impact is correct experiment attribution for the next model or fusion comparison.

## Validation command

```bash
python3 -m compileall src scripts
```

Final Colab validation:

```bash
python scripts/run_colab_exp.py
python scripts/commit_ai_runs.py
```

## Rollback plan

Revert this PR. It only removes logging/context changes and the iteration record; it does not touch model weights, dataset files, previous `ai_runs`, or training/evaluation algorithms.

## Next forbidden directions

- Do not implement tracking.
- Do not implement ReID.
- Do not implement trajectory prediction.
- Do not introduce BEVFormer, PETR, LSS, DETR3D, or other large BEV frameworks.
- Do not claim model improvement until a same-configuration metrics comparison exists.
