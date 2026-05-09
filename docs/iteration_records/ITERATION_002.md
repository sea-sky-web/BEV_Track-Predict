# ITERATION_002 Diagnostic Record

## Previous run

`ai_runs/latest_run.txt` points to `20260509_073835`.

The run completed successfully with `success=true`, `return_code=0`, and an empty `error.log` in the archived record.

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
- `ai_runs/20260509_073835/ai_context.md`
- `ai_runs/20260509_073835/error.log`
- `ai_runs/20260509_073835/train_tail.log`
- `src/models.py`
- `src/train_main.py`
- `src/evaluate_main.py`
- `scripts/run_colab_exp.py`
- `scripts/commit_ai_runs.py`

The training and evaluation chain is complete enough to proceed with analysis, but the archived context is not complete enough for safe model comparison.

## Model understanding

The current model takes synchronized WildTrack multi-view images as `(B, V, 3, H, W)`.

Each view is passed through a shared ResNet50 stride-8 trunk and interpolated to the configured feature plane. The per-view feature map is projected to the BEV grid with the geometry homography through `warp_perspective_torch`.

The current fusion in `src/models.py` concatenates projected BEV features from all views along the channel dimension, optionally appends coordinate channels, and sends the result to `BEVHeadDilated`. This is an implicit concatenation baseline, not an explicit recorded `fusion_mode` comparison.

Evaluation uses sigmoid BEV heatmaps, threshold sweep, local max extraction, greedy BEV point matching, and reports precision, recall, F1, false positives, missed detections, and localization error.

## Observed problem

The latest run has very low F1 with both high false positives and high missed detections. Localization error for matched detections is relatively small, so the most visible failure is not only point localization. The run also lacks comparison-critical metadata such as explicit `fusion_mode`, `views`, `max_frames`, and checkpoint path at the top level of `metrics.json`, and `ai_context.md` does not follow `docs/experiment_iteration_protocol.md`.

Because the evaluation record is not fully comparable, a model architecture change would be difficult to attribute in the next Colab run.

## Candidate causes

1. Incomplete experiment comparability metadata.
   - Supporting evidence: `metrics.json` has detection metrics but does not top-level record `fusion_mode`, `views`, `max_frames`, or `checkpoint_path`; `ai_context.md` is a generic instruction instead of the required iteration protocol.
   - Uncertainty: the training log contains some configuration lines, so a human can still infer part of the setup.
   - Minimal validation action: add structured run metadata and formal `ai_context.md`, then run the next Colab loop and verify the archived fields.

2. Fusion implementation may be only an implicit concat baseline.
   - Supporting evidence: `src/models.py` concatenates projected per-view BEV features before the BEV head; `docs/model_definition.md` requires naive vs spatial-aware confidence fusion comparison.
   - Uncertainty: the current BEV head can still learn some cross-view interaction after concatenation.
   - Minimal validation action: only after metadata is complete, compare explicit baseline and confidence fusion under the same settings.

3. Heatmap confidence or thresholding may be poorly calibrated.
   - Supporting evidence: best threshold is `0.1`, precision is low, recall is also low, and training logs show strongly negative raw prediction ranges after early epochs.
   - Uncertainty: threshold sweep already exists, and poor recall at the best threshold may reflect heatmap quality rather than just threshold selection.
   - Minimal validation action: archive the full threshold sweep and configuration consistently before changing thresholds or losses.

4. View projection coverage is imbalanced.
   - Supporting evidence: train log reports valid ratios `0.1195`, `0.8914`, and `0.3579` for views 0, 1, and 2.
   - Uncertainty: low valid ratio views may still contribute useful information, and dropping views would change the comparison setting.
   - Minimal validation action: after metadata is complete, run a controlled same-protocol view or fusion ablation.

## Selected hypothesis

Because the latest run is valid but not sufficiently self-describing for controlled comparison, we change the experiment logging and archive context, expecting the next iteration to be safely comparable because `metrics.json` and `ai_context.md` will record the dataset, views, max_frames, fusion_mode, checkpoint path, and detection metrics together.

## Rejected options

- Do not implement confidence fusion in this PR, because the current run record cannot yet support a clean before/after attribution.
- Do not tune thresholds or loss weights in this PR, because that would mix evaluation/protocol changes with model behavior.
- Do not drop low-coverage views in this PR, because changing selected views would make the next comparison inconclusive.

## Change boundary

This iteration is limited to experiment traceability:

- `scripts/run_colab_exp.py` may add comparison-critical metadata to `metrics.json`.
- `scripts/commit_ai_runs.py` may write a structured `ai_context.md` and normalize empty error logs to `No error.`.
- No model, dataset, geometry, loss, or evaluation metric definition is changed.

## Input output acceptance criteria

Input:

- A completed Colab run with `runs/<exp_name>/metrics.json`, `train.log`, and `error.log`.

Output:

- `ai_runs/YYYYMMDD_HHMMSS/metrics.json` contains detection metrics plus dataset, views, max_frames, fusion_mode, and checkpoint path.
- `ai_runs/YYYYMMDD_HHMMSS/ai_context.md` follows the required twelve-section iteration structure.
- `ai_runs/YYYYMMDD_HHMMSS/error.log` contains `No error.` when the run has no error.

Acceptance criteria:

- `python -m compileall src scripts` passes.
- The next Colab `python scripts/run_colab_exp.py` and `python scripts/commit_ai_runs.py` run still use the same training entrypoint.
- The next archived run can be compared without reading free-form logs to recover core configuration.

## Expected metric impact

No direct model metric improvement is claimed in this iteration. The expected impact is improved experiment comparability and reduced risk of making an untraceable model change.

## Validation command

```bash
python -m compileall src scripts
```

Final validation remains the Colab loop:

```bash
python scripts/run_colab_exp.py
python scripts/commit_ai_runs.py
```

## Rollback plan

Revert this PR. The rollback only removes structured logging changes and the iteration record; it does not affect model weights, dataset files, previous `ai_runs`, or training/evaluation algorithms.

## Next forbidden directions

- Do not implement tracking.
- Do not implement ReID.
- Do not implement trajectory prediction.
- Do not introduce BEVFormer, PETR, LSS, DETR3D, or other large BEV frameworks.
- Do not claim model improvement until a same-configuration metrics comparison exists.
