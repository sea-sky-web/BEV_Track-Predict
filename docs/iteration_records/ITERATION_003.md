# ITERATION_003 Diagnostic Record

## Previous run

`ai_runs/latest_run.txt` points to `20260510_050623`.

This run tested confidence-based multi-view fusion. While valid and clean, it resulted in a significant increase in false positives compared to the concat baseline.

Key metrics (Confidence Fusion):
- Precision: `0.078728`
- F1: `0.100518`
- False positives: `12252`
- Missed detections: `Unavailable` (Need to verify in actual metrics.json)

Comparison with Concat Fusion:
- Concat FP: `7881`
- Confidence FP: `12252`

## Observed problem

Confidence fusion produced nearly 50% more false positives than the concat baseline. Visual inspection or metric analysis suggests that the BEV point extraction (threshold + NMS) is allowing too many duplicate or noisy peaks, especially when confidence fusion produces more "active" heatmaps.

## Candidate causes

1. **Inadequate NMS:** Local-max NMS with a small kernel (3x3) might not be sufficient to suppress multiple peaks belonging to the same object, especially if the predicted heatmaps are slightly spread out or multi-modal.
2. **Duplicate Detections:** Points that are very close to each other (within one match radius) are not being suppressed during extraction, leading to multiple FPs for a single object.

## Selected hypothesis

Implementing an additional greedy distance-based suppression step after thresholding and local-max NMS will reduce false positives by ensuring that no two predicted points are closer than a specified minimum distance (e.g., 3.0 cells, matching the match radius).

## Change boundary

This iteration is limited to:
- `src/evaluate_main.py`: Add `--det_min_distance` and implement greedy distance suppression in `_extract_points`. Save point-extraction settings in the metrics payload for traceability.
- `scripts/commit_ai_runs.py`: Update `ai_context.md` template to display the new point-extraction configuration.
- `docs/experiment_protocol.md`: Document that point-extraction settings are comparison-critical.
- `docs/current_iteration_plan.md`: Update next actions.

No changes to model architecture, training loss, or dataset.

## Expected metric impact

- **Target:** Significant reduction in false positives (`det_fp`).
- **Target:** Improved precision and F1 for confidence fusion.
- **Acceptable Tradeoff:** Small loss in recall is acceptable if F1 improves.

## Validation plan

- All code changes are committed and pushed.
- Validation will be performed in Colab by running evaluation with `--det_min_distance 3.0` on the existing confidence fusion checkpoint.
- The result will be archived via `scripts/commit_ai_runs.py` and compared against the `20260510_050623` baseline.

## Next forbidden directions

- Do not change model architecture yet.
- Do not implement tracking or ReID.
- Do not add trajectory prediction.
- Do not introduce large BEV frameworks.
