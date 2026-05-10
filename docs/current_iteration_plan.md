# Current Iteration Plan

## 1. Purpose

This file defines the immediate development plan for the next few iterations.

It is intentionally short.

It is not a full research roadmap.

The goal is to keep AI-assisted development focused on the current BEV pedestrian detection stage.

---

## 2. Current State

The repository is being converted into an AI-assisted deep learning research workflow.

The current model definition is:

```text
WildTrack multi-view images
→ geometry-guided BEV projection
→ spatial-aware multi-view confidence fusion
→ BEV heatmap
→ BEV pedestrian point detections
```

The current priority is not to add new model ideas.

The current priority is to align project files, verify the baseline, and then compare the defined fusion module.

---

## 3. Stage 0: Repository Control File Alignment

### Objective

Make sure AI control files exist in the expected locations and reference each other correctly.

### Required Actions

```text
1. Move or place model definition at docs/model_definition.md.
2. Keep AGENTS.md at the repository root.
3. Place experiment iteration protocol at docs/experiment_iteration_protocol.md.
4. Place dataset contract at docs/dataset_contract.md.
5. Place experiment protocol at docs/experiment_protocol.md.
6. Ensure AGENTS.md references only existing files.
7. Fix Markdown formatting if any file was written as a single long line.
```

### Success Criteria

```text
AGENTS.md can be read as the root AI constraint.
docs/model_definition.md exists.
docs/dataset_contract.md exists.
docs/experiment_protocol.md exists.
docs/experiment_iteration_protocol.md exists.
All referenced files exist.
```

### Do Not Do In This Stage

```text
Do not change model architecture.
Do not start trajectory prediction.
Do not add new datasets.
Do not introduce large BEV frameworks.
```

---

## 4. Stage 1: Baseline Verification

### Objective

Verify that the current baseline pipeline can run and produce detection metrics.

### Required Actions

```text
1. Run a minimal smoke test.
2. Run a small training or load an existing checkpoint.
3. Run BEV detection evaluation.
4. Save metrics to JSON.
5. Create the first fully compliant ai_context.md if not already available.
```

### Expected Commands

Smoke test pattern:

```bash
python src/train_main.py --data_root wildtrack --views 0,1,2 --device cpu --max_frames 2
```

Evaluation pattern:

```bash
python src/evaluate_main.py \
  --data_root wildtrack \
  --views 0,1,2 \
  --model_path outputs/model_final.pth \
  --report_detection \
  --metrics_out outputs/eval_metrics.json
```

The exact command may be adjusted to match the current implementation.

### Success Criteria

```text
training or smoke test runs without immediate crash
evaluation produces Precision, Recall, F1, and localization error
metrics are archived under ai_runs/YYYYMMDD_HHMMSS/
ai_context.md follows docs/experiment_iteration_protocol.md
```

### Do Not Do In This Stage

```text
Do not optimize architecture before baseline metrics are known.
Do not tune many hyperparameters.
Do not claim improvement without a baseline comparison.
```

---

## 5. Stage 2: Fusion Module Verification

### Objective

Verify the defined model improvement:

```text
spatial-aware multi-view confidence fusion
```

### Required Actions

```text
1. Identify the current fusion implementation in src/models.py.
2. Ensure naive fusion is available as a baseline.
3. Ensure confidence fusion is available as the defined improvement.
4. Run both settings under the same evaluation configuration.
5. Compare metrics.
```

### Required Comparison

```text
naive fusion vs confidence fusion
```

Comparison metrics:

```text
Precision
Recall
F1
Localization error
false positives
missed detections
```

### Success Criteria

```text
both fusion modes run under the same selected views
both fusion modes produce metrics
comparison is recorded in ai_context.md
result is marked as improved, not improved, or inconclusive
```

### Do Not Do In This Stage

```text
Do not add calibration perturbation.
Do not add tracking.
Do not add prediction.
Do not introduce BEVFormer, PETR, LSS, or DETR3D.
```

---

## 6. Stage 3: Next Decision

Stage 3 is not fixed yet.

It must be selected based on Stage 2 metrics.

Allowed next decisions:

```text
fix fusion implementation if metrics fail
improve evaluation if metrics are incomplete
adjust training protocol if training is unstable
run view ablation if fusion comparison is valid
```

Forbidden next decisions:

```text
jump directly to trajectory prediction
replace the model with a large BEV framework
add unrelated modules because they seem advanced
```

---

## 7. Current Next Action

The immediate next action is:

```text
Evaluate the confidence fusion checkpoint with greedy distance suppression.
```

Reason:

```text
Confidence fusion showed high false positives (12252 vs 7881 concat). Reducing FPs via --det_min_distance 3.0 is needed to recover precision and F1 before further model optimization.
```

Expected validation:

```text
A new ai_runs timestamp in Colab with --det_min_distance 3.0 showing reduced det_fp and improved det_f1.
```

---

## 8. Summary

The current short-term plan is:

```text
Stage 0: align project control files (Completed)
Stage 1: verify baseline train/eval pipeline (Completed)
Stage 2: compare naive fusion with confidence fusion (Completed - Result: high FPs)
Stage 3: improve evaluation via distance-based suppression (In Progress)
Stage 4: decide based on metrics
```

This plan prevents scope drift and keeps the project focused on measurable BEV pedestrian detection progress.
