# Experiment Iteration Protocol

## 1. Purpose

This document defines how every formal experiment and AI-assisted model change must be recorded.

The goal is to ensure that model development follows a continuous research loop:

```text
previous result
→ observed problem
→ improvement hypothesis
→ minimal change
→ training/evaluation
→ metrics comparison
→ next action
```

This project must not evolve through isolated code edits.

Each iteration must preserve enough context for the next AI assistant or future researcher to understand:

```text
what was changed
why it was changed
what result it produced
what should be done next
what should not be done next
```

---

## 2. Formal Experiment Directory

Every formal experiment must be recorded under:

```text
ai_runs/YYYYMMDD_HHMMSS/
```

Required files:

```text
ai_context.md
metrics.json
train_tail.log
error.log
```

Optional files:

```text
config_snapshot.yaml
eval_summary.md
changed_files.txt
notes.md
```

The timestamped experiment directory is the authoritative record.

`latest` or `latest_run.txt` may only point to the most recent timestamped run.

Previous timestamped experiment directories must not be deleted, overwritten, or rewritten.

---

## 3. Required `ai_context.md` Structure

Every `ai_context.md` must follow this exact section structure:

```markdown
# AI Iteration Context

## 1. Iteration ID

## 2. Previous Iteration

## 3. Previous Metrics Summary

## 4. Observed Problem

## 5. Improvement Hypothesis

## 6. Changes Made

## 7. Training Configuration

## 8. Evaluation Configuration

## 9. Current Metrics

## 10. Result Interpretation

## 11. Next Iteration Recommendation

## 12. Do Not Do Next
```

Do not remove sections.

If a section is not applicable, write:

```text
N/A
```

---

## 4. Section Requirements

### 4.1 Iteration ID

Record the current timestamped experiment ID:

```text
YYYYMMDD_HHMMSS
```

Example:

```text
20260504_034152
```

---

### 4.2 Previous Iteration

Record the previous experiment directory.

Example:

```text
ai_runs/20260504_034152/
```

If this is the first formal iteration, write:

```text
No previous formal iteration.
```

---

### 4.3 Previous Metrics Summary

Summarize the previous experiment metrics.

Required fields:

```text
Precision:
Recall:
F1:
Localization error:
False positives:
Missed detections:
Main failure:
```

If a metric is unavailable, write:

```text
Unavailable
```

---

### 4.4 Observed Problem

Describe the exact problem found from the previous result.

Valid examples:

```text
Recall is low.
False positives are high.
Localization error is large.
Multi-view fusion does not outperform the naive baseline.
Evaluation metrics are incomplete.
Training runs but outputs are not archived.
```

Invalid examples:

```text
The model should be better.
The architecture is not advanced enough.
We need a stronger model.
```

The observed problem must be based on logs, metrics, or a concrete implementation issue.

---

### 4.5 Improvement Hypothesis

The improvement hypothesis must use this format:

```text
Because [observed problem],
we change [specific module/config],
expecting [target metric or behavior] to improve because [reason].
```

Example:

```text
Because naive multi-view fusion treats all projected views equally,
we change the fusion module from concatenation to spatial-aware confidence fusion,
expecting F1 and localization error to improve because unreliable views can receive lower BEV-space weights.
```

The hypothesis must be specific enough to validate.

Invalid hypotheses:

```text
Try a better model.
Improve the network.
Optimize training.
```

---

### 4.6 Changes Made

List changed files and describe each change briefly.

Format:

```text
Changed files:
- src/models.py: implemented spatial-aware confidence fusion.
- src/train_main.py: added config flag for fusion mode.
- docs/model_definition.md: no change.
```

The change list must distinguish between:

```text
model change
training config change
evaluation change
logging change
documentation change
bug fix
```

---

### 4.7 Training Configuration

Record the training configuration.

Required fields:

```text
dataset:
views:
epochs:
batch_size:
learning_rate:
max_frames:
device:
seed:
checkpoint_path:
fusion_mode:
```

If a training run was not performed, write:

```text
Training not performed in this iteration.
```

---

### 4.8 Evaluation Configuration

Record the evaluation configuration.

Required fields:

```text
model_path:
views:
threshold:
distance_threshold:
metrics_output:
device:
max_frames:
```

If evaluation was not performed, write:

```text
Evaluation not performed in this iteration.
```

---

### 4.9 Current Metrics

Record the current metrics.

Required fields:

```text
Precision:
Recall:
F1:
Localization error:
False positives:
Missed detections:
```

If evaluation failed, write:

```text
Evaluation failed.
```

Then summarize the failure and ensure the full error is stored in `error.log`.

---

### 4.10 Result Interpretation

Answer directly:

```text
Did the change improve the target metric?
Yes / No / Inconclusive
```

Then explain using metrics.

Valid example:

```text
Inconclusive. Evaluation ran successfully, but the comparison baseline used a different view configuration, so the F1 scores are not directly comparable.
```

Invalid example:

```text
The change should help.
```

---

### 4.11 Next Iteration Recommendation

Only one primary next action is allowed.

Format:

```text
Next action:
Reason:
Expected validation:
```

Example:

```text
Next action:
Run the same evaluation for naive fusion and confidence fusion under identical views and threshold settings.

Reason:
The current result cannot prove whether the fusion module improves performance.

Expected validation:
A table comparing Precision, Recall, F1, and localization error under the same evaluation configuration.
```

Do not list multiple unrelated next actions.

---

### 4.12 Do Not Do Next

List directions that should not be pursued in the next iteration.

This section is mandatory because it prevents scope drift.

Examples:

```text
Do not add trajectory prediction yet.
Do not introduce BEVFormer or other large BEV frameworks.
Do not tune hyperparameters before validating fusion behavior.
Do not modify the dataset format.
```

---

## 5. `metrics.json` Requirement

The file `metrics.json` must be machine-readable.

Recommended structure:

```json
{
  "iteration_id": "YYYYMMDD_HHMMSS",
  "previous_iteration": "YYYYMMDD_HHMMSS",
  "dataset": "WildTrack",
  "views": "0,1,2",
  "fusion_mode": "confidence",
  "precision": null,
  "recall": null,
  "f1": null,
  "localization_error": null,
  "false_positives": null,
  "missed_detections": null,
  "checkpoint_path": null,
  "metrics_output": null,
  "status": "success | failed | incomplete"
}
```

Fields may be extended, but the required metric fields must remain.

---

## 6. `train_tail.log` Requirement

The file `train_tail.log` should contain the final part of the training log.

It should include:

```text
training command
last visible training losses
checkpoint save message
training completion or failure message
```

Do not store only a success sentence.

---

## 7. `error.log` Requirement

The file `error.log` must exist for every formal experiment.

If no error occurred, it should contain:

```text
No error.
```

If an error occurred, it should contain:

```text
command
full error message
failing file or module if known
short diagnosis
```

Do not suppress errors.

---

## 8. Iteration Decision Rules

A new model change is allowed only if it is linked to a previous result.

Allowed reasons:

```text
metric is weak
evaluation is incomplete
training is unstable
logging is incomplete
implementation does not match docs/model_definition.md
smoke test fails
```

Forbidden reasons:

```text
the model feels too simple
a more advanced architecture exists
adding a module sounds interesting
the assistant suggests a larger framework
```

---

## 9. Improvement Claim Rule

AI assistants must not claim improvement unless metrics support it.

Acceptable statements:

```text
This change implements the defined fusion module.
The evaluation ran successfully.
The result is inconclusive because the baseline comparison is missing.
F1 improved from 0.42 to 0.48 under the same evaluation configuration.
```

Unacceptable statements without evidence:

```text
This improves the model.
This makes the model more robust.
This is better than the baseline.
```

---

## 10. Comparison Rule

Metric comparison is valid only when the following are consistent:

```text
dataset
views
max_frames
checkpoint selection rule
evaluation threshold
distance threshold
metric implementation
```

If these are not consistent, the result must be marked:

```text
Inconclusive
```

---

## 11. Minimal Next-Step Rule

Each iteration must end with exactly one primary next action.

This rule prevents scattered development.

The next action must be one of:

```text
run a specific evaluation
fix a specific failure
implement a specific defined module
compare two specific settings
update a specific documentation gap
```

The next action must not be vague.

Invalid:

```text
Improve the model further.
Try more experiments.
Optimize the architecture.
```

Valid:

```text
Compare naive fusion and confidence fusion on views 0,1,2 using the same checkpoint and threshold sweep.
```

---

## 12. Scope Drift Control

Each iteration must explicitly state what not to do next.

Default forbidden directions for the current stage:

```text
Do not implement tracking.
Do not implement ReID.
Do not implement trajectory prediction.
Do not implement occupancy-flow forecasting.
Do not introduce BEVFormer, PETR, LSS, DETR3D, or other large BEV frameworks.
Do not change the dataset away from WildTrack.
```

These can only change if `docs/model_definition.md` is explicitly updated.

---

## 13. Summary

The purpose of this protocol is to make every AI-assisted change traceable, measurable, and connected to the previous result.

The project must progress through controlled research iterations:

```text
previous result
→ observed problem
→ hypothesis
→ minimal change
→ validation
→ interpretation
→ next action
```

If an iteration cannot be connected to this chain, it should not be performed.
