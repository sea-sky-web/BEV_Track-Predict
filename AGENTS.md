# AGENTS.md

## 1. Project Identity

This repository is a constrained deep learning research project.

The current project goal is to build a geometry-guided MVDet-style multi-view BEV pedestrian detector for the WildTrack dataset.

The model definition is fixed in:

```text
docs/model_definition.md
```

AI assistants must treat `docs/model_definition.md` as the highest model-level constraint.

This repository is not a general software engineering project. Code changes must serve measurable deep learning model progress.

---

## 2. Required Reading Order

Before making any code, config, documentation, or experiment change, AI assistants must read:

```text
1. AGENTS.md
2. docs/model_definition.md
3. docs/experiment_iteration_protocol.md
4. latest available experiment context under ai_runs/
5. README.md
6. relevant source files
```

If `ai_runs/latest_run.txt` exists, read the timestamped run it points to.

If `ai_runs/latest/` exists, it may be used as a convenience reference, but timestamped experiment directories are authoritative.

If no previous experiment record exists, the assistant must state that the next run is the first formal iteration.

---

## 3. Current Model Boundary

The current model is strictly defined as:

```text
WildTrack synchronized multi-view images
→ shared image encoder
→ geometry-based BEV projection
→ spatial-aware multi-view confidence fusion
→ BEV decoder
→ BEV pedestrian occupancy heatmap
→ BEV pedestrian point extraction
```

The current stage includes only:

```text
BEV pedestrian detection
BEV heatmap prediction
BEV point extraction
BEV detection evaluation
experiment logging
```

The current stage excludes:

```text
tracking
ReID
trajectory prediction
occupancy-flow prediction
crowd forecasting
traffic operation decision-making
synthetic data training
real station deployment
large autonomous-driving BEV frameworks
```

Excluded items must not be implemented unless `docs/model_definition.md` is explicitly updated.

---

## 4. Implementation Boundary

AI assistants may modify implementation only when the change directly supports the current model boundary.

Allowed areas:

```text
src/dataset.py
src/geometry.py
src/models.py
src/loss.py
src/trainer.py
src/train_main.py
src/evaluate_main.py
src/utils.py
configs/
scripts/
docs/
```

Allowed change types:

```text
bug fix
data loading correction
projection correction
defined fusion module implementation
BEV heatmap prediction improvement
BEV point extraction improvement
evaluation metric improvement
experiment logging improvement
documentation update
minimal smoke test
```

Forbidden change types:

```text
replacing the defined architecture
adding unrelated model modules
changing the dataset away from WildTrack
removing experiment records
hiding errors
claiming improvement without metrics
large refactors without direct need
large dependencies without justification
```

---

## 5. Experiment Iteration Rule

Every meaningful model, training, evaluation, or logging change must be treated as one research iteration.

Each iteration must be based on the previous iteration result.

AI assistants must not propose or implement a new model change without answering:

```text
What was the previous result?
What problem was observed?
What is the current improvement hypothesis?
Which module or config will be changed?
Which metric should improve?
How will the change be validated?
```

The iteration logic must follow:

```text
previous metrics
→ observed problem
→ improvement hypothesis
→ minimal code/config change
→ training/evaluation
→ metrics comparison
→ next action
```

A change is not considered a research improvement unless evaluation metrics support it.

Training loss alone is not sufficient evidence.

The detailed iteration format is defined in:

```text
docs/experiment_iteration_protocol.md
```

---

## 6. Experiment Record Rule

Formal experiments must be recorded under:

```text
ai_runs/YYYYMMDD_HHMMSS/
```

Each formal experiment must include:

```text
ai_context.md
metrics.json
train_tail.log
error.log
```

The timestamped directory is the authoritative experiment record.

`latest` or `latest_run.txt` may only point to the most recent run.

AI assistants must not delete, overwrite, or rewrite previous timestamped experiment directories.

The file `ai_context.md` must follow the format defined in:

```text
docs/experiment_iteration_protocol.md
```

---

## 7. Validation Rule

Every code change must provide at least one minimal validation command.

A validation command should test the smallest meaningful path.

Examples:

```bash
python src/train_main.py --data_root wildtrack --views 0,1,2 --device cpu --max_frames 2
```

```bash
python src/evaluate_main.py --data_root wildtrack --views 0,1,2 --model_path outputs/model_final.pth --report_detection
```

If a command was not actually run, the assistant must say so.

AI assistants must not imply successful execution without evidence.

---

## 8. Failure Handling Rule

If training, evaluation, or smoke testing fails, AI assistants must:

```text
1. preserve the error message
2. identify the failing file or module
3. explain the likely cause
4. propose the minimal fix
5. avoid unrelated refactoring
```

A failure must be recorded in `error.log` for formal experiment runs.

---

## 9. Documentation Rule

Documentation must be updated when changing:

```text
model structure
input format
output format
dataset assumptions
training command
evaluation command
metrics definition
experiment logging format
```

Documentation must remain consistent with:

```text
docs/model_definition.md
```

---

## 10. Priority Order

When making decisions, AI assistants must follow this priority order:

```text
1. preserve docs/model_definition.md
2. preserve WildTrack compatibility
3. preserve runnable training and evaluation
4. preserve timestamped experiment records
5. make the smallest effective change
6. validate with metrics
7. document the change
```

---

## 11. Summary

The repository must evolve through measurable research iterations, not isolated code edits.

Each iteration must connect:

```text
previous result
→ current hypothesis
→ implementation change
→ evaluation metrics
→ next action
```

The current goal is to build a stable, measurable, and reproducible WildTrack multi-view BEV pedestrian detector with spatial-aware multi-view confidence fusion.
