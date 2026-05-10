# Experiment Protocol

## 1. Purpose

This document defines required training/evaluation metrics and command-level reporting for the current BEV pedestrian detection stage.

It complements `docs/experiment_iteration_protocol.md` (iteration traceability) and `docs/model_definition.md` (model boundary).

---

## 2. Scope

This protocol applies to:

```text
src/train_main.py
src/evaluate_main.py
ai_runs/YYYYMMDD_HHMMSS/
```

The current scope is limited to WildTrack BEV pedestrian detection.

---

## 3. Required Evaluation Metrics

Each formal evaluation must report:

```text
Precision
Recall
F1
Localization error
False positives
Missed detections
```

Metric comparisons are valid only when dataset, views, thresholds, distance threshold, point-extraction settings (NMS size, suppression distance), and loss configuration (type, weights) are held consistent.

---

## 4. Minimal Commands

Recommended smoke test:

```bash
python src/train_main.py --data_root wildtrack --views 0,1,2 --device cpu --max_frames 2
```

Recommended evaluation:

```bash
python src/evaluate_main.py \
  --data_root wildtrack \
  --views 0,1,2 \
  --model_path outputs/model_final.pth \
  --report_detection \
  --det_min_distance 3.0 \
  --metrics_out outputs/eval_metrics.json
```

---

## 5. Recording Rules

For each formal experiment directory under `ai_runs/YYYYMMDD_HHMMSS/`, include:

```text
ai_context.md
metrics.json
train_tail.log
error.log
```

If evaluation fails, `error.log` must preserve the command and full traceback.

---

## 6. Interpretation Rules

Do not claim improvement without direct metric comparison under identical configuration.

Allowed result labels:

```text
Improved
Not improved
Inconclusive
```

If any required comparison condition is mismatched, the result must be marked `Inconclusive`.

---

## 7. Forbidden Scope Drift

Do not introduce tracking, ReID, trajectory prediction, or large BEV framework replacements in this stage.

All protocol updates must remain consistent with `docs/model_definition.md`.
