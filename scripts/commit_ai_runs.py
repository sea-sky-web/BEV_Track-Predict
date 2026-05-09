from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "exp_colab.yaml"


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def tail_lines(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:]) + ("\n" if lines else "")


def _metric(metrics: dict[str, Any], key: str, default: str = "Unavailable") -> Any:
    actual = metrics.get("actual_metrics")
    if isinstance(actual, dict) and key in actual:
        return actual[key]
    return metrics.get(key, default)


def _fmt(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_ai_context(metrics: dict[str, Any], cfg: dict[str, Any], timestamp: str) -> str:
    actual = metrics.get("actual_metrics") if isinstance(metrics.get("actual_metrics"), dict) else {}
    exp_cfg = metrics.get("experiment_config") if isinstance(metrics.get("experiment_config"), dict) else {}
    success = bool(metrics.get("success"))
    status = "success" if success else "failed"
    views = exp_cfg.get("views", metrics.get("views", "Unavailable"))
    max_frames = exp_cfg.get("max_frames", metrics.get("max_frames", "Unavailable"))
    fusion_mode = exp_cfg.get("fusion_mode", metrics.get("fusion_mode", "concat"))
    checkpoint_path = exp_cfg.get("checkpoint_path", metrics.get("checkpoint_path", "Unavailable"))
    train_command = exp_cfg.get("train_command", cfg.get("train_command", []))
    train_command_text = " ".join(str(x) for x in train_command) if isinstance(train_command, list) else str(train_command)

    precision = _metric(metrics, "det_precision")
    recall = _metric(metrics, "det_recall")
    f1 = _metric(metrics, "det_f1")
    loc_err = _metric(metrics, "det_loc_err_m")
    fp = _metric(metrics, "det_fp")
    fn = _metric(metrics, "det_fn")

    if success and actual:
        observed_problem = (
            "Training and evaluation completed, but the detection result is weak: "
            f"F1={_fmt(f1)}, precision={_fmt(precision)}, recall={_fmt(recall)}, "
            f"false positives={_fmt(fp)}, missed detections={_fmt(fn)}."
        )
        interpretation = "Inconclusive. This run is a baseline measurement until a same-configuration comparison exists."
    elif success:
        observed_problem = "Training completed, but evaluation metrics are incomplete or unavailable."
        interpretation = "Inconclusive. Required detection metrics are missing."
    else:
        observed_problem = "Training or evaluation failed. See error.log."
        interpretation = "No. The run failed and must be fixed before model optimization."

    return f"""# AI Iteration Context

## 1. Iteration ID

{timestamp}

## 2. Previous Iteration

Previous formal run should be read from `ai_runs/latest_run.txt` before this run was created. If unavailable, treat this as the current baseline record.

## 3. Previous Metrics Summary

Precision: {_fmt(precision)}
Recall: {_fmt(recall)}
F1: {_fmt(f1)}
Localization error: {_fmt(loc_err)}
False positives: {_fmt(fp)}
Missed detections: {_fmt(fn)}
Main failure: {observed_problem}

## 4. Observed Problem

{observed_problem}

## 5. Improvement Hypothesis

Because the current run needs a controlled comparison before any model-level claim,
we preserve the training entrypoint and record the comparison-critical settings,
expecting the next iteration to compare metrics under the same dataset, views, max_frames, checkpoint rule, threshold sweep, and fusion_mode.

## 6. Changes Made

Changed files:
- scripts/run_colab_exp.py: records dataset, views, max_frames, fusion_mode, checkpoint_path, and train_command in metrics.json.
- scripts/commit_ai_runs.py: writes this structured ai_context.md format for archived runs.
- docs/iteration_records/ITERATION_002.md: records the diagnostic decision and change boundary.

## 7. Training Configuration

dataset: WildTrack
views: {_fmt(views)}
epochs: {_fmt(exp_cfg.get('epochs', 'Unavailable'))}
batch_size: {_fmt(exp_cfg.get('batch_size', 'Unavailable'))}
learning_rate: Unavailable
max_frames: {_fmt(max_frames)}
device: Unavailable
seed: Unavailable
checkpoint_path: {_fmt(checkpoint_path)}
fusion_mode: {_fmt(fusion_mode)}
train_command: {train_command_text}

## 8. Evaluation Configuration

model_path: {_fmt(checkpoint_path)}
views: {_fmt(views)}
threshold: {_fmt(_metric(metrics, 'det_best_threshold'))}
distance_threshold: Unavailable
metrics_output: metrics.json
nodevice: Unavailable
max_frames: {_fmt(max_frames)}

## 9. Current Metrics

Precision: {_fmt(precision)}
Recall: {_fmt(recall)}
F1: {_fmt(f1)}
Localization error: {_fmt(loc_err)}
False positives: {_fmt(fp)}
Missed detections: {_fmt(fn)}
Status: {status}

## 10. Result Interpretation

{interpretation}

## 11. Next Iteration Recommendation

Next action:
Run the next Colab training/evaluation after this logging change and verify that metrics.json contains dataset, views, max_frames, fusion_mode, checkpoint_path, and detection metrics.

Reason:
Without these fields, future model changes cannot be compared safely under the experiment protocol.

Expected validation:
A new ai_runs timestamp whose metrics.json has both detection metrics and comparison-critical configuration fields.

## 12. Do Not Do Next

Do not implement tracking.
Do not implement ReID.
Do not implement trajectory prediction.
Do not introduce BEVFormer, PETR, LSS, DETR3D, or other large BEV frameworks.
Do not claim model improvement until metrics are compared under the same configuration.
"""


def main() -> int:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}
    exp_name = str(cfg.get("exp_name", "colab_exp"))
    output_dir = Path(str(cfg.get("output_dir", ROOT / "runs" / exp_name)))
    branch = os.getenv("GITHUB_BRANCH", str(cfg.get("git", {}).get("branch", "main")))
    log_tail_n = int(cfg.get("log_tail_lines", 500))

    metrics_path = output_dir / "metrics.json"
    train_log = output_dir / "train.log"
    error_log = output_dir / "error.log"

    metrics: dict[str, Any] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            timestamp = str(metrics.get("timestamp") or timestamp)
        except Exception:
            metrics = {}

    run_dir = ROOT / "ai_runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if metrics_path.exists():
        (run_dir / "metrics.json").write_text(metrics_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    else:
        (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")

    if error_log.exists():
        error_text = error_log.read_text(encoding="utf-8", errors="ignore")
        if error_text.strip() == "":
            error_text = "No error.\n"
        (run_dir / "error.log").write_text(error_text, encoding="utf-8")
    else:
        (run_dir / "error.log").write_text("No error.\n", encoding="utf-8")

    (run_dir / "train_tail.log").write_text(tail_lines(train_log, log_tail_n), encoding="utf-8")
    (run_dir / "ai_context.md").write_text(build_ai_context(metrics, cfg, timestamp), encoding="utf-8")
    (ROOT / "ai_runs" / "latest_run.txt").write_text(f"{timestamp}\n", encoding="utf-8")

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not found, ai_runs prepared locally, skip git operations.")
        return 0

    user = os.getenv("GITHUB_USER")
    repo = os.getenv("GITHUB_REPO")
    if not user or not repo or not branch:
        print("GITHUB_USER, GITHUB_REPO, or GITHUB_BRANCH missing, skip git operations.")
        return 0

    set_email = run_git(["config", "user.email", "colab-runner@example.com"])
    if set_email.returncode != 0:
        print(f"set git user.email failed: {set_email.stderr.strip()}")
        return 1

    set_name = run_git(["config", "user.name", "colab-runner"])
    if set_name.returncode != 0:
        print(f"set git user.name failed: {set_name.stderr.strip()}")
        return 1

    remote_url = f"https://x-access-token:{token}@github.com/{user}/{repo}.git"
    set_url = run_git(["remote", "set-url", "origin", remote_url])
    if set_url.returncode != 0:
        print(f"set remote url failed: {set_url.stderr.strip()}")
        return 1

    add_res = run_git(["add", "ai_runs/"])
    if add_res.returncode != 0:
        print(f"git add ai_runs failed: {add_res.stderr.strip()}")
        return 1

    diff_res = run_git(["diff", "--cached", "--quiet"])
    if diff_res.returncode == 0:
        print("No ai_runs changes to commit.")
        return 0
    if diff_res.returncode != 1:
        print(f"git diff --cached --quiet failed: {diff_res.stderr.strip()}")
        return 1

    commit_msg = f"add training result: {exp_name} {timestamp}"
    commit_res = run_git(["commit", "-m", commit_msg])
    if commit_res.returncode != 0:
        print(commit_res.stdout)
        print(commit_res.stderr)
        return 1

    push_res = run_git(["push", "origin", branch])
    if push_res.returncode != 0:
        stderr = (push_res.stderr or "").lower()
        if "non-fast-forward" in stderr or "fetch first" in stderr or "rejected" in stderr:
            print(
                "git push failed because remote has new commits. "
                "Please pull/rebase manually and retry; this script will not auto-rebase."
            )
        else:
            print(f"git push failed on branch {branch}:\n{push_res.stdout}\n{push_res.stderr}")
        return 1

    print("ai_runs committed and pushed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
