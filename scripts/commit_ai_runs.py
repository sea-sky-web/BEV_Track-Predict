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
LATEST_RUN_PATH = ROOT / "ai_runs" / "latest_run.txt"


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def tail_lines(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:]) + ("\n" if lines else "")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def read_previous_iteration(current_timestamp: str) -> tuple[str, dict[str, Any]]:
    if not LATEST_RUN_PATH.exists():
        return "", {}
    previous_timestamp = LATEST_RUN_PATH.read_text(encoding="utf-8", errors="ignore").strip()
    if not previous_timestamp or previous_timestamp == current_timestamp:
        return "", {}
    previous_metrics = read_json(ROOT / "ai_runs" / previous_timestamp / "metrics.json")
    return previous_timestamp, previous_metrics


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


def metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "Precision": _metric(metrics, "det_precision"),
        "Recall": _metric(metrics, "det_recall"),
        "F1": _metric(metrics, "det_f1"),
        "Localization error": _metric(metrics, "det_loc_err_m"),
        "False positives": _metric(metrics, "det_fp"),
        "Missed detections": _metric(metrics, "det_fn"),
    }


def format_metric_block(metrics: dict[str, Any]) -> str:
    summary = metric_summary(metrics)
    return "\n".join(f"{name}: {_fmt(value)}" for name, value in summary.items())


def build_ai_context(
    current_metrics: dict[str, Any],
    previous_timestamp: str,
    previous_metrics: dict[str, Any],
    cfg: dict[str, Any],
    timestamp: str,
) -> str:
    actual = current_metrics.get("actual_metrics") if isinstance(current_metrics.get("actual_metrics"), dict) else {}
    exp_cfg = current_metrics.get("experiment_config") if isinstance(current_metrics.get("experiment_config"), dict) else {}
    success = bool(current_metrics.get("success"))
    status = "success" if success else "failed"
    views = exp_cfg.get("views", current_metrics.get("views", "Unavailable"))
    max_frames = exp_cfg.get("max_frames", current_metrics.get("max_frames", "Unavailable"))
    pretrained = exp_cfg.get("pretrained", current_metrics.get("pretrained", "Unavailable"))
    fusion_mode = exp_cfg.get("fusion_mode", current_metrics.get("fusion_mode", "concat"))
    optimizer = exp_cfg.get("optimizer", current_metrics.get("optimizer", "Unavailable"))
    scheduler = exp_cfg.get("scheduler", current_metrics.get("scheduler", "Unavailable"))
    lr_init = exp_cfg.get("lr_init", current_metrics.get("lr_init", "Unavailable"))
    max_lr = exp_cfg.get("max_lr", current_metrics.get("max_lr", "Unavailable"))
    momentum = exp_cfg.get("momentum", current_metrics.get("momentum", "Unavailable"))
    weight_decay = exp_cfg.get("weight_decay", current_metrics.get("weight_decay", "Unavailable"))
    freeze_backbone_epochs = exp_cfg.get(
        "freeze_backbone_epochs",
        current_metrics.get("freeze_backbone_epochs", "Unavailable"),
    )
    checkpoint_path = exp_cfg.get("checkpoint_path", current_metrics.get("checkpoint_path", "Unavailable"))
    train_command = exp_cfg.get("train_command", cfg.get("train_command", []))
    train_command_text = " ".join(str(x) for x in train_command) if isinstance(train_command, list) else str(train_command)
    alpha = exp_cfg.get("alpha", current_metrics.get("alpha", "Unavailable"))
    loss_config = exp_cfg.get("loss_config") if isinstance(exp_cfg.get("loss_config"), dict) else {}

    current_f1 = _metric(current_metrics, "det_f1")
    current_precision = _metric(current_metrics, "det_precision")
    current_recall = _metric(current_metrics, "det_recall")
    current_fp = _metric(current_metrics, "det_fp")
    current_fn = _metric(current_metrics, "det_fn")

    if success and actual:
        observed_problem = (
            "Training and evaluation completed, but the detection result is weak: "
            f"F1={_fmt(current_f1)}, precision={_fmt(current_precision)}, recall={_fmt(current_recall)}, "
            f"false positives={_fmt(current_fp)}, missed detections={_fmt(current_fn)}."
        )
        interpretation = "Inconclusive. This run is a baseline measurement until a same-configuration comparison exists."
    elif success:
        observed_problem = "Training completed, but evaluation metrics are incomplete or unavailable."
        interpretation = "Inconclusive. Required detection metrics are missing."
    else:
        observed_problem = "Training or evaluation failed. See error.log."
        interpretation = "No. The run failed and must be fixed before model optimization."

    previous_iteration = f"ai_runs/{previous_timestamp}/" if previous_timestamp else "No previous formal iteration."
    previous_block = format_metric_block(previous_metrics) if previous_metrics else (
        "Precision: Unavailable\n"
        "Recall: Unavailable\n"
        "F1: Unavailable\n"
        "Localization error: Unavailable\n"
        "False positives: Unavailable\n"
        "Missed detections: Unavailable"
    )
    current_block = format_metric_block(current_metrics)
    extraction_config = current_metrics.get("extraction_config")
    if not isinstance(extraction_config, dict):
        extraction_config = actual.get("extraction_config") if isinstance(actual.get("extraction_config"), dict) else {}

    def _extract_cfg(key: str, default: Any = "Unavailable") -> Any:
        if key in extraction_config:
            return extraction_config[key]
        if key in current_metrics:
            return current_metrics[key]
        return actual.get(key, default)

    return f"""# AI Iteration Context

## 1. Iteration ID

{timestamp}

## 2. Previous Iteration

{previous_iteration}

## 3. Previous Metrics Summary

{previous_block}
Main failure: {_fmt(_metric(previous_metrics, 'main_failure')) if previous_metrics else 'Unavailable'}

## 4. Observed Problem

{observed_problem}

## 5. Improvement Hypothesis

Because the latest successful WildTrack run still has low F1 and the training defaults previously used limited data/views and did not record pretrained status,
we train with an explicit pretrained backbone status, optimizer/scheduler settings, view set, and frame limit,
expecting the resulting metrics to show whether ImageNet initialization and full WildTrack coverage improve BEV detection.

## 6. Changes Made

Changed files:
- Run configuration: recorded pretrained status, optimizer, scheduler, learning rates, freeze_backbone_epochs, views, and max_frames.
- scripts/run_colab_exp.py: prepared metrics.json with comparison-critical training defaults.
- scripts/commit_ai_runs.py: recorded comparison-critical training defaults in ai_context.md.

## 7. Training Configuration

dataset: WildTrack
views: {_fmt(views)}
epochs: {_fmt(exp_cfg.get('epochs', 'Unavailable'))}
batch_size: {_fmt(exp_cfg.get('batch_size', 'Unavailable'))}
learning_rate: {_fmt(lr_init)}
max_frames: {_fmt(max_frames)}
device: Unavailable
seed: Unavailable
checkpoint_path: {_fmt(checkpoint_path)}
pretrained: {_fmt(pretrained)}
fusion_mode: {_fmt(fusion_mode)}
optimizer: {_fmt(optimizer)}
scheduler: {_fmt(scheduler)}
max_lr: {_fmt(max_lr)}
momentum: {_fmt(momentum)}
weight_decay: {_fmt(weight_decay)}
freeze_backbone_epochs: {_fmt(freeze_backbone_epochs)}
alpha: {_fmt(alpha)}
bev_pos_weight: {_fmt(loss_config.get('bev_pos_weight', 'Unavailable'))}
bev_neg_weight: {_fmt(loss_config.get('bev_neg_weight', 'Unavailable'))}
img_pos_weight: {_fmt(loss_config.get('img_pos_weight', 'Unavailable'))}
img_neg_weight: {_fmt(loss_config.get('img_neg_weight', 'Unavailable'))}
train_command: {train_command_text}

## 8. Evaluation Configuration

model_path: {_fmt(checkpoint_path)}
views: {_fmt(views)}
threshold: {_fmt(_metric(current_metrics, 'det_best_threshold'))}
distance_threshold: {_fmt(_extract_cfg('det_dist_thr'))}
min_distance: {_fmt(_extract_cfg('det_min_distance'))}
nms_ksize: {_fmt(_extract_cfg('det_nms_ksize'))}
max_preds: {_fmt(_extract_cfg('det_max_preds'))}
thresholds: {_fmt(_extract_cfg('det_thresholds'))}
metrics_output: metrics.json
device: Unavailable
max_frames: {_fmt(max_frames)}

## 9. Current Metrics

{current_block}
Status: {status}

## 10. Result Interpretation

{interpretation}

## 11. Next Iteration Recommendation

Next action:
Run confidence fusion with a stronger auxiliary supervision setting, for example ALPHA=2.0, under the same WildTrack views and evaluation sweep.

Reason:
Official MVDet balances BEV map loss with averaged per-view image loss through alpha, and the image auxiliary branch is intended to preserve useful per-view detection signal before BEV projection.

Expected validation:
A new ai_runs timestamp whose metrics.json reports alpha=2.0 and improves F1 over the previous confidence-fusion runs under the same evaluation settings.

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

    metrics = read_json(metrics_path)
    timestamp = str(metrics.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S"))
    previous_timestamp, previous_metrics = read_previous_iteration(timestamp)

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
    (run_dir / "ai_context.md").write_text(
        build_ai_context(metrics, previous_timestamp, previous_metrics, cfg, timestamp),
        encoding="utf-8",
    )
    LATEST_RUN_PATH.write_text(f"{timestamp}\n", encoding="utf-8")

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
