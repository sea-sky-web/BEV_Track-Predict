from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "exp_colab.yaml"


def run_git(args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def tail_lines(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:]) + ("\n" if lines else "")


def build_ai_context() -> str:
    return """# AI Training Context

This directory contains the latest Colab training result.

Please read the following files first:

1. ai_runs/latest/metrics.json
2. ai_runs/latest/error.log
3. ai_runs/latest/train_tail.log

Your task:
- If the previous training failed, fix the code according to error.log.
- If the training succeeded but metrics are poor, make a small and testable improvement.
- Keep the Colab entry command unchanged:
  python scripts/run_colab_exp.py
- Do not delete ai_runs.
- Do not commit model checkpoints or large datasets.
- Keep changes minimal and explain why they help.
"""


def main() -> int:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}
    output_dir = Path(str(cfg.get("output_dir", ROOT / "runs" / cfg.get("exp_name", "colab_exp"))))
    latest_dir = ROOT / str(cfg.get("git", {}).get("latest_dir", "ai_runs/latest"))
    history_root = ROOT / str(cfg.get("git", {}).get("history_dir", "ai_runs/history"))
    branch = os.getenv("GITHUB_BRANCH", str(cfg.get("git", {}).get("branch", "main")))
    exp_name = str(cfg.get("exp_name", "colab_exp"))
    log_tail_n = int(cfg.get("log_tail_lines", 500))

    metrics_path = output_dir / "metrics.json"
    train_log = output_dir / "train.log"
    error_log = output_dir / "error.log"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            timestamp = str(metrics.get("timestamp", timestamp))
        except Exception:
            pass

    latest_dir.mkdir(parents=True, exist_ok=True)
    history_dir = history_root / timestamp
    history_dir.mkdir(parents=True, exist_ok=True)

    if metrics_path.exists():
        shutil.copy2(metrics_path, latest_dir / "metrics.json")
        shutil.copy2(metrics_path, history_dir / "metrics.json")
    else:
        (latest_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
        (history_dir / "metrics.json").write_text("{}\n", encoding="utf-8")

    if error_log.exists():
        shutil.copy2(error_log, latest_dir / "error.log")
        shutil.copy2(error_log, history_dir / "error.log")
    else:
        (latest_dir / "error.log").write_text("", encoding="utf-8")
        (history_dir / "error.log").write_text("", encoding="utf-8")

    train_tail = tail_lines(train_log, log_tail_n)
    (latest_dir / "train_tail.log").write_text(train_tail, encoding="utf-8")
    (history_dir / "train_tail.log").write_text(train_tail, encoding="utf-8")

    context = build_ai_context()
    (latest_dir / "ai_context.md").write_text(context, encoding="utf-8")
    (history_dir / "ai_context.md").write_text(context, encoding="utf-8")

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not found, ai_runs prepared locally, skip git commit and push.")
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

    pull_res = run_git(["pull", "origin", branch, "--rebase"])
    if pull_res.returncode != 0:
        print(f"git pull --rebase failed on branch {branch}:\n{pull_res.stdout}\n{pull_res.stderr}")
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
        print(f"git push failed on branch {branch}:\n{push_res.stdout}\n{push_res.stderr}")
        return 1

    print("ai_runs committed and pushed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
