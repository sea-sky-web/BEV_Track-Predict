from __future__ import annotations
import json, os, shutil, subprocess
from datetime import datetime
from pathlib import Path
import yaml
from scripts.utils.log_utils import tail_text

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "exp_colab.yaml"


def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p


def ai_context() -> str:
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
    exp_name = cfg.get("exp_name", "wildtrack_baseline")
    output_dir = Path(cfg.get("output_dir", str(ROOT / "runs" / exp_name)))
    lines = int(cfg.get("log_tail_lines", 500))
    gcfg = cfg.get("git", {})

    metrics = output_dir / "metrics.json"
    train = output_dir / "train.log"
    error = output_dir / "error.log"
    if not metrics.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest = ROOT / gcfg.get("latest_dir", "ai_runs/latest")
    history = ROOT / gcfg.get("history_dir", "ai_runs/history") / timestamp
    latest.mkdir(parents=True, exist_ok=True)
    history.mkdir(parents=True, exist_ok=True)

    for d in (latest, history):
        shutil.copy2(metrics, d / "metrics.json")
        if error.exists():
            shutil.copy2(error, d / "error.log")
        (d / "train_tail.log").write_text(tail_text(train, lines), encoding="utf-8")
        (d / "ai_context.md").write_text(ai_context(), encoding="utf-8")

    run_git(["add", "ai_runs/"])
    commit_msg = f"add training result: {exp_name} {timestamp}"
    c = run_git(["commit", "-m", commit_msg], check=False)
    if c.returncode != 0 and "nothing to commit" not in (c.stdout + c.stderr).lower():
        raise RuntimeError(c.stderr.strip() or c.stdout.strip())

    token = os.getenv("GITHUB_TOKEN")
    user = os.getenv("GITHUB_USER")
    repo = os.getenv("GITHUB_REPO")
    branch = os.getenv("GITHUB_BRANCH", gcfg.get("branch", "main"))
    if not token:
        print("GITHUB_TOKEN not found, skip push.")
        return 0
    if not user or not repo:
        print("GITHUB_USER or GITHUB_REPO missing, skip push.")
        return 0

    remote_url = f"https://x-access-token:{token}@github.com/{user}/{repo}.git"
    run_git(["remote", "set-url", "origin", remote_url])
    pull = run_git(["pull", "origin", branch, "--rebase"], check=False)
    if pull.returncode != 0:
        raise RuntimeError(f"git pull --rebase failed: {pull.stderr.strip() or pull.stdout.strip()}")
    push = run_git(["push", "origin", branch], check=False)
    if push.returncode != 0:
        raise RuntimeError(f"git push failed: {push.stderr.strip() or push.stdout.strip()}")
    print(f"Pushed ai_runs to {branch}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
