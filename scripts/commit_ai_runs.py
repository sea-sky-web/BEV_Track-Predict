from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

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


def build_ai_context() -> str:
    return """# AI Training Context

Please follow this order:

1. Read `ai_runs/latest_run.txt` to get the latest timestamp.
2. Read `ai_runs/{timestamp}/metrics.json`.
3. Read `ai_runs/{timestamp}/error.log` and `ai_runs/{timestamp}/train_tail.log`.

Your task:
- If training failed, fix the error according to `error.log`.
- If training succeeded but metrics are poor, make a small, testable optimization.
- Keep the training entry command unchanged:
  `python scripts/run_colab_exp.py`
- Do not commit `runs/`, `wildtrack/`, model weights, or datasets.
- Keep changes minimal and explain why they help.
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            timestamp = str(metrics.get("timestamp") or timestamp)
        except Exception:
            pass

    run_dir = ROOT / "ai_runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if metrics_path.exists():
        (run_dir / "metrics.json").write_text(metrics_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    else:
        (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")

    if error_log.exists():
        (run_dir / "error.log").write_text(error_log.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    else:
        (run_dir / "error.log").write_text("", encoding="utf-8")

    (run_dir / "train_tail.log").write_text(tail_lines(train_log, log_tail_n), encoding="utf-8")
    (run_dir / "ai_context.md").write_text(build_ai_context(), encoding="utf-8")
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
