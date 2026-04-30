from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "exp_colab.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def try_read_existing_metrics(output_dir: Path) -> dict[str, Any]:
    candidates = [
        output_dir / "actual_metrics.json",
        output_dir / "eval_metrics.json",
        output_dir / "metrics_raw.json",
    ]
    merged: dict[str, Any] = {}
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    merged.update(data)
            except Exception:
                continue
    return merged


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start = time.time()

    success = False
    return_code = -1
    exc_text = ""

    cfg = load_config(CFG_PATH)
    exp_name = str(cfg.get("exp_name", "colab_exp"))
    output_dir = Path(str(cfg.get("output_dir", ROOT / "runs" / exp_name)))
    train_command = [str(x) for x in cfg.get("train_command", ["python", "scripts/train_main.py"])]
    target_metric = cfg.get("target_metric", "")
    target_value = cfg.get("target_value", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_log = output_dir / "train.log"
    error_log = output_dir / "error.log"
    metrics_file = output_dir / "metrics.json"

    try:
        with train_log.open("w", encoding="utf-8") as out_f, error_log.open("w", encoding="utf-8") as err_f:
            out_f.write(f"[launcher] train_command: {' '.join(train_command)}\n")
            out_f.flush()
            proc = subprocess.run(train_command, cwd=ROOT, stdout=out_f, stderr=err_f, check=False)
            return_code = proc.returncode
            success = return_code == 0
    except Exception as exc:
        exc_text = str(exc)
        success = False
        return_code = -1
        with error_log.open("a", encoding="utf-8") as err_f:
            err_f.write(f"[launcher_exception] {exc_text}\n")

    actual_metrics = try_read_existing_metrics(output_dir)

    status = "target_reached" if success else "need_fix"
    if success and target_metric and isinstance(actual_metrics, dict):
        v = actual_metrics.get(target_metric)
        if isinstance(v, (float, int)) and isinstance(target_value, (float, int)):
            status = "target_reached" if v >= target_value else "need_analysis"
        else:
            status = "need_analysis"

    metrics = {
        "exp_name": exp_name,
        "success": success,
        "return_code": return_code,
        "duration_seconds": round(time.time() - start, 3),
        "target_metric": target_metric,
        "target_value": target_value,
        "actual_metrics": actual_metrics,
        "log_path": str(train_log),
        "error_path": str(error_log),
        "timestamp": timestamp,
        "git_commit": git_commit_hash(),
        "ai_feedback": {
            "status": status,
            "instruction": "请读取 ai_runs/latest 中的文件，分析训练结果并给出下一步修改。",
        },
    }
    if exc_text:
        metrics["launcher_exception"] = exc_text

    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_colab_exp] metrics saved: {metrics_file}")
    return return_code if return_code >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
