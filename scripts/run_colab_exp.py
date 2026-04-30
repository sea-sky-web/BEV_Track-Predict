from __future__ import annotations
import json, subprocess, time
from datetime import datetime
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "exp_colab.yaml"


def git_commit_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def load_cfg() -> dict:
    with CFG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    start = time.time()
    rc = -1
    success = False
    cfg = {}
    exp_name = "unknown"
    output_dir = ROOT / "runs" / "unknown"
    log_path = output_dir / "train.log"
    err_path = output_dir / "error.log"
    metrics_path = output_dir / "metrics.json"

    try:
        cfg = load_cfg()
        exp_name = str(cfg.get("exp_name", "wildtrack_baseline"))
        output_dir = Path(cfg.get("output_dir", str(ROOT / "runs" / exp_name)))
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train.log"
        err_path = output_dir / "error.log"
        metrics_path = output_dir / "metrics.json"

        cmd = [str(x) for x in cfg.get("train_command", [])]
        if not cmd:
            raise ValueError("train_command is empty")

        with log_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
            out.write("[INFO] command: " + " ".join(cmd) + "\n")
            out.flush()
            p = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err, check=False)
            rc = p.returncode
            success = rc == 0

    except Exception as e:
        output_dir.mkdir(parents=True, exist_ok=True)
        with err_path.open("a", encoding="utf-8") as err:
            err.write(f"[FATAL] {e}\n")

    duration = round(time.time() - start, 3)
    actual_metrics = {}
    external_metrics = output_dir / "metrics.json"
    if external_metrics.exists():
        try:
            maybe = json.loads(external_metrics.read_text(encoding="utf-8"))
            if isinstance(maybe, dict) and "actual_metrics" in maybe and isinstance(maybe["actual_metrics"], dict):
                actual_metrics = maybe["actual_metrics"]
        except Exception:
            pass

    target_metric = str(cfg.get("target_metric", ""))
    target_value = cfg.get("target_value", None)
    status = "need_fix" if not success else "need_analysis"
    if success and isinstance(actual_metrics, dict) and target_metric in actual_metrics and isinstance(target_value, (int, float)):
        if float(actual_metrics[target_metric]) >= float(target_value):
            status = "target_reached"

    metrics = {
        "exp_name": exp_name,
        "success": success,
        "return_code": rc,
        "duration_seconds": duration,
        "target_metric": target_metric,
        "target_value": target_value,
        "actual_metrics": actual_metrics,
        "log_path": str(log_path),
        "error_path": str(err_path),
        "timestamp": ts,
        "git_commit": git_commit_hash(),
        "ai_feedback": {
            "status": status,
            "instruction": "请读取 ai_runs/latest 中的文件，分析训练结果并给出下一步修改。",
        },
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {exp_name} rc={rc} success={success}")
    print(f"[DONE] metrics={metrics_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
