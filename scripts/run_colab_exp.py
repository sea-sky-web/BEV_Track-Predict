"""Colab 统一实验入口：读取配置、执行训练、落盘标准化产物。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "exp_colab.yaml"


def _parse_scalar(raw: str) -> Any:
    v = raw.strip().strip('"').strip("'")
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _load_config(path: Path) -> Dict[str, Any]:
    """解析当前项目所需的精简 YAML 结构，避免额外依赖。"""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    cfg: Dict[str, Any] = {"data": {}, "output": {}, "train": {}, "target_metric": {}}
    train_args: List[Any] = []
    in_train_args = False
    args_indent = -1

    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        s = line.strip()

        if in_train_args:
            if indent <= args_indent:
                in_train_args = False
            elif s.startswith("- "):
                train_args.append(_parse_scalar(s[2:]))
                continue

        if s == "train:" or s == "data:" or s == "output:" or s == "target_metric:":
            continue
        if s == "args:":
            in_train_args = True
            args_indent = indent
            continue

        if ":" not in s:
            continue

        key, value = [x.strip() for x in s.split(":", 1)]
        val = _parse_scalar(value)

        if indent == 0:
            cfg[key] = val
        elif indent == 2:
            parent = None
            # 由结构约定判定归属
            if key in {"data_root"}:
                parent = "data"
            elif key in {"runs_root"}:
                parent = "output"
            elif key in {"train_script"}:
                parent = "train"
            elif key in {"name", "value"}:
                parent = "target_metric"
            if parent is not None:
                cfg[parent][key] = val

    cfg["train"]["args"] = train_args
    return cfg


def _build_command(cfg: Dict[str, Any]) -> list[str]:
    train_cfg = cfg.get("train", {})
    train_script = train_cfg.get("train_script", "scripts/train_main.py")
    args = train_cfg.get("args", [])
    if not isinstance(args, list):
        raise ValueError("train.args 必须是字符串列表")
    return [sys.executable, str(ROOT / train_script), *[str(x) for x in args]]


def main() -> int:
    start = time.time()
    exp_name = "colab_exp"
    run_dir = ROOT / "runs" / exp_name
    log_path = run_dir / "train.log"
    err_path = run_dir / "error.log"
    metrics_path = run_dir / "metrics.json"

    success = False
    return_code = -1
    target_metric_name = ""
    target_metric_value = None

    try:
        cfg = _load_config(CFG_PATH)
        exp_name = str(cfg.get("exp_name", "colab_exp"))

        runs_root = Path(cfg.get("output", {}).get("runs_root", "runs"))
        if not runs_root.is_absolute():
            runs_root = ROOT / runs_root

        run_dir = runs_root / exp_name
        log_path = run_dir / "train.log"
        err_path = run_dir / "error.log"
        metrics_path = run_dir / "metrics.json"
        run_dir.mkdir(parents=True, exist_ok=True)

        target_metric = cfg.get("target_metric", {})
        target_metric_name = target_metric.get("name", "")
        target_metric_value = target_metric.get("value", None)

        cmd = _build_command(cfg)

        with log_path.open("w", encoding="utf-8") as log_f, err_path.open("w", encoding="utf-8") as err_f:
            log_f.write(f"[INFO] command: {' '.join(cmd)}\n")
            log_f.flush()
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log_f, stderr=err_f, check=False)
            return_code = int(proc.returncode)
            success = return_code == 0

    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        with err_path.open("a", encoding="utf-8") as err_f:
            err_f.write(f"[FATAL] launcher exception: {exc}\n")
        success = False
        return_code = -1

    duration = time.time() - start
    metrics = {
        "exp_name": exp_name,
        "success": success,
        "return_code": return_code,
        "duration_seconds": round(duration, 3),
        "target_metric": target_metric_name,
        "target_value": target_metric_value,
        "actual_metrics": {},
        "log_path": str(log_path),
        "error_path": str(err_path),
        "ai_feedback": "success" if success else "training failed; check error.log",
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[DONE] exp={exp_name} success={success} rc={return_code}")
    print(f"[DONE] metrics: {metrics_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
