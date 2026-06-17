from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "exp_colab.yaml"
DEFAULT_CHECKPOINT_REL = Path("outputs/train_multicam_mvdet_style_v3/model_final.pth")


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


def _flag_value(command: list[str], flag: str, default: Any = None) -> Any:
    try:
        idx = command.index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(command):
        return default
    return command[idx + 1]


def _int_flag_value(command: list[str], flag: str, default: Any = None) -> Any:
    value = _flag_value(command, flag, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_flag_value(command: list[str], flag: str, default: Any = None) -> Any:
    value = _flag_value(command, flag, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_value(raw: Any, default: Any = None) -> Any:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return raw


def _append_float_override(
    command: list[str],
    cfg: dict[str, Any],
    *,
    flag: str,
    cfg_key: str,
    env_key: str,
) -> None:
    if flag in command:
        return
    raw = os.environ.get(env_key)
    if raw is None:
        raw = cfg.get(cfg_key)
    if raw is None:
        return
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_key} / {cfg_key} must be a float, got {raw!r}") from exc
    command.extend([flag, str(value)])


def _append_value_override(
    command: list[str],
    cfg: dict[str, Any],
    *,
    flag: str,
    cfg_key: str,
    env_key: str,
) -> None:
    if flag in command:
        return
    raw = os.environ.get(env_key)
    if raw is None:
        raw = cfg.get(cfg_key)
    if raw is None:
        return
    command.extend([flag, str(raw).lower() if isinstance(raw, bool) else str(raw)])


def _repo_path(raw: Union[str, Path]) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return ROOT / path


def _checkpoint_from_train_log(train_log: Path) -> Optional[str]:
    if not train_log.exists():
        return None
    text = train_log.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"\[OK\]\s+saved\s+(.+?model_final\.pth)", text)
    if not matches:
        return None
    return str(_repo_path(matches[-1].strip()))


def resolve_checkpoint_path(cfg: dict[str, Any], train_command: list[str], train_log: Path) -> str:
    configured = cfg.get("checkpoint_path")
    if configured:
        return str(_repo_path(str(configured)))

    logged = _checkpoint_from_train_log(train_log)
    if logged:
        return logged

    output_arg = _flag_value(train_command, "--output_dir", None)
    if output_arg:
        return str(_repo_path(output_arg) / "model_final.pth")

    return str(ROOT / DEFAULT_CHECKPOINT_REL)


def build_experiment_config(
    cfg: dict[str, Any],
    train_command: list[str],
    train_log: Path,
) -> dict[str, Any]:
    views = str(_flag_value(train_command, "--views", cfg.get("views", "0,1,2,3,4,5,6")))
    max_frames = _int_flag_value(train_command, "--max_frames", cfg.get("max_frames"))
    epochs = _int_flag_value(train_command, "--epochs", cfg.get("epochs"))
    batch_size = _int_flag_value(train_command, "--batch", cfg.get("batch_size"))

    if "--no-pretrained" in train_command:
        pretrained = False
    else:
        pretrained = _bool_value(_flag_value(train_command, "--pretrained", cfg.get("pretrained")))
    backbone = str(_flag_value(train_command, "--backbone", cfg.get("backbone", "resnet18")))
    fusion_mode = str(_flag_value(train_command, "--fusion_mode", cfg.get("fusion_mode", "confidence_v2")))
    optimizer = str(_flag_value(train_command, "--optimizer", cfg.get("optimizer", "adam")))
    scheduler = str(_flag_value(train_command, "--scheduler", cfg.get("scheduler", "cosine")))
    lr_init = _float_flag_value(train_command, "--lr_init", cfg.get("lr_init"))
    max_lr = _float_flag_value(train_command, "--max_lr", cfg.get("max_lr"))
    momentum = _float_flag_value(train_command, "--momentum", cfg.get("momentum"))
    weight_decay = _float_flag_value(train_command, "--weight_decay", cfg.get("weight_decay"))
    freeze_backbone_epochs = _int_flag_value(
        train_command,
        "--freeze_backbone_epochs",
        cfg.get("freeze_backbone_epochs"),
    )
    augment = False if "--no-augment" in train_command else _bool_value(
        _flag_value(train_command, "--augment", cfg.get("augment")),
        default=None,
    )
    augment_hflip_prob = _float_flag_value(
        train_command,
        "--augment_hflip_prob",
        cfg.get("augment_hflip_prob"),
    )
    augment_color_jitter = str(
        _flag_value(train_command, "--augment_color_jitter", cfg.get("augment_color_jitter", ""))
    )
    alpha = _float_flag_value(train_command, "--alpha", None)
    if alpha is None:
        try:
            alpha = float(cfg.get("alpha", 1.0))
        except (TypeError, ValueError):
            alpha = 1.0
    loss_config = {
        "bev_pos_weight": _float_flag_value(train_command, "--bev_pos_weight", 1.0),
        "bev_neg_weight": _float_flag_value(train_command, "--bev_neg_weight", 1.0),
        "img_pos_weight": _float_flag_value(train_command, "--img_pos_weight", 1.0),
        "img_neg_weight": _float_flag_value(train_command, "--img_neg_weight", 1.0),
    }

    return {
        "dataset": "WildTrack",
        "data_root": str(cfg.get("data_root") or _flag_value(train_command, "--data_root", "wildtrack")),
        "views": views,
        "max_frames": max_frames,
        "epochs": epochs,
        "batch_size": batch_size,
        "pretrained": pretrained,
        "backbone": backbone,
        "fusion_mode": fusion_mode,
        "augment": augment,
        "augment_hflip_prob": augment_hflip_prob,
        "augment_color_jitter": augment_color_jitter,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "lr_init": lr_init,
        "max_lr": max_lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "alpha": alpha,
        "loss_config": loss_config,
        "train_command": train_command,
        "checkpoint_path": resolve_checkpoint_path(cfg, train_command, train_log),
        "det_moda_dist_m": _float_flag_value(train_command, "--det_moda_dist_m", cfg.get("det_moda_dist_m")),
        "metrics_sources": ["actual_metrics.json", "eval_metrics.json", "metrics_raw.json"],
    }


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
    _append_value_override(
        train_command, cfg,
        flag="--views", cfg_key="views", env_key="VIEWS",
    )
    _append_value_override(
        train_command, cfg,
        flag="--max_frames", cfg_key="max_frames", env_key="MAX_FRAMES",
    )
    _append_value_override(
        train_command, cfg,
        flag="--epochs", cfg_key="epochs", env_key="EPOCHS",
    )
    _append_value_override(
        train_command, cfg,
        flag="--batch", cfg_key="batch_size", env_key="BATCH_SIZE",
    )
    _append_value_override(
        train_command, cfg,
        flag="--pretrained", cfg_key="pretrained", env_key="PRETRAINED",
    )
    _append_value_override(
        train_command, cfg,
        flag="--backbone", cfg_key="backbone", env_key="BACKBONE",
    )
    fusion_mode = str(os.environ.get("FUSION_MODE") or cfg.get("fusion_mode", "confidence_v2")).strip().lower()
    if "--fusion_mode" not in train_command:
        train_command.extend(["--fusion_mode", fusion_mode])
    _append_value_override(
        train_command, cfg,
        flag="--augment", cfg_key="augment", env_key="AUGMENT",
    )
    _append_float_override(
        train_command, cfg,
        flag="--augment_hflip_prob", cfg_key="augment_hflip_prob", env_key="AUGMENT_HFLIP_PROB",
    )
    _append_value_override(
        train_command, cfg,
        flag="--augment_color_jitter", cfg_key="augment_color_jitter", env_key="AUGMENT_COLOR_JITTER",
    )
    _append_value_override(
        train_command, cfg,
        flag="--optimizer", cfg_key="optimizer", env_key="OPTIMIZER",
    )
    _append_value_override(
        train_command, cfg,
        flag="--scheduler", cfg_key="scheduler", env_key="SCHEDULER",
    )
    _append_value_override(
        train_command, cfg,
        flag="--freeze_backbone_epochs", cfg_key="freeze_backbone_epochs", env_key="FREEZE_BACKBONE_EPOCHS",
    )
    _append_float_override(
        train_command, cfg,
        flag="--alpha", cfg_key="alpha", env_key="ALPHA",
    )
    _append_float_override(
        train_command, cfg,
        flag="--lr_init", cfg_key="lr_init", env_key="LR_INIT",
    )
    _append_float_override(
        train_command, cfg,
        flag="--max_lr", cfg_key="max_lr", env_key="MAX_LR",
    )
    _append_float_override(
        train_command, cfg,
        flag="--momentum", cfg_key="momentum", env_key="MOMENTUM",
    )
    _append_float_override(
        train_command, cfg,
        flag="--weight_decay", cfg_key="weight_decay", env_key="WEIGHT_DECAY",
    )
    _append_float_override(
        train_command, cfg,
        flag="--bev_pos_weight", cfg_key="bev_pos_weight", env_key="BEV_POS_WEIGHT",
    )
    _append_float_override(
        train_command, cfg,
        flag="--bev_neg_weight", cfg_key="bev_neg_weight", env_key="BEV_NEG_WEIGHT",
    )
    _append_float_override(
        train_command, cfg,
        flag="--img_pos_weight", cfg_key="img_pos_weight", env_key="IMG_POS_WEIGHT",
    )
    _append_float_override(
        train_command, cfg,
        flag="--img_neg_weight", cfg_key="img_neg_weight", env_key="IMG_NEG_WEIGHT",
    )
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

    if success and error_log.exists() and error_log.read_text(encoding="utf-8", errors="ignore").strip() == "":
        error_log.write_text("No error.\n", encoding="utf-8")

    actual_metrics = try_read_existing_metrics(output_dir)
    experiment_config = build_experiment_config(cfg, train_command, train_log)

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
        "experiment_config": experiment_config,
        "dataset": experiment_config["dataset"],
        "views": experiment_config["views"],
        "max_frames": experiment_config["max_frames"],
        "pretrained": experiment_config["pretrained"],
        "backbone": experiment_config["backbone"],
        "fusion_mode": experiment_config["fusion_mode"],
        "augment": experiment_config["augment"],
        "augment_hflip_prob": experiment_config["augment_hflip_prob"],
        "augment_color_jitter": experiment_config["augment_color_jitter"],
        "optimizer": experiment_config["optimizer"],
        "scheduler": experiment_config["scheduler"],
        "lr_init": experiment_config["lr_init"],
        "max_lr": experiment_config["max_lr"],
        "momentum": experiment_config["momentum"],
        "weight_decay": experiment_config["weight_decay"],
        "freeze_backbone_epochs": experiment_config["freeze_backbone_epochs"],
        "alpha": experiment_config["alpha"],
        "det_moda_dist_m": experiment_config["det_moda_dist_m"],
        "checkpoint_path": experiment_config["checkpoint_path"],
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
