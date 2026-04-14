# Notebook Result Output Protocol (MVP)

This protocol keeps Colab notebook changes minimal while giving the local orchestrator stable machine-readable outputs.

## 1. Required files

Write these files into one fixed folder (the same folder configured as `RESULT_DIR` in `.env`):

- `status.json`
- `metrics.json`
- `train.log`
- `last_error.txt`

## 2. `status.json` schema

```json
{
  "round_id": 3,
  "status": "running",
  "stage": "train",
  "started_at": "2026-04-14T10:22:10Z",
  "finished_at": null,
  "message": "epoch 5/50",
  "git_commit": "abc1234",
  "notebook_run_id": "20260414_102210"
}
```

Terminal statuses:

- success: `success`, `succeeded`, `completed`, `done`, `ok`
- failure: `failed`, `error`, `crashed`, `timeout`, `cancelled`

## 3. `metrics.json` schema

```json
{
  "round_id": 3,
  "primary_metrics": {
    "val_loss": 0.1452,
    "val_map": 0.681
  },
  "system_metrics": {
    "epoch": 50,
    "train_seconds": 4220.5,
    "gpu_memory_mb": 14250
  },
  "updated_at": "2026-04-14T11:32:44Z"
}
```

## 4. `last_error.txt`

On failure, write full traceback and the key command error here.  
On success, write an empty string.

## 5. Notebook-side helper snippet

Put this in one reusable notebook cell and call it from train steps:

```python
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path


RESULT_DIR = Path("/content/drive/MyDrive/colab_runs/bev_track_predict")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_status(round_id: int, status: str, stage: str, message: str, finished: bool = False):
    payload = {
        "round_id": round_id,
        "status": status,
        "stage": stage,
        "started_at": _utc_now() if status == "running" else None,
        "finished_at": _utc_now() if finished else None,
        "message": message,
    }
    _atomic_write(RESULT_DIR / "status.json", json.dumps(payload, ensure_ascii=False, indent=2))


def write_metrics(round_id: int, primary: dict, system: dict):
    payload = {
        "round_id": round_id,
        "primary_metrics": primary,
        "system_metrics": system,
        "updated_at": _utc_now(),
    }
    _atomic_write(RESULT_DIR / "metrics.json", json.dumps(payload, ensure_ascii=False, indent=2))


def write_error(exc: Exception):
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _atomic_write(RESULT_DIR / "last_error.txt", text)
```

