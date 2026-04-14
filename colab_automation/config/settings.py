"""Environment-based configuration loader."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for the automation loop."""

    colab_notebook_url: str
    repo_path: Path
    git_branch: str
    git_remote_name: str
    result_dir: Path
    status_file_name: str
    metrics_file_name: str
    error_file_name: str
    train_log_file_name: str
    poll_interval_seconds: int
    round_timeout_seconds: int
    max_rounds: int
    launch_retry: int
    loop_sleep_seconds: int
    selector_timeout_ms: int
    playwright_profile_dir: Path
    playwright_headless: bool
    playwright_channel: Optional[str]
    playwright_attach_existing_chrome: bool
    playwright_cdp_url: Optional[str]
    playwright_attach_new_tab: bool
    model_config_path: Optional[Path]
    patch_output_dir: Path
    log_dir: Path
    auto_apply_patch: bool

    @property
    def status_path(self) -> Path:
        return self.result_dir / self.status_file_name

    @property
    def metrics_path(self) -> Path:
        return self.result_dir / self.metrics_file_name

    @property
    def error_path(self) -> Path:
        return self.result_dir / self.error_file_name

    @property
    def train_log_path(self) -> Path:
        return self.result_dir / self.train_log_file_name

    def ensure_dirs(self) -> None:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.playwright_profile_dir.mkdir(parents=True, exist_ok=True)
        self.patch_output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def _parse_env_file(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_file.exists():
        return values

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def _get_value(source: dict[str, str], key: str, default: Optional[str] = None) -> Optional[str]:
    if key in os.environ:
        return os.environ[key]
    return source.get(key, default)


def _get_required_value(source: dict[str, str], key: str) -> str:
    value = _get_value(source, key)
    if value is None or not value.strip():
        raise ValueError(f"Missing required config key: {key}")
    return value


def _to_int(value: str, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Config key {key} must be an integer, got: {value!r}") from exc


def _to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def load_config(env_file: Optional[Path] = None) -> AppConfig:
    """Load application config from an env file and process environment."""

    default_env = Path(__file__).resolve().parents[1] / ".env"
    target_env = env_file.resolve() if env_file else default_env
    base_dir = target_env.parent
    values = _parse_env_file(target_env)

    colab_notebook_url = _get_required_value(values, "COLAB_NOTEBOOK_URL")
    repo_path = _resolve_path(_get_required_value(values, "REPO_PATH"), base_dir)
    result_dir = _resolve_path(_get_required_value(values, "RESULT_DIR"), base_dir)
    playwright_profile_dir = _resolve_path(
        _get_value(values, "PLAYWRIGHT_PROFILE_DIR", "./runtime/playwright_profile") or "./runtime/playwright_profile",
        base_dir,
    )
    patch_output_dir = _resolve_path(
        _get_value(values, "PATCH_OUTPUT_DIR", "runtime/patches") or "runtime/patches",
        base_dir,
    )
    log_dir = _resolve_path(_get_value(values, "LOG_DIR", "runtime/logs") or "runtime/logs", base_dir)

    model_config_value = _get_value(values, "MODEL_CONFIG_PATH")
    model_config_path = _resolve_path(model_config_value, base_dir) if model_config_value else None

    config = AppConfig(
        colab_notebook_url=colab_notebook_url,
        repo_path=repo_path,
        git_branch=_get_value(values, "GIT_BRANCH", "main") or "main",
        git_remote_name=_get_value(values, "GIT_REMOTE_NAME", "origin") or "origin",
        result_dir=result_dir,
        status_file_name=_get_value(values, "STATUS_FILE_NAME", "status.json") or "status.json",
        metrics_file_name=_get_value(values, "METRICS_FILE_NAME", "metrics.json") or "metrics.json",
        error_file_name=_get_value(values, "ERROR_FILE_NAME", "last_error.txt") or "last_error.txt",
        train_log_file_name=_get_value(values, "TRAIN_LOG_FILE_NAME", "train.log") or "train.log",
        poll_interval_seconds=_to_int(_get_value(values, "POLL_INTERVAL_SECONDS", "20") or "20", "POLL_INTERVAL_SECONDS"),
        round_timeout_seconds=_to_int(
            _get_value(values, "ROUND_TIMEOUT_SECONDS", "10800") or "10800",
            "ROUND_TIMEOUT_SECONDS",
        ),
        max_rounds=_to_int(_get_value(values, "MAX_ROUNDS", "5") or "5", "MAX_ROUNDS"),
        launch_retry=_to_int(_get_value(values, "LAUNCH_RETRY", "3") or "3", "LAUNCH_RETRY"),
        loop_sleep_seconds=_to_int(_get_value(values, "LOOP_SLEEP_SECONDS", "10") or "10", "LOOP_SLEEP_SECONDS"),
        selector_timeout_ms=_to_int(_get_value(values, "SELECTOR_TIMEOUT_MS", "15000") or "15000", "SELECTOR_TIMEOUT_MS"),
        playwright_profile_dir=playwright_profile_dir,
        playwright_headless=_to_bool(_get_value(values, "PLAYWRIGHT_HEADLESS", "false") or "false"),
        playwright_channel=_get_value(values, "PLAYWRIGHT_CHANNEL", None),
        playwright_attach_existing_chrome=_to_bool(
            _get_value(values, "PLAYWRIGHT_ATTACH_EXISTING_CHROME", "false") or "false"
        ),
        playwright_cdp_url=_get_value(values, "PLAYWRIGHT_CDP_URL", "http://127.0.0.1:9222"),
        playwright_attach_new_tab=_to_bool(_get_value(values, "PLAYWRIGHT_ATTACH_NEW_TAB", "true") or "true"),
        model_config_path=model_config_path,
        patch_output_dir=patch_output_dir,
        log_dir=log_dir,
        auto_apply_patch=_to_bool(_get_value(values, "AUTO_APPLY_PATCH", "false") or "false"),
    )

    config.ensure_dirs()
    return config
