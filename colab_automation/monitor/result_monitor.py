"""Training artifact polling and parsing."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from colab_automation.config import AppConfig


@dataclass
class TrainingResult:
    """Structured result for one training round."""

    round_id: Optional[int]
    status: str
    is_success: bool
    message: str
    status_payload: dict[str, Any] = field(default_factory=dict)
    metrics_payload: dict[str, Any] = field(default_factory=dict)
    error_text: str = ""
    train_log_tail: str = ""
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TrainingResultMonitor:
    """Poll result files produced by notebook training cells."""

    SUCCESS_STATES = {"success", "succeeded", "completed", "done", "ok"}
    FAILURE_STATES = {"failed", "error", "crashed", "timeout", "cancelled"}

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def read_status_payload(self) -> dict[str, Any]:
        return self._read_json(self.config.status_path) or {}

    def read_metrics_payload(self) -> dict[str, Any]:
        return self._read_json(self.config.metrics_path) or {}

    def read_error_text(self) -> str:
        return self._read_text(self.config.error_path)

    def read_train_log_tail(self, lines: int = 200) -> str:
        return self._tail_text(self.config.train_log_path, lines=lines)

    def wait_for_result(
        self,
        start_after: Optional[datetime],
        expected_round: Optional[int],
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> TrainingResult:
        """Poll status file until terminal status appears."""

        self.logger.info(
            "Start polling result files under %s (timeout=%ss, interval=%ss)",
            self.config.result_dir,
            timeout_seconds,
            poll_interval_seconds,
        )
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            status_payload = self.read_status_payload()
            if status_payload:
                result = self._try_build_result(status_payload, start_after, expected_round)
                if result is not None:
                    self.logger.info("Detected terminal result: status=%s, success=%s", result.status, result.is_success)
                    return result
            time.sleep(poll_interval_seconds)

        raise TimeoutError(
            f"No terminal status detected within {timeout_seconds}s. "
            f"Expected round={expected_round}, status file={self.config.status_path}"
        )

    def build_failed_result(self, round_id: Optional[int], message: str) -> TrainingResult:
        """Create synthetic failure result."""

        return TrainingResult(
            round_id=round_id,
            status="failed",
            is_success=False,
            message=message,
            status_payload=self.read_status_payload(),
            metrics_payload=self.read_metrics_payload(),
            error_text=self.read_error_text(),
            train_log_tail=self.read_train_log_tail(),
        )

    def _try_build_result(
        self,
        status_payload: dict[str, Any],
        start_after: Optional[datetime],
        expected_round: Optional[int],
    ) -> Optional[TrainingResult]:
        status_path = self.config.status_path
        if start_after and status_path.exists():
            status_mtime = datetime.fromtimestamp(status_path.stat().st_mtime, tz=timezone.utc)
            if status_mtime < start_after:
                return None

        status_raw = str(status_payload.get("status", "")).strip().lower()
        if status_raw not in self.SUCCESS_STATES and status_raw not in self.FAILURE_STATES:
            return None

        round_id = self._parse_int(status_payload.get("round_id"))
        if expected_round is not None and round_id is not None and round_id != expected_round:
            self.logger.debug("Ignoring status for round %s while expecting round %s", round_id, expected_round)
            return None

        message = str(status_payload.get("message") or "")
        is_success = status_raw in self.SUCCESS_STATES
        if not message:
            message = "success" if is_success else "failed"

        return TrainingResult(
            round_id=round_id,
            status=status_raw,
            is_success=is_success,
            message=message,
            status_payload=status_payload,
            metrics_payload=self.read_metrics_payload(),
            error_text=self.read_error_text(),
            train_log_tail=self.read_train_log_tail(),
        )

    @staticmethod
    def _parse_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _read_json(self, path: Path) -> Optional[dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            self.logger.warning("JSON file is not an object: %s", path)
            return None
        except json.JSONDecodeError as exc:
            self.logger.warning("Failed to parse JSON %s: %s", path, exc)
            return None
        except OSError as exc:
            self.logger.warning("Failed to read JSON %s: %s", path, exc)
            return None

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            self.logger.warning("Failed to read text %s: %s", path, exc)
            return ""

    def _tail_text(self, path: Path, lines: int) -> str:
        if not path.exists():
            return ""
        try:
            queue: deque[str] = deque(maxlen=max(1, lines))
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    queue.append(line.rstrip("\n"))
            return "\n".join(queue)
        except OSError as exc:
            self.logger.warning("Failed to tail file %s: %s", path, exc)
            return ""

