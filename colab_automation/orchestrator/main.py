"""Main automation loop for Colab training iteration."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from colab_automation.agent import (
    HeuristicPatchAgent,
    PatchApplier,
    PatchApplyResult,
    PatchPlan,
    PatchRequest,
)
from colab_automation.config import AppConfig
from colab_automation.gitops import GitClient, GitCommandError
from colab_automation.launcher import ColabPlaywrightLauncher
from colab_automation.monitor import TrainingResult, TrainingResultMonitor


class ColabAutomationOrchestrator:
    """Coordinate patch generation, git push, Colab launch, and monitor loop."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

        self.git = GitClient(config.repo_path, logger=logger, remote_name=config.git_remote_name)
        self.monitor = TrainingResultMonitor(config=config, logger=logger)
        self.patch_agent = HeuristicPatchAgent(logger=logger, repo_path=config.repo_path)
        self.patch_applier = PatchApplier(repo_path=config.repo_path, output_dir=config.patch_output_dir, logger=logger)

    def run(self) -> int:
        """Execute training loop until success or max rounds."""

        self.logger.info("Starting Colab automation loop. max_rounds=%d", self.config.max_rounds)

        for round_id in range(1, self.config.max_rounds + 1):
            self.logger.info("======== Round %d / %d ========", round_id, self.config.max_rounds)

            patch_request = self._build_patch_request(round_id)
            patch_plan = self.patch_agent.generate_patch(patch_request)
            apply_result = self.patch_applier.apply_plan(
                request=patch_request,
                plan=patch_plan,
                auto_apply_enabled=self.config.auto_apply_patch,
            )
            self.logger.info("Patch step result: %s", apply_result.message)

            try:
                self._commit_and_push_if_needed(round_id, patch_plan)
            except GitCommandError as exc:
                self.logger.error("Git operation failed on round %d: %s", round_id, exc)
                failed = self.monitor.build_failed_result(round_id=round_id, message=f"git_error: {exc}")
                self._write_round_report(round_id, failed, patch_plan, apply_result)
                return 1

            result = self._launch_and_wait(round_id=round_id)
            self._write_round_report(round_id, result, patch_plan, apply_result)

            if result.is_success:
                self.logger.info("Loop finished successfully at round %d.", round_id)
                return 0

            self.logger.warning("Round %d failed: %s", round_id, result.message)
            if round_id < self.config.max_rounds:
                self.logger.info("Sleeping %s seconds before next round.", self.config.loop_sleep_seconds)
                time.sleep(self.config.loop_sleep_seconds)

        self.logger.error("Reached max_rounds without success.")
        return 1

    def _build_patch_request(self, round_id: int) -> PatchRequest:
        config_text = ""
        if self.config.model_config_path and self.config.model_config_path.exists():
            config_text = self.config.model_config_path.read_text(encoding="utf-8", errors="replace")

        return PatchRequest(
            round_id=round_id,
            status_payload=self.monitor.read_status_payload(),
            metrics_payload=self.monitor.read_metrics_payload(),
            last_error_text=self.monitor.read_error_text(),
            config_file_path=self.config.model_config_path,
            config_text=config_text,
            git_diff=self.git.get_diff(),
        )

    def _commit_and_push_if_needed(self, round_id: int, patch_plan: PatchPlan) -> None:
        if not self.git.has_changes():
            self.logger.info("No repo changes to commit on round %d.", round_id)
            return

        commit_message = self._build_commit_message(round_id, patch_plan)
        committed = self.git.commit_all(commit_message)
        if not committed:
            self.logger.info("Commit skipped after staging check on round %d.", round_id)
            return
        self.git.push(self.config.git_branch)

    def _launch_and_wait(self, round_id: int) -> TrainingResult:
        last_exception: Exception | None = None

        for attempt in range(1, max(1, self.config.launch_retry) + 1):
            self.logger.info("Launcher attempt %d/%d", attempt, self.config.launch_retry)
            try:
                with ColabPlaywrightLauncher(config=self.config, logger=self.logger) as launcher:
                    launched_at = launcher.launch_training(round_id=round_id)
                    return self.monitor.wait_for_result(
                        start_after=launched_at,
                        expected_round=round_id,
                        timeout_seconds=self.config.round_timeout_seconds,
                        poll_interval_seconds=self.config.poll_interval_seconds,
                    )
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                self.logger.exception("Launcher attempt %d failed: %s", attempt, exc)
                if attempt < self.config.launch_retry:
                    time.sleep(min(10 * attempt, 30))

        assert last_exception is not None
        return self.monitor.build_failed_result(round_id=round_id, message=f"launcher_error: {last_exception}")

    def _write_round_report(
        self,
        round_id: int,
        result: TrainingResult,
        patch_plan: PatchPlan,
        apply_result: PatchApplyResult,
    ) -> Path:
        output = {
            "round_id": round_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "result": {
                "round_id": result.round_id,
                "status": result.status,
                "is_success": result.is_success,
                "message": result.message,
                "observed_at": result.observed_at.isoformat(),
            },
            "status_payload": result.status_payload,
            "metrics_payload": result.metrics_payload,
            "error_text": result.error_text,
            "train_log_tail": result.train_log_tail,
            "patch_plan": asdict(patch_plan),
            "patch_apply_result": {
                "applied": apply_result.applied,
                "patch_file": str(apply_result.patch_file) if apply_result.patch_file else None,
                "notes_file": str(apply_result.notes_file) if apply_result.notes_file else None,
                "message": apply_result.message,
            },
        }
        report_path = self.config.patch_output_dir / f"round_{round_id:03d}_report.json"
        report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Round report written to %s", report_path)
        return report_path

    @staticmethod
    def _build_commit_message(round_id: int, patch_plan: PatchPlan) -> str:
        summary = patch_plan.summary.strip()
        if len(summary) > 80:
            summary = summary[:80]
        summary = summary or "automated patch"
        return f"auto(colab): round {round_id} - {summary}"

