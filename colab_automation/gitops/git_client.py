"""Thin git subprocess wrapper."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class GitCommandError(RuntimeError):
    """Raised when git command exits with non-zero code."""

    def __init__(self, args: Sequence[str], stdout: str, stderr: str, returncode: int) -> None:
        self.args_used = list(args)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        joined = " ".join(args)
        super().__init__(f"Git command failed ({returncode}): {joined}\nSTDERR: {stderr.strip()}")


@dataclass
class GitResult:
    stdout: str
    stderr: str
    returncode: int


class GitClient:
    """Execute git commands inside a local repository."""

    def __init__(self, repo_path: Path, logger: logging.Logger, remote_name: str = "origin") -> None:
        self.repo_path = repo_path
        self.logger = logger
        self.remote_name = remote_name
        self._assert_repo()

    def has_changes(self) -> bool:
        result = self.run_git(("status", "--porcelain"))
        return bool(result.stdout.strip())

    def get_diff(self) -> str:
        return self.run_git(("diff", "--")).stdout

    def current_branch(self) -> str:
        return self.run_git(("rev-parse", "--abbrev-ref", "HEAD")).stdout.strip()

    def commit_all(self, message: str) -> bool:
        """Stage all tracked/untracked changes and commit."""

        if not self.has_changes():
            self.logger.info("No local changes, skipping commit.")
            return False

        self.run_git(("add", "-A"))
        staged = self.run_git(("diff", "--cached", "--name-only")).stdout.strip()
        if not staged:
            self.logger.info("No staged changes after git add, skipping commit.")
            return False

        self.logger.info("Committing changes with message: %s", message)
        self.run_git(("commit", "-m", message))
        return True

    def push(self, branch: str) -> None:
        """Push current HEAD to remote branch."""

        self.logger.info("Pushing HEAD to %s/%s", self.remote_name, branch)
        self.run_git(("push", self.remote_name, f"HEAD:{branch}"))

    def run_git(self, args: Sequence[str], check: bool = True) -> GitResult:
        """Run a git command and return output."""

        cmd = ["git", *args]
        proc = subprocess.run(
            cmd,
            cwd=self.repo_path,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        result = GitResult(stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
        if check and proc.returncode != 0:
            raise GitCommandError(args=cmd, stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
        return result

    def _assert_repo(self) -> None:
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a git repository: {self.repo_path}")
