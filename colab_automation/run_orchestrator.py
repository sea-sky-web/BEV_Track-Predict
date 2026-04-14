"""CLI entrypoint for the Colab automation loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from colab_automation.config import load_config
from colab_automation.orchestrator import ColabAutomationOrchestrator
from colab_automation.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Colab training automation loop.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env config file. Default: colab_automation/.env",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.env_file)
    logger = setup_logger(log_dir=config.log_dir, level=args.log_level)
    logger.info("Loaded config. repo=%s branch=%s", config.repo_path, config.git_branch)

    orchestrator = ColabAutomationOrchestrator(config=config, logger=logger)
    return orchestrator.run()


if __name__ == "__main__":
    raise SystemExit(main())

