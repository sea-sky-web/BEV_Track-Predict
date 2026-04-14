"""Bootstrap Playwright persistent login profile for Colab."""

from __future__ import annotations

import argparse
from pathlib import Path

from colab_automation.config import load_config
from colab_automation.launcher import ColabPlaywrightLauncher
from colab_automation.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize persistent Playwright login profile.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env config file. Default: colab_automation/.env",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.env_file)
    logger = setup_logger(log_dir=config.log_dir, level="INFO")

    if config.playwright_attach_existing_chrome:
        logger.info(
            "PLAYWRIGHT_ATTACH_EXISTING_CHROME=true. "
            "This script is only for persistent-profile login bootstrap."
        )
        logger.info("Set PLAYWRIGHT_ATTACH_EXISTING_CHROME=false if you want to initialize local profile login.")
        return 0

    logger.info("Opening browser with persistent profile: %s", config.playwright_profile_dir)
    logger.info("Please complete Google login manually in the opened window.")

    with ColabPlaywrightLauncher(config=config, logger=logger) as launcher:
        launcher.open_notebook()
        input("After login succeeds and notebook page is visible, press ENTER to save profile and exit...")

    logger.info("Persistent login profile has been saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
