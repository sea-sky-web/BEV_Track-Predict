"""Google Colab page automation via Playwright sync API."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import urlopen
from typing import Any, Optional

try:
    from playwright.sync_api import (
        Browser,
        BrowserContext,
        Dialog,
        Locator,
        Page,
        Playwright,
        TimeoutError as PlaywrightTimeoutError,
        sync_playwright,
    )

    PLAYWRIGHT_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    PLAYWRIGHT_AVAILABLE = False
    BrowserContext = Any  # type: ignore[misc,assignment]
    Browser = Any  # type: ignore[misc,assignment]
    Dialog = Any  # type: ignore[misc,assignment]
    Locator = Any  # type: ignore[misc,assignment]
    Page = Any  # type: ignore[misc,assignment]
    Playwright = Any  # type: ignore[misc,assignment]

    class PlaywrightTimeoutError(Exception):
        """Fallback timeout error placeholder when Playwright is missing."""

    def sync_playwright() -> Any:
        raise ModuleNotFoundError("playwright is not installed")

from colab_automation.config import AppConfig
from colab_automation.utils import retry


class ColabPlaywrightLauncher:
    """Launch Colab notebook, connect runtime, and trigger Run All."""

    CONNECT_BUTTON_PATTERNS = [
        re.compile(r"^\s*(connect|连接|連線)\s*$", re.IGNORECASE),
        re.compile(r"^\s*(reconnect|重新连接|重新連線)\s*$", re.IGNORECASE),
    ]

    CONFIRM_BUTTON_PATTERNS = [
        re.compile(r"(run anyway|仍要运行|仍然运行|继续执行)", re.IGNORECASE),
        re.compile(r"(ok|确定|確認|yes|继续|繼續)", re.IGNORECASE),
        re.compile(r"(connect anyway|仍然连接|仍要连接)", re.IGNORECASE),
    ]

    RUNTIME_MENU_PATTERNS = [
        re.compile(r"^\s*(runtime|代码执行程序|運算階段|运行时)\s*$", re.IGNORECASE),
    ]

    RUN_ALL_PATTERNS = [
        re.compile(r"(run all|run all cells|全部运行|运行所有|运行全部)", re.IGNORECASE),
    ]

    RUNNING_INDICATOR_PATTERNS = [
        re.compile(r"(busy|正在运行|运行中|執行中)", re.IGNORECASE),
        re.compile(r"(interrupt execution|中断执行|停止执行)", re.IGNORECASE),
    ]

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._owns_context: bool = False
        self._created_new_page: bool = False

    def __enter__(self) -> "ColabPlaywrightLauncher":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def start(self) -> "ColabPlaywrightLauncher":
        """Start browser session (persistent context or CDP attachment)."""

        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright is required but not installed. "
                "Run: pip install -r colab_automation/requirements.txt "
                "and: python -m playwright install chromium"
            )

        try:
            self._playwright = sync_playwright().start()
            if self.config.playwright_attach_existing_chrome:
                self._start_by_attach_cdp()
            else:
                self._start_by_persistent_context()

            assert self._page is not None
            self._page.set_default_timeout(self.config.selector_timeout_ms)
            self._page.on("dialog", self._handle_dialog)
            return self
        except Exception:
            # Ensure every failed startup attempt fully releases Playwright
            # so retry attempts do not hit stale event-loop state.
            self.close()
            raise

    def close(self) -> None:
        """Close browser resources."""

        if self._context and self._owns_context:
            self.logger.info("Closing Playwright-owned context")
            self._context.close()
            self._context = None
        elif self._context:
            if self._created_new_page and self._page:
                try:
                    self._page.close()
                except Exception:  # noqa: BLE001
                    pass
            self._context = None

        self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        self._page = None
        self._owns_context = False
        self._created_new_page = False

    def open_notebook(self) -> None:
        """Open target notebook with retries."""

        def _open_once() -> None:
            page = self._require_page()
            self.logger.info("Opening notebook URL: %s", self.config.colab_notebook_url)
            page.goto(self.config.colab_notebook_url, wait_until="domcontentloaded")
            page.wait_for_timeout(2_000)
            self._click_confirmation_buttons()

        retry(
            operation=_open_once,
            attempts=max(1, self.config.launch_retry),
            wait_seconds=2.0,
            backoff=1.7,
            logger=self.logger,
            operation_name="open_notebook",
        )

    def ensure_runtime_connected(self) -> None:
        """Attempt to connect runtime if Connect/Reconnect button is visible."""

        page = self._require_page()
        if not self._is_connect_button_visible():
            self.logger.info("Runtime already seems connected.")
            return

        self.logger.info("Runtime appears disconnected. Attempting to connect.")
        for attempt in range(1, 6):
            connect_button = self._find_first_visible_button(self.CONNECT_BUTTON_PATTERNS, timeout_ms=1_000)
            if connect_button is None:
                self.logger.info("Connect button no longer visible.")
                return
            connect_button.click(timeout=self.config.selector_timeout_ms)
            page.wait_for_timeout(1_500)
            self._click_confirmation_buttons()
            page.wait_for_timeout(2_500)
            if not self._is_connect_button_visible():
                self.logger.info("Runtime connected.")
                return
            self.logger.warning("Connect attempt %d did not finish.", attempt)

        raise RuntimeError("Failed to connect Colab runtime after retries.")

    def trigger_run_all(self) -> None:
        """Trigger Run All action with multiple fallback strategies."""

        strategies = [
            ("keyboard_shortcut", self._trigger_run_all_by_shortcut),
            ("runtime_menu", self._trigger_run_all_by_menu),
            ("command_palette", self._trigger_run_all_by_palette),
        ]
        errors: list[str] = []

        for name, strategy in strategies:
            try:
                self.logger.info("Trying Run All strategy: %s", name)
                strategy()
                self._click_confirmation_buttons()
                if self._wait_for_running_indicator(timeout_seconds=20):
                    self.logger.info("Detected running indicator after strategy: %s", name)
                else:
                    self.logger.warning("No explicit running indicator after strategy: %s", name)
                return
            except Exception as exc:  # noqa: BLE001
                msg = f"{name}: {exc}"
                self.logger.warning("Run All strategy failed: %s", msg)
                errors.append(msg)

        raise RuntimeError(f"Unable to trigger Run All. Errors: {errors}")

    def launch_training(self, round_id: int) -> datetime:
        """Open notebook, connect runtime, and run all cells."""

        self.logger.info("Launching training for round %d", round_id)
        self.open_notebook()
        self.ensure_runtime_connected()
        self.trigger_run_all()
        started_at = datetime.now(timezone.utc)
        self.logger.info("Round %d launched at %s", round_id, started_at.isoformat())
        return started_at

    def _trigger_run_all_by_shortcut(self) -> None:
        page = self._require_page()
        page.keyboard.press("Control+F9")
        page.wait_for_timeout(1_500)

    def _trigger_run_all_by_menu(self) -> None:
        page = self._require_page()
        if not self._click_by_role_patterns(("menuitem", "button"), self.RUNTIME_MENU_PATTERNS, timeout_ms=2_000):
            raise RuntimeError("Runtime menu entry not found.")
        page.wait_for_timeout(600)
        if not self._click_by_role_patterns(("menuitem", "button"), self.RUN_ALL_PATTERNS, timeout_ms=2_000):
            raise RuntimeError("Run All menu item not found.")
        page.wait_for_timeout(1_000)

    def _trigger_run_all_by_palette(self) -> None:
        page = self._require_page()
        page.keyboard.press("Control+Shift+P")
        page.wait_for_timeout(1_000)

        search_inputs = [
            page.locator("input[placeholder*='Search']"),
            page.locator("input[aria-label*='Search']"),
            page.locator("input[type='text']"),
        ]
        target_input: Optional[Locator] = None
        for locator in search_inputs:
            if locator.count() < 1:
                continue
            candidate = locator.first
            if self._is_locator_visible(candidate, timeout_ms=1_000):
                target_input = candidate
                break

        if target_input is None:
            raise RuntimeError("Command palette input not found.")

        target_input.fill("run all")
        page.wait_for_timeout(300)
        page.keyboard.press("Enter")
        page.wait_for_timeout(1_000)

    def _wait_for_running_indicator(self, timeout_seconds: int) -> bool:
        page = self._require_page()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._click_confirmation_buttons():
                page.wait_for_timeout(300)
            if self._any_text_visible(self.RUNNING_INDICATOR_PATTERNS, timeout_ms=500):
                return True
            stop_button = self._find_first_visible_button(
                [re.compile(r"(interrupt|stop|cancel|中断|停止)", re.IGNORECASE)],
                timeout_ms=500,
            )
            if stop_button is not None:
                return True
            page.wait_for_timeout(1_000)
        return False

    def _click_confirmation_buttons(self) -> bool:
        page = self._require_page()
        clicked_any = False
        for _ in range(3):
            clicked_this_round = False
            for pattern in self.CONFIRM_BUTTON_PATTERNS:
                locator = page.get_by_role("button", name=pattern)
                if locator.count() < 1:
                    continue
                candidate = locator.first
                if not self._is_locator_visible(candidate, timeout_ms=500):
                    continue
                candidate.click(timeout=1_500)
                clicked_this_round = True
                clicked_any = True
                page.wait_for_timeout(400)
            if not clicked_this_round:
                break
        return clicked_any

    def _is_connect_button_visible(self) -> bool:
        return self._find_first_visible_button(self.CONNECT_BUTTON_PATTERNS, timeout_ms=600) is not None

    def _find_first_visible_button(self, patterns: list[re.Pattern[str]], timeout_ms: int) -> Optional[Locator]:
        page = self._require_page()
        for pattern in patterns:
            locator = page.get_by_role("button", name=pattern)
            if locator.count() < 1:
                continue
            candidate = locator.first
            if self._is_locator_visible(candidate, timeout_ms=timeout_ms):
                return candidate
        return None

    def _click_by_role_patterns(
        self,
        roles: tuple[str, ...],
        patterns: list[re.Pattern[str]],
        timeout_ms: int,
    ) -> bool:
        page = self._require_page()
        for role in roles:
            for pattern in patterns:
                locator = page.get_by_role(role, name=pattern)
                if locator.count() < 1:
                    continue
                candidate = locator.first
                if not self._is_locator_visible(candidate, timeout_ms=timeout_ms):
                    continue
                candidate.click(timeout=timeout_ms)
                return True
        return False

    def _any_text_visible(self, patterns: list[re.Pattern[str]], timeout_ms: int) -> bool:
        page = self._require_page()
        for pattern in patterns:
            locator = page.get_by_text(pattern)
            if locator.count() < 1:
                continue
            if self._is_locator_visible(locator.first, timeout_ms=timeout_ms):
                return True
        return False

    @staticmethod
    def _is_locator_visible(locator: Locator, timeout_ms: int) -> bool:
        try:
            locator.wait_for(state="visible", timeout=timeout_ms)
            return True
        except PlaywrightTimeoutError:
            return False

    def _handle_dialog(self, dialog: Dialog) -> None:
        self.logger.info("Auto-accepting browser dialog: %s", dialog.message)
        dialog.accept()

    def _require_page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Playwright page is not initialized. Call start() first.")
        return self._page

    def _start_by_persistent_context(self) -> None:
        assert self._playwright is not None
        self.logger.info("Starting Playwright persistent context at %s", self.config.playwright_profile_dir)

        launch_kwargs: dict[str, object] = {
            "user_data_dir": str(self.config.playwright_profile_dir),
            "headless": self.config.playwright_headless,
            "viewport": {"width": 1600, "height": 900},
            "args": ["--start-maximized"],
        }
        if self.config.playwright_channel:
            launch_kwargs["channel"] = self.config.playwright_channel

        self._context = self._playwright.chromium.launch_persistent_context(**launch_kwargs)
        had_pages = bool(self._context.pages)
        self._page = self._context.pages[0] if had_pages else self._context.new_page()
        self._owns_context = True
        self._created_new_page = not had_pages

    def _start_by_attach_cdp(self) -> None:
        assert self._playwright is not None
        cdp_url = (self.config.playwright_cdp_url or "").strip()
        if not cdp_url:
            raise RuntimeError("PLAYWRIGHT_CDP_URL is required when PLAYWRIGHT_ATTACH_EXISTING_CHROME=true")

        self._ensure_cdp_endpoint_reachable(cdp_url)
        self.logger.info("Attaching Playwright to existing Chrome via CDP: %s", cdp_url)
        self._browser = self._playwright.chromium.connect_over_cdp(cdp_url)
        if not self._browser.contexts:
            raise RuntimeError("No browser contexts found via CDP attachment.")

        self._context = self._browser.contexts[0]
        if self.config.playwright_attach_new_tab or not self._context.pages:
            self._page = self._context.new_page()
            self._created_new_page = True
        else:
            self._page = self._context.pages[0]
            self._created_new_page = False
        self._owns_context = False

    def _ensure_cdp_endpoint_reachable(self, cdp_url: str) -> None:
        version_url = cdp_url.rstrip("/") + "/json/version"
        try:
            with urlopen(version_url, timeout=3) as resp:  # nosec B310
                status = getattr(resp, "status", 200)
                if int(status) >= 400:
                    raise RuntimeError(f"CDP endpoint returned HTTP {status}: {version_url}")
        except URLError as exc:
            raise RuntimeError(
                "Cannot reach Chrome CDP endpoint. "
                f"Expected {version_url}. Start Chrome with --remote-debugging-port=9222."
            ) from exc
