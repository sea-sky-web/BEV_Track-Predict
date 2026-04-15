"""Playwright浏览器启动器"""

import logging
import time
from contextlib import contextmanager
from typing import Optional

from playwright.sync_api import Playwright, sync_playwright, Browser, BrowserContext, Page


class ColabPlaywrightLauncher:
    """Colab Playwright浏览器启动器
    
    支持两种模式：
    1. 连接到已登录的Chrome（推荐）：通过CDP连接到用户手动启动的Chrome
    2. 启动新浏览器：直接启动新的浏览器实例（不推荐，无法登录Google）
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
    
    @contextmanager
    def start(self):
        """启动浏览器会话"""
        try:
            self.logger.info("启动Playwright...")
            self.playwright = sync_playwright().start()
            
            if self.config.playwright_attach_existing_chrome:
                # 模式1：连接到已登录的Chrome
                self._connect_to_existing_chrome()
            else:
                # 模式2：启动新浏览器（不推荐）
                self._launch_new_browser()
            
            yield self
            
        finally:
            self.stop()
    
    def _connect_to_existing_chrome(self):
        """连接到已登录的Chrome浏览器"""
        self.logger.info(f"连接到已登录的Chrome，CDP地址: {self.config.playwright_cdp_url}")
        
        browser_type = getattr(self.playwright, self.config.playwright_browser)
        
        try:
            self.browser = browser_type.connect_over_cdp(self.config.playwright_cdp_url)
            self.logger.info("成功连接到已登录的Chrome浏览器")
        except Exception as e:
            self.logger.error(f"连接到Chrome失败: {e}")
            self.logger.error("请确保已按照以下步骤操作：")
            self.logger.error("1. 关闭所有Chrome窗口")
            self.logger.error("2. 使用命令启动Chrome: chrome.exe --remote-debugging-port=9222")
            self.logger.error("3. 在Chrome中登录Google账户")
            self.logger.error("4. 再次运行本脚本")
            raise RuntimeError(f"无法连接到Chrome浏览器: {e}")
        
        # 获取现有页面或创建新页面
        contexts = self.browser.contexts
        if contexts:
            self.context = contexts[0]
            self.logger.info("使用现有浏览器上下文")
        else:
            self.context = self.browser.new_context()
            self.logger.info("创建新浏览器上下文")
        
        # 创建新页面（避免使用可能已关闭的现有页面）
        self.page = self.context.new_page()
        self.logger.info("创建新页面")
        
        self.page.set_default_timeout(self.config.playwright_timeout)
    
    def _launch_new_browser(self):
        """启动新浏览器（不推荐用于Google服务）"""
        self.logger.warning("⚠️ 正在启动新浏览器，Google服务可能需要手动登录")
        
        browser_type = getattr(self.playwright, self.config.playwright_browser)
        
        self.browser = browser_type.launch(
            headless=self.config.playwright_headless,
            args=[
                "--start-maximized",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ]
        )
        
        self.context = self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
        )
        
        self.page = self.context.new_page()
        self.page.set_default_timeout(self.config.playwright_timeout)
    
    def stop(self):
        """停止浏览器会话"""
        # 如果是连接到现有Chrome，不要关闭浏览器，只关闭我们创建的页面
        if self.config.playwright_attach_existing_chrome:
            if self.page and not self.page.is_closed():
                try:
                    # 不要关闭页面，保持用户的Chrome会话
                    self.logger.info("保持已登录的Chrome浏览器运行")
                except Exception:
                    pass
            # 不要关闭context和browser，保持用户的会话
        else:
            # 如果是我们启动的浏览器，正常关闭
            if self.page:
                try:
                    self.page.close()
                except Exception:
                    pass
            
            if self.context:
                try:
                    self.context.close()
                except Exception:
                    pass
            
            if self.browser:
                try:
                    self.browser.close()
                except Exception:
                    pass
        
        if self.playwright:
            try:
                self.playwright.stop()
            except Exception:
                pass
    
    def open_notebook(self, url: str = None):
        """打开Colab笔记本（使用轮询方式检查加载状态）"""
        notebook_url = url or self.config.colab_notebook_url
        self.logger.info(f"打开Colab笔记本: {notebook_url}")
        
        # 使用wait_until="load"而不是"networkidle"，避免等待所有网络请求完成
        self.page.goto(notebook_url, wait_until="load")
        
        # 轮询检查页面加载状态
        timeout_seconds = self.config.playwright_timeout / 1000
        check_interval = 2
        start_time = time.time()
        
        self.logger.info("等待笔记本加载...")
        
        while time.time() - start_time < timeout_seconds:
            elapsed = int(time.time() - start_time)
            
            # 检查页面是否加载完成（通过检查基本元素）
            try:
                # 检查页面标题
                title = self.page.title()
                if title and "Colab" in title:
                    self.logger.info(f"页面标题: {title}")
                
                # 检查基本元素是否存在
                toolbar_exists = self.page.locator('colab-notebook-toolbar').count() > 0
                scroller_exists = self.page.locator('colab-scroller').count() > 0
                
                if toolbar_exists and scroller_exists:
                    self.logger.info("✅ 笔记本加载完成")
                    return True
                    
            except Exception as e:
                self.logger.debug(f"检查页面状态时出错: {e}")
            
            self.logger.info(f"加载中... ({elapsed}s)")
            time.sleep(check_interval)
        
        self.logger.error(f"❌ 笔记本加载超时 ({timeout_seconds}s)")
        return False
