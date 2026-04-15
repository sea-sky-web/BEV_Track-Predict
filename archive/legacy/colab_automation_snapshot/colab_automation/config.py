"""配置加载模块"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ColabAutomationConfig:
    """Colab自动化配置类"""
    
    # GitHub配置
    github_repo_url: str
    github_branch: str
    
    # Colab配置
    colab_notebook_url: str
    
    # Playwright配置
    playwright_browser: str
    playwright_headless: bool
    playwright_timeout: int
    
    # CDP连接配置（连接到已登录的Chrome）
    playwright_attach_existing_chrome: bool
    playwright_cdp_url: str
    
    # 训练配置
    train_epochs: int
    train_batch: int
    train_max_frames: int
    
    # 日志配置
    error_log_dir: str
    max_error_logs: int


def load_config(config_path: str | Path = None) -> ColabAutomationConfig:
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / ".env"
    
    config_path = Path(config_path)
    
    # 从环境变量加载
    env = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env[key] = value
    
    # 优先使用系统环境变量
    def get_env(key: str, default: str = "") -> str:
        return os.environ.get(key, env.get(key, default))
    
    return ColabAutomationConfig(
        github_repo_url=get_env("GITHUB_REPO_URL", "https://github.com/sea-sky-web/BEV_Track-Predict.git"),
        github_branch=get_env("GITHUB_BRANCH", "main"),
        colab_notebook_url=get_env("COLAB_NOTEBOOK_URL", "https://colab.research.google.com/github/sea-sky-web/BEV_Track-Predict/blob/main/colab.ipynb"),
        playwright_browser=get_env("PLAYWRIGHT_BROWSER", "chromium"),
        playwright_headless=get_env("PLAYWRIGHT_HEADLESS", "false").lower() == "true",
        playwright_timeout=int(get_env("PLAYWRIGHT_TIMEOUT", "300000")),
        playwright_attach_existing_chrome=get_env("PLAYWRIGHT_ATTACH_EXISTING_CHROME", "true").lower() == "true",
        playwright_cdp_url=get_env("PLAYWRIGHT_CDP_URL", "http://localhost:9222"),
        train_epochs=int(get_env("TRAIN_EPOCHS", "10")),
        train_batch=int(get_env("TRAIN_BATCH", "1")),
        train_max_frames=int(get_env("TRAIN_MAX_FRAMES", "300")),
        error_log_dir=get_env("ERROR_LOG_DIR", "./errors"),
        max_error_logs=int(get_env("MAX_ERROR_LOGS", "10")),
    )
