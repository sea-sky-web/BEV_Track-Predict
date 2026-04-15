"""Colab训练自动化主脚本"""

import logging
import sys
from pathlib import Path

from colab_automation.config import load_config
from colab_automation.launcher.playwright_launcher import ColabPlaywrightLauncher
from colab_automation.executor.colab_executor import ColabExecutor
from colab_automation.logger.error_collector import ErrorCollector


def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("colab_automation.log", encoding="utf-8"),
        ]
    )
    return logging.getLogger(__name__)


def run_automation(config_path: str = None) -> bool:
    """运行Colab自动化训练"""
    logger = setup_logging()
    
    try:
        logger.info("=== 启动Colab训练自动化 ===")
        
        # 加载配置
        logger.info("加载配置文件...")
        config = load_config(config_path)
        logger.info(f"Colab笔记本: {config.colab_notebook_url}")
        logger.info(f"浏览器模式: {'无头' if config.playwright_headless else '可视化'}")
        
        # 创建错误收集器
        error_collector = ErrorCollector(
            log_dir=config.error_log_dir,
            max_logs=config.max_error_logs,
            logger=logger
        )
        
        # 启动浏览器并执行
        launcher = ColabPlaywrightLauncher(config, logger)
        
        with launcher.start():
            logger.info("打开Colab笔记本...")
            launcher.open_notebook()
            
            # 创建执行器
            executor = ColabExecutor(launcher.page, logger)
            
            # 等待笔记本就绪
            logger.info("等待笔记本加载完成...")
            if not executor.wait_for_notebook_ready(timeout=config.playwright_timeout):
                logger.error("笔记本加载超时")
                return False
            
            # 执行所有单元
            logger.info("开始执行所有代码单元...")
            success, results = executor.execute_all_cells()
            
            # 收集错误并生成报告
            logger.info("收集执行结果...")
            error_collector.save_report(results)
            
            if not success:
                errors = error_collector.collect_errors(results)
                error_collector.save_errors(errors)
                logger.error("部分单元执行失败，请查看错误日志")
                return False
            
            logger.info("=== Colab训练自动化完成 ===")
            return True
        
    except Exception as e:
        logger.error(f"自动化执行失败: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        return False


def main():
    """主函数"""
    config_path = Path(__file__).parent / ".env"
    
    if not config_path.exists():
        print(f"错误: 配置文件不存在 {config_path}")
        sys.exit(1)
    
    success = run_automation(str(config_path))
    
    if success:
        print("\n✓ 自动化执行成功完成")
        sys.exit(0)
    else:
        print("\n✗ 自动化执行失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
