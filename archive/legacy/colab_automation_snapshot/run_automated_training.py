"""
完整的自动化训练循环脚本

工作流程：
1. 检查本地代码变更（可选）
2. 启动浏览器连接到已登录的Chrome
3. 打开指定的Colab笔记本
4. 连接运行时（选择A100 GPU + 高RAM）
5. 执行所有代码单元
6. 监控训练过程
7. 收集错误日志
8. 如果有错误，提示用户进行修复
9. 循环执行直到训练成功
"""

import logging
import subprocess
import sys
import time
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
            logging.FileHandler("automated_training.log", encoding="utf-8"),
        ]
    )
    return logging.getLogger(__name__)


def check_git_changes() -> bool:
    """检查是否有未提交的代码变更"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        return len(result.stdout.strip()) > 0
    except Exception as e:
        logging.error(f"检查Git状态失败: {e}")
        return False


def push_to_github(logger: logging.Logger) -> bool:
    """推送代码到GitHub"""
    try:
        logger.info("检查是否有未提交的变更...")
        
        if not check_git_changes():
            logger.info("没有未提交的变更，跳过推送")
            return True
        
        logger.info("添加所有变更...")
        subprocess.run(["git", "add", "."], check=True, cwd=Path(__file__).parent)
        
        logger.info("提交变更...")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        subprocess.run(
            ["git", "commit", "-m", f"Automated training update - {timestamp}"],
            check=True,
            cwd=Path(__file__).parent
        )
        
        logger.info("推送到GitHub...")
        subprocess.run(["git", "push"], check=True, cwd=Path(__file__).parent)
        
        logger.info("代码已成功推送到GitHub")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"推送代码失败: {e}")
        return False


def run_colab_execution(config, logger: logging.Logger) -> bool:
    """运行Colab执行（完整流程）"""
    error_collector = ErrorCollector(
        log_dir=config.error_log_dir,
        max_logs=config.max_error_logs,
        logger=logger
    )
    
    launcher = ColabPlaywrightLauncher(config, logger)
    
    try:
        with launcher.start():
            logger.info("打开Colab笔记本...")
            launcher.open_notebook()
            
            executor = ColabExecutor(launcher.page, logger)
            
            logger.info("等待笔记本加载完成...")
            if not executor.wait_for_notebook_ready(timeout=config.playwright_timeout):
                logger.error("笔记本加载超时")
                return False
            
            # 获取运行时信息
            runtime_info = executor.get_runtime_info()
            logger.info(f"当前运行时状态: {runtime_info}")
            
            # 如果未连接，尝试连接运行时
            if not runtime_info['connected']:
                logger.info("运行时未连接，尝试连接...")
                success = executor.connect_runtime(gpu_type="A100", high_ram=True)
                if not success:
                    logger.warning("运行时连接失败，请手动连接后继续")
                    # 等待用户手动连接
                    logger.info("等待15秒让用户手动连接运行时...")
                    time.sleep(15)
            
            # 再次检查运行时状态
            runtime_info = executor.get_runtime_info()
            if runtime_info['connected']:
                logger.info(f"✅ 运行时已连接: {runtime_info}")
            else:
                logger.warning("⚠️ 运行时仍未连接，继续执行可能会失败")
            
            # 执行所有代码单元
            logger.info("开始执行所有代码单元...")
            success, results = executor.run_all_cells()
            
            # 保存报告
            error_collector.save_report(results)
            
            if not success:
                errors = error_collector.collect_errors(results)
                error_collector.save_errors(errors)
                logger.error("部分单元执行失败，请查看错误日志")
                return False
            
            logger.info("✅ Colab执行完成")
            return True
            
    except Exception as e:
        logger.error(f"Colab执行异常: {e}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        return False


def main():
    """主函数"""
    logger = setup_logging()
    
    logger.info("=== 启动自动化训练循环 ===")
    
    # 加载配置
    config_path = Path(__file__).parent / "colab_automation" / ".env"
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    logger.info(f"使用笔记本: {config.colab_notebook_url}")
    
    # 循环训练直到成功
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        logger.info(f"\n=== 第 {retry_count + 1}/{max_retries} 次尝试 ===")
        
        # 1. 推送代码到GitHub（可选）
        logger.info("步骤1: 检查并推送代码到GitHub")
        push_to_github(logger)
        
        # 等待GitHub同步
        logger.info("等待GitHub同步...")
        time.sleep(5)
        
        # 2. 执行Colab训练
        logger.info("步骤2: 执行Colab训练")
        success = run_colab_execution(config, logger)
        
        if success:
            logger.info("✓ 训练成功完成！")
            print("\n" + "=" * 60)
            print("✓ 自动化训练循环成功完成！")
            print("=" * 60)
            return
        
        retry_count += 1
        
        if retry_count < max_retries:
            logger.info(f"等待 {30 * retry_count} 秒后重试...")
            time.sleep(30 * retry_count)
        else:
            logger.error("已达到最大重试次数，训练失败")
    
    logger.error("=== 自动化训练循环失败 ===")
    print("\n" + "=" * 60)
    print("✗ 自动化训练循环失败")
    print("请查看错误日志并修复问题后重新运行")
    print("错误日志位置: errors/")
    print("训练日志位置: automated_training.log")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()
