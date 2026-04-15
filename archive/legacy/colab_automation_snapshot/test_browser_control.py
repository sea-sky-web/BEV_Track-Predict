"""测试 Colab 自动化浏览器控制功能"""

import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from colab_automation.config import load_config
from colab_automation.launcher.playwright_launcher import ColabPlaywrightLauncher
from colab_automation.executor.colab_executor import ColabExecutor


def test_browser_control():
    """测试浏览器控制功能"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("=== 开始测试 Colab 自动化浏览器控制功能 ===")

        config_path = Path(__file__).parent / "colab_automation" / ".env"
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return False

        logger.info("加载配置...")
        config = load_config(config_path)
        logger.info(f"Colab 笔记本 URL: {config.colab_notebook_url}")
        logger.info(f"浏览器模式: {'无头' if config.playwright_headless else '可视化'}")
        logger.info(f"CDP地址: {config.playwright_cdp_url}")

        logger.info("检查 Playwright 安装...")
        launcher = ColabPlaywrightLauncher(config, logger)

        logger.info("启动浏览器会话...")
        with launcher.start():
            logger.info("✅ 浏览器会话启动成功")

            logger.info("测试页面导航...")
            launcher.open_notebook()
            logger.info("✅ 页面导航成功")

            # 创建执行器
            executor = ColabExecutor(launcher.page, logger)

            # 等待笔记本就绪
            logger.info("等待笔记本加载...")
            executor.wait_for_notebook_ready(timeout=60000)
            logger.info("✅ 笔记本加载完成")

            # 获取页面数据
            logger.info("获取页面数据...")
            page_data = executor.get_page_data()
            logger.info(f"页面数据: {page_data}")
            logger.info("✅ 获取页面数据成功")

            # 获取运行时信息
            logger.info("获取运行时信息...")
            runtime_info = executor.get_runtime_info()
            logger.info(f"运行时信息: {runtime_info}")

            # 如果未连接，尝试连接运行时
            if not runtime_info['connected']:
                logger.info("运行时未连接，尝试连接...")
                success = executor.connect_runtime(gpu_type="A100", high_ram=True)
                if success:
                    logger.info("✅ 运行时连接成功")
                else:
                    logger.warning("⚠️ 运行时连接失败，请手动连接后继续测试")
            else:
                logger.info("✅ 运行时已连接")

            # 获取单元信息
            logger.info("获取单元信息...")
            cells = executor.get_cell_elements()
            logger.info(f"找到 {len(cells)} 个单元")

            # 测试运行全部按钮
            logger.info("测试运行全部按钮...")
            try:
                success, results = executor.run_all_cells()
                if success:
                    logger.info("✅ 运行全部成功")
                    logger.info(f"执行结果: {len(results)} 个单元")
                else:
                    logger.warning("⚠️ 运行全部失败或超时")
            except Exception as e:
                logger.error(f"运行全部时出错: {e}")

            logger.info("=== 浏览器控制功能测试完成 ===")
            return True

    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        import traceback
        logger.error(f"错误详情:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_browser_control()
    sys.exit(0 if success else 1)
