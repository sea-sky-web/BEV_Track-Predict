"""详细调试Colab页面结构"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from colab_automation.config import load_config
from colab_automation.launcher.playwright_launcher import ColabPlaywrightLauncher


def debug_colab_structure():
    """详细调试Colab页面结构"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        config_path = Path(__file__).parent / "colab_automation" / ".env"
        config = load_config(config_path)

        launcher = ColabPlaywrightLauncher(config, logger)

        with launcher.start():
            logger.info("打开Colab笔记本...")
            launcher.open_notebook()
            
            time.sleep(5)
            
            # 1. 检查连接按钮
            logger.info("\n=== 1. 检查连接按钮 ===")
            connect_buttons = launcher.page.locator('colab-connect-button').all()
            logger.info(f"colab-connect-button数量: {len(connect_buttons)}")
            
            if connect_buttons:
                btn = connect_buttons[0]
                try:
                    inner_html = btn.inner_html()
                    logger.info(f"连接按钮HTML: {inner_html[:500]}")
                except Exception as e:
                    logger.error(f"获取连接按钮HTML失败: {e}")
            
            # 2. 检查所有按钮
            logger.info("\n=== 2. 检查所有按钮 ===")
            all_buttons = launcher.page.evaluate("""
                Array.from(document.querySelectorAll('button, *[role="button"]'))
                    .map(el => ({
                        tag: el.tagName,
                        text: el.textContent.trim().substring(0, 50),
                        ariaLabel: el.getAttribute('aria-label'),
                        className: el.className.substring(0, 100)
                    }))
            """)
            logger.info(f"找到 {len(all_buttons)} 个按钮")
            for i, btn in enumerate(all_buttons[:20]):
                logger.info(f"按钮{i}: tag={btn['tag']}, text='{btn['text']}', ariaLabel='{btn['ariaLabel']}', class='{btn['className']}'")
            
            # 3. 检查单元格结构
            logger.info("\n=== 3. 检查单元格结构 ===")
            cells = launcher.page.evaluate("""
                Array.from(document.querySelectorAll('div.cell, colab-cell, *[data-type]'))
                    .map(el => ({
                        tag: el.tagName,
                        className: el.className,
                        dataType: el.getAttribute('data-type'),
                        innerText: el.textContent.substring(0, 100)
                    }))
            """)
            logger.info(f"找到 {len(cells)} 个单元格相关元素")
            for i, cell in enumerate(cells[:10]):
                logger.info(f"单元格{i}: tag={cell['tag']}, class={cell['className']}, dataType={cell['dataType']}")
            
            # 4. 检查工具栏
            logger.info("\n=== 4. 检查工具栏 ===")
            toolbars = launcher.page.locator('colab-notebook-toolbar').all()
            logger.info(f"工具栏数量: {len(toolbars)}")
            
            if toolbars:
                toolbar = toolbars[0]
                try:
                    buttons = toolbar.locator('button, colab-toolbar-button').all()
                    logger.info(f"工具栏按钮数量: {len(buttons)}")
                    for i, btn in enumerate(buttons):
                        try:
                            text = btn.inner_text()
                            logger.info(f"工具栏按钮{i}: {text[:50]}")
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"获取工具栏按钮失败: {e}")
            
            # 5. 检查运行相关按钮
            logger.info("\n=== 5. 检查运行相关按钮 ===")
            run_buttons = launcher.page.evaluate("""
                Array.from(document.querySelectorAll('button'))
                    .filter(btn => btn.textContent.includes('运行') || btn.textContent.includes('Run'))
                    .map(btn => ({
                        text: btn.textContent.trim(),
                        className: btn.className
                    }))
            """)
            logger.info(f"运行相关按钮: {run_buttons}")
            
            # 6. 检查运行时状态
            logger.info("\n=== 6. 检查运行时状态 ===")
            runtime_info = launcher.page.evaluate("""
                const status = document.querySelector('colab-connect-button');
                if (status) {
                    return {
                        exists: true,
                        textContent: status.textContent.trim(),
                        innerHTML: status.innerHTML.substring(0, 500),
                        classList: Array.from(status.classList)
                    };
                }
                return { exists: false };
            """)
            logger.info(f"运行时状态元素: {runtime_info}")
            
            logger.info("\n=== 调试完成 ===")
            return True

    except Exception as e:
        logger.error(f"调试失败: {e}")
        import traceback
        logger.error(f"错误详情:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = debug_colab_structure()
    sys.exit(0 if success else 1)