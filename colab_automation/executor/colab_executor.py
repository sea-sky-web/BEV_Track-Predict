"""Colab笔记本执行器"""

import logging
import time
from typing import List, Dict, Optional, Tuple

from playwright.sync_api import Page, Locator


class ColabExecutor:
    """Colab笔记本自动化执行器"""
    
    def __init__(self, page: Page, logger: Optional[logging.Logger] = None):
        self.page = page
        self.logger = logger or logging.getLogger(__name__)
        self.execution_results: List[Dict] = []
    
    def wait_for_notebook_ready(self, timeout: int = 300000):
        """等待笔记本完全加载（兼容新版Colab，使用轮询方式）"""
        self.logger.info("等待笔记本就绪...")
        
        start_time = time.time()
        check_interval = 2  # 检查间隔（秒）
        timeout_seconds = timeout / 1000
        
        # 需要检查的元素列表
        required_elements = [
            {'selector': 'colab-notebook-toolbar', 'name': '笔记本工具栏'},
            {'selector': 'colab-scroller', 'name': '笔记本滚动区域'},
            {'selector': 'colab-connect-button', 'name': '连接按钮'},
        ]
        
        # 已就绪的元素
        ready_elements = set()
        
        while time.time() - start_time < timeout_seconds:
            elapsed = int(time.time() - start_time)
            
            # 检查每个元素
            for element in required_elements:
                selector = element['selector']
                name = element['name']
                
                if name not in ready_elements:
                    try:
                        if self.page.locator(selector).first.is_visible():
                            ready_elements.add(name)
                            self.logger.info(f"✅ {name} 就绪")
                    except Exception:
                        pass
            
            # 检查进度
            progress = len(ready_elements) / len(required_elements) * 100
            self.logger.info(f"等待中... ({elapsed}s) [{len(ready_elements)}/{len(required_elements)}] {progress:.1f}%")
            
            # 所有元素都就绪
            if len(ready_elements) == len(required_elements):
                self.logger.info("✅ 笔记本结构就绪")
                return True
            
            # 检查页面标题是否包含错误信息
            try:
                title = self.page.title()
                if "Error" in title or "error" in title.lower():
                    self.logger.warning(f"页面标题包含错误: {title}")
            except Exception:
                pass
            
            # 检查是否需要登录
            try:
                login_forms = self.page.locator('input[type="email"]').count()
                if login_forms > 0:
                    self.logger.warning("⚠️ 检测到需要登录，请在浏览器中手动登录")
            except Exception:
                pass
            
            time.sleep(check_interval)
        
        # 超时处理
        self.logger.error(f"❌ 等待笔记本就绪超时 ({timeout_seconds}s)")
        self.logger.error(f"已就绪的元素: {ready_elements}")
        self.logger.error(f"未就绪的元素: {set(e['name'] for e in required_elements) - ready_elements}")
        return False
    
    def is_runtime_connected(self) -> bool:
        """检查运行时是否已连接"""
        try:
            # 检查连接状态指示器
            status = self.page.locator('colab-connect-button').first
            if status.is_visible():
                button_text = status.inner_text()
                self.logger.info(f"运行时状态按钮文本: {button_text}")
                if '已连接' in button_text or 'Connected' in button_text:
                    return True
            return False
        except Exception as e:
            self.logger.error(f"检查运行时状态失败: {e}")
            return False
    
    def connect_runtime(self, gpu_type: str = "A100", high_ram: bool = True) -> bool:
        """连接Colab运行时（简化版本，提示用户手动配置）
        
        Args:
            gpu_type: GPU类型，支持 "T4", "P100", "A100", "A00"
            high_ram: 是否选择高RAM
        
        Returns:
            是否成功连接
        """
        self.logger.info(f"尝试连接运行时: GPU={gpu_type}, HighRAM={high_ram}")
        
        try:
            # 等待页面完全加载
            time.sleep(3)
            
            # 检查当前运行时状态
            if self.is_runtime_connected():
                self.logger.info("运行时已连接")
                return True
            
            # 提示用户手动配置运行时
            self.logger.warning("=" * 60)
            self.logger.warning("请在浏览器中手动配置运行时：")
            self.logger.warning(f"1. 点击'连接'按钮")
            self.logger.warning(f"2. 选择'更改运行时类型'")
            self.logger.warning(f"3. 选择硬件加速器: GPU")
            self.logger.warning(f"4. 选择GPU类型: {gpu_type}")
            self.logger.warning(f"5. 勾选高RAM选项（如果需要）")
            self.logger.warning(f"6. 点击保存")
            self.logger.warning("=" * 60)
            
            # 等待用户手动配置
            wait_time = 60  # 1分钟等待时间
            start_time = time.time()
            
            while time.time() - start_time < wait_time:
                if self.is_runtime_connected():
                    self.logger.info("✅ 用户已成功连接运行时")
                    return True
                elapsed = int(time.time() - start_time)
                remaining = wait_time - elapsed
                self.logger.info(f"等待用户配置运行时... ({remaining}s 剩余)")
                time.sleep(5)
            
            self.logger.warning("用户未在规定时间内配置运行时")
            return False
            
        except Exception as e:
            self.logger.error(f"连接运行时失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return False
    
    def get_runtime_info(self) -> Dict:
        """获取当前运行时信息"""
        info = {
            'connected': False,
            'gpu_type': None,
            'ram_type': None,
            'status': None,
        }
        
        try:
            # 获取连接状态
            info['connected'] = self.is_runtime_connected()
            
            # 获取运行时状态文本
            status_element = self.page.locator('colab-connect-button').first
            if status_element.is_visible():
                info['status'] = status_element.inner_text()
            
            # 获取GPU信息
            gpu_info = self.page.locator('colab-gpu-badge').first
            if gpu_info.is_visible():
                info['gpu_type'] = gpu_info.inner_text()
            
            self.logger.info(f"运行时信息: {info}")
            
        except Exception as e:
            self.logger.error(f"获取运行时信息失败: {e}")
        
        return info
    
    def get_cell_elements(self) -> List[Locator]:
        """获取所有代码单元元素（兼容新版Colab）"""
        # 新版Colab使用 div.cell.code.notebook-cell 选择器
        selectors = [
            'div.cell.code.notebook-cell',
            '.cell.code',
            'colab-scroller >>> .cell',
            'div.cell',
            '.notebook-cell',
        ]
        
        for selector in selectors:
            cells = self.page.locator(selector).all()
            if len(cells) > 0:
                self.logger.info(f"找到 {len(cells)} 个单元，使用选择器: {selector}")
                return cells
        
        self.logger.warning("未找到单元格元素")
        return []
    
    def is_code_cell(self, cell_element: Locator) -> bool:
        """判断是否为代码单元"""
        try:
            class_name = cell_element.get_attribute('class') or ''
            data_type = cell_element.get_attribute('data-type') or ''
            return 'code' in class_name.lower() or data_type.lower() == 'code'
        except Exception:
            return False
    
    def execute_cell(self, cell_index: int) -> Dict:
        """执行指定索引的代码单元"""
        result = {
            'cell_index': cell_index,
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0,
        }
        
        try:
            cells = self.get_cell_elements()
            if cell_index >= len(cells):
                result['error'] = f"单元索引超出范围: {cell_index} >= {len(cells)}"
                return result
            
            cell = cells[cell_index]
            
            if not self.is_code_cell(cell):
                self.logger.info(f"跳过非代码单元 {cell_index}")
                result['success'] = True
                return result
            
            self.logger.info(f"执行单元 {cell_index}...")
            
            # 点击运行按钮
            run_button = cell.locator('.run-button').first
            if not run_button.is_visible():
                result['error'] = f"未找到运行按钮"
                return result
            
            start_time = time.time()
            run_button.click()
            
            # 等待执行完成（等待执行图标消失）
            try:
                cell.locator('.running-indicator').wait_for(state='hidden', timeout=600000)
            except Exception:
                self.logger.warning("运行指示器等待超时，继续检查")
            
            result['execution_time'] = time.time() - start_time
            
            # 获取输出
            output_area = cell.locator('.output-area').first
            if output_area.is_visible():
                output_text = output_area.inner_text() or ''
                result['output'] = output_text
                
                # 检查是否有错误
                if output_area.locator('.output-error').is_visible() or \
                   'Error' in output_text or 'Traceback' in output_text:
                    result['success'] = False
                    result['error'] = output_text
                else:
                    result['success'] = True
            else:
                result['success'] = True
            
            self.logger.info(f"单元 {cell_index} 执行完成: {'成功' if result['success'] else '失败'}")
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"执行单元 {cell_index} 时发生异常: {e}")
        
        self.execution_results.append(result)
        return result
    
    def run_all_cells(self) -> Tuple[bool, List[Dict]]:
        """点击"运行全部"按钮执行所有单元（兼容新版Colab）"""
        self.execution_results = []
        
        try:
            # 找到并点击"运行全部"按钮（尝试多种选择器，包括DIV按钮）
            run_all_button = None
            selectors = [
                'colab-notebook-toolbar-run-button',
                'colab-toolbar-button#run-all-button',
                'button:has-text("运行全部")',
                'button:has-text("Run all")',
                '.run-all-button',
                'div.goog-menu-button:has-text("全部运行")',
                'div:has-text("全部运行")',
                '[role="button"]:has-text("运行全部")',
                '[role="button"]:has-text("全部运行")',
            ]
            
            for selector in selectors:
                try:
                    button = self.page.locator(selector).first
                    if button.is_visible():
                        run_all_button = button
                        self.logger.info(f"找到'运行全部'按钮，使用选择器: {selector}")
                        break
                except Exception:
                    continue
            
            if run_all_button:
                self.logger.info("点击'运行全部'按钮")
                run_all_button.click()
            else:
                self.logger.error("未找到'运行全部'按钮")
                return False, []
            
            # 等待所有单元执行完成（最多等待30分钟）
            self.logger.info("等待所有单元执行完成...")
            max_wait = 1800  # 30分钟
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                # 检查是否还有单元正在运行（尝试多种选择器）
                running_cells = self.page.locator('.running-indicator').count()
                if running_cells == 0:
                    running_cells = self.page.locator('[aria-label="正在运行"]').count()
                
                if running_cells == 0:
                    self.logger.info("所有单元执行完成")
                    # 获取所有单元的执行结果
                    cells = self.get_cell_elements()
                    for i, cell in enumerate(cells):
                        if self.is_code_cell(cell):
                            result = self._get_cell_result(i, cell)
                            self.execution_results.append(result)
                    
                    all_success = all(r['success'] for r in self.execution_results)
                    return all_success, self.execution_results
                
                elapsed = int(time.time() - start_time)
                self.logger.info(f"执行中... ({elapsed}s)")
                time.sleep(10)
            
            self.logger.error("执行超时")
            return False, self.execution_results
            
        except Exception as e:
            self.logger.error(f"执行所有单元失败: {e}")
            return False, []
    
    def _get_cell_result(self, cell_index: int, cell: Locator) -> Dict:
        """获取单个单元的执行结果"""
        result = {
            'cell_index': cell_index,
            'success': True,
            'output': '',
            'error': '',
        }
        
        try:
            output_area = cell.locator('.output-area').first
            if output_area.is_visible():
                output_text = output_area.inner_text() or ''
                result['output'] = output_text
                
                if output_area.locator('.output-error').is_visible() or \
                   'Error' in output_text or 'Traceback' in output_text:
                    result['success'] = False
                    result['error'] = output_text
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    def get_notebook_output(self) -> str:
        """获取笔记本所有输出"""
        output_parts = []
        
        for i, result in enumerate(self.execution_results):
            output_parts.append(f"=== 单元 {i} ===")
            output_parts.append(f"状态: {'成功' if result['success'] else '失败'}")
            if result['output']:
                output_parts.append(f"输出:\n{result['output']}")
            if result['error']:
                output_parts.append(f"错误:\n{result['error']}")
            if 'execution_time' in result:
                output_parts.append(f"执行时间: {result['execution_time']:.2f}s")
            output_parts.append("")
        
        return "\n".join(output_parts)
    
    def get_page_data(self) -> Dict:
        """获取页面数据"""
        data = {
            'url': self.page.url,
            'title': self.page.title(),
            'runtime': self.get_runtime_info(),
            'cell_count': len(self.get_cell_elements()),
            'timestamp': time.time(),
        }
        
        # 获取页面状态信息
        try:
            # 检查是否需要登录
            login_needed = self.page.locator('input[type="email"]').is_visible()
            data['login_needed'] = login_needed
            
            if login_needed:
                self.logger.warning("检测到需要登录Google账户")
                
        except Exception as e:
            self.logger.error(f"获取页面数据失败: {e}")
        
        return data