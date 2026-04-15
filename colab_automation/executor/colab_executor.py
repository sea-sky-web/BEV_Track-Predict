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
        """连接Colab运行时（兼容新版Colab，修复下拉菜单问题）
        
        Args:
            gpu_type: GPU类型，支持 "T4", "P100", "A100", "A00"
            high_ram: 是否选择高RAM
        
        Returns:
            是否成功连接
        """
        self.logger.info(f"尝试连接运行时: GPU={gpu_type}, HighRAM={high_ram}")
        
        try:
            # 等待页面完全加载（2-3秒）
            self.logger.info("等待页面加载完成...")
            time.sleep(3)
            
            # 点击连接按钮（尝试多种选择器）
            connect_button = None
            button_selectors = [
                'colab-connect-button',
                'div.goog-menu-button:has-text("连接")',
                'button:has-text("连接")',
                '[role="button"]:has-text("连接")',
                'div:has-text("连接")',
            ]
            
            for selector in button_selectors:
                try:
                    button = self.page.locator(selector).first
                    if button.is_visible():
                        connect_button = button
                        self.logger.info(f"找到连接按钮，使用选择器: {selector}")
                        break
                except Exception:
                    continue
            
            if not connect_button:
                self.logger.error("未找到连接按钮")
                return False
            
            # 点击连接按钮（先尝试展开下拉菜单）
            self.logger.info("点击连接按钮展开下拉菜单")
            connect_button.click()
            time.sleep(2)
            
            # 在当前页面查找"更改运行时"选项（使用更广泛的选择器）
            self.logger.info("查找'更改运行时'选项...")
            change_runtime_option = None
            
            # 获取所有菜单项并检查文本
            menu_items = self.page.locator('[role="menuitem"]').all()
            self.logger.info(f"找到 {len(menu_items)} 个菜单项")
            
            for i, item in enumerate(menu_items):
                try:
                    text = item.inner_text()
                    self.logger.debug(f"菜单项{i}: {text}")
                    if "更改运行时" in text or "Change runtime" in text or "Runtime type" in text:
                        change_runtime_option = item
                        self.logger.info(f"找到'更改运行时'选项: {text}")
                        break
                except Exception:
                    continue
            
            # 如果没找到，尝试其他选择器
            if not change_runtime_option:
                option_selectors = [
                    'div.goog-menuitem:has-text("更改运行时")',
                    'div.goog-menuitem-content:has-text("更改运行时")',
                    'colab-runtime-menu-item',
                ]
                
                for selector in option_selectors:
                    try:
                        option = self.page.locator(selector).first
                        if option.is_visible():
                            text = option.inner_text()
                            change_runtime_option = option
                            self.logger.info(f"找到'更改运行时'选项，使用选择器: {selector}, 文本: {text}")
                            break
                    except Exception:
                        continue
            
            if change_runtime_option:
                self.logger.info("点击'更改运行时'选项")
                change_runtime_option.click()
                time.sleep(2)
            else:
                self.logger.warning("未找到'更改运行时'选项，尝试直接选择运行时")
            
            # 等待运行时配置对话框出现（使用更广泛的选择器）
            self.logger.info("等待运行时配置对话框...")
            dialog_found = False
            dialog_selectors = [
                'colab-runtime-selector',
                'div[role="dialog"]',
                '.modal-dialog',
                '[aria-label*="运行时"]',
                '[aria-label*="Runtime"]',
                'paper-dialog',
                '.mdc-dialog',
                'colab-dialog',
            ]
            start_time = time.time()
            timeout = 20  # 减少超时时间
            
            while time.time() - start_time < timeout:
                for selector in dialog_selectors:
                    try:
                        elements = self.page.locator(selector)
                        if elements.count() > 0 and elements.first.is_visible():
                            self.logger.info(f"找到运行时配置对话框，使用选择器: {selector}")
                            dialog_found = True
                            break
                    except Exception:
                        continue
                if dialog_found:
                    break
                time.sleep(1)
                elapsed = int(time.time() - start_time)
                self.logger.info(f"等待对话框中... ({elapsed}s)")
            
            if not dialog_found:
                self.logger.warning("未找到运行时配置对话框，尝试直接连接")
                # 如果找不到对话框，尝试直接点击连接按钮
                try:
                    connect_button.click()
                    time.sleep(5)
                    if self.is_runtime_connected():
                        self.logger.info("直接连接成功")
                        return True
                except Exception as e:
                    self.logger.error(f"直接连接失败: {e}")
                return False
            
            # 选择硬件加速器为GPU
            self.logger.info("选择硬件加速器为GPU...")
            accelerator_select = None
            accelerator_selectors = [
                'select#accelerator',
                'select[name="accelerator"]',
                'colab-select[name="accelerator"]',
            ]
            
            for selector in accelerator_selectors:
                try:
                    select = self.page.locator(selector).first
                    if select.is_visible():
                        accelerator_select = select
                        self.logger.info(f"找到加速器选择框，使用选择器: {selector}")
                        break
                except Exception:
                    continue
            
            if accelerator_select:
                self.logger.info("选择GPU加速器")
                accelerator_select.select_option("GPU")
                time.sleep(1)
            else:
                self.logger.warning("未找到加速器选择框")
            
            # 选择GPU类型
            self.logger.info(f"选择GPU类型: {gpu_type}...")
            gpu_select = None
            gpu_selectors = [
                'select#gpu-type',
                'select[name="gpuType"]',
                'colab-select[name="gpuType"]',
            ]
            
            for selector in gpu_selectors:
                try:
                    select = self.page.locator(selector).first
                    if select.is_visible():
                        gpu_select = select
                        self.logger.info(f"找到GPU类型选择框，使用选择器: {selector}")
                        break
                except Exception:
                    continue
            
            if gpu_select:
                # 获取所有选项并选择匹配的
                options = gpu_select.locator('option').all()
                selected = False
                for option in options:
                    try:
                        option_text = option.inner_text()
                        if gpu_type.lower() in option_text.lower():
                            gpu_select.select_option(label=option_text)
                            self.logger.info(f"选择GPU类型: {option_text}")
                            selected = True
                            break
                    except Exception:
                        continue
                
                if not selected:
                    self.logger.warning(f"未找到{gpu_type}，选择第一个GPU选项")
                    gpu_select.select_option(index=1)
                time.sleep(1)
            else:
                self.logger.warning("未找到GPU类型选择框")
            
            # 选择高RAM（如果可用）
            if high_ram:
                self.logger.info("选择高RAM...")
                high_ram_checkbox = None
                checkbox_selectors = [
                    'input[type="checkbox"][name="highram"]',
                    'input[type="checkbox"]:has-text("高RAM")',
                    'input[type="checkbox"]:has-text("High RAM")',
                ]
                
                for selector in checkbox_selectors:
                    try:
                        checkbox = self.page.locator(selector).first
                        if checkbox.is_visible():
                            high_ram_checkbox = checkbox
                            self.logger.info(f"找到高RAM复选框，使用选择器: {selector}")
                            break
                    except Exception:
                        continue
                
                if high_ram_checkbox:
                    is_checked = high_ram_checkbox.is_checked()
                    if not is_checked:
                        self.logger.info("勾选高RAM选项")
                        high_ram_checkbox.click()
                        time.sleep(1)
                    else:
                        self.logger.info("高RAM已勾选")
                else:
                    self.logger.warning("未找到高RAM复选框")
            
            # 点击保存按钮
            self.logger.info("点击保存按钮...")
            save_button = None
            save_selectors = [
                'button:has-text("保存")',
                'button:has-text("Save")',
                'colab-dialog-footer button',
            ]
            
            for selector in save_selectors:
                try:
                    button = self.page.locator(selector).first
                    if button.is_visible():
                        save_button = button
                        self.logger.info(f"找到保存按钮，使用选择器: {selector}")
                        break
                except Exception:
                    continue
            
            if save_button:
                self.logger.info("点击保存按钮应用配置")
                save_button.click()
            else:
                self.logger.error("未找到保存按钮")
                return False
            
            # 等待连接完成
            self.logger.info("等待运行时连接完成...")
            start_time = time.time()
            max_wait = 300  # 5分钟
            
            while time.time() - start_time < max_wait:
                if self.is_runtime_connected():
                    self.logger.info("✅ 运行时连接成功")
                    return True
                time.sleep(5)
                self.logger.info(f"等待连接中... ({int(time.time() - start_time)}s)")
            
            self.logger.error("运行时连接超时")
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