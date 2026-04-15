"""错误日志收集器"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ErrorCollector:
    """错误日志收集器"""
    
    def __init__(self, log_dir: str = "./errors", max_logs: int = 10, 
                 logger: Optional[logging.Logger] = None):
        self.log_dir = Path(log_dir)
        self.max_logs = max_logs
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_errors(self, execution_results: List[Dict]) -> List[Dict]:
        """收集执行错误"""
        errors = []
        
        for result in execution_results:
            if not result.get('success') and result.get('error'):
                errors.append({
                    'cell_index': result['cell_index'],
                    'error': result['error'],
                    'output': result.get('output', ''),
                    'execution_time': result.get('execution_time', 0),
                    'timestamp': datetime.now().isoformat(),
                })
        
        return errors
    
    def save_errors(self, errors: List[Dict], filename: str = None) -> str:
        """保存错误到日志文件"""
        if not errors:
            self.logger.info("没有错误需要保存")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"errors_{timestamp}.json"
        
        log_path = self.log_dir / filename
        
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"错误日志已保存: {log_path}")
        
        # 清理旧日志
        self._cleanup_old_logs()
        
        return str(log_path)
    
    def _cleanup_old_logs(self):
        """清理旧日志文件"""
        log_files = sorted(
            self.log_dir.glob("errors_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if len(log_files) > self.max_logs:
            for old_file in log_files[self.max_logs:]:
                old_file.unlink()
                self.logger.info(f"清理旧日志: {old_file}")
    
    def generate_report(self, execution_results: List[Dict]) -> str:
        """生成执行报告"""
        errors = self.collect_errors(execution_results)
        
        report = []
        report.append("=" * 60)
        report.append("Colab训练自动化执行报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")
        
        # 统计信息
        total = len(execution_results)
        success = sum(1 for r in execution_results if r.get('success'))
        failed = total - success
        
        report.append(f"执行单元总数: {total}")
        report.append(f"成功: {success}")
        report.append(f"失败: {failed}")
        report.append("")
        
        # 错误详情
        if errors:
            report.append("错误详情:")
            report.append("-" * 40)
            
            for error in errors:
                report.append(f"\n单元索引: {error['cell_index']}")
                report.append(f"错误信息:")
                report.append(error['error'][:2000] if len(error['error']) > 2000 else error['error'])
                if error.get('output'):
                    report.append(f"\n输出内容:")
                    report.append(error['output'][:1000] if len(error['output']) > 1000 else error['output'])
        else:
            report.append("✓ 所有单元执行成功")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_report(self, execution_results: List[Dict]) -> str:
        """保存执行报告"""
        report = self.generate_report(execution_results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.log_dir / f"report_{timestamp}.txt"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        self.logger.info(f"执行报告已保存: {report_path}")
        return str(report_path)
