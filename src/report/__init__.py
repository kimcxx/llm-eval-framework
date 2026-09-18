"""报告生成：把评测结果渲染成人类可读的 Markdown / 机器可读的 JSON。"""

from src.report.markdown import render_markdown, write_reports

__all__ = ["render_markdown", "write_reports"]
