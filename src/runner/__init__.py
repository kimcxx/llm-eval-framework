"""评测执行引擎：把「用例 × 模型」编排成可复现的评测运行。"""

from src.runner.results import CaseResult, EvalReport, GroupStats
from src.runner.runner import EvalRunner

__all__ = ["CaseResult", "EvalReport", "GroupStats", "EvalRunner"]
