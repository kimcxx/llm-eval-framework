"""评测执行引擎：把「用例 × 模型」编排成可复现的评测运行。"""

from src.runner.dimension import DEFAULT_DIMENSION_BY_CATEGORY, resolve_dimension
from src.runner.results import CaseResult, EvalReport, GroupStats
from src.runner.runner import EvalRunner

__all__ = [
    "DEFAULT_DIMENSION_BY_CATEGORY",
    "CaseResult",
    "EvalReport",
    "GroupStats",
    "EvalRunner",
    "resolve_dimension",
]
