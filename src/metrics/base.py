"""指标基类。

约定：
- score 统一为 0~1，越大越好，便于跨指标聚合；
- passed 为三态：True/False 参与通过率计算，None 表示「仅观测不判定」
  （例如裁判服务不可用、参考文本缺失时，不应该把这条算作模型失败）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.datasets.schema import EvalCase
from src.llm.base import LLMResponse


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool | None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "passed": self.passed,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[self.passed]
        return f"[{mark}] {self.name}={self.score:.3f} {self.detail}"


class BaseMetric(ABC):
    """所有指标的抽象基类。"""

    name: str = "base"
    counts_toward_pass: bool = True
    requires_judge: bool = False

    @abstractmethod
    def compute(self, case: EvalCase, response: LLMResponse) -> MetricResult:
        """计算该用例在此回答上的得分。"""
