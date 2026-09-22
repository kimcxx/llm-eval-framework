"""按分类推导默认能力维度（dimension）。

逐条给用例写 dimension 太啰嗦，而分类名本身已经能说明任务性质：
抽取任务看格式、数学与问答看正确性、红队看安全。这里维护一张
「分类 → 默认维度」的表，让没写 dimension 的用例也有稳定分组。

优先级：**用例显式声明 > 分类默认映射 > untagged**。
分类再怎么像，也不该盖掉用例自己写的标签——那是数据集作者更精确的表达。
"""

from __future__ import annotations

from src.datasets.schema import UNTAGGED_DIMENSION

# 只放「分类名足以确定任务性质」的映射；拿不准的一律留空，交回给用例声明，
# 免得猜错之后没人发现——归到 untagged 至少在报告里是可见的一行。
DEFAULT_DIMENSION_BY_CATEGORY: dict[str, str] = {
    "json_extract": "format",
    "math_reasoning": "correctness",
    "qa_open": "correctness",
    "qa_zh": "correctness",
    "safety_redteam": "safety",
}


def resolve_dimension(category: str | None, declared: str | None = None) -> str:
    """决定一条用例最终的能力维度。

    declared 非空时直接采用（用例声明优先）；否则查分类映射；
    分类也没登记就落到 untagged，让「没标签」始终在报告里可见。
    """
    if declared is not None and str(declared).strip():
        return str(declared).strip().lower()

    key = str(category or "").strip().lower()
    return DEFAULT_DIMENSION_BY_CATEGORY.get(key, UNTAGGED_DIMENSION)
