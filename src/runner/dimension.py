"""按分类推导默认能力维度（dimension）。

逐条给用例写 dimension 太啰嗦，而分类名本身已经能说明任务性质：
结构化抽取看指令遵循、数学与事实问答看准确性、开放题看相关性、红队看安全。
这里维护一张「分类 → 默认维度」的表，让没写 dimension 的用例也有稳定分组。

四类维度的划分口径（**按测法分，不按数据集分**）：

- ``correctness`` 准确性：有客观标准答案、断言可自动判定
- ``instruction_following`` 指令遵循：是否按要求的格式 / 结构输出
- ``safety`` 安全：越狱与拒答
- ``relevance`` 相关性：开放题，交给 LLM 裁判打分。它和客观断言不是同一种测法，
  单独成维度，否则「裁判松一点」会伪装成「模型答得准」

优先级：**用例显式声明 > 分类默认映射 > untagged**。
分类再怎么像，也不该盖掉用例自己写的标签——那是数据集作者更精确的表达。
"""

from __future__ import annotations

from src.datasets.schema import LEGACY_DIMENSION_ALIASES, UNTAGGED_DIMENSION

# 只放「分类名足以确定任务性质」的映射；拿不准的一律留空，交回给用例声明，
# 免得猜错之后没人发现——归到 untagged 至少在报告里是可见的一行。
DEFAULT_DIMENSION_BY_CATEGORY: dict[str, str] = {
    # JSON 抽取的核心考察点是「有没有按指令输出合法结构」，归指令遵循
    "json_extract": "instruction_following",
    "math_reasoning": "correctness",
    # 开放题走 LLM 裁判，单列「相关性」，不混进客观断言的准确性里
    "qa_open": "relevance",
    "qa_zh": "correctness",
    "safety_redteam": "safety",
}


def resolve_dimension(category: str | None, declared: str | None = None) -> str:
    """决定一条用例最终的能力维度。

    declared 非空时直接采用（用例声明优先）；否则查分类映射；
    分类也没登记就落到 untagged，让「没标签」始终在报告里可见。

    历史维度名（如 ``format``）在这里一并归一到新名，保证打标与聚合口径一致。
    """
    if declared is not None and str(declared).strip():
        value = str(declared).strip().lower()
        return LEGACY_DIMENSION_ALIASES.get(value, value)

    key = str(category or "").strip().lower()
    return DEFAULT_DIMENSION_BY_CATEGORY.get(key, UNTAGGED_DIMENSION)
