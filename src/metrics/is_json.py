"""输出是否为合法 JSON。

只回答一个问题：**下游程序能不能直接 json.loads 这段输出**。
字段对不对是 schema_match 的事，本指标完全不看内容。

与旧的 json_valid 的区别：json_valid 把「能否解析」和「字段是否正确」
混在一个 0~1 的分数里（4 个字段只对 3 个得 0.75，但判定为失败），
拆开之后，格式问题和内容问题可以分别归因。
"""

from __future__ import annotations

import json

from src.datasets.schema import EvalCase
from src.llm.base import LLMResponse
from src.metrics.base import BaseMetric, MetricResult
from src.metrics.normalize import shorten, strip_code_fence


class IsJsonMetric(BaseMetric):
    """输出整体能否被 json.loads 解析：能记 1，不能记 0。

    判定对象是「去掉 Markdown 代码围栏后的整段输出」。
    之所以宽容围栏、严格全文：```json 包裹是模型遵循指令的常见表现，
    而「前面加一句『好的，结果如下』再跟一个 JSON」并不算纯 JSON 输出，
    下游直接 json.loads 会失败，因此这里记 0（若想宽松统计，
    应该看 schema_match，它内部允许从解释性文字中抽出 JSON）。
    """

    name = "is_json"

    def compute(self, case: EvalCase, response: LLMResponse) -> MetricResult:
        text = strip_code_fence(response.text).strip()
        if not text:
            return MetricResult(self.name, 0.0, False, "输出为空，无法解析为 JSON")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return MetricResult(
                self.name,
                0.0,
                False,
                f"输出不是合法 JSON（{exc.msg}）：{shorten(text)}",
            )

        return MetricResult(self.name, 1.0, True, f"输出是合法 JSON（{type(parsed).__name__}）")

    def describe(self) -> str:
        return "is_json：输出整体可被 json.loads 解析记 1，否则记 0（只看格式，不校验字段内容）"
