"""JSON 字段（schema）匹配指标。

回答的是内容问题：必需字段是否都在，取值是否和期望一致。
格式能否解析由 is_json 回答，两个指标互不重叠。

- score = 匹配字段数 / 必需字段数（0~1，是比例而非布尔）；
- passed = score >= threshold，threshold 从 run 配置读取，默认 1.0
  （默认口径与旧 json_valid 一致：全部字段都对才算通过）。
"""

from __future__ import annotations

from src.datasets.schema import EvalCase
from src.llm.base import LLMResponse
from src.metrics.base import BaseMetric, MetricResult
from src.metrics.normalize import extract_json, scalar_equal, shorten

# 浮点比例与阈值比较时的容差：4/4 应等于 1.0，避免 0.9999999 误判失败
_EPS = 1e-9


class SchemaMatchMetric(BaseMetric):
    """必需字段的存在性与取值正确性。"""

    name = "schema_match"

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def compute(self, case: EvalCase, response: LLMResponse) -> MetricResult:
        required = _required_keys(case)
        if not required:
            # 没有 schema 就无从判定；按框架约定返回 None（仅记录），
            # 既不虚增通过率，也不把「用例没写 required_keys」算成模型失败。
            return MetricResult(
                self.name,
                0.0,
                None,
                "用例未声明 required_keys/expected，无法判定字段匹配（is_json 仍可判定格式）",
            )

        parsed = extract_json(response.text)
        if parsed is None:
            return MetricResult(
                self.name,
                0.0,
                False,
                f"输出无法解析为 JSON，字段匹配 0/{len(required)}",
            )
        if not isinstance(parsed, dict):
            return MetricResult(
                self.name,
                0.0,
                False,
                f"期望 JSON 对象，实际为 {type(parsed).__name__}，字段匹配 0/{len(required)}",
            )

        expected_map = case.expected if isinstance(case.expected, dict) else None
        hits = 0
        notes: list[str] = []
        for key in required:
            if key not in parsed:
                notes.append(f"{key}=缺失")
                continue
            if expected_map is not None and key in expected_map:
                if scalar_equal(expected_map[key], parsed[key]):
                    hits += 1
                    notes.append(f"{key}=✓")
                else:
                    notes.append(
                        f"{key}={shorten(str(parsed[key]), 20)}≠{shorten(str(expected_map[key]), 20)}"
                    )
            else:
                # expected 里没有该键的期望值时，存在即得分
                hits += 1
                notes.append(f"{key}=✓")

        score = hits / len(required)
        passed = score >= self.threshold - _EPS
        detail = (
            f"字段匹配 {hits}/{len(required)}（阈值 {self.threshold:g}）；" + "，".join(notes)
        )
        return MetricResult(self.name, score, passed, detail)

    def describe(self) -> str:
        return (
            "schema_match：必需字段存在且取值相等记 1，score=匹配字段比例，"
            f"通过阈值 {self.threshold:g}（数值比较前先 strip，并去掉千分位逗号与货币单位）"
        )


def _required_keys(case: EvalCase) -> list[str]:
    """必需字段：优先 required_keys，否则退回 expected 的键。"""
    if case.required_keys:
        return [str(k) for k in case.required_keys]
    if isinstance(case.expected, dict):
        return list(case.expected.keys())
    return []
