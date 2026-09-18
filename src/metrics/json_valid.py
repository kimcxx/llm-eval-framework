"""JSON 结构化输出合规指标。

Agent 与工程落地场景中，模型输出能否被下游程序消费，比"答得好不好"更致命。
本指标同时考察两件事：
1. 格式合规 —— 能否解析为 JSON；
2. 内容正确 —— 指定字段是否存在且取值正确。
"""

from __future__ import annotations

from src.datasets.schema import EvalCase
from src.llm.base import LLMResponse
from src.metrics.base import BaseMetric, MetricResult
from src.metrics.normalize import extract_json, normalize_text, shorten


class JsonValidMetric(BaseMetric):
    name = "json_valid"

    def compute(self, case: EvalCase, response: LLMResponse) -> MetricResult:
        parsed = extract_json(response.text)
        if parsed is None:
            return MetricResult(
                self.name, 0.0, False, f"输出不是合法 JSON：{shorten(response.text)}"
            )

        required = self._required_keys(case)
        if not required:
            return MetricResult(self.name, 1.0, True, "JSON 可解析")

        if not isinstance(parsed, dict):
            return MetricResult(
                self.name, 0.0, False, f"期望 JSON 对象，实际为 {type(parsed).__name__}"
            )

        hits = 0
        notes: list[str] = []
        for key in required:
            if key not in parsed:
                notes.append(f"{key}=缺失")
                continue
            actual = parsed[key]
            if isinstance(case.expected, dict) and key in case.expected:
                want = normalize_text(str(case.expected[key]))
                got = normalize_text(str(actual))
                if want and want == got:
                    hits += 1
                    notes.append(f"{key}=✓")
                else:
                    notes.append(f"{key}={shorten(str(actual), 20)}≠{shorten(str(case.expected[key]), 20)}")
            else:
                hits += 1
                notes.append(f"{key}=✓")

        score = hits / len(required)
        return MetricResult(
            self.name,
            score,
            hits == len(required),
            f"字段正确 {hits}/{len(required)}；" + "，".join(notes),
        )

    @staticmethod
    def _required_keys(case: EvalCase) -> list[str]:
        if case.required_keys:
            return [str(k) for k in case.required_keys]
        if isinstance(case.expected, dict):
            return list(case.expected.keys())
        return []
