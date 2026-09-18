"""精确匹配指标。

针对数字型答案做了「数值匹配」增强：模型输出 "23 × 47 = 1081" 这类
带推导过程的回答时，字符串比较会误判为失败，但数值比较能正确通过。
这是数学类评测的标准做法，也避免了把「答对但啰嗦」算成错。
"""

from __future__ import annotations

from src.datasets.schema import EvalCase
from src.llm.base import LLMResponse
from src.metrics.base import BaseMetric, MetricResult
from src.metrics.normalize import (
    extract_final_answer,
    extract_numbers,
    normalize_text,
    shorten,
    to_float,
)


class ExactMatchMetric(BaseMetric):
    name = "exact_match"

    def compute(self, case: EvalCase, response: LLMResponse) -> MetricResult:
        if case.expected is None:
            return MetricResult(self.name, 0.0, None, "用例未提供 expected，跳过")

        expected_text = str(case.expected)
        answer_text = extract_final_answer(response.text)

        # 路径一：数值匹配（仅当期望值本身是数字）
        # 只取「最后一个数值」而不是「任意一个数值」：数学题的题干会重复出现题目中的
        # 数字，若采用「任意命中」会把「抄了题目但没算」的回答误判为正确。
        expected_number = to_float(expected_text)
        if expected_number is not None:
            numbers = extract_numbers(response.text)
            if numbers and abs(numbers[-1] - expected_number) < 1e-6:
                return MetricResult(
                    self.name,
                    1.0,
                    True,
                    f"数值匹配命中 {expected_number:g}；回答={shorten(response.text)}",
                )

        # 路径二：归一化字符串比较
        passed = normalize_text(answer_text) == normalize_text(expected_text)
        return MetricResult(
            self.name,
            1.0 if passed else 0.0,
            passed,
            f"期望={shorten(expected_text)} | 实际={shorten(answer_text)}",
        )
