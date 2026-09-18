"""LLM-as-a-Judge 指标测试。

裁判本身也是模型，因此这里重点验证「裁判不可靠时框架如何自保」：
- 裁判给出合法评分 → 正常判定
- 裁判输出无法解析 → 判定为「跳过」，而不是算模型失败
- 裁判调用异常 → 同上
"""

from __future__ import annotations

import pytest

from conftest import FailingLLM, FakeLLM
from src.metrics.judge import JudgeMetric


def judge_case(make_case):
    return make_case(
        prompt="请解释什么是神经网络。",
        expected="神经网络是受生物神经元启发的计算模型。",
        metrics=["judge"],
    )


class TestJudgeMetric:
    def test_high_score_passes(self, make_case, make_response) -> None:
        judge = FakeLLM(default='{"score": 5, "reason": "准确完整"}', name="judge")
        metric = JudgeMetric(judge, threshold=4.0)

        result = metric.compute(
            judge_case(make_case),
            make_response("神经网络是受生物神经元启发的计算模型，由多层节点组成。"),
        )

        assert result.passed is True
        assert result.score == pytest.approx(1.0)

    def test_low_score_fails(self, make_case, make_response) -> None:
        judge = FakeLLM(default='{"score": 2, "reason": "答非所问"}', name="judge")
        metric = JudgeMetric(judge, threshold=4.0)

        result = metric.compute(judge_case(make_case), make_response("今天天气不错。"))

        assert result.passed is False
        assert result.score == pytest.approx(0.25)  # (2-1)/4

    def test_boundary_score_at_threshold_passes(self, make_case, make_response) -> None:
        judge = FakeLLM(default='{"score": 4, "reason": "基本正确"}', name="judge")
        metric = JudgeMetric(judge, threshold=4.0)
        assert metric.compute(judge_case(make_case), make_response("x")).passed is True

    def test_out_of_range_score_is_clamped(self, make_case, make_response) -> None:
        judge = FakeLLM(default='{"score": 9, "reason": "越界"}', name="judge")
        metric = JudgeMetric(judge, threshold=4.0)
        assert metric.compute(judge_case(make_case), make_response("x")).score == pytest.approx(1.0)

    def test_unparseable_verdict_is_skipped(self, make_case, make_response) -> None:
        """裁判胡言乱语时必须跳过，不能把评测基建故障算成模型能力差。"""
        judge = FakeLLM(default="我觉得还行吧", name="judge")
        metric = JudgeMetric(judge, threshold=4.0)

        result = metric.compute(judge_case(make_case), make_response("x"))

        assert result.passed is None
        assert "无法解析" in result.detail

    def test_judge_failure_is_skipped(self, make_case, make_response) -> None:
        judge = FailingLLM(retryable=False, name="judge")
        metric = JudgeMetric(judge, threshold=4.0)

        result = metric.compute(judge_case(make_case), make_response("x"))

        assert result.passed is None
        assert "裁判调用失败" in result.detail

    def test_falls_back_to_regex_when_json_missing(self, make_case, make_response) -> None:
        judge = FakeLLM(default="评分：5 分，理由：非常准确", name="judge")
        metric = JudgeMetric(judge, threshold=4.0)

        result = metric.compute(judge_case(make_case), make_response("x"))

        assert result.passed is True
