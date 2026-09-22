"""qa_open「只由 judge 判定」的判定口径测试。

背景：开放式问答没有标准答案，字面/字符相似度会把「换个说法但答对了」判成失败，
导致 qa_open 长期 0% 通过。现在的口径是：
- qa_open 分类只由 judge 判定（阈值 4/5）；
- similarity 仍然计算并写进指标明细，但不参与通过判定（passed=None，仅记录）；
- 其它分类的判定方式完全不变。
"""

from __future__ import annotations

import json

import pytest

from src.config import RunSettings
from src.datasets.schema import EvalCase
from src.metrics.registry import MetricFactory
from src.runner import EvalRunner

from conftest import FakeLLM  # noqa: E402  - 测试夹具里的确定性假模型


QA_OPEN_CASE = {
    "id": "open-001",
    "category": "qa_open",
    "prompt": "请解释分布式系统中的 CAP 定理。",
    "expected": "CAP 定理指出一致性、可用性和分区容错性三者不可兼得，网络分区不可避免时需在 C 与 A 之间取舍。",
    "metrics": ["similarity", "judge"],
    "source": "qa_open",
}

# 换个说法但答对了：字符相似度必然低，裁判应给高分
GOOD_BUT_DIFFERENT = "三者只能取二，分区一定会发生，所以要在一致性和可用性之间做选择。"

JUDGE_ONLY = ("qa_open",)


def judge_returning(score: int) -> FakeLLM:
    """裁判模型替身：忽略 prompt，固定返回指定分数。"""
    payload = json.dumps({"score": score, "reason": "测试用固定打分"}, ensure_ascii=False)
    return FakeLLM(default=payload, name="judge", model="judge-1")


def run_case(case: EvalCase, answer: str, judge_score: int | None = 5, judge_only=JUDGE_ONLY):
    judge = judge_returning(judge_score) if judge_score is not None else None
    factory = MetricFactory(
        judge_client=judge,
        prefer_embedding=False,
        judge_only_categories=judge_only,
    )
    runner = EvalRunner(
        clients={"m": FakeLLM(mapping={case.prompt: answer}, name="m")},
        factory=factory,
        run=RunSettings(workers=1, max_retries=1, judge_only_categories=judge_only),
        verbose=False,
    )
    report = runner.run_cases([case])
    return report.cases[0], report


class TestQaOpenJudgedByJudgeOnly:
    def test_similarity_low_but_judge_passes(self) -> None:
        """换个说法但答对了：相似度低不影响通过。"""
        case = EvalCase.from_dict(dict(QA_OPEN_CASE))
        result, _ = run_case(case, GOOD_BUT_DIFFERENT, judge_score=5)

        assert result.score_of("similarity") is not None
        assert result.score_of("similarity") < 0.75
        assert result.passed is True

    def test_similarity_still_recorded_not_judged(self) -> None:
        """similarity 仍在明细里，但不再参与判定（passed=None）。"""
        case = EvalCase.from_dict(dict(QA_OPEN_CASE))
        result, _ = run_case(case, GOOD_BUT_DIFFERENT, judge_score=5)

        similarity = next(m for m in result.metrics if m.name == "similarity")
        assert similarity.passed is None
        assert "仅记录" in similarity.detail
        assert similarity.score > 0  # 分数照旧写入报告

    def test_judge_below_threshold_still_fails(self) -> None:
        """裁判判定不达标，即使相似度很高也必须失败。"""
        case = EvalCase.from_dict(dict(QA_OPEN_CASE))
        result, _ = run_case(case, case.expected, judge_score=2)  # 照抄参考 → 相似度满分

        assert result.score_of("similarity") >= 0.99
        assert result.passed is False

    @pytest.mark.parametrize(("score", "expected"), [(4, True), (3, False), (5, True), (1, False)])
    def test_threshold_is_4_over_5(self, score: int, expected: bool) -> None:
        case = EvalCase.from_dict(dict(QA_OPEN_CASE))
        result, _ = run_case(case, GOOD_BUT_DIFFERENT, judge_score=score)
        assert result.passed is expected

    def test_judge_unavailable_case_is_skipped_not_failed(self) -> None:
        """裁判不可用时该用例「无可判定指标」→ 跳过，不算模型失败。"""
        case = EvalCase.from_dict(dict(QA_OPEN_CASE))
        result, report = run_case(case, GOOD_BUT_DIFFERENT, judge_score=None)

        assert result.passed is None
        stats = report.overall()[0]
        assert stats.skipped == 1
        assert stats.failed == 0


class TestOtherCategoriesUnchanged:
    def test_similarity_still_decides_for_other_categories(self) -> None:
        """其它分类：similarity 照旧参与判定。"""
        case = EvalCase.from_dict(
            {
                "id": "zh-001",
                "category": "qa_zh",
                "prompt": "中国的首都是哪里？",
                "expected": "北京",
                "metrics": ["similarity"],
                "source": "qa_zh",
            }
        )
        result, _ = run_case(case, "上海")

        similarity = next(m for m in result.metrics if m.name == "similarity")
        assert similarity.passed is False
        assert result.passed is False

    def test_other_category_high_similarity_passes(self) -> None:
        case = EvalCase.from_dict(
            {
                "id": "zh-002",
                "category": "qa_zh",
                "prompt": "中国的首都是哪里？",
                "expected": "北京",
                "metrics": ["similarity"],
                "source": "qa_zh",
            }
        )
        result, _ = run_case(case, "北京")
        assert result.passed is True


class TestConfiguration:
    def test_default_is_qa_open(self) -> None:
        assert RunSettings().judge_only_categories == ("qa_open",)

    def test_from_dict_accepts_yaml_list(self) -> None:
        settings = RunSettings.from_dict({"judge_only_categories": ["qa_open", "summary"]})
        assert settings.judge_only_categories == ("qa_open", "summary")

    def test_from_dict_accepts_single_string(self) -> None:
        settings = RunSettings.from_dict({"judge_only_categories": "qa_open"})
        assert settings.judge_only_categories == ("qa_open",)

    def test_from_dict_without_key_keeps_default(self) -> None:
        assert RunSettings.from_dict({"workers": 8}).judge_only_categories == ("qa_open",)

    def test_empty_list_disables_the_policy(self) -> None:
        settings = RunSettings.from_dict({"judge_only_categories": []})
        assert settings.judge_only_categories == ()


class TestDisclosure:
    def test_notes_mention_judge_only_categories(self) -> None:
        factory = MetricFactory(judge_client=judge_returning(5), judge_only_categories=JUDGE_ONLY)
        notes = factory.notes()
        assert any("qa_open" in note and "judge" in note for note in notes)

    def test_no_note_when_policy_empty(self) -> None:
        factory = MetricFactory(judge_client=None, judge_only_categories=())
        assert not factory.is_judge_only("qa_open")
