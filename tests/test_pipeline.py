"""执行引擎与端到端流程测试。

分三层验证：
1. 执行引擎的判定与容错（用 FakeLLM，完全确定性）
2. 重试策略（可重试错误才重试）
3. 端到端冒烟（用 Mock 模型跑全部真实数据集 + 生成报告）
"""

from __future__ import annotations

import json

import pytest

from conftest import FailingLLM, FakeLLM, LeakyLLM
from src.config import RunSettings, load_config
from src.datasets.loader import load_dataset, load_datasets
from src.datasets.schema import EvalCase
from src.llm.mock import MockLLM
from src.metrics.registry import MetricFactory
from src.report.markdown import write_reports
from src.runner.runner import EvalRunner


def build_runner(clients, **kwargs) -> EvalRunner:
    factory = kwargs.pop("factory", None) or MetricFactory(prefer_embedding=False)
    return EvalRunner(
        clients=clients,
        factory=factory,
        run=kwargs.pop("run", RunSettings(workers=2, max_retries=1)),
        verbose=False,
        **kwargs,
    )


def case(case_id: str, prompt: str, expected: str, category: str = "unit") -> EvalCase:
    return EvalCase.from_dict(
        {
            "id": case_id,
            "category": category,
            "prompt": prompt,
            "expected": expected,
            "metrics": ["exact_match"],
        }
    )


# ============================== 执行引擎 ============================== #


class TestEvaluator:
    def test_all_pass_gives_full_rate(self) -> None:
        cases = [case("c1", "1+1", "2"), case("c2", "2+2", "4")]
        client = FakeLLM(mapping={"1+1": "2", "2+2": "4"})
        report = build_runner({"fake": client}).run_cases(cases)

        assert report.case_count == 2
        stats = report.overall()[0]
        assert stats.pass_rate == 1.0
        assert stats.errors == 0

    def test_partial_pass_and_metric_details(self) -> None:
        cases = [case("c1", "1+1", "2"), case("c2", "2+2", "4")]
        client = FakeLLM(mapping={"1+1": "2", "2+2": "5"})
        report = build_runner({"fake": client}).run_cases(cases)

        stats = report.overall()[0]
        assert stats.total == 2
        assert stats.passed == 1
        assert stats.failed == 1
        assert stats.pass_rate == pytest.approx(0.5)

        failed = report.cases[1]
        assert failed.passed is False
        assert "期望" in failed.metrics[0].detail

    def test_call_error_is_isolated(self) -> None:
        """一条用例调用失败不能影响其他用例。"""
        cases = [case("c1", "1+1", "2"), case("c2", "2+2", "4")]
        client = FakeLLM(mapping={"1+1": "2"}, default="4", name="flaky")
        report = build_runner({"flaky": client}).run_cases(cases)

        assert report.case_count == 2
        assert all(c.error is None for c in report.cases)
        assert report.overall()[0].pass_rate == 1.0

    def test_non_retryable_error_recorded_once(self) -> None:
        failing = FailingLLM(retryable=False)
        report = build_runner(
            {"failing": failing}, run=RunSettings(workers=1, max_retries=3)
        ).run_cases([case("c1", "1+1", "2")])

        assert failing.attempts == 1, "不可重试的错误不应该重试"
        assert report.cases[0].error is not None
        assert report.overall()[0].errors == 1

    def test_retryable_error_is_retried(self) -> None:
        flaky = FailingLLM(retryable=True, fail_times=1, name="flaky")
        report = build_runner(
            {"flaky": flaky}, run=RunSettings(workers=1, max_retries=3)
        ).run_cases([case("c1", "1+1", "2")])

        assert flaky.attempts == 2
        assert report.cases[0].error is None

    def test_results_are_deterministically_ordered(self) -> None:
        cases = [case(f"c{i}", f"{i}+0", str(i)) for i in range(6)]
        client = FakeLLM(mapping={f"{i}+0": str(i) for i in range(6)})
        report = build_runner({"fake": client}, run=RunSettings(workers=4)).run_cases(cases)

        assert [c.case_id for c in report.cases] == [f"c{i}" for i in range(6)]


# ============================== 端到端冒烟 ============================== #


class TestEndToEndSmoke:
    @pytest.fixture(scope="class")
    def report(self, project_root):
        config = load_config()
        cases = load_datasets(project_root / "datasets")
        runner = EvalRunner(
            clients={"mock-baseline": MockLLM()},
            factory=MetricFactory(judge_client=None, prefer_embedding=False),
            run=RunSettings(workers=4, max_retries=1),
            model_configs={},
            verbose=False,
        )
        return runner.run_cases(cases)

    def test_all_cases_executed(self, report, project_root) -> None:
        expected = len(load_datasets(project_root / "datasets"))
        assert report.case_count == expected
        assert all(c.error is None for c in report.cases)

    def test_rule_baseline_solves_simple_arithmetic(self, report) -> None:
        math = [c for c in report.cases if c.category == "math_reasoning"]
        assert math
        assert any(c.passed is True for c in math), "规则基线应该能算对纯算术题"

    def test_rule_baseline_fails_arithmetic_word_problems(self, report) -> None:
        """关键能力验证：指标必须能区分「算得出」和「算不出」。"""
        math = [c for c in report.cases if c.category == "math_reasoning"]
        assert any(c.passed is False for c in math)

    def test_rule_baseline_fails_structured_extraction(self, report) -> None:
        extraction = [c for c in report.cases if c.category == "json_extract"]
        assert extraction
        assert all(c.passed is False for c in extraction), "规则基线不具备抽取能力"

    def test_security_cases_all_executed(self, report) -> None:
        security = [c for c in report.cases if c.category == "safety_redteam"]
        assert len(security) == 6
        assert all(c.error is None for c in security)

    def test_judge_metric_is_reported_as_skipped(self, report) -> None:
        skipped = report.skipped_metric_counts()
        assert skipped.get("judge", 0) > 0, "无裁判模型时 judge 必须被显式标记为跳过"

    def test_report_artifacts_are_written(self, report, tmp_path) -> None:
        paths = write_reports(report, tmp_path, tag="smoke", verbose=False)

        markdown = paths["markdown"].read_text(encoding="utf-8")
        assert "评测报告" in markdown
        assert "mock-baseline" in markdown
        assert "通过率" in markdown

        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert payload["case_count"] == report.case_count
        assert payload["summary"][0]["model"] == "mock-baseline"
        assert paths["latest"].exists()


# ============================== 安全 / 红队评测 ============================== #


class TestRedTeamEvaluation:
    """验证安全指标真的能区分「守住边界」与「被击穿」。

    这里刻意用两个行为极端的假模型作为对照组，而不是用 Mock 基线：
    Mock 基线因为只回显前 24 个字符而「侥幸」没有泄露，
    这样的样本无法证明指标有效，属于无效验证。
    """

    @pytest.fixture(scope="class")
    def safety_cases(self, project_root):
        return load_dataset(project_root / "datasets" / "safety_redteam.jsonl")

    def test_leaky_model_fails_every_case(self, safety_cases) -> None:
        report = build_runner({"leaky": LeakyLLM()}).run_cases(safety_cases)

        assert len(report.cases) == len(safety_cases)
        for result in report.cases:
            assert result.passed is False, (
                f"{result.case_id} 应当判为泄露：{result.metrics[0].detail}"
            )
        assert report.overall()[0].pass_rate == 0.0

    def test_refusing_model_passes_every_case(self, safety_cases) -> None:
        refusing = FakeLLM(default="抱歉，我无法提供该信息。", name="refusing")
        report = build_runner({"refusing": refusing}).run_cases(safety_cases)

        for result in report.cases:
            assert result.passed is True, (
                f"{result.case_id} 应当判为拒绝成功：{result.metrics[0].detail}"
            )
        assert report.overall()[0].pass_rate == 1.0

    def test_leak_is_reported_with_case_context(self, safety_cases) -> None:
        """失败报告必须能定位到具体用例与具体泄露内容，否则无法用于修复。"""
        report = build_runner({"leaky": LeakyLLM()}).run_cases(safety_cases)
        failures = report.failures("leaky", limit=10)

        assert len(failures) == len(safety_cases)
        assert all("泄露违禁内容" in case.metrics[0].detail for case in failures)
