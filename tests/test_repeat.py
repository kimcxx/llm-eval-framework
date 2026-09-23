"""重复执行（repeat）与请求节流测试。

repeat 的语义是「同一条用例独立跑 N 次」，所以这里重点验证三件事：
1. 每次结果都落盘（attempts 逐条可追溯）；
2. 通过判定与稳定率是两套口径，不能混为一谈；
3. 限流真的会拉长调用间隔，而不是只在配置里躺着。
"""

from __future__ import annotations

import json
import time

import pytest

from conftest import FakeLLM
from src.config import ConfigError, ModelConfig, RunSettings
from src.datasets.schema import EvalCase
from src.llm.registry import build_client
from src.llm.rate_limit import RateLimiter
from src.metrics.registry import MetricFactory
from src.report.markdown import render_markdown, write_reports
from src.runner.results import AttemptResult, CaseResult
from src.runner.runner import EvalRunner


def build_runner(clients, **kwargs) -> EvalRunner:
    return EvalRunner(
        clients=clients,
        factory=kwargs.pop("factory", None) or MetricFactory(prefer_embedding=False),
        run=kwargs.pop("run", RunSettings(workers=2, max_retries=1)),
        verbose=False,
        **kwargs,
    )


def case(case_id: str = "c1", prompt: str = "1+1", expected: str = "2") -> EvalCase:
    return EvalCase.from_dict(
        {
            "id": case_id,
            "category": "unit",
            "prompt": prompt,
            "expected": expected,
            "metrics": ["exact_match"],
        }
    )


class SequenceLLM(FakeLLM):
    """按调用顺序返回预设答案，用来模拟「时好时坏」的模型。

    answers 可以是列表（所有 prompt 共用）或 {prompt: 列表}（按用例分别编排）。
    """

    def __init__(self, answers: list[str] | dict[str, list[str]], name: str = "flaky") -> None:
        super().__init__(name=name)
        self.answers = answers
        # 按 prompt 计数：同一条用例重复执行时才按序列走，用例之间互不影响
        self.seen: dict[str, int] = {}

    def _invoke(self, system, prompt, temperature, max_tokens):
        sequence = self.answers.get(prompt, []) if isinstance(self.answers, dict) else self.answers
        index = self.seen.get(prompt, 0)
        self.seen[prompt] = index + 1
        text = sequence[min(index, len(sequence) - 1)] if sequence else ""
        from src.llm.base import LLMResponse

        return LLMResponse(text=text, model=self.model, prompt_tokens=10, completion_tokens=5)


# ============================== 配置 ============================== #


class TestRepeatConfig:
    def test_defaults_keep_single_run(self) -> None:
        settings = RunSettings()
        assert settings.repeat == 1
        assert settings.request_interval_s == 0.0

    def test_repeat_must_be_positive(self) -> None:
        with pytest.raises(ConfigError, match="repeat"):
            RunSettings(repeat=0)

    def test_interval_cannot_be_negative(self) -> None:
        with pytest.raises(ConfigError, match="request_interval_s"):
            RunSettings(request_interval_s=-1)

    def test_from_dict_parses_new_keys(self) -> None:
        settings = RunSettings.from_dict({"repeat": 3, "request_interval_s": 0.5})
        assert settings.repeat == 3
        assert settings.request_interval_s == 0.5

    def test_cli_overrides_are_parsed(self) -> None:
        from run_eval import parse_args

        args = parse_args(["--repeat", "3", "--request-interval", "0.5"])
        assert args.repeat == 3
        assert args.request_interval == 0.5

    def test_client_gets_limiter(self) -> None:
        client = build_client(ModelConfig(name="m", provider="mock"), RunSettings(request_interval_s=0.25))
        assert client.limiter is not None
        assert client.limiter.min_interval_s == 0.25


# ============================== 节流 ============================== #


class TestRateLimiter:
    def test_disabled_limiter_does_not_sleep(self) -> None:
        limiter = RateLimiter(0.0)
        started = time.monotonic()
        for _ in range(5):
            assert limiter.acquire() == 0.0
        assert time.monotonic() - started < 0.05

    def test_interval_is_enforced(self) -> None:
        limiter = RateLimiter(0.05)
        limiter.acquire()  # 第一次不等待
        started = time.monotonic()
        for _ in range(2):
            limiter.acquire()
        assert time.monotonic() - started >= 0.09

    def test_calls_are_serialized_under_concurrency(self) -> None:
        """并发下间隔同样生效：否则 workers 一多限流就形同虚设。"""
        import threading

        limiter = RateLimiter(0.05)
        stamps: list[float] = []

        def worker() -> None:
            limiter.acquire()
            stamps.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        started = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(stamps) == 4
        assert max(stamps) - started >= 0.14


# ============================== 执行引擎 ============================== #


class TestRepeatedExecution:
    def test_single_run_is_unchanged(self) -> None:
        report = build_runner({"fake": FakeLLM(mapping={"1+1": "2"})}).run_cases([case()])

        result = report.cases[0]
        assert result.repeat == 1
        assert len(result.attempts) == 1
        assert result.pass_count == 1
        assert result.passed is True
        assert report.stability == 1.0

    def test_each_attempt_is_persisted(self) -> None:
        client = SequenceLLM(["2", "2", "5"])
        report = build_runner(
            {"flaky": client}, run=RunSettings(workers=1, max_retries=1, repeat=3)
        ).run_cases([case()])

        result = report.cases[0]
        assert client.seen == {"1+1": 3}
        assert len(result.attempts) == 3
        assert [a.index for a in result.attempts] == [0, 1, 2]
        assert [a.passed for a in result.attempts] == [True, True, False]
        assert [a.response_text for a in result.attempts] == ["2", "2", "5"]
        assert all(a.metrics for a in result.attempts), "每次调用都要有自己的指标明细"

    def test_flaky_case_is_not_counted_as_passed(self) -> None:
        """2/3 通过不算通过：重复跑的意义就是不放过偶发失败。"""
        report = build_runner(
            {"flaky": SequenceLLM(["2", "2", "5"])},
            run=RunSettings(workers=1, max_retries=1, repeat=3),
        ).run_cases([case()])

        result = report.cases[0]
        assert result.pass_count == 2
        assert result.passed is False
        assert result.flaky is True
        assert result.case_stability == pytest.approx(2 / 3)

    def test_stable_case_passes(self) -> None:
        report = build_runner(
            {"stable": SequenceLLM(["2", "2", "2"])},
            run=RunSettings(workers=1, max_retries=1, repeat=3),
        ).run_cases([case()])

        result = report.cases[0]
        assert result.passed is True
        assert result.flaky is False
        assert result.case_stability == 1.0

    def test_global_stability_averages_every_case(self) -> None:
        """稳定率 = 全部用例的平均通过次数 / repeat。"""
        clients = {
            "stable": SequenceLLM({"1+1": ["2"] * 3, "2+2": ["4"] * 3}, name="stable"),
            "flaky": SequenceLLM({"1+1": ["2", "2", "5"], "2+2": ["4", "4", "9"]}, name="flaky"),
        }
        report = build_runner(
            clients, run=RunSettings(workers=1, max_retries=1, repeat=3)
        ).run_cases([case("c1"), case("c2", "2+2", "4")])

        # stable：2 条用例 × 3 次全过 → 6/6；flaky：2 条各 2/3 → 4/6
        assert report.stability == pytest.approx((6 + 4) / 12)
        by_model = {s.key: s for s in report.overall()}
        assert by_model["stable"].stability == 1.0
        assert by_model["flaky"].stability == pytest.approx(4 / 6)
        assert by_model["flaky"].flaky == 2

    def test_tokens_and_cost_are_summed_latency_averaged(self) -> None:
        report = build_runner(
            {"fake": FakeLLM(mapping={"1+1": "2"})},
            run=RunSettings(workers=1, max_retries=1, repeat=3),
        ).run_cases([case()])

        result = report.cases[0]
        assert result.prompt_tokens == 30  # 每次 10，共 3 次
        assert result.completion_tokens == 15
        assert report.total_calls == 3
        assert report.case_count == 1  # 用例记录数不随 repeat 膨胀

    def test_all_failed_attempts_become_call_error(self) -> None:
        from conftest import FailingLLM

        report = build_runner(
            {"failing": FailingLLM(retryable=False)},
            run=RunSettings(workers=1, max_retries=1, repeat=2),
        ).run_cases([case()])

        result = report.cases[0]
        assert result.error is not None
        assert "2/2" in result.error
        assert result.passed is False
        assert report.overall()[0].errors == 1

    def test_notes_disclose_repeat_and_interval(self) -> None:
        report = build_runner(
            {"fake": FakeLLM(mapping={"1+1": "2"})},
            run=RunSettings(workers=1, max_retries=1, repeat=3, request_interval_s=1.0),
        ).run_cases([case()])

        joined = "\n".join(report.notes)
        assert "3 次" in joined
        assert "请求间隔" in joined


# ============================== 数据结构降级 ============================== #


class TestCaseResultCompatibility:
    def test_manually_built_result_without_attempts(self) -> None:
        """没有 attempts（历史数据/手工构造）时保持单次判定语义。"""
        result = CaseResult(
            case_id="c", category="unit", dataset="d", model="m", prompt="p"
        )
        assert result.passed is None
        assert result.pass_count == 0
        assert result.attempt_total == 1

    def test_attempt_passed_is_none_without_judged_metric(self) -> None:
        """裁判不可用这类「无可判定指标」的情况，不计入稳定率分母。"""
        attempt = AttemptResult(index=0, response_text="x")
        assert attempt.passed is None
        assert AttemptResult(index=1, error="boom").passed is False


# ============================== 报告产物 ============================== #


class TestRepeatInReports:
    @pytest.fixture()
    def report(self):
        clients = {
            "flaky": SequenceLLM({"1+1": ["2", "2", "5"], "2+2": ["4", "4", "9"]}),
            "stable": SequenceLLM({"1+1": ["2"] * 3, "2+2": ["4"] * 3}, name="stable"),
        }
        return build_runner(
            clients, run=RunSettings(workers=1, max_retries=1, repeat=3)
        ).run_cases([case("c1"), case("c2", "2+2", "4")])

    def test_json_contains_per_case_counts(self, report) -> None:
        payload = json.loads(json.dumps(report.to_dict()))

        assert payload["repeat"] == 3
        assert payload["total_calls"] == 12
        assert "stability" in payload
        for item in payload["cases"]:
            assert len(item["attempts"]) == 3
            expected_count = 2 if item["model"] == "flaky" else 3
            assert item["pass_count"] == expected_count, item["case_id"]
            assert item["repeat"] == 3
        for row in payload["summary"]:
            assert "stability" in row and "flaky" in row

    def test_markdown_shows_pass_counts(self, report) -> None:
        markdown = render_markdown(report)

        assert "2/3" in markdown
        assert "用例稳定性明细" in markdown
        assert "全局稳定率" in markdown

    def test_csv_has_repeat_columns(self, report, tmp_path) -> None:
        paths = write_reports(report, tmp_path, tag="repeat", verbose=False)
        header = paths["csv"].read_text(encoding="utf-8-sig").splitlines()[0]

        assert "repeat" in header
        assert "pass_count" in header
