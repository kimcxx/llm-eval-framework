"""裁判模型不应默认参与评测。

背景：`configs/models.yaml` 里裁判模型同时躺在 `models:` 列表里，只要它的 Key 可用，
`build_clients` 就会把它收进被测列表 —— 于是 qa_open 这类「仅由 judge 判定」的分类
变成模型给自己打分。约定：**默认把裁判模型从被测列表移除，只有 --models 显式点名才保留**，
且保留时必须在报告 notes 里披露自评风险。
"""

from __future__ import annotations

import pytest
from conftest import FakeLLM

import scripts.run_eval as run_eval
from src.config import AppConfig, ModelConfig, RunSettings
from src.metrics.registry import MetricFactory
from src.runner import EvalRunner
from src.runner.results import EvalReport

JUDGE = "glm-4.5-air"  # 真实配置里的裁判模型名


class TestDropJudgeFromEval:
    """drop_judge_from_eval：三态判定 + 原地修改被测列表。"""

    @staticmethod
    def _clients() -> dict:
        return {"m1": FakeLLM(name="m1"), JUDGE: FakeLLM(name=JUDGE)}

    def test_default_removes_judge(self) -> None:
        clients = self._clients()
        assert run_eval.drop_judge_from_eval(clients, JUDGE) == "removed"
        assert list(clients) == ["m1"]

    def test_explicit_keeps_judge(self) -> None:
        clients = self._clients()
        state = run_eval.drop_judge_from_eval(clients, JUDGE, {JUDGE})
        assert state == "kept"
        assert JUDGE in clients

    def test_other_model_explicit_still_removes(self) -> None:
        """显式指定了别的模型，不代表同意让裁判下场。"""
        clients = self._clients()
        assert run_eval.drop_judge_from_eval(clients, JUDGE, {"m1"}) == "removed"
        assert JUDGE not in clients

    def test_absent_when_judge_not_evaluated(self) -> None:
        clients = {"m1": FakeLLM(name="m1")}
        assert run_eval.drop_judge_from_eval(clients, JUDGE) == "absent"
        assert list(clients) == ["m1"]

    def test_absent_when_no_judge(self) -> None:
        clients = self._clients()
        assert run_eval.drop_judge_from_eval(clients, None) == "absent"
        assert JUDGE in clients  # 没有裁判模型时不动被测列表


class TestRunnerExtraNotes:
    """CLI 传入的额外备注要进报告 notes，且不覆盖指标自身的口径披露。"""

    def test_extra_notes_appended(self, make_case) -> None:
        case = make_case(expected="ok", metrics=["exact_match"])
        clients = {"m1": FakeLLM(name="m1", mapping={"问题": "ok"})}
        factory = MetricFactory(
            similarity_threshold=0.75,
            judge_threshold=4.0,
            schema_match_threshold=1.0,
            judge_client=None,
            prefer_embedding=False,
        )
        runner = EvalRunner(
            clients=clients,
            factory=factory,
            run=RunSettings(),
            verbose=False,
            extra_notes=["自评警告"],
        )
        report = runner.run_cases([case])

        assert report.notes[-1] == "自评警告"
        assert len(report.notes) > 1  # 指标口径披露仍在

    def test_no_extra_notes_by_default(self, make_case) -> None:
        case = make_case(expected="ok", metrics=["exact_match"])
        clients = {"m1": FakeLLM(name="m1", mapping={"问题": "ok"})}
        factory = MetricFactory(
            similarity_threshold=0.75,
            judge_threshold=4.0,
            schema_match_threshold=1.0,
            judge_client=None,
            prefer_embedding=False,
        )
        runner = EvalRunner(clients=clients, factory=factory, run=RunSettings(), verbose=False)
        report = runner.run_cases([case])

        assert "自评警告" not in report.notes


class _RunnerRecorder:
    """替身 EvalRunner：只记录 main() 传进来的参数。"""

    captured: dict | None = None

    def __init__(self, **kwargs) -> None:
        _RunnerRecorder.captured = kwargs

    def run_cases(self, cases) -> EvalReport:
        return EvalReport(
            started_at="",
            finished_at="",
            duration_s=0.0,
            models=list(_RunnerRecorder.captured["clients"]),
            cases=[],
        )


@pytest.fixture
def patched_main(monkeypatch):
    """把 main() 的外部依赖换成假的，只考察模型列表与 extra_notes。"""

    def _patch() -> None:
        monkeypatch.setattr(
            run_eval,
            "build_clients",
            lambda *a, **k: {"m1": FakeLLM(name="m1"), JUDGE: FakeLLM(name=JUDGE)},
        )
        monkeypatch.setattr(
            AppConfig, "get_judge", lambda self: ModelConfig(name=JUDGE, provider="mock", model="judge-1")
        )
        monkeypatch.setattr(run_eval, "EvalRunner", _RunnerRecorder)
        monkeypatch.setattr(run_eval, "write_reports", lambda *a, **k: None)
        monkeypatch.setattr(run_eval, "print_console_summary", lambda report: None)
        monkeypatch.setattr(run_eval, "apply_limit", lambda cases, limit: cases[:1])
        _RunnerRecorder.captured = None

    return _patch


class TestMainJudgeExclusion:
    def test_judge_excluded_by_default(self, patched_main) -> None:
        patched_main()
        run_eval.main([])

        captured = _RunnerRecorder.captured
        assert list(captured["clients"]) == ["m1"]
        assert captured["extra_notes"] == []

    def test_judge_kept_when_explicit(self, patched_main) -> None:
        patched_main()
        run_eval.main(["--models", JUDGE])

        captured = _RunnerRecorder.captured
        assert set(captured["clients"]) == {"m1", JUDGE}
        assert any("自评" in note for note in captured["extra_notes"])
