"""dimension（能力维度）字段与分组统计测试。

dimension 是纯标签，只影响报告的分组视图，**绝不参与通过/失败判定**。
这两点都要被测试锁死，否则后续很容易在重构中被"顺手"接进判定逻辑。
"""

from __future__ import annotations

import pytest

from conftest import FakeLLM
from src.config import RunSettings
from src.datasets.schema import DIMENSIONS, DatasetError, EvalCase
from src.metrics.registry import MetricFactory
from src.runner.results import dimension_label
from src.runner.runner import EvalRunner


def _runner(model: str = "fake") -> EvalRunner:
    return EvalRunner(
        clients={model: FakeLLM(mapping={"1+1": "2", "2+2": "4"}, name=model)},
        factory=MetricFactory(prefer_embedding=False),
        run=RunSettings(workers=1, max_retries=1),
        verbose=False,
    )


def _case(case_id: str, prompt: str, expected: str, dimension: str | None = None) -> EvalCase:
    raw: dict = {
        "id": case_id,
        "prompt": prompt,
        "expected": expected,
        "metrics": ["exact_match"],
    }
    if dimension is not None:
        raw["dimension"] = dimension
    return EvalCase.from_dict(raw)


class TestDimensionField:
    def test_defaults_to_none(self) -> None:
        assert EvalCase.from_dict({"id": "a", "prompt": "q"}).dimension is None

    @pytest.mark.parametrize("value", DIMENSIONS)
    def test_accepts_every_enum_value(self, value: str) -> None:
        case = EvalCase.from_dict({"id": "a", "prompt": "q", "dimension": value})
        assert case.dimension == value

    def test_normalizes_case_and_surrounding_whitespace(self) -> None:
        case = EvalCase.from_dict({"id": "a", "prompt": "q", "dimension": "  SAFETY "})
        assert case.dimension == "safety"

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_blank_means_untagged(self, value) -> None:
        case = EvalCase.from_dict({"id": "a", "prompt": "q", "dimension": value})
        assert case.dimension is None

    def test_rejects_unknown_value(self) -> None:
        """拼错的维度必须报错，否则会被静默算进 untagged 而没人发现。"""
        with pytest.raises(DatasetError, match="dimension 非法"):
            EvalCase.from_dict({"id": "a", "prompt": "q", "dimension": "safty"})

    def test_rejects_non_string(self) -> None:
        with pytest.raises(DatasetError, match="应为字符串"):
            EvalCase.from_dict({"id": "a", "prompt": "q", "dimension": 3})

    def test_is_not_collected_into_meta(self) -> None:
        """dimension 是正式字段，不应残留在 meta（否则会重复出现在 extra 字段里）。"""
        case = EvalCase.from_dict({"id": "a", "prompt": "q", "dimension": "safety"})
        assert "dimension" not in case.meta


class TestDimensionAggregation:
    def _cases(self) -> list[EvalCase]:
        return [
            _case("c1", "1+1", "2", "correctness"),
            _case("c2", "2+2", "4", "format"),
            _case("c3", "1+1", "2"),  # 未打标签
        ]

    def test_groups_in_enum_order_with_untagged_last(self) -> None:
        report = _runner().run_cases(self._cases())
        assert [g.key for g in report.by_dimension("fake")] == [
            "correctness",
            "format",
            "untagged",
        ]

    def test_untagged_group_collects_unlabelled_cases(self) -> None:
        report = _runner().run_cases(self._cases())
        untagged = report.by_dimension("fake")[-1]
        assert untagged.total == 1

    def test_is_isolated_per_model(self) -> None:
        clients = {
            "a": FakeLLM(mapping={"1+1": "2"}, name="a"),
            "b": FakeLLM(default="错", name="b"),
        }
        report = EvalRunner(
            clients=clients,
            factory=MetricFactory(prefer_embedding=False),
            run=RunSettings(workers=1, max_retries=1),
            verbose=False,
        ).run_cases([_case("c1", "1+1", "2", "correctness")])

        assert report.by_dimension("a")[0].passed == 1
        assert report.by_dimension("b")[0].passed == 0

    def test_dimension_is_only_a_label(self) -> None:
        """同一条用例无论有没有 dimension，判定结果必须完全一致。"""
        tagged = _runner().run_cases([_case("c1", "1+1", "2", "safety")])
        plain = _runner().run_cases([_case("c1", "1+1", "2")])

        assert tagged.cases[0].passed is True
        assert plain.cases[0].passed is True
        assert tagged.overall()[0].pass_rate == plain.overall()[0].pass_rate

    def test_failures_still_counted_inside_dimension(self) -> None:
        """标签不改变判定：维度组里的失败照常计入 failed / pass_rate。"""
        report = _runner().run_cases([_case("c1", "1+1", "999", "robustness")])
        stats = report.by_dimension("fake")[0]

        assert stats.key == "robustness"
        assert stats.failed == 1
        assert stats.pass_rate == 0.0

    def test_to_dict_exposes_dimensions_and_case_dimension(self) -> None:
        payload = _runner().run_cases(self._cases()).to_dict()
        rows = payload["dimensions"]["fake"]

        assert rows[-1]["dimension"] == "untagged"
        assert rows[-1]["label"] == "未标注"
        assert {r["dimension"] for r in rows} == {"correctness", "format", "untagged"}
        assert payload["cases"][0]["dimension"] == "correctness"

    def test_untagged_survives_round_trip_when_no_labels_at_all(self) -> None:
        """存量数据集完全没打标签时，报告应只出现 untagged 一行而不是空表。"""
        payload = _runner().run_cases([_case("c1", "1+1", "2")]).to_dict()
        rows = payload["dimensions"]["fake"]

        assert len(rows) == 1
        assert rows[0]["dimension"] == "untagged"
        assert rows[0]["total"] == 1


class TestDimensionLabel:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("correctness", "正确性"),
            ("instruction_following", "指令遵循"),
            ("format", "格式合规"),
            ("safety", "安全"),
            ("robustness", "鲁棒性"),
            ("knowledge", "知识时效"),
            ("untagged", "未标注"),
        ],
    )
    def test_labels(self, value: str, expected: str) -> None:
        assert dimension_label(value) == expected

    def test_unknown_value_falls_back_to_itself(self) -> None:
        assert dimension_label("whatever") == "whatever"
