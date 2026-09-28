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
from src.runner.dimension import DEFAULT_DIMENSION_BY_CATEGORY, resolve_dimension
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
            _case("c2", "2+2", "4", "instruction_following"),
            _case("c3", "1+1", "2"),  # 未打标签
        ]

    def test_groups_in_enum_order_with_untagged_last(self) -> None:
        report = _runner().run_cases(self._cases())
        assert [g.key for g in report.by_dimension("fake")] == [
            "correctness",
            "instruction_following",
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
        assert {r["dimension"] for r in rows} == {
            "correctness",
            "instruction_following",
            "untagged",
        }
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
            ("correctness", "准确性"),
            ("instruction_following", "指令遵循"),
            ("safety", "安全"),
            ("relevance", "相关性"),
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


class TestDefaultDimensionByCategory:
    """没写 dimension 的用例按分类推导默认维度：声明 > 分类映射 > untagged。"""

    @pytest.mark.parametrize(
        ("category", "expected"),
        [
            ("json_extract", "instruction_following"),
            ("math_reasoning", "correctness"),
            ("qa_open", "relevance"),
            ("qa_zh", "correctness"),
            ("safety_redteam", "safety"),
        ],
    )
    def test_category_default(self, category: str, expected: str) -> None:
        assert resolve_dimension(category) == expected

    def test_declared_wins_over_category(self) -> None:
        """json_extract 默认 instruction_following，但用例写了 safety 就以用例为准。"""
        assert resolve_dimension("json_extract", "safety") == "safety"

    @pytest.mark.parametrize("category", ["default", "qa_en", "summary", ""])
    def test_unmapped_category_is_untagged(self, category: str) -> None:
        assert resolve_dimension(category) == "untagged"

    def test_matching_ignores_case_and_spaces(self) -> None:
        assert resolve_dimension("  JSON_Extract ") == "instruction_following"

    @pytest.mark.parametrize("declared", ["", "   "])
    def test_blank_declared_falls_back_to_category(self, declared: str) -> None:
        assert resolve_dimension("math_reasoning", declared) == "correctness"

    def test_every_default_is_a_known_dimension(self) -> None:
        """映射表写错维度名会让整组用例静默落进未知组，这里锁死。"""
        assert set(DEFAULT_DIMENSION_BY_CATEGORY.values()) <= set(DIMENSIONS)

    def test_unlabelled_case_is_tagged_by_category(self) -> None:
        case = EvalCase.from_dict(
            {
                "id": "c1",
                "category": "json_extract",
                "prompt": "1+1",
                "expected": "2",
                "metrics": ["exact_match"],
            }
        )
        report = _runner().run_cases([case])

        assert report.cases[0].dimension == "instruction_following"
        assert [g.key for g in report.by_dimension("fake")] == ["instruction_following"]

    def test_declared_dimension_survives_in_report(self) -> None:
        case = EvalCase.from_dict(
            {
                "id": "c1",
                "category": "json_extract",
                "dimension": "safety",
                "prompt": "1+1",
                "expected": "2",
                "metrics": ["exact_match"],
            }
        )
        assert _runner().run_cases([case]).cases[0].dimension == "safety"

    def test_default_tag_does_not_change_verdict(self) -> None:
        """默认标签只是分组依据，判定结果必须与分类无关。"""
        raw = {"id": "c1", "prompt": "1+1", "expected": "2", "metrics": ["exact_match"]}
        tagged = EvalCase.from_dict({**raw, "category": "json_extract"})
        plain = EvalCase.from_dict({**raw, "category": "default"})

        assert _runner().run_cases([tagged]).cases[0].passed is True
        assert _runner().run_cases([plain]).cases[0].passed is True


class TestLegacyDimensionBackCompat:
    """老数据集 / 老报告里写的是旧维度名（format 等）：不能报错，也不能静默消失。"""

    def test_format_is_normalized_to_instruction_following(self) -> None:
        """format 是指令遵循的旧称，打标与聚合必须同时换成新名。"""
        case = EvalCase.from_dict({"id": "a", "prompt": "q", "dimension": "format"})
        assert case.dimension == "instruction_following"

    def test_legacy_declared_value_is_normalized_on_resolve(self) -> None:
        assert resolve_dimension("math_reasoning", "FORMAT") == "instruction_following"

    @pytest.mark.parametrize("value", ["robustness", "knowledge"])
    def test_legacy_without_alias_is_kept_as_is(self, value: str) -> None:
        """没有对应新维度的旧名原样保留——报告里仍有一行，好过被悄悄丢掉。"""
        case = EvalCase.from_dict({"id": "a", "prompt": "q", "dimension": value})
        assert case.dimension == value

    def test_qa_open_is_no_longer_correctness(self) -> None:
        """开放题走 LLM 裁判，单列「相关性」：这次口径调整的核心，锁死。"""
        assert resolve_dimension("qa_open") == "relevance"

    def test_legacy_case_does_not_break_aggregation(self) -> None:
        """老用例写 format，聚合里应该出现在 instruction_following 组而不是新起一组。"""
        report = _runner().run_cases([_case("c1", "1+1", "2", "format")])
        assert [g.key for g in report.by_dimension("fake")] == ["instruction_following"]


class TestStoredReportUsesNewDimensions:
    """runner 必须把新维度值写进报告，而不是留给展示层去翻译。

    只在前端映射是兜底：新报告存的是旧三组的话，存储层与展示层就永远对不上。
    """

    @staticmethod
    def _cases() -> list[EvalCase]:
        return [
            EvalCase.from_dict({
                "id": "j1", "prompt": "返回 JSON", "expected": '{"a": 1}',
                "category": "json_extract", "metrics": ["exact_match"],
            }),
            EvalCase.from_dict({
                "id": "q1", "prompt": "开放问答", "expected": "参考答案",
                "category": "qa_open", "metrics": ["exact_match"],
            }),
        ]

    def test_case_result_carries_new_dimension_values(self) -> None:
        """CaseResult.dimension 直接就是新值，不需要展示层再映射一层。"""
        report = _runner().run_cases(self._cases())
        assert {c.dimension for c in report.cases} == {"instruction_following", "relevance"}

    def test_stored_dimensions_segment_is_new_scope(self) -> None:
        """to_dict() 的 dimensions 段：JSON 抽取不在准确性里，开放题单列相关性。"""
        payload = _runner().run_cases(self._cases()).to_dict()

        rows = {r["dimension"]: r for r in payload["dimensions"]["fake"]}
        assert set(rows) == {"instruction_following", "relevance"}
        assert "correctness" not in rows
        assert rows["instruction_following"]["label"] == "指令遵循"
        assert rows["relevance"]["label"] == "相关性"
        assert {c["dimension"] for c in payload["cases"]} == {"instruction_following", "relevance"}
