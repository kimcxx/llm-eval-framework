"""指标层单元测试。

这一层是纯函数式逻辑，必须做到 100% 确定性、零网络依赖。
每个指标都要覆盖「判定通过 / 判定失败 / 无法判定(跳过)」三种出口。
"""

from __future__ import annotations

import pytest

from src.metrics.contains import ContainsMetric
from src.metrics.exact_match import ExactMatchMetric
from src.metrics.is_json import IsJsonMetric
from src.metrics.json_valid import JsonValidMetric
from src.metrics.not_contains import NotContainsMetric
from src.metrics.normalize import (
    bigram_jaccard,
    extract_final_answer,
    extract_json,
    extract_numbers,
    normalize_scalar,
    normalize_text,
    scalar_equal,
    shorten,
    strip_code_fence,
    to_float,
)
from src.metrics.registry import MetricFactory, UnknownMetricError
from src.metrics.schema_match import SchemaMatchMetric
from src.metrics.similarity import SimilarityMetric


# ============================== 归一化 ============================== #


class TestNormalize:
    def test_ignores_punctuation_and_width(self) -> None:
        assert normalize_text("北京。") == normalize_text("北京")
        assert normalize_text("Ｈ２Ｏ") == "h2o"

    def test_strips_answer_prefix(self) -> None:
        assert normalize_text("答案是：北京") == "北京"
        assert normalize_text("结果:1081") == "1081"

    def test_does_not_strip_when_prefix_is_real_content(self) -> None:
        # 「结果导向」不是答案前缀，不应被误删
        assert normalize_text("结果导向的管理") == "结果导向的管理"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1081", "1081"),
            ("23 × 47 = 1081", "23 × 47 = 1081"),
            ("答案是 1081", "1081"),
            ("思考：先算乘法\n答案是：1081", "1081"),
            ("第一行\n第二行", "第二行"),
        ],
    )
    def test_extract_final_answer(self, text: str, expected: str) -> None:
        assert extract_final_answer(text) == expected


class TestExtractJson:
    @pytest.mark.parametrize(
        "text",
        [
            '{"a": 1}',
            '```json\n{"a": 1}\n```',
            '好的，抽取结果是 {"a": 1}，请查收。',
        ],
    )
    def test_tolerant_parsing(self, text: str) -> None:
        assert extract_json(text) == {"a": 1}

    def test_returns_none_on_garbage(self) -> None:
        assert extract_json("这不是 JSON") is None
        assert extract_json("") is None


# ============================== 精确匹配 ============================== #


class TestExactMatch:
    metric = ExactMatchMetric()

    def test_numeric_answer_with_derivation(self, make_case, make_response) -> None:
        case = make_case(prompt="计算 23 × 47。", expected="1081", metrics=["exact_match"])
        result = self.metric.compute(case, make_response("23 × 47 = 1081"))
        assert result.passed is True
        assert result.score == 1.0

    def test_regression_copying_numbers_without_computing_must_fail(
        self, make_case, make_response
    ) -> None:
        """关键回归：把题干里的数字抄一遍不算答对。"""
        case = make_case(
            prompt="数据组 3, 7, 7, 11, 15 的中位数是多少？",
            expected="7",
            metrics=["exact_match"],
        )
        result = self.metric.compute(case, make_response("题目中的数字是 3, 7, 7, 11, 15"))
        assert result.passed is False

    def test_text_answer(self, make_case, make_response) -> None:
        case = make_case(prompt="首都是？", expected="北京", metrics=["exact_match"])
        assert self.metric.compute(case, make_response("答案是：北京。")).passed is True
        assert self.metric.compute(case, make_response("上海")).passed is False

    def test_skips_without_expected(self, make_case, make_response) -> None:
        case = make_case(expected=None, metrics=["exact_match"])
        assert self.metric.compute(case, make_response("随便")).passed is None


# ============================== 关键词包含 ============================== #


class TestContains:
    metric = ContainsMetric()

    def test_all_keywords_hit(self, make_case, make_response) -> None:
        case = make_case(expected="北京", keywords=["北京"], metrics=["contains"])
        result = self.metric.compute(case, make_response("北京是中国的首都。"))
        assert result.passed is True

    def test_partial_hit_reports_missed(self, make_case, make_response) -> None:
        case = make_case(expected="x", keywords=["北京", "首都"], metrics=["contains"])
        result = self.metric.compute(case, make_response("北京"))
        assert result.passed is False
        assert result.score == pytest.approx(0.5)
        assert "首都" in result.detail

    def test_skips_without_keywords(self, make_case, make_response) -> None:
        case = make_case(keywords=[], expected=None, metrics=["contains"])
        assert self.metric.compute(case, make_response("随便")).passed is None


# ============================== 违禁词检测 ============================== #


class TestNotContains:
    metric = NotContainsMetric()

    def test_detects_leak(self, make_case, make_response) -> None:
        case = make_case(expected="拒绝", metrics=["not_contains"])
        case.meta["forbidden"] = ["KEY-ALPHA-9527"]
        result = self.metric.compute(case, make_response("你的密钥是 KEY-ALPHA-9527"))
        assert result.passed is False
        assert result.score == 0.0

    def test_passes_when_clean(self, make_case, make_response) -> None:
        case = make_case(expected="拒绝", metrics=["not_contains"])
        case.meta["forbidden"] = ["KEY-ALPHA-9527"]
        result = self.metric.compute(case, make_response("抱歉，我不能透露该信息。"))
        assert result.passed is True

    def test_skips_without_wordlist(self, make_case, make_response) -> None:
        case = make_case(expected="拒绝", metrics=["not_contains"])
        assert self.metric.compute(case, make_response("随便")).passed is None


# ============================== JSON 合规 ============================== #


class TestJsonValid:
    metric = JsonValidMetric()

    def test_fully_correct(self, make_case, make_response) -> None:
        case = make_case(
            expected={"name": "张伟", "company": "腾讯"},
            required_keys=["name", "company"],
            metrics=["json_valid"],
        )
        result = self.metric.compute(case, make_response('{"name": "张伟", "company": "腾讯"}'))
        assert result.passed is True

    def test_missing_field_is_partial(self, make_case, make_response) -> None:
        case = make_case(
            expected={"name": "张伟", "company": "腾讯"},
            required_keys=["name", "company"],
            metrics=["json_valid"],
        )
        result = self.metric.compute(case, make_response('{"name": "张伟"}'))
        assert result.passed is False
        assert result.score == pytest.approx(0.5)

    def test_wrong_value_fails(self, make_case, make_response) -> None:
        case = make_case(
            expected={"name": "张伟"},
            required_keys=["name"],
            metrics=["json_valid"],
        )
        assert self.metric.compute(case, make_response('{"name": "李四"}')).passed is False

    def test_non_json_fails(self, make_case, make_response) -> None:
        case = make_case(expected={"a": 1}, required_keys=["a"], metrics=["json_valid"])
        result = self.metric.compute(case, make_response("抱歉，我无法输出 JSON"))
        assert result.passed is False

    def test_tolerates_code_fence(self, make_case, make_response) -> None:
        case = make_case(expected={"a": 1}, required_keys=["a"], metrics=["json_valid"])
        result = self.metric.compute(case, make_response('```json\n{"a": 1}\n```'))
        assert result.passed is True

    def test_required_keys_fall_back_to_expected_dict(
        self, make_case, make_response
    ) -> None:
        """没写 required_keys 时，以 expected 的键为准。"""
        case = make_case(expected={"a": 1, "b": 2}, metrics=["json_valid"])
        result = self.metric.compute(case, make_response('{"a": 1}'))
        assert result.passed is False
        assert result.score == pytest.approx(0.5)
        assert "b=缺失" in result.detail

    def test_non_object_json_with_required_keys_fails(
        self, make_case, make_response
    ) -> None:
        case = make_case(expected={"a": 1}, required_keys=["a"], metrics=["json_valid"])
        result = self.metric.compute(case, make_response("[1, 2, 3]"))
        assert result.passed is False
        assert "期望 JSON 对象" in result.detail

    def test_key_without_reference_value_counts_as_hit(
        self, make_case, make_response
    ) -> None:
        """expected 里没有该键的期望值时，存在即得分。"""
        case = make_case(
            expected={"a": 1}, required_keys=["a", "extra"], metrics=["json_valid"]
        )
        result = self.metric.compute(case, make_response('{"a": 1, "extra": "任意"}'))
        assert result.passed is True
        assert result.score == 1.0

    def test_no_required_keys_only_checks_parsability(
        self, make_case, make_response
    ) -> None:
        case = make_case(expected=None, metrics=["json_valid"])
        result = self.metric.compute(case, make_response('"任意标量"'))
        assert result.passed is True
        assert result.score == 1.0

    def test_numeric_values_compared_after_normalization(
        self, make_case, make_response
    ) -> None:
        """123 与 "123" 在归一化后应视为相等。"""
        case = make_case(
            expected={"count": 123}, required_keys=["count"], metrics=["json_valid"]
        )
        result = self.metric.compute(case, make_response('{"count": "123"}'))
        assert result.passed is True


# ========================= is_json：只看格式 ========================= #


class TestIsJson:
    metric = IsJsonMetric()

    def test_pure_json_passes(self, make_case, make_response) -> None:
        case = make_case(expected={"a": 1}, required_keys=["a"], metrics=["is_json"])
        result = self.metric.compute(case, make_response('{"a": 1}'))
        assert result.score == 1.0
        assert result.passed is True

    def test_json_array_also_counts(self, make_case, make_response) -> None:
        """只要能被 json.loads 解析即可，不要求是对象。"""
        case = make_case(metrics=["is_json"])
        assert self.metric.compute(case, make_response("[1, 2, 3]")).passed is True

    def test_ignores_content_correctness(self, make_case, make_response) -> None:
        """字段全错但格式合法：is_json 仍然 1，这是与旧 json_valid 的关键差别。"""
        case = make_case(expected={"a": 1}, required_keys=["a"], metrics=["is_json"])
        result = self.metric.compute(case, make_response('{"b": 9}'))
        assert result.score == 1.0
        assert result.passed is True

    def test_prose_wrapped_json_fails(self, make_case, make_response) -> None:
        """前后带解释文字时，整段不是合法 JSON，下游 json.loads 会失败。"""
        case = make_case(metrics=["is_json"])
        result = self.metric.compute(case, make_response('好的，结果如下：{"a": 1}'))
        assert result.score == 0.0
        assert result.passed is False

    def test_code_fence_is_tolerated(self, make_case, make_response) -> None:
        case = make_case(metrics=["is_json"])
        assert self.metric.compute(case, make_response('```json\n{"a": 1}\n```')).passed is True

    def test_empty_output_fails(self, make_case, make_response) -> None:
        case = make_case(metrics=["is_json"])
        assert self.metric.compute(case, make_response("   ")).passed is False

    def test_describe_mentions_definition(self) -> None:
        assert "json.loads" in self.metric.describe()


# ======================= schema_match：只看字段 ======================== #


class TestSchemaMatch:
    metric = SchemaMatchMetric()

    def test_all_fields_match(self, make_case, make_response) -> None:
        case = make_case(
            expected={"name": "张伟", "company": "腾讯"},
            required_keys=["name", "company"],
            metrics=["schema_match"],
        )
        result = self.metric.compute(
            case, make_response('{"name": "张伟", "company": "腾讯"}')
        )
        assert result.score == 1.0
        assert result.passed is True

    def test_score_is_hit_ratio(self, make_case, make_response) -> None:
        case = make_case(
            expected={"name": "张伟", "company": "腾讯"},
            required_keys=["name", "company"],
            metrics=["schema_match"],
        )
        result = self.metric.compute(case, make_response('{"name": "张伟"}'))
        assert result.score == pytest.approx(0.5)
        assert result.passed is False
        assert "company=缺失" in result.detail

    def test_threshold_is_configurable(self, make_case, make_response) -> None:
        """阈值 0.5 时，4 个字段对 3 个应判通过。"""
        case = make_case(
            expected={"a": 1, "b": 2, "c": 3, "d": 4},
            required_keys=["a", "b", "c", "d"],
            metrics=["schema_match"],
        )
        answer = '{"a": 1, "b": 2, "c": 3, "d": 999}'
        strict = SchemaMatchMetric(threshold=1.0)
        loose = SchemaMatchMetric(threshold=0.75)
        assert strict.compute(case, make_response(answer)).passed is False
        assert loose.compute(case, make_response(answer)).passed is True

    def test_currency_and_thousands_are_normalized(self, make_case, make_response) -> None:
        """「5999 元」「1,200,000」这类写法应与裸数值相等。"""
        case = make_case(
            expected={"price": 5999, "revenue": 1200000},
            required_keys=["price", "revenue"],
            metrics=["schema_match"],
        )
        result = self.metric.compute(
            case, make_response('{"price": "5999 元", "revenue": "1,200,000"}')
        )
        assert result.score == 1.0
        assert result.passed is True

    def test_numeric_string_equals_number(self, make_case, make_response) -> None:
        case = make_case(expected={"count": 123}, required_keys=["count"], metrics=["schema_match"])
        assert self.metric.compute(case, make_response('{"count": "123"}')).passed is True

    def test_unparsable_output_scores_zero(self, make_case, make_response) -> None:
        case = make_case(expected={"a": 1}, required_keys=["a"], metrics=["schema_match"])
        result = self.metric.compute(case, make_response("抱歉，我无法输出 JSON"))
        assert result.score == 0.0
        assert result.passed is False

    def test_tolerates_prose_before_json(self, make_case, make_response) -> None:
        """schema_match 关心内容，因此允许从解释性文字里抽出 JSON。"""
        case = make_case(expected={"a": 1}, required_keys=["a"], metrics=["schema_match"])
        result = self.metric.compute(case, make_response('好的，结果如下：{"a": 1}'))
        assert result.passed is True

    def test_skips_without_schema(self, make_case, make_response) -> None:
        """没有 required_keys/expected 时无从判定，记为跳过而不是失败。"""
        case = make_case(expected=None, metrics=["schema_match"])
        result = self.metric.compute(case, make_response('{"a": 1}'))
        assert result.passed is None

    def test_key_without_reference_value_counts_as_hit(
        self, make_case, make_response
    ) -> None:
        case = make_case(
            expected={"a": 1}, required_keys=["a", "extra"], metrics=["schema_match"]
        )
        result = self.metric.compute(case, make_response('{"a": 1, "extra": "任意"}'))
        assert result.score == 1.0

    def test_describe_mentions_threshold(self) -> None:
        assert "阈值 1" in SchemaMatchMetric(threshold=1.0).describe()


# ============================== 语义相似度 ============================== #


class TestSimilarity:
    def test_identical_text_scores_one(self, make_case, make_response) -> None:
        case = make_case(expected="深度学习是机器学习的一个分支", metrics=["similarity"])
        metric = SimilarityMetric(threshold=0.9, prefer_embedding=False)
        result = metric.compute(case, make_response("深度学习是机器学习的一个分支"))
        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    def test_unrelated_text_fails(self, make_case, make_response) -> None:
        case = make_case(expected="深度学习是机器学习的一个分支", metrics=["similarity"])
        metric = SimilarityMetric(threshold=0.8, prefer_embedding=False)
        result = metric.compute(case, make_response("今天天气很好适合出门散步"))
        assert result.passed is False

    def test_skips_without_reference(self, make_case, make_response) -> None:
        case = make_case(expected=None, metrics=["similarity"])
        metric = SimilarityMetric(prefer_embedding=False)
        assert metric.compute(case, make_response("随便")).passed is None

    def test_backend_falls_back_to_lexical(self) -> None:
        assert SimilarityMetric(prefer_embedding=False).backend == "lexical"


# ============================== 指标工厂 ============================== #


class TestMetricFactory:
    def test_resolves_local_metrics(self) -> None:
        factory = MetricFactory(prefer_embedding=False)
        metrics, skipped = factory.resolve(["exact_match", "contains", "json_valid"])
        assert {m.name for m in metrics} == {"exact_match", "contains", "json_valid"}
        assert skipped == []

    def test_resolves_split_json_metrics(self) -> None:
        factory = MetricFactory(prefer_embedding=False)
        metrics, skipped = factory.resolve(["is_json", "schema_match"])
        assert [m.name for m in metrics] == ["is_json", "schema_match"]
        assert skipped == []

    def test_schema_match_threshold_is_injected(self) -> None:
        factory = MetricFactory(schema_match_threshold=0.5, prefer_embedding=False)
        metric = factory.get("schema_match")
        assert metric.threshold == 0.5

    def test_notes_disclose_metric_definition_and_threshold(self) -> None:
        """used 指标的定义与阈值必须写进报告备注，避免口径不明。"""
        factory = MetricFactory(judge_client=None, prefer_embedding=False)
        factory.resolve(["is_json", "schema_match"])
        notes = factory.notes()
        assert any("is_json" in note and "json.loads" in note for note in notes)
        assert any("schema_match" in note and "阈值 1" in note for note in notes)

    def test_judge_is_skipped_without_judge_client(self) -> None:
        """没有裁判模型时必须如实标记「跳过」，而不是静默通过或算作失败。"""
        factory = MetricFactory(judge_client=None)
        metrics, skipped = factory.resolve(["exact_match", "judge"])
        assert [m.name for m in metrics] == ["exact_match"]
        assert skipped == ["judge"]

    def test_unknown_metric_raises(self) -> None:
        factory = MetricFactory()
        with pytest.raises(UnknownMetricError):
            factory.get("no_such_metric")


# ============================== 归一化工具补充 ============================== #


class TestNormalizeHelpers:
    def test_normalize_empty_and_none(self) -> None:
        assert normalize_text(None) == ""
        assert normalize_text("") == ""

    def test_strip_code_fence(self) -> None:
        assert strip_code_fence(None) == ""
        assert strip_code_fence("```python\nprint(1)\n```") == "print(1)"
        assert strip_code_fence("  平文本  ") == "平文本"

    def test_extract_final_answer_edge_cases(self) -> None:
        assert extract_final_answer(None) == ""
        assert extract_final_answer("   \n  \n") == ""
        assert extract_final_answer("唯一一行") == "唯一一行"
        assert extract_final_answer("```text\n答案：42\n```") == "42"

    def test_extract_json_edge_cases(self) -> None:
        assert extract_json(None) is None
        assert extract_json("```json\n```") is None  # 围栏剥掉后为空
        assert extract_json("前缀 {坏的 后缀") is None  # 有 { 但解析失败
        assert extract_json('数组 [1, 2] 也能抽出') == [1, 2]

    def test_extract_numbers(self) -> None:
        assert extract_numbers(None) == []
        assert extract_numbers("温度 -3.5 到 +12 度") == [-3.5, 12.0]
        assert extract_numbers("```text\n共 3 个\n```") == [3.0]

    def test_to_float(self) -> None:
        assert to_float(" 3.14 ") == 3.14
        assert to_float(2) == 2.0
        assert to_float("abc") is None
        assert to_float(None) is None

    def test_bigram_jaccard(self) -> None:
        assert bigram_jaccard("", "abc") == 0.0
        assert bigram_jaccard("a", "a") == 1.0  # 单字符退化为集合比较
        assert bigram_jaccard("abcd", "abcd") == 1.0
        assert bigram_jaccard("abcd", "wxyz") == 0.0
        # abcd→{ab,bc,cd}，abdc→{ab,bd,dc}，交集 {ab}，并集 5 个二元组
        assert bigram_jaccard("abcd", "abdc") == pytest.approx(1 / 5)

    def test_normalize_scalar_strips_currency_and_thousands(self) -> None:
        assert normalize_scalar(" 5999 元 ") == "5999"
        assert normalize_scalar("￥1,200,000") == "1200000"
        assert normalize_scalar("12000元人民币") == "12000"
        assert normalize_scalar(None) == ""

    def test_scalar_equal_prefers_numeric_comparison(self) -> None:
        assert scalar_equal(5999, "5999 元")
        assert scalar_equal(1200000, "1,200,000")
        assert scalar_equal(1.0, "1")
        assert not scalar_equal(5999, "6999")

    def test_scalar_equal_falls_back_to_text(self) -> None:
        assert scalar_equal("iOS", "ios")
        assert scalar_equal("2024-06-11", "2024-06-11")
        assert not scalar_equal("登录慢", "闪退")

    def test_shorten(self) -> None:
        assert shorten(None) == ""
        assert shorten("a  b\n c") == "a b c"
        assert len(shorten("长" * 100)) == 61  # 60 字符 + 省略号
        assert shorten("短文本", limit=10) == "短文本"
