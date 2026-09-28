# -*- coding: utf-8 -*-
"""详情页三块改动的数据契约 / 模板契约测试。

背景：
    1. 汇总 tab 从「每个模型平铺一张维度表」改成「维度 × 模型」对比矩阵：
       行 = 维度（子行 = 分类），列 = 模型，格 = 通过率，末尾一列「最大差值」。
    2. 详情页顶部属性条：范围（全量 / 子集）× 重复次数 × 判定口径，三个正交
       属性并列，避免 full-repeat3 这种报告被单选分类压掉一半信息。

注意：
    * JS 跑不了（没有 node 环境），前端只测**模板契约**：渲染函数存在、被
      调用、关键阈值与样式类名到位；真正的数据计算放在 Python 侧测。
"""
from __future__ import annotations

import inspect

import pytest

import dashboard.app as app

ALL_FOUR = ["correctness", "instruction_following", "safety", "relevance"]


def _doc(cases=(), repeat=None, dimensions=None):
    doc = {"cases": list(cases)}
    if repeat is not None:
        doc["repeat"] = repeat
    if dimensions is not None:
        doc["dimensions"] = {"m": list(dimensions)}
    return doc


# ============================== 对比矩阵（模板契约） ============================== #


class TestMatrixTemplate:
    """矩阵渲染函数必须挂在汇总 tab 上，且关键规则都在模板里。"""

    def test_matrix_is_wired_into_summary_tab(self) -> None:
        assert "function buildDimensionMatrix" in app.PAGE
        assert "function dimensionMatrixCard" in app.PAGE
        assert "dimensionMatrixCard(matrix)" in app.PAGE

    def test_columns_come_from_summary_order(self) -> None:
        """列 = 本报告所有被测模型，顺序照 summary（= 配置顺序）。"""
        assert "buildDimensionMatrix(r.cases || [], (s || []).map(m => m.model))" in app.PAGE

    def test_mock_column_is_first_but_weakened(self) -> None:
        """mock-* 排最左（基线先看到），但灰显——它是管道基线不是被测对象。"""
        assert "table.matrix td.col-mock" in app.PAGE
        assert "table.matrix th.col-mock" in app.PAGE

    def test_spread_alert_threshold_is_five_points(self) -> None:
        """真模型间差距 ≥ 5pt 的行高亮；低于这个量级是噪声，不该提示。"""
        assert "const SPREAD_ALERT_PT = 5" in app.PAGE
        assert "spread-alert" in app.PAGE
        assert "spread-row" in app.PAGE

    def test_spread_ignores_mock_and_needs_two_real_models(self) -> None:
        """差值只在**真模型**之间算；真模型不足两个时返回 null（显示 —）。"""
        assert "rates.length < 2" in app.PAGE
        assert "!isMockModel(m)" in app.PAGE

    def test_single_model_degrades_to_one_column(self) -> None:
        """单模型报告：退化成单列，不渲染「最大差值」列（而不是显示 0 或报错）。"""
        assert "const single = models.length <= 1" in app.PAGE
        assert "single ? '' : spreadHtml(spread)" in app.PAGE

    def test_old_dimensions_fall_back_to_dim_key_of(self) -> None:
        """老报告的维度仍由 caseDimOf / dimKeyOf 归一（cases 里没有 dimension 字段）。"""
        assert "const caseDimOf" in app.PAGE
        assert "dimKeyOf((c && c.dimension) || cat)" in app.PAGE


# ============================== 报告属性条（数据契约） ============================== #


class TestComputeAttributes:
    """``compute_attributes``：范围 / 重复次数 / 判定口径。"""

    def test_full_scope_when_all_four_dimensions_covered(self) -> None:
        doc = _doc([{"dimension": d, "model": "m"} for d in ALL_FOUR])
        out = app.compute_attributes(doc)

        assert out["scope"] == "full"
        assert out["scope_label"] == "全量"
        assert out["coverage_text"] == "四维全覆盖"
        assert out["repeat"] == 1
        assert out["verdict"] == "通过率"
        assert out["is_stability"] is False

    def test_subset_scope_names_the_covered_dimensions(self) -> None:
        """dimcheck 这种只跑两组的报告：标注子集 + 具体覆盖到哪几维。"""
        doc = _doc([{"dimension": "instruction_following"}, {"dimension": "relevance"}])
        out = app.compute_attributes(doc)

        assert out["scope"] == "subset"
        assert out["scope_label"] == "子集"
        assert out["dimensions"] == ["指令遵循", "相关性"]
        assert out["coverage_text"] == "指令遵循、相关性"

    def test_full_and_repeat_three_are_expressed_together(self) -> None:
        """full-repeat3：既全量又是 repeat=3，两个属性必须同时表达（不是二选一）。"""
        doc = _doc([{"dimension": d} for d in ALL_FOUR], repeat=3)
        out = app.compute_attributes(doc)

        assert out["scope"] == "full"
        assert out["repeat"] == 3
        assert out["verdict"] == "稳定率"
        assert out["is_stability"] is True
        assert "稳定率" in out["verdict_note"]
        assert "3" in out["repeat_note"]

    def test_repeat_is_inferred_from_cases_for_old_reports(self) -> None:
        """老报告没有顶层 repeat：从 cases 的 repeat / attempts 长度推断。"""
        doc = {"cases": [{"category": "math_reasoning", "attempts": [{}, {}, {}]}]}
        assert app.compute_attributes(doc)["repeat"] == 3

        doc = {"cases": [{"category": "math_reasoning", "repeat": 2}]}
        assert app.compute_attributes(doc)["repeat"] == 2

    def test_old_report_without_any_signal(self) -> None:
        """什么都取不到时给合理回退值，而不是崩溃或显示空白。"""
        out = app.compute_attributes({})

        assert out["scope"] == "subset"
        assert out["coverage_text"] == "未识别到维度"
        assert out["repeat"] == 1
        assert out["verdict"] == "通过率"

    def test_non_dict_input_does_not_crash(self) -> None:
        out = app.compute_attributes(None)
        assert out["repeat"] == 1
        assert out["verdict"] == "通过率"

    def test_dimensions_segment_is_used_when_cases_missing(self) -> None:
        """没有 cases 的老报告：退回报告里存的 dimensions 段算覆盖。"""
        doc = _doc(dimensions=[{"dimension": "safety"}, {"dimension": "format"}])
        out = app.compute_attributes(doc)

        assert out["dimensions"] == ["指令遵循", "安全"]  # format → 指令遵循

    def test_covered_dimensions_are_sorted_by_dimension_order(self) -> None:
        doc = _doc([{"dimension": "safety"}, {"dimension": "correctness"}])
        assert app._covered_dimensions(doc) == ["correctness", "safety"]


class TestAttributesPlumbing:
    """属性条要真的被送到前端。"""

    def test_api_report_injects_attributes(self) -> None:
        src = inspect.getsource(app.Handler.do_GET)
        assert '_attributes' in src
        assert "compute_attributes(data)" in src

    def test_detail_page_renders_attribute_bar(self) -> None:
        assert "function attributesBar" in app.PAGE
        assert "attributesBar(r._attributes)" in app.PAGE
        assert 'class="attrs"' in app.PAGE

    def test_three_attributes_are_rendered(self) -> None:
        """范围 / 重复 / 判定：三个都渲染，缺一个就又变回单选分类了。"""
        assert "chip('范围'" in app.PAGE
        assert "chip('重复'" in app.PAGE
        assert "chip('判定'" in app.PAGE

    @pytest.mark.parametrize("report", [
        "report-full-regression-20260923-143903.json",
        "report-json-r3-20260923-142557.json",
    ])
    def test_real_reports_produce_attributes(self, report) -> None:
        import json

        path = app.REPORTS_DIR / report
        if not path.is_file():
            pytest.skip(f"{report} 不在 {app.REPORTS_DIR}，跳过")
        out = app.compute_attributes(json.loads(path.read_text(encoding="utf-8")))
        assert out["repeat"] >= 1
        assert out["verdict"] in ("通过率", "稳定率")
        assert out["scope_label"] in ("全量", "子集")
