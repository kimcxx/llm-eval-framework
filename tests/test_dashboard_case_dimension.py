# -*- coding: utf-8 -*-
"""模型层逐题明细「能力指标」列契约（与判定方式分开）。

背景：
    模型层逐题明细原先在指标位显示的是**判定器名**（contains / judge / similarity），
    抽屉里那张表干脆就叫「指标」——把「怎么判」当成了「测什么能力」在显示。
    能力指标其实早就存在：``dimension`` 四类（准确性 / 指令遵循 / 安全 / 相关性），
    报告顶部的维度通过率矩阵算的就是它。缺的只是逐题明细那一列。

契约要点（与 RAG 层同一条口径：**指标名 = 能力名，判定方式是实现细节**）：
    1. 逐题明细有「能力指标」列，读的是 dimension，**不许显示判定器名**；
    2. 「判定方式」列保留且与能力指标分成两列——它兼着「为什么判失败」的解释作用；
    3. 四个 dimension 的中文名齐全（沿用 src 的 DIMENSION_LABELS，不许自己编）；
    4. 未打标签的显示「未标注」，行为不变；
    5. 抽屉里那张表的标题不再是「指标」。
"""

from __future__ import annotations

import dashboard.app as app
from src.datasets.schema import DIMENSION_LABELS, UNTAGGED_DIMENSION_LABEL

NEW_DIMENSIONS = ("correctness", "instruction_following", "safety", "relevance")


class TestCaseDimensionColumn:
    """逐题明细：能力指标列存在，与判定方式分开。"""

    def test_two_columns_rendered(self) -> None:
        assert "caseDimCell" in app.PAGE, "逐题明细缺「能力指标」列的渲染函数"
        # 「能力指标」与「判定方式」必须是相邻的两列，不许合并
        assert "<th style=\"width:104px\">能力指标</th><th style=\"width:120px\">判定方式</th>" in app.PAGE
        assert "caseMethodCell(c)" in app.PAGE, "判定方式列不许删（失败原因还要靠它）"

    def test_column_uses_dimension_not_metric_name(self) -> None:
        """能力指标列读的是维度（caseDimOf / DIM_LABELS），不是判定器（metricLabel）。"""
        assert "caseDimLabel" in app.PAGE and "caseDimOf(c)" in app.PAGE
        # caseDimCell 的渲染体里不许出现 metricLabel（那是判定器名的渲染入口）
        start = app.PAGE.index("function caseDimCell")
        body = app.PAGE[start : start + 400]
        assert "metricLabel" not in body, "能力指标列里混进了判定器名的渲染"

    def test_untagged_shows_label(self) -> None:
        # 精确到常量赋值（页面别处也有「未标注」三个字，宽断言抓不住改动）
        assert "CASE_DIM_UNKNOWN = '未标注'" in app.PAGE, "未打标签的用例要显示「未标注」"
        assert "caseDimOf(c) === 'untagged'" in app.PAGE, "未标注要走 untagged 分支"

    def test_column_rendered_in_row(self) -> None:
        """每一行的单元格顺序：分类 → 能力指标 → 判定方式。"""
        assert "${caseDimCell(c)}\n              ${caseMethodCell(c)}" in app.PAGE


class TestDimensionNamesComplete:
    """四个能力名齐全，且与 src 侧同名（不许另编一套）。"""

    def test_labels_synced_with_src(self) -> None:
        assert DIMENSION_LABELS["correctness"] == "准确性"
        assert DIMENSION_LABELS["instruction_following"] == "指令遵循"
        assert DIMENSION_LABELS["safety"] == "安全"
        assert DIMENSION_LABELS["relevance"] == "相关性"
        assert UNTAGGED_DIMENSION_LABEL == "未标注"

    def test_four_names_present_in_page(self) -> None:
        """页面上这四个中文名必须都找得到（缺一个就说明没映射到）。"""
        for zh in ("准确性", "指令遵循", "安全", "相关性", "未标注"):
            assert zh in app.PAGE, f"页面缺少能力名「{zh}」"


class TestNoMetricNameInIndicatorSlot:
    """判定器名不许再占「指标」这个位置。"""

    def test_drawer_table_renamed(self) -> None:
        assert "<h3>指标 (${(c.metrics||[]).length})</h3>" not in app.PAGE, "抽屉里那张表还在叫「指标」"
        assert "<tr><th>指标</th><th>score</th>" not in app.PAGE
        assert "<h3>判定方式 (${(c.metrics||[]).length})</h3>" in app.PAGE
        assert "<tr><th>判定方式</th><th>score</th><th>通过</th><th>详情</th></tr>" in app.PAGE

    def test_drawer_shows_capability_label(self) -> None:
        """抽屉的「维度」改叫「能力指标」，且与列表列同源（caseDimLabel）。"""
        assert "能力指标：<b>${esc(caseDimLabel(c))}</b>" in app.PAGE
        assert "维度：<b>${esc(dimLabel(c.dimension))}</b>" not in app.PAGE
