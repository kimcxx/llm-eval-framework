# -*- coding: utf-8 -*-
"""看板用例列表的「判定方式」列：每条结果得说清是用什么判的。

背景（用户验收时提的）：
    安全集现在一半用例走 LLM 安全裁判（``safety_judge``），一半仍走
    ``not_contains`` 排除检查。列表页只显示「通过 / 失败」，看不出这条成绩
    是哪把尺子量出来的 —— 判定方式透明化是评测工具的基本操守。

契约要点：
    * 用例列表表格有一列「判定」，显示该 case 实际走过的指标的**中文标签**；
    * 多指标的用例（如「语义相似度 + LLM 裁判」）**全部列出**，不许只显示第一个；
    * 标签来自后端 ``METRIC_LABELS``（与 Python 侧同一份数据），未收录的原样显示；
    * 没有指标的用例显示「—」，不编造判定方式；
    * 安全集两类用例在真实归档报告里分别是「安全裁判」与「排除检查」。
"""
from __future__ import annotations

import json
from pathlib import Path

from dashboard.app import METRIC_LABELS, PAGE, metric_label

DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"

# 安全裁判重跑的归档报告：30 条口令型（safety_judge）+ 30 条机密型（not_contains）
SAFETY_REPORT = DASHBOARD_DATA_DIR / "report-safety-judge-20261005-112926.json"


def _cases_tab_body() -> str:
    body = PAGE[PAGE.index("function renderCasesTab"):]
    return body[: body.index("// 跨次对比 tab")]


def _drawer_body() -> str:
    body = PAGE[PAGE.index("function attemptsSection"):]
    return body[: body.index("// 报告属性条")] if "// 报告属性条" in body else body[:4000]


class TestMethodColumnExists:
    """表格必须有「判定」列，且每行都填。"""

    def test_header_has_method_column(self) -> None:
        header = _cases_tab_body()
        assert "<th" in header and "判定</th>" in header, "用例列表缺「判定」列"

    def test_method_column_sits_between_category_and_repeat(self) -> None:
        header = _cases_tab_body()
        i_cat = header.index("分类</th>")
        i_method = header.index("判定</th>")
        i_repeat = header.index("通过次数</th>")
        assert i_cat < i_method < i_repeat, "「判定」列应排在「分类」与「通过次数」之间"

    def test_every_row_renders_method_cell(self) -> None:
        body = _cases_tab_body()
        assert "${caseMethodCell(c)}" in body, "行渲染必须调用 caseMethodCell 填判定列"


class TestMethodCellRendering:
    """判定单元格：中文标签、多指标全列、无指标显示 —。"""

    def test_labels_come_from_metric_label(self) -> None:
        body = PAGE[PAGE.index("function caseMethodCell"):]
        body = body[: body.index("// 用例的「判定方式」")] if "// 用例的「判定方式」" in body else body[:900]
        assert "metricLabel(n)" in body, "标签必须走 metricLabel（与后端 METRIC_LABELS 同源）"

    def test_all_metrics_listed_not_only_first(self) -> None:
        """多指标用例要全部列出：实现里遍历 metrics 全量，不取 [0] / slice(0, 1)。"""
        body = PAGE[PAGE.index("function caseMethodLabels"):]
        body = body[: body.index("function caseMethodCell")]
        assert "c.metrics" in body and ".map(" in body, "应遍历 metrics 取全部指标名"
        assert "[0]" not in body and "slice(0, 1)" not in body, "不许只取第一个指标"

    def test_empty_metrics_shows_placeholder(self) -> None:
        body = PAGE[PAGE.index("function caseMethodCell"):]
        body = body[: body.index("// 用例的「判定方式」")] if "// 用例的「判定方式」" in body else body[:900]
        assert "names.length" in body, "无指标时要有占位（—）分支"

    def test_metric_label_covers_safety_metrics(self) -> None:
        assert metric_label("safety_judge") == "安全裁判"
        assert metric_label("not_contains") == "排除检查"
        assert "安全裁判" in METRIC_LABELS.values()
        assert "排除检查" in METRIC_LABELS.values()


class TestSafetyReportMethodLabels:
    """真实归档报告：两类用例的判定方式必须分得开（这是本列的由来）。"""

    def test_two_methods_coexist_in_archived_report(self) -> None:
        if not SAFETY_REPORT.exists():
            return  # 归档报告缺失时跳过（跑过安全集后即生效）
        doc = json.loads(SAFETY_REPORT.read_text(encoding="utf-8"))
        rows = doc.get("cases") or []
        assert rows, "归档报告无用例数据"

        by_method: dict[str, set[str]] = {}
        for c in rows:
            for m in c.get("metrics") or []:
                by_method.setdefault(metric_label(m.get("name")), set()).add(c.get("case_id"))

        assert "安全裁判" in by_method, "口令型用例应显示「安全裁判」"
        assert "排除检查" in by_method, "机密型用例应显示「排除检查」"
        # 一条用例只走一种判定方式，两类不重叠
        assert not (by_method["安全裁判"] & by_method["排除检查"]), "同一用例不该同时有两种判定方式"


class TestDrawerUsesChineseLabels:
    """抽屉里的评分明细同样显示中文标签，不露英文原名。"""

    def test_drawer_metric_name_localized(self) -> None:
        body = _drawer_body()
        assert "metricLabel(m.name)" in body, "抽屉评分明细应显示中文标签"

    def test_drawer_skipped_metrics_localized(self) -> None:
        body = _drawer_body()
        skipped = PAGE[PAGE.index("跳过："):]
        assert "metricLabel" in skipped[:200], "跳过的指标也要显示中文标签"
