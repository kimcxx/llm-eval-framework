# -*- coding: utf-8 -*-
"""#llm 页「报告维度对比」：同维度看两份真实评测报告谁涨谁跌。

背景：
    想看「同维度的报告对比」（如 9/23 全量回归 vs 之后某次真实评测），
    每个维度谁涨谁跌一眼看清，不用开两个详情页左右对照。

契约要点（对照阿灯的设计基准）：
    * 入口在 ``#llm`` 页，选择器的报告**只含真实模型**——复用 ``scoringReports()``，
      mock-only 的管道自检报告不进选择器（与 banner / 门户同一个选源）；
    * 一维度一行，**同名模型才互比**（chat 对 chat、pro 对 pro），
      模型名对不上（只有一份报告有该模型）的行标「—」；
    * 差值 = B − A（百分点），**带正负号**；提升绿 / 回退红 / 不变灰（测试质量语义）；
    * 页面上写清对比双方是哪两份报告（编号 + 时间 + 文件名）；
    * 凑不够两份真实报告时只给文字提示，**不拿 mock 凑数**；
    * 维度口径复用 ``dimensionSummary``（与后端 ``_dimension_summary`` 同契约），不另算一套。
"""
from __future__ import annotations

import json
from pathlib import Path

import dashboard.app as app
from dashboard.app import PAGE

DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"

# 两份含真实模型的全量回归报告（用于手算核对）：A = 较早，B = 较新
REPORT_A = DASHBOARD_DATA_DIR / "report-full-regression-20260923-143903.json"
REPORT_B = DASHBOARD_DATA_DIR / "report-full-regression-20261004-194618.json"


def _body(start: str, end: str) -> str:
    seg = PAGE[PAGE.index(start):]
    return seg[: seg.index(end)]


def _compare_card_body() -> str:
    return _body("function compareCard(", "async function bindCompare(")


def _compare_fn_body() -> str:
    return _body("function compareReports(", "// 差值列：带正负号")


def _delta_body() -> str:
    return _body("function deltaCell(", "// 对比入口：凑不够")


def _compare_view_body() -> str:
    """对比视图整段（入口 + 渲染 + 绑定）。"""
    return _body("// ============================== #llm 报告维度对比", "// 详情页属性条")


class TestEntryWiredIntoListPage:
    """对比入口挂在 #llm 页上，且渲染后被绑定。"""

    def test_card_rendered_in_list(self) -> None:
        assert "${compareCard(reports)}" in PAGE, "报告列表页未渲染对比入口"

    def test_binding_called_after_render(self) -> None:
        list_body = _body("async function renderList(", "// ============================== #llm 报告维度对比")
        assert "bindCompare(reports);" in list_body, "渲染后必须绑定选择器事件"


class TestSelectorOnlyRealReports:
    """选择器只含真实报告，mock-only 自检报告不出现。"""

    def test_pool_uses_scoring_reports(self) -> None:
        body = _compare_card_body()
        assert "scoringReports(reports)" in body, "必须复用成绩源选源（与 banner / 门户同源）"

    def test_options_built_from_pool_not_all_reports(self) -> None:
        body = _compare_card_body()
        assert "pool.map(" in body, "option 必须由 pool（真实报告）生成"
        assert "reports.map(" not in body, "不许拿全量 reports（含自检）生成 option"


class TestInsufficientReportsBehavior:
    """真实报告不足两份：给文字提示，不渲染选择器、不拿 mock 凑数。"""

    def test_guard_exists(self) -> None:
        body = _compare_card_body()
        assert "pool.length < 2" in body, "缺「不足两份」的判断"

    def test_placeholder_explains_why(self) -> None:
        body = _compare_card_body()
        assert "至少需要两份" in body, "要说明为什么对比不了"
        assert "自检" in body, "要点明自检报告不计入"

    def test_binding_skipped_when_no_selectors(self) -> None:
        bind_body = _body("async function bindCompare(", "// 详情页属性条")
        assert "if (!selA || !selB) return;" in bind_body, "没有选择器时绑定必须安静退出"


class TestDefaultPair:
    """默认对比哪两份：维度覆盖最全的两份，A 较早 / B 较新。

    直接取最新两份会撞上「全量报告 vs 只跑安全集的报告」——维度对不上就满屏「—」，
    第一眼看不出涨跌。覆盖维度最多的通常是同一套用例集的两次全量评测。
    """

    def test_uses_default_pair_helper(self) -> None:
        body = _compare_card_body()
        assert "defaultComparePair(pool)" in body, "不许硬取 pool[0] / pool[1] 当默认值"

    def test_picks_widest_dimension_coverage(self) -> None:
        body = _body("function defaultComparePair(", "// 对比入口：凑不够")
        assert "dimension_coverage" in body, "按维度覆盖面选，不是按新旧硬选"
        assert "Math.max(...pool.map(cover))" in body

    def test_a_is_older_b_is_newer(self) -> None:
        """pool 已按时间倒序：wide[0] 较新 → B，wide[1] 较早 → A。"""
        body = _body("function defaultComparePair(", "// 对比入口：凑不够")
        assert "[wide[1], wide[0]]" in body, "返回顺序必须是 [A(较早), B(较新)]"


class TestDeltaRendering:
    """差值：带正负号 + 颜色语义（提升绿 / 回退红 / 不变灰）。"""

    def test_sign_rendered(self) -> None:
        body = _delta_body()
        assert "'+'" in body and "'-'" in body, "差值必须带正负号"

    def test_color_semantics(self) -> None:
        body = _delta_body()
        assert "var(--ok)" in body and "var(--bad)" in body and "var(--muted)" in body
        # 提升 → ok、回退 → bad、不变 → muted
        assert body.index("v > 0 ? 'var(--ok)'") < body.index("? 'var(--bad)'")

    def test_missing_side_shows_dash(self) -> None:
        body = _delta_body()
        assert "pt == null" in body and "—" in body, "缺一侧应显示「—」而不是 0"

    def test_float_noise_flattened(self) -> None:
        body = _delta_body()
        assert "Math.abs(pt) < 0.05" in body, "浮点噪声要压平，避免显示 -0.0"


class TestCompareSemantics:
    """同名模型才互比；维度取并集，缺侧为 null。"""

    def test_model_pairs_by_name(self) -> None:
        body = _compare_fn_body()
        assert "modelsA.includes(m) && modelsB.includes(m)" in body, "必须按模型名配对"

    def test_in_both_flag_exposed(self) -> None:
        body = _compare_fn_body()
        assert "inBoth:" in body, "要暴露「两侧都有」标记，供页面标注无法对比"

    def test_delta_is_b_minus_a(self) -> None:
        body = _compare_fn_body()
        assert "(b.pass_rate - a.pass_rate) * 100" in body, "差值必须是 B − A（百分点）"

    def test_missing_side_is_null(self) -> None:
        body = _compare_fn_body()
        assert "a ? a.pass_rate : null" in body
        assert "b ? b.pass_rate : null" in body

    def test_dimension_union_sorted_like_dimension_summary(self) -> None:
        body = _compare_fn_body()
        assert "dimOrderIndex(" in body, "维度并集要按 DIMENSION_ORDER 排序"

    def test_dimension_source_is_the_shared_helper(self) -> None:
        body = _compare_fn_body()
        assert "dimensionSummary(" in body, "维度必须复用 dimensionSummary，不另算一套"
        assert "与后端 _dimension_summary 同契约" in PAGE


class TestBothReportsNamedOnPage:
    """页面要写清对比双方是哪两份报告。"""

    def test_shows_number_time_and_file(self) -> None:
        body = _compare_view_body()
        assert "报告A（基准）" in body and "报告B（对比）" in body
        assert "reportNo(reports, aMeta)" in body, "要带报告编号（与列表口径一致）"
        assert "esc(selA.value)" in body and "esc(selB.value)" in body, "要带报告文件名"


class TestHandCheckedNumbers:
    """手算核对：用后端同契约的 _dimension_summary 复算，钉住口径。

    抽查 deepseek-chat 的 correctness：报告A（0923）41/42 = 97.6%，
    报告B（1004）42/42 = 100.0%，差值 +2.4pt（页面应显示 +2.4）。
    """

    def _rate(self, doc: dict, model: str, dim: str):
        rows = {r["dimension"]: r for r in app._dimension_summary(doc, model)}
        return rows.get(dim)

    def test_chat_correctness_delta(self) -> None:
        if not (REPORT_A.exists() and REPORT_B.exists()):
            return
        a = self._rate(json.loads(REPORT_A.read_text(encoding="utf-8")), "deepseek-chat", "correctness")
        b = self._rate(json.loads(REPORT_B.read_text(encoding="utf-8")), "deepseek-chat", "correctness")
        assert (a["passed"], a["total"]) == (41, 42)
        assert (b["passed"], b["total"]) == (42, 42)
        delta = (b["pass_rate"] - a["pass_rate"]) * 100
        assert f"{delta:+.1f}" == "+2.4", f"手算应为 +2.4pt，实算 {delta:+.1f}"

    def test_relevance_regressed(self) -> None:
        """相关性维度两个模型都回退 12.5pt（8/8 → 7/8），负数必须显示负号。"""
        if not (REPORT_A.exists() and REPORT_B.exists()):
            return
        doc_a = json.loads(REPORT_A.read_text(encoding="utf-8"))
        doc_b = json.loads(REPORT_B.read_text(encoding="utf-8"))
        for model in ("deepseek-chat", "deepseek-pro"):
            a = self._rate(doc_a, model, "relevance")
            b = self._rate(doc_b, model, "relevance")
            assert (a["passed"], a["total"]) == (8, 8)
            assert (b["passed"], b["total"]) == (7, 8)
            delta = (b["pass_rate"] - a["pass_rate"]) * 100
            assert f"{delta:+.1f}" == "-12.5", f"{model} 手算应为 -12.5pt，实算 {delta:+.1f}"
