# -*- coding: utf-8 -*-
"""#llm 页头与门户文案：所有数字/模型名都来自「成绩源报告」。

背景：
    最新的一份报告往往不是真实评测——``cd.yml`` 每次合并后自动跑一遍，
    产物 ``report-cd-*`` 里只有 ``mock-baseline``。页头取 ``reports[0]`` 就会把
    「mock-baseline 37.8%」当成实验室结论（门户修过的病，#llm 页头没修）。

契约要点：
    * 选源唯一：门户卡片与 #llm 页头（banner / 统计行 / 统计卡）共用
      ``scoringReports()``，不许各算一套（dimensionSummary 的教训）；
    * banner 写出**每个**真实模型各自的成绩（不是区间、不是单一代表模型）；
    * 统计行区分「真实评测 X 份 / 自检 Y 次」，且 X + Y = 全量报告数；
    * 一份真实报告都没有时 banner 不渲染——宁可没有，不拿 mock 充数；
    * 门户简介段与局限块的模型名来自数据，不许写死我们的模型名。
"""

from __future__ import annotations

import json
from pathlib import Path

import dashboard.app as app

DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"


def _load(monkeypatch):
    monkeypatch.setattr(app, "REPORTS_DIR", DASHBOARD_DATA_DIR)
    return app._load_reports()


def _real_reports(reports):
    """与前端 scoringReports 同口径：只保留含真实模型的报告（保持时间倒序）。"""
    out = []
    for r in reports:
        rows = r.get("model_rows") or [{"model": r.get("model")}]
        models = [x.get("model") for x in rows if x and x.get("model")]
        if any(not app._is_mock(m) for m in models):
            out.append(r)
    return out


def _list_body() -> str:
    body = app.PAGE[app.PAGE.index("async function renderList"):]
    return body[:body.index("\n// 详情页属性条")]


def _banner_body() -> str:
    body = app.PAGE[app.PAGE.index("async function computeBannerData"):]
    return body[:body.index("\nfunction renderBanner")]


class TestSingleSourceOfScoring:
    """选源必须唯一，且门户与 #llm 共用。"""

    def test_scoring_reports_helper_exists(self) -> None:
        assert "function scoringReports(" in app.PAGE, "缺共用的成绩源筛选函数"
        assert "function pickScoringReport(" in app.PAGE
        assert "scoringReports(reports)[0]" in app.PAGE, "pickScoringReport 必须复用同一份实现"

    def test_banner_uses_scoring_pool(self) -> None:
        body = _banner_body()
        assert "scoringReports(reports)" in body, "banner 不能再用 reports[0] 取数"
        assert "if (!pool.length) return null" in body, "没有真实报告时 banner 必须整体不渲染"

    def test_list_header_uses_scoring_report(self) -> None:
        body = _list_body()
        assert "pickScoringReport(reports)" in body, "#llm 页头必须复用门户同一个选源"

    def test_portal_and_list_share_one_pick(self) -> None:
        """两处引用的必须是同一个函数（不是各写一遍过滤条件）。"""
        assert app.PAGE.count("scoringReports(") >= 2, "门户与 #llm 页头应共用同一个选源函数"


class TestHeaderNumbers:
    """页头数字口径。"""

    def test_stats_row_splits_real_and_selfcheck(self) -> None:
        body = _list_body()
        assert "真实评测报告 ${main.length} 份" in body
        assert "框架自检 ${selfChecks.length} 次" in body

    def test_case_count_deduped_by_models(self) -> None:
        """用例数是去重的（除以模型数），不是「用例 × 模型」的累计行数。"""
        body = _list_body()
        assert "modelsOf(scoring).length" in body, "用例数要除以模型数去重，否则 294 会当成 294 条用例"

    def test_score_card_lists_every_real_model(self) -> None:
        """卡3 每个真实模型各占一行，不用 min~max 区间。"""
        body = _list_body()
        assert "scoringRows.map(" in body
        assert "Math.min(...topRates)" not in body, "区间写法把模型间对比藏起来了，已废弃"

    def test_banner_writes_every_real_model(self) -> None:
        assert "data.models || []" in app.PAGE, "banner 要遍历所有真实模型"
        assert "最新真实评测 ·" in app.PAGE, "banner 开头要写明这是真实评测及其日期"

    def test_no_mock_figure_in_header(self) -> None:
        """页头不许再出现 mock 相关的取数写法。"""
        body = _banner_body()
        assert "const curr = pool[0]" in body, "banner 的当前报告必须来自成绩源序列"
        assert "representative" not in body, "「挑一个代表模型」的旧写法已废弃"


class TestPortalModelNamesNotHardcoded:
    """门户文案的模型名来自数据。"""

    @staticmethod
    def _portal_body() -> str:
        body = app.PAGE[app.PAGE.index("async function renderPortal"):]
        return body[:body.index("$app.innerHTML")]

    def test_intro_not_hardcoded(self) -> None:
        assert "横向对比 <b>deepseek-chat</b> 与 <b>deepseek-pro</b>" not in app.PAGE, (
            "简介段还在写死两个模型名"
        )

    def test_scale_line_not_hardcoded(self) -> None:
        assert "2 个模型（deepseek-chat / deepseek-pro）" not in app.PAGE, (
            "局限块「规模」行还在写死模型数与名字"
        )

    def test_three_places_share_real_names(self) -> None:
        body = self._portal_body()
        assert "realNames" in body, "模型名单要有唯一来源"
        assert "${modelsHtml}" in app.PAGE, "简介段要用同一份名单"
        assert "${scaleModels}" in app.PAGE, "局限块要用同一份名单"

    def test_fallback_wording_when_no_real_model(self) -> None:
        """没有真实模型数据时显示通用说法，不是空白也不是假名字。"""
        body = self._portal_body()
        assert "多个大模型" in body
        assert "暂无真实模型数据" in body


class TestDataContract:
    """当前数据下这些口径能算出什么（防止改完静默变空）。"""

    def test_scoring_pool_not_empty(self, monkeypatch) -> None:
        reports = _load(monkeypatch)
        pool = _real_reports(reports)
        assert pool, "数据里没有含真实模型的报告，页头会全空"
        assert len(pool) < len(reports), "数据里没有纯 mock 报告，折叠与自检计数无从验证"

    def test_scoring_report_has_datasets(self, monkeypatch) -> None:
        """「用例集 · N 个数据集」需要列表接口带 datasets 字段。"""
        reports = _load(monkeypatch)
        curr = _real_reports(reports)[0]
        assert curr.get("datasets"), "列表条目缺 datasets 字段，页头会显示「数据集信息缺失」"
        assert curr.get("case_count"), "列表条目缺 case_count"
