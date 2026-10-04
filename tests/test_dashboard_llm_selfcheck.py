# -*- coding: utf-8 -*-
"""#llm 报告列表：纯 mock 的「框架自检报告」折叠成底部统计行。

背景：
    cd.yml 在每次合并后自动跑一遍并归档，产物是 ``report-cd-*.json``，
    里面所有模型都是 ``mock-*``——那是**管道心跳**，证明评测框架自己还活着，
    不是模型成绩。它们逐条占行会把真报告挤下去（一度占掉列表的一大半）。

契约要点：
    * 判定：报告里**所有**模型都是 mock-* 才算自检；带真实模型的（哪怕也带
      mock-baseline 对照行）原样留在主列表；
    * 主列表只渲染非自检报告，自检折叠成底部一行「另有 N 次框架自检…· 展开」；
    * 编号仍按全量报告的序号走，折叠不让编号漂移（#63 永远是 #63）；
    * 「共 N 份报告」等计数口径不变（自检仍算在总数里）。
"""

from __future__ import annotations

from pathlib import Path

import dashboard.app as app

DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"


def _report_list_body() -> str:
    """取 renderList 全文（含模板里的渲染逻辑）：列表是运行时拼的，只能查源码。"""
    body = app.PAGE[app.PAGE.index("async function renderList"):]
    return body[:body.index("\n// 详情页属性条")]


def _load(monkeypatch):
    monkeypatch.setattr(app, "REPORTS_DIR", DASHBOARD_DATA_DIR)
    return app._load_reports()


def _is_mock_model(name) -> bool:
    """复用后端口径（``dashboard.app._is_mock``）：mock-* 即对照组/管道基线。"""
    return app._is_mock(name)


def _is_mock_only(report: dict) -> bool:
    """与前端 isSelfCheckReport 同口径的 Python 复算。"""
    rows = [x.get("model") for x in (report.get("model_rows") or [{"model": report.get("model")}])]
    models = [m for m in rows if m]
    return bool(models) and all(_is_mock_model(m) for m in models)


class TestSelfCheckClassification:
    """分类必须互斥且穷尽，否则会有报告既不在主列表也不在统计行。"""

    def test_partition_covers_every_report(self, monkeypatch) -> None:
        reports = _load(monkeypatch)
        assert reports, "数据目录里没有报告，分类测试失去意义"
        selfcheck = [r for r in reports if _is_mock_only(r)]
        main = [r for r in reports if not _is_mock_only(r)]
        assert len(selfcheck) + len(main) == len(reports)
        assert selfcheck, "数据里没有纯 mock 报告，折叠逻辑无从验证"

    def test_real_model_reports_stay_in_main_list(self, monkeypatch) -> None:
        """带真实模型的报告（含带 mock-baseline 对照行的）不许被折叠掉。"""
        reports = _load(monkeypatch)
        for r in reports:
            if _is_mock_only(r):
                continue
            models = [x.get("model") for x in (r.get("model_rows") or [])] or [r.get("model")]
            assert any(m and not _is_mock_model(m) for m in models), (
                f"{r.get('file')} 被留在主列表，但里面没有真实模型"
            )


class TestListRendering:
    """主列表只渲染真实报告；自检进底部统计行。"""

    def test_main_list_excludes_selfcheck(self) -> None:
        body = _report_list_body()
        assert "reports.filter(isSelfCheckReport)" in body, "列表要先分出自检报告"
        assert "main.map(" in body, "主列表必须只遍历非自检报告"

    def test_summary_row_wording(self) -> None:
        body = app.PAGE[app.PAGE.index("async function renderList"):]
        assert "次框架自检" in body, "底部缺「另有 N 次框架自检」统计行"
        assert "非模型成绩" in body, "统计行要点明自检不是模型成绩，否则仍会被误读"
        assert "· 展开" in body, "统计行要能展开看明细"

    def test_summary_row_keeps_detail_entry(self) -> None:
        """展开后每行仍能进详情页（data-go 与 rowlink 都要在）。"""
        body = app.PAGE[app.PAGE.index("async function renderList"):]
        tail = body[body.index("次框架自检"):]
        assert 'class="rowlink"' in tail and "data-go=" in tail, "展开区的行不能点进详情就失去意义"

    def test_numbering_not_renumbered(self) -> None:
        """编号按全量序号：折叠只影响显示，不该让「#55」指到别的报告。"""
        body = app.PAGE[app.PAGE.index("async function renderList"):]
        assert "reports.indexOf(r)" in body, "折叠后必须按全量列表定位序号，不能就地重排编号"

    def test_total_count_self_consistent(self) -> None:
        """统计行：真实 X 份 + 自检 Y 次 = 全量报告数（两个数字必须对得上）。"""
        body = app.PAGE[app.PAGE.index("async function renderList"):]
        assert "真实评测报告 ${main.length} 份" in body, "统计行要给出真实评测报告份数"
        assert "框架自检 ${selfChecks.length} 次" in body, "统计行要给出自检次数"
