# -*- coding: utf-8 -*-
"""看板「维度速览 / 维度覆盖」契约测试。

背景：
    维度口径从「六类旧名」收敛成四类（准确性 / 指令遵循 / 安全 / 相关性），
    开放题 qa_open 从 correctness 拆出来单列 relevance。首页每份报告下面要
    直接给出各维度通过率与覆盖范围，否则「mock 93.3%（只跑了安全）」会被
    误读成比「mock 34.7%（全量）」强。

契约要点：
    * 看板侧的维度表与 src 侧一一对应（两边各自改了会被这里抓住）；
    * 老报告（没有 dimensions 段、cases 也没 dimension 字段）按分类重算，
      口径与新报告一致——qa_open 不会留在 correctness 里；
    * 只跑部分分类的报告，未覆盖的维度不出现，并标出覆盖范围；
    * 历史报告（9/14 那批）加载不报错且有维度数据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import dashboard.app as app
from src.datasets.schema import DIMENSIONS, DIMENSION_LABELS
from src.runner.dimension import DEFAULT_DIMENSION_BY_CATEGORY

# 看板自带数据副本：沙盒/CI 里也能跑，不依赖 reports/ 工作区目录。
DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"


def _cat_doc(*rows) -> dict:
    """构造一份只有 categories 段的老格式报告。"""
    return {"categories": {"m": [{"category": c, "total": t, "passed": p} for c, t, p in rows]}}


class TestDimensionTablesStayInSync:
    """看板与 runner 各有一份维度表，靠这里防止改了一边忘了另一边。"""

    def test_category_mapping_matches_runner(self) -> None:
        assert app.CATEGORY_DIMENSION == DEFAULT_DIMENSION_BY_CATEGORY

    def test_order_matches_schema_enum(self) -> None:
        assert app.DIMENSION_ORDER == DIMENSIONS

    def test_labels_match_schema(self) -> None:
        for dim in DIMENSIONS:
            assert app._BANNER_DIM_LABELS[dim] == DIMENSION_LABELS[dim], f"{dim} 的中文名两边不一致"

    def test_relevance_is_known_to_the_dashboard(self) -> None:
        """relevance 是新维度：看板必须认识它，否则会显示成英文原词。"""
        assert "relevance" in app._BANNER_DIM_LABELS
        assert "relevance" in app.DIMENSION_ORDER


class TestDimensionSummary:
    def test_safety_only_report_has_one_dimension(self) -> None:
        """只跑安全分类：维度速览只显示安全，不能凭空补三个 0%。"""
        rows = app._dimension_summary(_cat_doc(("safety_redteam", 30, 25)), "m")

        assert [r["dimension"] for r in rows] == ["safety"]
        assert rows[0]["label"] == "安全"
        assert rows[0]["pass_rate"] == pytest.approx(25 / 30)

    def test_qa_open_is_relevance_not_correctness(self) -> None:
        """开放题走裁判，必须从准确性里拆出来。"""
        rows = {r["dimension"]: r for r in app._dimension_summary(
            _cat_doc(("qa_open", 8, 7), ("math_reasoning", 20, 20)), "m")}

        assert set(rows) == {"relevance", "correctness"}
        assert rows["relevance"]["total"] == 8
        assert rows["correctness"]["total"] == 20  # 只有数学，不含 qa_open

    def test_same_dimension_categories_are_merged(self) -> None:
        """数学与事实问答都归准确性，维度行要合并而不是分成两行。"""
        rows = app._dimension_summary(
            _cat_doc(("math_reasoning", 20, 18), ("qa_zh", 22, 21)), "m")

        assert [(r["dimension"], r["total"], r["passed"]) for r in rows] == [
            ("correctness", 42, 39),
        ]

    def test_falls_back_to_cases_when_no_categories(self) -> None:
        doc = {
            "cases": [
                {"model": "m", "category": "json_extract", "passed": True},
                {"model": "m", "category": "json_extract", "passed": False},
                {"model": "other", "category": "safety_redteam", "passed": True},
            ]
        }
        rows = app._dimension_summary(doc, "m")

        assert [(r["dimension"], r["total"], r["passed"]) for r in rows] == [
            ("instruction_following", 2, 1),
        ]

    def test_falls_back_to_legacy_dimensions_segment(self) -> None:
        """连 categories 都没有时用 dimensions 段，旧维度名 format 归一成指令遵循。"""
        doc = {"dimensions": {"m": [{"dimension": "format", "total": 18, "passed": 18}]}}
        rows = app._dimension_summary(doc, "m")

        assert [(r["dimension"], r["label"], r["pass_rate"]) for r in rows] == [
            ("instruction_following", "指令遵循", 1.0),
        ]

    def test_rows_follow_dimension_order(self) -> None:
        doc = _cat_doc(
            ("safety_redteam", 1, 1),
            ("math_reasoning", 1, 1),
            ("json_extract", 1, 1),
            ("qa_open", 1, 1),
        )
        assert [r["dimension"] for r in app._dimension_summary(doc, "m")] == list(DIMENSIONS)

    def test_nothing_to_show_returns_empty(self) -> None:
        """取不到维度就返回空列表，前端不渲染该行，而不是画一排 0%。"""
        assert app._dimension_summary({}, "m") == []
        assert app._dimension_summary(None, "m") == []
        assert app._dimension_summary({"categories": {"m": []}}, "m") == []


class TestListCoverage:
    """首页列表条目必须带上维度速览所需的字段。"""

    def test_every_report_exposes_dimensions_and_coverage(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "REPORTS_DIR", DASHBOARD_DATA_DIR)
        for item in app._load_reports():
            assert "dimensions" in item, f"{item['file']} 缺 dimensions"
            assert item["dimension_coverage"] == [r["label"] for r in item["dimensions"]]

    def test_partial_coverage_is_visible_in_list(self, monkeypatch, tmp_path) -> None:
        """只跑安全的报告：覆盖范围必须只剩「安全」，否则又会被误读成全量。"""
        payload = {
            "summary": [{"model": "mock-baseline", "total": 30, "passed": 28,
                         "failed": 2, "pass_rate": 28 / 30, "p95_latency_ms": 0}],
            "case_count": 30,
            "categories": {"mock-baseline": [
                {"category": "safety_redteam", "total": 30, "passed": 28, "pass_rate": 28 / 30},
            ]},
        }
        (tmp_path / "report-safety-20260923-000000.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)

        item = app._load_reports()[0]
        assert item["dimension_coverage"] == ["安全"]
        assert [r["dimension"] for r in item["dimensions"]] == ["safety"]

    def test_old_reports_do_not_crash(self, monkeypatch) -> None:
        """9/14 那批：没有 dimensions 段、cases 也没有 dimension 字段。

        必须按 categories 重算出维度（而不是报错或整份落进「未标注」）。
        """
        monkeypatch.setattr(app, "REPORTS_DIR", DASHBOARD_DATA_DIR)
        old = [it for it in app._load_reports() if "20260914" in it["file"]]

        assert old, f"{DASHBOARD_DATA_DIR} 下找不到 9/14 的老报告，本测试失去意义"
        for item in old:
            assert item["dimensions"], f"{item['file']} 老报告没算出任何维度"
            assert "untagged" not in [r["dimension"] for r in item["dimensions"]]


class TestDimensionRowTemplate:
    """模板契约：JS 跑不了，至少保证渲染函数与注入数据都到位。"""

    def test_dimension_tables_are_injected(self) -> None:
        """占位符没被替换会让整段 JS 语法错误，页面白屏。"""
        assert "__DIM_LABELS__" not in app.PAGE
        assert "__CATEGORY_DIMENSION__" not in app.PAGE
        assert "__LEGACY_DIM_ALIASES__" not in app.PAGE
        assert '"relevance": "相关性"' in app.PAGE

    def test_model_overview_row_is_rendered(self) -> None:
        """首页副行：模型总览 chips（谁的成绩一眼可见）+ 覆盖标注。"""
        assert "const modelOverviewRowHtml" in app.PAGE
        assert "modelOverviewRowHtml(r)" in app.PAGE
        assert "覆盖：" in app.PAGE
        assert "全部四维" in app.PAGE
        assert 'class="dimrow"' in app.PAGE
        # mock 是管道基线，灰显但保留
        assert "dimmock" in app.PAGE

    def test_model_overview_row_has_style(self) -> None:
        assert "tr.dimrow td" in app.PAGE

    def test_detail_groups_use_dimension_key(self) -> None:
        """详情页维度分组走 caseDimOf（已知分类按分类重算），老报告才不会落进「未标注」。"""
        assert "const dimKeyOf" in app.PAGE
        assert "const caseDimOf" in app.PAGE
        assert "const dim = caseDimOf(c);" in app.PAGE

    def test_known_category_beats_stored_dimension(self) -> None:
        """老报告的 cases 里存着旧打标（qa_open → correctness），必须让分类说话。

        否则详情页显示「准确性 50 题」、首页重算是 42 题，又变回两套数字。
        """
        assert "if (CATEGORY_DIMENSION[cat]) return CATEGORY_DIMENSION[cat];" in app.PAGE


class TestCaseDimension:
    """单条用例归维度：已知分类说话，老报告里存的旧打标不算数。"""

    def test_known_category_beats_legacy_stored_dimension(self) -> None:
        """老报告把 qa_open 记在 correctness 里，重算必须落到 relevance。"""
        assert app._case_dimension({"category": "qa_open", "dimension": "correctness"}) == "relevance"

    def test_unknown_category_keeps_declared_dimension(self) -> None:
        assert app._case_dimension({"category": "custom_x", "dimension": "safety"}) == "safety"

    def test_legacy_dimension_name_is_normalized(self) -> None:
        assert app._case_dimension({"dimension": "format"}) == "instruction_following"

    def test_detail_and_home_agree_on_the_real_report(self) -> None:
        """真报告端到端：详情页（按 cases）与首页（按 categories 重算）必须同一套维度。"""
        path = DASHBOARD_DATA_DIR / "report-full-regression-20260923-143903.json"
        if not path.is_file():
            pytest.skip(f"{path} 不在，跳过")
        doc = json.loads(path.read_text(encoding="utf-8"))
        model = "deepseek-chat"

        home = {r["dimension"] for r in app._dimension_summary(doc, model)}
        detail = {
            app._case_dimension(c)
            for c in (doc.get("cases") or [])
            if c.get("model") == model and app._case_dimension(c)
        }
        assert home == detail
        assert home == {"correctness", "instruction_following", "safety", "relevance"}


class TestSingleSourceOfTruth:
    """展示层只允许一套口径：所有地方都从同一个聚合函数取数。

    曾经 banner 直接读报告里存的 ``dimensions`` 段（生成时的旧口径快照），
    于是详情页看到「correctness 50 题」、首页看到 42 题，两套数字互相打脸。
    """

    def test_js_has_one_dimension_aggregator(self) -> None:
        assert "function dimensionSummary" in app.PAGE

    def test_banner_goes_through_the_aggregator(self) -> None:
        """banner 必须从 dimensionSummary 取数，不再自己读 dimensions 段。

        多模型时逐个真实模型都算一遍（取最低的那个并标出模型名），
        所以现在是循环里的 ``dimensionSummary(currDoc, row.model)``。
        """
        assert "dimensionSummary(currDoc, row.model)" in app.PAGE
        assert "dimensionSummary(currDoc, representative)" not in app.PAGE, (
            "「挑一个代表模型」的旧写法已废弃：多模型的成绩要都写出来，谁弱要标得清"
        )
        assert "findCandidates(currDoc.dimensions)" not in app.PAGE

    def test_dimension_order_is_injected(self) -> None:
        assert "__DIMENSION_ORDER__" not in app.PAGE
        assert '"relevance"]' in app.PAGE or '["correctness", "instruction_following", "safety", "relevance"]' in app.PAGE

    def test_weakest_dimension_reuses_homepage_summary(self) -> None:
        """后端同一条约束：_weakest_dimension 是 _dimension_summary 的消费者。"""
        doc = {
            "categories": {"m": [
                {"category": "math_reasoning", "total": 20, "passed": 20},
                {"category": "qa_open", "total": 8, "passed": 2},
            ]},
            "dimensions": {"m": [{"dimension": "correctness", "total": 50, "passed": 30}]},
        }
        rows = app._dimension_summary(doc, "m")
        weakest = app._weakest_dimension(doc, "m")

        assert [r["dimension"] for r in rows] == ["correctness", "relevance"]
        # banner 的最弱维度 = 速览里最弱的那一行，同一份结果
        assert weakest == (rows[-1]["label"], rows[-1]["pass_rate"])
