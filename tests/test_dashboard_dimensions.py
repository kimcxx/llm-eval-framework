# -*- coding: utf-8 -*-
"""看板「维度速览 / 维度覆盖」契约测试（2026-10-28 按 Q8 裁决重写口径）。

背景：
    维度口径从「六类旧名」收敛成四类（准确性 / 指令遵循 / 安全 / 相关性），
    开放题 qa_open 从 correctness 拆出来单列 relevance。

Q8 裁决（2026-10-28）定死四条：
    * **一个真相源**：维度统计只算一次，落在报告 JSON 的 ``dimensions`` 段；
      看板**照抄，不重算**。以前看板按 categories 重算（本意是迁就老报告），
      于是报告存一份、页面算一份，同一份报告同一模型对不上（报告 correctness=51、
      页面 42），还会把未知分类（rag_cases）当维度键、冒出假维度；
    * **老报告标注、不重算**：报告里写明口径版本（``dimension_schema_version``），
      缺这个字段或版本对不上就是老报告，看板照抄并标注「口径为旧版」；
    * **分类 ≠ 维度**：不得把 category 名当维度键，认不出的一律进「未标注」，
      绝不新造维度名；
    * **护栏**：同一报告同一模型，页面维度数/通过数 == 报告 dimensions 段的值；
      且不得出现四类白名单之外的维度名。

另两条契约（与本裁决无关，继续守住）：
    * 看板侧的维度表与 src 侧一一对应（两边各自改了会被这里抓住）；
    * 只跑部分分类的报告，未覆盖的维度不出现，并标出覆盖范围。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import dashboard.app as app
from src.datasets.schema import DIMENSIONS, DIMENSION_LABELS, DIMENSION_SCHEMA_VERSION
from src.runner.dimension import DEFAULT_DIMENSION_BY_CATEGORY

# 看板自带数据副本：沙盒/CI 里也能跑，不依赖 reports/ 工作区目录。
DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"


def _dim_doc(*rows, version=DIMENSION_SCHEMA_VERSION) -> dict:
    """构造一份带 ``dimensions`` 段（真相源）的报告。"""
    doc = {
        "dimensions": {"m": [
            {"dimension": d, "total": t, "passed": p} for d, t, p in rows
        ]},
    }
    if version:
        doc["dimension_schema_version"] = version
    return doc


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

    def test_schema_version_matches_src(self) -> None:
        """口径版本号两边一致：报告写的是 src 那份，看板认的是自己那份。"""
        assert app.DIMENSION_SCHEMA_VERSION == DIMENSION_SCHEMA_VERSION

    def test_whitelist_is_the_four_dimensions_plus_untagged(self) -> None:
        assert app.KNOWN_DIMENSIONS == frozenset(DIMENSIONS) | {"untagged"}


class TestDimensionSummary:
    """维度速览的数据源：**报告里的 dimensions 段，照抄**。"""

    def test_safety_only_report_has_one_dimension(self) -> None:
        """只跑安全分类：维度速览只显示安全，不能凭空补三个 0%。"""
        rows = app._dimension_summary(_dim_doc(("safety", 30, 25)), "m")

        assert [r["dimension"] for r in rows] == ["safety"]
        assert rows[0]["label"] == "安全"
        assert rows[0]["pass_rate"] == pytest.approx(25 / 30)

    def test_qa_open_is_relevance_not_correctness(self) -> None:
        """开放题走裁判，不混进准确性——**由报告侧打标决定，看板照抄**。

        以前这条测的是「看板按分类把 qa_open 重算成 relevance」；现在报告生成时
        就按用例自带标签聚合好了，看板只负责不把它改回去。
        """
        rows = {r["dimension"]: r for r in app._dimension_summary(
            _dim_doc(("relevance", 8, 7), ("correctness", 20, 20)), "m")}

        assert set(rows) == {"relevance", "correctness"}
        assert rows["relevance"]["total"] == 8
        assert rows["correctness"]["total"] == 20  # 只有数学，不含 qa_open

    def test_page_matches_report_segment_verbatim(self) -> None:
        """裁决④：页面维度数/通过数**必须等于**报告 dimensions 段的值。

        categories 与 cases 故意给成另一套数——谁要是把「按分类重算」改回来，
        这里立刻红（这就是本条的阴性对照）。
        """
        doc = {
            "dimension_schema_version": DIMENSION_SCHEMA_VERSION,
            "dimensions": {"m": [
                {"dimension": "correctness", "total": 51, "passed": 48},
                {"dimension": "relevance", "total": 12, "passed": 9},
            ]},
            # 干扰项：分类口径下算出来是 42 / 13，与真相源不同
            "categories": {"m": [
                {"category": "math_reasoning", "total": 42, "passed": 40},
                {"category": "rag_cases", "total": 13, "passed": 1},
            ]},
            "cases": [
                {"model": "m", "category": "math_reasoning", "dimension": "correctness", "passed": True},
            ],
        }
        rows = app._dimension_summary(doc, "m")

        assert [(r["dimension"], r["total"], r["passed"]) for r in rows] == [
            ("correctness", 51, 48),
            ("relevance", 12, 9),
        ]

    def test_category_never_becomes_a_dimension(self) -> None:
        """裁决③：分类 ≠ 维度，未知分类（rag_cases）不得冒充成一个维度。"""
        doc = {
            "dimension_schema_version": DIMENSION_SCHEMA_VERSION,
            "dimensions": {"m": [{"dimension": "correctness", "total": 98, "passed": 95}]},
            "categories": {"m": [{"category": "rag_cases", "total": 13, "passed": 1}]},
        }
        rows = app._dimension_summary(doc, "m")

        assert [r["dimension"] for r in rows] == ["correctness"], "不该冒出 rag_cases 维度"
        assert "rag_cases" not in [r["dimension"] for r in rows]

    def test_unknown_dimension_falls_into_untagged(self) -> None:
        """报告里出现认不出的维度名：归入「未标注」，不新造维度名。"""
        rows = app._dimension_summary(_dim_doc(("some_new_thing", 5, 5)), "m")

        assert [r["dimension"] for r in rows] == ["untagged"]

    def test_falls_back_to_cases_only_without_dimensions_segment(self) -> None:
        """连 dimensions 段都没有（更老的报告）才按用例自带标签兜底。"""
        doc = {
            "cases": [
                {"model": "m", "category": "json_extract", "dimension": "instruction_following", "passed": True},
                {"model": "m", "category": "json_extract", "dimension": "instruction_following", "passed": False},
                {"model": "other", "category": "safety_redteam", "dimension": "safety", "passed": True},
            ]
        }
        rows = app._dimension_summary(doc, "m")

        assert [(r["dimension"], r["total"], r["passed"]) for r in rows] == [
            ("instruction_following", 2, 1),
        ]

    def test_legacy_dimension_name_is_normalized(self) -> None:
        """旧维度名 format 归一成指令遵循。"""
        doc = {"dimensions": {"m": [{"dimension": "format", "total": 18, "passed": 18}]}}
        rows = app._dimension_summary(doc, "m")

        assert [(r["dimension"], r["label"], r["pass_rate"]) for r in rows] == [
            ("instruction_following", "指令遵循", 1.0),
        ]

    def test_rows_follow_dimension_order(self) -> None:
        doc = _dim_doc(
            ("safety", 1, 1),
            ("correctness", 1, 1),
            ("instruction_following", 1, 1),
            ("relevance", 1, 1),
        )
        assert [r["dimension"] for r in app._dimension_summary(doc, "m")] == list(DIMENSIONS)

    def test_nothing_to_show_returns_empty(self) -> None:
        """取不到维度就返回空列表，前端不渲染该行，而不是画一排 0%。"""
        assert app._dimension_summary({}, "m") == []
        assert app._dimension_summary(None, "m") == []
        assert app._dimension_summary({"categories": {"m": []}}, "m") == []


class TestLegacySchema:
    """老报告的口径标注（裁决②）：照抄 + 标注，不重算。"""

    def test_report_without_version_is_legacy(self) -> None:
        assert app._dimension_schema_legacy({"dimensions": {}}) is True

    def test_report_with_current_version_is_not_legacy(self) -> None:
        doc = {"dimension_schema_version": DIMENSION_SCHEMA_VERSION}
        assert app._dimension_schema_legacy(doc) is False

    def test_legacy_report_numbers_are_still_copied(self) -> None:
        """旧口径也照抄报告里的数——不因为「口径旧」就自己重算一份。"""
        doc = {"dimensions": {"m": [{"dimension": "correctness", "total": 50, "passed": 30}]}}
        rows = app._dimension_summary(doc, "m")

        assert app._dimension_schema_legacy(doc) is True
        assert [(r["dimension"], r["total"]) for r in rows] == [("correctness", 50)]

    def test_list_entry_carries_legacy_flag(self, monkeypatch, tmp_path) -> None:
        payload = {
            "summary": [{"model": "mock-baseline", "total": 30, "passed": 28,
                         "failed": 2, "pass_rate": 28 / 30, "p95_latency_ms": 0}],
            "case_count": 30,
            "dimensions": {"mock-baseline": [
                {"dimension": "safety", "total": 30, "passed": 28, "pass_rate": 28 / 30},
            ]},
        }
        (tmp_path / "report-safety-20260923-000000.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)

        item = app._load_reports()[0]
        assert item["dimension_schema_legacy"] is True, "没有口径版本号的老报告要被标出来"


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
            "dimensions": {"mock-baseline": [
                {"dimension": "safety", "total": 30, "passed": 28, "pass_rate": 28 / 30},
            ]},
        }
        (tmp_path / "report-safety-20260923-000000.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)

        item = app._load_reports()[0]
        assert item["dimension_coverage"] == ["安全"]
        assert [r["dimension"] for r in item["dimensions"]] == ["safety"]

    def test_old_reports_do_not_crash(self, monkeypatch) -> None:
        """9/14 那批最老的报告：没有 dimensions 段、cases 也没有 dimension 字段。

        新口径下它们会落进「未标注」并被标成旧口径（**这是预期的**）——
        要求只剩下：不崩、有维度行、且明确标了旧口径，让读者别拿它跟新报告横比。
        """
        monkeypatch.setattr(app, "REPORTS_DIR", DASHBOARD_DATA_DIR)
        old = [it for it in app._load_reports() if "20260914" in it["file"]]

        assert old, f"{DASHBOARD_DATA_DIR} 下找不到 9/14 的老报告，本测试失去意义"
        for item in old:
            assert item["dimension_schema_legacy"] is True
            assert isinstance(item["dimensions"], list)


class TestDimensionRowTemplate:
    """模板契约：JS 跑不了，至少保证渲染函数与注入数据都到位。"""

    def test_dimension_tables_are_injected(self) -> None:
        """占位符没被替换会让整段 JS 语法错误，页面白屏。"""
        assert "__DIM_LABELS__" not in app.PAGE
        assert "__CATEGORY_DIMENSION__" not in app.PAGE
        assert "__LEGACY_DIM_ALIASES__" not in app.PAGE
        assert "__KNOWN_DIMENSIONS__" not in app.PAGE
        assert "__DIMENSION_SCHEMA_VERSION__" not in app.PAGE
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
        """详情页维度分组走 caseDimOf（**只用用例自带 dimension**）。"""
        assert "const dimKeyOf" in app.PAGE
        assert "const caseDimOf" in app.PAGE
        assert "const dim = caseDimOf(c);" in app.PAGE

    def test_js_does_not_recompute_from_category(self) -> None:
        """裁决③：JS 里不许再出现「按分类换算维度」的写法。"""
        assert "if (CATEGORY_DIMENSION[cat]) return CATEGORY_DIMENSION[cat];" not in app.PAGE
        assert "add(c.dimension || c.category" not in app.PAGE

    def test_legacy_schema_is_annotated_on_the_page(self) -> None:
        """裁决②：页面要能标出「口径为旧版」。"""
        assert "口径为旧版" in app.PAGE
        assert "const legacyDimBadge" in app.PAGE


class TestCaseDimension:
    """单条用例归维度：**用例自带标签说话，分类不算数**（分类 ≠ 维度）。"""

    def test_declared_dimension_beats_category(self) -> None:
        """用例自己声明的维度说了算，分类只是「它来自哪个数据集」。"""
        assert app._case_dimension({"category": "qa_open", "dimension": "relevance"}) == "relevance"

    def test_unknown_category_keeps_declared_dimension(self) -> None:
        assert app._case_dimension({"category": "custom_x", "dimension": "safety"}) == "safety"

    def test_legacy_dimension_name_is_normalized(self) -> None:
        assert app._case_dimension({"dimension": "format"}) == "instruction_following"

    def test_missing_tag_is_untagged_not_a_new_name(self) -> None:
        """没打标签 / 认不出：归入「未标注」，不拿分类名顶替。"""
        assert app._case_dimension({"category": "rag_cases"}) == "untagged"
        assert app._case_dimension({}) == "untagged"

    def test_detail_and_home_agree_on_the_real_report(self) -> None:
        """真报告端到端：页面（照抄）必须等于报告 dimensions 段，不再各算一套。"""
        path = DASHBOARD_DATA_DIR / "report-full-regression-20260923-143903.json"
        if not path.is_file():
            pytest.skip(f"{path} 不在，跳过")
        doc = json.loads(path.read_text(encoding="utf-8"))
        model = "deepseek-chat"

        reported = {
            app._dimension_key(r.get("dimension"))
            for r in ((doc.get("dimensions") or {}).get(model) or [])
        }
        rows = app._dimension_summary(doc, model)
        page = {r["dimension"] for r in rows}
        # 报告段缺失（更老的报告）时才按用例兜底，此时页面与用例标签同源
        if reported:
            assert page == reported, "页面必须与报告 dimensions 段一致"

        # 总数也必须一致（裁决④）
        for row in rows:
            src = next((r for r in ((doc.get("dimensions") or {}).get(model) or [])
                        if app._dimension_key(r.get("dimension")) == row["dimension"]), None)
            if src and src.get("total"):
                assert row["total"] == src["total"], f"{row['dimension']} 的样本量与报告不一致"


class TestSingleSourceOfTruth:
    """展示层只允许一套口径：**报告里算好，页面照抄**。"""

    def test_js_has_one_dimension_aggregator(self) -> None:
        assert "function dimensionSummary" in app.PAGE

    def test_banner_goes_through_the_aggregator(self) -> None:
        """banner 必须从 dimensionSummary 取数，不再自己读 dimensions 段。"""
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
        doc = _dim_doc(("correctness", 20, 20), ("relevance", 8, 2))
        rows = app._dimension_summary(doc, "m")
        weakest = app._weakest_dimension(doc, "m")

        assert [r["dimension"] for r in rows] == ["correctness", "relevance"]
        # banner 的最弱维度 = 速览里最弱的那一行，同一份结果
        assert weakest == (rows[-1]["label"], rows[-1]["pass_rate"])


class TestDimensionWhitelist:
    """裁决④：报告/页面上不得出现四类白名单之外的维度名。"""

    def test_no_stray_dimension_names_in_real_reports(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "REPORTS_DIR", DASHBOARD_DATA_DIR)
        for item in app._load_reports():
            for row in item["dimensions"]:
                assert row["dimension"] in app.KNOWN_DIMENSIONS, (
                    f"{item['file']} 出现白名单外的维度名：{row['dimension']}"
                )

    def test_unknown_names_go_to_untagged(self) -> None:
        assert app._dimension_key("rag_cases") == "untagged"
        assert app._dimension_key("qa_open") == "untagged", "分类名不是维度"
        assert app._dimension_key("") == "untagged"
        assert app._dimension_key(None) == "untagged"
