# -*- coding: utf-8 -*-
"""RAG 的「考点 / 判定方式 / 结果」三层次 + 顶部指标汇总契约。

背景：
    逐题明细原先把「判定器名」当成「指标」显示（`metrics: ["contains"]`）。
    判定器 = **怎么判**（技术手段）、指标 = **结果数字**、考点 = **想测什么能力**
    ——三个层次挤在一列里，看报告的人分不清。本单把它们分开：
        * 用例集新增 ``focus`` 字段（考点）；
        * 结果 JSON 新增 ``metrics_summary``（四项指标数字）；
        * #rag 页顶部摆四项汇总，逐题明细拆成「考点 / 判定方式 / 结果」三列。

红线：``datasets/rag_cases.jsonl`` 的判据字段一字不改，只新增 focus；
``src/`` 的判定逻辑不动——**这是展示层改动，成绩一分不能变**。
"""

from __future__ import annotations

import json
from pathlib import Path

import dashboard.app as app
from rag_eval.run_eval import (
    FOCUS_LABELS,
    JUDGE_MISSING_TEXT,
    rag_focus_summary,
)

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "rag_cases.jsonl"
FOCUS_WHITELIST = tuple(FOCUS_LABELS)


def _cases() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def _row(case_id: str, focus: str, status: str, metrics: list[dict] | None = None) -> dict:
    return {
        "id": case_id,
        "focus": focus,
        "status": status,
        "metrics": metrics or [{"name": "contains", "score": 1.0, "passed": True}],
    }


class TestDatasetFocus:
    """13 条用例必须全部带考点，且取值在白名单内。"""

    def test_every_case_has_focus_in_whitelist(self) -> None:
        cases = _cases()
        assert len(cases) == 13, f"用例数变了（{len(cases)}），本单只动 focus 字段"
        for c in cases:
            assert "focus" in c, f"{c['id']} 缺 focus 字段"
            assert c["focus"] in FOCUS_WHITELIST, f"{c['id']} 的 focus 非法：{c['focus']!r}"

    def test_focus_distribution(self) -> None:
        """考点分布 = 现有 note 的归类：端到端 6 / 忠实度 4 / 无答案诚实 3。"""
        counts: dict[str, int] = {}
        for c in _cases():
            counts[c["focus"]] = counts.get(c["focus"], 0) + 1
        assert counts == {"end_to_end": 6, "faithfulness": 4, "honesty": 3}, counts

    def test_judging_fields_untouched(self) -> None:
        """判据是红线：除 focus 外字段不许动（缺字段同样要红）。"""
        for c in _cases():
            for key in ("id", "metrics", "prompt", "gold_doc_ids", "note"):
                assert key in c, f"{c['id']} 丢了字段 {key}"
            assert c["category"] == "rag_cases", f"{c['id']} 的 category 被改了"
            assert isinstance(c["metrics"], list) and c["metrics"], f"{c['id']} 的 metrics 被改了"

    def test_focus_matches_note(self) -> None:
        """focus 必须与 note 里写的归类一致——不能题面写着诚实题却标了端到端。"""
        tag_by_focus = {"end_to_end": "端到端命中", "faithfulness": "忠实度", "honesty": "无答案诚实"}
        for c in _cases():
            assert tag_by_focus[c["focus"]] in c["note"], f"{c['id']} 的 note 与 focus 不符：{c['note']}"


class TestMetricsSummary:
    """四项指标汇总。"""

    def test_four_items_present(self) -> None:
        rows = [
            _row("rag-001", "end_to_end", "通过"),
            _row("rag-011", "honesty", "通过"),
            _row("rag-007", "faithfulness", "通过"),
        ]
        ms = rag_focus_summary(rows, {"recall_at_k": 1.0, "k": 2, "scored": 10, "hit": 10})
        for key in ("recall_at_k", "end_to_end", "honesty", "faithfulness"):
            assert key in ms, f"缺指标项 {key}"
            assert ms[key]["label"], f"{key} 缺标签"
        assert ms["end_to_end"]["value"] == 1.0
        assert ms["end_to_end"]["scored"] == 3
        assert ms["honesty"]["value"] == 1.0
        assert ms["recall_at_k"]["label"] == "检索层 Recall@2"

    def test_skipped_cases_excluded_from_denominator(self) -> None:
        """跳过不计入分母（三态口径）：3 条里 2 通过 1 跳过 → 2/2 而不是 2/3。"""
        rows = [
            _row("rag-001", "end_to_end", "通过"),
            _row("rag-002", "end_to_end", "失败"),
            _row("rag-003", "end_to_end", "跳过"),
        ]
        ms = rag_focus_summary(rows, {"recall_at_k": 0.5, "k": 2, "scored": 4, "hit": 2})
        assert ms["end_to_end"]["scored"] == 2
        assert ms["end_to_end"]["passed"] == 1
        assert ms["end_to_end"]["value"] == 0.5

    def test_faithfulness_skipped_without_judge(self) -> None:
        """judge 缺席 → 忠实度显式写「跳过」，**不给数字**（0/100% 都会被误读）。"""
        rows = [_row("rag-007", "faithfulness", "通过", [{"name": "contains", "score": 1.0, "passed": True}])]
        ms = rag_focus_summary(rows, {"recall_at_k": 1.0, "k": 2, "scored": 1, "hit": 1})
        item = ms["faithfulness"]
        assert item["skipped"] is True
        assert item["value"] is None, "judge 缺席时不许给数字"
        assert item["reason"] == JUDGE_MISSING_TEXT
        assert "跳过" in item["reason"]

    def test_faithfulness_scored_when_judge_ran(self) -> None:
        rows = [
            _row("rag-007", "faithfulness", "通过", [
                {"name": "contains", "score": 1.0, "passed": True},
                {"name": "judge", "score": 4.0, "passed": True},
            ]),
            _row("rag-008", "faithfulness", "失败", [
                {"name": "contains", "score": 1.0, "passed": True},
                {"name": "judge", "score": 2.0, "passed": False},
            ]),
        ]
        ms = rag_focus_summary(rows, {"recall_at_k": 1.0, "k": 2, "scored": 2, "hit": 2})
        item = ms["faithfulness"]
        assert item["skipped"] is False
        assert item["value"] == 0.5


class TestDashboardRendering:
    """看板：顶部四项 + 逐题明细三列。"""

    def test_summary_and_three_columns_rendered(self) -> None:
        assert "ragMetricsSummaryCard" in app.PAGE, "页面缺顶部指标汇总区"
        assert "ragMetricStat" in app.PAGE, "汇总项没有单独的渲染函数"
        assert "FOCUS_LABELS" in app.PAGE, "考点中文名映射缺失"
        # 「判定方式 / 结果」必须拆成两列，不能再由「判定」一列兼任
        assert "<th>考点</th><th>判定方式</th><th>结果</th>" in app.PAGE, "明细表头不是三列分开"
        assert "<th>判定</th>" not in app.PAGE, "旧的「判定」单列还在"

    def test_skipped_metric_shows_text_not_number(self) -> None:
        """跳过项渲染的是 reason 文案，且不给百分比。"""
        assert "m.skipped" in app.PAGE and "ragMetricStat" in app.PAGE
        assert JUDGE_MISSING_TEXT in app.PAGE, "页面缺「跳过（未配置裁判模型）」文案"
