# -*- coding: utf-8 -*-
"""RAG 的「考点 / 判定方式 / 结果」三层次 + 顶部指标汇总契约。

背景：
    逐题明细原先把「判定器名」当成「指标」显示（`metrics: ["contains"]`）。
    判定器 = **怎么判**（技术手段）、指标 = **结果数字**、考点 = **想测什么能力**
    ——三个层次挤在一列里，看报告的人分不清。本单把它们分开：
        * 用例集新增 ``focus`` 字段（考点）；
        * 结果 JSON 新增 ``metrics_summary``（四项指标数字）；
        * #rag 页顶部摆四项汇总，逐题明细拆成「考点 / 判定方式 / 结果」三列。

红线：``rag_eval/datasets/rag_cases.jsonl`` 的判据字段一字不改，只新增 focus；
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

DATASET = Path(__file__).resolve().parent.parent / "rag_eval" / "datasets" / "rag_cases.jsonl"
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
        # 原先写死 13 条；rag-014~016（诱导编造）与 rag-017~022（加难度六题）之后是 22 条。
        # 条数不该随每次加题就要改 → 只守「不少于当初那批」这个真风险。
        assert len(cases) >= 13, "用例数变少了，是不是误删了题？"
        for c in cases:
            assert "focus" in c, f"{c['id']} 缺 focus 字段"
            assert c["focus"] in FOCUS_WHITELIST, f"{c['id']} 的 focus 非法：{c['focus']!r}"

    def test_focus_distribution(self) -> None:
        """考点分布随加难度变化，这里钉住当前值：端到端 11 / 忠实度 7 / 无答案诚实 4。

        改这个值前先确认是「加了题」还是「改动了既有题的 focus」——后者要说明理由。
        """
        counts: dict[str, int] = {}
        for c in _cases():
            counts[c["focus"]] = counts.get(c["focus"], 0) + 1
        assert counts == {"end_to_end": 11, "faithfulness": 7, "honesty": 4}, counts

    def test_judging_fields_untouched(self) -> None:
        """判据是红线：除 focus 外字段不许动（缺字段同样要红）。"""
        for c in _cases():
            for key in ("id", "metrics", "prompt", "gold_doc_ids", "note"):
                assert key in c, f"{c['id']} 丢了字段 {key}"
            assert c["category"] == "rag_cases", f"{c['id']} 的 category 被改了"
            assert isinstance(c["metrics"], list) and c["metrics"], f"{c['id']} 的 metrics 被改了"

    def test_focus_matches_note(self) -> None:
        """focus 必须与 note 里写的归类一致——不能题面写着诚实题却标了端到端。

        rag-014 起换成「加难度·」前缀的新格式（写的是难度意图，不是三类归类），
        这条口径只对旧格式（note 以「占位题」开头）用例生效——对它们的护栏强度不变。
        """
        tag_by_focus = {"end_to_end": "端到端命中", "faithfulness": "忠实度", "honesty": "无答案诚实"}
        legacy = [c for c in _cases() if c["note"].startswith("占位题")]
        assert len(legacy) >= 13, "旧格式用例（占位题）是不是被删了？"
        for c in legacy:
            assert tag_by_focus[c["focus"]] in c["note"], f"{c['id']} 的 note 与 focus 不符：{c['note']}"


class TestMetricsSummary:
    """四项指标汇总。"""

    def test_five_items_present(self) -> None:
        """五项指标：命中率@2 与 召回率@2 是**两项**，名字都不许再叫 Recall。"""
        rows = [
            _row("rag-001", "end_to_end", "通过"),
            _row("rag-011", "honesty", "通过"),
            _row("rag-007", "faithfulness", "通过"),
        ]
        ms = rag_focus_summary(rows, {"recall_at_k": 1.0, "k": 2, "scored": 10, "hit": 10})
        for key in ("recall_at_k", "recall_ratio_at_k", "end_to_end", "honesty", "faithfulness"):
            assert key in ms, f"缺指标项 {key}"
            assert ms[key]["label"], f"{key} 缺标签"
        assert ms["end_to_end"]["value"] == 1.0
        assert ms["end_to_end"]["scored"] == 3
        assert ms["honesty"]["value"] == 1.0
        # 名字写准：这是**命中率**（Hit Rate@2），不是召回率
        assert ms["recall_at_k"]["label"] == "检索层 命中率@2"
        assert ms["recall_ratio_at_k"]["label"] == "检索层 召回率@2"
        assert "Recall" not in json.dumps(ms, ensure_ascii=False), "指标标签里不许再出现 Recall 这个错名字"

    def test_missing_recall_ratio_is_not_guessed(self) -> None:
        """老结果没有召回率字段 → value 为 None + 给出原因，不许拿命中率顶替。"""
        rows = [_row("rag-001", "end_to_end", "通过")]
        ms = rag_focus_summary(rows, {"recall_at_k": 1.0, "k": 2, "scored": 1, "hit": 1})
        assert ms["recall_ratio_at_k"]["value"] is None, "没有这个数就是没有，不猜"
        assert "需重跑" in ms["recall_ratio_at_k"]["reason"]
        assert ms["recall_at_k"]["value"] == 1.0, "命中率照旧有值"

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
