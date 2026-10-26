# -*- coding: utf-8 -*-
"""模型层报告列表的「层别」契约测试。

背景：
    RAG 层的报告也写在 ``reports/`` 下（``report-rag-v1-*.json``，模型名
    ``rag-deepseek-chat`` 这类）。``_load_reports()`` 原本只排除 CI 的
    tests-summary/tests-detail，于是 RAG 报告被当成模型层报告——而且它是最新的，
    直接顶掉真正的模型层报告：门户「模型层评测」卡片显示 ``rag-mock-baseline``
    和 **NaN%**（RAG 报告里没有模型层要的通过率字段）。

契约要点：
    * 模型层的列表/卡片只吃模型层报告，文件名带 rag 的一律不进；
    * 排除规则只有一处（``_is_model_layer_report``），不许在调用点散写 if；
    * 页面上不该出现 NaN——缺字段/除零时显示「—」。
"""

from __future__ import annotations

import json
from pathlib import Path

import dashboard.app as app


def _model_report(model: str, stamp: str) -> dict:
    return {
        "started_at": f"2026-10-26T{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}",
        "case_count": 74,
        "repeat": 1,
        "datasets": ["qa_zh"],
        "summary": [
            {
                "model": model,
                "total": 74,
                "passed": 70,
                "failed": 4,
                "pass_rate": 0.946,
                "stability": 0.9,
                "p95_latency_ms": 1200,
            }
        ],
    }


def _rag_report(stamp: str) -> dict:
    return {
        "started_at": f"2026-10-26T{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}",
        "case_count": 13,
        "repeat": 1,
        "datasets": ["rag_cases"],
        "summary": [
            {
                "model": "rag-deepseek-chat",
                "total": 13,
                "passed": 13,
                "failed": 0,
                "pass_rate": 1.0,
            }
        ],
    }


class TestModelLayerOnly:
    """RAG 层的报告不能混进模型层的列表与卡片。"""

    def test_rag_report_not_in_model_list(self, tmp_path: Path, monkeypatch) -> None:
        reports = tmp_path
        (reports / "report-safety-judge-20261005-112926.json").write_text(
            json.dumps(_model_report("deepseek-chat", "112926"), ensure_ascii=False), encoding="utf-8"
        )
        (reports / "report-rag-v1-20261026-151434.json").write_text(
            json.dumps(_rag_report("151434"), ensure_ascii=False), encoding="utf-8"
        )
        monkeypatch.setattr(app, "REPORTS_DIR", reports)

        items = app._load_reports()
        files = [it["file"] for it in items]
        assert not any("rag" in f.lower() for f in files), f"RAG 报告混进了模型层列表：{files}"
        assert items[0]["file"] == "report-safety-judge-20261005-112926.json", "最近一项应是真模型层报告"
        assert items[0]["model"] == "deepseek-chat"

    def test_rag_selfcheck_report_also_excluded(self, tmp_path: Path, monkeypatch) -> None:
        """CI/CD 的 rag 冒烟报告同样不算（它正是页面 NaN 的那一份）。"""
        reports = tmp_path
        (reports / "report-cd-rag-20261026-063536.json").write_text(
            json.dumps(_rag_report("063536"), ensure_ascii=False), encoding="utf-8"
        )
        (reports / "report-cd-20261026-063536.json").write_text(
            json.dumps(_model_report("mock-baseline", "063536"), ensure_ascii=False), encoding="utf-8"
        )
        monkeypatch.setattr(app, "REPORTS_DIR", reports)

        files = [it["file"] for it in app._load_reports()]
        assert files == ["report-cd-20261026-063536.json"]

    def test_exclusion_rule_is_one_function(self) -> None:
        """排除规则必须收敛在一处——散写 if 迟早漏掉一种文件名。"""
        assert app._is_model_layer_report("report-rag-v1-20261026-151434.json") is False
        assert app._is_model_layer_report("report-ci-rag-20261026-151355.json") is False
        assert app._is_model_layer_report("report-cd-rag-20261026-063536.json") is False
        assert app._is_model_layer_report("report-full-v3-20260922-190023.json") is True
        assert app._is_model_layer_report("tests-summary.json") is False
        assert app._is_model_layer_report("tests-detail.json") is False


class TestNoNaNOnPage:
    """缺字段/除零时显示「—」而不是 NaN%。"""

    def test_pct_guards_non_finite(self) -> None:
        assert "Number.isFinite(Number(x))" in app.PAGE, "pct 没做有限性判断，缺字段会渲染成 NaN%"
        assert "'—'" in app.PAGE, "非数字时应显示「—」"
