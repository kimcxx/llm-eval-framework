"""看板 RAG 层的后端注入测试。

沿用 PR #51 栽过的教训：**测试不得依赖不入库的本机产物**。
`rag_eval/results/` 里的最新一份入库，但 archive/ 不入库，CI 沙盒里可能一份都没有——
所以逻辑一律用 tmp 里自造的数据验证。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard import app


def _result(total: int = 13, passed: int = 9) -> dict:
    return {
        "generated_at": "2026-10-26T12:00:00",
        "models": ["rag-deepseek-chat"],
        "top_k": 2,
        "retriever": "retriever(字符后端)",
        "knowledge_base": {
            "path": "rag_eval/fixtures/knowledge_base.jsonl",
            "doc_count": 14,
            "provenance": {"frozen_at": "2026-10-26", "status": "占位版"},
        },
        "report_file": "reports/report-rag-20261026-120000.json",
        "summary": [
            {
                "model": "rag-deepseek-chat",
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "skipped": 0,
                "pass_rate": passed / total,
            }
        ],
        "retrieval": {"k": 2, "scored": 10, "hit": 10, "recall_at_k": 1.0},
        "cases": [{"id": "rag-001", "status": "通过", "retrieved": ["kb-002"], "gold": ["kb-002"]}],
        "notes": ["被测对象是 mini-RAG 固定管道"],
    }


@pytest.fixture
def fake_rag_env(tmp_path: Path, monkeypatch):
    results = tmp_path / "results"
    (results / "archive").mkdir(parents=True)
    (results / "result-20261026-120000.json").write_text(
        json.dumps(_result(), ensure_ascii=False), encoding="utf-8"
    )
    (results / "result-20261025-120000.json").write_text(
        json.dumps(_result(passed=5), ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(app, "RAG_RESULTS_DIR", results)
    monkeypatch.setattr(app, "RAG_ARCHIVE_DIR", results / "archive")
    return results


class TestRagRuns:

    def test_latest_first(self, fake_rag_env) -> None:
        runs = app.load_rag_runs()
        assert len(runs) == 2
        assert runs[0]["file"] == "result-20261026-120000.json", "最新一次应排最前"
        assert runs[0]["time"] == "2026-10-26 12:00:00"
        assert runs[0]["model"] == "rag-deepseek-chat"

    def test_only_latest_carries_cases(self, fake_rag_env) -> None:
        """逐题明细只给最新一次（体积考虑），历史行不该带。"""
        runs = app.load_rag_runs()
        assert len(runs[0]["cases"]) == 1
        assert runs[1]["cases"] == []

    def test_retrieval_stats_carried(self, fake_rag_env) -> None:
        """recall@k 与通过率并列展示：答错时能分清是检索的锅还是生成的锅。"""
        run = app.load_rag_runs()[0]
        assert run["recall_at_k"] == 1.0
        assert run["recall_k"] == 2 and run["recall_scored"] == 10
        assert run["doc_count"] == 14
        assert run["provenance"]["status"] == "占位版"

    def test_missing_dir_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(app, "RAG_RESULTS_DIR", tmp_path / "nope")
        monkeypatch.setattr(app, "RAG_ARCHIVE_DIR", tmp_path / "nope" / "archive")
        assert app.load_rag_runs() == []

    def test_broken_file_skipped(self, fake_rag_env) -> None:
        (fake_rag_env / "result-bad.json").write_text("{不是 json", encoding="utf-8")
        assert len(app.load_rag_runs()) == 2, "坏文件应被跳过而不是让整页崩掉"


class TestRagPortalWiring:

    def test_page_has_rag_route_and_card(self) -> None:
        assert 'href="#rag"' in app.PAGE, "门户缺 RAG 层入口卡"
        assert "if (file === 'rag') return renderRag();" in app.PAGE, "路由没接 #rag"

    def test_portal_is_three_layers(self) -> None:
        assert "LLM &amp; Agent &amp; RAG 评测实验室" in app.PAGE

    def test_scope_no_longer_says_rag_uncovered(self) -> None:
        """RAG 已从「未覆盖」挪进已覆盖，但要写明只覆盖固定管道。"""
        assert "agentic RAG" in app.PAGE
