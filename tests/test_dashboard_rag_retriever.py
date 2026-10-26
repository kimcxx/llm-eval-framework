# -*- coding: utf-8 -*-
"""RAG「检索后端」卡片的展示契约：降级要显式标，异常原文不上页面。

背景：
    ``Retriever.describe()`` 降级时会把 ``init_error``（形如
    ``ModuleNotFoundError: No module named 'sentence_transformers'``）拼进说明串，
    看板又把这一串直接渲染到 #rag 页「检索后端」卡片上——页面上糊着一段 Python
    异常原文，读者看到的是「一堆报错」，而不是「现在是字符后端在跑、结论不是语义级的」。

契约要点（kb-007 同款口径：降级必须显式标注，不沉默）：
    * 卡片的标签是人话「字符相似度（降级）」，带「降级」标记与降级 hint；
    * 卡片上**不得出现** ``ModuleNotFoundError`` / ``No module named`` 等异常原文
      ——老结果 JSON 里已经存了带原文的串，展示层必须能挡住；
    * 异常原文只留在报告 JSON 的 notes / ``init_error`` 里供排查。
"""

from __future__ import annotations

import json
from pathlib import Path

import dashboard.app as app
from rag_eval.knowledge import load_knowledge_base
from rag_eval.retriever import Retriever

RAW_WITH_ERROR = "retriever(字符后端，原因：ModuleNotFoundError: No module named 'sentence_transformers')"
FORBIDDEN = ("ModuleNotFoundError", "No module named")


def _no_raw_error(text: str, where: str) -> None:
    for mark in FORBIDDEN:
        assert mark not in text, f"{where} 不该出现异常原文「{mark}」：{text}"


class TestRetrieverView:
    """展示层把原始串翻成人话。"""

    def test_degraded_label_is_human_readable(self) -> None:
        view = app._rag_retriever_view(RAW_WITH_ERROR)
        assert view["retriever_degraded"] is True
        assert view["retriever_label"] == "字符相似度（降级）", view["retriever_label"]
        assert "降级" in view["retriever_label"]
        _no_raw_error(view["retriever_label"], "卡片标签")
        _no_raw_error(view["retriever_hint"], "卡片 hint")

    def test_plain_lexical_also_marked_degraded(self) -> None:
        """没写原因的纯字符后端（--no-embedding）同样是降级，别漏标。"""
        view = app._rag_retriever_view("retriever(字符后端（降级）)")
        assert view["retriever_degraded"] is True
        assert view["retriever_label"] == "字符相似度（降级）"

    def test_embedding_backend_not_degraded(self) -> None:
        view = app._rag_retriever_view("retriever(向量后端 paraphrase-multilingual-MiniLM-L12-v2)")
        assert view["retriever_degraded"] is False
        assert "向量检索" in view["retriever_label"]
        assert "paraphrase-multilingual-MiniLM-L12-v2" in view["retriever_label"]
        assert view["retriever_hint"] == ""

    def test_unknown_string_falls_back_without_error(self) -> None:
        """字段缺失/不认识时给「—」，不许把原文原样透传。"""
        view = app._rag_retriever_view("")
        assert view["retriever_degraded"] is False


class TestEntryDoesNotLeakRawError:
    """端到端：结果 JSON 里带原文，交给前端的数据也必须是干净的。"""

    def test_entry_label_clean_while_raw_kept(self, tmp_path: Path) -> None:
        payload = {
            "generated_at": "2026-10-26T16:26:00",
            "models": ["rag-deepseek-chat"],
            "top_k": 2,
            "retriever": RAW_WITH_ERROR,
            "knowledge_base": {"doc_count": 14, "path": "x", "provenance": {}},
            "retrieval": {"recall_at_k": 1.0, "k": 2, "scored": 10, "hit": 10},
            "summary": [{"model": "rag-deepseek-chat", "total": 13, "passed": 13, "failed": 0, "skipped": 0, "pass_rate": 1.0}],
            "cases": [],
            "notes": ["检索后端降级原因（异常原文）：ModuleNotFoundError: No module named 'sentence_transformers'"],
        }
        f = tmp_path / "result-20261026-162600.json"
        f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        entry = app._rag_entry(f)
        _no_raw_error(entry["retriever_label"], "API 的 retriever_label")
        assert entry["retriever_label"] == "字符相似度（降级）"
        assert entry["retriever_degraded"] is True
        # 原始串保留在数据里供排查（notes 里也有原文）
        assert entry["retriever"] == RAW_WITH_ERROR


class TestSourceKeepsErrorOutOfCard:
    """源头：describe() 不许再拼异常原文。"""

    def test_describe_without_error(self) -> None:
        r = Retriever(load_knowledge_base(), prefer_embedding=False)
        r._init_error = RAW_WITH_ERROR.split("：", 1)[-1]
        text = r.describe()
        assert "字符后端" in text
        _no_raw_error(text, "Retriever.describe()")
        # 原文仍在 init_error，供 run_eval 写进 notes
        assert r.init_error and "No module named" in r.init_error


class TestPageRendersCleanFields:
    """页面渲染用的是清洗后的字段，不是原始串。"""

    def test_card_uses_view_fields(self) -> None:
        assert "ragRetrieverStat" in app.PAGE, "检索后端卡片应走专门的渲染函数"
        assert "retriever_label" in app.PAGE, "卡片没用清洗后的标签字段"
        assert "latest.retriever ||" not in app.PAGE, "卡片还在直接渲染原始 retriever 串"
