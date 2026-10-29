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
        # 新结果一定带 run_kind（rag_eval/run_eval.py 写的）；这里的 fixture 代表**正式跑**。
        # 「老结果没有这个字段」的兜底判定由 TestRunKindFallback 单独覆盖。
        "run_kind": "real",
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


def _selfcheck_result(passed: int = 0, total: int = 13) -> dict:
    """mock 自检跑的结果：0/13 是假模型的预期行为，不是成绩。"""
    data = _result(total=total, passed=passed)
    data["run_kind"] = "mock"
    data["models"] = ["rag-mock-baseline"]
    data["summary"] = [
        {
            "model": "rag-mock-baseline",
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "skipped": 0,
            "pass_rate": passed / total,
        }
    ]
    data["cases"] = [{"id": "rag-001", "status": "失败", "retrieved": ["kb-002"], "gold": ["kb-002"]}]
    return data


class TestSelfcheckIsNotAScore:
    """mock 自检跑是管道阴性对照，不能当「被测对象 / 最新成绩」展示。

    2026-10-26 工单：看板把 `rag-mock-baseline` 的 0/13 当成 RAG 层成绩，
    访问者（包括面试官）会读成「RAG 层全挂」——而 0/13 是假模型的预期行为，
    与模型层既有规矩（mock-* 不进通过率排名）直接冲突。
    """

    @staticmethod
    def _env(tmp_path: Path, monkeypatch, files: dict[str, dict]) -> Path:
        results = tmp_path / "results"
        (results / "archive").mkdir(parents=True)
        for name, payload in files.items():
            (results / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(app, "RAG_RESULTS_DIR", results)
        monkeypatch.setattr(app, "RAG_ARCHIVE_DIR", results / "archive")
        return results

    def test_latest_skips_mock_runs(self, tmp_path: Path, monkeypatch) -> None:
        self._env(
            tmp_path,
            monkeypatch,
            {
                "result-20261026-120000.json": _result(),
                "result-20261026-140000.json": _selfcheck_result(),
            },
        )
        runs = app.load_rag_runs()
        assert runs[0]["model"] == "rag-mock-baseline", "历史表按时间倒序，自检跑要看得见"
        assert runs[0]["is_mock"] is True
        latest = app.pick_latest_real_run(runs)
        assert latest is not None, "有真跑就必须取到真跑，不能因为自检跑更新就顶掉"
        assert latest["model"] == "rag-deepseek-chat"
        assert latest["is_mock"] is False

    def test_cases_follow_the_real_run(self, tmp_path: Path, monkeypatch) -> None:
        """逐题明细跟着真跑走：0/13 的自检明细没有可看的东西。"""
        self._env(
            tmp_path,
            monkeypatch,
            {
                "result-20261026-120000.json": _result(),
                "result-20261026-140000.json": _selfcheck_result(),
            },
        )
        runs = app.load_rag_runs()
        latest = app.pick_latest_real_run(runs)
        assert len(latest["cases"]) == 1, "真跑那份要带逐题明细"
        assert runs[0]["cases"] == [], "自检跑不该占用逐题明细"

    def test_only_selfcheck_has_no_score(self, tmp_path: Path, monkeypatch) -> None:
        """只有自检跑时，页面显示「暂无真实模型成绩」而不是把 0/13 当成绩。"""
        self._env(tmp_path, monkeypatch, {"result-20261026-140000.json": _selfcheck_result()})
        runs = app.load_rag_runs()
        assert len(runs) == 1
        assert app.pick_latest_real_run(runs) is None

    def test_mock_prefix_stripped_before_check(self) -> None:
        assert app._is_rag_mock(["rag-mock-baseline"]) is True
        assert app._is_rag_mock(["rag-deepseek-chat"]) is False
        assert app._is_rag_mock([]) is False

    def test_page_marks_selfcheck_runs(self) -> None:
        assert "自检 · 阴性对照" in app.PAGE, "历史表要标明哪些行是自检跑"
        assert "最新一次（真实模型）" in app.PAGE, "卡片要写明取的是真实模型的跑"


def _comparison_result(passed: int = 17, total: int = 22) -> dict:
    """对照跑的结果：人为做对比的实验（如 `--no-embedding` 强制降级），不是成绩。

    本单真跑过一次：字符后端 17/22（recall 83.3%），它排在时间倒序第一，
    于是 77.3% 占了「最新成绩」的位置——读者会以为那是系统当前水平。
    """
    data = _result(total=total, passed=passed)
    data["run_kind"] = "comparison"
    data["retriever"] = "retriever(字符后端（降级）)"
    data["summary"] = [
        {
            "model": "rag-deepseek-chat",
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "skipped": 0,
            "pass_rate": passed / total,
        }
    ]
    data["cases"] = [{"id": "rag-017", "status": "失败", "retrieved": ["kb-004"], "gold": ["kb-013"]}]
    return data


class TestComparisonRunIsNotAScore:
    """对照跑（实验）不是成绩：不许顶掉正式跑的位置。

    与 mock 自检同一类问题——**某些跑是实验，不是成绩**。本单的字符后端跑是人为
    用 `--no-embedding` 强制降级做的后端对比，77.3% 是设计使然（降级后端本来就差），
    拿它当「最新成绩」等于向访问者宣称系统的检索水平倒退了。
    """

    def _env(self, tmp_path: Path, monkeypatch, files: dict[str, dict]) -> Path:
        results = tmp_path / "results"
        (results / "archive").mkdir(parents=True)
        for name, payload in files.items():
            (results / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(app, "RAG_RESULTS_DIR", results)
        monkeypatch.setattr(app, "RAG_ARCHIVE_DIR", results / "archive")
        return results

    def test_latest_skips_comparison_runs(self, tmp_path: Path, monkeypatch) -> None:
        """最新一份是对照跑、次新一份是正式跑 → 取次新那份。"""
        self._env(
            tmp_path,
            monkeypatch,
            {
                "result-20261029-165000.json": _result(total=22, passed=20),
                "result-20261029-173000.json": _comparison_result(),
            },
        )
        runs = app.load_rag_runs()
        assert runs[0]["run_kind"] == "comparison", "历史表按时间倒序，对照跑要看得见"
        latest = app.pick_latest_real_run(runs)
        assert latest is not None, "有正式跑就必须取到，不能因为对照跑更新就被顶掉"
        assert latest["file"] == "result-20261029-165000.json"
        assert latest["run_kind"] == "real"

    def test_all_comparison_has_no_score(self, tmp_path: Path, monkeypatch) -> None:
        """全是对照跑时返回 None：页面显示「暂无正式成绩」，不许拿对照跑填数。"""
        self._env(
            tmp_path,
            monkeypatch,
            {
                "result-20261029-173000.json": _comparison_result(),
                "result-20261029-170000.json": _comparison_result(passed=15),
            },
        )
        runs = app.load_rag_runs()
        assert len(runs) == 2
        assert app.pick_latest_real_run(runs) is None

    def test_cases_follow_the_real_run_not_comparison(self, tmp_path: Path, monkeypatch) -> None:
        """逐题明细也要跟正式跑走，不能被对照跑顶掉。"""
        self._env(
            tmp_path,
            monkeypatch,
            {
                "result-20261029-165000.json": _result(total=22, passed=20),
                "result-20261029-173000.json": _comparison_result(),
            },
        )
        runs = app.load_rag_runs()
        real = next(r for r in runs if r["run_kind"] == "real")
        comparison = next(r for r in runs if r["run_kind"] == "comparison")
        assert len(real["cases"]) == 1, "正式跑那份要带逐题明细"
        assert comparison["cases"] == [], "对照跑不该占用逐题明细"

    # --- 阴性对照：证明上面两条断言真有区分力（把判定逻辑去掉就该红） ---
    def test_negative_control_skipping_only_mock_would_pick_comparison(self, tmp_path: Path, monkeypatch) -> None:
        """只跳 mock 的**旧逻辑**会取到对照跑——所以断言不是白写的。"""
        self._env(
            tmp_path,
            monkeypatch,
            {
                "result-20261029-165000.json": _result(total=22, passed=20),
                "result-20261029-173000.json": _comparison_result(),
            },
        )
        runs = app.load_rag_runs()

        def old_logic(rs):
            for r in rs:
                if not r.get("is_mock"):
                    return r
            return None

        picked = old_logic(runs)
        assert picked is not None and picked["file"] == "result-20261029-173000.json", (
            "旧逻辑应当取到对照跑（这正是本单要修的 bug）"
        )
        assert app.pick_latest_real_run(runs)["file"] != picked["file"], "新逻辑必须改掉这个结果"

    def test_negative_control_all_comparison_would_show_score(self, tmp_path: Path, monkeypatch) -> None:
        """旧逻辑在全是对照跑时会返回一份成绩填上去——所以「返回 None」不是空断言。"""
        self._env(
            tmp_path,
            monkeypatch,
            {
                "result-20261029-173000.json": _comparison_result(),
                "result-20261029-170000.json": _comparison_result(passed=15),
            },
        )
        runs = app.load_rag_runs()
        old_picked = next((r for r in runs if not r.get("is_mock")), None)
        assert old_picked is not None, "旧逻辑会拿对照跑当成绩填上去"
        assert app.pick_latest_real_run(runs) is None, "新逻辑必须返回 None"

    def test_page_says_no_formal_score_and_marks_comparison(self) -> None:
        """④兜底显示：全是对照跑时页面要写明「无正式成绩」，历史表要标出对照跑。"""
        assert "无正式成绩" in app.PAGE, "页面缺「暂无正式成绩」的兜底文案"
        assert "对照 · 不进成绩" in app.PAGE, "历史表要标明哪些行是对照跑"


class TestRunKindFallback:
    """老结果没有 `run_kind` 字段时的兜底判定（历史成绩不用重跑也不会消失）。"""

    def _entry(self, tmp_path: Path, payload: dict, name: str = "result-20261026-120000.json"):
        f = tmp_path / name
        f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return app._rag_entry(f)

    def test_legacy_degraded_run_is_comparison(self, tmp_path: Path) -> None:
        """老的字符后端跑（串里含「字符」/「降级」）→ 判为对照跑。"""
        data = _result()
        del data["run_kind"]
        data["retriever"] = "retriever(字符后端（降级）)"
        entry = self._entry(tmp_path, data)
        assert entry["run_kind"] == "comparison"

    def test_legacy_embedding_run_is_real(self, tmp_path: Path) -> None:
        """老的向量后端跑（非 mock、未降级）→ 判为正式跑，历史成绩不会消失。"""
        data = _result()
        del data["run_kind"]
        data["retriever"] = "retriever(向量后端 BAAI/bge-small-zh-v1.5)"
        entry = self._entry(tmp_path, data)
        assert entry["run_kind"] == "real"

    def test_legacy_mock_run_is_mock(self, tmp_path: Path) -> None:
        """老的自检跑（mock 模型）→ 判为 mock，不被降级兜底抢走。"""
        data = _selfcheck_result()
        del data["run_kind"]
        entry = self._entry(tmp_path, data)
        assert entry["run_kind"] == "mock"
        assert entry["is_mock"] is True

    def test_explicit_run_kind_wins(self, tmp_path: Path) -> None:
        """显式写了 run_kind 就以它为准（向量后端也能被声明成对照跑）。"""
        data = _result()
        data["run_kind"] = "comparison"
        data["retriever"] = "retriever(向量后端 BAAI/bge-small-zh-v1.5)"
        assert self._entry(tmp_path, data)["run_kind"] == "comparison"


class TestRagPortalWiring:

    def test_page_has_rag_route_and_card(self) -> None:
        assert 'href="#rag"' in app.PAGE, "门户缺 RAG 层入口卡"
        assert "if (file === 'rag') return renderRag();" in app.PAGE, "路由没接 #rag"

    def test_portal_is_three_layers(self) -> None:
        assert "LLM &amp; Agent &amp; RAG 评测实验室" in app.PAGE

    def test_answer_cell_is_wrappable(self) -> None:
        """回答是自由文本：全局 td 是 nowrap，不显式允许换行会把表格撑出横向滚动条。

        这是页面上真实出现过的版式问题（长回答把各列挤变形），故钉一条断言。
        """
        assert ".wrapcell" in app.PAGE, "缺 wrapcell 样式（长文本换行）"
        assert "ragAnswerCell" in app.PAGE, "回答列没走专门的单元格渲染"

    def test_scope_no_longer_says_rag_uncovered(self) -> None:
        """RAG 已从「未覆盖」挪进已覆盖，但要写明只覆盖固定管道。"""
        assert "agentic RAG" in app.PAGE
