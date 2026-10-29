"""RAG 层的测试。

分四组：
1. 知识库与检索器（冻结快照、确定性、无依赖可跑）；
2. 检索层 recall@k（纯规则，不调模型）；
3. mini-RAG 管道（上下文有没有真的拼进 prompt、被测对象命名）；
4. 端到端（用假模型跑通现有 runner，验证判分接得上）。

所有用例都不依赖本机跑出来的产物，也不依赖向量模型——CI 沙盒里必须绿。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import load_datasets  # noqa: E402
from src.datasets.schema import EvalCase  # noqa: E402
from src.llm.base import LLMResponse  # noqa: E402
from src.metrics import MetricFactory  # noqa: E402
from src.runner import EvalRunner  # noqa: E402

from rag_eval.knowledge import (  # noqa: E402
    KB_FILE,
    KB_META_FILE,
    check_provenance,
    load_knowledge_base,
    load_provenance,
)
from rag_eval.pipeline import CONTEXT_TEMPLATE, RAGLLM  # noqa: E402
from rag_eval.retrieval import evaluate_retrieval, gold_doc_ids  # noqa: E402
from rag_eval.retriever import Retriever  # noqa: E402

from conftest import FakeLLM  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. 知识库与检索器
# --------------------------------------------------------------------------- #
class TestKnowledgeBase:

    def test_loads_frozen_snapshot(self) -> None:
        docs = load_knowledge_base()
        assert len(docs) >= 10, "知识库至少 10 段才够撑起检索考点"
        assert len({d.doc_id for d in docs}) == len(docs), "文档 id 必须唯一"

    def test_provenance_declared_and_consistent(self) -> None:
        """provenance 是冻结快照的一部分：条数对不上要被发现，而不是静默漂移。"""
        meta = load_provenance()
        assert meta.get("frozen_at"), "provenance 必须写明冻结日期"
        assert meta.get("source"), "provenance 必须写明来源"
        assert check_provenance(load_knowledge_base(), meta) == "", "声明条数与实际不一致"

    def test_provenance_mismatch_is_reported(self) -> None:
        assert "声明" in check_provenance(load_knowledge_base(), {"doc_count": 999})

    def test_fixture_files_are_in_repo(self) -> None:
        assert KB_FILE.exists() and KB_META_FILE.exists(), "知识库是冻结快照，必须入库"


class TestRetriever:

    @pytest.fixture
    def retriever(self) -> Retriever:
        # 强制字符后端：测试不能依赖 sentence-transformers 是否安装
        return Retriever(load_knowledge_base(), prefer_embedding=False)

    def test_backend_is_lexical_without_embedding(self, retriever: Retriever) -> None:
        assert retriever.backend == "lexical"
        assert "字符后端" in retriever.describe()

    def test_deterministic(self, retriever: Retriever) -> None:
        """同输入必得同输出，否则评测不可复现。"""
        first = retriever.search("回归门禁的默认容差是多少？", top_k=2)
        second = retriever.search("回归门禁的默认容差是多少？", top_k=2)
        assert [h.doc_id for h in first] == [h.doc_id for h in second]
        assert [h.score for h in first] == [h.score for h in second]

    def test_top_k_respected(self, retriever: Retriever) -> None:
        assert len(retriever.search("模型层有多少条用例？", top_k=3)) == 3
        assert len(retriever.search("模型层有多少条用例？", top_k=1)) == 1

    def test_scores_descending(self, retriever: Retriever) -> None:
        hits = retriever.search("Agent 层有几道任务？", top_k=3)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    # kb-015 是 2026-10-29 加的「旧口径干扰段」：与 kb-002 语义相近、数值冲突，
    # 字符后端下会把真正正确的出处挤出 top-2（rag-001/rag-016/rag-022 实测 miss，
    # kb-015 在这三题上都排第 1）。这不是 bug，是本次加难度要看的信号——
    # 工单写明 rag-001 保留作基线对照、掉分是预期，**不许为了救它去改资料或判据**。
    #
    # 但豁免必须**显式且穷尽**：下面断言「实际 miss 集合恰好等于这张表」，
    # 于是①表外任何题一旦被推下去，测试立刻红；②表里的题若不再 miss，
    # 说明干扰失效了（比如 kb-015 被误删），同样立刻红。两边都不许悄悄放宽。
    DISTRACTOR_AFFECTED = {
        "rag-001": "基线对照：题干没写「现在」，字符后端分不清新旧口径（工单明示：掉分是预期信号）",
        "rag-016": "kb-015 把它挤出了 top-2（副作用，非设计意图，已如实报告）",
        "rag-022": "本题考点就是「会不会被引到旧口径」，命中低才是被测对象的真实差距",
    }

    def test_retrieves_gold_doc_for_every_answerable_case(self, retriever: Retriever) -> None:
        """端到端命中的前提：标了正确出处的题，检索必须找得回来。

        例外只有 ``DISTRACTOR_AFFECTED`` 那几题，且必须与实际 miss 完全一致。
        """
        cases = load_datasets(ROOT / "rag_eval" / "datasets", ["rag_cases"])
        scored = [c for c in cases if gold_doc_ids(c)]
        assert scored, "用例集里应有标注正确出处的题"
        missed: dict[str, tuple] = {}
        for case in scored:
            retrieved = [h.doc_id for h in retriever.search(case.prompt, top_k=2)]
            if not any(g in retrieved for g in gold_doc_ids(case)):
                missed[case.id] = (gold_doc_ids(case), retrieved)
        assert set(missed) == set(self.DISTRACTOR_AFFECTED), (
            "检索未命中的题与登记的干扰受害清单不一致，"
            f"请核对 kb-015 的影响后再更新 DISTRACTOR_AFFECTED（不许直接放宽）：{missed}"
        )

    def test_distractor_kb015_reaches_top2_on_old_caliber_questions(self, retriever: Retriever) -> None:
        """干扰段得真的能干扰——kb-015 必须挤进「多少条用例」类题目的 top-2。

        这条是 rag-022「旧口径干扰」考点的**成立前提**：若 kb-015 压根进不了 top-2，
        那 rag-022 考的就是运气而不是分辨能力（阿灯 2026-10-28 承认过的设计错误）。
        """
        cases = {c.id: c for c in load_datasets(ROOT / "rag_eval" / "datasets", ["rag_cases"])}
        top1_by_kb015: dict[str, int] = {}
        for cid in ("rag-001", "rag-022"):
            hits = retriever.search(cases[cid].prompt, top_k=2)
            doc_ids = [h.doc_id for h in hits]
            assert "kb-015" in doc_ids, f"{cid} 的 top-2 里没有 kb-015（{doc_ids}），干扰没生效"
            top1_by_kb015[cid] = doc_ids.index("kb-015") + 1
        assert top1_by_kb015, "应有被旧口径干扰的题目"
        # 顺带记录排名，便于 --retrieval-only 报告里核对（这里不断言必须第 1，
        # 排名会随资料变动；「进 top-2」才是考点成立的最低要求）
        print(f"\n[kb-015 排名] {top1_by_kb015}")


# --------------------------------------------------------------------------- #
# 2. 检索层 recall@k（纯规则）
# --------------------------------------------------------------------------- #
def _case(case_id: str, prompt: str, gold: list[str]) -> EvalCase:
    return EvalCase(id=case_id, category="rag_cases", prompt=prompt, meta={"gold_doc_ids": gold})


class TestRetrieval:

    @pytest.fixture
    def retriever(self) -> Retriever:
        return Retriever(load_knowledge_base(), prefer_embedding=False)

    def test_recall_counts_only_scored_cases(self, retriever: Retriever) -> None:
        """「无答案诚实」类没有正确出处，不该被算进 recall 的分母。"""
        cases = [
            _case("c1", "模型层的评测用例一共有多少条？", ["kb-002"]),
            _case("c2", "这个框架支持部署到哪几家云厂商？", []),
        ]
        stats = evaluate_retrieval(cases, retriever, top_k=2)
        assert stats["scored"] == 1
        assert stats["total"] == 2

    def test_miss_is_recorded(self, retriever: Retriever) -> None:
        cases = [_case("c1", "这个项目的 RAG 层用的是什么向量数据库？", ["kb-999"])]
        stats = evaluate_retrieval(cases, retriever, top_k=2)
        assert stats["hit"] == 0
        assert stats["recall_at_k"] == 0.0
        assert stats["missed"] == ["c1"]

    def test_hit_is_recorded(self, retriever: Retriever) -> None:
        # 用 rag-003 的题面而不是 rag-001：rag-001 自 2026-10-29 起被 kb-015（旧口径干扰段）
        # 在字符后端下挤出 top-2，是刻意保留的信号，不适合当「必中」样例。
        cases = [_case("c1", "Agent 层一共有多少道任务？", ["kb-008"])]
        stats = evaluate_retrieval(cases, retriever, top_k=2)
        assert stats["hit"] == 1
        assert stats["recall_at_k"] == 1.0
        assert stats["per_case"][0]["gold_hit"] is True


# --------------------------------------------------------------------------- #
# 3. mini-RAG 管道
# --------------------------------------------------------------------------- #
class TestRAGPipeline:

    def test_context_is_injected_into_prompt(self) -> None:
        """RAG 的定义就是「把检索到的资料拼进 prompt」——没拼进去就不是 RAG。"""
        inner = FakeLLM(name="fake", default="ok")
        rag = RAGLLM(inner, prefer_embedding=False)
        # 同上：换 rag-003（Agent 层任务数），避开被 kb-015 干扰的 rag-001
        question = "Agent 层一共有多少道任务？"
        rag.complete(question)
        sent = inner.calls[-1]
        assert "kb-008" in sent, "检索到的资料没进 prompt"
        assert "10" in sent, "资料正文没进 prompt"
        assert sent != question, "原问题应被包在上下文模板里"

    def test_hits_recorded_per_question(self) -> None:
        """并发下要能按问题查回「它依据了哪几段」，否则结果文件没法核对。"""
        rag = RAGLLM(FakeLLM(name="fake"), prefer_embedding=False)
        question = "Agent 层一共有多少道任务？"
        rag.complete(question)
        assert rag.hits_by_question[question], "没有记录检索结果"
        assert rag.hits_by_question[question][0].doc_id == "kb-008"

    def test_name_marked_as_rag(self) -> None:
        """报告里 rag- 前缀是「这是接了检索管道的被测对象」的唯一标识。"""
        rag = RAGLLM(FakeLLM(name="deepseek-chat"), prefer_embedding=False)
        assert rag.name == "rag-deepseek-chat"
        assert "top_k=2" in rag.describe()

    def test_empty_hits_render_placeholder(self) -> None:
        assert "未检索到" in RAGLLM.build_context([])

    def test_context_template_has_both_slots(self) -> None:
        assert "{context}" in CONTEXT_TEMPLATE and "{question}" in CONTEXT_TEMPLATE


# --------------------------------------------------------------------------- #
# 4. 端到端：mini-RAG 接进现有 runner
# --------------------------------------------------------------------------- #
class _EchoKnowledgeLLM(FakeLLM):
    """假模型：照抄检索到的资料作答（模拟一个「用对资料」的 RAG）。

    库里有的照资料答，库里没有的按 RAG_SYSTEM 的要求说「资料中没有相关信息」。
    """

    def _invoke(self, system, prompt, temperature, max_tokens) -> LLMResponse:  # type: ignore[override]
        self.calls.append(prompt)
        # rag-003 的分支要排在 rag-001 前面：它的上下文里同时含 kb-008（10 道任务）与
        # kb-001（能力全景，可能提到 98 条），按 98 先匹配会误答成模型层数字。
        if "以下是检索到的资料" in prompt and "10 道任务" in prompt:
            text = "Agent 层共 10 道任务。"
        elif "以下是检索到的资料" in prompt and "98" in prompt:
            text = "模型层共 5 个数据集、98 条用例。"
        elif "以下是检索到的资料" in prompt and "埋标记" in prompt:
            text = "采用埋标记加诱导泄露的方法。"
        elif "向量数据库" in prompt or "云厂商" in prompt or "毕业于哪所大学" in prompt:
            text = "资料中没有相关信息。"
        else:
            text = "（默认回答）"
        return LLMResponse(text=text, model=self.model, prompt_tokens=10, completion_tokens=5)


class TestRAGEvalEndToEnd:

    def test_pipeline_scores_against_existing_metrics(self) -> None:
        """判分器一行没改：命中类靠 contains，诚实类靠 not_contains + judge。

        样例取 rag-002 / rag-003（字符后端下检索稳定命中），不用 rag-001——
        它自 2026-10-29 起被 kb-015 抢走检索位，是刻意保留的干扰信号，
        拿它当「照资料答就应通过」的样例等于把信号当 bug 去救。
        """
        from src.config import RunSettings  # noqa: PLC0415

        cases = [c for c in load_datasets(ROOT / "rag_eval" / "datasets", ["rag_cases"]) if c.id in ("rag-002", "rag-003")]
        rag = RAGLLM(_EchoKnowledgeLLM(name="deepseek-chat"), prefer_embedding=False)
        runner = EvalRunner(
            clients={rag.name: rag},  # type: ignore[dict-item]
            factory=MetricFactory(prefer_embedding=False),
            run=RunSettings(),
            verbose=False,
        )
        report = runner.run_cases(cases)
        by_id = {c.case_id: c for c in report.cases}
        assert by_id["rag-002"].passed is True, "照资料答出埋标记应判通过"
        assert by_id["rag-003"].passed is True, "照资料答出 10 道应判通过"

    def test_honest_case_passes_when_not_fabricating(self) -> None:
        """「不知道就说不知道」是考点：没编造就该过，哪怕裁判缺席。"""
        from src.config import RunSettings  # noqa: PLC0415

        cases = [c for c in load_datasets(ROOT / "rag_eval" / "datasets", ["rag_cases"]) if c.id == "rag-011"]
        rag = RAGLLM(_EchoKnowledgeLLM(name="deepseek-chat"), prefer_embedding=False)
        runner = EvalRunner(
            clients={rag.name: rag},  # type: ignore[dict-item]
            factory=MetricFactory(prefer_embedding=False),
            run=RunSettings(),
            verbose=False,
        )
        report = runner.run_cases(cases)
        result = report.cases[0]
        not_contains = next(m for m in result.metrics if m.name == "not_contains")
        assert not_contains.passed is True, "没编造向量库名就不该判失败"

    def test_fabrication_is_caught(self) -> None:
        """反过来的阴性对照：编造库名必须被 not_contains 抓到。"""
        from src.config import RunSettings  # noqa: PLC0415

        cases = [c for c in load_datasets(ROOT / "rag_eval" / "datasets", ["rag_cases"]) if c.id == "rag-011"]
        rag = RAGLLM(FakeLLM(name="deepseek-chat", default="用了 Milvus 做向量检索。"), prefer_embedding=False)
        runner = EvalRunner(
            clients={rag.name: rag},  # type: ignore[dict-item]
            factory=MetricFactory(prefer_embedding=False),
            run=RunSettings(),
            verbose=False,
        )
        report = runner.run_cases(cases)
        result = report.cases[0]
        not_contains = next(m for m in result.metrics if m.name == "not_contains")
        assert not_contains.passed is False, "编造了库名却没被判失败"

    def test_model_name_carries_rag_prefix(self) -> None:
        """端到端跑完，报告里的被测对象名必须带 rag- 前缀。"""
        from src.config import RunSettings  # noqa: PLC0415

        cases = [c for c in load_datasets(ROOT / "rag_eval" / "datasets", ["rag_cases"]) if c.id == "rag-001"]
        rag = RAGLLM(_EchoKnowledgeLLM(name="deepseek-chat"), prefer_embedding=False)
        runner = EvalRunner(
            clients={rag.name: rag},  # type: ignore[dict-item]
            factory=MetricFactory(prefer_embedding=False),
            run=RunSettings(),
            verbose=False,
        )
        report = runner.run_cases(cases)
        assert report.models == ["rag-deepseek-chat"]
        assert report.overall()[0].key == "rag-deepseek-chat"


# --------------------------------------------------------------------------- #
# 5. 数据集本身的护栏（判据是红线，至少要自洽）
# --------------------------------------------------------------------------- #
class TestRagCasesDataset:

    @pytest.fixture
    def cases(self) -> list[EvalCase]:
        return load_datasets(ROOT / "rag_eval" / "datasets", ["rag_cases"])

    def test_loads_with_legal_metrics_and_dimensions(self, cases: list[EvalCase]) -> None:
        # 上限不放死写：加难度是往里添题（rag-014~016 一批、rag-017~022 一批），
        # 每次添题都要改这个数字没意义；下限守住「不许把题删回去」这个真风险。
        assert len(cases) >= 13, "用例数变少了，是不是误删了题？"
        for case in cases:
            assert case.dimension in ("correctness", "relevance"), f"{case.id} 维度未标注或非预期"
            assert case.metrics, f"{case.id} 没指定指标"

    def test_not_contains_cases_must_declare_forbidden(self, cases: list[EvalCase]) -> None:
        """空 forbidden 会让 not_contains 空转成「永远通过」，等于考点被悄悄取消。"""
        for case in cases:
            if "not_contains" in case.metrics:
                assert case.meta.get("forbidden"), f"{case.id} 用了 not_contains 却没给违禁词"

    def test_three_probe_types_present(self, cases: list[EvalCase]) -> None:
        """三类考点各得有：命中、忠实度、无答案诚实。"""
        answerable = [c for c in cases if gold_doc_ids(c)]
        honest = [c for c in cases if not gold_doc_ids(c)]
        judged = [c for c in cases if "judge" in c.metrics]
        assert len(answerable) >= 5, "端到端命中类至少 5 条"
        assert len(honest) >= 3, "无答案诚实类至少 3 条"
        assert len(judged) >= 3, "忠实度/诚实类应交裁判裁"


# --------------------------------------------------------------------------- #
# 5. 归档：历史结果不许丢
# --------------------------------------------------------------------------- #
class TestArchivePrevious:
    """上一份 result 必须进 archive，而不是被覆盖或删掉。

    2026-10-26 的教训：一次跑曾经让 12:10 的真跑结果「消失」——归档是移动不是删除，
    顺序上还得在新跑**开始时**就归位，否则失败/中断的跑会顶掉历史。
    """

    def test_previous_moved_into_archive(self, tmp_path: Path, monkeypatch) -> None:
        from rag_eval import run_eval

        results = tmp_path / "results"
        results.mkdir()
        old = results / "result-20261026-120000.json"
        old.write_text('{"keep": true}', encoding="utf-8")
        monkeypatch.setattr(run_eval, "RESULTS_DIR", results)
        monkeypatch.setattr(run_eval, "ARCHIVE_DIR", results / "archive")

        run_eval.archive_previous(results / "result-20261026-140000.json")

        assert not old.exists(), "上一份不该还留在 results/ 顶层等着被覆盖"
        moved = results / "archive" / "result-20261026-120000.json"
        assert moved.exists(), "上一份必须进 archive（移动，不是删除）"
        assert json.loads(moved.read_text(encoding="utf-8")) == {"keep": True}

    def test_current_file_is_never_archived(self, tmp_path: Path, monkeypatch) -> None:
        """正在写的这一份不能被自己归档掉。"""
        from rag_eval import run_eval

        results = tmp_path / "results"
        results.mkdir()
        current = results / "result-20261026-140000.json"
        current.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(run_eval, "RESULTS_DIR", results)
        monkeypatch.setattr(run_eval, "ARCHIVE_DIR", results / "archive")

        run_eval.archive_previous(current)
        assert current.exists()

    def test_no_previous_is_fine(self, tmp_path: Path, monkeypatch) -> None:
        from rag_eval import run_eval

        results = tmp_path / "results"
        results.mkdir()
        monkeypatch.setattr(run_eval, "RESULTS_DIR", results)
        monkeypatch.setattr(run_eval, "ARCHIVE_DIR", results / "archive")
        run_eval.archive_previous(results / "result-20261026-140000.json")  # 不抛异常即可
