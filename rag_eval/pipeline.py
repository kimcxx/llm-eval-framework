"""mini-RAG 流水线：检索 → 拼上下文 → 生成。

关键设计：**被测对象包装成一个 `BaseLLM`**。
这样现有执行引擎（`src/runner/`）、判分器、报告与看板一行都不用改——
runner 眼里它就是一个「模型」，只是这个名字叫 `rag-*`，而它的 `complete()`
内部先检索知识库、把上下文拼进 prompt，再交给真实模型生成。

被测对象因此是**固定管道**：每个问题无条件检索 top-k 再生成，不做「要不要检索」
的决策，也不产生多步轨迹——这正是它不属于 Agent 层的原因。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM, LLMResponse

from rag_eval.knowledge import KB_FILE, load_knowledge_base
from rag_eval.retriever import Hit, Retriever

DEFAULT_TOP_K = 2

RAG_SYSTEM = (
    "你是检索增强问答助手。只能依据给定的资料回答问题，不得使用资料之外的知识，"
    "也不得推测或编造。资料里没有答案时，明确回答“资料中没有相关信息”。"
    "回答保持简洁，直接给结论。"
)

CONTEXT_TEMPLATE = (
    "以下是检索到的资料：\n"
    "----------------\n"
    "{context}\n"
    "----------------\n\n"
    "问题：{question}\n\n"
    "请严格依据上述资料回答。资料中没有的，回答“资料中没有相关信息”，不要编造。"
)


class RAGLLM(BaseLLM):
    """把 mini-RAG 包装成模型客户端。

    `last_hits` 保存最近一次检索结果，供单题调试用；`hits_by_question` 按问题
    存下每次检索结果，供评测入口写进结果文件（可核对「它到底依据了哪几段」——
    并发下 `last_hits` 不可靠）。两者都不改变 `complete()` 的对外契约。
    """

    def __init__(
        self,
        inner: BaseLLM,
        *,
        retriever: Retriever | None = None,
        top_k: int = DEFAULT_TOP_K,
        knowledge_path: Path | str | None = None,
        prefer_embedding: bool = True,
    ) -> None:
        super().__init__(name=f"rag-{inner.name}", model=inner.model)
        self.inner = inner
        self.top_k = max(1, int(top_k))
        if retriever is not None:
            self.retriever = retriever
        else:
            docs = load_knowledge_base(knowledge_path or KB_FILE)
            self.retriever = Retriever(docs, prefer_embedding=prefer_embedding)
        self.last_hits: list[Hit] = []
        self.hits_by_question: dict[str, list[Hit]] = {}

    # ---------------- BaseLLM 接口 ---------------- #

    def _invoke(
        self,
        system: str | None,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        hits = self.retriever.search(prompt, self.top_k)
        self.last_hits = hits
        self.hits_by_question[prompt] = hits
        context = self.build_context(hits)
        generation_prompt = CONTEXT_TEMPLATE.format(context=context, question=prompt)
        # 系统提示词由流水线指定：RAG 的「只依据资料」约束是被测对象的一部分，
        # 用例自带的 system 会覆盖它就不成其为 RAG 了。
        return self.inner.complete(
            generation_prompt,
            system=RAG_SYSTEM,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ---------------- 辅助 ---------------- #

    @staticmethod
    def build_context(hits: list[Hit]) -> str:
        if not hits:
            return "（本次未检索到任何资料）"
        return "\n\n".join(hit.doc.render() for hit in hits)

    def describe(self) -> str:
        return f"rag(top_k={self.top_k}, {self.retriever.describe()}, llm={self.inner.model})"

    def __repr__(self) -> str:
        return f"<RAGLLM name={self.name!r} top_k={self.top_k} backend={self.retriever.backend!r}>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "top_k": self.top_k,
            "retriever": self.retriever.describe(),
        }
