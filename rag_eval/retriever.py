"""检索器：向量后端优先、字符后端兜底。

和 `src/metrics/similarity.py` 同一个套路——**框架不能因为缺一个重依赖就跑不起来**：
装了 sentence-transformers 就用中文向量模型算余弦；没装就降级为字符级打分，
并在 `describe()` 里标明后端，避免把字符级结果当成语义级结论。

两条硬要求：
1. **确定性** —— 同输入必得同输出（排序按 `(-score, doc_id)`），否则评测不可复现；
2. **CI 零成本** —— 词法后端不加载任何模型，纯标准库，CI 沙盒里照跑。
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass

from src.metrics.normalize import normalize_text

from rag_eval.knowledge import Document

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
MAX_EMBED_CHARS = 512

# 词法后端两项打分的权重：覆盖度为主（短查询对长文档也能高分），重合度为辅。
COVERAGE_WEIGHT = 0.6
JACCARD_WEIGHT = 0.4


@dataclass(frozen=True)
class Hit:
    """一条检索结果。"""

    doc: Document
    score: float

    @property
    def doc_id(self) -> str:
        return self.doc.doc_id


def _bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


class Retriever:
    def __init__(
        self,
        docs: list[Document],
        *,
        prefer_embedding: bool = True,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        if not docs:
            raise ValueError("检索器需要至少一个文档")
        self.docs = list(docs)
        self.prefer_embedding = prefer_embedding
        self.model_name = model_name
        self._encoder = None
        self._backend: str | None = None
        self._init_error: str | None = None
        self._lock = threading.Lock()
        # 预计算归一化文本与二元组，词法后端零成本
        self._norm: list[str] = [normalize_text(f"{d.title}{d.text}") for d in self.docs]
        self._grams: list[set[str]] = [_bigrams(t) for t in self._norm]

    # ---------------- 后端初始化（惰性，只做一次） ---------------- #

    @property
    def backend(self) -> str:
        self._ensure_backend()
        return self._backend or "lexical"

    @property
    def init_error(self) -> str | None:
        self._ensure_backend()
        return self._init_error

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return
        with self._lock:
            if self._backend is not None:
                return
            if not self.prefer_embedding:
                self._backend = "lexical"
                return
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.model_name)
                self._backend = "embedding"
            except Exception as exc:  # noqa: BLE001 - 任何失败都降级
                self._backend = "lexical"
                self._init_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[rag] 向量后端不可用（{type(exc).__name__}），检索已降级为字符级。",
                    file=sys.stderr,
                )

    # ---------------- 检索 ---------------- #

    def search(self, query: str, top_k: int = 2) -> list[Hit]:
        """返回得分最高的 top_k 条（不足则全部返回），顺序确定。"""
        self._ensure_backend()
        if self._backend == "embedding":
            scores = self._embedding_scores(query)
        else:
            scores = [self._lexical_score(query, i) for i in range(len(self.docs))]

        ranked = sorted(
            range(len(self.docs)),
            key=lambda i: (-scores[i], self.docs[i].doc_id),
        )
        k = max(1, int(top_k))
        return [Hit(doc=self.docs[i], score=round(scores[i], 6)) for i in ranked[:k]]

    def _embedding_scores(self, query: str) -> list[float]:
        import numpy as np

        texts = [query[:MAX_EMBED_CHARS]] + [f"{d.title}\n{d.text}"[:MAX_EMBED_CHARS] for d in self.docs]
        vectors = self._encoder.encode(texts, normalize_embeddings=True)  # type: ignore[union-cattr]
        query_vec = vectors[0]
        return [float(np.dot(query_vec, vectors[i + 1])) for i in range(len(self.docs))]

    def _lexical_score(self, query: str, index: int) -> float:
        q_norm = normalize_text(query)
        if not q_norm or not self._norm[index]:
            return 0.0
        q_grams = _bigrams(q_norm)
        doc_grams = self._grams[index]
        if not q_grams:
            return 0.0
        coverage = len(q_grams & doc_grams) / len(q_grams)
        union = q_grams | doc_grams
        jaccard = len(q_grams & doc_grams) / len(union) if union else 0.0
        return COVERAGE_WEIGHT * coverage + JACCARD_WEIGHT * jaccard

    def describe(self) -> str:
        """给人看的一句话说明（**不含异常原文**）。

        降级原因的原文（如 `ModuleNotFoundError: No module named 'sentence_transformers'`）
        是排查用的，只在 ``init_error`` 里、由 run_eval 写进报告 JSON 的 notes。
        describe() 会被看板直接渲染到卡片上——把异常名和路径糊到页面上，
        读者只会看到「一堆报错」，看不出「现在是字符后端在跑、结论不是语义级的」。
        """
        if self.backend == "embedding":
            return f"retriever(向量后端 {self.model_name})"
        # 主动关掉向量（--no-embedding）与装不上被迫降级，都是「字符后端」，统一标「降级」
        return "retriever(字符后端（降级）)"
