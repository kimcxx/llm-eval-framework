"""语义相似度指标（带自动降级）。

工程上最重要的一点：**评测框架自身不能因为缺一个重依赖就跑不起来**。
因此这里采用「向量后端优先、字符后端兜底」的双后端设计：
- 装了 sentence-transformers → 用中文向量模型算余弦相似度，判断同义改写；
- 没装 → 降级为字符级相似度，流程照跑，并在 detail 中明确标注已降级，
  避免使用者误以为得到了语义级结论。
"""

from __future__ import annotations

import difflib
import sys
import threading

from src.datasets.schema import EvalCase
from src.llm.base import LLMResponse
from src.metrics.base import BaseMetric, MetricResult
from src.metrics.normalize import bigram_jaccard, normalize_text, shorten

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
MAX_EMBED_CHARS = 512


class SimilarityMetric(BaseMetric):
    name = "similarity"

    def __init__(
        self,
        threshold: float = 0.75,
        model_name: str = DEFAULT_MODEL,
        prefer_embedding: bool = True,
    ) -> None:
        self.threshold = threshold
        self.model_name = model_name
        self.prefer_embedding = prefer_embedding
        self._encoder = None
        self._backend: str | None = None
        self._init_error: str | None = None
        # 实例级锁：保护 _ensure_backend 的惰性初始化，避免并发时重复加载模型
        self._init_lock = threading.Lock()

    # ---------------- 后端初始化（惰性，只做一次） ---------------- #

    @property
    def backend(self) -> str:
        self._ensure_backend()
        return self._backend or "lexical"

    def _ensure_backend(self) -> None:
        # 无锁快速检查：绝大多数调用发生在初始化完成之后
        if self._backend is not None:
            return

        # 加锁后二次检查，保证初始化只执行一次（双检锁）
        with self._init_lock:
            if self._backend is not None:
                return

            if not self.prefer_embedding:
                self._backend = "lexical"
                return

            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.model_name)
                self._backend = "embedding"
            except Exception as exc:  # noqa: BLE001 - 任何失败都降级，不能让评测挂掉
                self._backend = "lexical"
                self._init_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[similarity] 向量后端不可用（{type(exc).__name__}），已降级为字符级相似度。"
                    f"如需语义级评测请执行: pip install sentence-transformers",
                    file=sys.stderr,
                )

    # ---------------- 指标计算 ---------------- #

    def compute(self, case: EvalCase, response: LLMResponse) -> MetricResult:
        reference = self._reference(case)
        if not reference:
            return MetricResult(self.name, 0.0, None, "用例未提供参考文本，跳过")

        self._ensure_backend()
        if self._backend == "embedding":
            score = self._embedding_score(reference, response.text)
        else:
            score = self._lexical_score(reference, response.text)

        passed = score >= self.threshold
        detail = f"{self._backend} 相似度={score:.3f}（阈值 {self.threshold}）"
        if self._backend == "lexical" and self._init_error:
            detail += "（已降级，非语义结论）"

        return MetricResult(self.name, score, passed, detail)

    @staticmethod
    def _reference(case: EvalCase) -> str:
        if isinstance(case.expected, str) and case.expected.strip():
            return case.expected.strip()
        fallback = case.meta.get("reference")
        return str(fallback).strip() if fallback else ""

    def _embedding_score(self, reference: str, answer: str) -> float:
        import numpy as np

        vectors = self._encoder.encode(  # type: ignore[union-attr]
            [reference[:MAX_EMBED_CHARS], answer[:MAX_EMBED_CHARS]],
            normalize_embeddings=True,
        )
        return float(np.dot(vectors[0], vectors[1]))

    @staticmethod
    def _lexical_score(reference: str, answer: str) -> float:
        ref_norm = normalize_text(reference)
        ans_norm = normalize_text(answer)
        if not ref_norm or not ans_norm:
            return 0.0
        sequence_ratio = difflib.SequenceMatcher(None, ref_norm, ans_norm).ratio()
        return max(sequence_ratio, bigram_jaccard(ref_norm, ans_norm))

    def describe(self) -> str:
        if self.backend == "embedding":
            return f"similarity(向量后端 {self.model_name})"
        reason = f"，原因：{shorten(self._init_error or '未启用', 40)}" if self._init_error else ""
        return f"similarity(字符后端{reason})"
