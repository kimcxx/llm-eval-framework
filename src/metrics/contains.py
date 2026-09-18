"""关键词包含指标。

适用场景：中文知识问答。此时逐字精确匹配过严
（"北京是中国的首都" 与 "北京" 都不该算错），
但又不值得动用语义模型。关键词命中率是性价比最高的折中。
"""

from __future__ import annotations

from src.datasets.schema import EvalCase
from src.llm.base import LLMResponse
from src.metrics.base import BaseMetric, MetricResult
from src.metrics.normalize import normalize_text, shorten


class ContainsMetric(BaseMetric):
    name = "contains"

    def compute(self, case: EvalCase, response: LLMResponse) -> MetricResult:
        keywords = [k for k in case.keywords if str(k).strip()]
        if not keywords:
            return MetricResult(self.name, 0.0, None, "用例未提供 keywords，跳过")

        haystack = normalize_text(response.text)
        hits = [k for k in keywords if normalize_text(str(k)) in haystack]
        missed = [k for k in keywords if k not in hits]

        score = len(hits) / len(keywords)
        detail = f"命中 {len(hits)}/{len(keywords)}"
        if missed:
            detail += f"；缺失 {[shorten(str(m), 20) for m in missed]}"

        return MetricResult(self.name, score, not missed, detail)
