"""检索层评测：recall@k，纯规则、零成本。

这一段**不调模型**，只问一个可判定的问题：用例标注的正确出处（`gold_doc_ids`）
有没有出现在检索结果的 top-k 里。因此它可以：
- 在 CI 里每次都跑（字符后端，无依赖、无额度消耗）；
- 与端到端成绩分开看——答错时，先分清是「没检索到」还是「检索到了但生成时没用好」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.datasets.schema import EvalCase

from rag_eval.retriever import Hit, Retriever


@dataclass(frozen=True)
class CaseRetrieval:
    case_id: str
    retrieved: list[str]
    gold: list[str]
    hit: bool
    scored: bool  # False = 该用例没有标注正确出处（如「无答案诚实」类），不计入 recall
    top_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "retrieved": self.retrieved,
            "gold": self.gold,
            "gold_hit": self.hit,
            "scored": self.scored,
            "top_score": round(self.top_score, 4),
        }


def gold_doc_ids(case: EvalCase) -> list[str]:
    raw = case.meta.get("gold_doc_ids")
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


def evaluate_retrieval(
    cases: list[EvalCase],
    retriever: Retriever,
    top_k: int = 2,
) -> dict[str, Any]:
    """对每条用例做一次检索，统计 recall@k。"""
    per_case: list[CaseRetrieval] = []
    for case in cases:
        hits: list[Hit] = retriever.search(case.prompt, top_k)
        retrieved = [h.doc_id for h in hits]
        gold = gold_doc_ids(case)
        hit = bool(gold) and any(g in retrieved for g in gold)
        per_case.append(
            CaseRetrieval(
                case_id=case.id,
                retrieved=retrieved,
                gold=gold,
                hit=hit,
                scored=bool(gold),
                top_score=hits[0].score if hits else 0.0,
            )
        )

    scored = [c for c in per_case if c.scored]
    hit_count = sum(1 for c in scored if c.hit)
    answered = sum(1 for c in per_case if c.retrieved)
    return {
        "k": top_k,
        "backend": retriever.backend,
        "total": len(per_case),
        "scored": len(scored),
        "hit": hit_count,
        "recall_at_k": (hit_count / len(scored)) if scored else 0.0,
        "retrieved_any": answered,
        "missed": [c.case_id for c in scored if not c.hit],
        "per_case": [c.to_dict() for c in per_case],
    }


def format_retrieval(stats: dict[str, Any]) -> str:
    """给控制台看的一张小表。"""
    lines = [
        f"检索层 recall@{stats['k']}（{stats['backend']}）",
        "-" * 46,
        f"{'用例':<12}{'命中':<8}{'正确出处':<14}{'实际检索到'}",
    ]
    for row in stats["per_case"]:
        gold = ",".join(row["gold"]) or "—（无答案类）"
        lines.append(
            f"{row['id']:<12}{('是' if row['gold_hit'] else '否') if row['scored'] else '—':<8}"
            f"{gold:<14}{','.join(row['retrieved']) or '（空）'}"
        )
    lines.append("-" * 46)
    lines.append(
        f"可判 {stats['scored']} 条，命中 {stats['hit']} 条，"
        f"recall@{stats['k']} = {stats['recall_at_k']:.1%}"
    )
    return "\n".join(lines)
