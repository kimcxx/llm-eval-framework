"""检索层评测：命中率@k 与 召回率@k，纯规则、零成本。

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
    scored: bool  # False = 该用例没有标注正确出处（如「无答案诚实」类），不计入指标
    top_score: float
    # 逐题召回率用的两个数：top-k 里捞回了几段金标 / 金标一共几段。
    # 只有「一题对多段」的题（如 rag-017）才看得出与命中率的区别。
    gold_hit_count: int = 0

    @property
    def gold_total(self) -> int:
        return len(self.gold)

    @property
    def gold_recall(self) -> float:
        """逐题召回率：捞回的金标段数 ÷ 金标总段数。未标注出处的题返回 0（且不进分母）。"""
        if not self.gold:
            return 0.0
        return self.gold_hit_count / len(self.gold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "retrieved": self.retrieved,
            "gold": self.gold,
            "gold_hit": self.hit,
            "gold_hit_count": self.gold_hit_count,
            "gold_total": self.gold_total,
            "gold_recall": round(self.gold_recall, 4),
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
    """对每条用例做一次检索，统计两个**不是一回事**的指标。

    * ``recall_at_k``：**命中率** Hit Rate@k——「金标里只要有任意一段进了 top-k 就算命中」
      （`any`）。字段名与数值为兼容历史一律不动，**只有显示名改成「命中率@2」**。
    * ``recall_ratio_at_k``：**召回率** Recall@k——逐题 `|top-k ∩ gold| ÷ |gold|`，
      整体取所有可判题的**算术平均**（不是先合并再算，避免长题压过短题）。

    两者在「一题对一段」时数值相同，所以长期没暴露；rag-017（金标 kb-013 + kb-002）
    是这批里第一道「一题对两段」的题——命中率 100%、召回率 50%，**归因完全不同**：
    只看命中率会把失败推给「模型笨」，看召回率才看得出链条本来就断了一半。
    """
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
                gold_hit_count=sum(1 for g in gold if g in retrieved),
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
        # 兼容保留：字段名叫 recall_at_k、值不变，但它其实是**命中率**（见上）
        "recall_at_k": (hit_count / len(scored)) if scored else 0.0,
        # 真正的召回率（老结果没有这个字段 → 页面显示「需重跑」，不许猜数）
        "recall_ratio_at_k": (sum(c.gold_recall for c in scored) / len(scored)) if scored else 0.0,
        "retrieved_any": answered,
        "missed": [c.case_id for c in scored if not c.hit],
        "per_case": [c.to_dict() for c in per_case],
    }


def format_retrieval(stats: dict[str, Any]) -> str:
    """给控制台看的一张小表。"""
    # 名字要写准：这是**命中率**（只要有一段金标进 top-k 就算），不是召回率。
    # 两个指标并列，免得读者以为 Recall 一直是满分而漏掉「一题对多段」题断的那一半。
    ratio = stats.get("recall_ratio_at_k")
    lines = [
        f"检索层 命中率@{stats['k']} / 召回率@{stats['k']}（{stats['backend']}）",
        "-" * 62,
        f"{'用例':<12}{'命中':<8}{'捞回':<8}{'正确出处':<14}{'实际检索到'}",
    ]
    for row in stats["per_case"]:
        gold = ",".join(row["gold"]) or "—（无答案类）"
        got = f"{row.get('gold_hit_count', 0)}/{row.get('gold_total', 0)}" if row["scored"] else "—"
        lines.append(
            f"{row['id']:<12}{('是' if row['gold_hit'] else '否') if row['scored'] else '—':<8}"
            f"{got:<8}{gold:<14}{','.join(row['retrieved']) or '（空）'}"
        )
    lines.append("-" * 62)
    lines.append(
        f"可判 {stats['scored']} 条，命中 {stats['hit']} 条，"
        f"命中率@{stats['k']} = {stats['recall_at_k']:.1%}"
        + ("" if ratio is None else f"，召回率@{stats['k']} = {ratio:.1%}")
    )
    return "\n".join(lines)
