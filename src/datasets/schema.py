"""评测用例的数据结构。

一条评测用例 = 输入 + 期望 + 该用例要跑哪些指标。
把「跑哪些指标」下放到用例级别，是因为不同任务的判定方式天然不同：
数学题看精确匹配，抽取任务看 JSON 合规，开放问答才需要语义相似度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_METRICS = ("exact_match",)

# 能力维度：给用例打标签，只影响报告的分组统计，不参与任何通过/失败判定。
DIMENSIONS: tuple[str, ...] = (
    "correctness",
    "instruction_following",
    "format",
    "safety",
    "robustness",
    "knowledge",
)

DIMENSION_LABELS: dict[str, str] = {
    "correctness": "正确性",
    "instruction_following": "指令遵循",
    "format": "格式合规",
    "safety": "安全",
    "robustness": "鲁棒性",
    "knowledge": "知识时效",
}

# 未打 dimension 标签的用例在报告里归入该组，单独一行展示。
UNTAGGED_DIMENSION = "untagged"
UNTAGGED_DIMENSION_LABEL = "未标注"


class DatasetError(ValueError):
    """数据集格式非法。"""


@dataclass
class EvalCase:
    id: str
    category: str
    prompt: str
    system: str | None = None
    expected: Any = None
    keywords: list[str] = field(default_factory=list)
    required_keys: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    dimension: str | None = None
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = "") -> EvalCase:
        missing = [key for key in ("id", "prompt") if not raw.get(key)]
        if missing:
            raise DatasetError(f"用例缺少必填字段 {missing}: {raw}")

        known = {
            "id",
            "category",
            "prompt",
            "system",
            "expected",
            "keywords",
            "required_keys",
            "metrics",
            "dimension",
        }
        payload = {k: v for k, v in raw.items() if k in known}

        payload.setdefault("category", source or "default")
        # 只给了 expected 字符串、没显式写 keywords 时，自动用 expected 作为关键词
        if not payload.get("keywords") and isinstance(payload.get("expected"), str):
            payload["keywords"] = [payload["expected"]]

        payload["dimension"] = _normalize_dimension(raw.get("dimension"), raw.get("id", "?"))

        return cls(
            **payload,
            source=source,
            meta={k: v for k, v in raw.items() if k not in known},
        )


def _normalize_dimension(value: Any, case_id: str) -> str | None:
    """校验并归一化 dimension：空值视为未打标签（None）。

    大小写与首尾空白不敏感，便于人手写标签；非法值直接报错，
    避免把拼错的维度静默丢进 untagged 而没人发现。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetError(f"用例 {case_id} 的 dimension 应为字符串，实际是 {type(value).__name__}")

    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in DIMENSIONS:
        raise DatasetError(
            f"用例 {case_id} 的 dimension 非法: {value!r}；可选值: {', '.join(DIMENSIONS)}"
        )
    return normalized
