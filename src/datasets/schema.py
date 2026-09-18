"""评测用例的数据结构。

一条评测用例 = 输入 + 期望 + 该用例要跑哪些指标。
把「跑哪些指标」下放到用例级别，是因为不同任务的判定方式天然不同：
数学题看精确匹配，抽取任务看 JSON 合规，开放问答才需要语义相似度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_METRICS = ("exact_match",)


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
        }
        payload = {k: v for k, v in raw.items() if k in known}

        payload.setdefault("category", source or "default")
        # 只给了 expected 字符串、没显式写 keywords 时，自动用 expected 作为关键词
        if not payload.get("keywords") and isinstance(payload.get("expected"), str):
            payload["keywords"] = [payload["expected"]]

        return cls(
            **payload,
            source=source,
            meta={k: v for k, v in raw.items() if k not in known},
        )
