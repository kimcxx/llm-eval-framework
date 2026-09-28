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
# 四类对齐行业通用叫法，元组顺序即报告/看板里的展示顺序：
#   correctness 准确性 —— 有客观标准答案的断言（数学、事实问答）
#   instruction_following 指令遵循 —— 是否按要求的格式/结构输出（JSON 抽取等）
#   safety 安全 —— 红队越狱与拒答
#   relevance 相关性 —— 开放题，由 LLM 裁判打分；与客观断言不是同一种测法，
#                       单独成维度，避免把「裁判的主观分」混进「答对答错」
DIMENSIONS: tuple[str, ...] = (
    "correctness",
    "instruction_following",
    "safety",
    "relevance",
)

# 历史维度名：老数据集与老报告里出现过，保留以免加载老数据直接报错。
LEGACY_DIMENSIONS: tuple[str, ...] = ("format", "robustness", "knowledge")
# 其中 format 是 instruction_following 的旧称，统一口径时自动归一；
# robustness / knowledge 没有对应新维度，原样保留，报告里仍然可见（不会静默消失）。
LEGACY_DIMENSION_ALIASES: dict[str, str] = {
    "format": "instruction_following",
}

DIMENSION_LABELS: dict[str, str] = {
    "correctness": "准确性",
    "instruction_following": "指令遵循",
    "safety": "安全",
    "relevance": "相关性",
    # 旧维度中文名：仅供历史报告回退显示，新报告不会再出现
    "format": "格式合规",
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
    # 历史维度名放行（format 等），否则老数据集一加载就炸；归一时顺手换成新名
    if normalized not in DIMENSIONS and normalized not in LEGACY_DIMENSIONS:
        raise DatasetError(
            f"用例 {case_id} 的 dimension 非法: {value!r}；可选值: {', '.join(DIMENSIONS)}"
        )
    return LEGACY_DIMENSION_ALIASES.get(normalized, normalized)
