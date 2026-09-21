"""评测结果的数据结构与聚合统计。

聚合逻辑独立于执行逻辑，好处是：同一份原始结果可以被反复重新统计
（换阈值、换分组维度），不需要重新花钱调模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from src.datasets.schema import (
    DIMENSION_LABELS,
    DIMENSIONS,
    UNTAGGED_DIMENSION,
    UNTAGGED_DIMENSION_LABEL,
)
from src.metrics.base import MetricResult


@dataclass
class CaseResult:
    """单条用例在单个模型上的一次完整评测记录。"""

    case_id: str
    category: str
    dataset: str
    model: str
    prompt: str
    dimension: str | None = None
    response_text: str = ""
    expected: Any = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    metrics: list[MetricResult] = field(default_factory=list)
    skipped_metrics: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def judged(self) -> list[MetricResult]:
        """参与通过判定的指标（passed 不为 None）。"""
        return [m for m in self.metrics if m.passed is not None]

    @property
    def passed(self) -> bool | None:
        """None 表示该用例无可判定指标，不计入通过率分母。"""
        if self.error:
            return False
        judged = self.judged
        if not judged:
            return None
        return all(m.passed for m in judged)

    def score_of(self, name: str) -> float | None:
        for metric in self.metrics:
            if metric.name == name:
                return metric.score
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "dataset": self.dataset,
            "model": self.model,
            "dimension": self.dimension,
            "prompt": self.prompt,
            "expected": self.expected,
            "response": self.response_text,
            "passed": self.passed,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": round(self.cost, 6),
            "metrics": [m.to_dict() for m in self.metrics],
            "skipped_metrics": self.skipped_metrics,
        }


@dataclass
class GroupStats:
    """一组结果的统计口径（可以按模型分组，也可以按模型×类别分组）。"""

    key: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    latency_ms: list[float] = field(default_factory=list)
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metric_scores: dict[str, list[float]] = field(default_factory=dict)

    def add(self, case: CaseResult) -> None:
        self.total += 1

        verdict = case.passed
        if verdict is None:
            self.skipped += 1
        elif verdict:
            self.passed += 1
        else:
            self.failed += 1

        if case.error:
            self.errors += 1

        self.latency_ms.append(case.latency_ms)
        self.cost += case.cost
        self.prompt_tokens += case.prompt_tokens
        self.completion_tokens += case.completion_tokens

        for metric in case.metrics:
            self.metric_scores.setdefault(metric.name, []).append(metric.score)

    @property
    def judged(self) -> int:
        """实际参与判定的用例数（分母不应包含被跳过的）。"""
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.judged if self.judged else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return percentile(self.latency_ms, 0.95)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def avg_metric(self, name: str) -> float | None:
        scores = self.metric_scores.get(name)
        if not scores:
            return None
        return sum(scores) / len(scores)


def dimension_label(name: str) -> str:
    """维度的展示名；untagged 显示为「未标注」，未知值原样返回。"""
    if name == UNTAGGED_DIMENSION:
        return UNTAGGED_DIMENSION_LABEL
    return DIMENSION_LABELS.get(name, name)


def percentile(values: Iterable[float], ratio: float) -> float:
    """线性插值分位数，避免为了一个 P95 引入 numpy 依赖。"""
    data = sorted(values)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]

    position = (len(data) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(data) - 1)
    weight = position - lower
    return data[lower] * (1 - weight) + data[upper] * weight


def group_by(
    cases: Iterable[CaseResult],
    key: str | Callable[[CaseResult], str],
) -> dict[str, GroupStats]:
    """按指定维度分组统计，保持首次出现的顺序。"""
    key_fn: Callable[[CaseResult], str] = key if callable(key) else _field_getter(key)
    groups: dict[str, GroupStats] = {}

    for case in cases:
        name = key_fn(case)
        stats = groups.get(name)
        if stats is None:
            stats = GroupStats(key=name)
            groups[name] = stats
        stats.add(case)

    return groups


def _field_getter(field_name: str) -> Callable[[CaseResult], str]:
    def getter(case: CaseResult) -> str:
        return str(getattr(case, field_name, "unknown"))

    return getter


@dataclass
class EvalReport:
    """一次完整评测运行的全部产物。"""

    started_at: str
    finished_at: str
    duration_s: float
    models: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    config_path: str = ""

    # ---------------- 聚合视图 ---------------- #

    @property
    def case_count(self) -> int:
        return len(self.cases)

    def overall(self) -> list[GroupStats]:
        groups = group_by(self.cases, "model")
        return [groups[m] for m in self.models if m in groups]

    def by_category(self, model: str) -> list[GroupStats]:
        subset = [c for c in self.cases if c.model == model]
        groups = group_by(subset, "category")
        return sorted(groups.values(), key=lambda g: g.key)

    def by_dimension(self, model: str) -> list[GroupStats]:
        """按能力维度统计；未打 dimension 标签的用例归入 untagged 组。

        排序固定为枚举顺序，untagged 永远排在最后，便于跨报告横向对比。
        """
        subset = [c for c in self.cases if c.model == model]
        groups = group_by(subset, lambda c: c.dimension or UNTAGGED_DIMENSION)
        order = {name: index for index, name in enumerate(DIMENSIONS)}
        return sorted(
            groups.values(),
            key=lambda g: (order.get(g.key, len(DIMENSIONS)), g.key),
        )

    def metric_names(self) -> list[str]:
        names: list[str] = []
        for case in self.cases:
            for metric in case.metrics:
                if metric.name not in names:
                    names.append(metric.name)
        return sorted(names)

    def skipped_metric_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            for name in case.skipped_metrics:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def failures(self, model: str, limit: int = 20) -> list[CaseResult]:
        bad = [c for c in self.cases if c.model == model and c.passed is False]
        return bad[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 2),
            "models": self.models,
            "datasets": self.datasets,
            "case_count": self.case_count,
            "notes": self.notes,
            "summary": [
                {
                    "model": s.key,
                    "total": s.total,
                    "passed": s.passed,
                    "failed": s.failed,
                    "skipped": s.skipped,
                    "errors": s.errors,
                    "pass_rate": round(s.pass_rate, 4),
                    "avg_latency_ms": round(s.avg_latency_ms, 2),
                    "p95_latency_ms": round(s.p95_latency_ms, 2),
                    "cost": round(s.cost, 6),
                    "total_tokens": s.total_tokens,
                    "metric_avg": {
                        name: round(s.avg_metric(name) or 0.0, 4) for name in s.metric_scores
                    },
                }
                for s in self.overall()
            ],
            "categories": {
                model: [
                    {
                        "category": stats.key,
                        "total": stats.total,
                        "passed": stats.passed,
                        "failed": stats.failed,
                        "pass_rate": round(stats.pass_rate, 4),
                    }
                    for stats in self.by_category(model)
                ]
                for model in self.models
            },
            # 按能力维度汇总；未打 dimension 标签的用例归入 untagged
            "dimensions": {
                model: [
                    {
                        "dimension": stats.key,
                        "label": dimension_label(stats.key),
                        "total": stats.total,
                        "passed": stats.passed,
                        "failed": stats.failed,
                        "skipped": stats.skipped,
                        "pass_rate": round(stats.pass_rate, 4),
                    }
                    for stats in self.by_dimension(model)
                ]
                for model in self.models
            },
            "cases": [c.to_dict() for c in self.cases],
        }
