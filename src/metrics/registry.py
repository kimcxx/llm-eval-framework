"""指标工厂：按名称装配指标，并明确区分「本地指标」与「需要裁判的指标」。"""

from __future__ import annotations

from typing import Iterable, Sequence

from src.llm.base import BaseLLM
from src.metrics.base import BaseMetric, MetricResult
from src.metrics.contains import ContainsMetric
from src.metrics.exact_match import ExactMatchMetric
from src.metrics.is_json import IsJsonMetric
from src.metrics.json_valid import JsonValidMetric
from src.metrics.judge import JudgeMetric
from src.metrics.not_contains import NotContainsMetric
from src.metrics.schema_match import SchemaMatchMetric
from src.metrics.similarity import SimilarityMetric

# json_valid 已拆分为 is_json（格式）+ schema_match（字段），
# 保留名字只为兼容旧数据集与旧报告，新用例请直接用这两个。
DEPRECATED_METRICS = ("json_valid",)
LOCAL_METRICS = (
    "exact_match",
    "contains",
    "not_contains",
    "is_json",
    "schema_match",
    "similarity",
) + DEPRECATED_METRICS
JUDGE_METRICS = ("judge",)
SUPPORTED_METRICS = LOCAL_METRICS + JUDGE_METRICS


class UnknownMetricError(ValueError):
    """数据集里写了框架不认识的指标名。"""


class MetricFactory:
    """负责按需创建并缓存指标实例（向量模型只加载一次）。"""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.75,
        judge_threshold: float = 4.0,
        schema_match_threshold: float = 1.0,
        judge_client: BaseLLM | None = None,
        prefer_embedding: bool = True,
        judge_only_categories: Sequence[str] = (),
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.judge_threshold = judge_threshold
        self.schema_match_threshold = schema_match_threshold
        self.judge_client = judge_client
        self.prefer_embedding = prefer_embedding
        # 只由裁判判定的分类：其余指标降级为「仅记录」，不算模型失败
        self.judge_only_categories = frozenset(
            str(c).strip().lower() for c in judge_only_categories if str(c).strip()
        )
        self._cache: dict[str, BaseMetric] = {}

    def get(self, name: str) -> BaseMetric | None:
        """返回指标实例；不可用时返回 None（例如没有裁判模型）。"""
        if name in self._cache:
            return self._cache[name]

        metric: BaseMetric | None
        if name == "exact_match":
            metric = ExactMatchMetric()
        elif name == "contains":
            metric = ContainsMetric()
        elif name == "json_valid":
            # 兼容旧数据集：仍是「格式 + 字段」的旧口径，不会自动拆成两个指标
            metric = JsonValidMetric()
        elif name == "is_json":
            metric = IsJsonMetric()
        elif name == "schema_match":
            metric = SchemaMatchMetric(threshold=self.schema_match_threshold)
        elif name == "not_contains":
            metric = NotContainsMetric()
        elif name == "similarity":
            metric = SimilarityMetric(
                threshold=self.similarity_threshold,
                prefer_embedding=self.prefer_embedding,
            )
        elif name == "judge":
            if self.judge_client is None:
                return None
            metric = JudgeMetric(self.judge_client, threshold=self.judge_threshold)
        else:
            raise UnknownMetricError(
                f"未知指标 {name!r}；可用指标：{', '.join(SUPPORTED_METRICS)}"
            )

        self._cache[name] = metric
        return metric

    def resolve(self, names: list[str]) -> tuple[list[BaseMetric], list[str]]:
        """把用例声明的指标名解析为实例。

        返回 (可用指标, 被跳过的指标名)，让执行引擎可以在报告中如实披露
        「哪些指标其实没跑」，而不是静默通过。
        """
        resolved: list[BaseMetric] = []
        skipped: list[str] = []

        for name in names:
            metric = self.get(name)
            if metric is None:
                skipped.append(name)
            else:
                resolved.append(metric)

        return resolved, skipped

    # ---------------- 判定口径：某些分类只由裁判说了算 ---------------- #

    def is_judge_only(self, category: str | None) -> bool:
        """该分类是否只由 judge 判定（其余指标仅记录）。"""
        return str(category or "").strip().lower() in self.judge_only_categories

    def as_record_only(self, result: MetricResult, category: str | None) -> MetricResult:
        """把 judge-only 分类里的非裁判指标降级为「仅记录」（passed=None）。

        为什么不是直接丢弃结果：相似度仍然是有信息量的信号，报告里要能看见，
        但它不该把「换个说法但答对了」判成失败，所以只改判定口径、不改分数。
        """
        if result.passed is None or result.name == "judge":
            return result
        if not self.is_judge_only(category):
            return result

        detail = f"{result.detail}（仅记录，不参与通过判定：该分类由 judge 判定）".strip()
        return MetricResult(result.name, result.score, None, detail)

    def notes(self, active_categories: Iterable[str] | None = None) -> list[str]:
        """披露本次评测实际生效的指标实现，避免「静默降级」。

        active_categories 是本轮真正跑到的分类。给了就只披露与之相关的口径：
        judge_only_categories 是全局配置，只跑 json_extract 的报告不该出现
        「分类 qa_open 仅由 judge 判定」——那会让人以为这轮跑过 qa_open。
        不传（None）表示不过滤，保持「配置全貌」口径。
        """
        notes: list[str] = []

        # 本轮真正用到（已实例化）的指标才披露定义与阈值，避免噪声
        for name in ("similarity", "is_json", "schema_match"):
            metric = self._cache.get(name)
            if metric is None:
                continue
            describe = getattr(metric, "describe", None)
            notes.append(describe() if callable(describe) else name)

        if self.judge_client is not None:
            notes.append(f"裁判模型 {self.judge_client.name}（通过阈值 {self.judge_threshold:g}/5）")
        else:
            notes.append("未配置可用裁判模型，judge 指标已被跳过")

        involved = self.judge_only_categories
        if involved and active_categories is not None:
            ran = {str(c).strip().lower() for c in active_categories if str(c).strip()}
            involved = involved & ran
        if involved:
            names = "、".join(sorted(involved))
            notes.append(f"分类 {names} 仅由 judge 判定，同用例其它指标只记录不判定")

        return notes
