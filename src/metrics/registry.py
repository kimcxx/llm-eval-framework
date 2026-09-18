"""指标工厂：按名称装配指标，并明确区分「本地指标」与「需要裁判的指标」。"""

from __future__ import annotations

from src.llm.base import BaseLLM
from src.metrics.base import BaseMetric
from src.metrics.contains import ContainsMetric
from src.metrics.exact_match import ExactMatchMetric
from src.metrics.json_valid import JsonValidMetric
from src.metrics.judge import JudgeMetric
from src.metrics.not_contains import NotContainsMetric
from src.metrics.similarity import SimilarityMetric

LOCAL_METRICS = ("exact_match", "contains", "not_contains", "json_valid", "similarity")
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
        judge_client: BaseLLM | None = None,
        prefer_embedding: bool = True,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.judge_threshold = judge_threshold
        self.judge_client = judge_client
        self.prefer_embedding = prefer_embedding
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
            metric = JsonValidMetric()
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

    def notes(self) -> list[str]:
        """披露本次评测实际生效的指标实现，避免「静默降级」。"""
        notes: list[str] = []

        similarity = self._cache.get("similarity")
        if similarity is not None:
            describe = getattr(similarity, "describe", None)
            notes.append(describe() if callable(describe) else "similarity")

        if self.judge_client is not None:
            notes.append(f"裁判模型 {self.judge_client.name}（通过阈值 {self.judge_threshold:g}/5）")
        else:
            notes.append("未配置可用裁判模型，judge 指标已被跳过")

        return notes
