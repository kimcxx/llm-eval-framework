"""评测执行引擎。

三条设计原则：
1. **单点失败不拖垮整轮评测** —— 某条用例报错只记为该用例失败，
   整轮继续跑完，否则长评测跑一半崩掉会浪费已消耗的额度；
2. **重试只针对可重试错误** —— 限流/超时可重试，参数错误/鉴权失败立刻放弃；
3. **并发可控** —— 通过 workers 控制对模型侧的压力，避免触发限流。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterable, Sequence

from src.config import ModelConfig, RunSettings
from src.datasets.schema import EvalCase
from src.llm.base import BaseLLM, LLMError, LLMResponse
from src.metrics.base import MetricResult
from src.metrics.registry import MetricFactory
from src.runner.results import CaseResult, EvalReport


class EvalRunner:
    def __init__(
        self,
        *,
        clients: dict[str, BaseLLM],
        factory: MetricFactory,
        run: RunSettings,
        model_configs: dict[str, ModelConfig] | None = None,
        config_path: str = "",
        workers: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.clients = clients
        self.factory = factory
        self.run = run
        self.model_configs = model_configs or {}
        self.config_path = config_path
        self.workers = max(1, workers or run.workers)
        self.verbose = verbose

    # ---------------- 主流程 ---------------- #

    def run_cases(self, cases: Sequence[EvalCase]) -> EvalReport:
        started = time.perf_counter()
        started_at = datetime.now().isoformat(timespec="seconds")
        models = list(self.clients.keys())

        if not cases:
            raise ValueError("没有可执行的评测用例")

        if self.verbose:
            print(
                f"开始评测：{len(cases)} 条用例 × {len(models)} 个模型 "
                f"= {len(cases) * len(models)} 次调用（并发 {self.workers}）"
            )

        results: list[CaseResult] = []
        total = len(cases) * len(models)
        done = 0
        progress_step = max(1, total // 10)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [
                pool.submit(self._evaluate_one, client, name, case)
                for name, client in self.clients.items()
                for case in cases
            ]
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if self.verbose and done % progress_step == 0:
                    print(f"  进度 {done}/{total}")

        # 排序保证报告稳定可复现（否则并发完成顺序会影响阅读）
        order = {case.id: i for i, case in enumerate(cases)}
        results.sort(key=lambda r: (models.index(r.model), order.get(r.case_id, 0)))

        finished_at = datetime.now().isoformat(timespec="seconds")
        report = EvalReport(
            started_at=started_at,
            finished_at=finished_at,
            duration_s=time.perf_counter() - started,
            models=models,
            datasets=sorted({c.source for c in cases}),
            cases=results,
            notes=self.factory.notes(),
            config_path=self.config_path,
        )

        if self.verbose:
            print(f"评测完成，耗时 {report.duration_s:.1f}s")
        return report

    # ---------------- 单条用例 ---------------- #

    def _evaluate_one(self, client: BaseLLM, model_name: str, case: EvalCase) -> CaseResult:
        base = CaseResult(
            case_id=case.id,
            category=case.category,
            dataset=case.source,
            model=model_name,
            prompt=case.prompt,
            dimension=case.dimension,
            expected=case.expected,
        )

        metrics, skipped = self.factory.resolve(case.metrics)
        base.skipped_metrics = skipped

        try:
            response = self._call_with_retry(client, case)
        except LLMError as exc:
            base.error = str(exc)
            return base

        base.response_text = response.text
        base.latency_ms = response.latency_ms
        base.prompt_tokens = response.prompt_tokens
        base.completion_tokens = response.completion_tokens
        base.cost = self._estimate_cost(model_name, response)
        base.metrics = self._run_metrics(metrics, case, response)
        return base

    def _call_with_retry(self, client: BaseLLM, case: EvalCase) -> LLMResponse:
        attempts = max(1, self.run.max_retries)
        last_error: LLMError | None = None

        for attempt in range(attempts):
            try:
                return client.complete(case.prompt, system=case.system)
            except LLMError as exc:
                last_error = exc
                if not exc.retryable or attempt == attempts - 1:
                    raise
                time.sleep(min(2**attempt, 8))  # 指数退避，缓解限流

        raise last_error or LLMError("未知调用失败")

    @staticmethod
    def _run_metrics(metrics: Iterable, case: EvalCase, response: LLMResponse) -> list[MetricResult]:
        """逐个指标计算；单个指标异常不影响其他指标。"""
        results: list[MetricResult] = []
        for metric in metrics:
            try:
                results.append(metric.compute(case, response))
            except Exception as exc:  # noqa: BLE001 - 指标 bug 不应判定为模型失败
                results.append(
                    MetricResult(metric.name, 0.0, None, f"指标执行异常：{type(exc).__name__}: {exc}")
                )
        return results

    def _estimate_cost(self, model_name: str, response: LLMResponse) -> float:
        cfg = self.model_configs.get(model_name)
        if cfg is None:
            return 0.0
        return (
            response.prompt_tokens / 1000 * cfg.price_per_1k_input
            + response.completion_tokens / 1000 * cfg.price_per_1k_output
        )
