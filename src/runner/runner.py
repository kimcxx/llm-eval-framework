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
from src.runner.dimension import resolve_dimension
from src.runner.results import AttemptResult, CaseResult, EvalReport


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
        extra_notes: Sequence[str] | None = None,
    ) -> None:
        self.clients = clients
        self.factory = factory
        self.run = run
        self.model_configs = model_configs or {}
        self.config_path = config_path
        self.workers = max(1, workers or run.workers)
        self.verbose = verbose
        self.extra_notes = list(extra_notes or [])
        self.repeat = max(1, int(getattr(run, "repeat", 1) or 1))

    # ---------------- 主流程 ---------------- #

    def run_cases(self, cases: Sequence[EvalCase]) -> EvalReport:
        started = time.perf_counter()
        started_at = datetime.now().isoformat(timespec="seconds")
        models = list(self.clients.keys())

        if not cases:
            raise ValueError("没有可执行的评测用例")

        repeat = self.repeat
        calls = len(cases) * len(models) * repeat

        if self.verbose:
            print(
                f"开始评测：{len(cases)} 条用例 × {len(models)} 个模型 × {repeat} 次 "
                f"= {calls} 次调用（并发 {self.workers}）"
            )

        raw: list[CaseResult] = []
        total = calls
        done = 0
        progress_step = max(1, total // 10)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [
                pool.submit(self._evaluate_one, client, name, case, attempt)
                for name, client in self.clients.items()
                for case in cases
                for attempt in range(repeat)
            ]
            for future in as_completed(futures):
                raw.append(future.result())
                done += 1
                if self.verbose and done % progress_step == 0:
                    print(f"  进度 {done}/{total}")

        results = self._merge_attempts(cases, models, raw)

        finished_at = datetime.now().isoformat(timespec="seconds")
        report = EvalReport(
            started_at=started_at,
            finished_at=finished_at,
            duration_s=time.perf_counter() - started,
            models=models,
            datasets=sorted({c.source for c in cases}),
            cases=results,
            # 只披露本轮跑到的分类相关的口径，避免输出没跑的分类（如只跑 json_extract 却提 qa_open）
            notes=self.factory.notes(
                active_categories={c.category for c in cases}
            )
            + self.extra_notes
            + self._repeat_notes(repeat),
            config_path=self.config_path,
            repeat=repeat,
        )

        if self.verbose:
            print(f"评测完成，耗时 {report.duration_s:.1f}s")
        return report

    # ---------------- 单条用例 ---------------- #

    def _repeat_notes(self, repeat: int) -> list[str]:
        """把重复执行与限流的口径写进报告，避免读者误读通过率。"""
        notes: list[str] = []
        if repeat > 1:
            notes.append(
                f"每条用例独立执行 {repeat} 次（结果见 cases[].attempts）；"
                f"通过判定 = {repeat} 次全部通过，稳定率 = 逐次通过次数 / 总调用次数"
            )
        interval = getattr(self.run, "request_interval_s", 0.0) or 0.0
        if interval > 0:
            notes.append(
                f"全局请求间隔 {interval:g}s（约 {60 / interval:.0f} 次/分钟），用于规避供应商限流"
            )
        return notes

    def _merge_attempts(
        self,
        cases: Sequence[EvalCase],
        models: list[str],
        raw: list[CaseResult],
    ) -> list[CaseResult]:
        """把同一 (模型, 用例) 的 N 次执行合并成一条报告记录。

        合并后：延迟取平均、tokens 与成本取总和（因为真的消耗了 N 次），
        顶层 response/metrics 取第一次作为代表样本，每次详情留在 attempts 里。
        """
        order = {case.id: i for i, case in enumerate(cases)}
        model_order = {name: i for i, name in enumerate(models)}

        def sort_key(result: CaseResult) -> tuple[int, int]:
            return (model_order.get(result.model, len(models)), order.get(result.case_id, 0))

        if self.repeat == 1:
            raw.sort(key=sort_key)
            return raw

        grouped: dict[tuple[str, str], list[CaseResult]] = {}
        for item in raw:
            grouped.setdefault((item.model, item.case_id), []).append(item)

        merged: list[CaseResult] = []
        for items in grouped.values():
            items.sort(key=lambda r: r.attempts[0].index if r.attempts else 0)
            base = items[0]
            base.repeat = len(items)
            base.attempts = [item.attempts[0] for item in items if item.attempts]
            base.latency_ms = sum(item.latency_ms for item in items) / len(items)
            base.prompt_tokens = sum(item.prompt_tokens for item in items)
            base.completion_tokens = sum(item.completion_tokens for item in items)
            base.cost = sum(item.cost for item in items)

            errors = [item.error for item in items if item.error]
            # 部分失败不算「用例级调用异常」——那属于稳定性问题，交给 pass_count 表达
            base.error = (
                f"{len(errors)}/{len(items)} 次调用失败：{errors[-1]}"
                if len(errors) == len(items)
                else None
            )
            merged.append(base)

        merged.sort(key=sort_key)
        return merged

    def _evaluate_one(
        self,
        client: BaseLLM,
        model_name: str,
        case: EvalCase,
        attempt: int = 0,
    ) -> CaseResult:
        base = CaseResult(
            case_id=case.id,
            category=case.category,
            dataset=case.source,
            model=model_name,
            prompt=case.prompt,
            # 用例没声明 dimension 时按分类推导默认值，避免整份报告都落进 untagged
            dimension=resolve_dimension(case.category, case.dimension),
            expected=case.expected,
            repeat=1,
        )

        metrics, skipped = self.factory.resolve(case.metrics)
        base.skipped_metrics = skipped
        single = AttemptResult(index=attempt)

        try:
            response = self._call_with_retry(client, case)
        except LLMError as exc:
            base.error = str(exc)
            single.error = str(exc)
            base.attempts = [single]
            return base

        base.response_text = response.text
        base.latency_ms = response.latency_ms
        base.prompt_tokens = response.prompt_tokens
        base.completion_tokens = response.completion_tokens
        base.cost = self._estimate_cost(model_name, response)
        base.metrics = self._run_metrics(metrics, case, response)

        single.response_text = base.response_text
        single.latency_ms = base.latency_ms
        single.prompt_tokens = base.prompt_tokens
        single.completion_tokens = base.completion_tokens
        single.cost = base.cost
        single.metrics = base.metrics
        base.attempts = [single]
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

    def _run_metrics(self, metrics: Iterable, case: EvalCase, response: LLMResponse) -> list[MetricResult]:
        """逐个指标计算；单个指标异常不影响其他指标。

        judge-only 分类（如 qa_open）的通过与否只由裁判指标决定，
        其余指标仍计算并写入明细，但降级为「仅记录」，不计入通过判定。
        """
        results: list[MetricResult] = []
        for metric in metrics:
            try:
                result = metric.compute(case, response)
            except Exception as exc:  # noqa: BLE001 - 指标 bug 不应判定为模型失败
                result = MetricResult(
                    metric.name, 0.0, None, f"指标执行异常：{type(exc).__name__}: {exc}"
                )
            results.append(self.factory.as_record_only(result, case.category))
        return results

    def _estimate_cost(self, model_name: str, response: LLMResponse) -> float:
        cfg = self.model_configs.get(model_name)
        if cfg is None:
            return 0.0
        return (
            response.prompt_tokens / 1000 * cfg.price_per_1k_input
            + response.completion_tokens / 1000 * cfg.price_per_1k_output
        )
