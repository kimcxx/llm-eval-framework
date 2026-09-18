#!/usr/bin/env python
"""评测入口 CLI。

用法示例：
    python scripts/run_eval.py --models mock-baseline
    python scripts/run_eval.py --models deepseek-chat qwen-plus --tag v1
    python scripts/run_eval.py --datasets math_reasoning --limit 5
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from src.datasets import EvalCase, load_datasets  # noqa: E402
from src.llm import build_client, build_clients  # noqa: E402
from src.metrics import MetricFactory  # noqa: E402
from src.report import write_reports  # noqa: E402
from src.runner import EvalRunner  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM 评测框架：用同一套用例集横向对比多个模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="模型配置文件")
    parser.add_argument("--models", nargs="*", help="只评测指定模型，默认评测全部可用模型")
    parser.add_argument("--datasets", nargs="*", help="只评测指定数据集（文件名，不含后缀）")
    parser.add_argument("--categories", nargs="*", help="只评测指定类别")
    parser.add_argument("--workers", type=int, default=None, help="并发线程数")
    parser.add_argument("--limit", type=int, default=None, help="每个数据集最多取多少条用例")
    parser.add_argument("--tag", default=None, help="报告文件名前缀，便于版本间对比")
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="强制使用字符级相似度（不加载向量模型，速度更快）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="缺少 API Key 的模型直接报错，而不是跳过",
    )
    return parser.parse_args(argv)


def apply_limit(cases: list[EvalCase], limit: int | None) -> list[EvalCase]:
    """按数据集分组后各取前 N 条，避免某个数据集独占采样名额。"""
    if not limit or limit <= 0:
        return cases

    grouped: OrderedDict[str, list[EvalCase]] = OrderedDict()
    for case in cases:
        grouped.setdefault(case.source, []).append(case)
    return [case for group in grouped.values() for case in group[:limit]]


def print_console_summary(report) -> None:
    print("\n" + "=" * 74)
    print(f"{'模型':<20}{'用例':>6}{'通过':>6}{'失败':>6}{'跳过':>6}{'通过率':>10}{'P95(ms)':>10}")
    print("-" * 74)
    for stats in report.overall():
        print(
            f"{stats.key:<20}{stats.total:>6}{stats.passed:>6}{stats.failed:>6}"
            f"{stats.skipped:>6}{stats.pass_rate:>9.1%}{stats.p95_latency_ms:>10.0f}"
        )
    print("=" * 74)

    skipped = report.skipped_metric_counts()
    if skipped:
        print(f"注意：以下指标因不可用被跳过 —— {skipped}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    clients = build_clients(config, args.models, strict=args.strict)
    model_configs = {cfg.name: cfg for cfg in config.models}

    cases = load_datasets(config.paths.datasets_dir, args.datasets)
    if args.categories:
        wanted = set(args.categories)
        cases = [c for c in cases if c.category in wanted]
        if not cases:
            print(f"没有匹配到任何类别 {sorted(wanted)} 的用例", file=sys.stderr)
            return 2
    cases = apply_limit(cases, args.limit)

    judge_client = None
    judge_cfg = config.get_judge()
    if judge_cfg is not None:
        if judge_cfg.is_available:
            judge_client = build_client(judge_cfg, config.run)
            if judge_cfg.name not in clients:
                print(f"裁判模型 {judge_cfg.name} 已加载（不计入被测模型）")
        else:
            print(f"裁判模型 {judge_cfg.name} 不可用（{judge_cfg.unavailable_reason}），judge 指标将被跳过")

    factory = MetricFactory(
        similarity_threshold=config.run.similarity_threshold,
        judge_threshold=config.run.judge_threshold,
        judge_client=judge_client,
        prefer_embedding=not args.no_embedding,
    )

    runner = EvalRunner(
        clients=clients,
        factory=factory,
        run=config.run,
        model_configs=model_configs,
        config_path=str(args.config),
        workers=args.workers,
    )

    report = runner.run_cases(cases)
    print_console_summary(report)
    write_reports(report, config.paths.output_dir, tag=args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
