#!/usr/bin/env python
"""评测入口 CLI。

用法示例：
    python scripts/run_eval.py --models mock-baseline
    python scripts/run_eval.py --models deepseek-chat qwen-plus --tag v1
    python scripts/run_eval.py --datasets math_reasoning --limit 5
    python scripts/run_eval.py --models mock-baseline --no-embedding --categories qa_zh,math_reasoning
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG_PATH, ConfigError, load_config  # noqa: E402
from src.datasets import EvalCase, load_datasets  # noqa: E402
from src.llm import build_client, build_clients  # noqa: E402
from src.llm.base import BaseLLM  # noqa: E402
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
    parser.add_argument(
        "--categories",
        default=None,
        help="只评测指定类别（逗号分隔，如 --categories qa_zh,math；类别名大小写不敏感）",
    )
    parser.add_argument("--workers", type=int, default=None, help="并发线程数")
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="每条用例独立执行多少次（默认取配置的 run.repeat，通常 1）；>1 时统计稳定率",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=None,
        help="相邻两次模型调用的最小间隔秒数（默认取配置的 run.request_interval_s，0=不限流）",
    )
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


def drop_judge_from_eval(
    clients: dict[str, BaseLLM],
    judge_name: str | None,
    explicit_names: Iterable[str] = (),
) -> str:
    """决定裁判模型是否留在被测列表里。

    裁判模型同时当选手会自评（尤其 judge_only_categories 下的分类），所以默认移除；
    只有用户用 --models 显式点名时才保留，由调用方负责给出自评警告。

    返回：'removed'（已移除）/ 'kept'（显式指定，保留）/ 'absent'（本来就不在被测列表）。
    """
    if not judge_name or judge_name not in clients:
        return "absent"
    if judge_name in set(explicit_names):
        return "kept"
    clients.pop(judge_name)
    return "removed"


def print_console_summary(report) -> None:
    repeated = getattr(report, "repeat", 1) > 1
    width = 84 if repeated else 74

    print("\n" + "=" * width)
    header = f"{'模型':<20}{'用例':>6}{'通过':>6}{'失败':>6}{'跳过':>6}{'通过率':>10}"
    if repeated:
        header += f"{'稳定率':>10}"
    header += f"{'P95(ms)':>10}"
    print(header)
    print("-" * width)
    for stats in report.overall():
        line = (
            f"{stats.key:<20}{stats.total:>6}{stats.passed:>6}{stats.failed:>6}"
            f"{stats.skipped:>6}{stats.pass_rate:>9.1%}"
        )
        if repeated:
            line += f"{stats.stability:>9.1%}"
        line += f"{stats.p95_latency_ms:>10.0f}"
        print(line)
    print("=" * width)

    if repeated:
        print(
            f"稳定率 = 逐次通过次数 / 总调用次数（全局 {report.stability:.1%}）；"
            f"通过判定要求 {report.repeat} 次全部通过，波动用例 {report.flaky_count} 条"
        )

    skipped = report.skipped_metric_counts()
    if skipped:
        print(f"注意：以下指标因不可用被跳过 —— {skipped}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    # CLI 覆盖配置：命令行是最贴近当次运行的意图，优先级高于 YAML
    if args.repeat is not None:
        config.run.repeat = args.repeat
    if args.request_interval is not None:
        config.run.request_interval_s = args.request_interval

    clients = build_clients(config, args.models, strict=args.strict)
    model_configs = {cfg.name: cfg for cfg in config.models}

    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    cases = load_datasets(config.paths.datasets_dir, args.datasets, categories=categories)
    cases = apply_limit(cases, args.limit)

    # --models 里点名的模型做一次归一化，便于与裁判模型按配置名比对
    explicit_names: set[str] = set()
    for name in args.models or []:
        try:
            explicit_names.add(config.get_model(name).name)
        except ConfigError:
            pass

    judge_client = None
    judge_cfg = config.get_judge()
    extra_notes: list[str] = []
    if judge_cfg is not None:
        if judge_cfg.is_available:
            judge_client = build_client(judge_cfg, config.run)
            state = drop_judge_from_eval(clients, judge_cfg.name, explicit_names)
            if state == "removed":
                print(
                    f"裁判模型 {judge_cfg.name} 已加载，已从被测模型列表移除"
                    f"（如需同时评测，请用 --models 显式指定）"
                )
            elif state == "kept":
                extra_notes.append(
                    f"裁判模型 {judge_cfg.name} 同时作为被测模型（--models 显式指定），"
                    f"其 judge 判定属于自评，结果仅供参考"
                )
                print(
                    f"警告：裁判模型 {judge_cfg.name} 同时参与评测（--models 显式指定），"
                    f"相关 judge 判定存在自评风险"
                )
            else:
                print(f"裁判模型 {judge_cfg.name} 已加载（不计入被测模型）")
        else:
            print(f"裁判模型 {judge_cfg.name} 不可用（{judge_cfg.unavailable_reason}），judge 指标将被跳过")

    factory = MetricFactory(
        similarity_threshold=config.run.similarity_threshold,
        judge_threshold=config.run.judge_threshold,
        schema_match_threshold=config.run.schema_match_threshold,
        judge_client=judge_client,
        prefer_embedding=not args.no_embedding,
        judge_only_categories=config.run.judge_only_categories,
    )

    runner = EvalRunner(
        clients=clients,
        factory=factory,
        run=config.run,
        model_configs=model_configs,
        config_path=str(args.config),
        workers=args.workers,
        extra_notes=extra_notes,
    )

    report = runner.run_cases(cases)
    print_console_summary(report)
    write_reports(report, config.paths.output_dir, tag=args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
