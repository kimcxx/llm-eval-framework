"""报告渲染。

产物有三个，对应三种消费场景：
- report.md   —— 人看（代码评审、简历作品展示、周报）
- report.json —— CI 看（与上一版本做 diff，判断是否能力退化）
- cases.csv   —— 表格看（面试时展示数据量，也方便二次分析）
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.runner.results import EvalReport

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _pretty(value: Any) -> str:
    """字典/列表用 JSON 展示，比 Python repr 更易读，且保留中文不转义。"""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _flatten(value: Any, limit: int = 120) -> str:
    """压平换行，避免多行回答把 Markdown 列表结构撑坏。"""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=(), default=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = lambda value: f"{value * 100:.1f}%"
    env.filters["num"] = lambda value: f"{value:.3f}"
    env.filters["ms"] = lambda value: f"{value:.0f}"
    env.filters["money"] = lambda value: f"{value:.4f}"
    env.filters["pretty"] = _pretty
    env.filters["flat"] = _flatten
    return env


def _summary_rows(report: EvalReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stats in report.overall():
        rows.append(
            {
                "model": stats.key,
                "total": stats.total,
                "passed": stats.passed,
                "failed": stats.failed,
                "skipped": stats.skipped,
                "errors": stats.errors,
                "pass_rate": stats.pass_rate,
                "avg_latency_ms": stats.avg_latency_ms,
                "p95_latency_ms": stats.p95_latency_ms,
                "cost": stats.cost,
                "total_tokens": stats.total_tokens,
                "avg_similarity": stats.avg_metric("similarity"),
                "avg_judge": stats.avg_metric("judge"),
            }
        )
    return rows


def _category_rows(report: EvalReport, model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stats in report.by_category(model):
        rows.append(
            {
                "category": stats.key,
                "total": stats.total,
                "passed": stats.passed,
                "pass_rate": stats.pass_rate,
                "avg_latency_ms": stats.avg_latency_ms,
                "errors": stats.errors,
            }
        )
    return rows


def _failure_rows(report: EvalReport, model: str, limit: int = 15) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report.failures(model, limit=limit):
        failed = [m for m in case.metrics if m.passed is False]
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "error": case.error,
                "expected": case.expected,
                "response": case.response_text,
                "reasons": [f"{m.name}: {m.detail}" for m in failed] or ["（无判定指标失败，可能是调用异常）"],
            }
        )
    return rows


def build_context(report: EvalReport) -> dict[str, Any]:
    return {
        "report": report,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_rows": _summary_rows(report),
        "category_rows": {m: _category_rows(report, m) for m in report.models},
        "failure_rows": {m: _failure_rows(report, m) for m in report.models},
        "skipped_metrics": report.skipped_metric_counts(),
        "metric_names": report.metric_names(),
    }


def render_markdown(report: EvalReport) -> str:
    template = _environment().get_template("report.md.j2")
    return template.render(**build_context(report))


def write_reports(
    report: EvalReport,
    output_dir: Path,
    *,
    tag: str | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """写出 Markdown / JSON / CSV，返回产物路径。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"{tag}-{stamp}" if tag else stamp

    md_path = output_dir / f"report-{prefix}.md"
    json_path = output_dir / f"report-{prefix}.json"
    csv_path = output_dir / f"cases-{prefix}.csv"
    latest_path = output_dir / "latest.md"

    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(report, csv_path)

    latest_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    if verbose:
        print(f"报告已写入：\n  {md_path}\n  {json_path}\n  {csv_path}")

    return {"markdown": md_path, "json": json_path, "csv": csv_path, "latest": latest_path}


def _write_csv(report: EvalReport, path: Path) -> None:
    try:
        import pandas as pd
    except ImportError:  # 无 pandas 时退化为纯标准库写 CSV
        _write_csv_stdlib(report, path)
        return

    rows = []
    for case in report.cases:
        row: dict[str, Any] = {
            "model": case.model,
            "dataset": case.dataset,
            "category": case.category,
            "case_id": case.case_id,
            "passed": case.passed,
            "error": case.error,
            "latency_ms": round(case.latency_ms, 2),
            "prompt_tokens": case.prompt_tokens,
            "completion_tokens": case.completion_tokens,
            "cost": case.cost,
            "response": case.response_text,
        }
        for metric in case.metrics:
            row[f"score_{metric.name}"] = round(metric.score, 4)
            row[f"pass_{metric.name}"] = metric.passed
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_csv_stdlib(report: EvalReport, path: Path) -> None:
    import csv

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "dataset", "category", "case_id", "passed", "error", "latency_ms", "cost", "response"]
        )
        for case in report.cases:
            writer.writerow(
                [
                    case.model,
                    case.dataset,
                    case.category,
                    case.case_id,
                    case.passed,
                    case.error or "",
                    round(case.latency_ms, 2),
                    round(case.cost, 6),
                    case.response_text,
                ]
            )
