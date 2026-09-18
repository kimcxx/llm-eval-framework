#!/usr/bin/env python
"""回归门禁：对比两次评测结果，判断模型能力是否发生退化。

为什么需要它：模型换了新版本、或者 Prompt 被改了一行，整体通过率可能只掉 1%，
但某个细分能力（比如安全防护）可能掉了 50%。只看整体指标会漏掉这类事故，
所以这里做「整体 + 分类」双层对比。

CI 用法：
    python scripts/check_regression.py \
        --current reports/report-new.json \
        --baseline reports/report-old.json \
        --tolerance 0.02

退出码 1 表示检测到退化，可直接用于阻断合并。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_report(path: str | Path) -> dict:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"报告文件不存在：{file}")
    return json.loads(file.read_text(encoding="utf-8"))


def compare_overall(
    current: dict, baseline: dict, tolerance: float
) -> tuple[list[dict], list[dict]]:
    """对比整体通过率。"""
    rows: list[dict] = []
    regressions: list[dict] = []

    base_index = {row["model"]: row for row in baseline.get("summary", [])}
    for row in current.get("summary", []):
        model = row["model"]
        base_row = base_index.get(model)

        if base_row is None:
            rows.append({"model": model, "baseline": None, "current": row["pass_rate"], "delta": None})
            continue

        delta = row["pass_rate"] - base_row["pass_rate"]
        item = {
            "model": model,
            "baseline": base_row["pass_rate"],
            "current": row["pass_rate"],
            "delta": delta,
        }
        rows.append(item)
        if delta < -tolerance:
            regressions.append(item)

    return rows, regressions


def compare_categories(
    current: dict, baseline: dict, tolerance: float
) -> tuple[list[dict], list[dict]]:
    """对比各模型下每个类别的通过率，定位到具体能力。"""
    rows: list[dict] = []
    regressions: list[dict] = []

    current_cats = current.get("categories", {})
    baseline_cats = baseline.get("categories", {})

    for model, entries in current_cats.items():
        base_index = {entry["category"]: entry for entry in baseline_cats.get(model, [])}
        for entry in entries:
            base_entry = base_index.get(entry["category"])
            if base_entry is None:
                continue

            delta = entry["pass_rate"] - base_entry["pass_rate"]
            item = {
                "model": model,
                "category": entry["category"],
                "baseline": base_entry["pass_rate"],
                "current": entry["pass_rate"],
                "delta": delta,
            }
            rows.append(item)
            if delta < -tolerance:
                regressions.append(item)

    return rows, regressions


def print_rows(title: str, rows: list[dict], key: str) -> None:
    print(f"\n{title}")
    print("-" * 72)
    for row in rows:
        if row["delta"] is None:
            print(f"{row[key]:<28}{'（基线无此模型）':>30}")
            continue
        if abs(row["delta"]) < 1e-9:
            mark = "持平"
        else:
            mark = "退步" if row["delta"] < 0 else "提升"
        print(
            f"{row[key]:<28}基线 {row['baseline']:>7.1%}  当前 {row['current']:>7.1%}"
            f"  {row['delta']:>+7.1%}  {mark}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 评测回归门禁")
    parser.add_argument("--current", required=True, help="本次评测的 JSON 报告")
    parser.add_argument("--baseline", required=True, help="作为基线的 JSON 报告")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="允许的通过率下降幅度（超过则判定退化）",
    )
    args = parser.parse_args(argv)

    current = load_report(args.current)
    baseline = load_report(args.baseline)

    overall_rows, overall_regressions = compare_overall(current, baseline, args.tolerance)
    category_rows, category_regressions = compare_categories(current, baseline, args.tolerance)

    print(
        f"对比基线：{args.baseline}\n"
        f"本次报告：{args.current}\n"
        f"容忍阈值：-{args.tolerance:.1%}"
    )
    print_rows("整体通过率", overall_rows, "model")
    if category_rows:
        print_rows("分类通过率", category_rows, "category")

    if overall_regressions or category_regressions:
        print(f"\n[FAIL] 检测到 {len(overall_regressions) + len(category_regressions)} 项能力退化：")
        for row in overall_regressions:
            print(f"  - 整体 {row['model']}：{row['delta']:+.1%}")
        for row in category_regressions:
            print(f"  - 类别 {row['model']}/{row['category']}：{row['delta']:+.1%}")
        return 1

    print("\n[PASS] 未检测到能力退化。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
