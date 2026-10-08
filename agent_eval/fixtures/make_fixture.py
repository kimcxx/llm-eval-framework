"""从 9/23 全量回归报告生成 **agent 评测专用 fixture**。

为什么要有这个文件：
    评测语境下数据源必须冻结，否则「读最新报告」会让金标悄悄过期——10-04 新全量
    回归落地后，工具返回的数字变了而 tasks.json 的金标还停在 9/23，10 道题里 4 道
    变成莫名其妙的 fail。fixture 化的职责是**稳定与可复现**，不是新鲜。

为什么是「专用小文件」而不是复制整份报告：
    工具对外只有一个查询 ``get_eval_result(model_name, dimension)``，模型 2 × 维度 4
    = **8 个数据点**就是全部契约。照抄整份报告会把 98 条用例明细也搬进来，既没必要
    也没人维护。这里按契约存 8 行，provenance 写清来源与冻结日期。

数值**一律由本脚本算出**（复用 tools.py 的 ``CATEGORY_DIMENSION`` 重算口径），
禁止手敲——手敲等于埋下一次漂移。改完口径重跑本脚本即可。

用法：
    python agent_eval/fixtures/make_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_eval.tools import (  # noqa: E402
    DIMENSION_LABELS,
    _dimension_rows,
    _real_models,
)

# 源报告：全库唯一与 tasks.json 金标同源的真全量回归（10/4 那份会要求金标全部重标，
# 且会让 10/1 的历史成绩失去可比基准，故不追新——详见 TEAM_NOTES 的 fixture 选型评估）
SOURCE_REPORT = PROJECT_ROOT / "reports" / "report-full-regression-20260923-143903.json"
OUT_PATH = Path(__file__).resolve().parent / "eval_data_fixture.json"

DATA_DATE = "2026-09-23"   # 数据本身的日期（源报告的跑测日）
FROZEN_AT = "2026-10-06"   # 冻结成 fixture 的日期
# 维度落盘顺序：人读的顺序（工具渲染不依赖它，两种数据源的渲染顺序由 DIMENSION_ORDER 统一）
DIMENSION_ORDER = ("correctness", "instruction_following", "safety", "relevance")


def build() -> dict:
    """按工具现行口径重算出 8 个数据点。"""
    doc = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    models = [m for m in _real_models(doc)]  # 排除 mock-* 对照组
    rows = []
    for model in models:
        dims = _dimension_rows(doc, model)
        for dim in DIMENSION_ORDER:
            row = dims.get(dim)
            if not row:
                continue
            rows.append({
                "model": model,
                "dimension": dim,
                "label": DIMENSION_LABELS[dim],
                "total": int(row["total"]),
                "passed": int(row["passed"]),
            })
    return {
        "provenance": {
            "source_report": SOURCE_REPORT.name,
            "source_report_path": f"reports/{SOURCE_REPORT.name}",
            "data_date": DATA_DATE,
            "frozen_at": FROZEN_AT,
            "purpose": (
                "agent 评测专用数据快照：评测时数据源冻结在 "
                f"{DATA_DATE} 这份全量回归上，保证成绩跨时间、跨大脑可比。"
            ),
            "maintenance": (
                "此后手工维护。「升级 fixture」= 一次显式决策：改数 + 金标重标 + "
                "v2_changes 留痕，三者一起做；永不追新。"
            ),
            "generated_by": "agent_eval/fixtures/make_fixture.py",
            "note": (
                "金标必须独立于被测系统人工标定；这里冻结的是「被测系统读的数据」，"
                "不是金标。两者同源（都来自 9/23 那份）但职责不同，别混为一谈。"
            ),
        },
        "contract": {
            "query": "get_eval_result(model_name, dimension)",
            "models": models,
            "dimensions": {k: DIMENSION_LABELS[k] for k in DIMENSION_ORDER},
            "data_points": len(rows),
        },
        "data": rows,
    }


def main() -> int:
    if not SOURCE_REPORT.is_file():
        print(f"[失败] 找不到源报告：{SOURCE_REPORT}（fixture 只能从这份报告生成）")
        return 1
    payload = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[生成] {OUT_PATH}")
    print(f"[数据源] {SOURCE_REPORT.name}（数据日期 {DATA_DATE}）")
    for row in payload["data"]:
        rate = row["passed"] / row["total"] if row["total"] else 0.0
        print(f"  {row['model']:<14} {row['label']:<5} {rate * 100:.1f}%  "
              f"({row['passed']}/{row['total']})")
    print(f"[共 {len(payload['data'])} 个数据点]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
