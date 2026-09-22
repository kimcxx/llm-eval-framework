# -*- coding: utf-8 -*-
"""把看板导出为纯静态站点（用于 GitHub Pages 等静态托管）。

用法：
    python dashboard/build_static.py                          # 默认读取 dashboard/data
    python dashboard/build_static.py --reports-dir reports --out dashboard/dist

产物结构：
    <out>/index.html                 首页
    <out>/api/reports.json           报告列表（同 /api/reports）
    <out>/api/report/<报告文件名>    各报告详情（同 /api/report/<name>）
"""
import argparse
import json
from pathlib import Path

import app  # 复用看板的 PAGE 模板与数据加载逻辑


def build(reports_dir: Path, out_dir: Path) -> int:
    app.REPORTS_DIR = Path(reports_dir)
    reports = app._load_reports()

    out = Path(out_dir)
    (out / "api" / "report").mkdir(parents=True, exist_ok=True)

    (out / "index.html").write_text(app.PAGE, encoding="utf-8")
    (out / "api" / "reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for item in reports:
        src = app.REPORTS_DIR / item["file"]
        data = json.loads(src.read_text(encoding="utf-8"))
        # 与动态服务保持一致：汇总 tab 的结论由后端算好后一起导出
        if isinstance(data, dict):
            data["_conclusion"] = app.compute_conclusion(data)
        # 文件名保持与动态服务路由一致（api/report/<报告文件名>，本身已含 .json）
        (out / "api" / "report" / item["file"]).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    # CI 测试摘要（CD 流水线的 "解析测试摘要" 步骤生成），可缺失
    tests_summary = app._load_tests_summary()
    (out / "api" / "tests.json").write_text(
        json.dumps(tests_summary or {}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 测试详情（每条用例：classname/name/status/duration_s/message），可缺失
    detail_src = app.REPORTS_DIR / "tests-detail.json"
    if detail_src.is_file():
        # 不重序列化：直接拷贝原文更高效（detail 数组可能数千行）
        (out / "api" / "tests-detail.json").write_text(
            detail_src.read_text(encoding="utf-8"), encoding="utf-8"
        )

    print(f"静态站点已生成: {out}（{len(reports)} 份报告{'，含 CI 测试摘要' if tests_summary else ''}{' + 用例详情' if detail_src.is_file() else ''}）")
    return len(reports)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="导出静态评测报告看板")
    parser.add_argument(
        "--reports-dir",
        default=str(here / "data"),
        help="报告数据目录（默认 dashboard/data）",
    )
    parser.add_argument(
        "--out", default=str(here / "dist"), help="输出目录（默认 dashboard/dist）"
    )
    args = parser.parse_args()
    build(Path(args.reports_dir), Path(args.out))


if __name__ == "__main__":
    main()
