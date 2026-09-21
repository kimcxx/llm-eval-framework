# -*- coding: utf-8 -*-
"""pytest junit-xml → 看板 tests-summary.json 转换器。

用法：
    pytest --junit-xml=tests-junit.xml -q
    python dashboard/parse_junit.py tests-junit.xml > dashboard/data/tests-summary.json
    # 或直接 import：
    from dashboard.parse_junit import parse
    summary = parse(Path("tests-junit.xml"), meta={"branch": "main", "commit": "abc1234"})

输入：pytest 生成的 junit XML 文件（pytest --junit-xml=...）
输出 dict / JSON 字段：
    - generated_at   ISO8601 时间戳
    - total / passed / failed / skipped / errors    用例数
    - duration_s      总耗时（秒，保留 2 位小数）
    - failed_tests    list[{classname, name, message}]  失败用例明细（最多 50 条）
    - branch / commit / run_id / run_url  注入的元信息（来自 GitHub Actions env，可选）
    - pytest_version  pytest 版本（注入式，可选）

零依赖（仅 Python 标准库）。异常降级：
- 文件缺失 → 抛 FileNotFoundError
- XML 解析失败 → 抛 ET.ParseError
- 测试套件无用例 → 返回全 0 的摘要（不抛错）
"""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _meta_from_env() -> dict[str, str]:
    """从 GitHub Actions 环境变量读取元信息（非 GitHub 环境安全降级为空）。"""
    keys = {
        "branch": "GITHUB_REF_NAME",
        "commit": "GITHUB_SHA",
        "run_id": "GITHUB_RUN_ID",
        "run_url": "GITHUB_SERVER_URL",  # 配合 run_id 拼出完整 URL
        "pytest_version": "PYTEST_VERSION",
    }
    out: dict[str, str] = {}
    for k, env in keys.items():
        v = os.environ.get(env)
        if v:
            if k == "run_url" and "GITHUB_RUN_ID" in os.environ:
                out["run_url"] = f"{v.rstrip('/')}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
            elif k != "run_url":
                out[k] = v
    return out


def parse(junit_xml: Path, meta: dict[str, str] | None = None) -> dict[str, Any]:
    """解析 pytest junit XML，返回看板所需的 summary dict。

    参数：
        junit_xml: pytest --junit-xml 产物路径
        meta: 可选元信息（如 branch/commit/run_url），后续由 GitHub Actions 注入

    字段含义见模块 docstring。
    """
    p = Path(junit_xml)
    if not p.is_file():
        raise FileNotFoundError(f"junit-xml 文件不存在: {p}")
    tree = ET.parse(p)
    root = tree.getroot()

    total = passed = failed = skipped = errors = 0
    duration_s = 0.0
    failed_tests: list[dict[str, str]] = []

    # pytest 默认根标签是 <testsuites>，单个 suite 时是 <testsuite>
    suites = list(root) if root.tag == "testsuites" else [root]
    for suite in suites:
        # 顶层统计
        try:
            total += int(suite.get("tests", "0"))
        except (TypeError, ValueError):
            pass
        duration_s += float(suite.get("time", "0") or 0)
        # 逐个 <testcase>
        for tc in suite.findall("testcase"):
            name = tc.get("name", "")
            classname = tc.get("classname", "")
            if tc.find("failure") is not None:
                failed += 1
                if len(failed_tests) < 50:
                    fail_node = tc.find("failure")
                    msg = fail_node.get("message", "") if fail_node is not None else ""
                    failed_tests.append({"classname": classname, "name": name, "message": msg[:500]})
            elif tc.find("error") is not None:
                errors += 1
            elif tc.find("skipped") is not None:
                skipped += 1
            else:
                passed += 1

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "duration_s": round(duration_s, 2),
        "failed_tests": failed_tests,
    }
    if meta:
        summary.update({k: v for k, v in meta.items() if v})
    return summary


def main() -> int:
    """CLI 入口：junit-xml 路径 → 摘要 JSON 输出。

    用法：
        python parse_junit.py <junit-xml>             # 输出到 stdout（UTF-8 字节）
        python parse_junit.py <junit-xml> -o <文件>   # 写入文件（避免管道编码问题）
        python parse_junit.py <junit-xml> --out <文件>
    """
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("用法: python parse_junit.py <junit-xml> [-o <文件>]", file=sys.stderr)
        return 2
    junit_xml = Path(args[0])
    out_path: Path | None = None
    if len(args) >= 3 and args[1] in ("-o", "--out"):
        out_path = Path(args[2])

    meta = _meta_from_env()
    summary = parse(junit_xml, meta=meta)
    body = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")
    else:
        # 直接写字节，绕过 stdout 的编码（Windows PowerShell 默认 UTF-16 会破坏 JSON）
        sys.stdout.buffer.write(body.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())