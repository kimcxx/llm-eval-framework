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
    detail_tests: list[dict[str, Any]] = []

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
            time_s = round(float(tc.get("time", "0") or 0), 3)
            fail_node = tc.find("failure")
            err_node = tc.find("error")
            skip_node = tc.find("skipped")
            if fail_node is not None:
                failed += 1
                msg = fail_node.get("message", "") if fail_node is not None else ""
                if len(failed_tests) < 50:
                    failed_tests.append({"classname": classname, "name": name, "message": msg[:500]})
                if len(detail_tests) < 5000:
                    detail_tests.append({"classname": classname, "name": name, "status": "fail", "duration_s": time_s, "message": msg[:500]})
            elif err_node is not None:
                errors += 1
                msg = err_node.get("message", "") if err_node is not None else ""
                if len(detail_tests) < 5000:
                    detail_tests.append({"classname": classname, "name": name, "status": "error", "duration_s": time_s, "message": msg[:500]})
            elif skip_node is not None:
                skipped += 1
                msg = skip_node.get("message", "") if skip_node is not None else ""
                if len(detail_tests) < 5000:
                    detail_tests.append({"classname": classname, "name": name, "status": "skip", "duration_s": time_s, "message": msg[:200]})
            else:
                passed += 1
                if len(detail_tests) < 5000:
                    detail_tests.append({"classname": classname, "name": name, "status": "pass", "duration_s": time_s, "message": ""})

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "duration_s": round(duration_s, 2),
        "failed_tests": failed_tests,
        "detail_tests": detail_tests,
    }
    if meta:
        summary.update({k: v for k, v in meta.items() if v})
    return summary


def main() -> int:
    """CLI 入口：junit-xml 路径 → 摘要 JSON 输出。

    用法：
        python parse_junit.py <junit-xml>             # 输出到 stdout（UTF-8 字节）
        python parse_junit.py <junit-xml> -o <文件>   # 写入 summary 文件
        python parse_junit.py <junit-xml> --detail <文件>  # 同时输出用例明细到另一文件
    """
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("用法: python parse_junit.py <junit-xml> [-o <文件>] [--detail <文件>]", file=sys.stderr)
        return 2
    junit_xml = Path(args[0])
    out_path: Path | None = None
    detail_path: Path | None = None
    i = 1
    while i < len(args):
        a = args[i]
        if a in ("-o", "--out") and i + 1 < len(args):
            out_path = Path(args[i + 1]); i += 2
        elif a in ("-d", "--detail") and i + 1 < len(args):
            detail_path = Path(args[i + 1]); i += 2
        else:
            print(f"未知参数: {a}", file=sys.stderr)
            return 2

    meta = _meta_from_env()
    summary = parse(junit_xml, meta=meta)

    # summary 主体：剥掉 detail_tests（独立成文件，避免首页数据过大）
    summary_for_output = {k: v for k, v in summary.items() if k != "detail_tests"}
    body = json.dumps(summary_for_output, ensure_ascii=False, indent=2) + "\n"
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")
    else:
        sys.stdout.buffer.write(body.encode("utf-8"))

    # detail 文件：只包含用例列表 + 顶层元信息
    if detail_path:
        detail_payload = {
            "generated_at": summary["generated_at"],
            "total": summary["total"],
            "tests": summary["detail_tests"],
        }
        # 同样带 meta 字段（branch/commit/run_url），便于跳转
        for k in ("branch", "commit", "run_id", "run_url", "pytest_version"):
            if k in summary:
                detail_payload[k] = summary[k]
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(
            json.dumps(detail_payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())