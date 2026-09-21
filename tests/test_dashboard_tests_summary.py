# -*- coding: utf-8 -*-
"""看板 CI 测试摘要（junit-xml → summary）的单元测试。

背景：
    CD 流水线（.github/workflows/cd.yml）的"解析测试摘要"步骤会调用
    ``dashboard/parse_junit.parse``，输出 ``dashboard/data/tests-summary.json``。
    看板首页的「CI 测试」卡片读这个文件展示通过/失败/耗时。

契约要点：
    * ``parse(junit_xml, meta=None)``：返回字段必须包含
      ``generated_at / total / passed / failed / skipped / errors / duration_s /
       failed_tests``；meta 注入的字段（branch/commit/run_url 等）会与基础字段合并。
    * 文件缺失 → ``FileNotFoundError``；XML 解析失败 → ``ET.ParseError``。
    * ``failed_tests`` 上限 50 条（防止大失败集撑爆看板首页）。
    * ``<testsuites>``（pytest 默认根）和 ``<testsuite>``（单 suite）两种 XML
      根都要支持。
    * ``_load_tests_summary()``：看板 helper，文件缺失返回 None（让前端走空态
      渲染而不是崩页），文件存在则返回 dict。

注意：
    * 不需要 pytest 实际运行（不依赖 fixtures/conftest 里的真实 junit-xml）；
      测试在内存里构造 junit XML 再解析。
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dashboard.app import _load_tests_summary
from dashboard.parse_junit import parse


# ---------- helpers ----------

JUNIT_TPL = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest" tests="{total}" failures="{failed}" errors="{errors}" skipped="{skipped}" time="{time}">
  <testsuite name="t" tests="{total}" failures="{failed}" errors="{errors}" skipped="{skipped}" time="{time}" timestamp="2026-09-21T00:00:00">
{cases}
  </testsuite>
</testsuites>"""


def _case_xml(name: str, classname: str = "tests.t", *, fail: bool = False, error: bool = False, skip: bool = False) -> str:
    """构造一条 <testcase> XML；fail/error/skip 三选一（默认 pass）。"""
    child = ""
    if fail:
        child = f'<failure message="assert failed">assert 1 == 2</failure>'
    elif error:
        child = f'<error message="boom">Traceback...</error>'
    elif skip:
        child = f'<skipped message="skip reason" />'
    return f'    <testcase classname="{classname}" name="{name}" time="0.001">{child}</testcase>'


def _write_junit(tmp_path: Path, *, total: int = 0, failed: int = 0, errors: int = 0, skipped: int = 0, time: float = 1.5, cases: list[str] | None = None) -> Path:
    """写出 junit-xml 文件。cases 不指定时按 total 自动填充占位用例。"""
    if cases is None:
        n_pass = total - failed - errors - skipped
        cases = (
            [_case_xml(f"p{i}") for i in range(n_pass)]
            + [_case_xml(f"f{i}", fail=True) for i in range(failed)]
            + [_case_xml(f"e{i}", error=True) for i in range(errors)]
            + [_case_xml(f"s{i}", skip=True) for i in range(skipped)]
        )
    xml = JUNIT_TPL.format(total=total, failed=failed, errors=errors, skipped=skipped, time=time, cases="\n".join(cases))
    p = tmp_path / "junit.xml"
    p.write_text(xml, encoding="utf-8")
    return p


# ---------- parse() 基本正确性 ----------

def test_parse_basic_all_pass(tmp_path: Path):
    """全通过场景：10 个用例全 pass。"""
    cases = [_case_xml(f"p{i}") for i in range(10)]
    p = _write_junit(tmp_path, total=10, cases=cases)
    s = parse(p)
    assert s["total"] == 10
    assert s["passed"] == 10
    assert s["failed"] == 0
    assert s["errors"] == 0
    assert s["skipped"] == 0
    assert s["duration_s"] == 1.5
    assert s["failed_tests"] == []
    assert "generated_at" in s and s["generated_at"]


def test_parse_mixed_results(tmp_path: Path):
    """混合状态：1 pass + 2 failed + 1 error + 1 skipped。"""
    cases = (
        [_case_xml("ok1")]
        + [_case_xml(f"fail{i}", fail=True) for i in range(2)]
        + [_case_xml("err1", error=True)]
        + [_case_xml("skip1", skip=True)]
    )
    p = _write_junit(tmp_path, total=5, failed=2, errors=1, skipped=1, cases=cases)
    assert parse(p)["passed"] == 1
    s = parse(p)
    assert s["passed"] == 1
    assert s["failed"] == 2
    assert s["errors"] == 1
    assert s["skipped"] == 1
    assert s["total"] == 5


def test_parse_records_failed_tests_details(tmp_path: Path):
    """失败用例要写进 failed_tests：含 classname/name/message。"""
    cases = [_case_xml("ok"), _case_xml("boom", fail=True)]
    p = _write_junit(tmp_path, total=2, failed=1, cases=cases)
    ft = parse(p)["failed_tests"]
    assert len(ft) == 1
    assert ft[0]["classname"] == "tests.t"
    assert ft[0]["name"] == "boom"
    assert "assert" in ft[0]["message"]


def test_parse_failed_tests_capped_at_50(tmp_path: Path):
    """失败明细上限 50 条（防止大失败集撑爆看板首页）。"""
    cases = [_case_xml(f"f{i}", fail=True) for i in range(80)]
    p = _write_junit(tmp_path, total=80, failed=80, cases=cases)
    assert len(parse(p)["failed_tests"]) == 50


def test_parse_meta_injection_merges_into_summary(tmp_path: Path):
    """外部传入 meta（branch/commit/run_url）会合并进返回 dict。"""
    p = _write_junit(tmp_path, total=1, cases=[_case_xml("ok")])
    s = parse(p, meta={"branch": "main", "commit": "abc1234", "run_url": "https://github.com/x/y/actions/runs/1"})
    assert s["branch"] == "main"
    assert s["commit"] == "abc1234"
    assert s["run_url"].endswith("/runs/1")
    # 基础字段不丢
    assert s["total"] == 1


def test_parse_meta_empty_values_dropped(tmp_path: Path):
    """meta 里的空值不写入返回 dict（避免 None/空串污染 JSON）。"""
    p = _write_junit(tmp_path, total=1, cases=[_case_xml("ok")])
    s = parse(p, meta={"branch": "main", "commit": "", "run_url": None})
    assert s["branch"] == "main"
    assert "commit" not in s
    assert "run_url" not in s


def test_parse_handles_single_testsuite_root(tmp_path: Path):
    """当 XML 根是 <testsuite>（单 suite）而非 <testsuites> 也要能解析。"""
    xml = """<?xml version="1.0"?>
    <testsuite name="t" tests="2" failures="1" errors="0" skipped="0" time="0.5">
      <testcase classname="x" name="ok" time="0.001"/>
      <testcase classname="x" name="boom" time="0.001"><failure message="oops">trace</failure></testcase>
    </testsuite>"""
    p = tmp_path / "single.xml"
    p.write_text(xml, encoding="utf-8")
    s = parse(p)
    assert s["total"] == 2
    assert s["passed"] == 1
    assert s["failed"] == 1


# ---------- parse() 异常降级 ----------

def test_parse_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse(tmp_path / "no.xml")


def test_parse_invalid_xml_raises(tmp_path: Path):
    p = tmp_path / "bad.xml"
    p.write_text("<not-valid-xml", encoding="utf-8")
    with pytest.raises(ET.ParseError):
        parse(p)


def test_parse_no_cases_returns_zero(tmp_path: Path):
    """空套件（total=0、无 testcase）安全返回全 0 摘要，不抛错。"""
    p = _write_junit(tmp_path, total=0, cases=[])
    s = parse(p)
    assert s["total"] == 0
    assert s["passed"] == 0
    assert s["failed_tests"] == []


# ---------- _load_tests_summary()（看板端 helper） ----------

def test_load_tests_summary_returns_none_when_missing(tmp_path: Path, monkeypatch):
    """REPORTS_DIR 指向空目录时，_load_tests_summary 返回 None（不抛错）。"""
    from dashboard import app
    monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)
    assert app._load_tests_summary() is None


def test_load_tests_summary_returns_dict_when_valid(tmp_path: Path, monkeypatch):
    """REPORTS_DIR 有 tests-summary.json 时返回 dict。"""
    from dashboard import app
    monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)
    payload = {"total": 5, "passed": 5, "failed": 0, "duration_s": 0.3}
    (tmp_path / "tests-summary.json").write_text(json.dumps(payload), encoding="utf-8")
    assert app._load_tests_summary() == payload


def test_load_tests_summary_returns_none_on_broken_json(tmp_path: Path, monkeypatch):
    """JSON 损坏时降级为 None（让前端走空态而不是崩页）。"""
    from dashboard import app
    monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)
    (tmp_path / "tests-summary.json").write_text("{broken", encoding="utf-8")
    assert app._load_tests_summary() is None