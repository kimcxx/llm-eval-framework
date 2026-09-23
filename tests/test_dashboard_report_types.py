# -*- coding: utf-8 -*-
"""首页报告列表「区分报告类型」的单测。

覆盖两层：

1. ``_load_reports`` 的数据契约：列表要能区分「稳定性测试（repeat>1）」与
   「回归测试（repeat=1）」，就必须把 ``repeat`` 和逐模型的 ``stability``
   一起带出来——前端拿不到这两个字段，类型标签和「稳定率」列都会变空白。
2. PAGE 模板契约：类型标签 / 两种颜色 / 「30 × 3次」/ 列表取 stability 的分支，
   防止后续改模板时被删掉；老报告（无 stability）要能退回到 pass_rate。
"""
from __future__ import annotations

import json

import pytest

import dashboard.app as app
from dashboard.app import PAGE, _load_reports


def _write(tmp_path, name: str, payload: dict) -> None:
    (tmp_path / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _stability_report() -> dict:
    return {
        "started_at": "2026-09-23T11:46:38",
        "models": ["mock-baseline", "deepseek-chat"],
        "case_count": 60,
        "repeat": 3,
        "total_calls": 180,
        "stability": 0.8481,
        "summary": [
            {"model": "mock-baseline", "total": 30, "passed": 28, "failed": 2,
             "skipped": 0, "errors": 0, "pass_rate": 0.9333, "stability": 0.9333, "flaky": 0},
            {"model": "deepseek-chat", "total": 30, "passed": 18, "failed": 12,
             "skipped": 0, "errors": 0, "pass_rate": 0.6, "stability": 0.7556, "flaky": 9},
        ],
        "cases": [],
    }


def _legacy_report() -> dict:
    """老报告：没有 repeat / stability / flaky 字段。"""
    return {
        "started_at": "2026-09-22T19:00:23",
        "models": ["deepseek-chat"],
        "case_count": 98,
        "summary": [
            {"model": "deepseek-chat", "total": 98, "passed": 91, "failed": 7,
             "skipped": 0, "errors": 0, "pass_rate": 0.9286},
        ],
        "cases": [],
    }


@pytest.fixture()
def reports_dir(tmp_path, monkeypatch):
    _write(tmp_path, "report-safety-repeat3-20260923-114638.json", _stability_report())
    _write(tmp_path, "report-full-v3-20260922-190023.json", _legacy_report())
    monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)
    return tmp_path


# ============================== 数据契约 ============================== #


def test_repeat_is_exposed_for_type_label(reports_dir):
    items = {i["file"]: i for i in _load_reports()}
    assert items["report-safety-repeat3-20260923-114638.json"]["repeat"] == 3
    # 老报告没有 repeat 字段 → 按 1 处理，显示成「回归测试」而不是报错
    assert items["report-full-v3-20260922-190023.json"]["repeat"] == 1


def test_model_rows_expose_stability(reports_dir):
    items = {i["file"]: i for i in _load_reports()}
    rows = items["report-safety-repeat3-20260923-114638.json"]["model_rows"]

    assert [r["stability"] for r in rows] == [0.9333, 0.7556]
    # 老报告逐模型没有 stability：保留 None，前端退回 pass_rate
    legacy = items["report-full-v3-20260922-190023.json"]["model_rows"]
    assert legacy[0]["stability"] is None
    assert legacy[0]["pass_rate"] == 0.9286


def test_top_level_stability_is_exposed(reports_dir):
    items = {i["file"]: i for i in _load_reports()}
    assert items["report-safety-repeat3-20260923-114638.json"]["stability"] == 0.8481


# ============================== 模板契约 ============================== #


def test_page_has_both_type_labels():
    assert '稳定性测试' in PAGE
    assert '回归测试' in PAGE


def test_page_has_distinct_colors_per_type():
    assert '.badge.rtype-reg' in PAGE
    assert '.badge.rtype-stab' in PAGE


def test_page_renders_repeat_in_case_column():
    assert 'const caseCountText' in PAGE
    assert '× ${n}次' in PAGE


def test_page_list_rate_prefers_stability():
    """列表通过率列：稳定性报告取 stability，缺失时退回 pass_rate。"""
    assert 'const listRate = ' in PAGE
    assert 'x.stability != null' in PAGE


def test_page_column_header_switches():
    assert "hasStability ? '稳定率 / 通过率' : '通过率'" in PAGE
