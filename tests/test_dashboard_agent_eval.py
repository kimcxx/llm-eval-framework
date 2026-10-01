# -*- coding: utf-8 -*-
"""看板「Agent 评测」板块的数据层单元测试（dashboard/app.py）。

覆盖三个契约点：
    * ``load_agent_runs()``：扫 ``agent_eval/results/``（含 ``archive/``），
      按时间倒序返回；目录不存在返回空列表（前端走空态，不崩页）。
    * **INVALID 过滤**：文件名带 ``-INVALID-`` 的是作废存档，一条都不许返回。
      这是硬约束——那两份是「通过了自己出的题」的成绩，显示出来等于把事故
      又摆一遍（见 agent_eval/results/README.md）。
    * **报告配对**（``_pair_agent_reports``）：``result-<戳>.json`` ↔
      ``agent-eval-<戳>.html``；时间戳完全一致优先，配不上取「该结果之后生成的
      最新一份」，一份报告只认一份结果；都配不上就留 None（前端显示「—」）。

测试手法：monkeypatch 掉 app 上的三个目录常量指向 tmp_path，不碰仓库里真实
的 agent_eval/ 数据。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard import app


# ---------- helpers ----------

def _result_payload(*, passed: int = 10, total: int = 10, repeat: int = 1) -> dict:
    """构造一份 agent_eval 结果 JSON（字段与 run_eval.py 落盘口径一致）。"""
    payload = {
        "repeat": repeat,
        "started_at": "2026-10-01T10:00:00",
        "finished_at": "2026-10-01T10:05:00",
        "summary": {
            "任务总数": total,
            "通过": passed,
            "失败": total - passed,
            "无效": 0,
            "任务成功率": passed / total,
            "工具选择正确率": 1.0,
            "平均步数": 2.6,
            "幻觉题是否守住": passed == total,
            "幻觉题": [{"id": "t07-unknown-dimension-honesty", "status": "通过"}],
        },
        "results": [],
    }
    if repeat > 1:
        payload["repeat_summary"] = {
            "遍数": repeat,
            "逐题": [{"id": "t01-basic-lookup", "各遍状态": ["通过"] * repeat,
                      "各遍步数": [2] * repeat, "状态一致": True}],
            "状态一致题数": 1,
            "全部一致": True,
        }
    return payload


def _write_result(directory: Path, stamp: str, *, name: str | None = None, **kwargs) -> Path:
    """在 directory 下写一份 result-<stamp>.json，返回路径。"""
    directory.mkdir(parents=True, exist_ok=True)
    filename = name or f"result-{stamp}.json"
    path = directory / filename
    path.write_text(json.dumps(_result_payload(**kwargs), ensure_ascii=False), encoding="utf-8")
    return path


def _write_report(directory: Path, stamp: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"agent-eval-{stamp}.html"
    path.write_text(f"<html><!-- {stamp} --></html>", encoding="utf-8")
    return path


@pytest.fixture()
def agent_dirs(tmp_path: Path, monkeypatch):
    """把三个目录常量指向 tmp_path，返回 (results, archive, reports)。"""
    results = tmp_path / "results"
    archive = results / "archive"
    reports = tmp_path / "reports"
    for d in (results, archive, reports):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app, "AGENT_RESULTS_DIR", results)
    monkeypatch.setattr(app, "AGENT_ARCHIVE_DIR", archive)
    monkeypatch.setattr(app, "AGENT_REPORTS_DIR", reports)
    return results, archive, reports


# ---------- load_agent_runs() ----------

def test_load_agent_runs_reads_results_and_archive_sorted_desc(agent_dirs):
    """results/ 与 archive/ 都扫，按时间倒序；archived 标记要带出来。"""
    results, archive, _ = agent_dirs
    _write_result(results, "20261001-103426")
    _write_result(archive, "20260929-175350")

    runs = app.load_agent_runs()
    assert len(runs) == 2
    assert runs[0]["file"] == "result-20261001-103426.json"
    assert runs[0]["archived"] is False
    assert runs[1]["file"] == "result-20260929-175350.json"
    assert runs[1]["archived"] is True
    # 归一化字段：数值与 repeat_summary 都要在
    assert runs[0]["total"] == 10 and runs[0]["passed"] == 10
    assert runs[0]["repeat"] == 1
    assert "repeat_summary" not in runs[0]  # 单遍结果没有可比对象，不塞空数组


def test_load_agent_runs_repeat_summary_only_when_repeat_gt_1(agent_dirs):
    """repeat>1 才带逐题一致性；repeat=1 不带（否则前端渲染出「1 遍」的空表）。"""
    results, _, _ = agent_dirs
    _write_result(results, "20261001-103426", repeat=3)

    runs = app.load_agent_runs()
    assert runs[0]["repeat"] == 3
    assert runs[0]["repeat_summary"]["遍数"] == 3
    assert runs[0]["repeat_summary"]["全部一致"] is True


def test_load_agent_runs_missing_dir_returns_empty(tmp_path: Path, monkeypatch):
    """目录不存在（部署沙盒里没有 agent_eval/）返回空列表，不抛错。"""
    monkeypatch.setattr(app, "AGENT_RESULTS_DIR", tmp_path / "nope")
    monkeypatch.setattr(app, "AGENT_ARCHIVE_DIR", tmp_path / "nope" / "archive")
    assert app.load_agent_runs() == []


def test_load_agent_runs_skips_broken_json(agent_dirs):
    """坏 JSON 跳过而不是让整块崩掉（看板其余部分还得照常显示）。"""
    results, _, _ = agent_dirs
    (results / "result-20261001-103426.json").write_text("{broken", encoding="utf-8")
    _write_result(results, "20260929-175350")
    assert [r["file"] for r in app.load_agent_runs()] == ["result-20260929-175350.json"]


# ---------- INVALID 过滤 ----------

def test_load_agent_runs_filters_invalid(agent_dirs):
    """正式结果与 INVALID 混在一起：INVALID 一条都不返回。"""
    results, archive, _ = agent_dirs
    _write_result(results, "20261001-103426")
    _write_result(results, "20260929-173057", name="result-INVALID-自行出题-20260929-173057.json")
    _write_result(archive, "20260929-174406", name="result-INVALID-自行出题-20260929-174406.json")

    runs = app.load_agent_runs()
    assert [r["file"] for r in runs] == ["result-20261001-103426.json"]
    assert all("-INVALID-" not in r["file"] for r in runs)


def test_load_agent_runs_all_invalid_returns_empty(agent_dirs):
    """边界：目录里只有 INVALID 存档时返回空列表（前端显示「暂无结果」）。"""
    results, _, _ = agent_dirs
    _write_result(results, "20260929-173057", name="result-INVALID-自行出题-20260929-173057.json")
    _write_result(results, "20260929-174406", name="result-INVALID-自行出题-20260929-174406.json")
    assert app.load_agent_runs() == []


# ---------- 报告配对 ----------

def test_pair_report_exact_match(agent_dirs):
    """结果戳与报告戳完全一致 → report_exact=True（--stamp 之后的常态）。"""
    results, _, reports = agent_dirs
    _write_result(results, "20261001-103426")
    _write_report(reports, "20261001-103426")

    run = app.load_agent_runs()[0]
    assert run["report"] == "agent-eval-20261001-103426.html"
    assert run["report_exact"] is True


def test_pair_report_fallback_to_later_report(agent_dirs):
    """没有同戳报告时，取该结果之后生成的最新一份，并标记 report_exact=False。"""
    results, _, reports = agent_dirs
    _write_result(results, "20261001-103426")
    _write_report(reports, "20261001-115744")  # 报告生成时间晚于结果

    run = app.load_agent_runs()[0]
    assert run["report"] == "agent-eval-20261001-115744.html"
    assert run["report_exact"] is False


def test_pair_report_none_when_missing(agent_dirs):
    """边界：reports/ 为空（或目录不存在）→ 配不上，report 保持 None。"""
    results, _, _ = agent_dirs
    _write_result(results, "20261001-103426")

    run = app.load_agent_runs()[0]
    assert run["report"] is None
    assert run["report_exact"] is False


def test_pair_report_not_claimed_twice(agent_dirs):
    """一份报告只认一份结果（认最近的那次），不能多行指向同一份。"""
    results, _, reports = agent_dirs
    _write_result(results, "20261001-103426")
    _write_result(results, "20260929-175350")
    _write_report(reports, "20261001-115744")

    runs = app.load_agent_runs()
    paired = [r for r in runs if r["report"]]
    assert len(paired) == 1
    assert paired[0]["file"] == "result-20261001-103426.json"
    assert paired[0]["report_exact"] is False
