# -*- coding: utf-8 -*-
"""报告 ``time`` 取值与列表排序的回归测试。

背景（2026-11-03 工单，晓旭在页面发现「看不到新报告」）：
    文件名是 ``report-{tag}-{日期}-{时间}``。tag 自身带日期时
    （``report-3model-fix-20261103-20261103-104403``）文件里有**两段**日期，
    原先 ``re.search(r"\\d{8}-\\d{6}", stem)`` 抓**第一组**，切出
    ``20261103-202611``——而 ``_time_key`` 还能把它解析成 2026-11-03 20:26:11，
    **不报错**。于是两份报告的 ``time`` 撞车 → 稳定排序按文件名决定先后 →
    新报告排第二、首页「最新成绩」继续显示旧报告；跨次对比的默认基线也跟着错。

契约（本文件钉住）：
    * ``time`` 一律 ``YYYYMMDD-HHMMSS``——跨次对比 tab 拿它做字符串比大小，
      换成 ISO 会把前端比较弄坏；
    * 报告内部 ``started_at`` 优先，文件名只在没有内部时间时兜底；
    * 兜底取文件名**末尾**那段时间（tag 里的日期在前），不取第一段；
    * 列表严格按 time 倒序，不靠文件名撞运气。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

import dashboard.app as app

DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"

# 真实踩坑文件名：tag `3model-fix-20261103` 自带日期，与后缀日期撞成两段
REAL_NAME = "report-3model-fix-20261103-20261103-104403.json"

COMPACT = re.compile(r"^\d{8}-\d{6}$")


def _write_report(directory: Path, name: str, payload: dict) -> None:
    (directory / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _payload(started_at: str | None = None) -> dict:
    data = {
        "case_count": 1,
        "summary": [
            {"model": "m", "total": 1, "passed": 1, "failed": 0,
             "pass_rate": 1.0, "p95_latency_ms": 1.0}
        ],
    }
    if started_at:
        data["started_at"] = started_at
    return data


class TestStampSource:
    """``time`` 从哪儿取：内部优先、文件名末尾兜底。"""

    def test_tag_with_date_takes_tail_not_cross_segment(self, tmp_path, monkeypatch) -> None:
        """tag 带日期时只认末尾那段时间：``20261103-104403``，不是跨段的 ``20261103-202611``。"""
        _write_report(tmp_path, REAL_NAME, _payload())
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)
        item = app._load_reports()[0]
        assert item["time"] == "20261103-104403"
        assert item["time"] != "20261103-202611", "抓第一组会切出跨两段日期的假时刻"

    def test_started_at_wins_over_filename(self, tmp_path, monkeypatch) -> None:
        """报告里有 ``started_at`` 时以它为准——文件名时间戳本来就不可靠。"""
        # 文件名后缀是 10:44:03（落盘时刻），内部开始时刻是 10:38:32
        _write_report(tmp_path, REAL_NAME, _payload("2026-11-03T10:38:32"))
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)
        item = app._load_reports()[0]
        assert item["time"] == "20261103-103832", "内部 started_at 优先"

    def test_compact_14_digit_filename_fallback(self, tmp_path, monkeypatch) -> None:
        """没有内部时间、文件名又是 14 位连写时也要能认出来。"""
        _write_report(tmp_path, "report-clean-20261027214243.json", _payload())
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)
        assert app._load_reports()[0]["time"] == "20261027-214243"

    def test_no_source_at_all_does_not_crash(self, tmp_path, monkeypatch) -> None:
        """两处都认不出：给 ``?``，交给 ``_time_key`` 兜底（排到最后），不许抛错。"""
        _write_report(tmp_path, "report-weird.json", _payload())
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)
        assert app._load_reports()[0]["time"] == "?"

    def test_format_stays_compact_for_every_real_report(self, monkeypatch) -> None:
        """真实数据目录里所有报告：time 必须是紧凑串（前端拿它做字符串比较）。"""
        monkeypatch.setattr(app, "REPORTS_DIR", DASHBOARD_DATA_DIR)
        items = app._load_reports()
        assert items, "dashboard/data 里应有报告"
        bad = [(it["file"], it["time"]) for it in items if not COMPACT.match(str(it["time"]))]
        assert not bad, f"这些报告的 time 不是 YYYYMMDD-HHMMSS：{bad[:5]}"


class TestOrdering:
    """排序回归：新报告必须排在旧报告前面，且 time 不撞车。"""

    def test_newer_report_sorts_first_when_both_tags_carry_date(self, tmp_path, monkeypatch) -> None:
        """两份 tag 都带日期的报告：新的（10:38）必须排第一。

        旧实现下两份 time 都是 ``20261103-202611`` → 撞车 → 稳定排序按文件名
        定先后（``model-...`` 在 ``fix-...`` 前），新报告被挤到第二。
        """
        _write_report(
            tmp_path,
            "report-3model-20261103-20261103-095015.json",
            _payload("2026-11-03T09:45:26"),
        )
        _write_report(tmp_path, REAL_NAME, _payload("2026-11-03T10:38:32"))
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)
        items = app._load_reports()
        assert [it["file"] for it in items] == [
            REAL_NAME,
            "report-3model-20261103-20261103-095015.json",
        ]
        assert len({it["time"] for it in items}) == 2, "两份 time 不许撞车"

    def test_real_newest_report_is_first(self, monkeypatch) -> None:
        """真实数据：10:44 那份排第一（晓旭当时就是因为它没排第一才以为报告没生成）。"""
        if not (DASHBOARD_DATA_DIR / REAL_NAME).is_file():
            pytest.skip(f"{REAL_NAME} 不在 dashboard/data，跳过")
        monkeypatch.setattr(app, "REPORTS_DIR", DASHBOARD_DATA_DIR)
        items = app._load_reports()
        assert items[0]["file"] == REAL_NAME
        times = [it["time"] for it in items[:5]]
        assert len(set(times)) == len(times), f"前 5 份 time 有重复：{times}"

    def test_is_sorted_descending(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "REPORTS_DIR", DASHBOARD_DATA_DIR)
        items = app._load_reports()
        keys = [app._time_key(it["time"]) for it in items]
        assert keys == sorted(keys, reverse=True), "必须严格倒序"


class TestNegativeControl:
    """阴性对照：证明上面的断言不是白写的——旧写法确实会切出假时刻。"""

    def test_old_regex_produced_a_bogus_but_parseable_stamp(self) -> None:
        stem = "report-3model-fix-20261103-20261103-104403"
        buggy = re.search(r"\d{8}-\d{6}", stem).group(0)
        assert buggy == "20261103-202611", "旧写法抓第一组，跨在两段日期中间"
        # 关键：它**能**被解析，所以是静默错误，不会抛异常提醒你
        assert app._time_key(buggy) == datetime(2026, 11, 3, 20, 26, 11)
        # 正确取值
        assert app._report_stamp({}, stem + ".json") == "20261103-104403"

    def test_same_bogus_stamp_makes_two_reports_tie(self) -> None:
        """两份 tag 带日期的文件名，旧写法切出的 time 完全相同——这正是排序失灵的机制。"""
        a = re.search(r"\d{8}-\d{6}", "report-3model-20261103-20261103-095015").group(0)
        b = re.search(r"\d{8}-\d{6}", "report-3model-fix-20261103-20261103-104403").group(0)
        assert a == b == "20261103-202611", "撞车 → 稳定排序按文件名定先后"
