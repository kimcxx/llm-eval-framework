"""日期显示格式的护栏（晓旭 2026-10-28 在页面发现两处日期显示错误）。

两条根因都被钉在这里：

1. 后端用 ``\\d{8}-\\d{6}`` 取**第一组**——遇到 tag 里也带日期的文件名
   （``report-clean-20261027-20261027-214243``）就切出 ``20261027-202610``，
   #llm 页「成绩日期」于是显示成没人看得懂的半截串；
2. 前端 ``fmtTime`` 又按 15 位下标**二次切割**——RAG 侧的 ``time`` 本就已格式化
   成 ``2026-10-28 04:53:19``，再切一刀就成了乱码。

规矩：**时间戳只由后端解析一次**（``normalize_display_time`` → ``YYYY-MM-DD HH:MM``），
前端拿到什么显示什么；解析不出就显示 ``—``，**不许显示半截数字**。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dashboard import app

# 页面上允许出现的日期形态：'2026-10-27 21:42' / '2026-10-27' / '—'
DISPLAY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}( \d{2}:\d{2})?|—)$")


def _write(tmp_path: Path, name: str, payload: dict) -> None:
    (tmp_path / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _model_report(**overrides) -> dict:
    payload = {
        "started_at": "2026-10-27T21:35:55",
        "finished_at": "2026-10-27T21:42:43",
        "models": ["deepseek-chat"],
        "datasets": ["qa_zh"],
        "case_count": 98,
        "summary": [
            {"model": "deepseek-chat", "total": 98, "passed": 95, "failed": 3,
             "skipped": 0, "errors": 0, "pass_rate": 0.969},
        ],
        "cases": [],
    }
    payload.update(overrides)
    return payload


class TestNormalizeDisplayTime:
    """纯函数层：各种输入形态都要解析成人能读的样子。"""

    def test_iso_string(self) -> None:
        assert app.normalize_display_time("2026-10-27T21:35:55") == "2026-10-27 21:35"

    def test_compact_stamp(self) -> None:
        assert app.normalize_display_time("20261027-214243") == "2026-10-27 21:42"

    def test_already_formatted_is_not_re_sliced(self) -> None:
        # RAG 侧的 time 本来就是 'YYYY-MM-DD HH:MM:SS'，不许再被切一刀（只截到分钟）
        assert app.normalize_display_time("2026-10-28 04:53:19") == "2026-10-28 04:53"

    def test_unparsable_gives_dash_not_half_number(self) -> None:
        assert app.normalize_display_time(None, "", "??") == "—"
        assert app.normalize_display_time() == "—"

    @pytest.mark.parametrize(
        "src",
        [
            "2026-10-27T21:35:55",
            "20261027-214243",
            "2026-10-28 04:53:19",
            "report-full-local-20261027-20261027-184105.json",
            "result-20261026-120000.json",
            "2026-10-27",
            "",
        ],
    )
    def test_output_always_displayable(self, src: str) -> None:
        got = app.normalize_display_time(src)
        assert DISPLAY_RE.match(got), f"{src!r} 解析出 {got!r}，不是人能读的日期"

    def test_tagged_filename_uses_trailing_stamp(self) -> None:
        """阴性对照：tag 里也带日期的文件名，必须取**末尾**那组。

        当初的 bug 就是取第一组：``20261027-202610`` → 若实现改回取第一组，
        这里会得到 ``2026-10-27 20:21`` 而不是 ``2026-10-27 21:42``，必须变红。
        """
        got = app.normalize_display_time(None, "report-clean-20261027-20261027-214243.json")
        assert got == "2026-10-27 21:42"


class TestEntryCarriesDisplayTime:
    """入口层：列表/卡片拿到的对象里必须带规范化字段。"""

    def test_model_layer_entry_uses_started_at(self, tmp_path: Path, monkeypatch) -> None:
        _write(tmp_path, "report-clean-20261027-20261027-214243.json", _model_report())
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)

        item = app._load_reports()[0]
        assert item["time_display"] == "2026-10-27 21:35", "优先信任报告内的 started_at"
        assert DISPLAY_RE.match(item["time_display"])

    def test_model_layer_entry_falls_back_to_trailing_stamp(self, tmp_path: Path, monkeypatch) -> None:
        """阴性对照：报告里没有 started_at 时退回文件名，同样要取末尾那组。"""
        payload = _model_report()
        del payload["started_at"]
        del payload["finished_at"]
        _write(tmp_path, "report-clean-20261027-20261027-214243.json", payload)
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)

        item = app._load_reports()[0]
        assert item["time_display"] == "2026-10-27 21:42"

    def test_agent_entry_carries_display_time(self, tmp_path: Path) -> None:
        # Agent 结果文件名形如 result-20261028-125835.json；老结果里没有 finished_at
        path = tmp_path / "result-20261028-125835.json"
        path.write_text(
            json.dumps({"summary": {"任务总数": 10}, "finished_at": "2026-10-28T12:58:35"}, ensure_ascii=False),
            encoding="utf-8",
        )
        entry = app._agent_entry(path, archived=False)
        assert entry is not None
        assert entry["time_display"] == "2026-10-28 12:58"
        assert DISPLAY_RE.match(entry["time_display"])

    def test_rag_entry_carries_display_time(self, tmp_path: Path) -> None:
        path = tmp_path / "result-20261026-120000.json"
        path.write_text(
            json.dumps(
                {
                    "generated_at": "2026-10-26T12:00:00",
                    "models": ["rag-deepseek-chat"],
                    "summary": [{"model": "rag-deepseek-chat", "total": 13, "passed": 13,
                                 "failed": 0, "pass_rate": 1.0}],
                    "retrieval": {"k": 2, "scored": 10, "hit": 10, "recall_at_k": 1.0},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        entry = app._rag_entry(path)
        assert entry is not None
        assert DISPLAY_RE.match(entry["time_display"]), f"RAG 卡片拿到 {entry['time_display']!r}"


class TestFrontendDoesNotSlice:
    """前端不许再靠下标切字符串（改回切割就算回归）。"""

    def test_no_index_slicing_of_timestamps(self) -> None:
        src = Path(app.__file__).read_text(encoding="utf-8")
        for bad in ("slice(9,11)", "slice(11,13)", "slice(0,4)"):
            assert bad not in src, f"前端又出现了按下标切时间戳的写法：{bad}"

    def test_fmttime_is_passthrough(self) -> None:
        src = Path(app.__file__).read_text(encoding="utf-8")
        assert "const fmtTime = t => (t && String(t) !== '?') ? String(t) : '—';" in src, (
            "fmtTime 必须是原样显示（后端已规范化），不许再加工"
        )
