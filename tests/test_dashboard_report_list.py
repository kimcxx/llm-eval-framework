# -*- coding: utf-8 -*-
"""看板「报告列表」元数据契约测试。

背景：
    横向评测会把多个模型写进同一份报告（``summary`` 有多行）。
    ``_load_reports()`` 早期只取 ``summary[0]``，于是首页列表把整份横向
    报告显示成第一个模型（通常是被对照的 ``mock-baseline``），真实模型
    （如 ``deepseek-pro``）在列表里根本看不到，容易被误判为「报告没发布」。

契约要点：
    * 多模型报告的 ``model_rows`` 必须与 ``summary`` 一一对应且有序；
    * ``models`` 是全部模型名的有序列表；
    * 兼容旧字段：``model`` 仍为 ``summary[0]`` 的模型；
    * 单模型报告行为不变（``model_rows`` 只有一项）；
    * ``summary`` 缺失时给出兜底行，不能让看板出现 ``undefined``。
"""

from __future__ import annotations

import json
from pathlib import Path

import dashboard.app as app

# 看板自带数据副本：沙盒/CI 里也能跑，不依赖 reports/ 工作区目录。
DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"


def _write_report(directory: Path, name: str, payload: dict) -> None:
    (directory / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class TestMultiModelReport:
    """横向报告必须把每个模型都暴露给前端。"""

    def test_dashboard_data_contains_multi_model_report(self, monkeypatch) -> None:
        """数据目录里应至少有一份多模型报告，否则本组测试失去意义。"""
        monkeypatch.setattr(app, "REPORTS_DIR", DASHBOARD_DATA_DIR)
        multi = [it for it in app._load_reports() if len(it.get("model_rows") or []) > 1]
        assert multi, f"{DASHBOARD_DATA_DIR} 下找不到横向（多模型）报告，看板将只显示首个模型"

    def test_model_rows_match_summary_order(self, monkeypatch) -> None:
        """``models`` / ``model_rows`` 必须与报告 ``summary`` 的模型顺序一致。"""
        monkeypatch.setattr(app, "REPORTS_DIR", DASHBOARD_DATA_DIR)
        checked = 0
        for item in app._load_reports():
            if len(item.get("model_rows") or []) <= 1:
                continue
            raw = json.loads((DASHBOARD_DATA_DIR / item["file"]).read_text(encoding="utf-8"))
            expected = [row["model"] for row in raw["summary"]]
            assert item["models"] == expected, f"{item['file']} 的 models 与 summary 不一致"
            assert [r["model"] for r in item["model_rows"]] == expected, (
                f"{item['file']} 的 model_rows 与 summary 不一致"
            )
            # 兼容旧前端：model 仍取首个模型
            assert item["model"] == expected[0]
            checked += 1
        assert checked, "没有校验到任何多模型报告"


class TestSingleModelNotRegressed:
    """单模型报告与残缺数据的兜底行为。"""

    def test_single_model_report(self, tmp_path, monkeypatch) -> None:
        _write_report(
            tmp_path,
            "report-unit-20260101-000000.json",
            {
                "case_count": 2,
                "summary": [
                    {
                        "model": "solo",
                        "total": 2,
                        "passed": 1,
                        "failed": 1,
                        "pass_rate": 0.5,
                        "p95_latency_ms": 10.0,
                    }
                ],
            },
        )
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)
        items = app._load_reports()
        assert len(items) == 1
        assert items[0]["model"] == "solo"
        assert items[0]["models"] == ["solo"]
        assert items[0]["model_rows"] == [
            {"model": "solo", "total": 2, "passed": 1, "failed": 1, "pass_rate": 0.5, "p95": 10.0}
        ]

    def test_missing_summary_falls_back(self, tmp_path, monkeypatch) -> None:
        """没有 ``summary`` 段时仍要给出占位行，前端不能渲染出空行/undefined。"""
        _write_report(tmp_path, "report-unit-20260102-000000.json", {"case_count": 0})
        monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)
        items = app._load_reports()
        assert len(items) == 1
        assert items[0]["model_rows"] == [
            {"model": "?", "total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "p95": None}
        ]
