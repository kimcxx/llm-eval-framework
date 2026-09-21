# -*- coding: utf-8 -*-
"""看板历史报告 JSON 契约测试。

背景：
    新看板要在 ``cases`` 数组上做「用例列表 / 失败详情 / 跨次对比」等渲染。
    这意味着每份历史报告里的 ``cases`` 元素都必须字段齐全，否则前端在
    读 ``c.metrics[i].score`` / ``c.passed`` 等字段时会直接 ``undefined``，
    看板的整页渲染也会崩。本测试的目的就是把这种「残缺数据」在进入看板
    之前就挡住。

契约要点（每个被参数化的报告都生成独立断言项）：
    * 顶层必须存在 ``cases`` 数组；
    * ``len(cases) == case_count``；
    * 每条 ``case`` 必须包含：``case_id / category / passed / prompt /
      expected / response / metrics``，缺任一字段即失败；
    * ``passed`` 必须是 ``bool`` 或 ``None``（被跳过的用例允许 ``None``）；
    * ``metrics`` 必须是列表（允许空列表表示「无指标」），每个元素必须包含
      ``name / passed / score``，缺任一字段即失败。

注意：
    * 数据源是 ``dashboard/data/report-*.json``（看板自带数据副本），不依赖
      ``reports/`` 工作区目录；这样在沙盒/CI 里也能跑。
    * 一旦发现某份报告缺字段，请把 ``文件名 + 字段名`` 一并贴到 PR 评论里
      —— 这是给 developer 的修复指引，本测试不会自动修复数据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# 看板自带数据副本：沙盒/CI 里也能跑，不依赖 reports/ 工作区目录。
DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"

REQUIRED_CASE_KEYS: frozenset[str] = frozenset(
    {"case_id", "category", "passed", "prompt", "expected", "response", "metrics"}
)
REQUIRED_METRIC_KEYS: frozenset[str] = frozenset({"name", "passed", "score"})


def _all_report_paths() -> list[Path]:
    """扫描看板数据目录下所有 ``report-*.json``，按文件名排序保证稳定。"""
    if not DASHBOARD_DATA_DIR.is_dir():
        return []
    return sorted(DASHBOARD_DATA_DIR.glob("report-*.json"))


@pytest.fixture(scope="module")
def report_paths() -> list[Path]:
    """所有看板历史报告的绝对路径列表。"""
    return _all_report_paths()


# ============================== 顶层结构 ============================== #


class TestReportTopLevel:
    """每份报告都必须能被看板解析：cases 数组存在、长度等于 case_count。"""

    def test_has_at_least_one_report(self, report_paths: list[Path]) -> None:
        """看板数据目录里至少要有一份报告；否则新看板直接空转没有意义。"""
        assert report_paths, (
            f"在 {DASHBOARD_DATA_DIR} 下找不到任何 report-*.json，"
            "看板将无法展示任何历史报告"
        )

    def test_report_files_are_valid_json(self, report_paths: list[Path]) -> None:
        """任何一份报告 JSON 损坏都会让整个 ``/api/report/<name>`` 接口 500。"""
        assert report_paths, "前置条件：无报告可校验"
        bad: list[str] = []
        for p in report_paths:
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                bad.append(f"{p.name}: {e}")
        assert not bad, "下列报告 JSON 解析失败:\n" + "\n".join(bad)

    @pytest.mark.parametrize("report_path", _all_report_paths(), ids=lambda p: p.name)
    def test_cases_length_matches_case_count(self, report_path: Path) -> None:
        """看板列表页直接展示 ``case_count``，详情页用 ``cases.length``，
        两者不一致会让用户看到「通过率 50% 但列表只渲染 10 条」的诡异现象。"""
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert "cases" in data, f"{report_path.name} 缺少顶层 cases 字段"
        assert "case_count" in data, f"{report_path.name} 缺少顶层 case_count 字段"
        cases = data["cases"]
        assert isinstance(cases, list), (
            f"{report_path.name} cases 字段不是数组，实际是 {type(cases).__name__}"
        )
        case_count = data["case_count"]
        assert isinstance(case_count, int), (
            f"{report_path.name} case_count 不是整数，实际是 {type(case_count).__name__}"
        )
        assert len(cases) == case_count, (
            f"{report_path.name}: cases 数组长度 {len(cases)} 与 case_count "
            f"{case_count} 不一致"
        )


# ============================== 单条 case 字段完整性 ============================== #


class TestCaseFields:
    """每条 case 的字段必须齐全，缺一不可 —— 这是新看板渲染不崩的硬性保障。"""

    @pytest.mark.parametrize("report_path", _all_report_paths(), ids=lambda p: p.name)
    def test_every_case_has_required_fields(self, report_path: Path) -> None:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        cases = data.get("cases") or []
        problems: list[str] = []
        for idx, case in enumerate(cases):
            if not isinstance(case, dict):
                problems.append(f"#{idx} 不是对象，实际是 {type(case).__name__}")
                continue
            missing = REQUIRED_CASE_KEYS - case.keys()
            if missing:
                problems.append(f"#{idx}({case.get('case_id', '?')}) 缺字段: {sorted(missing)}")
        assert not problems, (
            f"{report_path.name} 共 {len(problems)} 条 case 不满足字段契约:\n"
            + "\n".join("  - " + p for p in problems)
        )

    @pytest.mark.parametrize("report_path", _all_report_paths(), ids=lambda p: p.name)
    def test_passed_is_bool_or_none(self, report_path: Path) -> None:
        """``passed`` 允许 ``True/False``（已判定）和 ``None``（被跳过），
        不允许其它类型 —— 前端做颜色判定时按真值判断，``0`` / ``""`` 会误判。"""
        data = json.loads(report_path.read_text(encoding="utf-8"))
        cases = data.get("cases") or []
        bad: list[tuple[int, object]] = []
        for idx, case in enumerate(cases):
            pv = case.get("passed")
            if not isinstance(pv, (bool, type(None))):
                bad.append((idx, pv))
        assert not bad, (
            f"{report_path.name} 共 {len(bad)} 条 case.passed 不是 bool/None: "
            + ", ".join(f"#{i}={v!r}" for i, v in bad[:5])
        )

    @pytest.mark.parametrize("report_path", _all_report_paths(), ids=lambda p: p.name)
    def test_metrics_is_list_and_each_metric_has_required_fields(
        self, report_path: Path
    ) -> None:
        """``metrics`` 必须是数组；数组里每个元素必须有 name/passed/score，
        否则看板「指标均值 / 单指标明细」视图直接读不到分数。"""
        data = json.loads(report_path.read_text(encoding="utf-8"))
        cases = data.get("cases") or []
        problems: list[str] = []
        for cidx, case in enumerate(cases):
            metrics = case.get("metrics")
            if not isinstance(metrics, list):
                problems.append(
                    f"case#{cidx}({case.get('case_id', '?')}) metrics 不是数组，"
                    f"实际是 {type(metrics).__name__}"
                )
                continue
            for midx, m in enumerate(metrics):
                if not isinstance(m, dict):
                    problems.append(
                        f"case#{cidx}({case.get('case_id', '?')}).metrics[{midx}] "
                        f"不是对象，实际是 {type(m).__name__}"
                    )
                    continue
                missing = REQUIRED_METRIC_KEYS - m.keys()
                if missing:
                    problems.append(
                        f"case#{cidx}({case.get('case_id', '?')}).metrics[{midx}] "
                        f"缺字段: {sorted(missing)}"
                    )
        assert not problems, (
            f"{report_path.name} metrics 字段不满足契约:\n"
            + "\n".join("  - " + p for p in problems)
        )


# ============================== 摘要一致性（兜底） ============================== #


class TestSummaryConsistency:
    """防御性检查：summary 里的 ``total`` 必须等于 ``case_count``。

    这一条不在看板渲染路径上，但能尽早暴露 runner 写报告时的统计错误，
    避免「看板列表显示 74 个用例，详情却只看到 70 条」的诡异 bug。
    """

    @pytest.mark.parametrize("report_path", _all_report_paths(), ids=lambda p: p.name)
    def test_summary_total_matches_case_count(self, report_path: Path) -> None:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        case_count = data.get("case_count")
        summary = data.get("summary") or []
        if not summary:
            pytest.skip(f"{report_path.name} 无 summary 段，跳过兜底校验")
        totals = [s.get("total") for s in summary if isinstance(s, dict)]
        # summary 是「每个被测模型一条」：单模型报告 case_count == total；
        # 多模型横向对比报告（同一批用例跑 N 个模型）case_count == 各模型 total 之和。
        assert case_count == sum(totals), (
            f"{report_path.name} summary[].total 合计 {sum(totals)} 与 case_count "
            f"{case_count} 不一致（各模型 total: {totals}）"
        )
