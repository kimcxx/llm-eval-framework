# -*- coding: utf-8 -*-
"""看板跨次对比 helper 的单元测试。

背景：
    新看板「跨次对比」标签的数据逻辑由 developer 在 ``dashboard/app.py``
    抽出的两个模块顶层函数负责：
        - ``case_status(c) -> 'pass' | 'fail' | 'skip'``
        - ``diff_cases(curr_cases, base_cases) -> {
              regressed, fixed, stillFailing, onlyInCurr, onlyInBase
          }``

    这些 helper 同时被看板的前端 JS 调用，必须保持「Python 与 JS
    行为完全一致」，所以本测试是看板正确性的硬性保障。

契约要点：
    * ``case_status``：error 优先 → 'skip'；passed 为 True → 'pass'；
      其余 → 'fail'。非 dict 安全降级为 'fail'。
    * ``diff_cases``：按 ``case_id`` 字符串排序后比较；返回 5 个分组，
      每条结果都是 ``{curr, base}`` 或单个 dict（onlyInCurr/onlyInBase）。
    * 允许 None / 空列表入参，绝不抛错。

注意：
    * 历史看板 JSON 里 ``passed`` 都是 ``bool``（参见
      ``tests/test_dashboard_history.py``），但 helper 文档允许
      ``passed=None``，因此补一组针对 ``passed=None`` 的用例，把
      「不抛错 + 不进入 regressed/fixed」这两条契约固定下来。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dashboard.app import case_status, diff_cases


# 看板数据目录（与 test_dashboard_history.py 保持一致）。
DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"


def _case(
    case_id: str,
    *,
    passed: bool | None = None,
    error: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """构造最小可用的 case dict，字段顺序无关。"""
    payload: dict[str, Any] = {"case_id": case_id}
    if passed is not None:
        payload["passed"] = passed
    if error is not None:
        payload["error"] = error
    payload.update(extra)
    return payload


# ============================== case_status ============================== #


class TestCaseStatus:
    """单条用例的状态判定 —— ``diff_cases`` 的基础原语。"""

    def test_passed_true_is_pass(self) -> None:
        assert case_status(_case("c-1", passed=True)) == "pass"

    def test_passed_false_is_fail(self) -> None:
        assert case_status(_case("c-1", passed=False)) == "fail"

    def test_passed_none_no_error_is_fail(self) -> None:
        """契约固定点：``passed=None`` 且 ``error`` 为空时判为 ``fail``。

        这是 helper 的当前行为 —— 与最初需求里「passed 为 None 应不参与
        对比」的描述不一致，但已被 developer 拍板：``error`` 才是
        「跳过」的信号，``passed=None`` 走 ``fail`` 分支。
        本测试防止后续重构悄悄改变这个语义。
        """
        assert case_status(_case("c-1", passed=None)) == "fail"

    def test_error_truthy_is_skip(self) -> None:
        """``error`` 非空即视为「跳过」，不管 ``passed`` 是什么。"""
        assert case_status(_case("c-1", error="调用超时")) == "skip"

    def test_error_overrides_passed_true(self) -> None:
        """``error`` 优先级高于 ``passed``，避免把崩溃的用例误标为通过。"""
        assert case_status(_case("c-1", passed=True, error="OOM")) == "skip"

    def test_empty_dict_is_fail(self) -> None:
        """缺字段安全降级：什么都不写的 dict 视为「未通过」。"""
        assert case_status({}) == "fail"

    def test_non_dict_is_fail(self) -> None:
        """异常输入不应让看板崩，必须降级到 ``fail``。"""
        assert case_status(None) == "fail"  # type: ignore[arg-type]
        assert case_status("not-a-dict") == "fail"  # type: ignore[arg-type]
        assert case_status(42) == "fail"  # type: ignore[arg-type]
        assert case_status(["c-1"]) == "fail"  # type: ignore[arg-type]

    def test_real_report_case_status_matches_passed_flag(
        self, real_cases: list[dict[str, Any]]
    ) -> None:
        """冒烟：用真实报告里的 case 验证 ``case_status`` 与 ``passed`` 字段
        100% 一致（真实数据里没有 ``passed=None`` 或 ``error`` 非空的情况）。"""
        for c in real_cases:
            status = case_status(c)
            if c.get("passed") is True:
                assert status == "pass", f"{c.get('case_id')} 应为 pass"
            else:
                assert status == "fail", f"{c.get('case_id')} 应为 fail"


# ============================== diff_cases: 入参容错 ============================== #


class TestDiffCasesEmpty:
    """空入参 / None 入参必须返回空分组，绝不抛错。"""

    def test_both_none(self) -> None:
        assert diff_cases(None, None) == {
            "regressed": [],
            "fixed": [],
            "stillFailing": [],
            "onlyInCurr": [],
            "onlyInBase": [],
        }

    def test_both_empty_lists(self) -> None:
        assert diff_cases([], []) == {
            "regressed": [],
            "fixed": [],
            "stillFailing": [],
            "onlyInCurr": [],
            "onlyInBase": [],
        }

    def test_none_and_list(self) -> None:
        """一边 None 一边 list 也必须能跑通。"""
        out = diff_cases(None, [_case("a", passed=True)])
        assert out["onlyInBase"] == [_case("a", passed=True)]


# ============================== diff_cases: 五种核心场景 ============================== #


class TestDiffCasesCore:
    """A→B 失败 / B→A 修复 / 双失败 / 单边独有 / 完全相同。"""

    def test_identical_passes_no_diff(self) -> None:
        """完全相同：通过 case 不变 → 不在 regressed / fixed。"""
        cases = [_case("a", passed=True), _case("b", passed=True)]
        out = diff_cases(cases, cases)
        assert out["regressed"] == []
        assert out["fixed"] == []
        assert out["stillFailing"] == []
        assert out["onlyInCurr"] == []
        assert out["onlyInBase"] == []

    def test_regressed_when_pass_to_fail(self) -> None:
        """A→B 失败：上次通过 → 这次失败 → 进 regressed。"""
        base = [_case("c-1", passed=True)]
        curr = [_case("c-1", passed=False)]
        out = diff_cases(curr, base)
        assert len(out["regressed"]) == 1
        assert out["regressed"][0] == {"curr": curr[0], "base": base[0]}
        assert out["fixed"] == []
        assert out["stillFailing"] == []

    def test_fixed_when_fail_to_pass(self) -> None:
        """B→A 修复：上次失败 → 这次通过 → 进 fixed。"""
        base = [_case("c-1", passed=False)]
        curr = [_case("c-1", passed=True)]
        out = diff_cases(curr, base)
        assert len(out["fixed"]) == 1
        assert out["fixed"][0] == {"curr": curr[0], "base": base[0]}
        assert out["regressed"] == []
        assert out["stillFailing"] == []

    def test_still_failing_when_both_fail(self) -> None:
        """双失败：两次都失败 → 进 stillFailing，不算 regressed / fixed。"""
        base = [_case("c-1", passed=False)]
        curr = [_case("c-1", passed=False)]
        out = diff_cases(curr, base)
        assert len(out["stillFailing"]) == 1
        assert out["stillFailing"][0] == {"curr": curr[0], "base": base[0]}
        assert out["regressed"] == []
        assert out["fixed"] == []

    def test_only_in_base(self) -> None:
        """只在 base 中有的 case_id → onlyInBase（不在三个状态桶里）。"""
        base = [_case("c-1", passed=True), _case("only-base", passed=True)]
        curr = [_case("c-1", passed=True)]
        out = diff_cases(curr, base)
        assert out["onlyInBase"] == [_case("only-base", passed=True)]
        assert out["regressed"] == []
        assert out["fixed"] == []
        assert out["stillFailing"] == []
        assert out["onlyInCurr"] == []

    def test_only_in_curr(self) -> None:
        """只在 curr 中有的 case_id → onlyInCurr（不在三个状态桶里）。"""
        base = [_case("c-1", passed=True)]
        curr = [_case("c-1", passed=True), _case("only-curr", passed=True)]
        out = diff_cases(curr, base)
        assert out["onlyInCurr"] == [_case("only-curr", passed=True)]
        assert out["regressed"] == []
        assert out["fixed"] == []
        assert out["stillFailing"] == []
        assert out["onlyInBase"] == []


# ============================== diff_cases: 排序与混合 ============================== #


class TestDiffCasesOrdering:
    """结果必须按 ``case_id`` 字符串排序，便于前端直接渲染。"""

    def test_results_sorted_by_case_id(self) -> None:
        """乱序输入，结果仍然按 case_id 排序。"""
        base = [
            _case("c-3", passed=True),
            _case("c-1", passed=True),
            _case("c-2", passed=True),
        ]
        curr = [
            _case("c-2", passed=False),  # regressed
            _case("c-3", passed=False),  # regressed
            _case("c-1", passed=True),   # 仍 pass，不进任何桶
        ]
        out = diff_cases(curr, base)
        assert [item["curr"]["case_id"] for item in out["regressed"]] == [
            "c-2",
            "c-3",
        ]

    def test_mixed_scenario(self) -> None:
        """综合：regressed + fixed + stillFailing + onlyInCurr + onlyInBase
        同时出现，验证五个分组互不串味。"""
        base = [
            _case("reg", passed=True),    # → regressed
            _case("fix", passed=False),   # → fixed
            _case("fail", passed=False),  # → stillFailing
            _case("only-base", passed=True),  # → onlyInBase
        ]
        curr = [
            _case("reg", passed=False),
            _case("fix", passed=True),
            _case("fail", passed=False),
            _case("only-curr", passed=True),  # → onlyInCurr
        ]
        out = diff_cases(curr, base)
        assert [x["curr"]["case_id"] for x in out["regressed"]] == ["reg"]
        assert [x["curr"]["case_id"] for x in out["fixed"]] == ["fix"]
        assert [x["curr"]["case_id"] for x in out["stillFailing"]] == ["fail"]
        assert [x["case_id"] for x in out["onlyInCurr"]] == ["only-curr"]
        assert [x["case_id"] for x in out["onlyInBase"]] == ["only-base"]


# ============================== diff_cases: 跳过语义 ============================== #


class TestDiffCasesSkip:
    """``error`` 触发的「跳过」也参与对比，但 ``passed=None`` 走 fail 分支。"""

    def test_skip_in_base_pass_in_curr_is_fixed(self) -> None:
        """上次 error 跳过 → 这次通过：算作修复。"""
        base = [_case("c-1", error="OOM")]
        curr = [_case("c-1", passed=True)]
        out = diff_cases(curr, base)
        assert len(out["fixed"]) == 1
        assert out["regressed"] == []
        assert out["stillFailing"] == []

    def test_pass_in_base_skip_in_curr_is_regressed(self) -> None:
        """上次通过 → 这次 error 跳过：算作回归（避免静默吞掉线上崩溃）。"""
        base = [_case("c-1", passed=True)]
        curr = [_case("c-1", error="OOM")]
        out = diff_cases(curr, base)
        assert len(out["regressed"]) == 1
        assert out["fixed"] == []
        assert out["stillFailing"] == []

    def test_both_skipped_is_still_failing(self) -> None:
        """两次都被跳过（error）→ stillFailing，不当 fixed。"""
        base = [_case("c-1", error="OOM")]
        curr = [_case("c-1", error="网络超时")]
        out = diff_cases(curr, base)
        assert len(out["stillFailing"]) == 1
        assert out["regressed"] == []
        assert out["fixed"] == []

    def test_passed_none_does_not_crash(self) -> None:
        """``passed=None`` 必须不抛错（dashboard 是给用户用的，不能 500）。"""
        base = [_case("c-1", passed=None)]
        curr = [_case("c-1", passed=None)]
        out = diff_cases(curr, base)
        # 当前契约：passed=None（无 error）走 fail 分支 → stillFailing
        assert len(out["stillFailing"]) == 1
        assert out["regressed"] == []
        assert out["fixed"] == []

    def test_passed_none_in_curr_pass_in_base_is_regressed(self) -> None:
        """契约固定点：``passed=None``（无 error）= fail，所以
        base pass → curr passed=None 也算回归。
        与「passed=None 应不参与对比」的初版描述不同，本测试钉死当前行为。
        """
        base = [_case("c-1", passed=True)]
        curr = [_case("c-1", passed=None)]
        out = diff_cases(curr, base)
        assert len(out["regressed"]) == 1
        assert out["fixed"] == []


# ============================== diff_cases: 真实报告冒烟 ============================== #


class TestDiffCasesRealReport:
    """用一份真实报告当 fixture，验证 ``diff_cases`` 在看板数据上能跑通。"""

    def test_diff_same_report_against_itself_keeps_failing(
        self, real_cases: list[dict[str, Any]]
    ) -> None:
        """同一份报告对自身 diff：所有 case 两端状态一致。

        注意 ``stillFailing`` **不会**为空 —— 真实报告里大量 ``passed=False``
        的用例，两端都判为 ``fail``，按契约必须落进 ``stillFailing``。
        反过来说，这才是 helper 在防误报：两端相同就老老实实放 stillFailing，
        不会冒出来一堆假的 regressed / fixed。
        """
        out = diff_cases(real_cases, real_cases)
        assert out["regressed"] == []
        assert out["fixed"] == []
        assert out["onlyInCurr"] == []
        assert out["onlyInBase"] == []
        # stillFailing 的条目数 == 原报告中 passed=False 的 case 数
        expected_still = sum(1 for c in real_cases if c.get("passed") is False)
        assert len(out["stillFailing"]) == expected_still

    def test_real_report_with_constructed_regression(
        self, real_cases: list[dict[str, Any]]
    ) -> None:
        """构造「上次 pass → 这次 fail」的反事实 diff：
        以原报告为 base（上次），把 curr 里所有 ``passed=True`` 的 case
        翻成 ``passed=False``，其余保持不变。

        期望：原报告中所有原本通过（``passed=True``）的 case 都出现在
        ``regressed`` 里；``fixed`` / ``onlyInCurr`` / ``onlyInBase`` 均为空。
        """
        if not real_cases:
            pytest.skip("看板数据目录里没有可用的真实报告")
        base: list[dict[str, Any]] = [dict(c) for c in real_cases]  # 上次
        curr: list[dict[str, Any]] = []
        for c in real_cases:
            if c.get("passed") is True:
                # curr 里这次挂掉了
                bad = dict(c)
                bad["passed"] = False
                curr.append(bad)
            else:
                curr.append(dict(c))
        out = diff_cases(curr, base)
        expected_regressed = sum(1 for c in real_cases if c.get("passed") is True)
        assert len(out["regressed"]) == expected_regressed
        assert out["fixed"] == []
        assert out["onlyInCurr"] == []
        assert out["onlyInBase"] == []

    def test_real_report_constructed_fix(
        self, real_cases: list[dict[str, Any]]
    ) -> None:
        """构造「上次全挂 → 这次全过」的反事实 diff：每个原 fail 的用例都该
        出现在 ``fixed`` 里。"""
        if not real_cases:
            pytest.skip("看板数据目录里没有可用的真实报告")
        base: list[dict[str, Any]] = []
        curr: list[dict[str, Any]] = []
        for c in real_cases:
            base.append(dict(c))
            if c.get("passed") is False:
                # base 里是 fail，把 curr 改成通过
                good = dict(c)
                good["passed"] = True
                curr.append(good)
            else:
                curr.append(dict(c))
        out = diff_cases(curr, base)
        expected_fixed = sum(1 for c in real_cases if c.get("passed") is False)
        assert len(out["fixed"]) == expected_fixed
        assert out["regressed"] == []
        assert out["onlyInCurr"] == []
        assert out["onlyInBase"] == []


# ============================== fixture ============================== #


@pytest.fixture(scope="module")
def real_cases() -> list[dict[str, Any]]:
    """加载看板数据目录下第一份 ``report-*.json`` 的 ``cases`` 数组。

    用于 ``TestCaseStatus`` / ``TestDiffCasesRealReport`` 的冒烟。
    选「第一份」而不是随机选是为了让结果可复现。
    """
    if not DASHBOARD_DATA_DIR.is_dir():
        return []
    for p in sorted(DASHBOARD_DATA_DIR.glob("report-*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        cases = data.get("cases") or []
        if cases:
            return cases
    return []
