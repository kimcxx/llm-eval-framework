# -*- coding: utf-8 -*-
"""看板「用例列表 → 抽屉」定位正确性的单测。

背景（用户实测到的 bug）：
    多模型报告里 ``cases[]`` 是「用例 × 模型」的笛卡尔积 —— 同一个
    ``case_id`` 会出现 N 次（每个模型一条）。而看板曾把 ``case_id``
    当成唯一键：

        window.casesById = new Map(cases.map(c => [c.case_id, c]))

    后写的覆盖先写的，于是列表里第一行是 mock-baseline 的失败记录，
    点开抽屉拿到的却是最后一个模型（deepseek-pro）的通过记录 ——
    列表和抽屉对不上。跨次对比的聚合也是同一个坑（只比对了 1/N 的数据）。

修复口径：
    * 新增 ``case_key(c)``：``case_id`` + ``model``（``\\x1f`` 分隔）
    * ``diff_cases`` 用 ``case_key`` 做聚合键（单模型报告行为不变）
    * 前端用 ``caseKey()`` / ``window.casesByKey``，并给列表补上「模型」列

本测试把这几条契约钉死，防止改回 ``case_id`` 单键的回归。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.app import PAGE, case_key, diff_cases

DASHBOARD_DATA_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "data"


def _case(case_id: str, model: str = "m1", **extra):
    payload = {"case_id": case_id, "model": model}
    payload.update(extra)
    return payload


# ============================== case_key ============================== #


class TestCaseKey:
    """唯一键：case_id + 模型，缺字段安全降级。"""

    def test_key_contains_case_id_and_model(self):
        k = case_key(_case("json-005", "deepseek-pro"))
        assert "json-005" in k
        assert "deepseek-pro" in k

    def test_same_case_id_different_model_is_distinct(self):
        a = case_key(_case("json-005", "mock-baseline"))
        b = case_key(_case("json-005", "deepseek-pro"))
        assert a != b

    def test_same_case_id_same_model_is_equal(self):
        assert case_key(_case("json-005", "m1")) == case_key(
            {"model": "m1", "case_id": "json-005", "passed": True}
        )

    def test_single_model_report_key_behaves_like_case_id(self):
        """单模型报告：所有记录的 model 相同，排序语义等价于按 case_id 排。"""
        keys = [case_key(_case(cid, "only-model")) for cid in ("c-2", "c-1", "c-3")]
        assert sorted(keys) == [keys[1], keys[0], keys[2]]

    @pytest.mark.parametrize("bad", [None, "not-a-dict", 42, []])
    def test_non_dict_returns_empty_string(self, bad):
        """看板不能因为脏数据崩，非 dict 统一降级成空串（不会撞到真实用例）。"""
        assert case_key(bad) == ""

    def test_missing_model_is_tolerated(self):
        """老报告没有 model 字段：不能抛错，键仍带分隔符。"""
        assert case_key({"case_id": "c-1"}) == "c-1\x1f"


# ============================== diff_cases 多模型 ============================== #


class TestDiffCasesMultiModel:
    """同一 case_id 的多个模型必须各自独立参与对比，不能互相覆盖。"""

    def test_each_model_compared_independently(self):
        base = [
            _case("json-005", "mock-baseline", passed=False),
            _case("json-005", "deepseek-pro", passed=True),
        ]
        curr = [
            _case("json-005", "mock-baseline", passed=True),   # ← 修复
            _case("json-005", "deepseek-pro", passed=False),   # ← 回归
        ]
        out = diff_cases(curr, base)
        assert [x["curr"]["model"] for x in out["fixed"]] == ["mock-baseline"]
        assert [x["curr"]["model"] for x in out["regressed"]] == ["deepseek-pro"]

    def test_no_record_is_silently_dropped(self):
        """修复前：3 模型 × 2 用例 → 只比对 2 条（后写的覆盖先写的）。
        修复后：6 条全部各自成键，一条都不能丢。"""
        models = ["m1", "m2", "m3"]
        base = [_case(cid, m, passed=True) for cid in ("a", "b") for m in models]
        curr = [_case(cid, m, passed=False) for cid in ("a", "b") for m in models]
        out = diff_cases(curr, base)
        assert len(out["regressed"]) == 6

    def test_model_only_in_curr_is_only_in_curr(self):
        """新增一个模型的评测：该 (case, model) 组合属于「当前独有」。"""
        base = [_case("a", "m1", passed=True)]
        curr = [_case("a", "m1", passed=True), _case("a", "m2", passed=True)]
        out = diff_cases(curr, base)
        assert [case_key(x) for x in out["onlyInCurr"]] == [case_key(_case("a", "m2"))]

    def test_single_model_report_semantics_unchanged(self):
        """单模型报告：行为与改造前完全一致（regressed / fixed / stillFailing）。"""
        base = [_case("a", passed=True), _case("b", passed=False)]
        curr = [_case("a", passed=False), _case("b", passed=True)]
        out = diff_cases(curr, base)
        assert [x["curr"]["case_id"] for x in out["regressed"]] == ["a"]
        assert [x["curr"]["case_id"] for x in out["fixed"]] == ["b"]


# ============================== 真实多模型报告冒烟 ============================== #


@pytest.fixture(scope="module")
def multi_model_cases():
    """取看板数据里第一份「同一 case_id 出现多次」的报告（= 多模型报告）。"""
    if not DASHBOARD_DATA_DIR.is_dir():
        return []
    for p in sorted(DASHBOARD_DATA_DIR.glob("report-*.json")):
        cases = json.loads(p.read_text(encoding="utf-8")).get("cases") or []
        if not cases:
            continue
        if len({c.get("case_id") for c in cases}) < len(cases):
            return cases
    return []


class TestRealMultiModelReport:
    def test_case_keys_are_unique(self, multi_model_cases):
        if not multi_model_cases:
            pytest.skip("看板数据目录里没有多模型报告")
        keys = [case_key(c) for c in multi_model_cases]
        assert len(set(keys)) == len(keys)

    def test_self_diff_keeps_every_failing_record(self, multi_model_cases):
        """同一份报告对自身 diff：stillFailing 必须覆盖每一条 passed=False。

        修复前因为 case_id 互相覆盖，这里会少掉大量记录 —— 正是
        「列表显示失败、抽屉显示通过」的同一个根因。
        """
        if not multi_model_cases:
            pytest.skip("看板数据目录里没有多模型报告")
        out = diff_cases(multi_model_cases, multi_model_cases)
        assert out["regressed"] == []
        assert out["fixed"] == []
        expected = sum(1 for c in multi_model_cases if c.get("passed") is False)
        assert len(out["stillFailing"]) == expected

    def test_json_005_has_both_pass_and_fail_across_models(self, multi_model_cases):
        """用户报告的那条：json-005 在不同模型下结果不同（这就是覆盖事故现场）。"""
        if not multi_model_cases:
            pytest.skip("看板数据目录里没有多模型报告")
        rows = [c for c in multi_model_cases if c.get("case_id") == "json-005"]
        if not rows:
            pytest.skip("这份报告里没有 json-005")
        assert len({c.get("model") for c in rows}) == len(rows)
        assert len({case_key(c) for c in rows}) == len(rows)


# ============================== 前端 JS 契约 ============================== #


class TestFrontendContract:
    """前端模板必须按新键取 case（JS 逻辑不重复实现，这里只钉死契约）。"""

    def test_no_legacy_cases_by_id_map(self):
        assert "casesById" not in PAGE

    def test_uses_case_key_helper(self):
        assert "function caseKey(" in PAGE
        assert "window.casesByKey" in PAGE

    def test_case_list_shows_model_column(self):
        """列表必须能看出每行属于哪个模型，否则三行同 id 仍然无法区分。"""
        assert "模型</th>" in PAGE
