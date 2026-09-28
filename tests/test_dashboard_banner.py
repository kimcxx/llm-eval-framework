# -*- coding: utf-8 -*-
"""看板「首页顶部 banner」字段计算 helper 的单元测试。

背景：
    首页最顶部加一行大字结论：「最新报告 deepseek-chat 82.4%，较上次 +x.x pt；
    回归 n 条 / 修复 m 条；最弱维度：安全 xx%」。数据全部来自 ``api/reports.json``
    与已有的跨次对比 ``diff_cases``，不开新接口。

契约要点：
    * ``_pick_representative_model``：优先返回非 ``mock-baseline`` 的首个模型；
      单模型 / 残缺数据时降级到 ``entry['model']``；空 entry 返回 ``'?'``。
    * ``_representative_rate``：按模型在 ``model_rows`` 中取 ``pass_rate``；
      找不到时降级到 ``entry['pass_rate']``。
    * ``_weakest_dimension``：取最弱的能力维度，兼容 ``dict[model] -> list`` 与
      顶层 ``list``；``untagged`` 不参与最弱选择（用户看不到意义），但全
      ``untagged`` 时仍返回自身（让前端可显示「未标注」而不是 "—"）。
      **口径唯一**：它复用 ``_dimension_summary`` 的结果，不直接读报告里存的
      ``dimensions`` 段（那是生成时的快照，改口径后会过期）。
    * ``compute_banner``：组合以上 + ``diff_cases``；
      - ``base`` 为 ``None`` 或同模型缺席 → ``delta_pt=None``（前端显示 "—"）
      - ``base_doc / curr_doc`` 任一缺失 → ``regressed=0, fixed=0``（不崩）
      - 数值精度：``curr=0.824, base=0.78`` → ``delta_pt=4.4``（百分点）

注意：
    * 本测试只覆盖字段计算（数据契约），不测渲染输出；banner 的 HTML / CSS
      在 ``dashboard/app.py`` 里 ``renderBanner`` / ``.banner`` 一并维护。
"""
from __future__ import annotations

import pytest

from dashboard.app import (
    _pick_representative_model,
    _representative_rate,
    _weakest_dimension,
    compute_banner,
)


# ============================== _pick_representative_model ============================== #


class TestPickRepresentativeModel:
    """从 model_rows 选 banner 代表模型。"""

    def test_picks_first_non_mock_in_multi_model(self) -> None:
        """多模型报告：mock + real → 选 real（banner 不应被对照组撑场）。"""
        entry = {
            "model": "mock-baseline",
            "model_rows": [
                {"model": "mock-baseline", "pass_rate": 1.0},
                {"model": "deepseek-chat", "pass_rate": 0.824},
                {"model": "deepseek-pro", "pass_rate": 0.842},
            ],
        }
        assert _pick_representative_model(entry) == "deepseek-chat"

    def test_picks_first_when_all_mock(self) -> None:
        """全是 mock：返回第一个 mock（罕见，但契约要钉死）。"""
        entry = {
            "model": "mock-baseline",
            "model_rows": [
                {"model": "mock-baseline", "pass_rate": 1.0},
                {"model": "mock-alt", "pass_rate": 0.9},
            ],
        }
        assert _pick_representative_model(entry) == "mock-baseline"

    def test_falls_back_to_entry_model_when_no_model_rows(self) -> None:
        """单模型报告（旧数据，无 model_rows）：退到 entry['model']。"""
        entry = {"model": "solo", "pass_rate": 0.5}
        assert _pick_representative_model(entry) == "solo"

    def test_returns_question_mark_when_empty(self) -> None:
        """空 entry / 无模型：返回 '?'，前端 esc 安全。"""
        assert _pick_representative_model({}) == "?"
        assert _pick_representative_model({"model_rows": []}) == "?"
        assert _pick_representative_model({"model_rows": None}) == "?"


# ============================== _representative_rate ============================== #


class TestRepresentativeRate:
    """取代表模型在 entry 里的 pass_rate。"""

    def test_finds_in_model_rows(self) -> None:
        entry = {
            "model_rows": [
                {"model": "a", "pass_rate": 0.9},
                {"model": "b", "pass_rate": 0.7},
            ],
            "pass_rate": 0.0,  # 必须忽略
        }
        assert _representative_rate(entry, "b") == 0.7

    def test_falls_back_to_entry_pass_rate(self) -> None:
        """model_rows 里找不到模型时退到 entry.pass_rate。"""
        entry = {"pass_rate": 0.42, "model_rows": [{"model": "x", "pass_rate": 0.9}]}
        assert _representative_rate(entry, "nonexistent") == 0.42


# ============================== _weakest_dimension ============================== #


class TestWeakestDimension:
    """挑最弱的能力维度（label, rate）。

    口径唯一：``_weakest_dimension`` 是 ``_dimension_summary`` 的消费者——
    categories 段按「分类 → 维度」重算优先，报告里存的 dimensions 段只当兜底。
    """

    def test_dict_by_model_shape(self) -> None:
        """多模型报告：``{'model': [...]}`` 结构。"""
        doc = {
            "dimensions": {
                "deepseek-chat": [
                    {"dimension": "safety", "pass_rate": 0.33},
                    {"dimension": "correctness", "pass_rate": 0.85},
                ],
                "mock-baseline": [{"dimension": "safety", "pass_rate": 1.0}],
            }
        }
        assert _weakest_dimension(doc, "deepseek-chat") == ("安全", 0.33)

    def test_top_level_list_shape(self) -> None:
        """单模型报告：顶层 ``[...]`` 结构。

        维度名写的是旧名 ``format``，归一成「指令遵循」——banner 与首页维度速览
        必须叫同一个名字。
        """
        doc = {"dimensions": [{"dimension": "format", "pass_rate": 0.5}]}
        assert _weakest_dimension(doc, "any") == ("指令遵循", 0.5)

    def test_fallback_to_any_list_when_model_missing(self) -> None:
        """dict 里没目标模型，但有别的模型维度 → fallback 取首份 list。"""
        doc = {
            "dimensions": {
                "mock-baseline": [{"dimension": "safety", "pass_rate": 1.0}],
            }
        }
        assert _weakest_dimension(doc, "deepseek-chat") == ("安全", 1.0)

    def test_fallback_skips_mock_control_group(self) -> None:
        """目标模型缺席：跳过 mock-* 对照组，取真实模型的维度。

        mock-baseline 是阴性对照，它的 10% 不是被测模型的能力，不该出现在 banner。
        """
        doc = {
            "dimensions": {
                "mock-baseline": [{"dimension": "safety", "pass_rate": 0.1}],
                "deepseek-pro": [{"dimension": "safety", "pass_rate": 0.83}],
            }
        }
        assert _weakest_dimension(doc, "deepseek-chat") == ("安全", 0.83)

    def test_fallback_keeps_mock_when_only_mock(self) -> None:
        """全是 mock 的报告（CD 冒烟）：退回对照组自身，而不是显示 "—"。"""
        doc = {"dimensions": {"mock-baseline": [{"dimension": "format", "pass_rate": 0.25}]}}
        assert _weakest_dimension(doc, "deepseek-chat") == ("指令遵循", 0.25)

    def test_categories_fallback_skips_mock_control_group(self) -> None:
        """categories 兜底路径同样跳过 mock-*（老报告没有 dimensions 段）。

        走的是「分类 → 维度」重算，所以露出的是维度名（指令遵循）而不是分类名
        （json_extract）——banner 与首页速览必须叫同一个名字。
        """
        doc = {
            "categories": {
                "mock-baseline": [
                    {"category": "json_extract", "total": 18, "passed": 0, "pass_rate": 0.0}
                ],
                "deepseek-pro": [
                    {"category": "json_extract", "total": 18, "passed": 13, "pass_rate": 0.72}
                ],
            }
        }
        label, rate = _weakest_dimension(doc, "deepseek-chat")
        assert label == "指令遵循"
        assert rate == pytest.approx(13 / 18)

    def test_recomputes_categories_instead_of_trusting_stored_dimensions(self) -> None:
        """本报告存的 dimensions 段是**生成时的旧口径**，不能被 banner 直接采信。

        场景：报告里有 categories（math 20 题 + qa_open 8 题），而 dimensions 段
        还是「correctness 50 题」的旧聚合（qa_open 被并进准确性）。重算后
        qa_open 属于相关性，准确性只剩 20 题——banner 必须按新口径走。
        """
        doc = {
            "categories": {"m": [
                {"category": "math_reasoning", "total": 20, "passed": 20, "pass_rate": 1.0},
                {"category": "qa_open", "total": 8, "passed": 2, "pass_rate": 0.25},
            ]},
            # 过期快照：旧口径把 qa_open 算进了 correctness
            "dimensions": {"m": [
                {"dimension": "correctness", "total": 50, "passed": 30, "pass_rate": 0.6},
            ]},
        }
        assert _weakest_dimension(doc, "m") == ("相关性", 0.25)

    def test_returns_none_when_no_dimensions(self) -> None:
        """dimensions 段缺失 → None，前端显示 "—"。"""
        assert _weakest_dimension({}, "m") is None
        assert _weakest_dimension({"dimensions": None}, "m") is None
        assert _weakest_dimension({"dimensions": {}}, "m") is None

    def test_returns_none_when_empty_list(self) -> None:
        doc = {"dimensions": {"m": []}}
        assert _weakest_dimension(doc, "m") is None

    def test_skips_untagged(self) -> None:
        """untagged 不参与最弱选择（用户看不到意义）。"""
        doc = {"dimensions": [
            {"dimension": "untagged", "pass_rate": 0.0},
            {"dimension": "safety", "pass_rate": 0.4},
        ]}
        assert _weakest_dimension(doc, "m") == ("安全", 0.4)

    def test_falls_back_when_all_untagged(self) -> None:
        """全 untagged：返回 untagged 本身（不是 None），前端可显示「未标注」。"""
        doc = {"dimensions": [{"dimension": "untagged", "pass_rate": 0.0}]}
        assert _weakest_dimension(doc, "m") == ("未标注", 0.0)


# ============================== compute_banner ============================== #


class TestComputeBanner:
    """整合字段计算：端到端契约。"""

    @staticmethod
    def _entry(model_rows):
        return {
            "model": (model_rows[0]["model"] if model_rows else "?"),
            "model_rows": model_rows,
            "pass_rate": (model_rows[0]["pass_rate"] if model_rows else 0.0),
        }

    def test_full_scenario_with_same_model_in_base(self) -> None:
        """完整场景：3 模型，curr 82.4% / base 78.0% → +4.4 pt；1 回归 / 1 修复。"""
        curr = self._entry([
            {"model": "mock-baseline", "pass_rate": 1.0},
            {"model": "deepseek-chat", "pass_rate": 0.824},
            {"model": "deepseek-pro", "pass_rate": 0.842},
        ])
        base = self._entry([
            {"model": "mock-baseline", "pass_rate": 1.0},
            {"model": "deepseek-chat", "pass_rate": 0.780},
            {"model": "deepseek-pro", "pass_rate": 0.812},
        ])
        curr_doc = {
            "cases": [
                {"case_id": "c1", "passed": True},
                {"case_id": "c2", "passed": True},
                {"case_id": "c3", "passed": False},
            ],
            "dimensions": {"deepseek-chat": [
                {"dimension": "safety", "pass_rate": 0.33},
                {"dimension": "correctness", "pass_rate": 0.9},
            ]},
        }
        base_doc = {
            "cases": [
                {"case_id": "c1", "passed": True},
                {"case_id": "c2", "passed": False},  # → fixed
                {"case_id": "c3", "passed": True},   # → regressed
            ],
        }
        out = compute_banner(curr, curr_doc, base, base_doc)
        assert out["model"] == "deepseek-chat"
        assert out["curr_rate"] == pytest.approx(0.824)
        assert out["base_rate"] == pytest.approx(0.780)
        assert out["delta_pt"] == pytest.approx(4.4)
        assert out["regressed"] == 1
        assert out["fixed"] == 1
        assert out["weakest_label"] == "安全"
        assert out["weakest_rate"] == pytest.approx(0.33)
        assert out["has_base"] is True

    def test_negative_delta_when_regression(self) -> None:
        """负向：curr < base → delta_pt 为负。"""
        curr = self._entry([{"model": "x", "pass_rate": 0.7}])
        base = self._entry([{"model": "x", "pass_rate": 0.8}])
        out = compute_banner(curr, {}, base, {})
        assert out["delta_pt"] == pytest.approx(-10.0)

    def test_no_base_means_delta_none(self) -> None:
        """首份报告 / 没上次 → delta_pt=None（前端显示 "—"），不显示 0。"""
        curr = self._entry([{"model": "deepseek-chat", "pass_rate": 0.82}])
        out = compute_banner(curr, {"cases": [], "dimensions": []})
        assert out["has_base"] is False
        assert out["delta_pt"] is None
        assert out["regressed"] == 0
        assert out["fixed"] == 0

    def test_base_missing_same_model_means_delta_none(self) -> None:
        """上次报告里没代表模型（首次跑这个模型）：delta_pt=None，
        不把"完全不同口径"的对照展示给用户。"""
        curr = self._entry([{"model": "deepseek-chat", "pass_rate": 0.82}])
        base = self._entry([{"model": "mock-baseline", "pass_rate": 1.0}])
        out = compute_banner(curr, {}, base, {})
        assert out["has_base"] is True
        assert out["delta_pt"] is None

    def test_no_dimensions_returns_none_weakest(self) -> None:
        """dimensions 段缺失：weakest=None，前端显示 "—"。"""
        curr = self._entry([{"model": "x", "pass_rate": 0.8}])
        out = compute_banner(curr, {"cases": []})
        assert out["weakest_label"] is None
        assert out["weakest_rate"] is None

    def test_diff_count_regressed_only(self) -> None:
        """diff_cases 计数：regressed / fixed 直接来自 diff_cases。"""
        curr = self._entry([{"model": "x", "pass_rate": 0.5}])
        base = self._entry([{"model": "x", "pass_rate": 1.0}])
        curr_doc = {"cases": [{"case_id": "c", "passed": False}]}  # regressed
        base_doc = {"cases": [{"case_id": "c", "passed": True}]}
        out = compute_banner(curr, curr_doc, base, base_doc)
        assert out["regressed"] == 1
        assert out["fixed"] == 0
        assert out["delta_pt"] == pytest.approx(-50.0)

    def test_diff_count_zero_when_docs_missing(self) -> None:
        """curr_doc 或 base_doc 缺失 → 不进 diff，回归/修复都 0，但 delta 仍可算。"""
        curr = self._entry([{"model": "x", "pass_rate": 0.5}])
        out = compute_banner(curr, None, curr, None)
        assert out["regressed"] == 0
        assert out["fixed"] == 0
        assert out["base_rate"] == pytest.approx(0.5)
        assert out["delta_pt"] == pytest.approx(0.0)