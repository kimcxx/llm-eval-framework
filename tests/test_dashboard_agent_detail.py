# -*- coding: utf-8 -*-
"""看板 #agent 页的逐题明细：报告必须自解释。

背景（验收时提的）：
    #agent 页原来只有 4 个汇总数字（任务成功率 / 工具选择正确率 / 平均步数 /
    陷阱防线）+ 一张历史表 —— 没有逐题明细、看不到各层断言过没过、看不到失败
    原因、看不到轨迹。后果是 **10/10 全绿和 0/10 全红提供的信息量几乎一样**，
    访问者一头雾水、发现不了任何问题。

设计原则（验收方定的，以后所有页面适用）：
    凡是需要阿灯在旁边解说才能看懂的页面，都是没做完的页面。

契约要点：
    * 后端给每次 run 注入 ``tasks``：考点文案（``tasks.json`` 的 design_note，
      **原样搬运不另编**）、各层断言、失败原因、判据修订留痕（meta.v2_changes
      按题拆分）、final_answer；
    * 轨迹只给最新一次（一份 30–60KB，全量注入会把 api/agent.json 撑到 MB 级），
      老结果在页面上降级显示「无轨迹」；
    * 页面逐题一行：考点 + 各层亮灯 + 「步数 实际/上限」+ repeat 稳定性；
    * 展开区：失败原因 / 失败层 / 判据修订留痕 / 逐步轨迹（思考 → 调用 → 观察）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.app import (
    AGENT_TASKS_FILE,
    PAGE,
    _load_task_notes,
    _split_task_notes,
    _trajectory_steps,
    load_agent_runs,
)


def _fn_body(name: str, until: str = "") -> str:
    body = PAGE[PAGE.index(name):]
    return body[: body.index(until)] if until and until in body else body[:4000]


# --------------------------------------------------------------------------- #
# 考点文案与判据修订留痕（tasks.json 只读，判据本体不动）
# --------------------------------------------------------------------------- #
class TestTaskNotes:
    """考点取自 tasks.json 的 design_note；v2 留痕按题拆，不整段贴。"""

    def test_design_note_taken_from_tasks_json(self) -> None:
        notes = _load_task_notes()
        assert notes, "读不到 tasks.json，考点文案无从谈起"
        raw = json.loads(AGENT_TASKS_FILE.read_text(encoding="utf-8"))
        first = raw["tasks"][0]
        assert notes[first["id"]]["design_note"] == first["design_note"], (
            "考点文案必须与 tasks.json 的 design_note 一字不差（原样搬运）"
        )

    def test_trap_task_has_note(self) -> None:
        """陷阱题（t07 诚实性）也得有考点，访问者才知道这题在考什么。"""
        notes = _load_task_notes()
        t07 = notes.get("t07-unknown-dimension-honesty")
        assert t07 and t07["design_note"], "陷阱题缺考点文案"

    def test_v2_changes_split_by_task(self) -> None:
        """整段 v2_changes 按题号拆开：提到哪题算哪题的留痕。"""
        meta = json.loads(AGENT_TASKS_FILE.read_text(encoding="utf-8"))["meta"]
        split = _split_task_notes(meta)
        assert "t08" in split and "放宽" in split["t08"], "t08 的判据放宽留痕没拆出来"
        assert "t06" in split and "上限" in split["t06"], "t06 的步数上限改动没拆出来"

    def test_global_reason_not_pinned_to_a_task(self) -> None:
        """「依据：……」是对整批改动的说明，挂到某一题上就答非所问。"""
        split = _split_task_notes({"v2_changes": "t06 步数上限 4→6。依据：v1 首跑两个 FAIL 均为测试设计问题。"})
        assert "t06" in split
        assert "依据" not in split["t06"], "全局说明不该算进某一题的留痕"

    def test_missing_meta_is_empty(self) -> None:
        assert _split_task_notes({}) == {}
        assert _split_task_notes({"v2_changes": ""}) == {}

    def test_unreadable_tasks_file_falls_back(self, tmp_path: Path, monkeypatch) -> None:
        """tasks.json 读不动（如部署沙盒里没有 agent_eval/）时降级为空，不许崩。"""
        monkeypatch.setattr("dashboard.app.AGENT_TASKS_FILE", tmp_path / "nope.json")
        assert _load_task_notes() == {}


# --------------------------------------------------------------------------- #
# 轨迹：逐步「思考 → 调用 → 观察」，缺了就缺了
# --------------------------------------------------------------------------- #
class TestTrajectorySteps:

    def test_no_path_returns_none(self) -> None:
        assert _trajectory_steps(None) is None
        assert _trajectory_steps("") is None

    def test_missing_file_returns_none(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("dashboard.app.ROOT", tmp_path)
        assert _trajectory_steps("agent_eval/trajectories/nope.json") is None

    def test_real_trajectory_has_calls_and_observation(self) -> None:
        runs = load_agent_runs()
        traj = [s for t in runs[0]["tasks"] for s in (t.get("trajectory") or [])]
        assert traj, "最新一次没有轨迹，页面就只剩汇总数字了"
        assert any(s["calls"] for s in traj), "轨迹里看不到工具调用"
        assert any(s["observation"] for s in traj), "轨迹里看不到观察结果"
        # 第 0 条是任务信息，不是步骤
        assert all(s["n"] is not None for s in traj)


# --------------------------------------------------------------------------- #
# 后端注入：每次 run 都有逐题明细，轨迹只给最新一次
# --------------------------------------------------------------------------- #
class TestAgentEntryTasks:

    @pytest.fixture(scope="class")
    def runs(self) -> list:
        return load_agent_runs()

    def test_every_run_has_tasks(self, runs) -> None:
        assert runs, "没有 agent 结果可测"
        for r in runs:
            assert r.get("tasks"), f"{r['file']} 缺逐题明细"

    def test_task_fields(self, runs) -> None:
        t = runs[0]["tasks"][0]
        for key in ("id", "level", "task", "status", "steps", "max_steps",
                    "layers", "failure_reasons", "final_answer", "design_note", "v2_note"):
            assert key in t, f"逐题明细缺字段 {key}"

    def test_layers_carried_through(self, runs) -> None:
        """四层断言逐题带过来，页面才能逐项亮灯。"""
        layers = runs[0]["tasks"][0]["layers"]
        assert layers.get("答案层") and "pass" in layers["答案层"]
        assert layers.get("工具层") and "pass" in layers["工具层"]
        assert layers.get("步骤效率") and "pass" in layers["步骤效率"]

    def test_only_latest_has_trajectory(self, runs) -> None:
        assert "trajectory" in runs[0]["tasks"][0], "最新一次应带轨迹"
        if len(runs) > 1:
            assert "trajectory" not in runs[1]["tasks"][0], "历史 run 不该注入轨迹（体积）"


# --------------------------------------------------------------------------- #
# 页面渲染：访问者不看解说也能回答「考什么 / 哪层没过 / 为什么挂」
# --------------------------------------------------------------------------- #
class TestPageRendering:

    def test_render_agent_draws_task_list(self) -> None:
        body = _fn_body("async function renderAgent", "// Agent 逐题明细")
        assert "agentTaskList(latest)" in body, "renderAgent 没渲染逐题明细"

    def test_four_layers_lit_individually(self) -> None:
        """四层断言逐项亮灯，不许只给一个总评。"""
        assert "const AGENT_LAYERS" in PAGE
        body = _fn_body("const AGENT_LAYERS", "function agentLayerChips")
        for layer in ("答案层", "工具层", "步骤效率", "工具健康"):
            assert layer in body, f"缺断言层 {layer}"

    def test_layer_chip_shows_pass_fail(self) -> None:
        body = _fn_body("function agentLayerChips", "function agentTrajStep")
        assert "l.pass ? 'pass' : 'fail'" in body, "断言层没有 ✓/✗ 亮灯"
        assert "'✓' : '✗'" in body or "? '✓' : '✗'" in body

    def test_row_shows_point_and_steps(self) -> None:
        body = _fn_body("function agentTaskRow", "function agentTaskList")
        assert "design_note" in body, "行里没有考点文案"
        assert "t.steps" in body and "t.max_steps" in body, "步数要显示「实际/上限」"

    def test_trap_level_highlighted(self) -> None:
        body = _fn_body("function agentTaskRow", "function agentTaskList")
        assert "agtrap" in body, "陷阱题考点没有醒目标记"

    def test_detail_has_failure_and_v2_note(self) -> None:
        body = _fn_body("function agentTaskDetail", "function agentTaskRow")
        assert "failure_reasons" in body, "展开区没有失败原因"
        assert "v2_note" in body, "展开区没有判据修订留痕"

    def test_missing_trajectory_degrades(self) -> None:
        body = _fn_body("function agentTaskDetail", "function agentTaskRow")
        assert "无轨迹" in body, "轨迹缺失时要有降级文案，不许留空白"

    def test_stability_column_degrades(self) -> None:
        """repeat=1 没有可比对象，显示「稳定性 —」而不是编一个一致。"""
        body = _fn_body("function agentTaskRow", "function agentTaskList")
        assert "稳定性 —" in body

    def test_narrative_keeps_agent_as_subject(self) -> None:
        """叙事红线：被测对象是 agent 系统本身，大脑只是可替换组件。"""
        body = _fn_body("function agentTaskList", "// 一份报告里参与")
        assert "agent 系统本身" in body, "页面得写清被测对象是 agent 系统，不是脑子的成绩"
