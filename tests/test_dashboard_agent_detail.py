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


# agent_eval/results/ 与 agent_eval/trajectories/ **不入库**（.gitignore），CI 沙盒里一份都
# 没有。所以依赖本机产物的用例必须能跳过，逻辑本身由「自造数据」的用例覆盖——
# 第一版就是因为断言了真实轨迹，本地 949 passed 而 GitHub CI fail。
LOCAL_RESULTS = sorted(Path("agent_eval/results").rglob("result-*.json"))
LOCAL_TRAJECTORIES = sorted(Path("agent_eval/trajectories").glob("*.json"))


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

    def test_steps_extracted_from_logs(self, tmp_path: Path, monkeypatch) -> None:
        """逐步抽取「思考 → 调用 → 观察」，用**自己造的**轨迹验证。

        测试不能依赖本机跑出来的产物：``agent_eval/trajectories/`` 不入库，CI 沙盒里
        一份都没有（第一版就是这么挂在 CI 上的：本地 949 passed，GitHub 上 fail）。
        """
        traj = {"logs": [
            {"task": "题目"},  # 第 0 条是任务信息，不是步骤
            {"step_number": 1,
             "model_output_message": {"content": "先查安全维度"},
             "tool_calls": [{"function": {"name": "get_eval_result",
                                          "arguments": {"model": "deepseek-chat", "dimension": "安全"}}}],
             "observations": "86.7%", "error": None, "is_final_answer": False},
            {"step_number": 2, "model_output_message": {"content": ""}, "tool_calls": [],
             "observations": "86.7%（26/30）", "error": None, "is_final_answer": True},
        ]}
        (tmp_path / "tj.json").write_text(json.dumps(traj, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("dashboard.app.ROOT", tmp_path)
        steps = _trajectory_steps("tj.json")
        assert steps is not None and len(steps) == 2, "logs[0] 是任务信息，不该算进步骤"
        assert steps[0]["calls"][0]["name"] == "get_eval_result", "看不到工具调用"
        assert steps[0]["calls"][0]["args"] == {"model": "deepseek-chat", "dimension": "安全"}
        assert steps[0]["observation"] == "86.7%", "看不到观察结果"
        assert steps[0]["thought"] == "先查安全维度"
        assert steps[0]["final"] is False and steps[1]["final"] is True

    def test_tool_error_extracted(self, tmp_path: Path, monkeypatch) -> None:
        """工具报错要能单独显示——那是「工具层失败」的证据。"""
        traj = {"logs": [{"step_number": 1, "model_output_message": {"content": "查一下"},
                          "tool_calls": [{"function": {"name": "get_eval_result", "arguments": {}}}],
                          "observations": None, "error": "ValueError: 模型名不存在",
                          "is_final_answer": False}]}
        (tmp_path / "tj.json").write_text(json.dumps(traj, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("dashboard.app.ROOT", tmp_path)
        steps = _trajectory_steps("tj.json")
        assert steps[0]["error"] == "ValueError: 模型名不存在"

    @pytest.mark.skipif(not LOCAL_TRAJECTORIES,
                        reason="轨迹文件不入库，CI 沙盒里没有（测试不得依赖本机产物）")
    def test_real_trajectory_has_calls_and_observation(self) -> None:
        runs = load_agent_runs()
        traj = [s for t in runs[0]["tasks"] for s in (t.get("trajectory") or [])]
        assert traj, "最新一次没有轨迹，页面就只剩汇总数字了"
        assert any(s["calls"] for s in traj), "轨迹里看不到工具调用"
        assert any(s["observation"] for s in traj), "轨迹里看不到观察结果"
        # 第 0 条是任务信息，不是步骤
        assert all(s["n"] is not None for s in traj)


# --------------------------------------------------------------------------- #
# 后端注入（自造数据，CI 与本地都跑得到）
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_env(tmp_path: Path, monkeypatch):
    """自造一套 agent 结果环境：两份结果 + 一份轨迹 + 出题文件。

    结果文件与轨迹都不入库，CI 沙盒里没有；**逻辑用例必须自给自足**，不能指着
    本机跑出来的产物（否则就是「本地绿、GitHub 红」）。
    """
    root = tmp_path
    for sub in ("results", "results/archive", "trajectories"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    (root / "trajectories" / "tj.json").write_text(json.dumps({"logs": [
        {"task": "题目"},
        {"step_number": 1, "model_output_message": {"content": "查安全维度"},
         "tool_calls": [{"function": {"name": "get_eval_result",
                                      "arguments": {"model": "deepseek-chat", "dimension": "安全"}}}],
         "observations": "86.7%", "error": None, "is_final_answer": False},
        {"step_number": 2, "model_output_message": {"content": ""}, "tool_calls": [],
         "observations": "86.7%（26/30）", "error": None, "is_final_answer": True},
    ]}, ensure_ascii=False), encoding="utf-8")

    def _result(status, steps):
        ok = status == "通过"
        return {
            "model": "deepseek-chat", "repeat": 1,
            "summary": {"任务总数": 1, "通过": int(ok), "任务成功率": 1.0 if ok else 0.0},
            "results": [{
                "id": "t06-pure-calc-trap", "level": "陷阱·纯计算",
                "task": "不调工具直接算/猜数，应该被判失败",
                "status": status, "steps": steps, "max_steps": 4,
                "layers": {"有效性": {"pass": True, "detail": "有效"},
                           "答案层": {"pass": ok, "detail": "金标命中" if ok else "缺少金标数字"},
                           "工具层": {"pass": True, "detail": "命中"},
                           "工具健康": {"pass": True, "detail": "无报错"},
                           "步骤效率": {"pass": True, "detail": "2 步（上限 4）"}},
                "failure_reasons": [] if ok else ["答案层：缺少「83.3」"],
                "final_answer": "86.7%", "trajectory_path": "trajectories/tj.json",
            }],
        }

    # 时间戳在文件名里（_agent_stamp 取的就是它），决定谁排最前
    (root / "results" / "result-20261006-120000.json").write_text(
        json.dumps(_result("通过", 2), ensure_ascii=False), encoding="utf-8")
    (root / "results" / "result-20261005-120000.json").write_text(
        json.dumps(_result("失败", 2), ensure_ascii=False), encoding="utf-8")

    (root / "tasks.json").write_text(json.dumps({
        "meta": {"v2_changes": "t06 步数上限 4→6。依据：v1 首跑（8/10）两个 FAIL 均为测试设计问题。"},
        "tasks": [{"id": "t06-pure-calc-trap", "level": "陷阱·纯计算",
                   "design_note": "不调工具直接算/猜数字，应该被判失败"}],
    }, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr("dashboard.app.AGENT_RESULTS_DIR", root / "results")
    monkeypatch.setattr("dashboard.app.AGENT_ARCHIVE_DIR", root / "results" / "archive")
    monkeypatch.setattr("dashboard.app.AGENT_TASKS_FILE", root / "tasks.json")
    monkeypatch.setattr("dashboard.app.ROOT", root)
    return root


class TestAgentEntryInjection:

    def test_latest_first_and_has_trajectory(self, fake_env) -> None:
        runs = load_agent_runs()
        assert len(runs) == 2
        assert runs[0]["file"] == "result-20261006-120000.json", "最新一次应排最前"
        t = runs[0]["tasks"][0]
        assert len(t["trajectory"]) == 2, "最新一次要带逐步轨迹"
        assert t["trajectory"][0]["calls"][0]["name"] == "get_eval_result"

    def test_history_has_no_trajectory(self, fake_env) -> None:
        """轨迹只给最新一次（体积考虑），历史行不该有这个键。"""
        runs = load_agent_runs()
        assert "trajectory" not in runs[1]["tasks"][0]

    def test_design_note_from_tasks_json(self, fake_env) -> None:
        t = load_agent_runs()[0]["tasks"][0]
        assert t["design_note"] == "不调工具直接算/猜数字，应该被判失败", "考点要原样来自出题文件"

    def test_v2_note_split_and_cleaned(self, fake_env) -> None:
        """留痕按题拆，且不能把全局「依据」说明挂到题上。"""
        t = load_agent_runs()[1]["tasks"][0]
        assert "步数上限" in t["v2_note"], "t06 的处置留痕没落到本题"
        assert "依据" not in t["v2_note"], "全局说明不该算进某一题"

    def test_failure_reasons_carried(self, fake_env) -> None:
        t = load_agent_runs()[1]["tasks"][0]
        assert t["failure_reasons"] == ["答案层：缺少「83.3」"]
        assert t["layers"]["答案层"]["pass"] is False
        assert t["steps"] == 2 and t["max_steps"] == 4


# --------------------------------------------------------------------------- #
# 后端注入：真实结果（本机产物，CI 里跳过）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not LOCAL_RESULTS, reason="结果文件不入库，CI 沙盒里没有（逻辑由自造数据用例覆盖）")
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
        assert "agentTaskList(latest, ds)" in body, "renderAgent 没渲染逐题明细"

    def test_agent_page_declares_data_source(self) -> None:
        """⑤ 页面透明化：数据源必须写在页面上——它是判分的前提。

        藏起来的后果已经发生过一次：金标停在 9/23、工具读 10-04，10 道题里 4 道
        变成没人看得懂的 fail。
        """
        body = _fn_body("function agentTaskList", "\nfunction ")
        assert "data_source" in _fn_body("async function renderAgent", "// Agent 逐题明细"), \
            "renderAgent 要把后端给的数据源传给明细区"
        assert "agent 评测专用快照" in body, "要写明数据来自冻结的快照，不是「最新报告」"
        assert "ds.source_report" in body and "href=\"#" in body, \
            "要给源报告在 #llm 详情页的链接，访问者才能自己核"

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
