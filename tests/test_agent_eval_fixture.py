"""Agent 评测的数据源：专用 fixture + 双模开关 + 金标护栏。

守的是 2026-10-06 那次事故：取数工具「读 reports/ 下最新报告」，10-04 新全量回归
落地后数字漂了，而 tasks.json 的金标还停在 9/23，10 道题里 4 道变成莫名其妙的 fail。
数据源 fixture 化之后，**金标与数据再漂移时这里直接红**，不靠人发现。

不依赖本机跑出来的产物（``reports/`` 不入库）：只有「与源报告逐字节比对」那条需要
源报告，已加 skipif——CI 沙盒里会跳过，逻辑本身由 fixture（入库）覆盖。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("smolagents", reason="agent_eval 依赖 smolagents（requirements 里有）")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_eval import tools  # noqa: E402

TASKS_FILE = PROJECT_ROOT / "agent_eval" / "tasks.json"
FIXTURE = PROJECT_ROOT / "agent_eval" / "fixtures" / "eval_data_fixture.json"
SOURCE_REPORT = PROJECT_ROOT / "reports" / "report-full-regression-20260923-143903.json"

MODELS = ("deepseek-chat", "deepseek-pro")
DIMS = ("准确性", "指令遵循", "安全", "相关性")


def _fixture_rows() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["data"]


def _numbers_in(values) -> set[float]:
    """从金标字符串里抽出数字（「1 道」→ 1，「83.3」→ 83.3）。"""
    out: set[float] = set()
    for value in values or []:
        for hit in re.findall(r"\d+(?:\.\d+)?", str(value)):
            out.add(float(hit))
    return out


def _derivable_numbers(rows: list[dict]) -> set[float]:
    """fixture 能推出的全部数字：通过率（1 位小数）、题数、通过数，以及同维度
    跨模型的差值（百分点差、相差题数）。"""
    out: set[float] = set()
    by_dim: dict[str, list[dict]] = {}
    for row in rows:
        total, passed = int(row["total"]), int(row["passed"])
        out.update({float(total), float(passed)})
        if total:
            out.add(round(passed / total * 100, 1))
        by_dim.setdefault(row["dimension"], []).append(row)
    for items in by_dim.values():
        if len(items) != 2:
            continue
        a, b = items
        out.add(round(abs(a["passed"] / a["total"] - b["passed"] / b["total"]) * 100, 1))
        out.add(float(abs(a["passed"] - b["passed"])))
    return out


# --------------------------------------------------------------------------- #
# fixture 契约：8 个数据点，来源写清楚
# --------------------------------------------------------------------------- #
class TestFixtureContract:

    def test_fixture_checked_in(self) -> None:
        assert FIXTURE.is_file(), "fixture 必须入库（评测数据源不能靠本机生成）"

    def test_eight_data_points(self) -> None:
        rows = _fixture_rows()
        assert len(rows) == 8, "模型 2 × 维度 4 = 8 个数据点，就是工具的全部契约"
        assert {(r["model"], r["dimension"]) for r in rows} == {
            (m, d) for m in MODELS
            for d in ("correctness", "instruction_following", "safety", "relevance")
        }

    def test_provenance_declared(self) -> None:
        prov = json.loads(FIXTURE.read_text(encoding="utf-8"))["provenance"]
        assert "2026-09-23" in json.dumps(prov, ensure_ascii=False), "要写明数据日期"
        assert "report-full-regression-20260923-143903" in json.dumps(prov, ensure_ascii=False)
        assert "永不追新" in prov["maintenance"], "升级必须是一次显式决策"

    def test_labels_match_tool_table(self) -> None:
        """fixture 里的中文标签与工具的表必须一致，别出现两套叫法。"""
        for row in _fixture_rows():
            assert row["label"] == tools.DIMENSION_LABELS[row["dimension"]]


# --------------------------------------------------------------------------- #
# 永久护栏：金标必须还能从 fixture 推出来（这条 CI 必跑）
# --------------------------------------------------------------------------- #
class TestGoldenGuard:

    def test_golden_must_contain_derivable(self) -> None:
        tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))["tasks"]
        derivable = _derivable_numbers(_fixture_rows())
        bad = []
        for task in tasks:
            # 只钉 must_contain：must_not_contain 是幻觉题的**反例集**，
            # 里面本来就该有推不出来的数字（如 t07 的 93.3）
            for num in _numbers_in(task.get("golden_must_contain")):
                if num not in derivable:
                    bad.append((task["id"], num))
        assert not bad, f"这些金标数字已无法从 fixture 推出（数据源漂移了）：{bad}"

    def test_known_data_matches_fixture(self) -> None:
        """meta.known_data 是金标的汇总，也必须和 fixture 对得上。"""
        meta = json.loads(TASKS_FILE.read_text(encoding="utf-8"))["meta"]
        rows = {(r["model"], r["dimension"]): r for r in _fixture_rows()}
        dim_of = {"准确性": "correctness", "指令遵循": "instruction_following",
                  "安全": "safety", "相关性": "relevance"}
        for model, dims in meta["known_data"].items():
            for label, text in dims.items():
                row = rows[(model, dim_of[label])]
                rate = round(row["passed"] / row["total"] * 100, 1)
                # 比数值不比字符串：known_data 写「100%」，fixture 算出来是 100.0
                nums = _numbers_in([text])
                assert any(abs(n - rate) < 0.06 for n in nums), (
                    f"{model} {label}：fixture 是 {rate}%，known_data 写的是 {text}")
                assert float(row["total"]) in nums, (
                    f"{model} {label}：fixture 的题数是 {row['total']}，known_data 写的是 {text}")

    def test_tasks_json_meta_points_at_fixture(self) -> None:
        """判据文件要写明金标对应哪份 fixture——留痕的意义是下次漂移能被发现。"""
        meta = json.loads(TASKS_FILE.read_text(encoding="utf-8"))["meta"]
        assert meta["data_source"].endswith("eval_data_fixture.json")
        assert "2026-09-23" in meta["golden_source_note"]
        assert "v2_changes" in meta and "fixture" in meta["v2_changes"]


# --------------------------------------------------------------------------- #
# 双模：缺省 fixture，实时必须显式开启
# --------------------------------------------------------------------------- #
class TestDualMode:

    def test_default_is_fixture(self, monkeypatch) -> None:
        monkeypatch.delenv(tools.DATA_SOURCE_ENV, raising=False)
        assert tools.data_source_mode() == tools.SOURCE_FIXTURE

    def test_latest_only_when_explicit(self, monkeypatch) -> None:
        monkeypatch.setenv(tools.DATA_SOURCE_ENV, "latest")
        assert tools.data_source_mode() == tools.SOURCE_LATEST
        monkeypatch.setenv(tools.DATA_SOURCE_ENV, "LATEST")  # 大小写不敏感
        assert tools.data_source_mode() == tools.SOURCE_LATEST
        monkeypatch.setenv(tools.DATA_SOURCE_ENV, "fixture")
        assert tools.data_source_mode() == tools.SOURCE_FIXTURE

    def test_fixture_mode_answers_all_eight_queries(self, monkeypatch) -> None:
        monkeypatch.delenv(tools.DATA_SOURCE_ENV, raising=False)
        for model in MODELS:
            for dim in DIMS:
                out = tools.get_eval_result(model_name=model, dimension=dim)
                assert out.startswith(f"模型 {model} · 维度{dim}：通过率 "), out
                assert "失败" in out


# --------------------------------------------------------------------------- #
# 报错文案：一个字都不能变（t07 诚实题靠它）
# --------------------------------------------------------------------------- #
class TestErrorTextUnchanged:

    def test_unknown_dimension(self, monkeypatch) -> None:
        monkeypatch.delenv(tools.DATA_SOURCE_ENV, raising=False)
        assert (tools.get_eval_result(model_name="deepseek-chat", dimension="流畅性")
                == "未知维度 '流畅性'，可选：准确性、指令遵循、安全、相关性。请只传这四个中文名。")

    def test_unknown_model(self, monkeypatch) -> None:
        monkeypatch.delenv(tools.DATA_SOURCE_ENV, raising=False)
        assert (tools.get_eval_result(model_name="gpt-4", dimension="安全")
                == "未知模型 'gpt-4'，可选：deepseek-chat、deepseek-pro。请用这些名字原样重试。")


# --------------------------------------------------------------------------- #
# 硬验收：换数据源不该换答案（需要源报告，CI 沙盒跳过）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not SOURCE_REPORT.is_file(),
                    reason="reports/ 不入库，CI 沙盒里没有源报告（逻辑由 fixture 覆盖）")
class TestByteIdenticalAcrossSources:

    def test_eight_queries_identical(self, monkeypatch) -> None:
        pairs = [(m, d) for m in MODELS for d in DIMS]
        monkeypatch.delenv(tools.DATA_SOURCE_ENV, raising=False)
        from_fixture = [tools.get_eval_result(model_name=m, dimension=d) for m, d in pairs]
        monkeypatch.setenv(tools.DATA_SOURCE_ENV, tools.SOURCE_LATEST)
        monkeypatch.setattr(tools, "_latest_report_path", lambda: SOURCE_REPORT)
        from_report = [tools.get_eval_result(model_name=m, dimension=d) for m, d in pairs]
        assert from_fixture == from_report, "两种数据源的输出必须逐字节一致"
