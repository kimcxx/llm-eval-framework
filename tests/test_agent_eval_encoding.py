"""终端编码：模型输出里带特殊字符时，不能把整道题打成「无效」。

事故（2026-10-06）：Windows 控制台默认 GBK，模型输出里出现一个 U+2212（数学减号
「−」，模型爱用它代替 ASCII 的 -），``print`` 就抛 UnicodeEncodeError，那道题被记成
「无效」（步数 0）——不进通过率，却污染稳定性判定（「状态一致 8/10、存在抖动」里
就有它的功劳）。打印是给人看的，终端编码不该决定评测结果。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("smolagents", reason="agent_eval 依赖 smolagents（requirements 里有）")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_eval.common import (  # noqa: E402
    dump_trajectory,
    ensure_utf8_stdio,
    print_trajectory_summary,
)

# U+2212：数学减号。模型写「−3」而不是「-3」，GBK 控制台打不出来
MATH_MINUS = "−"
LOGS = [{
    "step_number": 1,
    "model_output": f"结果是 {MATH_MINUS}3 与 +5，差值为 {MATH_MINUS}8",
    "tool_calls": [{"name": "calculator", "arguments": {"expression": f"{MATH_MINUS}3 - 5"}}],
    "observations": [f"{MATH_MINUS}8"],
    "error": None,
}]


class TestMathMinusSafe:

    def test_stdio_is_utf8(self) -> None:
        ensure_utf8_stdio()
        assert (sys.stdout.encoding or "").lower().replace("-", "") == "utf8"

    def test_print_does_not_crash(self, capsys) -> None:
        """打印含 U+2212 的轨迹：不能抛，且字符要真的打出来。"""
        ensure_utf8_stdio()
        print_trajectory_summary(LOGS, f"最终答案：{MATH_MINUS}8")
        out = capsys.readouterr().out
        assert MATH_MINUS in out, "数学减号应该原样打印出来，不是被吞掉"

    def test_dump_does_not_crash(self, tmp_path, monkeypatch) -> None:
        """轨迹落盘同理：含 U+2212 不崩，且落进去的还是那个字符。"""
        monkeypatch.setattr("agent_eval.common.TRAJECTORY_DIR", tmp_path)
        path = dump_trajectory(
            task=f"算一算 {MATH_MINUS}3 与 5 的差",
            logs=LOGS,
            final_answer=f"{MATH_MINUS}8",
            usage={"calculator": 1},
        )
        text = path.read_text(encoding="utf-8")
        assert MATH_MINUS in text
        assert "calculator" in text
