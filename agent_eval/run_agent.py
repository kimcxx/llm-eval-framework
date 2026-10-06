"""组装 Agent、跑冒烟任务、把轨迹落盘并在终端打印人类可读摘要。

一条命令跑通：``python agent_eval/run_agent.py``

大脑用 deepseek-chat 那组配置（model_id / base_url / api_key 全部来自
configs/models.yaml + .env，代码里没有硬编码的 key）。手上两个工具在
agent_eval/tools.py：查报告的 get_eval_result 和算数的 calculator。

造大脑 / 收轨迹 / 落盘这些活都在 common.py 里，和 run_eval.py 共用一份实现，
保证「冒烟跑的」和「评测跑的」是同一套东西。

为什么强调「轨迹」：Agent 的结论对不对，光看最终答案判断不了——同样的答案
可能是查了数据算出来的，也可能是蒙的。轨迹是它做事过程的证据，必须落盘。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# smolagents 是可选依赖（requirements.txt 里注释着），没装时别甩一屏
# traceback，直接给一句能照做的提示。提示文案与 common.MISSING_DEP_HINT 一致。
try:
    from smolagents import ToolCallingAgent  # noqa: E402
    from agent_eval.common import (  # noqa: E402
        build_model,
        collect_logs,
        dump_trajectory,
        ensure_utf8_stdio,
        print_trajectory_summary,
    )
    from agent_eval.tools import calculator, get_eval_result  # noqa: E402
except ImportError:
    raise SystemExit(
        "缺少依赖：请取消 requirements.txt 第 28 行注释并 pip install smolagents"
    )

TRAJECTORY_DIR = Path(__file__).resolve().parent / "trajectories"
MAX_STEPS = 8

SMOKE_TASK = (
    "deepseek-chat 和 deepseek-pro 在安全维度上各是多少通过率？差多少个百分点？"
    "样本量是多少题？差值相当于几道题？"
    "要求：先用 get_eval_result 分别查两个模型的「安全」维度拿到真实数据，"
    "再用 calculator 计算百分点差值和等效题数，最后按 "
    "「chat 通过率 / pro 通过率 / 差值 / 题数 / 等效题数」的顺序给出结论。"
)

# 金标（来自 9/23 全量报告），只用于跑完后自检，不喂给 Agent
GOLDEN = {
    "chat_pass_rate": "83.3",
    "pro_pass_rate": "80",
    "diff_pt": "3.3",
    "total": "30",
    "diff_cases": "1",
}


def main() -> int:
    # 与 run_eval 同一套：终端编码钉成 utf-8，免得打印轨迹时被特殊字符崩掉
    ensure_utf8_stdio()
    model = build_model()
    agent = ToolCallingAgent(
        tools=[calculator, get_eval_result],
        model=model,
        max_steps=MAX_STEPS,
    )

    print(f"\n[任务] {SMOKE_TASK}\n")
    final_answer = str(agent.run(SMOKE_TASK))

    logs = collect_logs(agent)
    usage = print_trajectory_summary(logs, final_answer)

    path = dump_trajectory(
        SMOKE_TASK,
        logs,
        final_answer,
        usage,
        extra={"golden": GOLDEN, "max_steps": MAX_STEPS},
    )
    print(f"\n[轨迹] 已落盘：{path}")

    missing = [k for k, v in GOLDEN.items() if v not in final_answer]
    if missing:
        print(f"[自检] 最终答案里缺少金标数字：{missing}（金标 {GOLDEN}）")
    else:
        print(f"[自检] 金标数字全部命中：{GOLDEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
