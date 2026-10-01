"""run_agent.py 与 run_eval.py 共用的部分：造大脑、跑一次、收轨迹、落盘。

抽出来的原因不是「代码复用」这四个字本身，而是**两个入口必须跑同一套东西**：
冒烟脚本和评测脚本要是各组装一遍 agent，温差就出现了——一边换模型另一边没换、
一边记轨迹另一边不记，评测结果就不可比。共用一份实现，差异才只可能来自任务本身。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 缺依赖时给的可执行提示（行号指向 requirements.txt 里被注释掉的那行）
MISSING_DEP_HINT = "缺少依赖：请取消 requirements.txt 第 28 行注释并 pip install smolagents"

try:
    from smolagents import OpenAIServerModel, ToolCallingAgent
except ImportError:  # pragma: no cover - 只有没装依赖时才走得到
    raise SystemExit(MISSING_DEP_HINT)

from agent_eval.tools import calculator, get_eval_result  # noqa: E402
from src.config import ConfigError, load_config  # noqa: E402

AGENT_MODEL_NAME = "deepseek-chat"  # configs/models.yaml 里的引用名
TRAJECTORY_DIR = Path(__file__).resolve().parent / "trajectories"
DEFAULT_TOOLS = [calculator, get_eval_result]
DEFAULT_MAX_STEPS = 8


def build_model() -> OpenAIServerModel:
    """从项目配置构造大脑：model_id / api_key 都来自 models.yaml + .env。"""
    config = load_config()
    try:
        cfg = config.get_model(AGENT_MODEL_NAME)
    except ConfigError as exc:
        raise SystemExit(f"[配置错误] {exc}")

    if not cfg.is_available:
        raise SystemExit(
            f"[配置错误] 模型 {cfg.name} 不可用：{cfg.unavailable_reason}"
            "（请在项目根目录 .env 里配置后重试）"
        )

    print(f"[配置] 大脑 = {cfg.name}（model_id={cfg.model}, base_url={cfg.base_url}）")
    return OpenAIServerModel(
        model_id=cfg.model,
        api_base=cfg.base_url,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        # DeepSeek 这一档模型默认开 thinking，而 thinking 模式不接受 smolagents
        # 默认带的 tool_choice="required"（会 400：Thinking mode does not support
        # this tool_choice）。显式关掉思考：Agent 场景要的是稳定调工具，不是长推理。
        extra_body={"thinking": {"type": "disabled"}},
    )


def json_safe(value: Any) -> Any:
    """把轨迹里可能存在的非序列化对象（TokenUsage、异常等）变安全。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if hasattr(value, "model_dump_json"):
        try:
            return json.loads(value.model_dump_json())
        except Exception:
            pass
    return str(value)


def step_kind(step: dict[str, Any]) -> str:
    """给人看的步骤类型：TaskStep 只有 task，ActionStep 带 step_number。"""
    if "task" in step:
        return "TaskStep（任务输入）"
    number = step.get("step_number")
    return f"ActionStep #{number}" if number is not None else "ActionStep"


def collect_logs(agent: Any) -> list[dict[str, Any]]:
    """取轨迹步骤。

    smolagents 1.26 里「agent.logs」是空的——轨迹实际存在 ``agent.memory.steps``
    里，每个 step 是 ActionStep / TaskStep 对象，要 ``.dict()`` 才成字典
    （``get_full_steps()`` 干的就是这件事）。这里两条路都留着：优先内存里的
    完整步骤，拿不到再退回 ``logs``，免得换版本后轨迹突然变成空文件。
    """
    memory = getattr(agent, "memory", None)
    if memory is not None and hasattr(memory, "get_full_steps"):
        steps = memory.get_full_steps()
        if steps:
            return [s for s in steps if isinstance(s, dict)]

    fallback = getattr(agent, "logs", []) or []
    out: list[dict[str, Any]] = []
    for step in fallback:
        if isinstance(step, dict):
            out.append(step)
        elif hasattr(step, "dict"):
            out.append(step.dict())
        elif hasattr(step, "__dict__"):
            out.append(dict(vars(step)))
    return out


def tool_calls_of(step: dict[str, Any]) -> list[dict[str, Any]]:
    """取出这一步的工具调用（兼容 dict 和 ToolCall 对象两种形态）。"""
    raw = step.get("tool_calls") or []
    calls: list[dict[str, Any]] = []
    for call in raw:
        if isinstance(call, dict):
            name = call.get("name") or call.get("function", {}).get("name") or "?"
            args = call.get("arguments")
            if args is None:
                args = call.get("function", {}).get("arguments")
        else:
            name = getattr(call, "name", "?")
            args = getattr(call, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        calls.append({"name": str(name), "arguments": args})
    return calls


def observations_of(step: dict[str, Any]) -> list[str]:
    """同一步里的多个工具返回是用换行串起来的，拆开才看得出「谁返回了什么」。"""
    raw = step.get("observations") or []
    if isinstance(raw, str):
        return [line for line in raw.splitlines() if line.strip()]
    lines: list[str] = []
    for item in raw:
        lines.extend(line for line in str(item).splitlines() if line.strip())
    return lines


def tool_usage(logs: list[dict[str, Any]]) -> dict[str, int]:
    """统计每个工具被调了几次（关键断言的输入：有没有调、调了几次）。"""
    usage: dict[str, int] = {}
    for step in logs:
        for call in tool_calls_of(step):
            usage[call["name"]] = usage.get(call["name"], 0) + 1
    return usage


def print_trajectory_summary(logs: list[dict[str, Any]], final_answer: str) -> dict[str, int]:
    """在终端打印每步做了什么，并统计每个工具各被调了几次。"""
    usage: dict[str, int] = {}

    print("\n" + "=" * 72)
    print(f"轨迹摘要：共 {len(logs)} 步")
    print("=" * 72)

    for index, step in enumerate(logs, 1):
        calls = tool_calls_of(step)
        model_output = str(step.get("model_output") or "").strip()
        error = step.get("error")

        print(f"\n[步骤 {index}] {step_kind(step)}")
        if model_output:
            print(f"  思考/输出：{model_output[:300]}")
        if not calls:
            print("  工具调用：无")
        for call in calls:
            usage[call["name"]] = usage.get(call["name"], 0) + 1
            print(f"  调用工具：{call['name']}({json.dumps(call['arguments'], ensure_ascii=False)})")
        for obs in observations_of(step):
            print(f"  工具返回：{obs[:300]}")
        if error:
            print(f"  错误：{error}")

    print("\n" + "-" * 72)
    print("工具调用统计：" + ("、".join(f"{k} × {v}" for k, v in usage.items()) or "无"))
    print(f"最终答案：{final_answer}")
    print("-" * 72)
    return usage


def dump_trajectory(
    task: str,
    logs: list[dict[str, Any]],
    final_answer: str,
    usage: dict[str, int],
    tag: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    """轨迹落盘，返回文件路径。

    落盘的是完整步骤（含模型输入输出），不是只存结论——事后复盘要看的是
    「它凭什么得出这个结论」，只留答案的轨迹等于没留。
    """
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{tag}" if tag else ""
    path = TRAJECTORY_DIR / f"trajectory-{stamp}{suffix}.json"
    payload: dict[str, Any] = {
        "task": task,
        "model": AGENT_MODEL_NAME,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "steps": len(logs),
        "tool_usage": usage,
        "final_answer": final_answer,
        "logs": json_safe(logs),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_agent_task(task: str, max_steps: int = DEFAULT_MAX_STEPS, tag: str = "") -> dict[str, Any]:
    """新建 agent 跑一个任务，返回 {final_answer, logs, usage, trajectory_path, steps}。

    **每题新建 agent**：agent 是带记忆的对象，复用会让上一题的对话渗进下一题，
    评测就不是在评同一个东西了。
    """
    model = build_model()
    agent = ToolCallingAgent(tools=list(DEFAULT_TOOLS), model=model, max_steps=max_steps)

    final_answer = str(agent.run(task))
    logs = collect_logs(agent)
    usage = tool_usage(logs)
    path = dump_trajectory(task, logs, final_answer, usage, tag=tag)
    return {
        "final_answer": final_answer,
        "logs": logs,
        "usage": usage,
        "trajectory_path": str(path),
        "steps": len(logs),
    }
