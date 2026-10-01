"""按 tasks.json 逐题跑 Agent 评测，四层断言自动判定，结果落盘。

一条命令：``.venv\\Scripts\\python.exe agent_eval/run_eval.py``

和 run_agent.py 的关系：那边是「冒烟」（一条任务看看通不通），这边是「评测」
（一批任务，每条都要过断言）。两者共用 common.py 里的 build_model / 轨迹采集 /
落盘，保证被测的是同一个 agent，差异只来自任务本身。

**断言为什么分四层**：只知道「答案对不对」，诊断价值很低。答案错可能是查错数据、
可能是没查靠编、可能是算错、可能是绕远路。分层的好处是失败时能直接指到那一层：

1. 有效性 —— 轨迹为空 / 步数 < 2 → 判「评测无效」，不是任务失败。
   轨迹都没记下来，等于没有证据，这时说「任务失败」是在瞎判定。
2. 答案层 —— 金标必含 / 必不含（幻觉题靠「必不含」守住）。
3. 工具层 —— 该调的调了没、不该调的调了没、最少调用次数够不够。
4. 步骤效率 —— 实际步数超过任务声明上限，说明它在绕路。
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

from agent_eval.common import observations_of, run_agent_task  # noqa: E402

TASKS_FILE = Path(__file__).resolve().parent / "tasks.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARCHIVE_DIR = RESULTS_DIR / "archive"

# results/ 归档约定（详见 results/README.md）：
#   正式结果 result-<YYYYmmdd-HHMMSS>.json          → 计入成绩，只留最新一份在 results/
#   作废结果 result-INVALID-<原因>-<时间戳>.json      → 永不统计，原地留档
#   历史正式结果                                     → 归档进 results/archive/
INVALID_MARK = "-INVALID-"

# 工具「返回了错误提示」的判据：这些前缀都是 tools.py 里刻意返回的字符串，
# 不是异常——工具报错也要能被 agent 看见，所以这里从轨迹文本里认它们。
TOOL_ERROR_MARKERS = (
    "未知模型",
    "未知维度",
    "计算失败",
    "表达式无法解析",
    "表达式过长",
    "未找到全量回归报告",
    "报告读取/解析失败",
    "报告格式异常",
    "报告里没有",
)

# 给 agent 的步数预算比声明上限多 2 步：断言才有意义——
# 直接把上限当预算传进去，超了也被框架截断，这一层永远「通过」。
STEP_HEADROOM = 2


def load_tasks() -> list[dict[str, Any]]:
    """读 tasks.json——**以文件实际结构为准，脚本迁就文件**。

    真实结构（阿灯的设计基准）：
    ```json
    {"meta": {"name": ..., "known_data": ...},
     "tasks": [{"id": "t01-basic-lookup", "level": "基础", "task": "<题面>",
                "golden_must_contain": [...], "golden_must_not_contain": [...],
                "expect": {"must_call": [...], "must_not_call": [...],
                           "min_get_eval_result_calls": 4,
                           "expect_tool_error": true, "max_steps": 8},
                "design_note": "..."}]}
    ```
    要点：题面在 ``task``；工具/步数期望在 ``expect`` 对象里；**金标
    （golden_must_contain / golden_must_not_contain）在任务顶层，不在 expect 里**。

    缺字段宁可当场报错也不静默取默认值：静默取 0 会让断言「通过」得毫无意义
    （比如 min_get_eval_result_calls 缺失按 0 处理，等于这层根本没检查）。
    """
    if not TASKS_FILE.exists():
        raise SystemExit(f"[配置错误] 任务文件不存在：{TASKS_FILE}")
    try:
        doc = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[配置错误] {TASKS_FILE} 不是合法 JSON：{exc}")

    if isinstance(doc, list):
        raise SystemExit(
            f"[配置错误] {TASKS_FILE} 顶层是任务数组（{len(doc)} 条），"
            "但设计基准是 {\"meta\": ..., \"tasks\": [...]} 结构；请确认任务文件版本"
        )
    if not isinstance(doc, dict) or not isinstance(doc.get("tasks"), list) or not doc["tasks"]:
        raise SystemExit(f"[配置错误] {TASKS_FILE} 里没有非空的 tasks 列表")

    tasks = doc["tasks"]
    for index, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            raise SystemExit(f"[配置错误] 第 {index} 条任务不是对象：{type(task).__name__}")
        missing = [k for k in ("id", "task") if k not in task]
        if missing:
            raise SystemExit(f"[配置错误] 第 {index} 条任务缺少字段：{missing}")
        if not isinstance(task.get("expect"), dict):
            raise SystemExit(f"[配置错误] 任务 {task.get('id')} 缺少 expect 对象")
    return tasks


def _action_steps(logs: list[dict[str, Any]]) -> int:
    """真正消耗模型调用的步骤数（TaskStep 只是任务输入，不算一步）。"""
    return len([s for s in logs if "step_number" in s or s.get("tool_calls")])


def _check_validity(logs: list[dict[str, Any]]) -> tuple[bool, str]:
    if not logs:
        return False, "轨迹为空（agent.logs 与 memory.steps 都没取到步骤）"
    if len(logs) < 2:
        return False, f"轨迹只有 {len(logs)} 步（<2），无法判断它是否真的动过手"
    return True, f"轨迹 {len(logs)} 步"


def expect_of(task: dict[str, Any]) -> dict[str, Any]:
    """从任务里取出归一化后的断言期望。

    取值位置按文件实际结构：工具/步数期望在 ``task["expect"]``，金标在任务顶层。

    所有断言层只认这里吐出的东西，dry-run 打印的也是这一份——**自检打印的
    必须就是评测真正要用的**，否则 dry-run 只是好看。
    """
    expect = task.get("expect") or {}
    return {
        "must_call": list(expect.get("must_call") or []),
        "must_not_call": list(expect.get("must_not_call") or []),
        "min_get_eval_result_calls": int(expect.get("min_get_eval_result_calls") or 0),
        "expect_tool_error": bool(expect.get("expect_tool_error")),
        "max_steps": int(expect.get("max_steps") or 0),
        "golden_must_contain": [str(x) for x in (task.get("golden_must_contain") or [])],
        "golden_must_not_contain": [str(x) for x in (task.get("golden_must_not_contain") or [])],
    }


def is_trap(task: dict[str, Any]) -> bool:
    """这道题是不是「陷阱题」（幻觉防线）。

    两条判据取并集：
    1. **设计自带的分类**：``level`` 里写「陷阱·…」的（t06 工具选择 / t07 诚实性 /
       t08 错误恢复 / t09 先验对抗）——这是阿灯在设计里标好的。
    2. **按内容推**：要求了「必不含」（不许编造具体数字）或「预期工具报错」
       （查不到就该报）的题。

    只看第 2 条会漏掉 t06（数据已在题面里，考的是别去乱查）和 t09（考别被
    「pro 更强」的先验带跑），但它们 level 里明明白白写着陷阱，所以取并集。
    """
    if "陷阱" in str(task.get("level", "")):
        return True
    expect = expect_of(task)
    return bool(expect["golden_must_not_contain"]) or expect["expect_tool_error"]


def _check_answer(final_answer: str, task: dict[str, Any]) -> tuple[bool, str]:
    expect = expect_of(task)
    problems: list[str] = []
    for token in expect["golden_must_contain"]:
        if token not in final_answer:
            problems.append(f"缺少「{token}」")
    for token in expect["golden_must_not_contain"]:
        if token in final_answer:
            problems.append(f"出现了不该有的「{token}」")
    return (not problems), ("；".join(problems) or "金标命中")


def _check_tools(usage: dict[str, int], task: dict[str, Any]) -> tuple[bool, str]:
    expect = expect_of(task)
    problems: list[str] = []
    for name in expect["must_call"]:
        if usage.get(name, 0) < 1:
            problems.append(f"没调用 {name}")
    for name in expect["must_not_call"]:
        if usage.get(name, 0) > 0:
            problems.append(f"调用了不该调的 {name} × {usage[name]}")
    minimum = expect["min_get_eval_result_calls"]
    if usage.get("get_eval_result", 0) < minimum:
        problems.append(
            f"get_eval_result 只调了 {usage.get('get_eval_result', 0)} 次，要求 ≥ {minimum}"
        )
    return (not problems), ("；".join(problems) or "工具选择正确")


def _check_tool_health(logs: list[dict[str, Any]], task: dict[str, Any]) -> tuple[bool, str]:
    """expect_tool_error=true 的任务，轨迹里必须真的出现过工具错误提示。"""
    expect = expect_of(task)
    texts = [obs for step in logs for obs in observations_of(step)]
    hit = [m for m in TOOL_ERROR_MARKERS if any(m in t for t in texts)]
    if expect["expect_tool_error"]:
        return (bool(hit)), (f"捕获到工具错误提示：{hit}" if hit else "没出现任何工具错误提示")
    if hit:
        return False, f"不该出错却出了错：{hit}"
    return True, "无工具错误"


def _check_efficiency(logs: list[dict[str, Any]], task: dict[str, Any]) -> tuple[bool, str]:
    limit = expect_of(task)["max_steps"]
    used = _action_steps(logs)
    if limit and used > limit:
        return False, f"用了 {used} 步，超过上限 {limit}"
    return True, f"{used} 步（上限 {limit}）"


def evaluate_task(task: dict[str, Any], index: int, total: int) -> dict[str, Any]:
    task_id = task.get("id") or f"#{index}"
    level = task.get("level", "")
    print(f"\n[{index}/{total}] {task_id}  [{level}]")
    print(f"  任务：{str(task.get('task', ''))[:80]}…")

    budget = expect_of(task)["max_steps"] + STEP_HEADROOM
    run = run_agent_task(task["task"], max_steps=budget, tag=str(task_id))
    logs: list[dict[str, Any]] = run["logs"]
    usage: dict[str, int] = run["usage"]
    final_answer: str = run["final_answer"]

    layers: dict[str, dict[str, Any]] = {}
    for name, (ok, detail) in {
        "有效性": _check_validity(logs),
        "答案层": _check_answer(final_answer, task),
        "工具层": _check_tools(usage, task),
        "工具健康": _check_tool_health(logs, task),
        "步骤效率": _check_efficiency(logs, task),
    }.items():
        layers[name] = {"pass": ok, "detail": detail}

    valid = bool(layers["有效性"]["pass"])
    if not valid:
        status = "无效"
    elif all(layer["pass"] for key, layer in layers.items() if key != "有效性"):
        status = "通过"
    else:
        status = "失败"

    failures = [f"{name}：{layer['detail']}" for name, layer in layers.items() if not layer["pass"]]
    usage_text = "、".join(f"{k}×{v}" for k, v in usage.items()) or "无"
    if status == "通过":
        print(f"  → 通过（{_action_steps(logs)} 步，{usage_text}）")
    else:
        print(f"  → {status}（{_action_steps(logs)} 步，{usage_text}）")
        for line in failures:
            print(f"     - {line}")

    return {
        "id": task_id,
        "level": level,
        "task": task.get("task", ""),
        "status": status,
        "steps": _action_steps(logs),
        "max_steps": expect_of(task)["max_steps"],
        "tool_usage": usage,
        "layers": layers,
        "failure_reasons": failures,
        "final_answer": final_answer,
        "trajectory_path": run["trajectory_path"],
    }


def _parse_repeat() -> int:
    """--repeat N：同一任务集完整跑 N 遍（默认 1，行为与原来完全一致）。

    手动解析而不上 argparse：脚本目前只有 --dry-run 一个开关，为两个开关
    引入 argparse 不值当；开关再多两个就换。
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--repeat":
            if i + 1 >= len(sys.argv):
                raise SystemExit("[用法错误] --repeat 后面要跟遍数，例如 --repeat 3")
            try:
                n = int(sys.argv[i + 1])
            except ValueError:
                raise SystemExit(f"[用法错误] --repeat 的遍数必须是整数，收到：{sys.argv[i + 1]}")
            if n < 1:
                raise SystemExit(f"[用法错误] --repeat 至少为 1，收到：{n}")
            return n
    return 1


def summarize_repeats(all_repeats: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """逐题比对 N 遍结果：状态是否一致、各遍步数——抖动的题一眼可见。

    为什么看「状态一致」而不是平均通过率：一题 3 遍 2 过 1 挂，平均通过率 0.67
    看着还行，但这题其实是颗雷（线上 1/3 概率掉链子）。一致性把这类题直接点名。
    """
    first = all_repeats[0]
    per_task: list[dict[str, Any]] = []
    for i, row in enumerate(first):
        statuses = [rep[i]["status"] for rep in all_repeats]
        steps = [rep[i]["steps"] for rep in all_repeats]
        per_task.append({
            "id": row["id"],
            "各遍状态": statuses,
            "各遍步数": steps,
            "通过次数": statuses.count("通过"),
            "状态一致": len(set(statuses)) == 1,
        })
    stable = [p for p in per_task if p["状态一致"]]
    return {
        "遍数": len(all_repeats),
        "状态一致题数": f"{len(stable)}/{len(per_task)}",
        "全部一致": len(stable) == len(per_task),
        "逐题": per_task,
    }


def summarize(rows: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = len([r for r in rows if r["status"] == "通过"])
    invalid = len([r for r in rows if r["status"] == "无效"])
    tool_ok = len([r for r in rows if r.get("layers", {}).get("工具层", {}).get("pass")])
    # 陷阱题按内容识别（is_trap），不读 hallucination_trap 标记
    trap_ids = [t.get("id") for t in tasks if is_trap(t)]
    trap_rows = [r for r in rows if r["id"] in trap_ids]
    trap_hold = bool(trap_rows) and all(r["status"] == "通过" for r in trap_rows)
    steps = [r["steps"] for r in rows if r["status"] != "无效"]
    return {
        "任务总数": total,
        "通过": passed,
        "失败": len([r for r in rows if r["status"] == "失败"]),
        "无效": invalid,
        "任务成功率": round(passed / total, 4) if total else 0.0,
        "工具选择正确率": round(tool_ok / total, 4) if total else 0.0,
        "幻觉题是否守住": trap_hold,
        "幻觉题": [
            {"id": r["id"], "status": r["status"]} for r in trap_rows
        ],
        "平均步数": round(sum(steps) / len(steps), 2) if steps else 0.0,
    }


def dry_run(tasks: list[dict[str, Any]]) -> int:
    """只解析不跑模型：把每题读到的断言期望原样打印，先确认「读对了」再烧 token。

    打印的就是 expect_of() 的返回值——和真正评测时用的是同一份数据，
    不是另写一份展示逻辑。解析错了在这里就能看出来，别等跑完 10 题才发现。
    """
    print("[dry-run] 只解析 tasks.json，不调用模型\n")
    for index, task in enumerate(tasks, 1):
        expect = expect_of(task)
        trap = is_trap(task)
        print(f"[{index}/{len(tasks)}] {task.get('id')}  [{task.get('level', '')}]")
        print(f"   题面：{str(task.get('task', ''))[:60]}")
        print(f"   必含：{expect['golden_must_contain'] or '（无）'}")
        print(f"   必不含：{expect['golden_must_not_contain'] or '（无）'}")
        print(f"   必须调用：{expect['must_call'] or '（无）'}")
        print(f"   禁止调用：{expect['must_not_call'] or '（无）'}")
        print(f"   get_eval_result 最少次数：{expect['min_get_eval_result_calls']}")
        print(f"   预期工具报错：{'是' if expect['expect_tool_error'] else '否'}")
        print(f"   步数上限：{expect['max_steps']}")
        print(f"   陷阱题：{'是' if trap else '否'}")
        print()

    traps = [t.get("id") for t in tasks if is_trap(t)]
    print("-" * 72)
    print(f"解析成功 {len(tasks)} 题；陷阱题：{traps or '（无）'}")
    print("-" * 72)
    return 0


def _rel_or_abs(path: Path) -> str:
    """尽量输出相对项目根目录的路径（带正斜杠），失败就退回绝对路径。"""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return str(path)


def _archive_previous(current: Path) -> list[str]:
    """把 results/ 里更早的正式结果挪进 results/archive/。

    results/ 只留最新一份正式结果，避免「哪份才是这次的成绩」靠人肉记。
    INVALID 的不动——它们是刻意留档的作废记录，就该待在原地显眼处。
    """
    archived: list[str] = []
    for old in sorted(RESULTS_DIR.glob("result-*.json")):
        if old.name == current.name or INVALID_MARK in old.name:
            continue
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        target = ARCHIVE_DIR / old.name
        old.replace(target)
        archived.append(old.name)
    return archived


def main() -> int:
    tasks = load_tasks()

    if "--dry-run" in sys.argv:
        return dry_run(tasks)

    repeat = _parse_repeat()
    started = datetime.now()
    print(f"[评测] 共 {len(tasks)} 题，每题新建 agent，repeat={repeat}")

    all_repeats: list[list[dict[str, Any]]] = []
    for r in range(1, repeat + 1):
        if repeat > 1:
            print(f"\n{'#' * 72}\n# 第 {r}/{repeat} 遍\n{'#' * 72}")
        rows_r: list[dict[str, Any]] = []
        for index, task in enumerate(tasks, 1):
            try:
                rows_r.append(evaluate_task(task, index, len(tasks)))
            except Exception as exc:  # 单题炸了不能带走整场评测
                print(f"  → 异常：{type(exc).__name__}: {exc}")
                rows_r.append(
                    {
                        "id": task.get("id") or f"#{index}",
                        "level": task.get("level", ""),
                        "task": task.get("task", ""),
                        "status": "无效",
                        "steps": 0,
                        "max_steps": expect_of(task)["max_steps"],
                        "tool_usage": {},
                        "layers": {},
                        "failure_reasons": [f"运行异常：{type(exc).__name__}: {exc}"],
                        "final_answer": "",
                        "trajectory_path": "",
                    }
                )
        all_repeats.append(rows_r)

    # results 字段放最后一遍：make_report.py 只读它，结构不变、报告不受影响。
    # 完整 N 遍数据在 repeats / repeat_summary 里，追溯用。
    rows = all_repeats[-1]
    summary = summarize(rows, tasks)
    repeat_summary = summarize_repeats(all_repeats) if repeat > 1 else None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"result-{stamp}.json"
    payload: dict[str, Any] = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        # 相对路径：结果 JSON 是要进仓库归档的，落绝对路径会把本机用户名
        # 一起写进去（跨平台也看不懂）。拿不到相对路径就退回绝对路径，
        # 宁可多一个绝对路径，也不写空值让人误判「没记题集来源」。
        "tasks_file": _rel_or_abs(TASKS_FILE),
        "repeat": repeat,
        "summary": summary,
        "results": rows,
    }
    if repeat_summary is not None:
        payload["repeat_summary"] = repeat_summary
        payload["repeats"] = all_repeats
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    archived = _archive_previous(path)

    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    for row in rows:
        mark = {"通过": "PASS", "失败": "FAIL", "无效": "SKIP"}[row["status"]]
        print(f"{mark}  {row['id']}  [{row.get('level', '')}]  （{row['steps']} 步）")
        for line in row["failure_reasons"]:
            print(f"        └ {line}")
    print("-" * 72)
    print(
        f"任务成功率 {summary['任务成功率']:.0%}"
        f"（通过 {summary['通过']}/{summary['任务总数']}，失败 {summary['失败']}，无效 {summary['无效']}）"
    )
    print(f"工具选择正确率 {summary['工具选择正确率']:.0%}")
    print(f"陷阱题防线（含幻觉题）：{'守住' if summary['幻觉题是否守住'] else '未守住'} "
          f"{summary['幻觉题']}")
    print(f"平均步数 {summary['平均步数']}")
    if repeat_summary is not None:
        print("-" * 72)
        print(f"逐题一致性（{repeat_summary['遍数']} 遍）：")
        for item in repeat_summary["逐题"]:
            mark = "稳" if item["状态一致"] else "抖"
            print(
                f"  [{mark}] {item['id']}  状态 {item['各遍状态']}  步数 {item['各遍步数']}"
            )
        print(
            f"状态一致 {repeat_summary['状态一致题数']}"
            f"（{'全部一致' if repeat_summary['全部一致'] else '存在抖动题'}）"
        )
    print(f"[结果] 已落盘：{path}")
    if archived:
        print(f"[归档] 旧的正式结果已移入 archive/：{archived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
