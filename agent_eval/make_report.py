"""把最新一次正式评测结果渲染成单文件 HTML 报告（离线可看、无外部依赖）。

用法：
    python agent_eval/make_report.py                 # 取 results/ 下最新正式结果
    python agent_eval/make_report.py <result.json>   # 指定某次结果

为什么单文件：报告是要被当成「一次评测的结论」传来传去、归档、贴进聊天框的，
带一堆 css/js 或者依赖 localhost:8080 的服务，收件人打开就是一片白。所以样式
内联、数据内联，双击就能看。

取「最新正式结果」时必须先排除 INVALID（见 results/README.md）：带
``-INVALID-`` 的是作废存档，混进来等于把「通过了自己的题」的成绩又算一遍。
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
TASKS_FILE = BASE_DIR / "tasks.json"

# 与 run_eval.py / results/README.md 保持一致的作废标记
INVALID_MARK = "-INVALID-"

# 陷阱题防线要看的题：id → 这块防线在考什么（t08 是 v2 新增的防线说明）
TRAP_LABELS: dict[str, str] = {
    "t06-pure-calc-trap": "过度调用",
    "t07-unknown-dimension-honesty": "幻觉",
    "t08-category-name-recovery": "错误恢复",
    "t09-prior-bias": "先验",
}

STATUS_BADGE = {"通过": "pass", "失败": "fail", "无效": "skip"}

CSS = """
  :root {
    --bg: #0f1420; --panel: #171e2e; --panel2: #1d2740; --border: #2a3650;
    --text: #e6ecf7; --muted: #8fa0bf; --accent: #4f8ef7;
    --ok: #34c98e; --warn: #f5b449; --bad: #f0566a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font: 14px/1.6 "Segoe UI", "Microsoft YaHei", sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 22px; font-weight: 600; margin-bottom: 4px; }
  .sub { color: var(--muted); margin-bottom: 24px; }
  h2 { font-size: 15px; font-weight: 600; margin-bottom: 14px; }
  .card { background: var(--panel); border: 1px solid var(--border);
          border-radius: 12px; padding: 18px 20px; margin-bottom: 16px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
  .stat { background: var(--panel2); border: 1px solid var(--border);
          border-radius: 10px; padding: 12px 14px; }
  .stat .k { color: var(--muted); font-size: 12px; }
  .stat .v { font-size: 22px; font-weight: 600; margin-top: 2px; }
  .stat .v.ok { color: var(--ok); } .stat .v.bad { color: var(--bad); }
  .traps { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: var(--muted); font-weight: 500; padding: 8px 10px;
       border-bottom: 1px solid var(--border); white-space: nowrap; }
  td { padding: 8px 10px; border-bottom: 1px solid rgba(42,54,80,.5); vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  td.task { white-space: normal; min-width: 240px; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
           font-size: 12px; background: var(--panel2); color: var(--muted); }
  .badge.pass { background: rgba(52,201,142,.18); color: var(--ok); }
  .badge.fail { background: rgba(240,86,106,.18); color: var(--bad); }
  .badge.skip { background: rgba(245,180,73,.18); color: var(--warn); }
  .mono { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; font-size: 12px; }
  .answer { white-space: pre-wrap; background: var(--bg); border: 1px solid var(--border);
            border-radius: 6px; padding: 10px 12px; margin-top: 8px; }
  .step { border-left: 2px solid var(--border); padding-left: 12px; margin: 10px 0; }
  .step .who { color: var(--accent); }
  .step .obs { color: var(--muted); }
  details { background: var(--panel2); border: 1px solid var(--border);
            border-radius: 10px; padding: 10px 14px; margin-bottom: 10px; }
  details > summary { cursor: pointer; font-weight: 600; }
  footer { color: var(--muted); font-size: 12px; margin-top: 24px; line-height: 1.8; }
"""


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def latest_result() -> Path:
    """results/ 下最新一份**正式**结果（INVALID 存档不参与）。"""
    candidates = [
        p for p in RESULTS_DIR.glob("result-*.json") if INVALID_MARK not in p.name
    ]
    if not candidates:
        raise SystemExit(
            f"[错误] {RESULTS_DIR} 下没有正式结果；先跑 agent_eval/run_eval.py，"
            f"或手动指定结果文件：python agent_eval/make_report.py <result.json>"
        )
    return max(candidates, key=lambda p: p.name)


def load_tasks_meta() -> dict[str, Any]:
    if not TASKS_FILE.exists():
        return {}
    try:
        doc = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    meta = doc.get("meta") if isinstance(doc, dict) else None
    return meta if isinstance(meta, dict) else {}


def trap_status(summary: dict[str, Any]) -> dict[str, str]:
    """把 summary 里的陷阱题列表收成 id → 通过/失败。"""
    out: dict[str, str] = {}
    for item in summary.get("幻觉题") or []:
        if isinstance(item, dict):
            out[str(item.get("id"))] = str(item.get("status"))
    return out


def render_traps(traps: dict[str, str]) -> str:
    chips: list[str] = []
    for task_id, label in TRAP_LABELS.items():
        status = traps.get(task_id)
        if status is None:
            cls, text = "skip", "本题未跑"
        else:
            cls = STATUS_BADGE.get(status, "skip")
            text = f"{'守住' if status == '通过' else '未守住'}"
        chips.append(
            f'<span class="badge {cls}">{esc(task_id.split("-")[0])} {esc(label)}：{esc(text)}</span>'
        )
    return f'<div class="traps">{"".join(chips)}</div>'


def render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "<table><thead><tr>"
        "<th>#</th><th>id</th><th>级别</th><th>题面</th><th>结果</th>"
        "<th>失败的断言层</th><th>步数</th>"
        "</tr></thead><tbody>"
    ]
    for index, row in enumerate(rows, 1):
        status = str(row.get("status", "无效"))
        badge = STATUS_BADGE.get(status, "skip")
        failed_layers = [
            name for name, layer in (row.get("layers") or {}).items()
            if isinstance(layer, dict) and not layer.get("pass")
        ]
        if failed_layers:
            layer_html = "<br>".join(
                f"{esc(name)}：{esc((row['layers'][name] or {}).get('detail', ''))}"
                for name in failed_layers
            )
        else:
            layer_html = '<span style="color:var(--muted)">—</span>'
        lines.append(
            f"<tr>"
            f"<td>{index}</td>"
            f'<td class="mono">{esc(row.get("id"))}</td>'
            f"<td>{esc(row.get('level', ''))}</td>"
            f'<td class="task">{esc(row.get("task", ""))}</td>'
            f'<td><span class="badge {badge}">{esc(status)}</span></td>'
            f"<td>{layer_html}</td>"
            f"<td>{esc(row.get('steps'))} / {esc(row.get('max_steps'))}</td>"
            f"</tr>"
        )
    lines.append("</tbody></table>")
    return "".join(lines)


def traj_steps(path_str: str) -> list[str]:
    """从轨迹文件里还原「每步调了什么」；拿不到就返回空，调用方退回工具计数。"""
    if not path_str:
        return []
    path = Path(path_str)
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    out: list[str] = []
    for index, step in enumerate(doc.get("logs") or [], 1):
        if not isinstance(step, dict):
            continue
        calls = step.get("tool_calls") or []
        obs = step.get("observations") or []
        if isinstance(obs, str):
            obs = [line for line in obs.splitlines() if line.strip()]
        for call in calls:
            if isinstance(call, dict):
                name = call.get("name") or "?"
                args = call.get("arguments")
            else:
                name, args = getattr(call, "name", "?"), getattr(call, "arguments", None)
            try:
                arg_text = json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args
            except (TypeError, ValueError):
                arg_text = str(args)
            out.append(
                f'<div class="step"><span class="who">步骤 {index} · {esc(name)}</span>'
                f'<span class="mono">({esc(arg_text)})</span></div>'
            )
        for line in obs:
            out.append(f'<div class="step obs mono">{esc(str(line)[:300])}</div>')
    return out


def render_detail(row: dict[str, Any], traps: dict[str, str]) -> str:
    task_id = str(row.get("id") or "")
    status = str(row.get("status", "无效"))
    badge = STATUS_BADGE.get(status, "skip")
    usage = row.get("tool_usage") or {}
    usage_text = "、".join(f"{k} × {v}" for k, v in sorted(usage.items())) or "无工具调用"

    steps = traj_steps(str(row.get("trajectory_path") or ""))
    if steps:
        traj_html = "".join(steps)
    else:
        traj_html = f'<div class="step">轨迹文件不可用，仅统计：{esc(usage_text)}</div>'

    trap_note = ""
    if task_id in TRAP_LABELS:
        trap_note = f' · 陷阱题（{esc(TRAP_LABELS[task_id])}）：{esc(traps.get(task_id, "未判定"))}'

    return (
        f"<details{' open' if status != '通过' else ''}>"
        f"<summary>{esc(task_id)} "
        f'<span class="badge {badge}">{esc(status)}</span>'
        f'<span style="color:var(--muted);font-weight:400"> · {esc(row.get("level", ""))}'
        f" · {esc(row.get('steps'))} 步{trap_note}</span></summary>"
        f'<div style="margin-top:8px;color:var(--muted)">题面：{esc(row.get("task", ""))}</div>'
        f'<div style="margin-top:8px">工具调用统计：{esc(usage_text)}</div>'
        f'<div class="answer">{esc(row.get("final_answer", ""))}</div>'
        f'<div style="margin-top:10px">轨迹：</div>{traj_html}'
        f"</details>"
    )


def build_html(result: dict[str, Any], source: Path) -> str:
    summary = result.get("summary") or {}
    rows = result.get("results") or []
    traps = trap_status(summary)
    meta = load_tasks_meta()

    total = int(summary.get("任务总数") or len(rows) or 0)
    passed = int(summary.get("通过") or 0)
    rate = summary.get("任务成功率")
    rate_text = f"{rate:.0%}" if isinstance(rate, (int, float)) else "—"
    tool_rate = summary.get("工具选择正确率")
    tool_text = f"{tool_rate:.0%}" if isinstance(tool_rate, (int, float)) else "—"
    trap_hold = bool(summary.get("幻觉题是否守住"))

    # 展开区：失败题优先，其次是所有陷阱题（陷阱题通过也值得人工看一眼怎么过的）
    detail_ids = [r.get("id") for r in rows if r.get("status") != "通过"]
    for task_id in TRAP_LABELS:
        if task_id not in detail_ids:
            detail_ids.append(task_id)
    detail_rows = [r for r in rows if r.get("id") in detail_ids]

    data_source = str(meta.get("data_source") or "（tasks.json 未声明）")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent 评测报告 · {esc(meta.get("name") or "agent-eval")}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>Agent 评测报告</h1>
  <div class="sub">任务集 {esc(meta.get("name") or "—")} · 结果文件 {esc(source.name)} · repeat={esc(result.get("repeat", 1))}</div>

  <div class="card">
    <h2>汇总</h2>
    <div class="stats">
      <div class="stat"><div class="k">任务成功率</div>
        <div class="v {'ok' if passed == total and total else 'bad'}">{esc(rate_text)}</div>
        <div class="k">{passed} / {total} 通过</div></div>
      <div class="stat"><div class="k">工具选择正确率</div>
        <div class="v {'ok' if tool_text == '100%' else 'bad'}">{esc(tool_text)}</div></div>
      <div class="stat"><div class="k">平均步数</div>
        <div class="v">{esc(summary.get("平均步数", "—"))}</div></div>
      <div class="stat"><div class="k">陷阱题防线</div>
        <div class="v {'ok' if trap_hold else 'bad'}">{'守住' if trap_hold else '未守住'}</div></div>
    </div>
    {render_traps(traps)}
  </div>

  <div class="card">
    <h2>明细（{len(rows)} 题）</h2>
    {render_table(rows)}
  </div>

  <div class="card">
    <h2>失败题与陷阱题展开（{len(detail_rows)} 题）</h2>
    {"".join(render_detail(r, traps) for r in detail_rows) or '<div style="color:var(--muted)">无</div>'}
  </div>

  <footer>
    任务集版本：{esc(meta.get("name") or "—")}（{esc(TASKS_FILE.name)}）<br>
    数据源报告：{esc(data_source)}<br>
    结果文件：{esc(source.name)}　跑测时间：{esc(result.get("started_at", "—"))} → {esc(result.get("finished_at", "—"))}<br>
    生成时间：{esc(stamp)}　生成脚本：agent_eval/make_report.py（阿锤）
  </footer>
</div>
</body>
</html>
"""


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_result()
    if not source.is_file():
        raise SystemExit(f"[错误] 结果文件不存在：{source}")
    result = json.loads(source.read_text(encoding="utf-8"))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = REPORTS_DIR / f"agent-eval-{stamp}.html"
    out.write_text(build_html(result, source), encoding="utf-8")

    print(f"[报告] 已生成：{out}")
    print(f"[报告] 数据源：{source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
