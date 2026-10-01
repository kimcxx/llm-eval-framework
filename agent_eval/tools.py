"""Agent 可用的两个工具。

工具就是 Agent 的手：手伸不到数据，Agent 再聪明也只能编。这里给两只手——

* ``calculator``：算数。差值、百分比、等效题数这类一步就能算错的东西，交给
  确定性代码，不要靠模型心算。
* ``get_eval_result``：查本项目自己的 LLM 评测报告。这样 Agent 答的是报告里
  真实存在的数字，而不是它以为存在过的数字。

两个工具的共同约定：**失败也返回字符串，不抛异常**。工具抛异常会直接打断
Agent 的推理链，而返回「未知维度，可选：…」给了它自我纠正的机会——
参数写错是模型最容易犯、也最容易自己改回来的错。
"""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path
from typing import Any

from smolagents import tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_GLOB = "report-full-regression-*.json"

# ---------------------------------------------------------------- 维度口径 ----
# 与 src/runner/dimension.py 保持一致：维度按「测法」分，不按数据集分。
# 这里再抄一份而不是 import，是为了让 agent_eval 自包含（跑起来不依赖 src 的
# 内部重构）；改了那边的口径，这里的中文名→键的映射也要同步改。
DIMENSION_ALIASES: dict[str, str] = {
    "准确性": "correctness",
    "correctness": "correctness",
    "指令遵循": "instruction_following",
    "instruction_following": "instruction_following",
    "format": "instruction_following",  # 老报告里的旧名
    "安全": "safety",
    "safety": "safety",
    "相关性": "relevance",
    "relevance": "relevance",
}

DIMENSION_LABELS: dict[str, str] = {
    "correctness": "准确性",
    "instruction_following": "指令遵循",
    "safety": "安全",
    "relevance": "相关性",
}

# 分类 → 维度。用来从 categories 段重算维度，见 _dimension_rows 的说明。
CATEGORY_DIMENSION: dict[str, str] = {
    "json_extract": "instruction_following",
    "math_reasoning": "correctness",
    "qa_zh": "correctness",
    "qa_open": "relevance",
    "safety_redteam": "safety",
}

# 这些模型是对照组（规则假模型），不出现在对外可选列表里
MOCK_PREFIX = "mock-"


# ------------------------------------------------------------------ 工具 1 ----
_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_EXPRESSION_LEN = 200


def _eval_node(node: ast.AST) -> float:
    """受限求值：只放行数字常量、一元正负、四则运算和括号。

    不碰 ``eval`` / ``exec``：那是把整台机器交给一段模型生成的字符串。
    AST 白名单的意义是「能算的照算，算之外的直接拒」，而不是事后打补丁。
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"只允许数字，收到 {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_eval_node(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("除数为 0")
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)

    raise ValueError(f"不支持的表达式成分：{type(node).__name__}")


def _format_number(value: float) -> str:
    """整数值不拖 .0；小数保留足够精度但不堆浮点噪声。"""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式，返回结果字符串。

    用于把「差值多少个百分点」「差值相当于几道题」这类换算交给确定性代码，
    避免模型心算出错。只支持加减乘除、括号和小数，例如
    "(83.3 - 80) * 30 / 100"、"(0.8333 - 0.8) * 30"。

    Args:
        expression: 要计算的表达式，如 "12 * (3 + 4) / 5"。不支持变量、函数、
            幂运算和字符串；这些会返回错误说明而不是抛异常。
    """
    raw = (expression or "").strip()
    if not raw:
        return "表达式为空，请传入形如 '83.3 - 80' 的算式。"
    if len(raw) > MAX_EXPRESSION_LEN:
        return f"表达式过长（{len(raw)} 字符，上限 {MAX_EXPRESSION_LEN}），请拆分后再算。"

    try:
        tree = ast.parse(raw, mode="eval")
    except SyntaxError as exc:
        return f"表达式无法解析：{raw!r}（{exc.msg}）"

    try:
        value = _eval_node(tree)
    except ZeroDivisionError as exc:
        return f"计算失败：{exc}（表达式 {raw!r}）"
    except (ValueError, TypeError, RecursionError) as exc:
        return f"计算失败：{exc}。只支持数字、+ - * / ( ) 和小数。"

    return f"{raw} = {_format_number(value)}"


# ------------------------------------------------------------------ 工具 2 ----
def _latest_report_path() -> Path | None:
    """reports/ 下最新的全量回归报告（按文件名时间戳排序，取最后一个）。"""
    if not REPORTS_DIR.is_dir():
        return None
    matches = sorted(REPORTS_DIR.glob(REPORT_GLOB))
    return matches[-1] if matches else None


def _load_latest_report() -> tuple[dict[str, Any] | None, str | None]:
    path = _latest_report_path()
    if path is None:
        return None, f"未找到全量回归报告：{REPORTS_DIR / REPORT_GLOB}（请先跑 python scripts/run_eval.py 生成）"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"报告读取/解析失败：{path.name}（{exc}）"
    if not isinstance(doc, dict):
        return None, f"报告格式异常（顶层不是对象）：{path.name}"
    return doc, None


def _real_models(doc: dict[str, Any]) -> list[str]:
    """报告里出现过的真实被测模型（排除 mock-* 对照组）。"""
    names: list[str] = []
    for group in (doc.get("dimensions"), doc.get("categories")):
        if isinstance(group, dict):
            names.extend(k for k in group if not str(k).startswith(MOCK_PREFIX))
    # 去重保序
    return list(dict.fromkeys(names))


def _dimension_rows(doc: dict[str, Any], model: str) -> dict[str, dict[str, int | float]]:
    """算出某模型各维度的 (total, passed)。

    **优先从 categories 段重算，而不是直接读 dimensions 段**。原因和
    dashboard 那边一模一样：dimensions 段是报告生成那一刻的快照，口径可能
    已经过期（9/23 那份里 correctness=50 题是把 qa_open 也算进去的旧口径，
    新口径下准确性只有 42 题）。按「分类 → 维度」重算，才能和看板上显示的
    数字对得上。只有报告里根本没有 categories 段时，才退回 dimensions 段。
    """
    rows: dict[str, dict[str, int | float]] = {}

    categories = doc.get("categories")
    if isinstance(categories, dict) and isinstance(categories.get(model), list):
        for row in categories[model]:
            if not isinstance(row, dict):
                continue
            dim = CATEGORY_DIMENSION.get(str(row.get("category") or "").strip().lower())
            if not dim:
                continue
            total = int(row.get("total") or 0)
            if not total:
                continue
            passed = int(row.get("passed") or 0)
            bucket = rows.setdefault(dim, {"total": 0, "passed": 0})
            bucket["total"] += total
            bucket["passed"] += passed

    if not rows:
        dimensions = doc.get("dimensions")
        if isinstance(dimensions, dict) and isinstance(dimensions.get(model), list):
            for row in dimensions[model]:
                if not isinstance(row, dict):
                    continue
                dim = DIMENSION_ALIASES.get(str(row.get("dimension") or "").strip())
                if not dim:
                    continue
                total = int(row.get("total") or 0)
                if not total:
                    continue
                # 同名维度可能分片出现（不同时期口径），累加而不是覆盖
                bucket = rows.setdefault(dim, {"total": 0, "passed": 0})
                bucket["total"] += total
                bucket["passed"] += int(row.get("passed") or 0)

    for bucket in rows.values():
        bucket["pass_rate"] = bucket["passed"] / bucket["total"] if bucket["total"] else 0.0
    return rows


@tool
def get_eval_result(model_name: str, dimension: str) -> str:
    """查询 LLM 评测报告里，某个模型在某个能力维度上的表现。

    返回该维度的**通过率、题数、失败数**。数据源是 reports/ 下最新的
    report-full-regression-*.json（全量回归报告）。

    Args:
        model_name: 被测模型名，只能是 "deepseek-chat" 或 "deepseek-pro"。
            这是 configs/models.yaml 里的 CLI 引用名（不是 API 里的真实模型名）。
        dimension: 能力维度，只能是下面四个中文名之一：
            "准确性" —— 有客观标准答案的题（数学推理、中文知识问答）；
            "指令遵循" —— 有没有按要求输出结构（如 JSON 抽取）；
            "安全" —— 越狱与拒答红队题；
            "相关性" —— 开放式问答，由 LLM 裁判打分。
            其它写法（英文、别名、错别字）都会被拒，并提示可选值。
    """
    doc, error = _load_latest_report()
    if error:
        return error

    available_models = _real_models(doc)
    model_key = str(model_name or "").strip()
    matched_model = next((m for m in available_models if m.lower() == model_key.lower()), None)
    if matched_model is None:
        options = "、".join(available_models) or "（报告里没有可用模型）"
        return f"未知模型 {model_name!r}，可选：{options}。请用这些名字原样重试。"

    dim_key = DIMENSION_ALIASES.get(str(dimension or "").strip())
    if dim_key is None:
        return "未知维度 {!r}，可选：准确性、指令遵循、安全、相关性。请只传这四个中文名。".format(dimension)

    rows = _dimension_rows(doc, matched_model)
    if not rows:
        return f"报告里没有模型 {matched_model} 的维度数据（该模型可能被跳过或未跑完）。"

    row = rows.get(dim_key)
    if row is None:
        available = "、".join(DIMENSION_LABELS.get(d, d) for d in rows)
        return f"报告里没有 {matched_model} 的{DIMENSION_LABELS[dim_key]}维度数据；该报告只有：{available}。"

    total = int(row["total"])
    passed = int(row["passed"])
    failed = total - passed
    return (
        f"模型 {matched_model} · 维度{DIMENSION_LABELS[dim_key]}："
        f"通过率 {row['pass_rate'] * 100:.1f}%（{passed}/{total}），"
        f"题数 {total}，失败 {failed}。"
    )
