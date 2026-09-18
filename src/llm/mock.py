"""Mock 模型：规则基线，不发起任何网络请求。

它解决两个真实问题：
1. **无 Key 也能跑通全流程** —— 学习/CI 场景不必消耗额度；
2. **提供对照基线** —— 真实模型与规则基线的分数差距，
   可以说明评测框架确实具备区分强弱模型的能力（而不是所有模型都得高分）。
"""

from __future__ import annotations

import json
import operator
import re
from typing import Any

from src.llm.base import BaseLLM, LLMResponse

_OPS: dict[str, Any] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "×": operator.mul,
    "x": operator.mul,
    "/": operator.truediv,
    "÷": operator.truediv,
}

_PAIR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([+\-*/×÷x])\s*(\d+(?:\.\d+)?)")


def _condense(text: str, limit: int = 24) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat[:limit]


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文约 1 字 1 token，英文约 4 字符 1 token。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return cjk + other // 4


def _solve_arithmetic(prompt: str) -> str | None:
    """从题目里找第一组「数 运算符 数」并计算，模拟最朴素的基线。"""
    match = _PAIR_RE.search(prompt)
    if not match:
        return None
    left, op, right = match.group(1), match.group(2), match.group(3)
    func = _OPS.get(op)
    if func is None:
        return None
    try:
        value = func(float(left), float(right))
    except ZeroDivisionError:
        return None
    return str(int(value)) if value == int(value) else f"{value:g}"


class MockLLM(BaseLLM):
    """确定性输出的规则基线模型。"""

    def __init__(self, name: str = "mock-baseline", model: str = "mock-rule-based", **options: Any) -> None:
        super().__init__(name=name, model=model, **options)

    def _invoke(
        self,
        system: str | None,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        text = self._answer(prompt)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=_estimate_tokens(prompt),
            completion_tokens=_estimate_tokens(text),
        )

    def _answer(self, prompt: str) -> str:
        lowered = prompt.lower()

        # 结构化抽取类：故意输出结构不合规的结果，用于验证指标能捕捉失败
        if "json" in lowered:
            return json.dumps({"answer": _condense(prompt)}, ensure_ascii=False)

        # 算术类
        solved = _solve_arithmetic(prompt)
        if solved is not None:
            return solved

        # 其余情况：固定话术
        return f"规则基线回答，题目要点：{_condense(prompt)}"
