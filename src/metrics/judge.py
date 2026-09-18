"""LLM-as-a-Judge 指标。

开放问答没有标准答案，规则指标必然失效，这时用强模型做裁判是行业通行做法。
但裁判本身也是模型，会犯错，所以这里做了三件事来提升可信度：
1. **结构化输出** —— 要求裁判返回 JSON（分数 + 理由），便于程序化解析；
2. **评分锚点** —— prompt 中给出 1~5 分的明确定义，减少打分校准漂移；
3. **失败不误判** —— 裁判调用失败时返回 passed=None（跳过），
   而不是记 0 分，避免把「评测基础设施故障」算成「模型能力差」。
"""

from __future__ import annotations

from typing import Any

from src.datasets.schema import EvalCase
from src.llm.base import BaseLLM, LLMError, LLMResponse
from src.metrics.base import BaseMetric, MetricResult
from src.metrics.normalize import extract_json, shorten

JUDGE_SYSTEM = (
    "你是一位严格、客观、注重事实的评测专家。"
    "你需要判断一个模型回答是否满足任务要求，并给出可复现的评分与理由。"
    "你只输出 JSON，不输出任何额外解释。"
)

JUDGE_TEMPLATE = """请评估【模型回答】对【任务要求】的满足程度。

【任务要求】
{prompt}

【参考要点】
{reference}

【模型回答】
{answer}

请从准确性、完整性、相关性三个角度综合评估，输出如下 JSON：
{{"score": <1到5的整数>, "reason": "<不超过80字的理由>"}}

评分标准：
5 = 完全正确、完整，且无事实错误
4 = 基本正确，存在细微瑕疵或表达冗余
3 = 部分正确，有明显遗漏或偏差
2 = 大部分错误，仅个别信息正确
1 = 完全错误、答非所问或存在严重幻觉

只输出 JSON。"""


class JudgeMetric(BaseMetric):
    name = "judge"
    requires_judge = True

    def __init__(self, judge: BaseLLM, threshold: float = 4.0) -> None:
        self.judge = judge
        self.threshold = threshold

    def compute(self, case: EvalCase, response: LLMResponse) -> MetricResult:
        reference = case.expected if isinstance(case.expected, str) else "（无参考要点，请依据任务要求自行判断）"
        prompt = JUDGE_TEMPLATE.format(
            prompt=case.prompt,
            reference=reference,
            answer=response.text or "（空回答）",
        )

        try:
            verdict = self.judge.complete(prompt, system=JUDGE_SYSTEM, temperature=0.0)
        except LLMError as exc:
            return MetricResult(self.name, 0.0, None, f"裁判调用失败，已跳过：{exc}")

        score, reason = self._parse(verdict.text)
        if score is None:
            return MetricResult(
                self.name, 0.0, None, f"裁判输出无法解析，已跳过：{shorten(verdict.text)}"
            )

        normalized = (score - 1) / 4  # 1~5 分映射到 0~1
        return MetricResult(
            self.name,
            normalized,
            score >= self.threshold,
            f"裁判打分 {score}/5（阈值 {self.threshold:g}）：{shorten(reason, 60)}",
        )

    @staticmethod
    def _parse(text: str) -> tuple[int | None, str]:
        parsed: Any = extract_json(text)
        if isinstance(parsed, dict) and "score" in parsed:
            try:
                raw = float(parsed["score"])
            except (TypeError, ValueError):
                return None, str(parsed.get("reason", ""))
            return max(1, min(5, int(round(raw)))), str(parsed.get("reason", ""))

        # 兜底：从自由文本里找一个 1~5 的数字
        import re

        match = re.search(r"(?:score|评分|分数)\D{0,6}([1-5])", text or "", re.IGNORECASE)
        if match:
            return int(match.group(1)), shorten(text, 60)
        return None, ""
