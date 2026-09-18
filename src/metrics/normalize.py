"""文本归一化与抽取工具。

评测里大量「假失败」都来自格式差异（全角半角、标点、多余解释），
因此归一化必须集中在一处实现，所有指标复用同一套规则。
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

# 需要在比较前抹掉的噪声字符：空白 + 中英文标点
# 中文引号一律用 \u 转义书写，避免与围栏字符混淆导致字符串被意外截断
_NOISE_RE = re.compile(
    r"[\s\u3000\u201c\u201d\u2018\u2019，。！？、；：（）()《》〈〉【】"
    r"\[\]{},.!?;:~`\-_/\\|+*=&^%$#@]+"
)

# 「答案：xxx」「答案是 xxx」「结果为 xxx」这类前缀
_PREFIX_RE = re.compile(
    r"^(?:最终答案|正确答案|答案|答|结果|输出|回答)\s*(?:[:：]|是|为)\s*[:：]?\s*"
)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

_FENCE_RE = re.compile(r"```(?:json|python|text)?\s*(.*?)```", re.DOTALL)


def normalize_text(text: str | None) -> str:
    """NFKC 归一化 → 去「答案：」前缀 → 小写 → 去标点空白。"""
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).strip()
    value = _PREFIX_RE.sub("", value)
    value = value.lower()
    return _NOISE_RE.sub("", value)


def strip_code_fence(text: str | None) -> str:
    """剥掉 Markdown 代码块围栏，只留内容。"""
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_final_answer(text: str | None) -> str:
    """从可能很啰嗦的回答里抽出「裸答案」。

    优先级：显式「答案：」标记 > 单行全文 > 最后一行。
    """
    if not text:
        return ""
    cleaned = strip_code_fence(text)
    match = re.search(r"(?:最终答案|正确答案|答案|结果)\s*(?:[:：]|是|为)\s*[:：]?\s*([^\n]+)", cleaned)
    if match:
        return match.group(1).strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[0] if len(lines) == 1 else lines[-1]


def extract_json(text: str | None) -> Any | None:
    """从文本中抽出第一个合法 JSON（容忍前后有解释性文字）。"""
    if not text:
        return None

    cleaned = strip_code_fence(text).strip()
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char in "{[":
            try:
                parsed, _ = decoder.raw_decode(cleaned[index:])
                return parsed
            except json.JSONDecodeError:
                continue
    return None


def extract_numbers(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(m) for m in _NUMBER_RE.findall(strip_code_fence(text))]


def to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def bigram_jaccard(a: str, b: str) -> float:
    """字符二元组的 Jaccard 相似度，作为无语义模型时的兜底方案。"""
    if not a or not b:
        return 0.0

    def grams(s: str) -> set[str]:
        if len(s) < 2:
            return {s}
        return {s[i : i + 2] for i in range(len(s) - 1)}

    ga, gb = grams(a), grams(b)
    union = ga | gb
    if not union:
        return 0.0
    return len(ga & gb) / len(union)


def shorten(text: str | None, limit: int = 60) -> str:
    flat = re.sub(r"\s+", " ", (text or "")).strip()
    return flat if len(flat) <= limit else flat[:limit] + "…"
