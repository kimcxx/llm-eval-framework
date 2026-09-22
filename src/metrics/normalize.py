"""文本归一化与抽取工具。

评测里大量「假失败」都来自格式差异（全角半角、标点、多余解释），
因此归一化必须集中在一处实现，所有指标复用同一套规则。
"""

from __future__ import annotations

import json
import math
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


# 货币单位：抽取任务里「5999 元」「￥5,999」和 5999 是同一个值，
# 但字面比较会把它们判成错误，属于典型的假失败。
_CURRENCY_RE = re.compile(r"(?:人民币|rmb|cny|usd|￥|¥|\$|元|块钱)", re.IGNORECASE)

# 千分位分隔符：只删「数字,数字」这种位置的逗号，避免误伤 "a,b" 这类文本
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d)")


def _strip_numeric_noise(value: Any) -> str:
    """strip → 去货币单位 → 去千分位逗号（保留小数点，供数值解析用）。"""
    if value is None:
        return ""
    text = _CURRENCY_RE.sub("", str(value).strip())
    return _THOUSANDS_RE.sub("", text)


def normalize_scalar(value: Any) -> str:
    """标量值归一化：先抹掉货币/千分位噪声，再走通用文本归一化。"""
    return normalize_text(_strip_numeric_noise(value))


# 季度写法：同一时间段的多种合法写法（「2024Q2」「2024年第二季度」「2024年第2季度」）
# 在归一化层做格式互转，避免因写法不同被判成字段取值错误。
# 只做格式归一，不做语义包含 —— 例如「神舟十六号载人飞船」与「神舟十六号」
# 的等价关系只能显式写在用例的 expected 列表里，归一化层不会把二者判等。
_QUARTER_CN_DIGITS = {"一": "1", "二": "2", "三": "3", "四": "4", "1": "1", "2": "2", "3": "3", "4": "4"}

_QUARTER_RE = re.compile(
    r"(?P<year>(?:19|20)\d{2})\s*年?\s*(?:"
    r"第?\s*(?P<cn>[一二三四1-4])\s*季度"
    r"|q(?P<q>[1-4])"
    r")",
    re.IGNORECASE,
)


def normalize_quarter(value: Any) -> str | None:
    """把季度文本统一成 `YYYYQn`，非季度文本返回 None。

    例："2024Q2" / "2024年第二季度" / "2024 年第 2 季度" → "2024Q2"。
    """
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    match = _QUARTER_RE.search(text)
    if not match:
        return None
    digit = _QUARTER_CN_DIGITS.get(match.group("cn") or match.group("q") or "")
    if digit is None:
        return None
    return f"{match.group('year')}Q{digit}"


def scalar_equal(expected: Any, actual: Any) -> bool:
    """判断两个标量是否相等：优先按数值比，数值不可比时再按归一化文本比。

    数值优先是因为 5999 与 "5999"、"5,999.0" 都应该是同一个值；
    文本兜底则覆盖 "iOS"、"2024-06-11" 这类非数值字段。季度这类
    同一时间的不同写法（2024Q2 / 2024年第二季度）也会先统一格式再比。
    注意数值比较必须在抹掉标点（含小数点）之前进行，否则 1.0 会被文本归一化成 10。
    """
    want_raw = _strip_numeric_noise(expected)
    got_raw = _strip_numeric_noise(actual)
    if not want_raw or not got_raw:
        return want_raw == got_raw

    want_num = to_float(want_raw)
    got_num = to_float(got_raw)
    if want_num is not None and got_num is not None:
        return math.isclose(want_num, got_num, rel_tol=1e-9, abs_tol=1e-9)

    want_quarter = normalize_quarter(want_raw)
    got_quarter = normalize_quarter(got_raw)
    if want_quarter is not None and got_quarter is not None:
        return want_quarter == got_quarter

    return normalize_text(want_raw) == normalize_text(got_raw)


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
