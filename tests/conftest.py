"""测试夹具。

核心思路：**测试执行引擎时绝不真实调用模型**。
用 FakeLLM 按 prompt 返回预设回答，这样执行引擎、指标、报告的逻辑
都能被确定性地验证，且不需要 API Key、不会产生费用、不会因网络抖动而 flaky。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
for extra in (SCRIPTS, ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from src.datasets.schema import EvalCase  # noqa: E402
from src.llm.base import BaseLLM, LLMError, LLMResponse  # noqa: E402


class FakeLLM(BaseLLM):
    """按 prompt 精确匹配返回预设回答的假模型。"""

    def __init__(
        self,
        mapping: dict[str, Any] | None = None,
        name: str = "fake",
        model: str = "fake-1",
        default: str = "（默认回答）",
    ) -> None:
        super().__init__(name=name, model=model)
        self.mapping = mapping or {}
        self.default = default
        self.calls: list[str] = []

    def _invoke(self, system: str | None, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        self.calls.append(prompt)
        text = self.mapping.get(prompt, self.default)
        if callable(text):
            text = text(prompt)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
        )


class LeakyLLM(BaseLLM):
    """把所有上下文（含 system prompt）原样吐出的假模型。

    用来模拟最典型的安全事故：模型无法区分「指令」与「上下文」，
    把系统提示词、内部密钥一并返回给用户。
    """

    def __init__(self, name: str = "leaky") -> None:
        super().__init__(name=name, model="leaky-1")

    def _invoke(self, system: str | None, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        return LLMResponse(text=f"{system or ''}\n{prompt}", model=self.model)


class FailingLLM(BaseLLM):
    """始终抛异常的假模型，用于验证「单点失败不拖垮整轮评测」。"""

    def __init__(self, name: str = "failing", retryable: bool = True, fail_times: int = 99) -> None:
        super().__init__(name=name, model="failing-1")
        self.retryable = retryable
        self.fail_times = fail_times
        self.attempts = 0

    def _invoke(self, system: str | None, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise LLMError("模拟调用失败", retryable=self.retryable)
        return LLMResponse(text="重试成功", model=self.model)


@pytest.fixture
def make_case() -> Callable[..., EvalCase]:
    def _make(**overrides: Any) -> EvalCase:
        payload: dict[str, Any] = {
            "id": "case-1",
            "category": "unit",
            "prompt": "问题",
            "metrics": ["exact_match"],
        }
        payload.update(overrides)
        return EvalCase.from_dict(payload)

    return _make


@pytest.fixture
def make_response() -> Callable[..., LLMResponse]:
    def _make(text: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(text=text, model="fake-1", **kwargs)

    return _make


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT
