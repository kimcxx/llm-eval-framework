"""LLM 客户端抽象层。

评测框架的第一条铁律：**被测对象必须是可替换的**。
所有指标、执行引擎只依赖 BaseLLM，因此换模型、加模型不需要改评测逻辑。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """一次模型调用的完整结果，包含文本与工程指标。"""

    text: str
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    """模型调用异常。retryable 决定执行引擎是否重试。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class BaseLLM(ABC):
    """模型客户端基类。子类只需实现 `_invoke`。

    `limiter` 由工厂注入：节流放在这一层，被测模型调用与裁判调用才会走同一套限速。
    """

    def __init__(self, name: str, model: str, **options: Any) -> None:
        self.name = name
        self.model = model
        self.options = options
        self.limiter: Any | None = None

    @abstractmethod
    def _invoke(
        self,
        system: str | None,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """真正执行一次模型调用（不包含计时与重试）。"""

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """对外统一入口，负责计时与节流。"""
        temp = self.options.get("temperature", 0.0) if temperature is None else temperature
        limit = self.options.get("max_tokens", 1024) if max_tokens is None else max_tokens

        if self.limiter is not None:
            self.limiter.acquire()

        started = time.perf_counter()
        response = self._invoke(system, prompt, float(temp), int(limit))
        response.latency_ms = (time.perf_counter() - started) * 1000
        return response

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} model={self.model!r}>"
