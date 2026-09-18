"""OpenAI 兼容协议客户端。

DeepSeek、通义千问（百炼兼容模式）、智谱 GLM 都提供 OpenAI 兼容端点，
因此一套代码即可覆盖，这是选型上最省事也最贴近生产实践的做法。
"""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLM, LLMError, LLMResponse

# 这些状态码代表「稍后重试可能成功」
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
    ) -> None:
        super().__init__(name=name, model=model, temperature=temperature, max_tokens=max_tokens)
        if not api_key:
            raise LLMError(f"模型 {name} 缺少 API Key，无法初始化")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失属于环境问题
            raise LLMError("未安装 openai 依赖，请执行: pip install -r requirements.txt") from exc

        # max_retries=0：重试策略由评测框架统一控制，便于统计「真实失败率」
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,
        )

    def _invoke(
        self,
        system: str | None,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - 统一转成框架内异常
            raise self._translate(exc) from exc

        if not completion.choices:
            raise LLMError(f"模型 {self.name} 返回空 choices", retryable=True)

        text = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)

        return LLMResponse(
            text=text,
            model=getattr(completion, "model", self.model),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            raw=completion,
        )

    @staticmethod
    def _translate(exc: Exception) -> LLMError:
        """把 SDK 异常翻译成带 retryable 标记的框架异常。"""
        status = getattr(exc, "status_code", None)
        message = f"{type(exc).__name__}: {exc}"

        if status in RETRYABLE_STATUS:
            return LLMError(message, retryable=True)

        # 网络类异常通常可重试
        name = type(exc).__name__
        if any(k in name for k in ("Timeout", "Connection", "InternalServer")):
            return LLMError(message, retryable=True)

        return LLMError(message, retryable=False)


def as_dict(response: LLMResponse) -> dict[str, Any]:
    """便于落盘序列化。"""
    return {
        "model": response.model,
        "text": response.text,
        "latency_ms": response.latency_ms,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
    }
