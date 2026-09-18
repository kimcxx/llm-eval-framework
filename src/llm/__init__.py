"""LLM 客户端层：屏蔽不同厂商的接入差异。"""

from src.llm.base import BaseLLM, LLMError, LLMResponse
from src.llm.mock import MockLLM
from src.llm.openai_compat import OpenAICompatLLM
from src.llm.registry import build_client, build_clients

__all__ = [
    "BaseLLM",
    "LLMError",
    "LLMResponse",
    "MockLLM",
    "OpenAICompatLLM",
    "build_client",
    "build_clients",
]
