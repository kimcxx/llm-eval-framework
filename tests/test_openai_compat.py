"""OpenAI 兼容客户端单元测试。

核心原则：**绝不真实调用 API**。
通过 monkeypatch 替换 openai.OpenAI 为记录参数、返回预设结果的假客户端，
覆盖参数拼装、成功路径、空响应与异常翻译（可重试/不可重试分类）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.llm.base import LLMError, LLMResponse
from src.llm.openai_compat import RETRYABLE_STATUS, OpenAICompatLLM, as_dict


def make_completion(
    text: str = "好的",
    model: str = "fake-model",
    prompt_tokens: int = 3,
    completion_tokens: int = 5,
    with_usage: bool = True,
    with_model: bool = True,
    choices: list[Any] | None = None,
) -> Any:
    """构造一个形如 SDK ChatCompletion 的对象。"""
    completion: dict[str, Any] = {
        "choices": choices if choices is not None else [SimpleNamespace(message=SimpleNamespace(content=text))],
        "usage": SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if with_usage
        else None,
    }
    if with_model:
        completion["model"] = model
    return SimpleNamespace(**completion)


class FakeTimeoutError(Exception):
    """名字里带 Timeout，用于模拟 SDK 的超时异常。"""


class FakeConnectionError(Exception):
    """名字里带 Connection，用于模拟 SDK 的网络异常。"""


class FakeInternalServerError(Exception):
    """名字里带 InternalServer，用于模拟网关 5xx。"""


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> Any:
    """把 openai.OpenAI 替换为假客户端，返回可配置的容器。"""
    holder = SimpleNamespace(
        completion=make_completion(), error=None, init_kwargs=None, calls=[]
    )

    class _Completions:
        def create(self, **kwargs: Any) -> Any:
            holder.calls.append(kwargs)
            if holder.error is not None:
                raise holder.error
            return holder.completion

    class _OpenAI:
        def __init__(self, **kwargs: Any) -> None:
            holder.init_kwargs = kwargs
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr("openai.OpenAI", _OpenAI)
    return holder


def make_llm(**overrides: Any) -> OpenAICompatLLM:
    defaults: dict[str, Any] = {"name": "compat", "model": "compat-1", "api_key": "sk-test"}
    defaults.update(overrides)
    return OpenAICompatLLM(**defaults)


# ============================== 初始化 ============================== #


class TestInit:
    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(LLMError, match="缺少 API Key"):
            OpenAICompatLLM(name="compat", model="m", api_key=None)
        with pytest.raises(LLMError, match="缺少 API Key"):
            OpenAICompatLLM(name="compat", model="m", api_key="")

    def test_sdk_options_are_forwarded(self, fake_openai) -> None:
        """构造参数必须原样传给 SDK，且 max_retries=0（重试交给框架）。"""
        make_llm(base_url="https://example.com/v1", timeout_s=12.5)

        assert fake_openai.init_kwargs == {
            "api_key": "sk-test",
            "base_url": "https://example.com/v1",
            "timeout": 12.5,
            "max_retries": 0,
        }


# ============================== 调用成功路径 ============================== #


class TestInvokeSuccess:
    def test_assembles_request_with_system(self, fake_openai) -> None:
        llm = make_llm()
        response = llm.complete("问题", system="你是助手", temperature=0.7, max_tokens=256)

        kwargs = fake_openai.calls[0]
        assert kwargs["model"] == "compat-1"
        assert kwargs["messages"] == [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "问题"},
        ]
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 256

        assert isinstance(response, LLMResponse)
        assert response.text == "好的"
        assert response.model == "fake-model"
        assert response.prompt_tokens == 3
        assert response.completion_tokens == 5

    def test_no_system_message_when_system_is_none(self, fake_openai) -> None:
        make_llm().complete("问题")
        assert fake_openai.calls[0]["messages"] == [{"role": "user", "content": "问题"}]

    def test_empty_content_becomes_empty_string(self, fake_openai) -> None:
        fake_openai.completion = make_completion(text=None)
        response = make_llm().complete("问题")
        assert response.text == ""

    def test_missing_usage_and_model_fall_back(self, fake_openai) -> None:
        """SDK 响应缺 usage / model 字段时用 0 和本地配置兜底。"""
        fake_openai.completion = make_completion(with_usage=False, with_model=False)
        response = make_llm().complete("问题")

        assert response.model == "compat-1"
        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0

    def test_zero_usage_is_kept(self, fake_openai) -> None:
        fake_openai.completion = make_completion(prompt_tokens=0, completion_tokens=0)
        response = make_llm().complete("问题")
        assert response.total_tokens == 0

    def test_empty_choices_raises_retryable(self, fake_openai) -> None:
        fake_openai.completion = make_completion(choices=[])
        with pytest.raises(LLMError, match="空 choices") as exc_info:
            make_llm().complete("问题")
        assert exc_info.value.retryable is True


# ============================== 异常翻译 ============================== #


class TestTranslate:
    @pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
    def test_retryable_status_codes(self, status: int) -> None:
        exc = Exception("服务暂不可用")
        exc.status_code = status
        error = OpenAICompatLLM._translate(exc)
        assert error.retryable is True
        assert "Exception" in str(error)

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_non_retryable_status_codes(self, status: int) -> None:
        exc = Exception("鉴权失败")
        exc.status_code = status
        assert OpenAICompatLLM._translate(exc).retryable is False

    @pytest.mark.parametrize(
        "exc",
        [FakeTimeoutError("超时"), FakeConnectionError("连接断开"), FakeInternalServerError("网关错误")],
        ids=["timeout", "connection", "internal-server"],
    )
    def test_network_like_errors_are_retryable(self, exc: Exception) -> None:
        assert OpenAICompatLLM._translate(exc).retryable is True

    def test_plain_exception_is_not_retryable(self) -> None:
        error = OpenAICompatLLM._translate(ValueError("参数错误"))
        assert error.retryable is False
        assert "ValueError" in str(error) and "参数错误" in str(error)

    def test_error_contains_exception_type(self) -> None:
        error = OpenAICompatLLM._translate(FakeTimeoutError("读超时"))
        assert "FakeTimeoutError" in str(error)


class TestInvokeErrorTranslation:
    def test_invoke_wraps_sdk_exception(self, fake_openai) -> None:
        fake_openai.error = Exception("限流")
        fake_openai.error.status_code = 429

        with pytest.raises(LLMError) as exc_info:
            make_llm().complete("问题")
        assert exc_info.value.retryable is True
        assert exc_info.value.__cause__ is fake_openai.error

    def test_invoke_non_retryable_error(self, fake_openai) -> None:
        fake_openai.error = Exception("无效 Key")
        fake_openai.error.status_code = 401

        with pytest.raises(LLMError) as exc_info:
            make_llm().complete("问题")
        assert exc_info.value.retryable is False

    def test_invoke_network_error_without_status(self, fake_openai) -> None:
        fake_openai.error = FakeTimeoutError("请求超时")
        with pytest.raises(LLMError) as exc_info:
            make_llm().complete("问题")
        assert exc_info.value.retryable is True


# ============================== 序列化 ============================== #


class TestAsDict:
    def test_serializes_response_fields(self) -> None:
        response = LLMResponse(
            text="答案",
            model="compat-1",
            latency_ms=12.5,
            prompt_tokens=7,
            completion_tokens=9,
        )
        assert as_dict(response) == {
            "model": "compat-1",
            "text": "答案",
            "latency_ms": 12.5,
            "prompt_tokens": 7,
            "completion_tokens": 9,
        }
