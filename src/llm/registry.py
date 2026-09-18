"""模型客户端工厂：按配置装配客户端，并处理「缺 Key」的降级。"""

from __future__ import annotations

import warnings

from src.config import AppConfig, ConfigError, ModelConfig, RunSettings
from src.llm.base import BaseLLM
from src.llm.mock import MockLLM
from src.llm.openai_compat import OpenAICompatLLM


def build_client(cfg: ModelConfig, run: RunSettings) -> BaseLLM:
    """根据单条模型配置构建客户端。"""
    if cfg.provider == "mock":
        return MockLLM(
            name=cfg.name,
            model=cfg.model,
            temperature=0.0,
            max_tokens=cfg.max_tokens,
        )

    if cfg.provider == "openai_compat":
        return OpenAICompatLLM(
            name=cfg.name,
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_s=run.timeout_s,
        )

    raise ConfigError(f"未知的 provider: {cfg.provider}（模型 {cfg.name}）")


def build_clients(
    config: AppConfig,
    model_names: list[str] | None = None,
    *,
    strict: bool = False,
) -> dict[str, BaseLLM]:
    """批量构建客户端。

    strict=False 时，缺 Key 的模型会被跳过并给出警告，
    这样「只想跑 mock」的场景不会被未配置的模型卡住。
    """
    clients: dict[str, BaseLLM] = {}

    for cfg in config.select_models(model_names):
        if not cfg.is_available:
            message = f"跳过模型 {cfg.name}：{cfg.unavailable_reason}"
            if strict:
                raise ConfigError(message)
            warnings.warn(message, stacklevel=2)
            continue
        clients[cfg.name] = build_client(cfg, config.run)

    if not clients:
        raise ConfigError(
            "没有任何可用模型。请在 .env 中配置 API Key，"
            "或指定 mock 模型（--models mock-baseline）先验证框架。"
        )
    return clients
