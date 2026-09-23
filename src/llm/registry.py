"""模型客户端工厂：按配置装配客户端，并处理「缺 Key」的降级。"""

from __future__ import annotations

import warnings

from src.config import AppConfig, ConfigError, ModelConfig, RunSettings
from src.llm.base import BaseLLM
from src.llm.mock import MockLLM
from src.llm.openai_compat import OpenAICompatLLM
from src.llm.rate_limit import RateLimiter


def build_client(cfg: ModelConfig, run: RunSettings, limiter: RateLimiter | None = None) -> BaseLLM:
    """根据单条模型配置构建客户端。

    limiter 省略时按 run.request_interval_s 自建一个——单客户端场景（如裁判）
    与批量场景行为一致。
    """
    if cfg.provider == "mock":
        client: BaseLLM = MockLLM(
            name=cfg.name,
            model=cfg.model,
            temperature=0.0,
            max_tokens=cfg.max_tokens,
        )
    elif cfg.provider == "openai_compat":
        client = OpenAICompatLLM(
            name=cfg.name,
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_s=run.timeout_s,
        )
    else:
        raise ConfigError(f"未知的 provider: {cfg.provider}（模型 {cfg.name}）")

    client.limiter = limiter if limiter is not None else RateLimiter(run.request_interval_s)
    return client


def build_clients(
    config: AppConfig,
    model_names: list[str] | None = None,
    *,
    strict: bool = False,
    limiter: RateLimiter | None = None,
) -> dict[str, BaseLLM]:
    """批量构建客户端。

    strict=False 时，缺 Key 的模型会被跳过并给出警告，
    这样「只想跑 mock」的场景不会被未配置的模型卡住。

    所有被测客户端共享同一个 limiter：限速要按「对供应商的总压力」算，
    而不是每个模型各限一份，否则模型一多就又把 QPS 翻上去了。
    """
    clients: dict[str, BaseLLM] = {}
    shared = limiter if limiter is not None else RateLimiter(config.run.request_interval_s)

    for cfg in config.select_models(model_names):
        if not cfg.is_available:
            message = f"跳过模型 {cfg.name}：{cfg.unavailable_reason}"
            if strict:
                raise ConfigError(message)
            warnings.warn(message, stacklevel=2)
            continue
        clients[cfg.name] = build_client(cfg, config.run, limiter=shared)

    if not clients:
        raise ConfigError(
            "没有任何可用模型。请在 .env 中配置 API Key，"
            "或指定 mock 模型（--models mock-baseline）先验证框架。"
        )
    return clients
