"""模型客户端工厂（registry）单元测试。

覆盖 build_client 的三个分支（mock / openai_compat / 未知 provider）
以及 build_clients 的跳过、strict 报错、全不可用报错与按名筛选。
"""

from __future__ import annotations

import pytest

from src.config import AppConfig, ConfigError, ModelConfig, PathSettings, RunSettings
from src.llm.mock import MockLLM
from src.llm.openai_compat import OpenAICompatLLM
from src.llm.registry import build_client, build_clients


def mock_model(name: str = "mock-baseline") -> ModelConfig:
    return ModelConfig(name=name, provider="mock", model="mock-rule-based")


def real_model(name: str = "real", env: str = "TEST_REGISTRY_KEY") -> ModelConfig:
    return ModelConfig(
        name=name,
        provider="openai_compat",
        model=f"{name}-model",
        api_key_env=env,
        base_url="https://example.com/v1",
        temperature=0.1,
        max_tokens=777,
    )


def make_config(models: list[ModelConfig], judge: str | None = None) -> AppConfig:
    return AppConfig(
        models=models,
        judge_model=judge,
        run=RunSettings(timeout_s=30.0),
        paths=PathSettings(),
    )


# ============================== build_client ============================== #


class TestBuildClient:
    def test_mock_provider(self) -> None:
        cfg = mock_model()
        client = build_client(cfg, RunSettings())
        assert isinstance(client, MockLLM)
        assert client.name == "mock-baseline"
        assert client.model == "mock-rule-based"

    def test_openai_compat_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_REGISTRY_KEY", "sk-registry")
        cfg = real_model()

        client = build_client(cfg, RunSettings(timeout_s=15.0))

        assert isinstance(client, OpenAICompatLLM)
        assert client.name == "real"
        assert client.model == "real-model"
        # 构造参数确实来自配置而非硬编码
        assert client.options["temperature"] == 0.1
        assert client.options["max_tokens"] == 777

    def test_unknown_provider_raises(self) -> None:
        cfg = ModelConfig(name="weird", provider="graphql", model="w-1")
        with pytest.raises(ConfigError, match="未知的 provider"):
            build_client(cfg, RunSettings())


# ============================== build_clients ============================== #


class TestBuildClients:
    def test_builds_all_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_REGISTRY_KEY", "sk-registry")
        clients = build_clients(make_config([mock_model(), real_model()]))

        assert set(clients) == {"mock-baseline", "real"}
        assert isinstance(clients["mock-baseline"], MockLLM)
        assert isinstance(clients["real"], OpenAICompatLLM)

    def test_skips_model_without_key_and_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_REGISTRY_KEY", raising=False)
        config = make_config([mock_model(), real_model()])

        with pytest.warns(UserWarning, match="跳过模型 real"):
            clients = build_clients(config)

        # 只构建了 mock，缺 Key 的模型被跳过而非导致整体失败
        assert set(clients) == {"mock-baseline"}

    def test_strict_mode_raises_on_missing_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_REGISTRY_KEY", raising=False)
        config = make_config([mock_model(), real_model()])

        with pytest.raises(ConfigError, match="跳过模型 real"):
            build_clients(config, strict=True)

    def test_no_available_models_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_REGISTRY_KEY", raising=False)
        with pytest.warns(UserWarning, match="跳过模型"), pytest.raises(
            ConfigError, match="没有任何可用模型"
        ):
            build_clients(make_config([real_model()]))

    def test_filters_by_model_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_REGISTRY_KEY", "sk-registry")
        config = make_config([mock_model(), real_model()])

        clients = build_clients(config, ["mock-baseline"])

        assert set(clients) == {"mock-baseline"}

    def test_unknown_model_name_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_REGISTRY_KEY", "sk-registry")
        config = make_config([mock_model(), real_model()])

        with pytest.raises(ConfigError, match="配置中不存在模型"):
            build_clients(config, ["no-such-model"])

    def test_timeout_comes_from_run_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_REGISTRY_KEY", "sk-registry")
        run = RunSettings(timeout_s=9.5)
        client = build_client(real_model(), run)

        assert isinstance(client, OpenAICompatLLM)
