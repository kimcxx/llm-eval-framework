"""配置加载（config）单元测试。

覆盖 YAML 加载的异常分支（文件不存在 / 格式错误 / 未定义模型 / 缺 name）、
.env 的手工解析降级路径、ModelConfig 的可用性判定，以及按名检索模型。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import (
    AppConfig,
    ConfigError,
    ModelConfig,
    PathSettings,
    RunSettings,
    load_config,
    load_dotenv,
)


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(content, encoding="utf-8")
    return path


VALID_YAML = """
models:
  - name: mock-baseline
    provider: mock
    model: mock-rule-based
    unknown_field: 保留到 extra
  - name: deepseek-chat
    provider: openai_compat
    api_key_env: TEST_CONFIG_KEY
    base_url: https://example.com/v1
    temperature: 0.3
    max_tokens: 512

judge_model: deepseek-chat

run:
  workers: 8
  timeout_s: 42.5
  similarity_threshold: 0.6

paths:
  datasets_dir: data
  output_dir: out
"""


# ============================== load_config ============================== #


class TestLoadConfig:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="配置文件不存在"):
            load_config(tmp_path / "nope.yaml")

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, "models: [:: 未闭合")
        with pytest.raises(yaml.YAMLError):
            load_config(path)

    def test_no_models_raises(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, "run:\n  workers: 2\n")
        with pytest.raises(ConfigError, match="未定义任何模型"):
            load_config(path)

    def test_model_without_name_raises(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, "models:\n  - provider: mock\n")
        with pytest.raises(ConfigError, match="缺少 name 字段"):
            load_config(path)

    def test_loads_full_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_CONFIG_KEY", "sk-config")
        config = load_config(write_config(tmp_path, VALID_YAML))

        assert [m.name for m in config.models] == ["mock-baseline", "deepseek-chat"]

        mock = config.models[0]
        assert mock.model == "mock-rule-based"
        assert mock.extra == {"unknown_field": "保留到 extra"}

        real = config.models[1]
        assert real.model == "deepseek-chat"  # 未写 model 时回退为 name
        assert real.temperature == 0.3
        assert real.max_tokens == 512

        assert config.judge_model == "deepseek-chat"
        assert config.run.workers == 8
        assert config.run.timeout_s == 42.5
        assert config.run.similarity_threshold == 0.6
        assert config.paths.datasets_dir.name == "data"
        assert config.paths.output_dir.name == "out"

    def test_blank_judge_becomes_none(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            "models:\n  - name: mock-baseline\n    provider: mock\njudge_model:\n",
        )
        assert load_config(path).judge_model is None

    def test_defaults_when_sections_missing(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path, "models:\n  - name: mock-baseline\n    provider: mock\n"
        )
        config = load_config(path)

        assert config.run == RunSettings()
        assert config.paths == PathSettings()


# ============================== .env 加载 ============================== #


class TestLoadDotenv:
    ENV_CONTENT = (
        "# 注释行\n"
        "\n"
        "TEST_DOTENV_PLAIN=hello\n"
        "TEST_DOTENV_QUOTED=\"引号值\"\n"
        "没有等号的行\n"
    )

    def test_missing_file_is_noop(self, tmp_path: Path) -> None:
        load_dotenv(tmp_path / ".env")  # 不应抛错

    def test_falls_back_to_manual_parsing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """python-dotenv 不可用时，退化为手工解析并剥掉引号。"""
        env_file = tmp_path / ".env"
        env_file.write_text(self.ENV_CONTENT, encoding="utf-8")

        monkeypatch.delenv("TEST_DOTENV_PLAIN", raising=False)
        monkeypatch.delenv("TEST_DOTENV_QUOTED", raising=False)
        monkeypatch.setitem(sys.modules, "dotenv", None)  # 强制 import 失败

        load_dotenv(env_file)

        assert os.environ["TEST_DOTENV_PLAIN"] == "hello"
        assert os.environ["TEST_DOTENV_QUOTED"] == "引号值"

    def test_does_not_override_existing_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_DOTENV_EXISTING=from-file\n", encoding="utf-8")

        monkeypatch.setenv("TEST_DOTENV_EXISTING", "from-env")
        monkeypatch.setitem(sys.modules, "dotenv", None)

        load_dotenv(env_file)

        assert os.environ["TEST_DOTENV_EXISTING"] == "from-env"


# ============================== ModelConfig ============================== #


class TestModelConfig:
    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = ModelConfig(name="m", api_key_env="TEST_KEY_A")
        monkeypatch.setenv("TEST_KEY_A", "  sk-abc  ")
        assert cfg.api_key == "sk-abc"  # 前后空白应被剥掉

    def test_api_key_blank_or_unset_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = ModelConfig(name="m", api_key_env="TEST_KEY_B")
        monkeypatch.setenv("TEST_KEY_B", "   ")
        assert cfg.api_key is None

        monkeypatch.delenv("TEST_KEY_B", raising=False)
        assert cfg.api_key is None

    def test_no_api_key_env_is_none(self) -> None:
        assert ModelConfig(name="m").api_key is None

    def test_mock_is_always_available(self) -> None:
        assert ModelConfig(name="m", provider="mock").is_available is True

    def test_real_model_availability_depends_on_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = ModelConfig(name="m", provider="openai_compat", api_key_env="TEST_KEY_C")

        monkeypatch.setenv("TEST_KEY_C", "sk-x")
        assert cfg.is_available is True

        monkeypatch.delenv("TEST_KEY_C", raising=False)
        assert cfg.is_available is False
        assert cfg.unavailable_reason == "环境变量 TEST_KEY_C 未设置"

    def test_from_dict_defaults(self) -> None:
        cfg = ModelConfig.from_dict({"name": "only-name"})
        assert cfg.provider == "openai_compat"
        assert cfg.model == "only-name"
        assert cfg.temperature == 0.0


# ============================== 模型检索 ============================== #


def sample_config() -> AppConfig:
    return AppConfig(
        models=[
            ModelConfig(name="mock-baseline", provider="mock"),
            ModelConfig(name="real", provider="openai_compat", api_key_env="NOPE"),
        ],
        judge_model=None,
        run=RunSettings(),
        paths=PathSettings(),
    )


class TestModelLookup:
    def test_get_model(self) -> None:
        assert sample_config().get_model("real").name == "real"

    def test_get_model_unknown_raises(self) -> None:
        with pytest.raises(ConfigError, match="配置中不存在模型"):
            sample_config().get_model("ghost")

    def test_select_all_when_no_names(self) -> None:
        assert len(sample_config().select_models(None)) == 2
        assert len(sample_config().select_models([])) == 2

    def test_select_subset(self) -> None:
        picked = sample_config().select_models(["mock-baseline"])
        assert [m.name for m in picked] == ["mock-baseline"]

    def test_select_unknown_name_raises(self) -> None:
        with pytest.raises(ConfigError, match="配置中不存在模型"):
            sample_config().select_models(["ghost"])

    def test_get_judge_returns_none_when_unset(self) -> None:
        assert sample_config().get_judge() is None

    def test_get_judge_resolves_configured_model(self) -> None:
        config = sample_config()
        config.judge_model = "real"
        assert config.get_judge().name == "real"

    def test_get_judge_returns_none_when_broken_reference(self) -> None:
        """裁判模型配置错了应返回 None（跳过 judge），而不是让框架崩掉。"""
        config = sample_config()
        config.judge_model = "ghost"
        assert config.get_judge() is None


# ============================== from_dict 容错 ============================== #


class TestSettingsFromDict:
    def test_run_settings_none_uses_defaults(self) -> None:
        assert RunSettings.from_dict(None) == RunSettings()

    def test_run_settings_partial_override(self) -> None:
        settings = RunSettings.from_dict({"workers": 16})
        assert settings.workers == 16
        assert settings.max_retries == 3

    def test_path_settings_defaults(self) -> None:
        paths = PathSettings.from_dict(None)
        assert paths.datasets_dir.name == "datasets"
        assert paths.output_dir.name == "reports"

    def test_path_settings_custom(self, tmp_path: Path) -> None:
        paths = PathSettings.from_dict({"datasets_dir": "d", "output_dir": "o"})
        assert paths.datasets_dir.name == "d"
        assert paths.output_dir.name == "o"
