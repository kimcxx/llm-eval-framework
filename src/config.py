"""配置加载：把 configs/models.yaml + 环境变量聚合成强类型对象。

设计意图：评测框架必须做到「一次配置、多处复用」，
所以模型参数、阈值、并发度全部外置到 YAML，代码里不出现魔法数字。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"


class ConfigError(RuntimeError):
    """配置非法时抛出。"""


def load_dotenv(env_file: Path | None = None) -> None:
    """加载 .env。优先用 python-dotenv，缺失时退化为手工解析，保证零依赖可跑。"""
    path = env_file or (PROJECT_ROOT / ".env")
    if not path.exists():
        return

    try:
        from dotenv import load_dotenv as _load  # type: ignore

        _load(path, override=False)
        return
    except ImportError:
        pass

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass
class ModelConfig:
    """单个被测模型的配置。"""

    name: str
    provider: str = "openai_compat"
    model: str = ""
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    price_per_1k_input: float = 0.0
    price_per_1k_output: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        value = os.getenv(self.api_key_env, "").strip()
        return value or None

    @property
    def is_available(self) -> bool:
        """Mock 模型永远可用；真实模型必须有 Key。"""
        if self.provider == "mock":
            return True
        return bool(self.api_key)

    @property
    def unavailable_reason(self) -> str:
        return f"环境变量 {self.api_key_env} 未设置"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        if "name" not in raw:
            raise ConfigError(f"模型配置缺少 name 字段: {raw}")
        known = {
            "name",
            "provider",
            "model",
            "api_key_env",
            "base_url",
            "temperature",
            "max_tokens",
            "price_per_1k_input",
            "price_per_1k_output",
        }
        payload = {k: v for k, v in raw.items() if k in known}
        payload.setdefault("model", payload["name"])
        extra = {k: v for k, v in raw.items() if k not in known}
        return cls(**payload, extra=extra)


@dataclass
class RunSettings:
    """评测运行参数。"""

    workers: int = 4
    max_retries: int = 3
    timeout_s: float = 60.0
    similarity_threshold: float = 0.75
    judge_threshold: float = 4.0
    # schema_match 的通过阈值：score（匹配字段比例）不低于该值才算通过。
    # 默认 1.0 = 所有必需字段都必须正确，与旧 json_valid 的口径一致。
    schema_match_threshold: float = 1.0
    # 这些分类只由 LLM 裁判判定：同类用例里的其它指标（如 similarity）
    # 仍然计算并写进报告明细，但不参与通过/失败判定（仅作记录）。
    # 背景：开放式问答没有标准答案，字面相似度会把「换个说法但答对了」误判为失败。
    judge_only_categories: tuple[str, ...] = ("qa_open",)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RunSettings:
        raw = dict(raw or {})
        categories = raw.pop("judge_only_categories", None)
        if categories is None:
            return cls(**raw)
        if isinstance(categories, str):
            categories = [categories]
        normalized = tuple(str(item).strip() for item in categories if str(item).strip())
        return cls(**raw, judge_only_categories=normalized)


@dataclass
class PathSettings:
    datasets_dir: Path = PROJECT_ROOT / "datasets"
    output_dir: Path = PROJECT_ROOT / "reports"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PathSettings:
        raw = raw or {}
        datasets = PROJECT_ROOT / raw.get("datasets_dir", "datasets")
        output = PROJECT_ROOT / raw.get("output_dir", "reports")
        return cls(datasets_dir=datasets, output_dir=output)


@dataclass
class AppConfig:
    """整个评测框架的运行时配置。"""

    models: list[ModelConfig]
    judge_model: str | None
    run: RunSettings
    paths: PathSettings

    def get_model(self, name: str) -> ModelConfig:
        for model in self.models:
            if model.name == name:
                return model
        raise ConfigError(f"配置中不存在模型: {name}")

    def select_models(self, names: list[str] | None = None) -> list[ModelConfig]:
        if not names:
            return list(self.models)
        return [self.get_model(n) for n in names]

    def get_judge(self) -> ModelConfig | None:
        if not self.judge_model:
            return None
        try:
            return self.get_model(self.judge_model)
        except ConfigError:
            return None


def load_config(path: str | Path | None = None) -> AppConfig:
    """读取 YAML 配置并加载 .env。"""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"配置文件不存在: {config_path}")

    load_dotenv()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    models_raw = raw.get("models") or []
    if not models_raw:
        raise ConfigError(f"{config_path} 中未定义任何模型")

    return AppConfig(
        models=[ModelConfig.from_dict(item) for item in models_raw],
        judge_model=raw.get("judge_model") or None,
        run=RunSettings.from_dict(raw.get("run")),
        paths=PathSettings.from_dict(raw.get("paths")),
    )
