"""数据集加载：支持 .jsonl（每行一条）与 .json（数组）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.datasets.schema import DatasetError, EvalCase

SUPPORTED_SUFFIXES = {".jsonl", ".json"}


def list_datasets(datasets_dir: Path) -> list[Path]:
    if not datasets_dir.exists():
        return []
    return sorted(
        p for p in datasets_dir.iterdir() if p.suffix in SUPPORTED_SUFFIXES and p.is_file()
    )


def load_dataset(path: Path) -> list[EvalCase]:
    """加载单个数据集文件。"""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"数据集不存在: {path}")

    text = path.read_text(encoding="utf-8")
    source = path.stem

    if path.suffix == ".jsonl":
        raws: list[dict] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raws.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path.name} 第 {lineno} 行不是合法 JSON: {exc}") from exc
    else:
        loaded = json.loads(text)
        if not isinstance(loaded, list):
            raise DatasetError(f"{path.name} 顶层结构应为数组")
        raws = loaded

    cases = [EvalCase.from_dict(item, source=source) for item in raws]
    _check_duplicate_ids(cases, source)
    return cases


def load_datasets(
    datasets_dir: Path,
    names: Iterable[str] | None = None,
    categories: list[str] | None = None,
) -> list[EvalCase]:
    """加载目录下的全部数据集，或用 names 指定若干文件名（不含后缀）。

    categories 非空时只保留类别在列表内的用例（类别名大小写不敏感）；
    过滤后为空则抛 ValueError。
    """
    available = list_datasets(datasets_dir)
    if not available:
        raise DatasetError(f"目录 {datasets_dir} 下没有找到任何 .jsonl/.json 数据集")

    if names:
        wanted = set(names)
        selected = [p for p in available if p.stem in wanted]
        missing = wanted - {p.stem for p in selected}
        if missing:
            raise DatasetError(f"数据集不存在: {sorted(missing)}；可用: {[p.stem for p in available]}")
    else:
        selected = available

    cases: list[EvalCase] = []
    for path in selected:
        cases.extend(load_dataset(path))

    if categories:
        wanted = {c.strip().lower() for c in categories if c and c.strip()}
        if wanted:
            cases = [case for case in cases if case.category.lower() in wanted]
            if not cases:
                raise ValueError(f"以下类别均无匹配用例: {', '.join(categories)}")
    return cases


def _check_duplicate_ids(cases: list[EvalCase], source: str) -> None:
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise DatasetError(f"数据集 {source} 中存在重复的用例 id: {case.id}")
        seen.add(case.id)
