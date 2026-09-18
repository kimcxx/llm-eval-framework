"""按类别过滤评测用例的测试（接口契约驱动）。

契约要点：
- ``load_datasets(datasets_dir, categories=None)`` 全量返回，向后兼容；
- ``categories`` 非空时只保留 ``EvalCase.category`` 命中的用例，多类别取并集、
  保持原有用例顺序、类别名大小写不敏感；
- 传入的类别**全部**无匹配时抛 ``ValueError``，消息中包含无效类别名；
- 部分有效部分无效时只保留有效类别的用例，不报错；
- CLI 支持 ``--categories qa_zh,math``（逗号分隔），不传时行为与现状一致。

功能落地前这些过滤测试预期失败（TypeError/AttributeError 之类），落地后应全绿。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.datasets.loader import load_datasets


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def datasets_dir(tmp_path: Path) -> Path:
    """构造一个含 3 个文件、4 种类别的临时数据集目录。

    全量加载顺序（按文件名排序 + 文件内行序）：
        qa-1, qa-2, math-1, mix-1, mix-2
    """
    _write_jsonl(
        tmp_path / "a_qa.jsonl",
        [
            {"id": "qa-1", "category": "qa_zh", "prompt": "问题一", "metrics": ["contains"]},
            {"id": "qa-2", "category": "qa_zh", "prompt": "问题二", "metrics": ["contains"]},
        ],
    )
    _write_jsonl(
        tmp_path / "b_math.jsonl",
        [{"id": "math-1", "category": "math", "prompt": "1+1=?", "metrics": ["exact_match"]}],
    )
    _write_jsonl(
        tmp_path / "c_mixed.jsonl",
        [
            {"id": "mix-1", "category": "math", "prompt": "2*3=?", "metrics": ["exact_match"]},
            {"id": "mix-2", "category": "reasoning", "prompt": "推理题", "metrics": ["contains"]},
        ],
    )
    return tmp_path


ALL_IDS = ["qa-1", "qa-2", "math-1", "mix-1", "mix-2"]


class TestCategoryFilter:
    def test_no_filter_returns_all(self, datasets_dir: Path) -> None:
        """categories=None → 全量返回，向后兼容。"""
        cases = load_datasets(datasets_dir, categories=None)
        assert [c.id for c in cases] == ALL_IDS

    def test_single_category(self, datasets_dir: Path) -> None:
        """单类别过滤 → 只剩该类。"""
        cases = load_datasets(datasets_dir, categories=["qa_zh"])
        assert [c.id for c in cases] == ["qa-1", "qa-2"]
        assert all(c.category == "qa_zh" for c in cases)

    def test_union_and_order_preserved(self, datasets_dir: Path) -> None:
        """多类别取并集，且保持原有用例顺序（而非按 categories 参数顺序）。"""
        cases = load_datasets(datasets_dir, categories=["reasoning", "math"])
        assert [c.id for c in cases] == ["math-1", "mix-1", "mix-2"]

    def test_case_insensitive(self, datasets_dir: Path) -> None:
        """类别匹配大小写不敏感："QA_ZH" 能命中 "qa_zh"。"""
        cases = load_datasets(datasets_dir, categories=["QA_ZH"])
        assert [c.id for c in cases] == ["qa-1", "qa-2"]

        # 混合大小写 + 多类别同样生效
        cases = load_datasets(datasets_dir, categories=["QA_zh", "MATH"])
        assert [c.id for c in cases] == ["qa-1", "qa-2", "math-1", "mix-1"]

    def test_all_unknown_categories_raise(self, datasets_dir: Path) -> None:
        """全部类别无匹配 → ValueError，消息包含无效类别名。"""
        with pytest.raises(ValueError, match="nonexistent") as exc_info:
            load_datasets(datasets_dir, categories=["nonexistent", "ghost"])
        assert "ghost" in str(exc_info.value)

    def test_partial_valid_keeps_valid_only(self, datasets_dir: Path) -> None:
        """部分有效部分无效 → 只保留有效类别的用例，不报错。"""
        cases = load_datasets(datasets_dir, categories=["qa_zh", "ghost"])
        assert [c.id for c in cases] == ["qa-1", "qa-2"]

    def test_backward_compatible_without_categories(self, datasets_dir: Path) -> None:
        """不传 categories 关键字 → 与现状一致（全量），旧调用方式不受影响。"""
        cases = load_datasets(datasets_dir)
        assert [c.id for c in cases] == ALL_IDS


class TestCategoryCli:
    """CLI：scripts/run_eval.py --categories qa_zh,math（逗号分隔）。"""

    @staticmethod
    def _normalize(value) -> list[str]:
        """兼容实现差异：无论 argparse 产出字符串还是（已/未）切分的列表，统一为类别列表。"""
        if value is None:
            return []
        if isinstance(value, str):
            parts = [value]
        else:
            parts = list(value)
        out: list[str] = []
        for part in parts:
            out.extend(x.strip() for x in str(part).split(",") if x.strip())
        return out

    def test_comma_separated_categories(self) -> None:
        from scripts.run_eval import parse_args

        args = parse_args(["--categories", "qa_zh,math"])
        assert self._normalize(args.categories) == ["qa_zh", "math"]

    def test_comma_separated_with_spaces(self) -> None:
        from scripts.run_eval import parse_args

        args = parse_args(["--categories", "qa_zh, math"])
        assert self._normalize(args.categories) == ["qa_zh", "math"]

    def test_no_categories_flag_keeps_default(self) -> None:
        """不传 --categories → 解析结果为空，行为与现状一致。"""
        from scripts.run_eval import parse_args

        args = parse_args([])
        assert not self._normalize(args.categories)
