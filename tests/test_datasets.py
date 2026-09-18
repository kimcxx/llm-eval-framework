"""数据集加载与校验测试。

数据集是评测的「标尺」，标尺本身出错会让所有结论失效，
因此这里重点验证：格式错误能被及时发现、id 不重复、字段映射正确。
"""

from __future__ import annotations

import json

import pytest

from src.datasets.loader import list_datasets, load_dataset, load_datasets
from src.datasets.schema import DatasetError, EvalCase


class TestSchema:
    def test_required_fields(self) -> None:
        with pytest.raises(DatasetError, match="必填字段"):
            EvalCase.from_dict({"prompt": "缺少 id"})

    def test_auto_keywords_from_expected(self) -> None:
        case = EvalCase.from_dict({"id": "a", "prompt": "q", "expected": "北京"})
        assert case.keywords == ["北京"]

    def test_extra_fields_go_to_meta(self) -> None:
        case = EvalCase.from_dict({"id": "a", "prompt": "q", "forbidden": ["X"]})
        assert case.meta["forbidden"] == ["X"]


class TestLoader:
    def test_loads_datasets_from_jsonl(self, tmp_path) -> None:
        path = tmp_path / "demo.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"id": "a", "prompt": "q1", "metrics": ["contains"]}, ensure_ascii=False),
                    "",
                    json.dumps({"id": "b", "prompt": "q2", "metrics": ["contains"]}, ensure_ascii=False),
                ]
            ),
            encoding="utf-8",
        )
        cases = load_dataset(path)
        assert [c.id for c in cases] == ["a", "b"]
        assert cases[0].source == "demo"

    def test_loads_datasets_from_json_array(self, tmp_path) -> None:
        path = tmp_path / "demo.json"
        path.write_text(
            json.dumps([{"id": "a", "prompt": "q"}], ensure_ascii=False), encoding="utf-8"
        )
        assert len(load_dataset(path)) == 1

    def test_reports_line_number_on_bad_json(self, tmp_path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "a", "prompt": "q"}\n{不是 JSON}\n', encoding="utf-8")
        with pytest.raises(DatasetError, match="第 2 行"):
            load_dataset(path)

    def test_rejects_duplicate_ids(self, tmp_path) -> None:
        path = tmp_path / "dup.jsonl"
        path.write_text(
            '{"id": "a", "prompt": "q"}\n{"id": "a", "prompt": "q2"}\n', encoding="utf-8"
        )
        with pytest.raises(DatasetError, match="重复"):
            load_dataset(path)

    def test_missing_dataset_name_lists_available(self, tmp_path) -> None:
        (tmp_path / "one.jsonl").write_text('{"id": "a", "prompt": "q"}\n', encoding="utf-8")
        with pytest.raises(DatasetError, match="数据集不存在"):
            load_datasets(tmp_path, ["two"])


class TestShippedDatasets:
    """确保仓库自带的数据集始终合法。"""

    def test_all_bundled_datasets_load(self, project_root) -> None:
        paths = list_datasets(project_root / "datasets")
        assert len(paths) >= 5

        cases = load_datasets(project_root / "datasets")
        assert len(cases) >= 60
        assert len({c.id for c in cases}) == len(cases)

    def test_every_case_declares_known_metrics(self, project_root) -> None:
        from src.metrics import SUPPORTED_METRICS

        cases = load_datasets(project_root / "datasets")
        for case in cases:
            assert case.metrics, f"{case.id} 未声明任何指标"
            unknown = set(case.metrics) - set(SUPPORTED_METRICS)
            assert not unknown, f"{case.id} 声明了未知指标 {unknown}"

    def test_metric_declaration_matches_case_data(self, project_root) -> None:
        """指标与数据必须自洽，否则会出现大量无意义的「跳过」。"""
        cases = load_datasets(project_root / "datasets")
        for case in cases:
            if "json_valid" in case.metrics:
                assert case.required_keys or isinstance(case.expected, dict), (
                    f"{case.id} 使用 json_valid 但未提供 required_keys/expected"
                )
            if "not_contains" in case.metrics:
                assert case.meta.get("forbidden"), f"{case.id} 使用 not_contains 但未提供 forbidden"
            if "contains" in case.metrics:
                assert case.keywords, f"{case.id} 使用 contains 但未提供 keywords"
            if "exact_match" in case.metrics:
                assert case.expected is not None, f"{case.id} 使用 exact_match 但未提供 expected"
