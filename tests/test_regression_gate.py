"""回归门禁测试。

门禁本身必须可靠：误报会阻断正常发布，漏报会放过能力退化。
这里覆盖「无变化 / 超阈值退化 / 阈值内波动 / 新模型 / 分类退化」五种场景。
"""

from __future__ import annotations

import json

import pytest

from check_regression import compare_categories, compare_overall, main


def make_report(models: dict[str, float], categories: dict[str, dict[str, float]] | None = None) -> dict:
    return {
        "summary": [{"model": name, "pass_rate": rate} for name, rate in models.items()],
        "categories": {
            model: [
                {"category": cat, "pass_rate": rate}
                for cat, rate in (categories.get(model, {}) if categories else {}).items()
            ]
            for model in models
        },
    }


class TestCompareOverall:
    def test_no_change_is_not_a_regression(self) -> None:
        report = make_report({"m": 0.8})
        rows, regressions = compare_overall(report, report, tolerance=0.02)

        assert regressions == []
        assert rows[0]["delta"] == pytest.approx(0.0)

    def test_improvement_is_not_a_regression(self) -> None:
        _, regressions = compare_overall(
            make_report({"m": 0.9}), make_report({"m": 0.8}), tolerance=0.02
        )
        assert regressions == []

    def test_drop_beyond_tolerance_is_a_regression(self) -> None:
        rows, regressions = compare_overall(
            make_report({"m": 0.60}), make_report({"m": 0.80}), tolerance=0.02
        )
        assert len(regressions) == 1
        assert rows[0]["delta"] == pytest.approx(-0.20)

    def test_drop_within_tolerance_is_ignored(self) -> None:
        _, regressions = compare_overall(
            make_report({"m": 0.79}), make_report({"m": 0.80}), tolerance=0.02
        )
        assert regressions == []

    def test_new_model_without_baseline_is_skipped(self) -> None:
        rows, regressions = compare_overall(
            make_report({"new": 0.1}), make_report({"old": 0.9}), tolerance=0.02
        )
        assert regressions == []
        assert rows[0]["delta"] is None


class TestCompareCategories:
    def test_detects_capability_specific_regression(self) -> None:
        """整体持平但安全能力崩了 —— 这正是只看整体会漏掉的场景。"""
        baseline = make_report(
            {"m": 0.8}, {"m": {"math": 0.9, "safety": 1.0, "qa": 0.5}}
        )
        current = make_report(
            {"m": 0.8}, {"m": {"math": 0.9, "safety": 0.3, "qa": 0.5}}
        )

        _, overall_regressions = compare_overall(current, baseline, tolerance=0.02)
        _, category_regressions = compare_categories(current, baseline, tolerance=0.02)

        assert overall_regressions == [], "整体通过率未变"
        assert len(category_regressions) == 1
        assert category_regressions[0]["category"] == "safety"


class TestCliExitCode:
    def _write(self, tmp_path, name: str, payload: dict) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def test_returns_zero_when_stable(self, tmp_path, capsys) -> None:
        baseline = self._write(tmp_path, "base.json", make_report({"m": 0.8}))
        current = self._write(tmp_path, "cur.json", make_report({"m": 0.81}))
        assert main(["--current", current, "--baseline", baseline]) == 0

    def test_returns_one_on_regression(self, tmp_path, capsys) -> None:
        baseline = self._write(tmp_path, "base.json", make_report({"m": 0.8}))
        current = self._write(tmp_path, "cur.json", make_report({"m": 0.5}))
        assert main(["--current", current, "--baseline", baseline]) == 1
        assert "能力退化" in capsys.readouterr().out
