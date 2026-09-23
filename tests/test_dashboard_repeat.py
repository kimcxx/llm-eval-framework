# -*- coding: utf-8 -*-
"""看板 repeat（重复执行）展示的单测。

前端 JS 无法在 pytest 里执行，这里守住模板契约：PAGE 里确实渲染了
稳定率 / 抖动用例 / 通过次数 / attempts 抽屉，且都带「老报告降级」分支
（字段缺失显示 —、整块不渲染）。目的不是测 JS 语法，而是防止后续改模板
时把这几处悄悄删掉。

数据契约（``EvalReport.to_dict()`` 是否真的产出 stability / flaky /
pass_count / attempts）在 ``tests/test_repeat.py`` 里覆盖，那里跑的是真实
runner，不依赖看板代码。
"""
from __future__ import annotations

from dashboard.app import PAGE


# ============================== 模板契约 ============================== #


def test_summary_table_has_stability_and_flaky_columns():
    assert '<th>稳定率</th>' in PAGE
    assert '<th>抖动用例</th>' in PAGE
    # 列说明小字：稳定率 / 抖动用例的口径
    assert '稳定率 = 全部调用中通过的比例' in PAGE
    assert '抖动用例 = 同一用例重复跑结果不一致的条数' in PAGE


def test_cases_table_renders_pass_count():
    assert '<th style="width:96px">通过次数</th>' in PAGE
    assert 'repeatCell(c)' in PAGE


def test_drawer_renders_attempts():
    assert 'function attemptsSection(c)' in PAGE
    assert '重复执行（' in PAGE
    assert 'class="attempts"' in PAGE


def test_conclusion_card_gets_report_for_flaky_note():
    """结论卡片必须拿到整份报告才能统计抖动用例，只传 _conclusion 算不出来。"""
    assert 'conclusionCard(r._conclusion, r)' in PAGE
    assert 'function flakyNote(r)' in PAGE
    assert '看稳定率而非单次通过率' in PAGE


def test_new_columns_fall_back_to_dash():
    """老报告没有 stability / flaky / attempts：显示 —，不报错。"""
    assert 'stabCell' in PAGE
    assert "m.flaky == null" in PAGE
    assert 'if (!attempts.length) return' in PAGE
