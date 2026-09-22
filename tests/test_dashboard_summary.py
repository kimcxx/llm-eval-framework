# -*- coding: utf-8 -*-
"""看板「详情页汇总 tab」改造的单测。

覆盖的是后端新增的两个纯函数（前端 JS 只负责渲染，逻辑不重复实现）：

    * ``metric_label``：指标名 → 中文标签（json_valid=格式校验 等）；
      未收录的指标原样返回，保证新增指标不会显示成空白。
    * ``compute_conclusion``：汇总 tab 顶部「本报告结论」
      - ``rank``：按通过率降序（相同保持原顺序）
      - ``judge``：从 notes 解析「裁判模型 xxx」；无则 None
      - ``anomalies``：跳过用例 / 分类明显偏低 / 未启用裁判，最多 4 条
      - ``text``：拼好的一句话结论

另外校验：
    * ``METRIC_LABELS`` 会被注入到 PAGE 模板（模板里不能残留 ``__METRIC_LABELS__``）
    * ``compute_conclusion`` 对 None / 空 dict 不抛异常（老报告兼容）
"""
from __future__ import annotations

import pytest

from dashboard.app import (
    METRIC_LABELS,
    PAGE,
    _parse_judge_model,
    compute_conclusion,
    metric_label,
)


# ============================== metric_label ============================== #

def test_known_metrics_have_chinese_labels():
    assert metric_label('json_valid') == '格式校验'
    assert metric_label('judge') == 'LLM 裁判'
    assert metric_label('similarity') == '语义相似度'


@pytest.mark.parametrize('name', ['exact_match', 'contains', 'not_contains'])
def test_all_metric_names_are_labeled(name):
    """框架实际会输出的指标名必须都有中文标签，否则抽屉里会露出英文原名。"""
    assert name in METRIC_LABELS
    assert metric_label(name) != name


def test_unknown_metric_falls_back_to_raw_name():
    assert metric_label('brand_new_metric') == 'brand_new_metric'


@pytest.mark.parametrize('bad', [None, 123, [], {}])
def test_non_string_metric_is_returned_as_is(bad):
    assert metric_label(bad) == bad


# ============================== _parse_judge_model ============================== #

def test_parse_judge_model_from_notes():
    report = {'notes': ['similarity(字符后端)', '裁判模型 glm-4.5-air（通过阈值 4/5）']}
    assert _parse_judge_model(report) == 'glm-4.5-air'


def test_parse_judge_model_accepts_half_width_bracket():
    report = {'notes': ['裁判模型 deepseek-chat(通过阈值 4/5)']}
    assert _parse_judge_model(report) == 'deepseek-chat'


@pytest.mark.parametrize('report', [{}, {'notes': []}, {'notes': ['没有裁判']}, None, 'x'])
def test_parse_judge_model_returns_none_when_absent(report):
    assert _parse_judge_model(report) is None


# ============================== compute_conclusion ============================== #

def _report(summary=None, categories=None, notes=None):
    return {
        'summary': summary or [],
        'categories': categories or {},
        'notes': notes or [],
    }


def test_rank_is_sorted_by_pass_rate_desc():
    r = _report(summary=[
        {'model': 'mock-baseline', 'pass_rate': 0.347, 'skipped': 0},
        {'model': 'deepseek-pro', 'pass_rate': 0.908, 'skipped': 0},
        {'model': 'deepseek-chat', 'pass_rate': 0.898, 'skipped': 0},
    ], notes=['裁判模型 glm-4.5-air（通过阈值 4/5）'])
    c = compute_conclusion(r)
    assert [x['model'] for x in c['rank']] == ['deepseek-pro', 'deepseek-chat']
    assert c['judge'] == 'glm-4.5-air'
    assert 'deepseek-pro 90.8% > deepseek-chat 89.8%' in c['text']
    assert 'mock-baseline' not in c['text']


def test_rank_excludes_mock_control_group():
    """通过率排名里排除 mock-*：它是阴性对照，不是被测模型。"""
    r = _report(summary=[
        {'model': 'mock-baseline', 'pass_rate': 0.347, 'skipped': 0},
        {'model': 'deepseek-pro', 'pass_rate': 0.908, 'skipped': 0},
    ], notes=['裁判模型 glm-4.5-air（通过阈值 4/5）'])
    c = compute_conclusion(r)
    assert [x['model'] for x in c['rank']] == ['deepseek-pro']
    assert 'mock-baseline' not in c['text']


def test_rank_keeps_mock_when_all_mock():
    """全是 mock 的报告（CD 冒烟）：排名不能为空，退回所有模型。"""
    r = _report(summary=[
        {'model': 'mock-baseline', 'pass_rate': 0.347, 'skipped': 0},
        {'model': 'mock-alt', 'pass_rate': 0.5, 'skipped': 0},
    ], notes=['裁判模型 glm-4.5-air（通过阈值 4/5）'])
    c = compute_conclusion(r)
    assert [x['model'] for x in c['rank']] == ['mock-alt', 'mock-baseline']


def test_text_contains_judge_and_no_anomaly_when_clean():
    r = _report(summary=[{'model': 'm1', 'pass_rate': 0.9, 'skipped': 0}],
                notes=['裁判模型 glm-4.5-air（通过阈值 4/5）'])
    c = compute_conclusion(r)
    assert '裁判模型：glm-4.5-air' in c['text']
    assert '注意' not in c['text']
    assert c['anomalies'] == []


def test_anomaly_when_cases_skipped():
    r = _report(summary=[{'model': 'deepseek-pro', 'pass_rate': 0.88, 'skipped': 5}],
                notes=['裁判模型 glm-4.5-air（通过阈值 4/5）'])
    c = compute_conclusion(r)
    assert 'deepseek-pro 有 5 条用例跳过' in c['text']


def test_anomaly_when_category_obviously_low():
    r = _report(
        summary=[{'model': 'm1', 'pass_rate': 0.8, 'skipped': 0}],
        categories={'m1': [
            {'category': 'json_extract', 'total': 18, 'pass_rate': 0.75},
            {'category': 'qa_zh', 'total': 22, 'pass_rate': 0.18},
        ]},
        notes=['裁判模型 glm-4.5-air（通过阈值 4/5）'],
    )
    c = compute_conclusion(r)
    assert any('分类 qa_zh 明显偏低' in a for a in c['anomalies'])
    assert not any('json_extract' in a for a in c['anomalies'])


def test_mock_control_group_is_not_reported_as_anomaly():
    """mock-* 是对照组，本来就差；只有真实模型偏低才值得报警。"""
    r = _report(
        summary=[
            {'model': 'mock-baseline', 'pass_rate': 0.347, 'skipped': 0},
            {'model': 'deepseek-chat', 'pass_rate': 0.898, 'skipped': 0},
        ],
        categories={
            'mock-baseline': [{'category': 'json_extract', 'total': 18, 'pass_rate': 0.0}],
            'deepseek-chat': [{'category': 'json_extract', 'total': 18, 'pass_rate': 0.72}],
        },
        notes=['裁判模型 glm-4.5-air（通过阈值 4/5）'],
    )
    assert compute_conclusion(r)['anomalies'] == []


def test_all_mock_report_still_reports_low_category():
    """全是 mock 的报告（CD 冒烟）没有真实模型可参照，退回所有模型。"""
    r = _report(
        summary=[{'model': 'mock-baseline', 'pass_rate': 0.347, 'skipped': 0}],
        categories={'mock-baseline': [{'category': 'qa_zh', 'total': 22, 'pass_rate': 0.0}]},
        notes=['裁判模型 glm-4.5-air（通过阈值 4/5）'],
    )
    assert any('分类 qa_zh 明显偏低' in a for a in compute_conclusion(r)['anomalies'])


def test_tiny_category_sample_is_not_an_anomaly():
    """样本量 < 3 的分类不报异常：1~2 条失败没有统计意义，只会制造噪音。"""
    r = _report(
        summary=[{'model': 'm1', 'pass_rate': 0.8, 'skipped': 0}],
        categories={'m1': [{'category': 'qa_zh', 'total': 2, 'pass_rate': 0.0}]},
        notes=['裁判模型 glm-4.5-air（通过阈值 4/5）'],
    )
    assert compute_conclusion(r)['anomalies'] == []


def test_anomaly_when_judge_missing():
    r = _report(summary=[{'model': 'm1', 'pass_rate': 0.8, 'skipped': 0}])
    c = compute_conclusion(r)
    assert c['judge'] is None
    assert any('未启用裁判模型' in a for a in c['anomalies'])


def test_anomalies_are_capped_at_four():
    summary = [{'model': f'm{i}', 'pass_rate': 0.5, 'skipped': 1} for i in range(6)]
    r = _report(
        summary=summary,
        categories={f'm{i}': [{'category': f'c{i}', 'total': 5, 'pass_rate': 0.1}] for i in range(6)},
    )
    assert len(compute_conclusion(r)['anomalies']) <= 4


@pytest.mark.parametrize('bad', [None, {}, [], 'not a dict'])
def test_compute_conclusion_never_raises(bad):
    c = compute_conclusion(bad)
    assert isinstance(c, dict)
    assert 'text' in c


def test_empty_report_text_is_graceful():
    assert compute_conclusion({})['text'] == '通过率排名：无模型汇总数据；裁判模型：未启用；注意：本次未启用裁判模型（judge 指标未生效）'


# ============================== 模板注入 ============================== #

def test_metric_labels_injected_into_page():
    """模板占位符必须被真实映射替换，否则前端拿到的是无效 JS。"""
    assert '__METRIC_LABELS__' not in PAGE
    assert '"json_valid": "格式校验"' in PAGE or "'json_valid': '格式校验'" in PAGE
