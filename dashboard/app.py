# -*- coding: utf-8 -*-
"""评测实验室看板 —— 零依赖（仅 Python 标准库）。

用法：
    python dashboard/app.py            # 默认 0.0.0.0:8080
    PORT=9000 python dashboard/app.py  # 自定义端口

数据源：../reports/ 目录下 report-*.json（runner 产物）。
"""
import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
# 优先使用看板自带数据（部署沙盒用），本地开发则回退到项目 reports/
REPORTS_DIR = Path(__file__).resolve().parent / "data"
if not REPORTS_DIR.is_dir():
    REPORTS_DIR = ROOT / "reports"
PORT = int(os.environ.get("PORT", "8080"))

# Agent 评测（agent_eval/）的数据位置——只读，不写、不改。
# 之所以另外扫一份目录而不是并进 reports/：评测报告的「通过率」是模型答对率，
# Agent 评测的「通过率」是 agent 有没有调对工具，两个通过率混进一张表会互相污染。
AGENT_EVAL_DIR = ROOT / "agent_eval"
AGENT_RESULTS_DIR = AGENT_EVAL_DIR / "results"
AGENT_ARCHIVE_DIR = AGENT_RESULTS_DIR / "archive"
AGENT_REPORTS_DIR = AGENT_EVAL_DIR / "reports"
# 文件名里带这个标记的是作废存档（见 agent_eval/results/README.md）：
# 它是事故留档，可以在仓库里查，但**绝不能出现在看板上**——显示了就等于
# 把「通过了自己出的题」的成绩又摆出来一遍。
AGENT_INVALID_MARK = "-INVALID-"

# ============================== 模块顶层 helper（供 tester 单测 / Python 调用） ============================== #


def case_status(c):
    """单条用例状态：'pass' / 'fail' / 'skip'。

    判定规则（与前端 JS `statusOf` 保持一致）：
    - c.error 非空 → 'skip'
    - c.passed 为 True → 'pass'
    - 其余 → 'fail'

    参数 c 为 dict（来自 report.json 的 cases[]），缺字段安全降级。
    """
    if not isinstance(c, dict):
        return 'fail'
    if c.get('error'):
        return 'skip'
    if c.get('passed') is True:
        return 'pass'
    return 'fail'


def case_key(c):
    """一条 case 记录的唯一键：``case_id`` + ``model``（用 ``\\x1f`` 分隔）。

    多模型报告里同一个 ``case_id`` 会出现 N 次（每个模型一条），只用
    ``case_id`` 当键会让后写的覆盖先写的：列表行明明是 A 模型那条，
    点开抽屉拿到的却是 B 模型那条（状态可能相反）。

    所以凡是「case → 单条记录」的映射（抽屉查找、跨次对比聚合）
    都必须用这个复合键；单模型报告下它与 ``case_id`` 等价。
    """
    if not isinstance(c, dict):
        return ""
    cid = c.get("case_id")
    model = c.get("model")
    return f"{'' if cid is None else cid}\x1f{'' if model is None else model}"


def diff_cases(curr_cases, base_cases):
    """跨次对比：按用例唯一键（case_id + 模型）排序后比较 passed 状态变化。

    返回 dict：
    - regressed: 上次通过 → 这次失败（list[{curr, base}]）
    - fixed:     上次失败 → 这次通过（list[{curr, base}]）
    - stillFailing: 两次都失败（list[{curr, base}]）
    - onlyInCurr:   当前报告独有（list[curr]）
    - onlyInBase:   基线报告独有（list[base]）

    入参允许 None / list；空入参返回全空分组，绝不抛错。
    """
    curr = list(curr_cases or [])
    base = list(base_cases or [])
    curr_map = {case_key(c): c for c in curr if isinstance(c, dict) and c.get("case_id") is not None}
    base_map = {case_key(c): c for c in base if isinstance(c, dict) and c.get("case_id") is not None}
    all_keys = sorted(set(curr_map) | set(base_map))

    regressed, fixed, still_failing = [], [], []
    only_in_curr, only_in_base = [], []
    for key in all_keys:
        cur = curr_map.get(key)
        b = base_map.get(key)
        if cur is not None and b is None:
            only_in_curr.append(cur)
            continue
        if b is not None and cur is None:
            only_in_base.append(b)
            continue
        cs, bs = case_status(cur), case_status(b)
        if bs == "pass" and cs != "pass":
            regressed.append({"curr": cur, "base": b})
        elif bs != "pass" and cs == "pass":
            fixed.append({"curr": cur, "base": b})
        elif cs != "pass":
            still_failing.append({"curr": cur, "base": b})
    return {
        "regressed": regressed,
        "fixed": fixed,
        "stillFailing": still_failing,
        "onlyInCurr": only_in_curr,
        "onlyInBase": only_in_base,
    }


# ============================== 顶部 banner 字段计算 ============================== #
# 首页最顶部一行大字结论：「最新报告 <model> XX.X%，较上次 +/-X.X pt；
# 回归 n 条 / 修复 m 条；最弱维度：<label> YY.Y%」。
# 数据全部来自 api/reports.json + 已有的 diff_cases 逻辑，不加新接口。

_BANNER_DIM_LABELS = {
    'correctness': '准确性', 'instruction_following': '指令遵循', 'safety': '安全',
    'relevance': '相关性', 'format': '格式合规', 'robustness': '鲁棒性',
    'knowledge': '知识时效', 'untagged': '未标注',
}

# ============================== 能力维度口径 ============================== #
# 四类维度：准确性 / 指令遵循 / 安全 / 相关性。开放题（qa_open）走 LLM 裁判，
# 与「有客观答案的断言」不是同一种测法，单列「相关性」。
# 下面三张表与 src 侧保持一一对应（tests/test_dashboard_dimensions.py 守护）：
DIMENSION_ORDER = ('correctness', 'instruction_following', 'safety', 'relevance')
CATEGORY_DIMENSION = {
    'json_extract': 'instruction_following',
    'math_reasoning': 'correctness',
    'qa_open': 'relevance',
    'qa_zh': 'correctness',
    'safety_redteam': 'safety',
}
# 旧维度名 → 新维度名（format 是指令遵循的旧称）
LEGACY_DIMENSION_ALIASES = {'format': 'instruction_following'}


def _dimension_key(raw):
    """把报告里的维度名 / 分类名归一成四类维度之一。

    认不出的原样返回：宁可在报告里看到一行陌生维度，也不要静默丢掉数据。
    """
    key = str(raw or '').strip().lower()
    if not key:
        return None
    if key in LEGACY_DIMENSION_ALIASES:
        return LEGACY_DIMENSION_ALIASES[key]
    if key in CATEGORY_DIMENSION:  # 老报告给的是 category，按表换算
        return CATEGORY_DIMENSION[key]
    return key


def _case_dimension(case):
    """单条用例归到哪个维度：**已知分类一律按分类重算**，未知分类才用自带 dimension。

    老报告的 ``cases[]`` 里存的 dimension 是旧口径（qa_open 记在 correctness），
    照抄会让详情页出现「准确性 50 题」而首页重算是 42 题——又是两套数字。新报告
    的打标与分类映射本来就一致，重算不改变它们；自定义分类才退回用例自带的标签
    （那是数据集作者更精确的表达）。
    """
    if not isinstance(case, dict):
        return None
    key = str(case.get('category') or '').strip().lower()
    if key in CATEGORY_DIMENSION:
        return CATEGORY_DIMENSION[key]
    return _dimension_key(case.get('dimension') or case.get('category'))


def _dimension_summary(doc, model):
    """算一份报告的「各维度通过率」行，供首页维度速览展示。

    数据源按优先级：

    1. ``categories`` 段 —— 按「分类 → 维度」**重算**。老报告的 dimensions 段是
       生成时按旧口径分组的（qa_open 并进了 correctness），改不了；重算才能让
       老报告与新报告同一口径，满足「qa_open 不再出现在准确性里」。
    2. ``cases[]`` —— 逐条按 category 归并（categories 段缺失时的兜底）
    3. ``dimensions`` 段 —— 前两者都没有时的最后兜底（老维度名在此归一）

    返回 ``[{dimension, label, total, passed, pass_rate}]``，按 DIMENSION_ORDER
    排序；什么都取不到时返回空列表（前端不渲染该行，而不是画一排 0%）。
    """
    if not isinstance(doc, dict):
        return []

    agg: dict[str, list[int]] = {}
    # 已经聚合过、但没留样本量的行（dimensions 段里只有 pass_rate 的老数据）
    precounted: list[dict] = []

    def add(dim, total, passed):
        if not dim or not total:
            return
        bucket = agg.setdefault(dim, [0, 0])
        bucket[0] += int(total)
        bucket[1] += int(passed)

    def build_row(dim, total, passed):
        return {
            'dimension': dim,
            'label': _BANNER_DIM_LABELS.get(dim, dim),
            'total': total,
            'passed': passed,
            'pass_rate': (passed / total) if total else 0.0,
        }

    # 1) categories 段（老报告 / dashboard/data 普遍有）
    for row in (_candidates_for(doc.get('categories'), model) or []):
        if isinstance(row, dict):
            add(_dimension_key(row.get('category')), row.get('total', 0), row.get('passed', 0))

    # 2) cases[] 逐条归并
    if not agg:
        for case in (doc.get('cases') or []):
            if not isinstance(case, dict):
                continue
            if model and case.get('model') and case.get('model') != model:
                continue
            add(_case_dimension(case), 1, 1 if case.get('passed') is True else 0)

    # 3) dimensions 段（口径是生成时的，仅作兜底）
    if not agg:
        for row in (_candidates_for(doc.get('dimensions'), model) or []):
            if not isinstance(row, dict):
                continue
            dim = _dimension_key(row.get('dimension'))
            if not dim:
                continue
            if row.get('total'):
                add(dim, row.get('total', 0), row.get('passed', 0))
            else:
                # 只有通过率、没留样本量：原样保留这一行，别因为缺字段整份丢掉
                precounted.append({
                    'dimension': dim,
                    'label': _BANNER_DIM_LABELS.get(dim, dim),
                    'total': None,
                    'passed': None,
                    'pass_rate': float(row.get('pass_rate') or 0.0),
                })

    order = {name: index for index, name in enumerate(DIMENSION_ORDER)}
    rows = [build_row(dim, counts[0], counts[1]) for dim, counts in agg.items()] or precounted
    rows.sort(key=lambda row: (order.get(str(row['dimension']), len(DIMENSION_ORDER)), str(row['dimension'])))
    return rows


def _is_mock(name):
    """是否为对照组模型（``mock-*``）。

    mock-baseline 这类是阴性对照：它的分数低是设计使然，不是被测模型的能力，
    所以凡是「挑一个模型出来给人看」的场合（banner / 最弱维度 / 结论排名）
    都要把它排除掉，全是对照组时才退回。
    """
    return str(name or '').startswith('mock-')


def _candidates_for(group, model):
    """从 ``dict[model] -> list`` 里挑一组候选行。

    优先指定模型；指定模型缺席时跳过 ``mock-*`` 对照组，避免把对照组的
    分数当成被测模型的结论展示出来。全是 mock 的报告（如 CD 冒烟）才退回任意一组。
    """
    if group is None:
        return None
    if isinstance(group, list):
        return group
    if not isinstance(group, dict):
        return None

    own = group.get(model)
    if isinstance(own, list):
        return own
    for key, value in group.items():
        if isinstance(value, list) and not _is_mock(key):
            return value
    for value in group.values():
        if isinstance(value, list):
            return value
    return None


def _pick_representative_model(entry):
    """从 model_rows 选 banner 代表模型：优先真实模型（名字不以 ``mock-`` 开头）。

    横向评测的报告里通常有 ``mock-baseline`` 等对照组（可能多个 mock-*），
    把对照组推到 banner 会让人误以为「真实模型怎么才 100%」，所以必须挑
    真实模型。全 mock 时退到 ``entry['model']``；单模型报告 / 残缺数据
    时同样退到 ``entry['model']``。
    """
    rows = entry.get('model_rows') or []
    for row in rows:
        if isinstance(row, dict) and row.get('model') and not _is_mock(row['model']):
            return row['model']
    return entry.get('model', '?')


def _representative_rate(entry, model):
    """取代表模型在 entry 里的 pass_rate；找不到则降级到 entry.pass_rate。"""
    for row in (entry.get('model_rows') or []):
        if isinstance(row, dict) and row.get('model') == model:
            return row.get('pass_rate', 0.0)
    return entry.get('pass_rate', 0.0)


def _weakest_dimension(doc, model):
    """从报告里挑最弱的能力维度，返回 ``(label, rate)`` 或 ``None``。

    **口径唯一**：委托给 ``_dimension_summary()``（categories 重算 → cases →
    dimensions 兜底，旧维度名在此归一）。以前这里直接读报告里存的 ``dimensions``
    段，而那份聚合是生成时的快照：改口径后就变成「correctness 50 题（含 qa_open）」
    这种过期数字，和首页速览的 42 题对不上。现在首页卡片、banner、详情页维度表
    全部走 ``_dimension_summary`` 的同一份结果。

    ``untagged`` 不参与最弱选择（用户看不到意义），但全 ``untagged`` 时仍返回
    自身；取不到数据时返回 ``None``，前端显示 "—"。``mock-*`` 是对照组：只在没有
    真实模型可挑时才参与（详见 ``_candidates_for``）。
    """
    rows = _dimension_summary(doc, model)
    if not rows:
        return None

    pool = [r for r in rows if r['dimension'] != 'untagged'] or rows
    weakest = min(pool, key=lambda r: r['pass_rate'])
    return (weakest['label'], weakest['pass_rate'])


def compute_banner(curr_entry, curr_doc, base_entry=None, base_doc=None):
    """计算 banner 字段。纯函数无副作用；供 Python 单测与前端 JS 共享契约。

    入参：
    - curr_entry: ``_load_reports()`` 返回的最新一项
    - curr_doc: curr_entry.file 对应的完整报告 JSON（含 cases / dimensions / summary）
    - base_entry / base_doc: 上一次报告的同名对象；可为 None

    返回 dict，banner 直接消费：
        ``model / curr_rate / base_rate / delta_pt / regressed / fixed /
        weakest_label / weakest_rate / has_base``
    """
    model = _pick_representative_model(curr_entry)
    curr_rate = _representative_rate(curr_entry, model)

    base_rate = None
    if base_entry:
        # 上次报告里能精确匹配到代表模型才有"较上次"意义；找不到则 None，让 banner 显示 "—"
        rows = base_entry.get('model_rows') or []
        if any(isinstance(row, dict) and row.get('model') == model for row in rows):
            base_rate = _representative_rate(base_entry, model)

    delta_pt = (curr_rate - base_rate) * 100 if base_rate is not None else None

    if base_doc and curr_doc and isinstance(base_doc, dict) and isinstance(curr_doc, dict):
        diff = diff_cases(curr_doc.get('cases') or [], base_doc.get('cases') or [])
        regressed = len(diff['regressed'])
        fixed = len(diff['fixed'])
    else:
        regressed = 0
        fixed = 0

    weakest = _weakest_dimension(curr_doc, model) if isinstance(curr_doc, dict) else None

    return {
        'model': model,
        'curr_rate': curr_rate,
        'base_rate': base_rate,
        'delta_pt': delta_pt,
        'regressed': regressed,
        'fixed': fixed,
        'weakest_label': weakest[0] if weakest else None,
        'weakest_rate': weakest[1] if weakest else None,
        'has_base': base_entry is not None,
    }


# ============================== 详情页「汇总」结论 ============================== #
# 汇总 tab 顶部一句话结论：模型通过率排名 + 裁判是谁 + 值得注意的异常（如某分类明显偏低）。
# 服务端算好后塞进 /api/report/<file> 的 `_conclusion` 字段，前端只负责渲染：
# 这样动态服务与静态导出（build_static）结果完全一致，且逻辑能被单测覆盖。

# 指标名 → 中文标签（用例抽屉里展示用；未收录的指标原样显示，不会变空白）
METRIC_LABELS = {
    'json_valid': '格式校验',  # 旧指标，历史报告仍在，保留标签避免显示空白
    'is_json': 'JSON 解析',
    'schema_match': '字段匹配',
    'exact_match': '精确匹配',
    'similarity': '语义相似度',
    'judge': 'LLM 裁判',
    'contains': '包含检查',
    'not_contains': '排除检查',
}

# notes 里 runner 写的是「裁判模型 glm-4.5-air（通过阈值 4/5）」
_JUDGE_MODEL_RE = re.compile(r'裁判模型\s*([^\s（(]+)')

# 样本量太小不报异常（1~2 条失败没有统计意义），阈值以下才算「明显偏低」
_ANOMALY_MIN_TOTAL = 3
_ANOMALY_MAX_RATE = 0.5


def metric_label(name):
    """指标名 → 中文标签；未收录 / 非字符串原样返回。"""
    if not isinstance(name, str):
        return name
    return METRIC_LABELS.get(name, name)


def _parse_judge_model(report):
    """从 ``report['notes']`` 解析裁判模型名，没有则 None（老报告没有这条 note）。"""
    if not isinstance(report, dict):
        return None
    for note in report.get('notes') or []:
        if not isinstance(note, str):
            continue
        m = _JUDGE_MODEL_RE.search(note)
        if m:
            return m.group(1)
    return None


def _conclusion_anomalies(report):
    """挑出值得注意的异常：跳过用例、某分类明显偏低、未启用裁判。"""
    anomalies = []

    # 1) 跳过：通常是模型 / 裁判调用失败，会让通过率虚高或虚低，必须提示
    for row in report.get('summary') or []:
        if not isinstance(row, dict):
            continue
        skipped = row.get('skipped') or 0
        if skipped:
            anomalies.append(f"{row.get('model', '?')} 有 {skipped} 条用例跳过（判定不完整）")

    # 2) 分类明显偏低：跨模型取该分类最差的一条，避免只盯着某一个模型。
    #    mock-* 是对照组，本来就差，拿它报警只会制造噪音（与 banner 选代表模型同理）；
    #    全是 mock 的报告（如 CD 冒烟）才退回到所有模型。
    cats = report.get('categories')
    real_models = {
        r.get('model') for r in (report.get('summary') or [])
        if isinstance(r, dict) and not _is_mock(r.get('model'))
    }
    worst_by_cat = {}
    pairs = cats.items() if isinstance(cats, dict) else []
    for _model, rows in pairs:
        if real_models and _model not in real_models:
            continue
        for c in rows or []:
            if not isinstance(c, dict) or not c.get('category'):
                continue
            if (c.get('total') or 0) < _ANOMALY_MIN_TOTAL:
                continue
            rate = c.get('pass_rate', 0.0)
            prev = worst_by_cat.get(c['category'])
            if prev is None or rate < prev[0]:
                worst_by_cat[c['category']] = (rate, _model)
    for cat, (rate, model) in sorted(worst_by_cat.items(), key=lambda kv: kv[1][0]):
        if rate < _ANOMALY_MAX_RATE:
            anomalies.append(f"分类 {cat} 明显偏低（{model} 仅 {rate * 100:.1f}%）")

    # 3) 没裁判：judge 指标没生效，qa_open 这类只能靠字面对比，结论不可信
    if _parse_judge_model(report) is None:
        anomalies.append('本次未启用裁判模型（judge 指标未生效）')

    return anomalies[:4]


def compute_conclusion(report):
    """汇总 tab 顶部「本报告结论」。纯函数无副作用，供单测与前端渲染共享。

    返回 dict：
    - ``rank``: ``[{model, pass_rate}]``，按通过率降序（相同则保持原顺序）
    - ``judge``: 裁判模型名或 ``None``
    - ``anomalies``: ``[str]`` 值得注意的异常（最多 4 条）
    - ``text``: 拼好的一句话结论，前端直接展示
    """
    if not isinstance(report, dict):
        return {'rank': [], 'judge': None, 'anomalies': [], 'text': '本报告无汇总数据'}

    rows = [r for r in (report.get('summary') or []) if isinstance(r, dict)]
    # mock-* 是对照组，不进通过率排名：结论句里出现「mock-baseline 34.7%」会让人
    # 以为被测模型里有个 34.7% 的。全是 mock 的报告才退回所有模型，避免排名为空。
    ranked_rows = [r for r in rows if not _is_mock(r.get('model'))] or rows
    rank = sorted(
        [{'model': r.get('model', '?'), 'pass_rate': r.get('pass_rate', 0.0)} for r in ranked_rows],
        key=lambda x: -(x['pass_rate'] or 0.0),
    )
    judge = _parse_judge_model(report)
    anomalies = _conclusion_anomalies(report)

    if rank:
        rank_text = ' > '.join(f"{r['model']} {r['pass_rate'] * 100:.1f}%" for r in rank)
    else:
        rank_text = '无模型汇总数据'
    text = f"通过率排名：{rank_text}；裁判模型：{judge or '未启用'}"
    if anomalies:
        text += '；注意：' + '；'.join(anomalies)

    return {'rank': rank, 'judge': judge, 'anomalies': anomalies, 'text': text}


# ============================== 详情页「报告属性」 ============================== #
# 首页筛选器把报告分成「回归 / 稳定性」两类，那是**单选**维度，够筛但不够描述：
# full-repeat3 这种「既全量又是 repeat=3」的报告会被压成其中一个标签，另一半信息丢掉。
# 详情页顶部改成并排列出三个正交属性：范围 × 重复次数 × 判定口径。
#
# 口径说明（四维齐 = 全量）：数据集全集本身就覆盖四个维度（数学/事实问答 → 准确性，
# JSON 抽取 → 指令遵循，红队 → 安全，开放题 → 相关性），所以「四维都在」≈ 跑了全量；
# 只跑了其中几维的报告（如 --categories json_extract,qa_open）标为子集。

ATTRIBUTE_NOTES = {
    'full': '覆盖四个能力维度，等于跑了数据集全集',
    'subset': '只跑了部分数据集/分类，横向对比时要先看覆盖到哪几维',
}


def _covered_dimensions(doc):
    """报告实际覆盖到哪些维度（去重，按 DIMENSION_ORDER 排序）。

    优先按 ``cases[]`` 逐条算（老报告没有 dimensions 段也能算出来）；
    没有 cases 时才退回报告里存的 ``dimensions`` 段。
    """
    keys = set()
    for case in (doc.get('cases') or []):
        key = _case_dimension(case)
        if key:
            keys.add(key)

    if not keys:
        for row in (_candidates_for(doc.get('dimensions'), None) or []):
            if isinstance(row, dict):
                key = _dimension_key(row.get('dimension'))
                if key:
                    keys.add(key)

    order = {name: index for index, name in enumerate(DIMENSION_ORDER)}
    return sorted(keys, key=lambda k: (order.get(k, len(DIMENSION_ORDER)), k))


def _report_repeat(doc):
    """报告的重复次数：优先顶层 repeat，老报告从 cases 里推断。"""
    repeat = int(doc.get('repeat') or 0)
    if repeat > 1:
        return repeat
    for case in (doc.get('cases') or []):
        if not isinstance(case, dict):
            continue
        n = int(case.get('repeat') or 0) or len(case.get('attempts') or [])
        if n > 1:
            return n
    return 1


def compute_attributes(doc):
    """详情页属性条的数据。纯函数，供 Python 单测与前端共享。

    返回 dict：
    - ``scope``: ``'full'`` / ``'subset'``；``scope_label``: 全量 / 子集
    - ``dimensions``: 覆盖到的维度中文名列表
    - ``coverage_text``: 「四维全覆盖」或「准确性、安全」
    - ``repeat``: 每条用例执行次数（老报告缺省 1）
    - ``verdict``: 判定口径——「稳定率」（repeat>1）/「通过率」
    - 三者的 ``*_note``：hover 提示文案
    """
    if not isinstance(doc, dict):
        doc = {}

    dims = _covered_dimensions(doc)
    labels = [_BANNER_DIM_LABELS.get(d, d) for d in dims]
    is_full = set(DIMENSION_ORDER).issubset(set(dims))
    repeat = _report_repeat(doc)

    if not dims:
        coverage_text = '未识别到维度'
    elif is_full:
        coverage_text = '四维全覆盖'
    else:
        coverage_text = '、'.join(labels)

    verdict = '稳定率' if repeat > 1 else '通过率'

    return {
        'scope': 'full' if is_full else 'subset',
        'scope_label': '全量' if is_full else '子集',
        'dimensions': labels,
        'coverage_text': coverage_text,
        'scope_note': ATTRIBUTE_NOTES['full' if is_full else 'subset'],
        'repeat': repeat,
        'repeat_note': (
            f"每条用例独立跑 {repeat} 次，重复 {repeat} 次全通过才算通过"
            if repeat > 1 else '每条用例跑 1 次，单次结果即通过与否'
        ),
        'verdict': verdict,
        'verdict_note': (
            '以稳定率为准：单次通过率会被随机性带偏'
            if repeat > 1 else '以单次通过率为准'
        ),
        'is_stability': repeat > 1,
    }


# 模板里的 __METRIC_LABELS__ 在导入时替换成真实的中文标签映射，
# 保证「Python 单测可见的 METRIC_LABELS」与「前端渲染用的 METRIC_LABELS」是同一份数据。
PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM &amp; Agent 评测实验室</title>
<style>
  :root {
    --bg: #0f1420; --panel: #171e2e; --panel2: #1d2740; --border: #2a3650;
    --text: #e6ecf7; --muted: #8fa0bf; --accent: #4f8ef7;
    --ok: #34c98e; --warn: #f5b449; --bad: #f0566a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font: 14px/1.6 "Segoe UI", "Microsoft YaHei", sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 22px; font-weight: 600; margin-bottom: 4px; }
  .sub { color: var(--muted); margin-bottom: 24px; }
  /* 首页最顶部的"大字结论"横幅 */
  .banner { background: linear-gradient(90deg, rgba(79,142,247,.14), rgba(79,142,247,.03));
            border: 1px solid var(--border); border-left: 3px solid var(--accent);
            border-radius: 10px; padding: 14px 20px; margin-bottom: 18px;
            font-size: 15px; line-height: 1.75; }
  .banner .big { font-size: 26px; font-weight: 700; }
  .banner .delta-up { color: var(--ok); font-weight: 600; }
  .banner .delta-down { color: var(--bad); font-weight: 600; }
  .banner .delta-na { color: var(--muted); font-weight: 600; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: var(--muted); font-weight: 500; padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  td { padding: 8px 10px; border-bottom: 1px solid rgba(42,54,80,.5); white-space: nowrap; }
  tr:last-child td { border-bottom: none; }
  tr.rowlink { cursor: pointer; transition: background .15s; }
  tr.rowlink:hover { background: var(--panel2); }
  .errbox { background: rgba(240,86,106,.15); border: 1px solid var(--bad); color: var(--bad); border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; font-size: 13px; word-break: break-all; }
  .errbox b { color: var(--bad); }
  .rate { font-weight: 600; }
  .bar { position: relative; background: #232d47; border-radius: 5px; height: 8px; width: 130px; overflow: hidden; }
  .bar > i { position: absolute; inset: 0 auto 0 0; border-radius: 5px; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; background: var(--panel2); color: var(--muted); }
  .badge.pass { background: rgba(52,201,142,.18); color: var(--ok); }
  .badge.fail { background: rgba(240,86,106,.18); color: var(--bad); }
  .badge.skip { background: rgba(245,180,73,.18); color: var(--warn); }
  /* repeat>1 报告：抖动用例标记 / 重复执行并排卡片 */
  .badge.warn { background: rgba(245,180,73,.18); color: var(--warn); }
  /* 报告类型标签：回归测试（单次）与稳定性测试（repeat>1）用不同颜色，避免视觉混淆 */
  .badge.rtype-reg { background: rgba(79,142,247,.18); color: var(--accent); }
  .badge.rtype-stab { background: rgba(163,113,247,.18); color: #a371f7; }
  /* 首页卡片副行：紧贴报告行下方，不参与点击跳转 */
  tr.dimrow td { padding: 0 10px 9px; border-bottom: 1px solid rgba(42,54,80,.5); color: var(--muted); font-size: 13px; white-space: normal; }
  tr.dimrow .dimchip { display: inline-block; }
  tr.dimrow .dimchip.dimmock { opacity: .55; }
  /* 详情页「维度 × 模型」对比矩阵 */
  table.matrix th, table.matrix td { text-align: center; white-space: nowrap; }
  table.matrix th:first-child, table.matrix td:first-child { text-align: left; white-space: normal; }
  table.matrix th.col-mock { font-weight: 400; opacity: .65; }
  table.matrix td.col-mock { opacity: .5; }
  table.matrix tr.subrow { background: var(--panel2); }
  table.matrix tr.spread-row { background: rgba(255,214,102,.11); }
  table.matrix tr.subrow.spread-row { background: rgba(255,214,102,.06); }
  table.matrix td.spread-alert { color: var(--warn); font-weight: 600; }
  /* 详情页顶部属性条：范围 × 重复 × 判定口径（三个正交属性并排，不做单选分类） */
  .attrs { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 14px; }
  .attr { background: var(--panel2); border: 1px solid var(--border); border-radius: 6px;
          padding: 3px 10px; font-size: 13px; color: var(--text); }
  .attr .attrlabel { color: var(--muted); margin-right: 6px; font-size: 12px; }
  tr.dimrow .dimsep { opacity: .45; margin: 0 4px; }
  tr.dimrow .dimlabel { opacity: .55; margin-right: 10px; }
  tr.dimrow .dimcover { color: var(--muted); opacity: .9; margin-left: 10px; }
  .badge.dimmodel { background: rgba(122,162,247,.14); color: var(--accent); }
  .attempts { display: flex; gap: 10px; flex-wrap: wrap; }
  .attempt { flex: 1 1 220px; min-width: 200px; background: var(--panel2);
             border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
  .attempt-head { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 12px; }
  .stat { background: var(--panel2); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .stat .k { color: var(--muted); font-size: 12px; }
  .stat .v { font-size: 20px; font-weight: 600; margin-top: 2px; }
  .back { color: var(--accent); text-decoration: none; display: inline-block; margin-bottom: 14px; font-size: 13px; }
  .back:hover { text-decoration: underline; }
  .catname { color: var(--muted); }
  /* 表标题下的灰色小字：说明这张表回答什么问题 */
  .hint { color: var(--muted); font-size: 12px; margin: -2px 0 10px; }
  /* 详情页顶部「本报告结论」 */
  .conclusion { border-left: 3px solid var(--accent); }
  .anomaly-list { margin: 8px 0 0 18px; color: var(--warn); font-size: 13px; line-height: 1.8; }
  details.card > summary { cursor: pointer; list-style: none; }
  details.card > summary::-webkit-details-marker { display: none; }
  details.card > summary::before { content: '▸ '; color: var(--muted); }
  details.card[open] > summary::before { content: '▾ '; }
  .empty { color: var(--muted); padding: 40px; text-align: center; }
  /* 门户页（空 hash）：两张等大入口卡。grid 的 1fr/1fr 保证同宽，
     默认 align-items:stretch 保证同高，短的那张不会被压成半张。 */
  .portal { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
  .pcard { display: block; background: var(--panel); border: 1px solid var(--border);
           border-radius: 12px; padding: 20px 22px; color: inherit; text-decoration: none;
           transition: border-color .15s, transform .15s; }
  .pcard:hover { border-color: var(--accent); transform: translateY(-2px); }
  .ptitle { font-size: 17px; font-weight: 600; }
  .psub { color: var(--muted); font-size: 13px; margin-top: 4px; }
  .pgo { color: var(--accent); font-size: 13px; margin-top: 14px; }

  /* 标签页导航 */
  .tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border); margin-bottom: 18px; }
  .tab { padding: 8px 16px; color: var(--muted); cursor: pointer; border: none; background: none;
         font: inherit; border-bottom: 2px solid transparent; margin-bottom: -1px; transition: color .12s; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  /* 用例列表工具栏 */
  .toolbar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }
  .toolbar select, .toolbar input { background: var(--panel2); border: 1px solid var(--border);
    color: var(--text); padding: 6px 10px; border-radius: 6px; font: inherit; }
  .toolbar input { min-width: 220px; }
  .toolbar .stat-inline { color: var(--muted); margin-left: auto; font-size: 12px; }

  /* 行底色（用于跨次对比） */
  tr.row-regressed td { background: rgba(240,86,106,.18); }
  tr.row-regressed:hover td { background: rgba(240,86,106,.30); }
  tr.row-fixed td { background: rgba(52,201,142,.18); }
  tr.row-fixed:hover td { background: rgba(52,201,142,.30); }

  /* 等宽 / 可滚动文本块 */
  .mono { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
          white-space: pre-wrap; word-break: break-word; font-size: 13px; }
  .scroll-box { max-height: 320px; overflow: auto; background: var(--bg);
    border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; }

  /* 抽屉（用例详情） */
  .drawer-mask { position: fixed; inset: 0; background: rgba(0,0,0,.45); z-index: 50;
                  opacity: 0; pointer-events: none; transition: opacity .18s; }
  .drawer-mask.open { opacity: 1; pointer-events: auto; }
  .drawer { position: fixed; top: 0; right: 0; bottom: 0; width: min(720px, 96vw);
            background: var(--panel); border-left: 1px solid var(--border); z-index: 51;
            transform: translateX(100%); transition: transform .22s ease-out;
            display: flex; flex-direction: column; }
  .drawer.open { transform: translateX(0); }
  .drawer-head { padding: 14px 18px; border-bottom: 1px solid var(--border);
                  display: flex; align-items: center; gap: 10px; }
  .drawer-head h2 { font-size: 16px; font-weight: 600; flex: 1; word-break: break-all; }
  .drawer-body { flex: 1; overflow: auto; padding: 18px 20px; }
  .drawer-close { background: none; border: 1px solid var(--border); color: var(--text);
                   width: 30px; height: 30px; border-radius: 6px; cursor: pointer;
                   font-size: 16px; line-height: 1; }
  .drawer-close:hover { background: var(--panel2); }
  .drawer-section { margin-bottom: 18px; }
  .drawer-section h3 { font-size: 13px; color: var(--muted); margin-bottom: 8px; font-weight: 500; }
  .err { color: var(--bad); }
  .info-row { display: flex; gap: 14px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }
  .info-row span b { color: var(--text); font-weight: 500; }

  /* 小屏：drawer 全屏，工具栏单列 */
  @media (max-width: 640px) {
    .drawer { width: 100vw; }
    .toolbar { flex-direction: column; align-items: stretch; }
    .toolbar input { min-width: 0; width: 100%; }
  }
</style>
</head>
<body>
<div class="wrap" id="app"><div class="empty">加载中…</div></div>
<div class="drawer-mask" id="drawerMask"></div>
<aside class="drawer" id="drawer" role="dialog" aria-modal="true">
  <div class="drawer-head">
    <h2 id="drawerTitle">case</h2>
    <span id="drawerBadge"></span>
    <button class="drawer-close" id="drawerClose" aria-label="关闭">×</button>
  </div>
  <div class="drawer-body" id="drawerBody"></div>
</aside>
<script>
const $app = document.getElementById('app');
const $drawer = document.getElementById('drawer');
const $drawerMask = document.getElementById('drawerMask');
const $drawerTitle = document.getElementById('drawerTitle');
const $drawerBadge = document.getElementById('drawerBadge');
const $drawerBody = document.getElementById('drawerBody');
const $drawerClose = document.getElementById('drawerClose');

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = x => (x * 100).toFixed(1) + '%';
// 维度中文标签 / 分类→维度映射：由 Python 侧注入（与 _BANNER_DIM_LABELS /
// CATEGORY_DIMENSION 是同一份数据），首页速览、详情页维度表、banner 共用一套口径
const DIM_LABELS = __DIM_LABELS__;
const CATEGORY_DIMENSION = __CATEGORY_DIMENSION__;
const LEGACY_DIM_ALIASES = __LEGACY_DIM_ALIASES__;
const DIMENSION_ORDER = __DIMENSION_ORDER__;
const dimLabel = d => d ? (DIM_LABELS[d] || d) : '未标注';
// 老报告的 cases 没有 dimension 字段（或写的是旧名 format）：按分类换算成四类之一，
// 否则整份老报告的维度表会全部落进「未标注」
const dimKeyOf = raw => {
  const k = String(raw || '').trim().toLowerCase();
  if (!k) return 'untagged';
  if (LEGACY_DIM_ALIASES[k]) return LEGACY_DIM_ALIASES[k];
  if (CATEGORY_DIMENSION[k]) return CATEGORY_DIMENSION[k];
  return k;
};
// 单条用例归到哪个维度：已知分类一律按分类重算，未知分类才用用例自带的 dimension。
// 老报告的 cases 里存的 dimension 是旧口径（qa_open 记在 correctness），照抄会让
// 详情页出现「准确性 50 题」而首页是 42 题——又是两套数字。新报告的打标与分类
// 映射本来就一致，重算不会改变它们。
const caseDimOf = c => {
  const cat = String((c && c.category) || '').trim().toLowerCase();
  if (CATEGORY_DIMENSION[cat]) return CATEGORY_DIMENSION[cat];
  return dimKeyOf((c && c.dimension) || cat);
};
// 指标中文标签：由后端 METRIC_LABELS 注入（与 Python 侧同一份数据），未收录的原样显示
const METRIC_LABELS = __METRIC_LABELS__;
const metricLabel = m => METRIC_LABELS[m] || m;
const rateColor = x => x >= 0.8 ? 'var(--ok)' : x >= 0.5 ? 'var(--warn)' : 'var(--bad)';
const rateInner = x => `<div style="display:flex;align-items:center;gap:8px"><span class="rate" style="color:${rateColor(x)}">${pct(x)}</span><span class="bar"><i style="width:${(x*100).toFixed(1)}%;background:${rateColor(x)}"></i></span></div>`;
const rateCell = x => `<td>${rateInner(x)}</td>`;

// ---------- repeat（重复执行）相关：老报告没有这些字段，一律降级为 '—' ----------
// 用例重复次数：优先用 repeat 字段，缺失时从 attempts 长度推断，都没有则 0（= 单次评测）
const repeatOf = c => Number((c && c.repeat) || 0)
  || ((c && Array.isArray(c.attempts) && c.attempts.length) || 0);
// 稳定率单元格：null/undefined（老报告）显示 '—'
const stabCell = v => `<td>${(v == null) ? '<span class="catname">—</span>' : rateInner(v)}</td>`;
// 报告类型：repeat>1 = 稳定性测试（同一条用例跑多次，看稳定率）；否则回归测试
const isStability = r => Number((r && r.repeat) || 1) > 1;
const typeBadge = r => isStability(r)
  ? `<span class="badge rtype-stab" title="每条用例重复跑 ${r.repeat} 次，看稳定率">稳定性测试</span>`
  : `<span class="badge rtype-reg" title="每条用例跑 1 次">回归测试</span>`;
// 列表「用例」列：稳定性报告显示「30 × 3次」（单模型用例数 × 重复次数）
const caseCountText = r => {
  const n = Number((r && r.repeat) || 1);
  const models = (r && r.model_rows && r.model_rows.length) || 1;
  if (n > 1) return `${Math.round((r.case_count || 0) / models)} × ${n}次`;
  return `${r.case_count || 0}`;
};
// 列表「通过率 / 稳定率」列：稳定性报告取 stability（严格通过率放详情页），
// 老报告没有 stability 时退回 pass_rate，不出现空白
const listRate = (r, x) => (isStability(r) && x && x.stability != null) ? x.stability : (x ? x.pass_rate : 0);
// 首页卡片副行：一行 chips 列出**每个模型**的整体通过率（chat 93.9% · pro 92.9% …）。
// 为什么是模型总览而不是维度速览：维度速览只显示一个代表模型（跳过 mock 取第一个真模型），
// 读者会把它当成整份报告的成绩——标注了模型名也只是缓解，仍然是「一堆维度 + 一个模型」。
// 横向评测的价值在于对比，所以首页给模型之间的对比数字；维度 × 模型的对比留给
// 详情页的矩阵（信息更全，还带分类子行和最大差值）。末尾保留覆盖标注：
// 「只跑了安全」和「全量」必须一眼能分开，否则 93.3% 会被误读成比 34.7% 强。
const modelOverviewRowHtml = r => {
  if (!r) return '';
  const rows = (r.model_rows && r.model_rows.length)
    ? r.model_rows
    : (r.model && r.model !== '?' ? [{model: r.model, pass_rate: r.pass_rate || 0}] : []);
  const dims = r.dimensions || [];
  if (!rows.length && !dims.length) return '';
  const chips = rows.map(m =>
    `<span class="dimchip${isMockModel(m.model) ? ' dimmock' : ''}">${esc(m.model)} `
    + `<b class="rate" style="color:${rateColor(m.pass_rate || 0)}">${pct(m.pass_rate || 0)}</b></span>`
  ).join('<span class="dimsep">·</span>');
  const coverage = dims.length >= 4
    ? '全部四维'
    : ((r.dimension_coverage && r.dimension_coverage.length)
        ? r.dimension_coverage.join('、')
        : '—');
  return `<tr class="dimrow"><td colspan="7"><span class="dimlabel">模型</span>${chips}`
    + `<span class="dimcover">覆盖：${esc(coverage)}</span></td></tr>`;
};
// 「2/3」+ 抖动标记
const repeatCell = c => {
  const n = repeatOf(c);
  if (!n || n <= 1 || c.pass_count == null) return '<td><span class="catname">—</span></td>';
  const mark = c.flaky ? ' <span class="badge warn" title="同一用例重复跑结果不一致">不稳定</span>' : '';
  return `<td><span class="badge">${c.pass_count}/${n}</span>${mark}</td>`;
};
// 横向报告一份文件含多个模型，列表里必须按模型逐行展示（只显示第一个会让人以为整份报告是 mock）
const isMultiModel = r => !!(r.model_rows && r.model_rows.length > 1);
const multiCell = (r, fn) => r.model_rows.map(fn).join('');

// 把任意 Python/JSON 值渲染成 HTML（dict/list 原样展示，str 保留换行）
const formatAny = v => {
  if (v === null || v === undefined) return '<span class="catname">（空）</span>';
  if (typeof v === 'string') return esc(v);
  if (typeof v === 'number' || typeof v === 'boolean') return esc(String(v));
  try { return esc(JSON.stringify(v, null, 2)); } catch (_) { return esc(String(v)); }
};

// 单条用例的状态：pass / fail / skip
function statusOf(c) {
  if (c.error) return 'skip';
  if (c.passed === true) return 'pass';
  return 'fail';
}

// 用例唯一键：case_id + 模型（与后端 case_key 同口径，分隔符 \x1f）
// 多模型报告里同一 case_id 有 N 条（每模型一条），只用 case_id 当键会让
// 后写的覆盖先写的 —— 列表行是 A 模型，点开抽屉却是 B 模型（状态可能相反）。
function caseKey(c) {
  if (!c) return '';
  return String(c.case_id ?? '') + '\x1f' + String(c.model ?? '');
}

// 跨次对比：按用例唯一键（case_id + 模型）排序
function diffCases(currCases, baseCases) {
  const keyOf = caseKey;
  const ids = new Set();
  (currCases || []).forEach(c => ids.add(keyOf(c)));
  (baseCases || []).forEach(c => ids.add(keyOf(c)));
  const allIds = Array.from(ids).sort();
  const regressed = [], fixed = [], stillFailing = [], onlyInCurr = [], onlyInBase = [];
  const mapOf = arr => new Map((arr || []).map(c => [keyOf(c), c]));
  const currMap = mapOf(currCases), baseMap = mapOf(baseCases);
  for (const id of allIds) {
    const cur = currMap.get(id);
    const base = baseMap.get(id);
    if (cur && !base) { onlyInCurr.push(cur); continue; }
    if (base && !cur) { onlyInBase.push(base); continue; }
    const cs = statusOf(cur), bs = statusOf(base);
    if (bs === 'pass' && cs !== 'pass') regressed.push({curr: cur, base});
    else if (bs !== 'pass' && cs === 'pass') fixed.push({curr: cur, base});
    else if (cs !== 'pass') stillFailing.push({curr: cur, base});
  }
  return {regressed, fixed, stillFailing, onlyInCurr, onlyInBase};
}

// 维度聚合的唯一实现（与后端 _dimension_summary 同契约）：
// categories 段按「分类 → 维度」重算 → cases[] 逐条归并 → dimensions 段兜底。
// 报告里存的 dimensions 段是生成时的快照（口径可能是旧的），所以只能兜底：
// banner、首页速览、详情页维度表都必须走这里，否则各处各算一套就会互相打脸。
function dimensionSummary(doc, model) {
  if (!doc || typeof doc !== 'object') return [];
  const agg = new Map();
  const add = (rawDim, total, passed) => {
    if (!rawDim) return;
    const dim = dimKeyOf(rawDim);
    const t = Number(total) || 0;
    if (!t) return;
    const b = agg.get(dim) || [0, 0];
    b[0] += t;
    b[1] += Number(passed) || 0;
    agg.set(dim, b);
  };
  // 与后端 _candidates_for 同契约：指定模型缺席时跳过 mock-* 对照组
  const pickGroup = (g) => {
    if (!g) return null;
    if (Array.isArray(g)) return g;
    if (typeof g !== 'object') return null;
    const own = g[model];
    if (Array.isArray(own)) return own;
    const entries = Object.entries(g);
    const real = entries.find(([k, v]) => Array.isArray(v) && !String(k).startsWith('mock-'));
    if (real) return real[1];
    const any = entries.find(([, v]) => Array.isArray(v));
    return any ? any[1] : null;
  };

  (pickGroup(doc.categories) || []).forEach(row => row && add(row.category, row.total, row.passed));
  if (!agg.size) {
    (doc.cases || []).forEach(c => {
      if (!c) return;
      if (model && c.model && c.model !== model) return;
      add(c.dimension || c.category, 1, c.passed === true ? 1 : 0);
    });
  }
  // dimensions 段兜底：样本量缺失的老数据只带 pass_rate，原样保留别丢
  const precounted = [];
  if (!agg.size) {
    (pickGroup(doc.dimensions) || []).forEach(row => {
      if (!row || !row.dimension) return;
      const dim = dimKeyOf(row.dimension);
      if (Number(row.total) || 0) { add(row.dimension, row.total, row.passed); return; }
      precounted.push({
        dimension: dim, label: dimLabel(dim), total: null, passed: null,
        pass_rate: Number(row.pass_rate) || 0,
      });
    });
  }
  if (!agg.size && !precounted.length) return [];

  const order = Array.isArray(DIMENSION_ORDER) ? DIMENSION_ORDER : [];
  const rows = agg.size
    ? Array.from(agg.entries()).map(([dim, counts]) => ({
        dimension: dim,
        label: dimLabel(dim),
        total: counts[0],
        passed: counts[1],
        pass_rate: counts[0] ? counts[1] / counts[0] : 0,
      }))
    : precounted;
  return rows.sort((a, b) => {
    const ia = order.indexOf(a.dimension), ib = order.indexOf(b.dimension);
    const ra = ia < 0 ? order.length : ia, rb = ib < 0 ? order.length : ib;
    return (ra - rb) || String(a.dimension).localeCompare(String(b.dimension));
  });
}

// 首页最顶部"大字结论"banner：复用现有 api/reports.json + api/report/{file}，不加新接口
async function computeBannerData(reports) {
  if (!reports || !reports.length) return null;
  const curr = reports[0];
  const base = reports[1] || null;

  // 静默 fetch；失败让 banner 不显示而不是让首页崩
  const safeFetchDoc = (entry) => entry
    ? fetch('api/report/' + encodeURIComponent(entry.file))
        .then(r => r && r.ok ? r.json() : null)
        .catch(() => null)
    : Promise.resolve(null);

  const [currDoc, baseDoc] = await Promise.all([safeFetchDoc(curr), safeFetchDoc(base)]);

  // 代表模型：优先真实模型（名字不以 mock- 开头）；全 mock / 无 model_rows 时退到 entry.model
  const rows = curr.model_rows || [];
  const real = rows.find(r => r && r.model && !r.model.startsWith('mock-'));
  const representative = (real && real.model) || curr.model;

  const repRateRow = rows.find(r => r && r.model === representative);
  const currRate = (repRateRow && typeof repRateRow.pass_rate === 'number') ? repRateRow.pass_rate : curr.pass_rate;

  // base 报告里能找到同一模型才算"较上次"，否则显示 "—"
  let baseRate = null;
  if (base) {
    const baseRows = base.model_rows || [];
    if (baseRows.some(r => r && r.model === representative)) {
      const r = baseRows.find(x => x.model === representative);
      baseRate = (r && typeof r.pass_rate === 'number') ? r.pass_rate : null;
    }
  }
  const deltaPt = baseRate == null ? null : (currRate - baseRate) * 100;

  // 跨次对比复用 diffCases（已有 helper）
  const diff = (baseDoc && currDoc) ? diffCases(currDoc.cases || [], baseDoc.cases || []) : null;
  const regressed = diff ? diff.regressed.length : 0;
  const fixed = diff ? diff.fixed.length : 0;

  // 最弱维度：走 dimensionSummary（categories 重算优先），与首页速览同一口径。
  // 不能直接读 currDoc.dimensions —— 那是报告生成时的旧口径快照。
  let weakest = null;
  if (currDoc) {
    const dimRows = dimensionSummary(currDoc, representative);
    const pool = dimRows.filter(d => d.dimension !== 'untagged');
    const usable = pool.length ? pool : dimRows;
    if (usable.length) {
      const w = usable.reduce((a, b) => (a.pass_rate <= b.pass_rate ? a : b));
      weakest = { label: w.label, rate: w.pass_rate };
    }
  }

  return { model: representative, currRate, baseRate, deltaPt, regressed, fixed, weakest, hasBase: !!base };
}

function renderBanner(data) {
  if (!data) return '';
  const rateText = (data.currRate * 100).toFixed(1) + '%';
  let deltaHtml;
  if (data.deltaPt == null) {
    deltaHtml = '<span class="delta-na">—</span>';
  } else {
    const cls = data.deltaPt > 0 ? 'delta-up' : (data.deltaPt < 0 ? 'delta-down' : 'delta-na');
    const sign = data.deltaPt > 0 ? '+' : '';
    deltaHtml = `<span class="${cls}">${sign}${data.deltaPt.toFixed(1)} pt</span>`;
  }
  const weakestHtml = data.weakest
    ? `${esc(data.weakest.label)} <b>${(data.weakest.rate * 100).toFixed(1)}%</b>`
    : '<span class="delta-na">—</span>';
  return `
    <div class="banner">
      最新报告 <b style="color:var(--accent)">${esc(data.model)}</b>
      <span class="big" style="color:${rateColor(data.currRate)}">${rateText}</span>
      ，较上次 ${deltaHtml}
      ；回归 <b style="color:var(--bad)">${data.regressed}</b> 条 / 修复 <b style="color:var(--ok)">${data.fixed}</b> 条
      ；最弱维度：${weakestHtml}
    </div>`;
}

// ---------- Agent 评测板块 ---------- //
// 数据源与评测报告完全分开：这里读的是 agent_eval/results/ 下的 result-*.json，
// 「通过率」的含义是「agent 有没有调对工具/答对题」，不是模型的答对率——
// 两个数并在一张表里会互相污染口径，所以单列一块。
// INVALID 作废存档由后端直接过滤掉（带 -INVALID- 的结果不返回），前端不再判一次。
function agentReportLink(r) {
  if (!r || !r.report) return '<span class="catname">—</span>';
  const tip = r.report_exact
    ? '与本次结果时间戳完全对应的 HTML 报告'
    : '未找到时间戳完全对应的报告，取本次结果之后生成的最新一份';
  return `<a class="back" style="margin:0" title="${esc(tip)}" href="api/agent/report/${encodeURIComponent(r.report)}" target="_blank" rel="noopener">HTML 报告 →</a>`;
}

function agentRepeatCard(latest) {
  const rs = latest && latest.repeat_summary;
  if (!rs) return '';
  const rows = rs.逐题 || [];
  if (!rows.length) return '';
  return `
    <div style="margin-top:14px">
      <div style="font-weight:600;margin-bottom:4px">逐题一致性（${rs.遍数} 遍）</div>
      <div class="hint">同一任务集跑 ${rs.遍数} 遍，逐题比对各遍状态与步数：状态一致 = 这题没抖。
        状态一致 <b>${esc(rs.状态一致题数)}</b>${rs.全部一致 ? ' <span class="badge pass">全部一致</span>' : ' <span class="badge warn">存在抖动题</span>'}</div>
      <table>
        <tr><th>任务</th><th>各遍状态</th><th>各遍步数</th><th>一致性</th></tr>
        ${rows.map(t => `
          <tr>
            <td class="mono">${esc(t.id)}</td>
            <td>${(t.各遍状态 || []).map(s => `<span class="badge ${s === '通过' ? 'pass' : 'fail'}" style="margin-right:4px">${esc(s)}</span>`).join('')}</td>
            <td class="catname">${esc((t.各遍步数 || []).join(' / '))}</td>
            <td>${t.状态一致 ? '<span class="badge pass">一致</span>' : '<span class="badge warn">抖动</span>'}</td>
          </tr>`).join('')}
      </table>
    </div>`;
}

// Agent 评测页（#agent）：最新一次的四个指标 + 逐题一致性 + 历史结果表
async function renderAgent() {
  const a = await fetch('api/agent.json').then(r => r && r.ok ? r.json() : null).catch(() => null);
  const runs = (a && a.runs) || [];
  const latest = (a && a.latest) || null;
  const holdBadge = latest && latest.trap_hold
    ? '<span class="badge pass">守住</span>'
    : (latest && latest.trap_hold === false ? '<span class="badge fail">未守住</span>' : '<span class="catname">—</span>');
  const statHtml = (k, v, color) => `<div class="stat"><div class="k">${esc(k)}</div>
      <div class="v"${color ? ` style="color:${color}"` : ''}>${v}</div></div>`;

  $app.innerHTML = `
    <a class="back" href="#">← 返回实验室首页</a>
    <h1>Agent 评测 <span class="sub" style="font-size:13px;font-weight:400">（agent 有没有调对工具 / 答对题）</span></h1>
    <div class="sub">数据源：<code>agent_eval/results/</code>（含 <code>archive/</code>）；
      这里的「通过率」是 agent 调对工具的比例，与上面「LLM 评测报告」的模型答对率不是一回事，
      所以分开看，不并在一张表里。</div>
    ${latest ? `
    <div class="card">
      <div class="hint">最新一次：${esc(latest.time)}${latest.archived ? ' <span class="badge">历史归档</span>' : ''}
        · ${esc(latest.file)}${latest.report ? ` · ${agentReportLink(latest)}` : ''}</div>
      <div class="grid" style="margin-bottom:4px">
        ${statHtml('任务成功率', `${pct(latest.pass_rate || 0)}<div class="k">${latest.passed ?? 0} / ${latest.total ?? 0} 通过</div>`, rateColor(latest.pass_rate || 0))}
        ${statHtml('工具选择正确率', pct(latest.tool_rate || 0), rateColor(latest.tool_rate || 0))}
        ${statHtml('平均步数', esc(latest.avg_steps ?? '—'))}
        ${statHtml('陷阱题防线', holdBadge)}
      </div>
      ${agentRepeatCard(latest)}
    </div>
    <div class="card">
      <div style="font-weight:600;margin-bottom:4px">历史结果（${runs.length} 次）</div>
      <table>
        <tr><th>时间</th><th>通过率</th><th>repeat</th><th>状态</th><th>报告</th></tr>
        ${runs.map(r => `
          <tr>
            <td class="catname">${esc(r.time)}${r.archived ? ' <span class="badge">归档</span>' : ''}</td>
            ${rateCell(r.pass_rate || 0)}
            <td><span class="badge">${esc(r.repeat)} 遍</span></td>
            <td>${Number(r.failed || 0) === 0 && Number(r.invalid || 0) === 0
                ? `<span class="badge pass">全通过</span>`
                : `<span class="badge fail">${esc(r.failed ?? 0)} 失败${Number(r.invalid || 0) ? ` · ${esc(r.invalid)} 无效` : ''}</span>`}
              <span class="catname" style="margin-left:6px">${esc(r.passed ?? 0)}/${esc(r.total ?? 0)}</span></td>
            <td>${agentReportLink(r)}</td>
          </tr>`).join('') || `<tr><td colspan="5" class="empty">暂无结果</td></tr>`}
      </table>
      <div class="hint" style="padding:10px 12px 0">带 <code>-INVALID-</code> 的作废存档不进看板；
        报告链接按时间戳配对（精确匹配优先，配不上取该结果之后生成的最新一份），配不上显示「—」。</div>
    </div>` : `
    <div class="card"><div class="sub" style="margin:0">暂无结果。先跑
      <code>python agent_eval/run_eval.py</code> 生成 <code>agent_eval/results/result-*.json</code>。</div></div>`}`;
}

// 门户页（空 hash）：两层评测各一张入口卡，等大并排，各带「最新成绩」摘要。
// 这一页只负责导航和一眼看完，模型层细节在 #llm，Agent 层细节在 #agent。
async function renderPortal(reports) {
  const agent = await fetch('api/agent.json').then(r => r && r.ok ? r.json() : null).catch(() => null);
  const statHtml = (k, v, color) => `<div class="stat"><div class="k">${esc(k)}</div>
      <div class="v"${color ? ` style="color:${color}"` : ''}>${v}</div></div>`;

  // ①模型层：横向报告一份文件含多个模型，通过率要按模型取范围（与 #llm 顶部卡片同口径）
  const latest = reports[0] || null;
  const multi = latest && isMultiModel(latest);
  const rates = multi ? latest.model_rows.map(x => listRate(latest, x))
    : (latest ? [listRate(latest, latest)] : []);
  const modelCount = multi ? latest.model_rows.length : (latest ? 1 : 0);
  const llmSummary = latest ? `
      <div class="grid" style="margin-top:12px">
        ${statHtml(`最新${isStability(latest) ? '稳定率' : '通过率'}`,
          rates.length > 1 ? `${pct(Math.min(...rates))} ~ ${pct(Math.max(...rates))}` : pct(rates[0] || 0),
          rateColor(rates.length ? rates[0] : 0))}
        ${statHtml('对比模型数', modelCount)}
        ${statHtml('报告数', reports.length)}
      </div>`
    : `<div class="sub" style="margin:12px 0 0">暂无报告。先跑一次模型层评测。</div>`;

  // ②Agent 层：成功率 + repeat 一致性 + 历史次数，口径同 #agent 页
  const aLatest = (agent && agent.latest) || null;
  const aRuns = (agent && agent.runs) || [];
  const aRs = aLatest && aLatest.repeat_summary;
  const agentSummary = aLatest ? `
      <div class="grid" style="margin-top:12px">
        ${statHtml('任务成功率', `${pct(aLatest.pass_rate || 0)}<div class="k">${aLatest.passed ?? 0} / ${aLatest.total ?? 0} 通过</div>`, rateColor(aLatest.pass_rate || 0))}
        ${statHtml('工具选择正确率', pct(aLatest.tool_rate || 0), rateColor(aLatest.tool_rate || 0))}
        ${statHtml('累计评测', `${aRuns.length}<div class="k">次</div>`)}
      </div>
      ${aRs ? `<div class="sub" style="margin:8px 0 0">repeat ${esc(aRs.遍数)} 遍一致性：状态一致
        <b>${esc(aRs.状态一致题数)}</b> / ${esc(aLatest.total ?? 0)} 题
        ${aRs.全部一致 ? '<span class="badge pass">全部一致</span>' : '<span class="badge warn">存在抖动题</span>'}</div>` : ''}`
    : `<div class="sub" style="margin:12px 0 0">暂无结果。先跑
        <code>python agent_eval/run_eval.py</code>。</div>`;

  $app.innerHTML = `
    <h1>LLM &amp; Agent 评测实验室</h1>
    <div class="sub">两层评测：<b>模型层</b>比能力——同一用例集横向对比多个大模型；
      <b>Agent 层</b>考执行——固定工具集下，agent 有没有调对工具、答对题。</div>
    <div class="portal">
      <a class="pcard" href="#llm">
        <div class="ptitle">① 模型层评测</div>
        <div class="psub">同一用例集横向对比多个大模型</div>
        ${llmSummary}
        <div class="pgo">进入 →</div>
      </a>
      <a class="pcard" href="#agent">
        <div class="ptitle">② Agent 层评测</div>
        <div class="psub">${esc(aLatest && aLatest.total ? aLatest.total : 10)} 题含 4 道陷阱 · 四层断言归因</div>
        ${agentSummary}
        <div class="pgo">进入 →</div>
      </a>
    </div>`;
}

// 渲染报告列表（#llm，原首页）
async function renderList(reports) {
  const tests = await (await fetch('api/tests.json')).json();
  const hasTests = tests && tests.total;
  const ciAllPass = hasTests && tests.failed === 0 && tests.errors === 0;
  // 列表里只要有一份稳定性报告，通过率列就要改成「稳定率 / 通过率」双含义表头
  const hasStability = (reports || []).some(isStability);
  // 顶部「最新通过率」卡片同样跟随：最新报告是稳定性测试时显示稳定率，
  // 否则列表与卡片两个数打架（一个 75.6% 一个 60.0%）反而看不懂
  const topRate = reports[0] ? listRate(reports[0], reports[0]) : 0;
  const topRates = (reports[0] && isMultiModel(reports[0]))
    ? reports[0].model_rows.map(x => listRate(reports[0], x))
    : [topRate];
  // 顶部 banner：复用现有 api 接口；数据计算是异步的但已与 tests.json 并行 fetch
  const bannerData = await computeBannerData(reports);
  $app.innerHTML = `
    <a class="back" href="#">← 返回实验室首页</a>
    ${bannerData ? renderBanner(bannerData) : ''}
    <h1>LLM 评测</h1>
    <div class="sub">共 ${reports.length} 份报告 · ${reports.reduce((a,r)=>a+r.case_count,0)} 个用例 · 最新 ${esc(reports[0].time)}</div>
    <div class="grid" style="margin-bottom:16px">
      <div class="stat"><div class="k">报告数</div><div class="v">${reports.length}</div></div>
      <div class="stat"><div class="k">单次最多用例</div><div class="v">${Math.max(...reports.map(r=>r.case_count))}</div></div>
      <div class="stat"><div class="k">最新${isStability(reports[0]) ? '稳定率' : '通过率'}</div><div class="v" style="color:${rateColor(topRate)}">${topRates.length > 1 ? `${pct(Math.min(...topRates))} ~ ${pct(Math.max(...topRates))}` : pct(topRate)}</div></div>
      <div class="stat"><div class="k">最新模型</div><div class="v" style="font-size:14px">${esc(isMultiModel(reports[0]) ? reports[0].models.join(' / ') : reports[0].model)}</div></div>
    </div>
    ${hasTests ? `
    <div class="card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap">
        <div>
          <div style="font-weight:600; margin-bottom:4px">CI 测试 <span class="badge ${ciAllPass?'pass':'fail'}">${ciAllPass?'全绿':(tests.failed+tests.errors)+' 失败'}</span></div>
          <div class="sub" style="margin-bottom:0">
            ${tests.total} 用例 · ${tests.passed} 通过 · ${tests.failed} 失败 · ${tests.errors} 异常 · ${tests.skipped} 跳过 · 耗时 ${tests.duration_s}s
            <span class="catname"> · 生成于 ${esc(tests.generated_at || '')}</span>
            ${tests.branch ? `<span class="catname"> · 分支 <code>${esc(tests.branch)}</code></span>` : ''}
            ${tests.commit ? `<span class="catname"> · <code>${esc((tests.commit||'').slice(0,7))}</code></span>` : ''}
          </div>
        </div>
        ${tests.run_url ? `<a class="back" href="${esc(tests.run_url)}" target="_blank" rel="noopener">查看运行日志 →</a>` : ''}
        <a class="back" href="#tests">查看用例详情 →</a>
      </div>
      ${tests.failed_tests && tests.failed_tests.length ? `
      <details style="margin-top:12px">
        <summary style="cursor:pointer;color:var(--muted);font-size:13px">${tests.failed_tests.length} 条失败明细</summary>
        <div class="mono scroll-box" style="margin-top:10px; max-height:240px">
          ${tests.failed_tests.map(t => `<div style="padding:4px 0; border-bottom:1px solid rgba(42,54,80,.4)"><span style="color:var(--bad)">✗</span> <code>${esc(t.classname)}.${esc(t.name)}</code>${t.message ? `<div style="color:var(--muted); margin-left:22px; margin-top:2px">${esc(t.message.slice(0,300))}${t.message.length>300?'…':''}</div>` : ''}</div>`).join('')}
        </div>
      </details>` : ''}
    </div>` : `
    <div class="card"><div class="sub" style="margin:0">暂无 CI 测试数据（dashboard/data/tests-summary.json 不存在）</div></div>`}
    <div class="card"><table>
      <tr><th>报告</th><th>时间</th><th>模型</th><th>用例</th><th>通过 / 失败</th><th>${hasStability ? '稳定率 / 通过率' : '通过率'}</th><th>P95 延迟</th></tr>
      ${reports.map((r, i) => {
        const multi = isMultiModel(r);
        const modelCell = multi ? multiCell(r, x => `<div>${esc(x.model)}</div>`) : esc(r.model);
        const totalCell = caseCountText(r);
        const pfCell = multi ? multiCell(r, x => `<div>${x.passed} / ${x.failed}</div>`) : `${r.passed} / ${r.failed}`;
        const rateCellHtml = multi
          ? `<td>${multiCell(r, x => rateInner(listRate(r, x)))}</td>`
          : rateCell(listRate(r, r));
        const p95Cell = multi
          ? multiCell(r, x => `<div>${x.p95 != null ? x.p95.toFixed(2) + ' ms' : '—'}</div>`)
          : (r.p95 != null ? r.p95.toFixed(2) + ' ms' : '—');
        return `
        <tr class="rowlink" data-go="${encodeURIComponent(r.file)}/summary">
          <td>#${reports.length - i} ${typeBadge(r)} <span class="badge">${r.tag || 'run'}</span>${multi ? ` <span class="badge">${r.model_rows.length} 模型</span>` : ''}</td>
          <td class="catname">${esc(r.time)}</td>
          <td>${modelCell}</td>
          <td>${totalCell}</td>
          <td>${pfCell}</td>
          ${rateCellHtml}
          <td>${p95Cell}</td>
        </tr>
        ${modelOverviewRowHtml(r)}`;
      }).join('')}
    </table>
    ${hasStability ? `<div class="hint" style="padding:10px 12px 0">「稳定性测试」= 每条用例重复跑 N 次，用例列显示「单模型用例数 × 重复次数」，通过率列显示<b>稳定率</b>（要求 N 次全过的严格通过率见报告详情）；「回归测试」= 每条用例跑 1 次。</div>` : ''}
    </div>`;
}

// 详情页属性条：范围 × 重复次数 × 判定口径，三个正交属性并排列出。
// 首页筛选器要的是「单选分类」（回归 / 稳定性），但 full-repeat3 这种「既全量又
// repeat=3」的报告用单选会丢掉一半信息，所以详情页改成并列的三个属性。
function attributesBar(a) {
  if (!a) return '';
  const chip = (label, text, tip) => `<span class="attr"${tip ? ` title="${esc(tip)}"` : ''}>`
    + `<span class="attrlabel">${esc(label)}</span>${esc(text)}</span>`;
  return `<div class="attrs">
    ${chip('范围', a.scope_label + (a.coverage_text ? `（${a.coverage_text}）` : ''), a.scope_note)}
    ${chip('重复', `repeat=${a.repeat}`, a.repeat_note)}
    ${chip('判定', a.verdict, a.verdict_note)}
  </div>`;
}

// 渲染详情页框架（标题 + tabs + 当前 tab 内容）
async function renderDetail(file, tab) {
  const r = await (await fetch('api/report/' + encodeURIComponent(file))).json();
  $app.innerHTML = `
    <a class="back" href="#llm">← 返回报告列表</a>
    <h1>${esc(file)}</h1>
    <div class="sub">${esc(r.started_at || '')} · ${r.case_count} 用例 · ${((r.duration_s ?? 0)).toFixed(2)}s
      · 数据集：${esc((r.datasets || []).join(', '))}</div>
    ${attributesBar(r._attributes)}
    ${r.notes && r.notes.length ? `<div class="card" style="color:var(--muted);font-size:13px">备注：${esc(r.notes.join('；'))}</div>` : ''}
    <div class="tabs" id="tabs">
      ${[['summary','汇总'],['cases','用例列表'],['diff','跨次对比']].map(([k,name]) =>
        `<a class="tab ${k===tab?'active':''}" href="#${encodeURIComponent(file)}/${k}">${name}</a>`).join('')}
    </div>
    <div id="tabBody"></div>`;
  const body = document.getElementById('tabBody');
  if (tab === 'cases') renderCasesTab(r, file);
  else if (tab === 'diff') renderDiffTab(r, file);
  else renderSummaryTab(r);
}

// 维度通过率表：按 case 的 dimension 标签分组，未打标签的归入 untagged 一行。
// 注：r.dimensions 为 {model: [{dimension,label,total,passed,failed,skipped,pass_rate}]}；
// 老报告没有该字段，返回空串不渲染。
// 结论卡片：一句话写明排名 / 裁判 / 异常，由后端 compute_conclusion 算好塞进 _conclusion
function conclusionCard(c, r) {
  if (!c || !c.text) return '';
  return `
    <div class="card conclusion">
      <div style="font-weight:600;margin-bottom:6px">本报告结论</div>
      <div style="font-size:14px;line-height:1.7">${esc(c.text)}</div>
      ${flakyNote(r)}
      ${(c.anomalies || []).length ? `
        <ul class="anomaly-list">${c.anomalies.map(a => `<li>${esc(a)}</li>`).join('')}</ul>` : ''}
    </div>`;
}

// 有抖动用例时补一句：repeat>1 的场景下单次通过率会被随机性带偏，应看稳定率
function flakyNote(r) {
  const rows = (r && Array.isArray(r.summary)) ? r.summary : [];
  const hits = rows.filter(s => Number((s && s.flaky) || 0) > 0);
  if (!hits.length) return '';
  const text = hits.map(s => `${s.model} 有 ${s.flaky} 条用例结果不稳定`).join('；');
  return `<div class="hint" style="margin-top:6px">⚠️ ${esc(text)}，看稳定率而非单次通过率。</div>`;
}

// 维度（主行）→ 分类（子行）两层分组：维度是能力面，分类是具体任务类型。
// 直接按 cases 现算，避免 dimensions / categories 两份聚合口径不一致（老报告只有其一）。
function buildDimCatGroups(cases) {
  const byModel = new Map();
  (cases || []).forEach(c => {
    const model = c.model || '（无）';
    const dim = caseDimOf(c);
    const cat = c.category || '（无）';
    if (!byModel.has(model)) byModel.set(model, new Map());
    const dims = byModel.get(model);
    if (!dims.has(dim)) dims.set(dim, new Map());
    const cats = dims.get(dim);
    if (!cats.has(cat)) cats.set(cat, {category: cat, total: 0, passed: 0, skipped: 0});
    const b = cats.get(cat);
    b.total++;
    const st = statusOf(c);
    if (st === 'pass') b.passed++;
    else if (st === 'skip') b.skipped++;
  });

  const out = [];
  for (const [model, dims] of byModel) {
    for (const [dim, cats] of dims) {
      const catList = Array.from(cats.values())
        .map(c => ({...c, pass_rate: c.total ? c.passed / c.total : 0}))
        .sort((a, b) => String(a.category).localeCompare(String(b.category)));
      const total = catList.reduce((a, c) => a + c.total, 0);
      const passed = catList.reduce((a, c) => a + c.passed, 0);
      const skipped = catList.reduce((a, c) => a + c.skipped, 0);
      out.push({model, dimension: dim, total, passed, skipped,
                pass_rate: total ? passed / total : 0, cats: catList});
    }
  }
  // 按模型名 + 维度顺序稳定输出，避免每次渲染行序跳动
  return out.sort((a, b) => String(a.model).localeCompare(String(b.model))
    || String(a.dimension).localeCompare(String(b.dimension)));
}

// ---------- 维度 × 模型 对比矩阵 ---------- //
// 横向评测的核心视图：行 = 能力维度（子行 = 具体分类），列 = 被测模型，格 = 通过率。
// 把三个模型平铺成三张「模型 → 维度」表没法横向比，必须放进一张表里才看得出
// 「谁在哪一维掉队」。mock-* 是管道基线不是被测对象，灰显但不隐藏。
const SPREAD_ALERT_PT = 5;   // 真模型间差距 ≥ 5pt 才高亮，低于这个量级是噪声
const isMockModel = n => String(n || '').startsWith('mock-');

// 真模型之间通过率的最大差距（百分点）；真模型不足两个时返回 null（显示 —）
function spreadOf(cellMap, models) {
  const rates = models
    .filter(m => !isMockModel(m) && cellMap.get(m))
    .map(m => cellMap.get(m).pass_rate);
  if (rates.length < 2) return null;
  const max = Math.max(...rates), min = Math.min(...rates);
  return {pt: (max - min) * 100, max, min};
}

// 把 buildDimCatGroups 的「每个模型一组」结果透视成「维度 × 模型」矩阵
function buildDimensionMatrix(cases, models) {
  const groups = buildDimCatGroups(cases || []);
  const order = Array.from(new Set((models && models.length) ? models : groups.map(g => g.model)));
  const idx = new Map(order.map((m, i) => [m, i]));
  // 列序：mock-* 基线放最左，真模型照报告里的顺序（= 配置顺序），避免每次渲染跳列
  order.sort((a, b) => ((isMockModel(a) ? 0 : 1) - (isMockModel(b) ? 0 : 1))
    || ((idx.get(a) ?? 99) - (idx.get(b) ?? 99)));

  const dims = new Map();
  groups.forEach(g => {
    if (!dims.has(g.dimension)) {
      dims.set(g.dimension, {
        dimension: g.dimension, label: dimLabel(g.dimension), cells: new Map(), cats: new Map(),
      });
    }
    const d = dims.get(g.dimension);
    d.cells.set(g.model, {
      total: g.total, passed: g.passed, skipped: g.skipped, pass_rate: g.pass_rate,
    });
    (g.cats || []).forEach(c => {
      if (!d.cats.has(c.category)) d.cats.set(c.category, {category: c.category, cells: new Map()});
      d.cats.get(c.category).cells.set(g.model, {
        total: c.total, passed: c.passed, skipped: c.skipped, pass_rate: c.pass_rate,
      });
    });
  });

  const rank = d => {
    const i = DIMENSION_ORDER.indexOf(d);
    return i < 0 ? DIMENSION_ORDER.length : i;
  };
  const rows = Array.from(dims.values())
    .sort((a, b) => (rank(a.dimension) - rank(b.dimension))
      || String(a.dimension).localeCompare(String(b.dimension)))
    .map(d => ({
      dimension: d.dimension,
      label: d.label,
      cells: order.map(m => d.cells.get(m) || null),
      spread: spreadOf(d.cells, order),
      cats: Array.from(d.cats.values())
        .sort((a, b) => String(a.category).localeCompare(String(b.category)))
        .map(c => ({
          category: c.category,
          cells: order.map(m => c.cells.get(m) || null),
          spread: spreadOf(c.cells, order),
        })),
    }));
  return {models: order, rows};
}

function dimensionMatrixCard(matrix) {
  if (!matrix || !matrix.rows.length) return '';
  const models = matrix.models;
  const single = models.length <= 1;

  const head = models.map(m => `
    <th class="${isMockModel(m) ? 'col-mock' : ''}">${esc(m)}${
      isMockModel(m) ? '<span class="catname" style="font-weight:400"> 基线</span>' : ''}</th>`).join('');

  const cellHtml = (cell, model) => {
    const cls = isMockModel(model) ? ' class="col-mock"' : '';
    if (!cell) return `<td${cls}><span class="catname">—</span></td>`;
    const sub = cell.total
      ? `<div class="catname" style="font-size:11px">${cell.passed}/${cell.total}${
          cell.skipped ? ` · 跳过 ${cell.skipped}` : ''}</div>`
      : '';
    return `<td${cls} title="${esc(model)}：${cell.passed}/${cell.total}">`
      + `<b style="color:${rateColor(cell.pass_rate)}">${pct(cell.pass_rate)}</b>${sub}</td>`;
  };
  const spreadHtml = s => (s == null
    ? '<td><span class="catname">—</span></td>'
    : `<td class="${s.pt >= SPREAD_ALERT_PT ? 'spread-alert' : ''}" title="真模型之间：${
        pct(s.min)} ~ ${pct(s.max)}">${s.pt.toFixed(1)} pt</td>`);

  // 单模型报告：矩阵退化成一列，「最大差值」无从比较，整列不渲染
  const rowHtml = (labelHtml, cells, spread, extraCls) => {
    const hl = (spread && spread.pt >= SPREAD_ALERT_PT) ? ' spread-row' : '';
    return `<tr class="${extraCls || ''}${hl}">${labelHtml}`
      + `${cells.map((c, i) => cellHtml(c, models[i])).join('')}`
      + `${single ? '' : spreadHtml(spread)}</tr>`;
  };

  const body = matrix.rows.map(r => {
    const main = rowHtml(
      `<td><span class="badge">${esc(r.label)}</span>${
        r.dimension === 'untagged' ? '<span class="catname" style="font-size:12px"> 未打标签</span>' : ''}</td>`,
      r.cells, r.spread, r.dimension === 'untagged' ? 'dim-untagged' : '');
    const subs = r.cats.map(c => rowHtml(
      `<td style="padding-left:22px"><span class="catname">↳ ${esc(c.category)}</span></td>`,
      c.cells, c.spread, 'subrow')).join('');
    return main + subs;
  }).join('');

  return `
    <div class="card">
      <div style="font-weight:600;margin-bottom:4px">维度 × 模型 对比矩阵</div>
      <div class="hint">行 = 能力维度（子行 = 具体分类），列 = 被测模型，格 = 通过率（下方小字为通过 / 总数）。
        <code>mock-*</code> 是管道基线，灰显、不作为被测对象；${single
          ? '本报告只有一个模型，矩阵退化为单列。'
          : `「最大差值」= 真模型之间通过率的最大差距，≥ ${SPREAD_ALERT_PT} pt 的行标黄——那是模型间有实质差异的信号。`}</div>
      <table class="matrix">
        <tr><th>维度 / 分类</th>${head}${single ? '' : '<th title="真模型之间通过率的最大差距">最大差值</th>'}</tr>
        ${body}
      </table>
    </div>`;
}

// 汇总 tab：结论 → 模型总览 → 维度/分类 → 指标均值（折叠）
function renderSummaryTab(r) {
  const s = r.summary || [];
  // 列顺序照 summary（= 配置顺序），mock-* 由矩阵自己排到最左
  const matrix = buildDimensionMatrix(r.cases || [], (s || []).map(m => m.model));
  document.getElementById('tabBody').innerHTML = `
    ${conclusionCard(r._conclusion, r)}
    <div class="card">
      <div style="font-weight:600;margin-bottom:4px">模型总览</div>
      <div class="hint">各模型整体表现对比。稳定率 = 全部调用中通过的比例；抖动用例 = 同一用例重复跑结果不一致的条数（仅 repeat&gt;1 的评测有数据，老报告显示 —）。</div>
      <table>
        <tr><th>模型</th><th>通过 / 总数</th><th>通过率</th><th>稳定率</th><th>抖动用例</th><th>avg 延迟</th><th>P95</th><th>tokens</th><th>成本</th></tr>
        ${s.map(m => `
          <tr>
            <td>${esc(m.model)}</td>
            <td>${m.passed} / ${m.total}</td>
            ${rateCell(m.pass_rate)}
            ${stabCell(m.stability)}
            <td>${(m.flaky == null) ? '<span class="catname">—</span>' : m.flaky}</td>
            <td>${(m.avg_latency_ms ?? 0).toFixed(2)} ms</td>
            <td>${(m.p95_latency_ms ?? 0).toFixed(2)} ms</td>
            <td>${m.total_tokens ?? '—'}</td>
            <td>${m.cost ?? 0}</td>
          </tr>`).join('')}
      </table>
    </div>
    ${dimensionMatrixCard(matrix)}
    ${(s[0] && s[0].metric_avg) ? `
    <details class="card">
      <summary style="font-weight:600">指标均值（首个模型）</summary>
      <div class="hint" style="margin-top:8px">每种判定方法的健康度，不是模型能力。</div>
      <table>
        ${Object.entries(s[0].metric_avg).map(([k, v]) => `
          <tr><td><span class="badge">${esc(metricLabel(k))}</span>
            <span class="catname" style="font-size:12px"> ${esc(k)}</span></td><td>${pct(v)}</td>
          <td><span class="bar" style="width:220px"><i style="width:${(v*100).toFixed(1)}%;background:${rateColor(v)}"></i></span></td></tr>`).join('')}
      </table>
    </details>` : ''}`;
}

// 用例列表 tab：表格 + 工具栏（状态筛选 / 分类下拉 / 搜索）+ 点击行打开抽屉
function renderCasesTab(r, file) {
  const cases = (r.cases || []).slice().sort((a, b) => String(a.case_id).localeCompare(String(b.case_id)));
  const cats = Array.from(new Set(cases.map(c => c.category || '（无）'))).sort();
  // 多模型报告里同一 case_id 会有多行，模型下拉用来只看某个模型的结果
  const models = Array.from(new Set(cases.map(c => c.model || '（无）'))).sort();
  let state = {status: 'all', model: 'all', category: 'all', q: ''};
  // 单行预览：压平换行 + 截断（promptfoo 式列表直接可见输入/输出摘要）
  const prev = (s, n) => {
    if (s == null) return '';
    const t = String(s).replace(/\s+/g, ' ').trim();
    return t.length > n ? t.slice(0, n) + '…' : t;
  };

  const draw = () => {
    const filtered = cases.filter(c => {
      if (state.status !== 'all' && statusOf(c) !== state.status) return false;
      if (state.model !== 'all' && (c.model || '（无）') !== state.model) return false;
      if (state.category !== 'all' && (c.category || '（无）') !== state.category) return false;
      if (state.q) {
        const q = state.q.toLowerCase();
        const hay = String(c.case_id) + ' ' + String(c.prompt || '') + ' ' + String(c.response || '');
        if (!hay.toLowerCase().includes(q)) return false;
      }
      return true;
    });
    document.getElementById('tabBody').innerHTML = `
      <div class="sub" style="margin-bottom:10px">
        数据流：<code>datasets/*.jsonl</code>（每行 = 一条评测用例：输入 + 期望输出）
        → runner 调模型 → 按分类打分 → <code>report.json</code> 的 <code>cases[]</code> → 本页。
        每行 = <b>某个模型跑某条用例</b>的一条结果（同一 case_id 会有多行，
        一行一个模型）；展示<b>输入 / 模型输出</b>摘要，点行看完整 prompt、
        期望输出、评分明细（抽屉里显示的就是这一行所属模型的那条结果）。
      </div>
      <div class="toolbar">
        <select id="fltStatus">
          <option value="all" ${state.status==='all'?'selected':''}>全部状态</option>
          <option value="pass" ${state.status==='pass'?'selected':''}>通过</option>
          <option value="fail" ${state.status==='fail'?'selected':''}>失败</option>
          <option value="skip" ${state.status==='skip'?'selected':''}>跳过</option>
        </select>
        <select id="fltModel" ${models.length <= 1 ? 'disabled title="本报告只有一个模型"' : ''}>
          <option value="all" ${state.model==='all'?'selected':''}>${models.length > 1 ? `全部模型（${models.length}）` : '全部模型'}</option>
          ${models.map(m => `<option value="${esc(m)}" ${state.model===m?'selected':''}>${esc(m)}</option>`).join('')}
        </select>
        <select id="fltCat">
          <option value="all" ${state.category==='all'?'selected':''}>全部分类</option>
          ${cats.map(c => `<option value="${esc(c)}" ${state.category===c?'selected':''}>${esc(c)}</option>`).join('')}
        </select>
        <input id="fltQ" placeholder="搜索 case_id / 输入 / 输出内容" value="${esc(state.q)}">
        <span class="stat-inline">共 ${cases.length} 条 · 筛选后 ${filtered.length}</span>
      </div>
      <div class="card" style="padding:0;overflow:auto">
        <table>
          <tr><th style="width:56px">状态</th><th style="width:110px">case_id</th><th style="width:110px">模型</th><th>输入（prompt）</th><th>模型输出（response）</th><th style="width:90px">分类</th><th style="width:96px">通过次数</th><th style="width:80px">延迟</th></tr>
          ${filtered.map(c => {
            const st = statusOf(c);
            const lat = c.latency_ms != null ? c.latency_ms.toFixed(2) + ' ms' : '—';
            const failMetric = (c.metrics || []).find(m => m.passed === false);
            return `
            <tr class="rowlink" data-case="${esc(caseKey(c))}">
              <td><span class="badge ${st}">${st==='pass'?'✓':st==='fail'?'✗':'-'}</span></td>
              <td><code>${esc(c.case_id)}</code></td>
              <td><span class="badge">${esc(c.model || '（无）')}</span></td>
              <td class="catname" style="white-space:normal;max-width:280px">${esc(prev(c.prompt, 80))}</td>
              <td class="catname" style="white-space:normal;max-width:280px">${esc(prev(c.response, 80))}${st==='fail' && failMetric ? `<div class="err" style="font-size:12px;margin-top:3px">${esc(prev(failMetric.detail || failMetric.name, 60))}</div>` : ''}</td>
              <td><span class="badge">${esc(c.category || '（无）')}</span></td>
              ${repeatCell(c)}
              <td>${lat}</td>
            </tr>`;
          }).join('')}
        </table>
      </div>`;
    document.getElementById('fltStatus').onchange = e => { state.status = e.target.value; draw(); };
    document.getElementById('fltModel').onchange = e => { state.model = e.target.value; draw(); };
    document.getElementById('fltCat').onchange = e => { state.category = e.target.value; draw(); };
    document.getElementById('fltQ').oninput = e => { state.q = e.target.value; draw(); };
  };
  // 用例唯一键 → case 映射（用于行点击打开 drawer；多模型报告下 case_id 不唯一）
  window.casesByKey = new Map(cases.map(c => [caseKey(c), c]));
  draw();
}

// 跨次对比 tab：选基线 → diffCases → 三个分组表格（行点击复用 drawer）
async function renderDiffTab(r, file) {
  const reports = await (await fetch('api/reports.json')).json();
  // 基线默认选「最近一次更早的报告」（reports 按时间倒序，当前 reports[0] 是当前查看的）
  const currIdx = reports.findIndex(x => x.file === file);
  const candidates = reports.filter(x => x.file !== file && x.time < (reports[currIdx]?.time || ''));
  const defaultBase = candidates[0]?.file || '';
  const baseSel = document.createElement('select');
  baseSel.id = 'baseSel';
  baseSel.style.cssText = 'background:var(--panel2);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:6px;font:inherit';
  baseSel.innerHTML = `<option value="">（请选择基线）</option>` +
    reports.filter(x => x.file !== file).map(x =>
      `<option value="${esc(x.file)}" ${x.file===defaultBase?'selected':''}>${esc(x.time)} · ${esc(x.model)} · ${esc(x.file)}</option>`
    ).join('');
  document.getElementById('tabBody').innerHTML = `
    <div class="toolbar">
      <span style="color:var(--muted)">基线报告：</span>
      <span id="baseWrap"></span>
      <span class="stat-inline" id="diffStat"></span>
    </div>
    <div id="diffBody"></div>`;
  document.getElementById('baseWrap').appendChild(baseSel);
  baseSel.onchange = () => runDiff();

  async function runDiff() {
    const baseFile = baseSel.value;
    if (!baseFile) {
      document.getElementById('diffBody').innerHTML = '<div class="empty">请选择一份基线报告</div>';
      document.getElementById('diffStat').textContent = '';
      return;
    }
    const base = await (await fetch('api/report/' + encodeURIComponent(baseFile))).json();
    const d = diffCases(r.cases || [], base.cases || []);
    window.casesByKey = new Map((r.cases || []).map(c => [caseKey(c), c]));
    const renderGroup = (title, color, rows, label) => {
      if (!rows.length) return `<div class="card"><div style="font-weight:600;margin-bottom:10px">${esc(title)} <span class="badge" style="margin-left:6px">0</span></div><div class="catname" style="padding:8px 0">无</div></div>`;
      return `
        <div class="card">
          <div style="font-weight:600;margin-bottom:10px">${esc(title)} <span class="badge ${color}" style="margin-left:6px">${rows.length}</span></div>
          <table>
            <tr><th>状态变化</th><th>case_id</th><th>分类</th><th>模型</th><th>基线</th><th>当前</th></tr>
            ${rows.map(row => {
              const cur = row.curr || row, base = row.base;
              const cs = statusOf(cur), bs = base ? statusOf(base) : '—';
              return `
              <tr class="rowlink ${color}" data-case="${esc(caseKey(cur))}">
                <td><span class="badge ${color}">${esc(label)}</span></td>
                <td>${esc(cur.case_id)}</td>
                <td><span class="badge">${esc(cur.category || '（无）')}</span></td>
                <td>${esc(cur.model || '')}</td>
                <td><span class="badge ${bs}">${bs==='pass'?'✓':bs==='fail'?'✗':bs==='skip'?'-':'-'}</span></td>
                <td><span class="badge ${cs}">${cs==='pass'?'✓':cs==='fail'?'✗':'-'}</span></td>
              </tr>`;
            }).join('')}
          </table>
        </div>`;
    };
    document.getElementById('diffBody').innerHTML = `
      ${(d.onlyInCurr.length || d.onlyInBase.length) ? `
        <div class="card" style="border-color:var(--warn)">
          <div style="font-weight:600;margin-bottom:6px">⚠ 用例集合不一致</div>
          <div class="catname" style="font-size:13px">
            当前独有 ${d.onlyInCurr.length} 条 · 基线独有 ${d.onlyInBase.length} 条
            ${d.onlyInCurr.length ? '· 当前独有：' + d.onlyInCurr.slice(0,5).map(c=>esc(c.case_id)).join(', ') + (d.onlyInCurr.length>5?' …':'') : ''}
            ${d.onlyInBase.length ? '· 基线独有：' + d.onlyInBase.slice(0,5).map(c=>esc(c.case_id)).join(', ') + (d.onlyInBase.length>5?' …':'') : ''}
          </div>
        </div>` : ''}
      ${renderGroup('回退完成（上次通过 → 这次失败）', 'fail', d.regressed, '回归')}
      ${renderGroup('修复完成（上次失败 → 这次通过）', 'pass', d.fixed, '修复')}
      ${renderGroup('仍失败（两次都失败）', 'fail', d.stillFailing, '未变')}`;
    document.getElementById('diffStat').textContent =
      `回归 ${d.regressed.length} · 修复 ${d.fixed.length} · 仍失败 ${d.stillFailing.length}`;
  }
  if (defaultBase) runDiff();
}

// 抽屉：显示用例完整详情
// 重复执行明细：每次尝试并排展示通过/失败 + 耗时（老报告无 attempts，整块不渲染）
function attemptsSection(c) {
  const attempts = (c && Array.isArray(c.attempts)) ? c.attempts : [];
  if (!attempts.length) return '';
  const total = attempts.length;
  const passed = (c.pass_count != null) ? c.pass_count : attempts.filter(a => a.passed === true).length;
  const cards = attempts.map(a => {
    const cls = a.passed === true ? 'pass' : (a.passed === false ? 'fail' : 'skip');
    const label = a.passed === true ? '通过' : (a.passed === false ? '失败' : '未判定');
    const lat = (a.latency_ms != null) ? a.latency_ms.toFixed(2) + ' ms' : '—';
    const bad = (a.metrics || []).find(m => m.passed === false);
    return `
      <div class="attempt">
        <div class="attempt-head">
          <span class="badge">第 ${Number(a.index ?? 0) + 1} 次</span>
          <span class="badge ${cls}">${label}</span>
          <span class="catname">${lat}</span>
        </div>
        <div class="scroll-box mono">${esc(a.response || '')}</div>
        ${a.error ? `<div class="err" style="font-size:12px;margin-top:4px">${esc(a.error)}</div>` : ''}
        ${bad ? `<div class="err" style="font-size:12px;margin-top:4px">${esc(bad.detail || bad.name)}</div>` : ''}
      </div>`;
  }).join('');
  return `
    <div class="drawer-section">
      <h3>重复执行（${total} 次，通过 ${passed} 次）${c.flaky ? ' <span class="badge warn">不稳定</span>' : ''}</h3>
      <div class="attempts">${cards}</div>
    </div>`;
}

function openDrawer(c) {
  if (!c) return;
  const st = statusOf(c);
  const tok = (c.prompt_tokens || 0) + (c.completion_tokens || 0);
  $drawerTitle.textContent = c.case_id;
  $drawerBadge.innerHTML = `<span class="badge ${st}">${st==='pass'?'通过':st==='fail'?'失败':'跳过'}</span>`;
  $drawerBody.innerHTML = `
    <div class="info-row" style="margin-bottom:14px">
      <span>分类：<b>${esc(c.category || '（无）')}</b></span>
      <span>维度：<b>${esc(dimLabel(c.dimension))}</b></span>
      <span>数据集：<b>${esc(c.dataset || '（无）')}</b></span>
      <span>模型：<b>${esc(c.model || '（无）')}</b></span>
    </div>
    <div class="drawer-section">
      <h3>prompt</h3>
      <div class="scroll-box mono">${esc(c.prompt || '')}</div>
    </div>
    <div class="drawer-section">
      <h3>expected</h3>
      <div class="scroll-box mono">${formatAny(c.expected)}</div>
    </div>
    <div class="drawer-section">
      <h3>response</h3>
      <div class="scroll-box mono">${esc(c.response || '')}</div>
    </div>
    ${attemptsSection(c)}
    ${c.error ? `
    <div class="drawer-section">
      <h3>error</h3>
      <div class="scroll-box mono err">${esc(c.error)}</div>
    </div>` : ''}
    <div class="drawer-section">
      <h3>指标 (${(c.metrics||[]).length})</h3>
      ${(c.metrics||[]).length ? `
        <table>
          <tr><th>指标</th><th>score</th><th>通过</th><th>详情</th></tr>
          ${c.metrics.map(m => `
            <tr>
              <td><span class="badge" title="${esc(m.name)}">${esc(metricLabel(m.name))}</span></td>
              <td>${m.score != null ? pct(m.score) : '—'}</td>
              <td><span class="badge ${m.passed?'pass':'fail'}">${m.passed?'✓':'✗'}</span></td>
              <td style="white-space:normal">${esc(m.detail || '')}</td>
            </tr>`).join('')}
        </table>` : '<div class="catname">（无）</div>'}
      ${(c.skipped_metrics||[]).length ? `
        <div class="catname" style="margin-top:8px;font-size:12px">跳过：${esc((c.skipped_metrics||[]).map(metricLabel).join(', '))}</div>` : ''}
    </div>
    <div class="info-row" style="margin-top:6px">
      <span>延迟：<b>${c.latency_ms != null ? c.latency_ms.toFixed(2) + ' ms' : '—'}</b></span>
      <span>tokens：<b>${tok || '—'}</b>（prompt ${c.prompt_tokens ?? 0} / completion ${c.completion_tokens ?? 0}）</span>
      <span>成本：<b>${c.cost != null ? c.cost : '—'}</b></span>
    </div>`;
  $drawer.classList.add('open');
  $drawerMask.classList.add('open');
}
function closeDrawer() {
  $drawer.classList.remove('open');
  $drawerMask.classList.remove('open');
}
$drawerClose.addEventListener('click', closeDrawer);
$drawerMask.addEventListener('click', closeDrawer);
addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

// 测试详情页（#tests）：每条 pytest 用例的状态/含义/耗时
// 状态保存在模块顶层 closure，避免每次筛选都重算
let _testsState = null;

async function renderTests() {
  const detail = await (await fetch('api/tests-detail.json')).json();
  const summary = await (await fetch('api/tests.json')).json();
  // 最新评测报告（用于跳「用例列表」看评测输入输出）
  let latestReport = '';
  try {
    const reports = await (await fetch('api/reports.json')).json();
    latestReport = (reports && reports[0] && reports[0].file) || '';
  } catch (e) {}
  const evalLink = latestReport
    ? `<a class="back" href="#${encodeURIComponent(latestReport)}/cases">看评测输入/输出（用例列表）→</a>`
    : '';
  const tests = (detail && detail.tests) || [];
  if (!tests.length) {
    $app.innerHTML = `
      <a class="back" href="#llm">← 返回报告列表</a>
      <h1>测试详情</h1>
      <div class="card"><div class="sub" style="margin:0">暂无测试详情（api/tests-detail.json 不存在）</div></div>`;
    return;
  }

  // 文件（classname 前缀）去重 + 排序
  const files = Array.from(new Set(tests.map(t => (t.classname || '').split('.')[0] || '(其他)'))).sort();
  const summaryInfo = summary || {};

  _testsState = { tests, files, summary: summaryInfo, filter: 'all', file: '', q: '', latestReport };

  $app.innerHTML = `
    <a class="back" href="#llm">← 返回报告列表</a>
    <h1>测试详情 <span class="sub" style="font-size:13px;font-weight:400">（pytest 框架单测）</span></h1>
    <div class="sub">
      本页是框架自身的单元测试。要看的<b>评测输入 / 模型输出</b>在这里：${evalLink || '（暂无评测报告）'}
    </div>
    <div class="sub">
      ${summaryInfo.generated_at ? `生成于 ${esc(summaryInfo.generated_at)} · ` : ''}
      ${summaryInfo.total != null ? `${summaryInfo.total} 用例 · ` : ''}
      ${summaryInfo.passed != null ? `${summaryInfo.passed} 通过 · ` : ''}
      ${summaryInfo.failed != null ? `<span style="color:${summaryInfo.failed?'var(--bad)':'var(--muted)'}">${summaryInfo.failed} 失败</span> · ` : ''}
      ${summaryInfo.errors != null ? `<span style="color:${summaryInfo.failed?'var(--bad)':'var(--muted)'}">${summaryInfo.errors} 异常</span> · ` : ''}
      ${summaryInfo.skipped != null ? `${summaryInfo.skipped} 跳过 · ` : ''}
      ${summaryInfo.duration_s != null ? `耗时 ${summaryInfo.duration_s}s` : ''}
      ${summaryInfo.branch ? ` · 分支 <code>${esc(summaryInfo.branch)}</code>` : ''}
      ${summaryInfo.commit ? ` · <code>${esc((summaryInfo.commit||'').slice(0,7))}</code>` : ''}
      ${summaryInfo.run_url ? ` · <a class="back" href="${esc(summaryInfo.run_url)}" target="_blank" rel="noopener">查看运行日志 →</a>` : ''}
    </div>
    <div class="toolbar">
      <select id="testsFilter">
        <option value="all">全部状态</option>
        <option value="pass">✓ 通过</option>
        <option value="fail">✗ 失败</option>
        <option value="error">⚠ 异常</option>
        <option value="skip">— 跳过</option>
      </select>
      <select id="testsFile">
        <option value="">全部文件</option>
        ${files.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join('')}
      </select>
      <input id="testsSearch" placeholder="搜索用例名 / 类名" />
      <span class="stat-inline" id="testsCount"></span>
    </div>
    <div class="card"><table id="testsTable">
      <tr><th style="width:80px">状态</th><th>文件</th><th>用例</th><th style="width:90px">耗时</th></tr>
      <tbody id="testsBody"></tbody>
    </table></div>
    <div style="color:var(--muted);font-size:12px;margin-top:8px">点击行看完整错误信息（弹出抽屉）</div>`;

  document.getElementById('testsFilter').addEventListener('change', e => { _testsState.filter = e.target.value; _renderTestsBody(); });
  document.getElementById('testsFile').addEventListener('change', e => { _testsState.file = e.target.value; _renderTestsBody(); });
  document.getElementById('testsSearch').addEventListener('input', e => { _testsState.q = e.target.value.toLowerCase(); _renderTestsBody(); });
  _renderTestsBody();
}

function _renderTestsBody() {
  const { tests, filter, file, q } = _testsState;
  const rows = tests.filter(t => {
    if (filter !== 'all' && t.status !== filter) return false;
    if (file && (t.classname || '').split('.')[0] !== file) return false;
    if (q && !(t.name || '').toLowerCase().includes(q) && !(t.classname || '').toLowerCase().includes(q)) return false;
    return true;
  });
  _testsState.filtered = rows;  // 抽屉查找时直接索引这里
  document.getElementById('testsCount').textContent = `命中 ${rows.length} / ${tests.length} 条`;
  const badge = s => `<span class="badge ${s==='pass'?'pass':s==='fail'?'fail':s==='skip'?'skip':''}">${s}</span>`;
  document.getElementById('testsBody').innerHTML = rows.map((t, i) => `
    <tr class="rowlink" data-i="${i}">
      <td>${badge(t.status || '')}</td>
      <td class="catname">${esc((t.classname || '').split('.')[0] || '')}</td>
      <td><code>${esc(t.name || '')}</code>${t.message ? `<div class="err" style="font-size:12px;margin-top:4px">${esc(t.message.slice(0,160))}${t.message.length>160?'…':''}</div>` : ''}</td>
      <td class="catname">${(t.duration_s || 0).toFixed(3)}s</td>
    </tr>`).join('') || `<tr><td colspan="4" class="empty">无匹配用例</td></tr>`;
  document.querySelectorAll('#testsBody tr.rowlink').forEach(tr => {
    tr.addEventListener('click', () => {
      const idx = Number(tr.dataset.i);
      _openTestDrawer(_testsState.filtered[idx] || null);
    });
  });
}

// 渲染抽屉（详情用同一个 drawer 组件）
function _openTestDrawer(t) {
  if (!t) return;
  $drawerTitle.textContent = `${t.classname || ''}.${t.name || ''}`;
  // 重置 badge 类名并更新文本（避免 outerHTML 丢引用）
  $drawerBadge.className = `badge ${t.status === 'pass' ? 'pass' : t.status === 'fail' ? 'fail' : t.status === 'skip' ? 'skip' : ''}`;
  $drawerBadge.textContent = t.status || '';
  $drawerBody.innerHTML = `
    <div class="drawer-section">
      <h3>基本信息</h3>
      <div class="info-row">
        <span>文件：<b>${esc(t.classname || '')}</b></span>
        <span>用例：<b><code>${esc(t.name || '')}</code></b></span>
        <span>耗时：<b>${(t.duration_s || 0).toFixed(3)}s</b></span>
      </div>
    </div>
    ${t.message ? `
    <div class="drawer-section">
      <h3>失败详情</h3>
      <div class="mono scroll-box">${esc(t.message)}</div>
    </div>` : `
    <div class="drawer-section">
      <h3>说明</h3>
      <div class="sub" style="margin:0">pytest 单元测试通过。这是<b>框架自身的质量测试</b>，不含评测输入输出——评测的 prompt / 期望输出 / 模型实际输出请看 ${_testsState.latestReport ? `<a class="back" href="#${encodeURIComponent(_testsState.latestReport)}/cases">最新报告的用例列表 →</a>` : '报告的「用例列表」页'}。</div>
    </div>`}`;
  $drawer.classList.add('open');
  $drawerMask.classList.add('open');
}

// 全局错误兜底：把未捕获错误显示到页面顶部，避免「点不开/无反应」静默失败
function _showError(msg) {
  const banner = `<div class="errbox"><b>运行错误：</b>${esc(String(msg))}<div style="margin-top:6px;color:var(--muted);font-size:12px">按 F12 打开控制台可看堆栈。请截图给我以便定位。</div></div>`;
  // 插到 $app 顶部；不要清空原内容
  const tmp = document.createElement('div');
  tmp.innerHTML = banner;
  $app.prepend(tmp.firstChild);
}
addEventListener('error', e => _showError(e.message || (e.error && e.error.message) || 'unknown'));
addEventListener('unhandledrejection', e => _showError((e.reason && (e.reason.message || e.reason)) || 'unknown'));

// 行点击改事件委托：避开内联 onclick + 反引号 template literal 替换的引号转义陷阱
$app.addEventListener('click', e => {
  const trGo = e.target.closest && e.target.closest('tr.rowlink[data-go]');
  if (trGo) { location.hash = trGo.dataset.go; return; }
  const trCase = e.target.closest && e.target.closest('tr.rowlink[data-case]');
  if (trCase) { openDrawer(window.casesByKey.get(trCase.dataset.case)); return; }
});

// 路由入口：hash 形如 #<file>/<tab>，旧 #<file> 默认汇总；
// 空 hash = 门户页（两层入口）；#llm 模型层原报告列表；#tests 用例详情；#agent Agent 评测
async function main() {
  const hash = location.hash.slice(1);
  const isPortal = !hash || hash === '/';
  try {
    const reports = await (await fetch('api/reports.json')).json();
    if (isPortal) {
      // 门户页即使没有报告也要能出（Agent 层可能是有数据的），空态由卡片自己兜
      return renderPortal(reports);
    }
    const [file, tab = 'summary'] = hash.split('/');
    if (!file) return renderPortal(reports);
    if (file === 'llm') return renderList(reports);
    if (file === 'tests') return renderTests();
    if (file === 'agent') return renderAgent();
    return renderDetail(decodeURIComponent(file), tab);
  } catch (e) {
    _showError(`main() 失败：${e.message || e}\n（hash="${hash}", file="${hash.split('/')[0]}", tab="${hash.split('/')[2] || 'summary'}"）`);
    // 退回报告列表，避免白屏
    try { return renderList(await (await fetch('api/reports.json')).json()); } catch (e2) { _showError(`连报告列表都拉不到：${e2.message || e2}`); }
  }
}
main();
addEventListener('hashchange', () => main());
</script>
</body>
</html>"""

PAGE = (
    PAGE_TEMPLATE
    .replace("__METRIC_LABELS__", json.dumps(METRIC_LABELS, ensure_ascii=False))
    .replace("__DIM_LABELS__", json.dumps(_BANNER_DIM_LABELS, ensure_ascii=False))
    .replace("__DIMENSION_ORDER__", json.dumps(list(DIMENSION_ORDER), ensure_ascii=False))
    .replace("__CATEGORY_DIMENSION__", json.dumps(CATEGORY_DIMENSION, ensure_ascii=False))
    .replace("__LEGACY_DIM_ALIASES__", json.dumps(LEGACY_DIMENSION_ALIASES, ensure_ascii=False))
)


def _load_reports():
    """扫描 reports/*.json，返回按时间倒序的元数据列表。"""
    items = []
    for p in sorted(REPORTS_DIR.glob("*.json")):
        # CI 测试摘要/明细不是评测报告，不进列表（否则会出现 model=?/用例=0 的幽灵条目）
        if p.name in ("tests-summary.json", "tests-detail.json"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = (data.get("summary") or [{}])[0]
        m = re.search(r"\d{8}-\d{6}", p.stem)
        # 横向报告一份文件含多个模型（summary 有多行）。若只暴露 summary[0]，
        # 列表里整份报告会被显示成第一个模型（通常是被对照的 mock），
        # 让人误以为「看不到 deepseek-pro」；这里额外给出 model_rows 供前端逐模型展示。
        model_rows = [
            {
                "model": s.get("model", "?"),
                "total": s.get("total", s.get("passed", 0) + s.get("failed", 0)),
                "passed": s.get("passed", 0),
                "failed": s.get("failed", 0),
                "pass_rate": s.get("pass_rate", 0.0),
                "stability": s.get("stability"),
                "p95": s.get("p95_latency_ms"),
            }
            for s in (data.get("summary") or [])
            if isinstance(s, dict)
        ] or [{
            "model": summary.get("model", "?"),
            "total": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "pass_rate": summary.get("pass_rate", 0.0),
            "stability": summary.get("stability"),
            "p95": summary.get("p95_latency_ms"),
        }]
        # repeat：老报告（无该字段）按 1 处理 = 单次回归测试
        repeat = int(data.get("repeat", 1) or 1)
        entry = {
            "file": p.name,
            "time": (m.group(0) if m else data.get("started_at", "?")),
            "tag": p.stem.split("-")[0] if "-" in p.stem else "run",
            "model": summary.get("model", "?"),
            "models": [row["model"] for row in model_rows],
            "model_rows": model_rows,
            "case_count": data.get("case_count", 0),
            "repeat": repeat,
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "pass_rate": summary.get("pass_rate", 0.0),
            "stability": data.get("stability"),
            "p95": summary.get("p95_latency_ms"),
        }
        # 维度速览：取代表模型（跳过 mock 对照组）的各维度通过率。
        # 老报告没有 dimensions 段时按 categories 重算，保证与新报告同口径。
        entry["dim_model"] = _pick_representative_model(entry)
        entry["dimensions"] = _dimension_summary(data, entry["dim_model"])
        entry["dimension_coverage"] = [row["label"] for row in entry["dimensions"]]
        items.append(entry)
    items.sort(key=lambda x: _time_key(x["time"]), reverse=True)
    return items


def _time_key(value: str):
    """把各种 time 字符串转成 datetime，兼容文件名数字串与 ISO 字符串。"""
    from datetime import datetime
    candidate = str(value).strip()
    if not candidate or candidate == "?":
        return datetime.min
    # 数字串 20260921-074502 → 当天时间
    try:
        return datetime.strptime(candidate, "%Y%m%d-%H%M%S")
    except ValueError:
        pass
    # ISO 格式 2026-09-21T15:59:26
    cleaned = candidate.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return datetime.min


def _load_tests_summary():
    """读取 CI 生成的 pytest 摘要（dashboard/data/tests-summary.json）。

    由 .github/workflows/cd.yml 的 "解析测试摘要" 步骤生成，零依赖。
    文件不存在或读取失败返回 None（看板会显示空状态而非崩页）。
    """
    p = REPORTS_DIR / "tests-summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _agent_stamp(name: str):
    """从 result-20261001-103426.json / agent-eval-20261001-103426.html 里取时间戳。"""
    m = re.search(r"\d{8}-\d{6}", str(name))
    return m.group(0) if m else None


def _agent_entry(path: Path, archived: bool):
    """读一份 Agent 评测结果，归一化成看板要的字段；读不动返回 None（跳过）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    summary = data.get("summary") or {}
    repeat = int(data.get("repeat") or 1)
    entry = {
        "file": path.name,
        "archived": archived,
        "stamp": _agent_stamp(path.name),
        "time": _agent_stamp(path.name) or data.get("finished_at") or data.get("started_at") or "?",
        "repeat": repeat,
        "total": summary.get("任务总数"),
        "passed": summary.get("通过"),
        "failed": summary.get("失败"),
        "invalid": summary.get("无效"),
        "pass_rate": summary.get("任务成功率"),
        "tool_rate": summary.get("工具选择正确率"),
        "avg_steps": summary.get("平均步数"),
        "trap_hold": summary.get("幻觉题是否守住"),
        "traps": summary.get("幻觉题") or [],
        "report": None,        # agent_eval/reports/ 里配上的 HTML 报告（配不上为 None）
        "report_exact": False,  # 时间戳是否精确匹配（False = 取该结果之后生成的最新一份）
    }
    # 逐题一致性只有 repeat>1 才有；单遍结果没有可比的对象，不塞空数组误导
    rs = data.get("repeat_summary")
    if repeat > 1 and isinstance(rs, dict):
        entry["repeat_summary"] = rs
    return entry


def _pair_agent_reports(runs: list):
    """把 agent_eval/reports/ 里的 HTML 报告配到结果行上（就地改 runs）。

    两级配对：
    1. **时间戳完全相同**（`result-20261001-103426.json` ↔ `agent-eval-20261001-103426.html`）
    2. 配不上时，取「该结果之后生成的最新一份报告」——报告总是在结果之后生成的。
       一份报告只认一份结果（认最近的那次），避免多行指向同一份报告。

    配不上就留空、前端显示「—」：宁可少一个链接，也不要链到一份不是这次的报告。
    """
    if not AGENT_REPORTS_DIR.is_dir():
        return
    reports = sorted(AGENT_REPORTS_DIR.glob("agent-eval-*.html"), key=lambda p: p.name, reverse=True)
    claimed = set()
    for rep in reports:
        rep_stamp = _agent_stamp(rep.name)
        if not rep_stamp:
            continue
        candidates = [
            r for r in runs
            if (r.get("stamp") or "") <= rep_stamp and r["file"] not in claimed
        ]
        if not candidates:
            continue
        # runs 已按时间倒序：exact 优先，其次取时间戳不晚于报告的最新一份
        exact = [r for r in candidates if r.get("stamp") == rep_stamp]
        target = (exact or candidates)[0]
        target["report"] = rep.name
        target["report_exact"] = bool(exact)
        claimed.add(target["file"])


def load_agent_runs() -> list:
    """扫描 agent_eval/results/（含 archive/）下的 result-*.json，按时间倒序。

    **INVALID 一律不返回**：那是作废存档，进看板就是误导（详见 agent_eval/results/README.md）。
    目录不存在（如部署沙盒里没有 agent_eval/）返回空列表，看板显示空状态而不是崩。
    """
    if not AGENT_RESULTS_DIR.is_dir():
        return []
    paths = [(p, False) for p in AGENT_RESULTS_DIR.glob("result-*.json")]
    if AGENT_ARCHIVE_DIR.is_dir():
        paths += [(p, True) for p in AGENT_ARCHIVE_DIR.glob("result-*.json")]

    runs = []
    for path, archived in paths:
        if AGENT_INVALID_MARK in path.name:
            continue
        entry = _agent_entry(path, archived)
        if entry:
            runs.append(entry)
    runs.sort(key=lambda x: _time_key(x["time"]), reverse=True)
    _pair_agent_reports(runs)
    return runs


def _load_tests_detail():
    """读取 pytest 用例明细（dashboard/data/tests-detail.json）。

    由 .github/workflows/cd.yml 的 "解析测试摘要" 步骤用 parse_junit --detail 写出。
    文件不存在或读取失败返回 None（看板会显示空状态而非崩页）。
    """
    p = REPORTS_DIR / "tests-detail.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


class Handler(SimpleHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(self.path)
        # 兼容前端相对路径（GitHub Pages 子路径下绝对路径会 404）：浏览器发起
        # 的相对请求 `api/...` 在 HTTPServer 里通常带有前导斜杠，但偶尔（如某些
        # 反代）会缺，统一补成 `/api/...`。
        if path.startswith("api/"):
            path = "/" + path
        if path == "/" or path == "/index.html":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/api/reports", "/api/reports.json"):
            self._json(_load_reports())
        elif path in ("/api/tests", "/api/tests.json"):
            self._json(_load_tests_summary() or {})
        elif path in ("/api/tests/detail", "/api/tests/detail.json", "/api/tests-detail.json"):
            self._json(_load_tests_detail() or {})
        elif path in ("/api/agent", "/api/agent.json"):
            runs = load_agent_runs()
            self._json({"latest": runs[0] if runs else None, "runs": runs})
        elif path.startswith("/api/agent/report/"):
            # Agent 评测的 HTML 报告：走 API 而不是直接链 agent_eval/ 下的文件，
            # 这样动态服务与静态导出（build_static 会把它写进 dist/api/agent/report/）
            # 用的是同一个相对地址，静态托管下也不会变成死链。
            name = os.path.basename(path[len("/api/agent/report/"):])
            fp = AGENT_REPORTS_DIR / name
            if fp.suffix == ".html" and fp.is_file():
                try:
                    body = fp.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as e:
                    self._json({"error": str(e)}, 500)
            else:
                self._json({"error": "not found"}, 404)
        elif path.startswith("/api/report/"):
            name = os.path.basename(path[len("/api/report/"):])
            fp = REPORTS_DIR / name
            if fp.suffix == ".json" and fp.is_file():
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    # 汇总 tab 结论 + 顶部属性条由后端算好，前端只渲染
                    # （与静态导出保持一致，且逻辑可被单测覆盖）
                    if isinstance(data, dict):
                        data["_conclusion"] = compute_conclusion(data)
                        data["_attributes"] = compute_attributes(data)
                    self._json(data)
                except Exception as e:
                    self._json({"error": str(e)}, 500)
            else:
                self._json({"error": "not found"}, 404)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main():
    # 必须是多线程：首页会并发请求 reports.json / tests.json / 多份 report.json，
    # 单线程 HTTPServer 遇到 keep-alive 连接会串行排队，第二次打开页面就容易卡死。
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"评测实验室看板已启动: http://0.0.0.0:{PORT}  (数据源: {REPORTS_DIR})")
    server.serve_forever()


if __name__ == "__main__":
    main()
