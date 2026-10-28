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
# 出题文件：**只读**。考点文案（design_note）与判据修订留痕（meta.v2_changes）从这里搬，
# 判据本体一个字不许动。
AGENT_TASKS_FILE = AGENT_EVAL_DIR / "tasks.json"

# RAG 评测（rag_eval/）的数据位置——只读，不写、不改。
# 与 Agent 层同理单独扫目录：RAG 层的「通过率」是「检索找对 + 生成忠于资料 + 不编造」，
# 和模型层的答对率、Agent 层的调对工具率是三件事，混进一张表就互相污染口径。
RAG_EVAL_DIR = ROOT / "rag_eval"
RAG_RESULTS_DIR = RAG_EVAL_DIR / "results"
RAG_ARCHIVE_DIR = RAG_RESULTS_DIR / "archive"

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
# 维度口径版本：与 src/datasets/schema.py 的 DIMENSION_SCHEMA_VERSION 一一对应
# （tests/test_dashboard_dimensions.py 守护两侧同步）。报告里写明它，看板据此判断
# 「这份维度数是按哪套口径算的」——缺这个字段的就是老报告，看板**标注旧口径**，
# 而不是自己再算一份新的（Q8 裁决）。
DIMENSION_SCHEMA_VERSION = "v2"
# 用例没打标签 / 维度名认不出时归入这一组，页面显示「未标注」
UNTAGGED_DIMENSION = 'untagged'
KNOWN_DIMENSIONS = frozenset(DIMENSION_ORDER) | {UNTAGGED_DIMENSION}


def _dimension_key(raw):
    """把报告里的维度名归一成四类之一；**认不出的归入「未标注」，绝不新造维度名**。

    Q8 裁决：分类 ≠ 维度。以前这里认不出就**原样返回**（还拿分类名换算），于是
    数据集名 ``rag_cases`` 直接冒充成了一个维度，页面多出一行假的「rag_cases」。
    现在只认白名单（四类 + 未标注）与旧名别名，其余一律进「未标注」。
    """
    key = str(raw or '').strip().lower()
    if not key:
        return UNTAGGED_DIMENSION
    if key in LEGACY_DIMENSION_ALIASES:
        return LEGACY_DIMENSION_ALIASES[key]
    if key in KNOWN_DIMENSIONS:
        return key
    return UNTAGGED_DIMENSION


def _case_dimension(case):
    """单条用例归到哪个维度：**只用用例自带的 ``dimension`` 标签，不按分类重算**。

    分类是「这道题来自哪个数据集」，维度是「这道题测什么能力」——两回事。按分类
    重算会让页面算出一份与报告不同的数（Q8：报告 correctness=51、页面 42）。
    用例没打标签或标签认不出时归入「未标注」。
    """
    if not isinstance(case, dict):
        return None
    return _dimension_key(case.get('dimension'))


def _dimension_schema_legacy(doc):
    """这份报告的维度口径是不是旧版（页面该标注、读者不该拿它跟新报告横向比）。

    报告里写明口径版本（``dimension_schema_version``）；缺这个字段或版本对不上
    的就是老报告。此时页面**照抄**报告里存的维度数并标注「口径为旧版」，
    绝不自己按分类重算一份新数（那正是 Q8 的病根）。
    """
    if not isinstance(doc, dict):
        return True
    return doc.get('dimension_schema_version') != DIMENSION_SCHEMA_VERSION


def _dimension_summary(doc, model):
    """算一份报告的「各维度通过率」行——**只读报告里的 dimensions 段，不重算**。

    Q8 裁决：维度统计**只算一次**，落在报告 JSON（``src/runner/results.py`` 按用例
    自带的 dimension 标签聚合）；看板原样显示。以前这里按 categories 重算（本意是
    迁就老报告、让它们也用新口径），结果是报告存一份、页面算一份，同一份报告同一
    模型对不上（报告 correctness=51、页面 42），而且未知分类（rag_cases）被当成
    维度键，冒出一个假维度。

    只有报告**连 dimensions 段都没有**（更老的报告）时才从 cases[] 兜底算——那份
    数是页面算的，前端会标注「口径为旧版」（见 ``_dimension_schema_legacy``），
    不让读者误以为它是当前口径的数。

    返回 ``[{dimension, label, total, passed, pass_rate}]``，按 DIMENSION_ORDER
    排序；什么都取不到时返回空列表（前端不渲染该行，而不是画一排 0%）。
    """
    if not isinstance(doc, dict):
        return []

    def build_row(dim, total, passed):
        return {
            'dimension': dim,
            'label': _BANNER_DIM_LABELS.get(dim, dim),
            'total': total,
            'passed': passed,
            'pass_rate': (passed / total) if total else 0.0,
        }

    rows: list[dict] = []
    # 真相源：报告里存好的维度统计，照抄
    for row in (_candidates_for(doc.get('dimensions'), model) or []):
        if not isinstance(row, dict):
            continue
        dim = _dimension_key(row.get('dimension'))
        if not dim:
            continue
        if row.get('total'):
            rows.append(build_row(dim, int(row.get('total', 0)), int(row.get('passed', 0))))
        else:
            # 只有通过率、没留样本量：原样保留这一行，别因为缺字段整份丢掉
            rows.append({
                'dimension': dim,
                'label': _BANNER_DIM_LABELS.get(dim, dim),
                'total': None,
                'passed': None,
                'pass_rate': float(row.get('pass_rate') or 0.0),
            })

    # 更老的报告连 dimensions 段都没有：按用例自带标签兜底（前端标注「口径为旧版」）
    if not rows:
        agg: dict[str, list[int]] = {}
        for case in (doc.get('cases') or []):
            if not isinstance(case, dict):
                continue
            if model and case.get('model') and case.get('model') != model:
                continue
            dim = _case_dimension(case)
            if not dim:
                continue
            bucket = agg.setdefault(dim, [0, 0])
            bucket[0] += 1
            bucket[1] += 1 if case.get('passed') is True else 0
        rows = [build_row(dim, counts[0], counts[1]) for dim, counts in agg.items()]

    order = {name: index for index, name in enumerate(DIMENSION_ORDER)}
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
    'safety_judge': '安全裁判',
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

    # 污染报告：这次跑的数据集范围本身是错的（例：模型层报告混进了别层的用例），
    # 连「测了什么」都不成立，由它算出的排名与异常句全是误导——典型是拿一个
    # 本不该出现的分类判成「某分类明显偏低」。这种报告留着留证，但**不许产出结论**。
    if report.get('contaminated'):
        return {
            'rank': [],
            'judge': None,
            'anomalies': [report.get('contamination_note') or '本报告已被标注为污染数据'],
            'text': '⚠️ 本报告为污染数据，不进任何结论（报告本身保留，仅作事故留证）',
            'contaminated': True,
        }

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

    与 ``_dimension_summary`` 同一个规矩（Q8 裁决）：**能照抄就照抄**——真相源是
    报告里的 ``dimensions`` 段；连这段都没有（更老的报告）才按用例自带标签兜底。
    """
    keys = set()
    for row in (_candidates_for(doc.get('dimensions'), None) or []):
        if isinstance(row, dict):
            key = _dimension_key(row.get('dimension'))
            if key:
                keys.add(key)

    if not keys:
        for case in (doc.get('cases') or []):
            key = _case_dimension(case)
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
  /* Agent 逐题明细：一题一行，点开看失败原因 / 判据修订留痕 / 逐步轨迹 */
  .agcase { border-top: 1px solid rgba(42,54,80,.5); }
  .agsrow { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 9px 10px;
            cursor: pointer; list-style: none; font-size: 13px; }
  .agsrow::-webkit-details-marker { display: none; }
  .agsrow::before { content: '▸ '; color: var(--muted); }
  .agcase[open] > .agsrow::before { content: '▾ '; }
  .agsrow:hover { background: var(--panel2); }
  .agcase[open] > .agsrow { background: var(--panel2); }
  .agid { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  .agpoint { color: var(--muted); font-size: 12px; flex: 1 1 240px; min-width: 160px; }
  .badge.agtrap { background: rgba(245,180,73,.18); color: var(--warn); }
  .agdetail { padding: 6px 14px 14px; font-size: 13px; line-height: 1.8; background: rgba(29,39,64,.35); }
  .agsec { margin-top: 6px; }
  .aglab { display: inline-block; min-width: 92px; color: var(--muted); font-size: 12px; vertical-align: top; }
  .agsec > .agbody { display: inline-block; max-width: calc(100% - 100px); white-space: normal; }
  .agfail { color: var(--bad); }
  .agv2 { color: var(--warn); }
  .agtraj { margin-top: 6px; }
  .agstep { border-left: 2px solid var(--border); padding: 2px 0 2px 10px; margin-bottom: 6px; }
  .agstepn { color: var(--muted); font-size: 12px; }
  .agthink, .agobs { color: var(--muted); white-space: pre-wrap; word-break: break-word; }
  .agcall { color: var(--accent); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 12px; word-break: break-all; }
  /* 详情页顶部「本报告结论」 */
  .conclusion { border-left: 3px solid var(--accent); }
  /* 污染报告：结论区换成警示赭色，别让人把它当成一条正常成绩结论读下去 */
  .conclusion.concl-warn { border-left-color: var(--bad); background: rgba(239,68,68,.08); }
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
  /* 卡片三段式：被测对象 / 怎么测 / 最新成绩 */
  .psec { margin-top: 10px; font-size: 13px; line-height: 1.7; }
  .plabel { display: inline-block; min-width: 62px; color: var(--muted); font-size: 12px; }
  .pchip { display: inline-block; background: var(--panel2); border: 1px solid var(--border);
           border-radius: 8px; padding: 2px 8px; margin: 0 4px 4px 0; }
  .pwhen { color: var(--muted); font-size: 12px; margin-left: 6px; }
  .pgo { color: var(--accent); font-size: 13px; margin-top: 14px; }
  .pfoot { color: var(--muted); font-size: 12px; margin-top: 14px; text-align: center; }
  /* h1 下的定位语：先把调门降到「自学实战项目」，免得访问者按生产框架的预期来读 */
  .plead { margin: 6px 0 10px; font-size: 14px; color: var(--muted); }
  /* 定位与局限：三组等宽，访问者一眼看完规模 / 没做什么 / 做到了什么 */
  .pscope { margin-top: 18px; background: var(--panel); border: 1px solid var(--border);
            border-radius: 12px; padding: 16px 20px; }
  .pscopetitle { font-weight: 600; font-size: 14px; margin-bottom: 10px; }
  .pscopegrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
  .pscopeitem .plabel { display: block; min-width: 0; margin-bottom: 2px; }
  .pvalue { font-size: 13px; line-height: 1.7; }

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
  /* 自由文本单元格（RAG 回答、问题原文）：全局 td 是 nowrap，长回答会把表格撑出
     横向滚动条、把其它列挤变形。这里显式允许换行 + 限宽，超出部分按字断行。 */
  .wrapcell { white-space: pre-wrap; word-break: break-word; max-width: 30em; }
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
// 缺字段/除零时显示「—」而不是 NaN%：页面上一个 NaN 会让整张卡的可信度归零，
// 而「没这个数」和「这个数是 0」是两回事，必须分开表达。
const pct = x => (Number.isFinite(Number(x)) ? (Number(x) * 100).toFixed(1) + '%' : '—');
// 维度中文标签 / 分类→维度映射：由 Python 侧注入（与 _BANNER_DIM_LABELS /
// CATEGORY_DIMENSION 是同一份数据），首页速览、详情页维度表、banner 共用一套口径
const DIM_LABELS = __DIM_LABELS__;
const CATEGORY_DIMENSION = __CATEGORY_DIMENSION__;
const LEGACY_DIM_ALIASES = __LEGACY_DIM_ALIASES__;
const DIMENSION_ORDER = __DIMENSION_ORDER__;
const KNOWN_DIMENSIONS = __KNOWN_DIMENSIONS__;
const DIM_SCHEMA_VERSION = __DIMENSION_SCHEMA_VERSION__;
// 这份报告的维度口径是不是旧版（报告里写明版本，缺字段的就是老报告）。
// 旧口径**照抄报告里的数**并显式标注，绝不重算一份新的（Q8 裁决）。
const isLegacyDimSchema = doc => !doc || doc.dimension_schema_version !== DIM_SCHEMA_VERSION;
const legacyDimBadge = legacy => (legacy
  ? ' <span class="badge" title="这份报告生成时用的维度口径与当前不同，数字照抄报告原文，未重算">口径为旧版</span>'
  : '');
const legacyDimNote = doc => legacyDimBadge(isLegacyDimSchema(doc));
const dimLabel = d => d ? (DIM_LABELS[d] || d) : '未标注';
// 维度：只认四类白名单 + 「未标注」。**分类不是维度**（Q8 裁决）——以前认不出就
// 原样返回、还拿分类名换算，于是数据集名 rag_cases 冒充成了一个维度。
const dimKeyOf = raw => {
  const k = String(raw || '').trim().toLowerCase();
  if (!k) return 'untagged';
  if (LEGACY_DIM_ALIASES[k]) return LEGACY_DIM_ALIASES[k];
  return KNOWN_DIMENSIONS.includes(k) ? k : 'untagged';
};
// 单条用例归到哪个维度：**只用用例自带的 dimension 标签**，不按分类重算——
// 重算会让页面算出一份与报告不同的数（Q8：报告 correctness=51、页面 42）。
const caseDimOf = c => dimKeyOf(c && c.dimension);
// 指标中文标签：由后端 METRIC_LABELS 注入（与 Python 侧同一份数据），未收录的原样显示
const METRIC_LABELS = __METRIC_LABELS__;
const metricLabel = m => METRIC_LABELS[m] || m;
const rateColor = x => x >= 0.8 ? 'var(--ok)' : x >= 0.5 ? 'var(--warn)' : 'var(--bad)';
const rateInner = x => `<div style="display:flex;align-items:center;gap:8px"><span class="rate" style="color:${rateColor(x)}">${pct(x)}</span><span class="bar"><i style="width:${(x*100).toFixed(1)}%;background:${rateColor(x)}"></i></span></div>`;
const rateCell = x => `<td>${rateInner(x)}</td>`;

// 用例的「能力指标」：这条题测的是**什么能力**（四类之一）。
// 与「判定方式」（怎么判：contains / judge …）是两个层次，必须分开两列——
// 把判定器名摆在能力指标的位置上，等于告诉读者「这道考 contains」，那是把实现细节
// 当成了测试意图（HELM / Open LLM Leaderboard / DeepEval 都按「能力名」组织指标）。
// 判定方式列**保留**：它兼着「为什么判失败」的解释作用。
const CASE_DIM_UNKNOWN = '未标注';
const caseDimLabel = c => {
  const key = caseDimOf(c);
  return key === 'untagged' ? CASE_DIM_UNKNOWN : (DIM_LABELS[key] || String(key || CASE_DIM_UNKNOWN));
};
function caseDimCell(c) {
  const untagged = caseDimOf(c) === 'untagged';
  const label = caseDimLabel(c);
  // 未标注用灰字而不是徽章：这是「还没打标签」的提示，不是一个已确定的能力结论
  return untagged
    ? `<td class="catname">${esc(label)}</td>`
    : `<td><span class="badge">${esc(label)}</span></td>`;
}

// 用例的「判定方式」：这条结果是用哪些指标判出来的（中文标签，多指标全部列出）。
// 安全集现在一半走安全裁判、一半走排除检查，列表里看不出区别就没法判断成绩怎么来的
// ——判定方式透明化是评测工具的基本操守，所以单独成列。
function caseMethodLabels(c) {
  return ((c && c.metrics) || []).map(m => m && m.name).filter(Boolean);
}
// 没有指标（老报告 / 整条跳过）显示「—」，不编造判定方式
function caseMethodCell(c) {
  const names = caseMethodLabels(c);
  if (!names.length) return '<td class="catname">—</td>';
  return `<td style="white-space:normal"><span style="display:inline-flex;flex-wrap:wrap;gap:4px">${
    names.map(n => `<span class="badge" title="${esc(n)}">${esc(metricLabel(n))}</span>`).join('')
  }</span></td>`;
}

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
    + `<span class="dimcover">覆盖：${esc(coverage)}${legacyDimBadge(r.dimension_schema_legacy)}</span></td></tr>`;
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

  // 真相源：报告里存好的维度统计，照抄（Q8 裁决：页面不许重算）
  const precounted = [];
  (pickGroup(doc.dimensions) || []).forEach(row => {
    if (!row || !row.dimension) return;
    const dim = dimKeyOf(row.dimension);
    if (Number(row.total) || 0) { add(row.dimension, row.total, row.passed); return; }
    precounted.push({
      dimension: dim, label: dimLabel(dim), total: null, passed: null,
      pass_rate: Number(row.pass_rate) || 0,
    });
  });
  // 更老的报告连 dimensions 段都没有：按用例自带标签兜底（页面会标注「口径为旧版」）
  if (!agg.size && !precounted.length) {
    (doc.cases || []).forEach(c => {
      if (!c) return;
      if (model && c.model && c.model !== model) return;
      add(c.dimension, 1, c.passed === true ? 1 : 0);
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
  // 数据源 = 成绩源报告序列（含真实模型的最新一份），与门户卡片同源。
  // 用 reports[0] 会取到 cd.yml 的自检报告（最新但全是 mock），页头就成了假数字。
  const pool = scoringReports(reports);
  if (!pool.length) return null;      // 一份真实报告都没有 → banner 不渲染，不拿 mock 充数
  const curr = pool[0];
  const base = pool[1] || null;

  // 静默 fetch；失败让 banner 不显示而不是让首页崩
  const safeFetchDoc = (entry) => entry
    ? fetch('api/report/' + encodeURIComponent(entry.file))
        .then(r => r && r.ok ? r.json() : null)
        .catch(() => null)
    : Promise.resolve(null);

  const [currDoc, baseDoc] = await Promise.all([safeFetchDoc(curr), safeFetchDoc(base)]);

  // 所有真实模型的成绩都写出来，不再只挑一个「代表模型」——多模型的价值就在对比
  const realRows = modelsOf(curr).filter(r => !isMockModel(r.model));
  const models = realRows.map(r => {
    const rate = (r && typeof r.pass_rate === 'number') ? r.pass_rate : null;
    // base 报告里能找到同一模型才算「较上次」，否则显示 —
    let deltaPt = null;
    if (base) {
      const b = modelsOf(base).find(x => x.model === r.model);
      if (b && typeof b.pass_rate === 'number' && rate != null) deltaPt = (rate - b.pass_rate) * 100;
    }
    return { model: r.model, rate: rate == null ? curr.pass_rate : rate, deltaPt };
  });

  // 跨次对比复用 diffCases（已有 helper）
  const diff = (baseDoc && currDoc) ? diffCases(currDoc.cases || [], baseDoc.cases || []) : null;
  const regressed = diff ? diff.regressed.length : 0;
  const fixed = diff ? diff.fixed.length : 0;

  // 最弱维度：走 dimensionSummary（categories 重算优先），与速览同一口径。
  // 不能直接读 currDoc.dimensions —— 那是报告生成时的旧口径快照。
  // 多模型时逐个真实模型都算一遍，取最低的那个并标出模型名（否则不知道是谁弱）。
  let weakest = null;
  if (currDoc) {
    for (const row of realRows) {
      const dimRows = dimensionSummary(currDoc, row.model);
      const dimPool = dimRows.filter(d => d.dimension !== 'untagged');
      const usable = dimPool.length ? dimPool : dimRows;
      if (!usable.length) continue;
      const w = usable.reduce((a, b) => (a.pass_rate <= b.pass_rate ? a : b));
      if (!weakest || w.pass_rate < weakest.rate) {
        weakest = { label: `${row.model} · ${w.label}`, rate: w.pass_rate };
      }
    }
  }

  return { time: curr.time, time_display: curr.time_display, models, regressed, fixed, weakest, hasBase: !!base };
}

function renderBanner(data) {
  if (!data) return '';
  // 每个真实模型一行：「模型名 93.9%（较上次 +0.5 pt）」，不是区间——区间把对比藏起来了
  const deltaHtml = (d) => {
    if (d == null) return '<span class="delta-na">—</span>';
    const cls = d > 0 ? 'delta-up' : (d < 0 ? 'delta-down' : 'delta-na');
    const sign = d > 0 ? '+' : '';
    return `<span class="${cls}">${sign}${d.toFixed(1)} pt</span>`;
  };
  const modelsHtml = (data.models || []).map(m =>
    `<b style="color:var(--accent)">${esc(m.model)}</b> `
    + `<span class="big" style="color:${rateColor(m.rate)}">${(m.rate * 100).toFixed(1)}%</span>`
    + `（较上次 ${deltaHtml(m.deltaPt)}）`
  ).join(' <span class="catname">/</span> ');
  const weakestHtml = data.weakest
    ? `${esc(data.weakest.label)} <b>${(data.weakest.rate * 100).toFixed(1)}%</b>`
    : '<span class="delta-na">—</span>';
  return `
    <div class="banner">
      最新真实评测 · ${esc(data.time_display || '')}：${modelsHtml}
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
  const ds = (a && a.data_source) || {};
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
      <div class="hint">最新一次：${esc(latest.time_display)}${latest.archived ? ' <span class="badge">历史归档</span>' : ''}
        · 大脑 <b>${esc(latest.brain || 'deepseek-chat')}</b>
        · ${esc(latest.file)}${latest.report ? ` · ${agentReportLink(latest)}` : ''}</div>
      <div class="grid" style="margin-bottom:4px">
        ${statHtml('任务成功率', `${pct(latest.pass_rate || 0)}<div class="k">${latest.passed ?? 0} / ${latest.total ?? 0} 通过</div>`, rateColor(latest.pass_rate || 0))}
        ${statHtml('工具选择正确率', pct(latest.tool_rate || 0), rateColor(latest.tool_rate || 0))}
        ${statHtml('平均步数', esc(latest.avg_steps ?? '—'))}
        ${statHtml('陷阱题防线', holdBadge)}
      </div>
      ${agentRepeatCard(latest)}
    </div>
    ${agentTaskList(latest, ds)}
    <div class="card">
      <div style="font-weight:600;margin-bottom:4px">历史结果（${runs.length} 次）</div>
      <table>
        <tr><th>时间</th><th>大脑</th><th>通过率</th><th>repeat</th><th>状态</th><th>报告</th></tr>
        ${runs.map(r => `
          <tr>
            <td class="catname">${esc(r.time_display)}${r.archived ? ' <span class="badge">归档</span>' : ''}</td>
            <td><span class="badge">${esc(r.brain || 'deepseek-chat')}</span></td>
            ${rateCell(r.pass_rate || 0)}
            <td><span class="badge">${esc(r.repeat)} 遍</span></td>
            <td>${Number(r.failed || 0) === 0 && Number(r.invalid || 0) === 0
                ? `<span class="badge pass">全通过</span>`
                : `<span class="badge fail">${esc(r.failed ?? 0)} 失败${Number(r.invalid || 0) ? ` · ${esc(r.invalid)} 无效` : ''}</span>`}
              <span class="catname" style="margin-left:6px">${esc(r.passed ?? 0)}/${esc(r.total ?? 0)}</span></td>
            <td>${agentReportLink(r)}</td>
          </tr>`).join('') || `<tr><td colspan="6" class="empty">暂无结果</td></tr>`}
      </table>
      <div class="hint" style="padding:10px 12px 0">「大脑」= 这次驱动 agent 的模型（<code>--model</code>），
        不同大脑的成绩各占一行、不合并：同一个 agent 换脑子跑出来的才是可比对比。
        带 <code>-INVALID-</code> 的作废存档不进看板；
        报告链接按时间戳配对（精确匹配优先，配不上取该结果之后生成的最新一份），配不上显示「—」。</div>
    </div>` : `
    <div class="card"><div class="sub" style="margin:0">暂无结果。先跑
      <code>python agent_eval/run_eval.py</code> 生成 <code>agent_eval/results/result-*.json</code>。</div></div>`}`;
}

// RAG 评测页（#rag）：与模型层、Agent 层并列的第三层。
// 这里要看三件事：检索找没找对（recall@k）、生成有没有忠于资料、资料里没有时会不会编。
// 逐题把「它依据了哪几段」摆出来，是因为 RAG 的错误有两种完全不同的病因——
// 没检索到（检索层的锅）和检索到了没用好（生成层的锅），混在一起看就没法归因。
// 「检索后端」卡片：后端降级时必须显式标出来（kb-007 口径：降级不是沉默的），
// 但**异常原文不上卡片**——`ModuleNotFoundError: No module named ...` 是排查信息，
// 只在报告 JSON 的 notes 里。读者要看见的是「现在是字面匹配、不是语义结论」。
function ragRetrieverStat(latest) {
  // statHtml 是各视图函数里的局部定义，这里自包含渲染同样的结构，不依赖外部
  const degraded = !!latest.retriever_degraded;
  const label = latest.retriever_label || (degraded ? '字符相似度（降级）' : '—');
  const hint = degraded
    ? `<div class="k">${esc(latest.retriever_hint || '未安装向量库 sentence-transformers，本次检索为字面匹配、非语义结论')}</div>`
    : '';
  const badge = degraded ? ' <span class="badge" style="background:#b45309">降级</span>' : '';
  const v = `<span style="font-size:13px">${esc(label)}</span>${badge}${hint}`;
  return `<div class="stat"><div class="k">${esc('检索后端')}</div><div class="v">${v}</div></div>`;
}

function ragProvenanceLine(p) {
  if (!p || !Object.keys(p).length) return '';
  const bits = [];
  if (p.frozen_at) bits.push('冻结于 ' + esc(p.frozen_at));
  if (p.status) bits.push(esc(p.status));
  if (p.source) bits.push('来源：' + esc(p.source));
  return `<div class="hint" style="padding:10px 12px 0">知识库 provenance：${bits.join(' · ')}</div>`;
}

// 回答是自由文本（可能几百字、带换行、带 Markdown 符号），直接塞进 nowrap 的 td
// 会把表格撑出横向滚动条。做法：限宽 + 允许换行 + 超 140 字截断，全文放 title 悬停看。
function ragAnswerCell(c) {
  const full = String(c.answer || '');
  if (!full) return '<td class="catname">—</td>';
  const shown = full.length > 140 ? full.slice(0, 140) + '…' : full;
  return `<td class="wrapcell catname" title="${esc(full)}">${esc(shown)}</td>`;
}

// 考点（想测什么能力）——展示用的中文名，取值白名单与 rag_eval 侧一致
const FOCUS_LABELS = {end_to_end: '端到端命中', faithfulness: '忠实度', honesty: '无答案诚实'};
const focusTag = f => FOCUS_LABELS[f] || '<span class="catname">—</span>';

// 顶部「指标汇总」四项：这里只摆**结果数字**。
// 考点（想测什么）、判定方式（怎么判）、指标（结果）是三个层次，不许再让判定器名
// 占着「指标」的位置——那是把它们混为一谈的源头。
function ragMetricStat(m) {
  if (!m) return '';
  if (m.skipped) {
    // 跳过必须写得跟通过率一样醒目：给 0 或 100% 都会被误读成「这家伙跑出来了」
    // 跳过必须写得跟通过率一样醒目：给 0 或 100% 都会被误读成「这家伙跑出来了」；
    // 文案与 rag_eval 侧的 JUDGE_MISSING_TEXT 一致（数据里没给 reason 时用它兜底）
    return `<div class="stat"><div class="k">${esc(m.label || '')}</div>
      <div class="v" style="color:#b45309;font-size:15px">${esc(m.reason || '跳过（未配置裁判模型）')}</div>
      <div class="k">judge 缺席，不计入通过率</div></div>`;
  }
  const hasValue = Number.isFinite(Number(m.value));
  return `<div class="stat"><div class="k">${esc(m.label || '')}</div>
    <div class="v"${hasValue ? ` style="color:${rateColor(m.value)}"` : ''}>${hasValue ? pct(m.value) : '—'}</div>
    <div class="k">${esc(m.passed ?? 0)} / ${esc(m.scored ?? 0)}</div></div>`;
}

function ragMetricsSummaryCard(ms) {
  const keys = ['recall_at_k', 'end_to_end', 'honesty', 'faithfulness'];
  const has = keys.some(k => ms && ms[k]);
  if (!has) {
    return `<div class="card">
      <div style="font-weight:600;margin-bottom:4px">指标汇总</div>
      <div class="hint">本次结果文件没有指标汇总（旧格式）——重跑一次 RAG 评测即可生成。没有数就写「没有」，不给假数字。</div>
    </div>`;
  }
  return `
    <div class="card">
      <div style="font-weight:600;margin-bottom:4px">指标汇总（结果数字）</div>
      <div class="hint">「考点 = 想测什么能力」「判定方式 = 怎么判」「指标 = 结果数字」是三个层次：
        这一栏只看数字，三件事在下方逐题明细里分三列列出。跳过不计入分母，也会显式写出。</div>
      <div class="grid" style="margin-top:6px">${keys.map(k => ragMetricStat(ms[k])).join('')}</div>
    </div>`;
}

function ragCaseRow(c) {
  const statusBadge = c.status === '通过'
    ? '<span class="badge pass">通过</span>'
    : (c.status === '跳过' ? '<span class="badge">跳过</span>' : '<span class="badge fail">失败</span>');
  const gold = (c.gold && c.gold.length) ? esc(c.gold.join(', ')) : '<span class="catname">—（无答案类）</span>';
  const hit = (c.gold && c.gold.length)
    ? (c.gold_hit ? '<span class="badge pass">命中</span>' : '<span class="badge fail">未命中</span>')
    : '<span class="catname">—</span>';
  // 「判定方式」= 怎么判（技术手段）：只列判定器名
  const judges = (c.metrics || []).map(m =>
    `<span class="badge" title="${esc(m.detail || '')}">${esc(m.name)}</span>`).join(' ') || '<span class="catname">—</span>';
  // 「结果」= 状态 + 各判定器得分（度量），与「怎么判」分开
  const result = (c.metrics || []).map(m => {
    const cls = m.passed === true ? 'pass' : (m.passed === false ? 'fail' : '');
    return `<span class="badge ${cls}" title="${esc(m.detail || '')}">${esc(m.name)} ${esc(m.score)}${m.passed === true ? ' ✓' : (m.passed === false ? ' ✗' : ' —')}</span>`;
  }).join(' ') || '<span class="catname">—</span>';
  return `
    <tr>
      <td class="mono">${esc(c.id)}</td>
      <td class="wrapcell">${esc((c.prompt || '').slice(0, 80))}</td>
      <td class="catname">${esc((c.retrieved || []).join(', ') || '（空）')}</td>
      <td>${gold}</td>
      <td>${hit}</td>
      <td>${focusTag(c.focus)}</td>
      <td class="wrapcell">${judges}</td>
      <td class="wrapcell">${statusBadge}<div style="margin-top:4px">${result}</div></td>
      ${ragAnswerCell(c)}
    </tr>`;
}

async function renderRag() {
  const a = await fetch('api/rag.json').then(r => r && r.ok ? r.json() : null).catch(() => null);
  const runs = (a && a.runs) || [];
  // latest 是「最近一次真实模型的跑」（后端已跳过 mock 自检）；latest_any 是最近一次跑（可能是自检）
  const latest = (a && a.latest) || null;
  const latestAny = (a && a.latest_any) || null;
  const onlySelfcheck = !latest && !!latestAny;
  const selfcheckIsNewer = !!(latest && latestAny && latestAny.is_mock && latestAny.file !== latest.file);
  const statHtml = (k, v, color) => `<div class="stat"><div class="k">${esc(k)}</div>
      <div class="v"${color ? ` style="color:${color}"` : ''}>${v}</div></div>`;
  const cases = (latest && latest.cases) || [];

  $app.innerHTML = `
    <a class="back" href="#">← 返回实验室首页</a>
    <h1>RAG 评测 <span class="sub" style="font-size:13px;font-weight:400">（检索找得对不对 / 生成忠不忠于资料 / 没有资料时编不编）</span></h1>
    <div class="sub">数据源：<code>rag_eval/results/</code>（含 <code>archive/</code>）。
      被测对象是项目内自建的 mini-RAG <b>固定管道</b>：每问无条件检索 top-k 再生成，
      不做「要不要检索」的决策、不产生多步轨迹——所以它是独立的一层，不是 Agent 层的子集。</div>
    ${latest ? `
    <div class="card">
      <div class="hint">最新一次（真实模型）：${esc(latest.time_display)}${latest.archived ? ' <span class="badge">历史归档</span>' : ''}
        · 被测对象 <b>${esc(latest.model || '—')}</b>
        · ${esc(latest.file)}${latest.report_file ? ` · <span class="catname">${esc(latest.report_file)}</span>` : ''}
        ${selfcheckIsNewer ? `<br>另有 ${esc(latestAny.time_display)} 的自检跑（<code>${esc(latestAny.model || 'mock')}</code>，阴性对照，不进成绩），见下方历史结果。` : ''}</div>
      <div class="grid" style="margin-bottom:4px">
        ${statHtml('任务通过率', `${pct(latest.pass_rate || 0)}<div class="k">${latest.passed ?? 0} / ${latest.total ?? 0} 通过${Number(latest.skipped || 0) ? ` · ${esc(latest.skipped)} 跳过` : ''}</div>`, rateColor(latest.pass_rate || 0))}
        ${statHtml(`检索层 recall@${esc(latest.recall_k ?? 2)}`, `${pct(latest.recall_at_k || 0)}<div class="k">可判 ${esc(latest.recall_scored ?? 0)} 条 · 命中 ${esc(latest.recall_hit ?? 0)} 条</div>`, rateColor(latest.recall_at_k || 0))}
        ${statHtml('知识库', `${esc(latest.doc_count ?? '—')} 段<div class="k">top-${esc(latest.top_k ?? 2)} 拼进上下文</div>`)}
        ${ragRetrieverStat(latest)}
      </div>
      ${ragProvenanceLine(latest.provenance)}
    </div>
    ${ragMetricsSummaryCard(latest.metrics_summary)}
    <div class="card">
      <div style="font-weight:600;margin-bottom:4px">逐题明细（${cases.length} 条）</div>
      <div class="hint">「检索到」是这次真正拼进上下文的资料编号，「正确出处」是题目标注的答案所在文档——
        两者对不上就是检索层的锅；对得上却答错，那是生成层没用好资料。<br>
        <b>考点</b>（想测什么能力）与<b>判定方式</b>（怎么判：contains / not_contains / judge）、
        <b>结果</b>（状态与得分）是三个层次，分列显示——判定器名不再占「指标」的位置。</div>
      <table>
        <tr><th>用例</th><th>问题</th><th>检索到</th><th>正确出处</th><th>命中</th><th>考点</th><th>判定方式</th><th>结果</th><th>回答（截断）</th></tr>
        ${cases.map(ragCaseRow).join('') || `<tr><td colspan="9" class="empty">暂无明细</td></tr>`}
      </table>
    </div>
    </div>` : `
    <div class="card"><div class="sub" style="margin:0">${onlySelfcheck
      ? `目前只有自检跑（<code>${esc(latestAny.model || 'mock')}</code>，${esc(latestAny.time_display)}）——它是管道阴性对照，不算成绩。跑一次真实模型：<code>python rag_eval/run_eval.py --models deepseek-chat</code>。`
      : `暂无结果。先跑 <code>python rag_eval/run_eval.py</code>（零成本冒烟：<code>--models mock-baseline --no-embedding</code>）生成 <code>rag_eval/results/result-*.json</code>。`}</div></div>`}
    ${runs.length ? `
    <div class="card">
      <div style="font-weight:600;margin-bottom:4px">历史结果（${runs.length} 次）</div>
      <div class="hint">标「自检」的是 mock 模型跑的管道阴性对照——它答不对是预期行为，不进成绩，只用来证明管道活着。</div>
      <table>
        <tr><th>时间</th><th>被测对象</th><th>通过率</th><th>recall@k</th><th>状态</th></tr>
        ${runs.map(r => `
          <tr>
            <td class="catname">${esc(r.time_display)}${r.archived ? ' <span class="badge">归档</span>' : ''}</td>
            <td><span class="badge">${esc(r.model || '—')}</span>${r.is_mock ? ' <span class="catname">自检 · 阴性对照</span>' : ''}</td>
            ${rateCell(r.pass_rate || 0)}
            <td class="catname">${r.recall_at_k === undefined || r.recall_at_k === null ? '—' : pct(r.recall_at_k)}</td>
            <td>${Number(r.failed || 0) === 0
                ? '<span class="badge pass">无失败</span>'
                : `<span class="badge fail">${esc(r.failed)} 失败</span>`}
              <span class="catname" style="margin-left:6px">${esc(r.passed ?? 0)}/${esc(r.total ?? 0)}</span></td>
          </tr>`).join('') || `<tr><td colspan="5" class="empty">暂无结果</td></tr>`}
      </table>
    </div>` : ``}`;
}

// Agent 逐题明细：4 个汇总数字回答不了「哪道题挂了、为什么挂」——10/10 全绿和 0/10 全红
// 提供的信息量几乎一样。这里把每题的考点、各层断言、失败原因、判据修订留痕、逐步轨迹
// 摆出来，访问者不看解说也能回答三件事：这题在考什么、哪层没过、为什么挂、后来怎么处理的。
// 层名与结果 JSON 的 layers 字段一致（「有效性」不单列：它是这次跑没跑成，不是断言层）。
const AGENT_LAYERS = [['答案层', '答案'], ['工具层', '工具'], ['步骤效率', '步数'], ['工具健康', '工具健康']];

function agentLayerChips(t) {
  const L = t.layers || {};
  return AGENT_LAYERS.map(([key, short]) => {
    const l = L[key];
    if (!l) return `<span class="badge" title="${esc(key)}：本次未判">${short} —</span>`;
    return `<span class="badge ${l.pass ? 'pass' : 'fail'}" title="${esc(key)}：${esc(l.detail || '')}">${l.pass ? '✓' : '✗'} ${esc(short)}</span>`;
  }).join('');
}

function agentTrajStep(s) {
  const calls = (s.calls || []).map(c => `<div class="agcall">→ ${esc(c.name)}(${esc(JSON.stringify(c.args || {}))})</div>`).join('');
  const obs = String(s.observation || '');
  return `<div class="agstep">
      <div class="agstepn">第 ${esc(s.n)} 步${s.final ? ' · 给出终答' : ''}</div>
      ${s.thought ? `<div class="agthink">思考：${esc(s.thought.slice(0, 300))}</div>` : ''}
      ${calls}
      ${s.error ? `<div class="agfail">工具报错：${esc(s.error)}</div>` : ''}
      ${obs ? `<div class="agobs">${s.final ? '终答' : '观察'}：${esc(obs.slice(0, 300))}</div>` : ''}
    </div>`;
}

function agentTaskDetail(t) {
  const L = t.layers || {};
  const failed = Object.keys(L).filter(k => L[k] && !L[k].pass);
  const sec = (label, body) => `<div class="agsec"><span class="aglab">${esc(label)}</span><span class="agbody">${body}</span></div>`;
  let html = sec('题目', esc(t.task || '—')) + sec('最终答案', esc(t.final_answer || '—'));
  if ((t.failure_reasons || []).length) {
    html += sec('失败原因', t.failure_reasons.map(r => `<div class="agfail">✗ ${esc(r)}</div>`).join(''));
  } else if (failed.length) {
    html += sec('失败层', failed.map(k => `<div class="agfail">✗ ${esc(k)}：${esc((L[k] || {}).detail || '')}</div>`).join(''));
  }
  if (t.v2_note) html += sec('判据修订留痕', `<span class="agv2">${esc(t.v2_note)}</span>`);
  if (t.repeat) {
    html += sec('repeat 稳定性',
      `各遍：${esc((t.repeat.各遍状态 || []).join(' / '))} · ${t.repeat.状态一致
        ? '<span style="color:var(--ok)">一致</span>' : '<span style="color:var(--warn)">不一致</span>'}`);
  }
  const tr = t.trajectory;
  html += (Array.isArray(tr) && tr.length)
    ? sec('轨迹', `<div class="agtraj">${tr.map(agentTrajStep).join('')}</div>`)
    : sec('轨迹', '<span class="catname">无轨迹（本次结果未归档轨迹文件）</span>');
  return html;
}

function agentTaskRow(t) {
  const isTrap = /陷阱/.test(String(t.level || ''));
  const rep = t.repeat;
  return `<details class="agcase">
      <summary class="agsrow">
        <span class="agid">${esc(t.id || '—')}</span>
        <span class="badge ${isTrap ? 'agtrap' : ''}">${esc(t.level || '—')}</span>
        <span class="agpoint" title="考点：tasks.json 的 design_note，原样搬运">${esc(t.design_note || '—')}</span>
        <span class="badge ${t.status === '通过' ? 'pass' : 'fail'}">${esc(t.status || '—')}</span>
        ${agentLayerChips(t)}
        <span class="catname">步数 ${esc(t.steps ?? '—')}/${esc(t.max_steps ?? '—')}</span>
        ${rep
          ? `<span class="badge ${rep.状态一致 ? 'pass' : 'skip'}" title="各遍：${esc((rep.各遍状态 || []).join(' / '))}">稳定性 ${rep.状态一致 ? '✓' : '✗'}</span>`
          : '<span class="badge" title="本次只跑 1 遍，不判稳定性">稳定性 —</span>'}
      </summary>
      <div class="agdetail">${agentTaskDetail(t)}</div>
    </details>`;
}

function agentTaskList(latest, ds) {
  const tasks = (latest && latest.tasks) || [];
  if (!tasks.length) return '';
  // 数据源必须写在页面上：它是判分的前提，藏起来访问者就发现不了
  // 「金标还停在旧报告上」这种漂移（2026-10-06 那次正是这么藏了很久）
  const srcFile = (ds && ds.source_report) ? String(ds.source_report).split('/').pop() : '';
  const dsLine = srcFile
    ? `<div class="hint" style="padding:0 12px 6px">数据源：<b>agent 评测专用快照</b>（源自 ${esc(ds.data_date || '—')} 全量回归 · <code>${esc(ds.fixture || '')}</code>）· <a href="#${encodeURIComponent(srcFile)}">看该报告 →</a></div>`
    : '';
  return `<div class="card">
      <div style="font-weight:600;margin-bottom:4px">逐题明细 · 最新一次（${esc(latest.time_display || '—')} · 大脑 ${esc(latest.brain || 'deepseek-chat')}）</div>
      ${dsLine}
      <div class="hint" style="padding:0 12px 6px">被测对象是 <b>agent 系统本身</b>，大脑只是可替换组件（换脑跑的成绩在下方历史表里分行看）。
        考点取自 <code>agent_eval/tasks.json</code> 的 design_note，原样搬运不另编；四层断言逐项亮灯（悬停看判定依据）；点开一行看失败原因、判据修订留痕与逐步轨迹。</div>
      ${tasks.map(agentTaskRow).join('')}
    </div>`;
}

// 一份报告里参与「成绩」的模型行：横向报告取 model_rows，单模型报告取 model 一个。
const modelsOf = r => (isMultiModel(r) ? r.model_rows : [{ model: r.model }]).filter(x => x && x.model);
// Agent 结果的历史里有一段没有 brain 字段（那时只有一个大脑），按这个缺省补齐，
// 与后端 _agent_entry 的口径一致。
const DEFAULT_BRAIN = 'deepseek-chat';
// 时间戳**只由后端解析一次**（normalize_display_time → 'YYYY-MM-DD HH:MM'），
// 前端原样显示、**不做二次切割**。之前这里靠硬编码下标切字符串（假设 15 位
// 紧凑串），遇到带 tag 的文件名（report-clean-20261027-20261027-214243）就切出
// `20261027-202610`，对 RAG 那种已经格式化好的串又切成乱码——晓旭 2026-10-28 发现。
const fmtTime = t => (t && String(t) !== '?') ? String(t) : '—';

// 门户成绩取哪份报告：**从最新往回找第一份含真实模型（非 mock-*）的报告**。
// CD 冒烟报告只有 mock-baseline（管道自检的假模型），拿它当成绩会把「mock 基线」
// 显示成实验室的结论——那是给访问者看的第一屏，误导最严重。mock-only 的报告
// 只计入「共 N 份报告」，不进成绩。
// 成绩源报告序列（按时间倒序）：只保留含真实模型（非 mock-*）的报告。
// 门户卡片、#llm 页头（banner / 统计行 / 统计卡）**共用这一个选源**——
// 两处各算一套就会出现「门户看真实模型、#llm 页头看 mock」的撕裂（dimensionSummary 的教训）。
function scoringReports(reports) {
  return (reports || []).filter(r => modelsOf(r).some(x => !isMockModel(x.model)));
}

function pickScoringReport(reports) {
  return scoringReports(reports)[0] || null;
}

// 「框架自检报告」：报告里**所有**模型都是 mock-*（cd.yml 每次合并后自动跑的管道心跳）。
// 它是「评测框架自己还活着」的证据，不是模型成绩，逐条占行会把真报告挤下去。
// 含真实模型的报告哪怕也带 mock-baseline 对照行，都不算自检。
function isSelfCheckReport(r) {
  const ms = modelsOf(r);
  return ms.length > 0 && ms.every(x => isMockModel(x.model));
}

// 门户页（空 hash）：两层评测各一张入口卡，等大并排。
// 每张卡三段式：被测对象 / 怎么测 / 最新成绩——访问者一眼看明白测的是谁、怎么测、结果如何。
async function renderPortal(reports) {
  const agent = await fetch('api/agent.json').then(r => r && r.ok ? r.json() : null).catch(() => null);
  const rag = await fetch('api/rag.json').then(r => r && r.ok ? r.json() : null).catch(() => null);
  const sec = (label, body) => `<div class="psec"><span class="plabel">${esc(label)}</span>${body}</div>`;

  // 模型层：成绩来自「第一份含真实模型的报告」，逐真实模型给通过率 + 报告日期
  const scoring = pickScoringReport(reports);
  const realRows = scoring ? modelsOf(scoring).filter(x => !isMockModel(x.model)) : [];
  const realNames = realRows.map(x => x.model);
  // 模型名单的**唯一数据源**：卡片「被测对象」、简介段、局限块「规模」三处都从这里取。
  // 别人 clone 后配自己的模型跑评测，这三处跟着数据走；写死我们的模型名就成了假话。
  const noRealModels = '<span class="pwhen">（暂无真实模型数据）</span>';
  const modelsHtml = realNames.length
    ? realNames.map(n => `<b>${esc(n)}</b>`).join(' / ')
    : '<b>多个大模型</b>';
  // 局限块那行的口径是「N 个模型（a / b）」，要带数量，所以另拼一份（仍是同一 realNames）
  const scaleModels = realNames.length
    ? `${realNames.length} 个模型（${realNames.map(esc).join(' / ')}）`
    : '模型（暂无真实模型数据）';
  const llmScore = scoring ? `
      ${sec('最新成绩', realRows.map(x =>
        `<span class="pchip">${esc(x.model)} <b class="rate" style="color:${rateColor(listRate(scoring, x))}">${pct(listRate(scoring, x))}</b></span>`
      ).join(' ') + `<span class="pwhen">${esc(fmtTime(scoring.time_display))} · 共 ${reports.length} 份报告</span>`)}`
    : sec('最新成绩', `<span class="pwhen">暂无含真实模型的报告</span>`);
  const llmCard = `
      <div class="ptitle">模型层评测</div>
      ${sec('被测对象', realNames.length
        ? realNames.map(n => `<b>${esc(n)}</b>`).join(' vs ')
        : noRealModels)}
      ${sec('怎么测', '同一用例集横向对比 · 接入 CI 回归门禁')}
      ${llmScore}`;

  // Agent 层：被测对象/怎么测/最新成绩，口径同 #agent 页
  const aLatest = (agent && agent.latest) || null;
  const aRs = aLatest && aLatest.repeat_summary;
  const trapCount = (aLatest && Array.isArray(aLatest.traps)) ? aLatest.traps.length : 4;
  // 被测大脑名单：从 agent.json 的 runs 里去重取全部大脑（后端给每条补了 brain，
  // 老结果缺省 deepseek-chat）。runs 按时间倒序，翻转一遍就是「首次出场」顺序，
  // 后面再跑旧大脑也不会把名单顺序打乱。写死某个大脑名，换脑跑完就成了假话。
  const aRuns = (agent && agent.runs) || [];
  const brains = [];
  for (const r of [...aRuns].reverse()) {
    const name = (r && r.brain) || DEFAULT_BRAIN;
    if (!brains.includes(name)) brains.push(name);
  }
  const brainsHtml = (brains.length ? brains : [DEFAULT_BRAIN]).map(b => `<b>${esc(b)}</b>`).join(' / ');
  const latestBrain = (aLatest && aLatest.brain) || DEFAULT_BRAIN;
  const agentCard = `
      <div class="ptitle">Agent 层评测</div>
      ${sec('被测对象', `smolagents <b>ToolCallingAgent</b> · 大脑：${brainsHtml} · 2 个真实回归工具`)}
      ${sec('怎么测', `${esc(aLatest ? aLatest.total : 10)} 道任务含 ${esc(trapCount)} 道陷阱 · 四层断言 · repeat=${esc(aLatest ? aLatest.repeat : 3)}`)}
      ${aLatest ? sec('最新成绩', `
        <span class="pchip">大脑 <b>${esc(latestBrain)}</b></span>
        <span class="pchip">任务 <b class="rate" style="color:${rateColor(aLatest.pass_rate || 0)}">${esc(aLatest.passed ?? 0)}/${esc(aLatest.total ?? 0)}</b></span>
        <span class="pchip">工具选择 <b class="rate" style="color:${rateColor(aLatest.tool_rate || 0)}">${pct(aLatest.tool_rate || 0)}</b></span>
        ${aRs ? `<span class="badge ${aRs.全部一致 ? 'pass' : 'warn'}">repeat ${esc(aRs.遍数)} 遍${aRs.全部一致 ? '全部一致' : '存在抖动'}</span>` : ''}
        <span class="pwhen">${esc(fmtTime(aLatest.time_display))}</span>`)
      : sec('最新成绩', `<span class="pwhen">暂无结果</span>`)}`;

  // RAG 层：第三张卡。口径同 #rag 页——检索层与生成层分开看，才能归因。
  // RAG「最新成绩」取最近一次真实模型的跑（后端 pick_latest_real_run 已跳过 mock 自检）
  const rLatest = (rag && rag.latest) || null;
  const rAny = (rag && rag.latest_any) || null;
  const rOnlySelfcheck = !rLatest && !!rAny;   // 只有自检跑，还没有真实模型的成绩
  const rTotal = rLatest ? rLatest.total : 13;
  const ragCard = `
      <div class="ptitle">RAG 层评测</div>
      ${sec('被测对象', `项目内自建 mini-RAG <b>固定管道</b>${rLatest && rLatest.model ? ` · 大脑 ${esc(String(rLatest.model).replace(/^rag-/, ''))}` : ''}`)}
      ${sec('怎么测', `冻结知识库 ${esc(rLatest ? rLatest.doc_count : 14)} 段 · top-${esc(rLatest ? rLatest.top_k : 2)} 检索 · 三类考点（命中 / 忠实度 / 无答案诚实）· 检索层 recall@k`)}
      ${rLatest ? sec('最新成绩', `
        <span class="pchip">任务 <b class="rate" style="color:${rateColor(rLatest.pass_rate || 0)}">${esc(rLatest.passed ?? 0)}/${esc(rLatest.total ?? 0)}</b></span>
        <span class="pchip">recall@${esc(rLatest.recall_k ?? 2)} <b class="rate" style="color:${rateColor(rLatest.recall_at_k || 0)}">${pct(rLatest.recall_at_k || 0)}</b></span>
        <span class="pwhen">${esc(fmtTime(rLatest.time_display))}</span>`)
      : sec('最新成绩', rOnlySelfcheck
          ? `<span class="pwhen">暂无真实模型成绩（最近一次 ${esc(fmtTime(rAny.time_display))} 是自检跑，不进成绩）</span>`
          : `<span class="pwhen">暂无结果</span>`)}`;

  $app.innerHTML = `
    <h1>LLM &amp; Agent &amp; RAG 评测实验室</h1>
    <div class="plead">一个 AI 测试方向的自学实战项目 —— 不是生产级框架，但评测闭环的每个环节都真实跑通。</div>
    <div class="sub">模型层：用同一套用例集横向对比 ${modelsHtml} 的能力、稳定性与工程成本，并接入 CI 回归门禁；
      Agent 层：一个 <b>smolagents ToolCallingAgent</b>，大脑 ${brainsHtml} · 2 个真实回归工具，
      10 道任务含 4 道陷阱，看工具调用、答题质量与多遍稳定性；
      RAG 层：项目内自建的 mini-RAG 固定管道，看检索找得对不对、生成忠不忠于资料、没有资料时会不会编。</div>
    <div class="portal">
      <a class="pcard" href="#llm">
        ${llmCard}
        <div class="pgo">查看模型层详情 →</div>
      </a>
      <a class="pcard" href="#agent">
        ${agentCard}
        <div class="pgo">查看 Agent 层详情 →</div>
      </a>
      <a class="pcard" href="#rag">
        ${ragCard}
        <div class="pgo">查看 RAG 层详情 →</div>
      </a>
    </div>
    <div class="pscope">
      <div class="pscopetitle">项目定位与局限</div>
      <div class="pscopegrid">
        <div class="pscopeitem">
          <div class="plabel">规模</div>
          <div class="pvalue">${scaleModels}、98 条模型层用例、
            10 道 Agent 任务、2 个工具、${esc(rTotal)} 条 RAG 用例与 ${esc(rLatest ? rLatest.doc_count : 14)} 段冻结知识库、
            单轮任务、单一模型供应商。</div>
        </div>
        <div class="pscopeitem">
          <div class="plabel">未覆盖</div>
          <div class="pvalue">多轮对话状态、多步组合绕开；RAG 层只覆盖固定管道（无条件检索再生成），
            未覆盖 agentic RAG（模型自主决定是否检索 / 改写 query / 多跳）；安全红队 30 条已按考点分两类判定（15 条排除检查 + 15 条安全裁判，裁判与被测同供应商）；
            repeat=3 的一致性验证样本量有限，只能看抖动、不足以做统计结论。</div>
        </div>
        <div class="pscopeitem">
          <div class="plabel">已跑通的闭环</div>
          <div class="pvalue">出题 → 判分 → 报告 → CI 门禁 → 误报归因 → 判据修订留痕（meta.v2_changes）
            → repeat=N 一致性验证。</div>
        </div>
      </div>
    </div>
    <div class="pfoot">数据由 CI 回归与本地评测自动更新。</div>`;
}

// 渲染报告列表（#llm，原首页）
async function renderList(reports) {
  const tests = await (await fetch('api/tests.json')).json();
  const hasTests = tests && tests.total;
  const ciAllPass = hasTests && tests.failed === 0 && tests.errors === 0;
  // 列表里只要有一份稳定性报告，通过率列就要改成「稳定率 / 通过率」双含义表头
  const hasStability = (reports || []).some(isStability);
  // 顶部 banner：复用现有 api 接口；数据计算是异步的但已与 tests.json 并行 fetch
  const bannerData = await computeBannerData(reports);
  // 列表只展示含真实模型的报告；纯 mock 的自检报告折叠到底部一行统计。
  // 报告编号仍按全量 reports 的序号走（#63 永远是 #63），折叠不会让编号漂移。
  const selfChecks = reports.filter(isSelfCheckReport);
  const main = reports.filter(r => !isSelfCheckReport(r));
  // 页头所有数字都只看成绩源报告（含真实模型的最新一份），与门户卡片同源。
  // 取 reports[0] 会取到 cd.yml 的自检报告——那是最新的，但里面只有 mock。
  const scoring = pickScoringReport(reports);
  const scoringRows = scoring ? modelsOf(scoring).filter(x => !isMockModel(x.model)) : [];
  // 用例集口径取成绩源报告覆盖的去重用例数与数据集数：跨报告累计 case_count
  // （同一批用例跑几十遍的累加）对外行是噪音，也不能反映「这套题有多大」。
  // case_count 是「用例 × 模型」的行数，除以模型数才是去重用例数（294 / 3 = 98）。
  const caseCount = scoring
    ? Math.round((scoring.case_count || 0) / Math.max(modelsOf(scoring).length, 1))
    : 0;
  const datasetCount = (scoring && scoring.datasets)
    ? (Array.isArray(scoring.datasets) ? scoring.datasets.length : Object.keys(scoring.datasets).length)
    : 0;
  $app.innerHTML = `
    <a class="back" href="#">← 返回实验室首页</a>
    ${bannerData ? renderBanner(bannerData) : ''}
    <h1>LLM 评测</h1>
    <div class="sub">真实评测报告 ${main.length} 份 · 框架自检 ${selfChecks.length} 次${scoring ? ` · 最新真实评测 ${esc(scoring.time_display)}` : ' · 暂无真实评测'}</div>
    <div class="grid" style="margin-bottom:16px">
      <div class="stat"><div class="k">真实评测报告</div><div class="v">${main.length}</div>
        <div class="catname">自检 ${selfChecks.length} 次见列表底部</div></div>
      <div class="stat"><div class="k">用例集</div><div class="v">${caseCount}</div>
        <div class="catname">${datasetCount ? `${datasetCount} 个数据集` : '数据集信息缺失'}</div></div>
      <div class="stat"><div class="k">最新成绩</div>
        <div class="v" style="font-size:15px">${scoringRows.length
          ? scoringRows.map(x => `<div>${esc(x.model)} <b style="color:${rateColor(listRate(scoring, x))}">${pct(listRate(scoring, x))}</b></div>`).join('')
          : '<span class="catname">—</span>'}</div></div>
      <div class="stat"><div class="k">成绩日期</div><div class="v" style="font-size:14px">${esc(scoring ? scoring.time_display : '—')}</div>
        <div class="catname">${scoring ? '此后的更新为管道自检' : '尚无真实评测'}</div></div>
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
      ${main.map((r) => {
        const i = reports.indexOf(r);
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
          <td class="catname">${esc(r.time_display)}</td>
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
    ${selfChecks.length ? `
    <details style="margin-top:12px; border-top:1px solid rgba(42,54,80,.4); padding-top:10px">
      <summary style="cursor:pointer;color:var(--muted);font-size:13px">另有 ${selfChecks.length} 次框架自检 ✓（管道心跳，非模型成绩）· 展开</summary>
      <table style="margin-top:10px">
        <tr><th>报告</th><th>时间</th><th>模型</th><th>用例</th><th>通过率</th></tr>
        ${selfChecks.map((r) => {
          const multi = isMultiModel(r);
          return `
        <tr class="rowlink" data-go="${encodeURIComponent(r.file)}/summary">
          <td>#${reports.length - reports.indexOf(r)} ${typeBadge(r)} <span class="badge">${r.tag || 'run'}</span></td>
          <td class="catname">${esc(r.time_display)}</td>
          <td>${multi ? multiCell(r, x => `<div>${esc(x.model)}</div>`) : esc(r.model)}</td>
          <td>${caseCountText(r)}</td>
          ${multi ? `<td>${multiCell(r, x => rateInner(listRate(r, x)))}</td>` : rateCell(listRate(r, r))}
        </tr>`;
        }).join('')}
      </table>
    </details>` : ''}
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
  // 污染报告：结论区整体换成警示（不出现排名与「某分类明显偏低」这类误导句）
  const warn = !!c.contaminated;
  return `
    <div class="card conclusion${warn ? ' concl-warn' : ''}">
      <div style="font-weight:600;margin-bottom:6px">${warn ? '⚠️ 本报告为污染数据' : '本报告结论'}</div>
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
      <div style="font-weight:600;margin-bottom:4px">维度 × 模型 对比矩阵${legacyDimNote(r)}</div>
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
        期望输出、评分明细（抽屉里显示的就是这一行所属模型的那条结果）。<br>
        <b>能力指标</b>（测什么能力：准确性 / 指令遵循 / 安全 / 相关性）与<b>判定方式</b>
        （怎么判：contains / judge …）是两回事，分两列显示——把判定器名摆在能力指标的位置上，
        等于告诉读者「这道考 contains」，那是把实现细节当成了测试意图。
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
          <tr><th style="width:56px">状态</th><th style="width:110px">case_id</th><th style="width:110px">模型</th><th>输入（prompt）</th><th>模型输出（response）</th><th style="width:90px">分类</th><th style="width:104px">能力指标</th><th style="width:120px">判定方式</th><th style="width:96px">通过次数</th><th style="width:80px">延迟</th></tr>
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
              ${caseDimCell(c)}
              ${caseMethodCell(c)}
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
      `<option value="${esc(x.file)}" ${x.file===defaultBase?'selected':''}>${esc(x.time_display)} · ${esc(x.model)} · ${esc(x.file)}</option>`
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
      <span>能力指标：<b>${esc(caseDimLabel(c))}</b></span>
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
      <h3>判定方式 (${(c.metrics||[]).length})</h3>
      <div class="hint" style="margin:-2px 0 6px">能力指标 = 这条题测什么能力（见上方「能力指标」）；
        这张表是<b>怎么判</b>——每个判定器的得分与判词，「为什么会判失败」在这里查。</div>
      ${(c.metrics||[]).length ? `
        <table>
          <tr><th>判定方式</th><th>score</th><th>通过</th><th>详情</th></tr>
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
// 空 hash = 门户页（三层入口）；#llm 模型层原报告列表；#tests 用例详情；
// #agent Agent 评测；#rag RAG 评测
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
    if (file === 'rag') return renderRag();
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
    .replace("__KNOWN_DIMENSIONS__", json.dumps(sorted(KNOWN_DIMENSIONS), ensure_ascii=False))
    .replace("__DIMENSION_SCHEMA_VERSION__", json.dumps(DIMENSION_SCHEMA_VERSION, ensure_ascii=False))
)


# 文件名里带这些标记的报告**不是模型层报告**，不进模型层的列表/卡片。
# RAG 层的报告也写在 reports/ 下（`report-rag-v1-*.json`，模型名 `rag-deepseek-chat`），
# 混进模型层列表会让它顶掉真正的模型层报告——卡片于是显示 rag-mock-baseline / NaN%。
# 与 Q6「模型层自检数据集不含 rag_cases」是同一条口径：模型层只吃模型层的东西。
NON_MODEL_REPORT_MARKS = ("rag",)
# CI 测试摘要/明细不是评测报告
NON_REPORT_FILES = ("tests-summary.json", "tests-detail.json")


def _is_model_layer_report(name: str) -> bool:
    """这份报告是不是模型层的？（排除规则只此一处，别在调用点散写 if。）"""
    lowered = name.lower()
    if lowered in NON_REPORT_FILES:
        return False
    return not any(mark in lowered for mark in NON_MODEL_REPORT_MARKS)


def _load_reports():
    """扫描 reports/*.json，返回按时间倒序的元数据列表。"""
    items = []
    for p in sorted(REPORTS_DIR.glob("*.json")):
        # CI 测试摘要/明细不是评测报告，不进列表（否则会出现 model=?/用例=0 的幽灵条目）；
        # 同理排除 RAG 层报告——它是另一层的成绩，不是模型层的。
        if not _is_model_layer_report(p.name):
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
            # 给前端显示用的规范化时间（文件名时间戳同样会被 tag 里的日期带偏，交给 normalize_display_time）
            "time_display": normalize_display_time(data.get("started_at"), p.name),
            "tag": p.stem.split("-")[0] if "-" in p.stem else "run",
            "model": summary.get("model", "?"),
            "models": [row["model"] for row in model_rows],
            "model_rows": model_rows,
            "case_count": data.get("case_count", 0),
            "repeat": repeat,
            # 数据集名单：#llm 页头「用例集 · N 个数据集」卡片要用，
            # 让前端不必为了数数据集再回源拉一次报告文档。
            "datasets": data.get("datasets", []),
            # 污染留证：被标过污染的报告照样进列表（历史不该抹掉），
            # 但详情页的结论区会换成警示，不产出任何能力结论。
            "contaminated": bool(data.get("contaminated")),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "pass_rate": summary.get("pass_rate", 0.0),
            "stability": data.get("stability"),
            "p95": summary.get("p95_latency_ms"),
        }
        # 维度速览：取代表模型（跳过 mock 对照组）的各维度通过率。
        # 「照抄报告里存的那份」，不再按 categories 重算（Q8 裁决：一个真相源）。
        entry["dim_model"] = _pick_representative_model(entry)
        entry["dimensions"] = _dimension_summary(data, entry["dim_model"])
        entry["dimension_coverage"] = [row["label"] for row in entry["dimensions"]]
        # 口径版本对不上（或缺字段）的老报告：页面标注「口径为旧版」，不拿它跟新报告横比
        entry["dimension_schema_legacy"] = _dimension_schema_legacy(data)
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


_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})")
_DATE_ONLY_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _stamp_to_display(s):
    """从文件名/紧凑串里认出「8 位日期 + 6 位时间」，**从后往前**取第一对。

    文件名形态不统一：``report-clean-20261027-20261027-214243`` 里 tag 后面紧跟的
    那段也是 8 位日期，按位置切必然错位。**只看数字段、且从末尾配对**，才能稳定
    拿到真正的时间戳（末尾那对），顺带跳过 ``repeat3`` 这类夹在中间的数字。
    认不出返回 None，交给上层决定兜底。
    """
    nums = re.findall(r"\d+", s)
    for i in range(len(nums) - 1, 0, -1):
        if len(nums[i - 1]) == 8 and len(nums[i]) == 6:
            day, clock = nums[i - 1], nums[i]
            return f"{day[:4]}-{day[4:6]}-{day[6:8]} {clock[:2]}:{clock[2:4]}"
    # 14 位连写（20261027214243）：前 8 位日期 + 后 6 位时间
    for n in nums:
        if len(n) == 14:
            return f"{n[:4]}-{n[4:6]}-{n[6:8]} {n[8:10]}:{n[10:12]}"
    return None


def normalize_display_time(*sources) -> str:
    """把各种来源的时间戳统一成 ``YYYY-MM-DD HH:MM``；认不出来就给 ``—``。

    由来（晓旭 2026-10-28 在页面发现）：报告文件名形如
    ``report-clean-20261027-20261027-214243``——**tag 里也带日期**，用
    ``\\d{8}-\\d{6}`` 取**第一组**会切出 ``20261027-202610`` 这种半截串，#llm 页
    「成绩日期」就显示成那样；RAG 侧的 ``time`` 本就已是 ``2026-10-28 04:53:19``，
    前端再按 15 位下标二次切割又切成乱码。

    所以这里定一条规矩：**时间戳只由后端解析一次**，取文件名里**末尾**那组
    （真正的时间戳在末尾，tag 里的日期在前面），并优先信任报告内的 ISO 时间；
    解析不出就给 ``—``，**不许显示半截数字**。前端拿到什么显示什么，不再切。
    """
    for src in sources:
        if not src:
            continue
        s = str(src)
        m = _ISO_RE.search(s)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}"
        out = _stamp_to_display(s)
        if out:
            return out
        m = _DATE_ONLY_RE.search(s)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return "—"


def _agent_stamp(name: str):
    """从 result-20261001-103426.json / agent-eval-20261001-103426.html 里取时间戳。"""
    m = re.search(r"\d{8}-\d{6}", str(name))
    return m.group(0) if m else None


def _split_task_notes(meta: dict) -> dict:
    """把 tasks.json 里整段的 ``meta.v2_changes`` 按题号拆开：一句提到哪题，就算哪题的留痕。

    留痕的价值在「这道题当初为什么改判据」——误报归因 → 判据修订 → 留痕，挂在题上才
    读得出因果；整段贴在页面顶部等于没写。tasks.json 只读，判据本体不动。
    """
    raw = str((meta or {}).get("v2_changes") or "").strip()
    if not raw:
        return {}
    out: dict = {}
    for seg in re.split(r"[；;]", raw):
        seg = seg.strip()
        if not seg:
            continue
        # 「依据：……」是对整批改动的说明，常跟在某题那句后面（用句号连接），
        # 挂到某一题上就答非所问，按子句剔掉
        clauses = [c.strip() for c in re.split(r"[。]", seg) if c.strip()]
        clauses = [c for c in clauses if not c.startswith("依据")]
        if not clauses:
            continue
        seg = "。".join(clauses)
        for tid in re.findall(r"t\d{2}", seg):
            bucket = out.setdefault(tid, [])
            if seg not in bucket:
                bucket.append(seg)
    return {k: "；".join(v) for k, v in out.items()}


def _load_task_notes() -> dict:
    """读 ``agent_eval/tasks.json``，取每道题的考点文案与判据修订留痕（**只读**）。

    考点文案是出题人写的 ``design_note``，看板原样搬运、不另编：自己概括的考点会和判据
    对不上，等于给访问者讲了一个没有出处故事。
    """
    try:
        data = json.loads(AGENT_TASKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return {}
    v2 = _split_task_notes(data.get("meta") or {})
    notes = {}
    for t in tasks:
        if not isinstance(t, dict) or not t.get("id"):
            continue
        tid = str(t["id"])
        notes[tid] = {
            "design_note": str(t.get("design_note") or ""),
            "level": str(t.get("level") or ""),
            "v2_note": v2.get(tid.split("-")[0], ""),
        }
    return notes


def _agent_data_source() -> dict:
    """#agent 页要写明的「数据从哪来」（**只读** tasks.json 的 meta）。

    数据源是判分的前提——不写出来，访问者根本发现不了「金标还停在旧报告上」这种
    漂移（2026-10-06 那次就是这么藏了很久）。给前端三样东西：fixture 文件名、
    源报告文件名、数据日期（源报告在 #llm 详情页可点开）。
    """
    try:
        meta = json.loads(AGENT_TASKS_FILE.read_text(encoding="utf-8")).get("meta") or {}
    except Exception:
        return {}
    note = str(meta.get("golden_source_note") or "")
    match = re.search(r"\d{4}-\d{2}-\d{2}", note)
    return {
        "fixture": str(meta.get("data_source") or ""),
        "source_report": str(meta.get("source_report") or ""),
        "data_date": match.group(0) if match else "",
    }


def _trajectory_steps(rel_path) -> list:
    """读一份落盘轨迹，抽出「思考 → 调用工具 → 观察」的逐步明细；读不动返回 None。

    老结果没归档轨迹是常态（轨迹是后加的能力），缺就缺，页面上降级显示「无轨迹」，
    不拿别的题的轨迹凑数。
    """
    if not rel_path:
        return None
    fp = ROOT / str(rel_path)
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        logs = (data or {}).get("logs") or []
    except Exception:
        return None
    steps = []
    for log in logs:
        # logs[0] 是任务信息（只有 task 字段），不是步骤
        if not isinstance(log, dict) or "step_number" not in log:
            continue
        msg = log.get("model_output_message")
        msg = msg if isinstance(msg, dict) else {}
        calls = []
        for c in log.get("tool_calls") or []:
            fn = (c or {}).get("function") if isinstance(c, dict) else None
            fn = fn if isinstance(fn, dict) else {}
            args = fn.get("arguments")
            calls.append({
                "name": str(fn.get("name") or ""),
                "args": args if isinstance(args, dict) else {},
            })
        obs = log.get("observations")
        err = log.get("error")
        steps.append({
            "n": log.get("step_number"),
            "thought": str(msg.get("content") or "").strip(),
            "calls": calls,
            "observation": "" if obs is None else str(obs)[:600],
            "error": "" if err is None else str(err)[:300],
            "final": str(log.get("is_final_answer")).lower() == "true",
        })
    return steps


def _agent_entry(path: Path, archived: bool, with_trajectory: bool = False):
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
        # 这次用的是哪个大脑（models.yaml 的引用名）。老结果没有 model 字段
        # （那时只有一个大脑），按缺省 deepseek-chat 补齐——不能让「换了脑子」
        # 和「没换脑子」的成绩在同一行里混着看。
        "brain": str(data.get("model") or "deepseek-chat"),
        "time": _agent_stamp(path.name) or data.get("finished_at") or data.get("started_at") or "?",
        "time_display": normalize_display_time(data.get("finished_at"), data.get("started_at"), path.name),
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

    # 逐题明细：考点文案 / 各层断言 / 失败原因 / 判据修订留痕，页面靠它自解释。
    # 4 个汇总数字（成功率 / 工具正确率 / 平均步数 / 陷阱防线）回答不了「哪道题挂了、
    # 为什么挂」，10/10 全绿和 0/10 全红提供的信息量几乎一样。
    notes = _load_task_notes()
    by_task = {}
    if isinstance(rs, dict):
        for it in rs.get("逐题") or []:
            if isinstance(it, dict) and it.get("id"):
                by_task[str(it["id"])] = it
    tasks = []
    for t in data.get("results") or []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id") or "")
        note = notes.get(tid) or {}
        item = {
            "id": tid,
            "level": str(t.get("level") or note.get("level") or ""),
            "task": str(t.get("task") or ""),
            "status": str(t.get("status") or ""),
            "steps": t.get("steps"),
            "max_steps": t.get("max_steps"),
            "layers": t.get("layers") or {},
            "failure_reasons": t.get("failure_reasons") or [],
            "final_answer": str(t.get("final_answer") or ""),
            # 考点文案取 tasks.json 的 design_note，不另编（出处可查）
            "design_note": note.get("design_note", ""),
            "v2_note": note.get("v2_note", ""),
        }
        if tid in by_task:
            item["repeat"] = by_task[tid]
        # 轨迹只给最新一次：一份 30–60KB，全量注入会把 api/agent.json 撑到 MB 级
        if with_trajectory:
            item["trajectory"] = _trajectory_steps(t.get("trajectory_path"))
        tasks.append(item)
    entry["tasks"] = tasks
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

    pairs = []
    for path, archived in paths:
        if AGENT_INVALID_MARK in path.name:
            continue
        entry = _agent_entry(path, archived)
        if entry:
            pairs.append((entry, path, archived))
    pairs.sort(key=lambda x: _time_key(x[0]["time"]), reverse=True)
    runs = [p[0] for p in pairs]
    # 轨迹只给最新一次重读一遍（体积考虑，见 _agent_entry）；历史行没有轨迹，
    # 页面上显示「无轨迹」而不是把老结果伪装成也留了档。
    if pairs:
        entry, path, archived = pairs[0]
        runs[0] = _agent_entry(path, archived, with_trajectory=True)
    _pair_agent_reports(runs)
    return runs


def _rag_time(stamp: str) -> str:
    """文件名时间戳 `20261026-115958` → `2026-10-26 11:59:58`（与 Agent 层显示一致）。"""
    try:
        day, clock = stamp.split("-", 1)
        return f"{day[:4]}-{day[4:6]}-{day[6:8]} {clock[:2]}:{clock[2:4]}:{clock[4:6]}"
    except Exception:
        return stamp


# 检索后端卡片的展示口径（kb-007 同款：降级必须显式标注，且不能把异常原文糊到页面上）。
# 老结果 JSON 的 retriever 字段里带着 `ModuleNotFoundError: No module named ...` 原文，
# 那是排查信息，**只在报告 JSON 的 notes 里**，卡片上要换成「人话 + 降级标记」。
_RAG_DEGRADED_LABEL = "字符相似度（降级）"
_RAG_DEGRADED_HINT = "未安装向量库 sentence-transformers，本次检索为字面匹配、非语义结论"


def _rag_retriever_view(raw: str) -> dict:
    """把 retriever 原始串翻成卡片能直接显示的三件套：label / degraded / hint。

    判定不看异常文本，只看后端是不是字符后端——**降级这件事本身**才是读者要看见的，
    「缺哪个包」是排查细节。
    """
    text = str(raw or "")
    degraded = ("字符" in text) or ("降级" in text)
    if degraded:
        model = ""
    else:
        # 向量后端：`retriever(向量后端 <model>)` —— 模型名保留，便于知道用的哪个模型
        model = text.split("向量后端", 1)[-1].strip(" )") if "向量后端" in text else ""
    return {
        "retriever_label": _RAG_DEGRADED_LABEL if degraded else (f"向量检索（{model}）" if model else "向量检索"),
        "retriever_degraded": degraded,
        "retriever_hint": _RAG_DEGRADED_HINT if degraded else "",
    }


def _is_rag_mock(models) -> bool:
    """这份 RAG 跑是不是自检跑（mock 模型）？

    RAG 层的模型名带 `rag-` 前缀（`rag-deepseek-chat` / `rag-mock-baseline`），
    判断前先剥掉前缀，再按模型层同款规矩认 `mock-`。
    """
    name = str((models or [""])[0])
    return name.removeprefix("rag-").startswith("mock-")


def _rag_entry(path: Path, archived: bool = False, with_cases: bool = False):
    """把一份 RAG 结果 JSON 归一化成看板用的结构。

    结果文件由 `rag_eval/run_eval.py` 写出（同源：既落模型层同款报告，也落这份
    RAG 专属结果）。读失败返回 None——看板显示空状态，不崩页。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    summary = data.get("summary") or []
    first = summary[0] if summary else {}
    kb = data.get("knowledge_base") or {}
    retrieval = data.get("retrieval") or {}
    models = data.get("models") or []
    stamp = path.stem.replace("result-", "", 1)

    entry = {
        "file": path.name,
        "time": _rag_time(stamp),
        "time_display": normalize_display_time(data.get("started_at"), stamp),
        "archived": archived,
        "models": models,
        "model": models[0] if models else "—",
        # 自检跑（mock 模型）是阴性对照，不是被测对象——看板据此决定它能不能进「最新成绩」
        "is_mock": _is_rag_mock(models),
        "top_k": data.get("top_k"),
        # 原始串留给排查（`retriever(字符后端（降级）)` 这类），卡片上**只用清洗后的三个字段**
        "retriever": data.get("retriever") or "—",
        **_rag_retriever_view(data.get("retriever") or ""),
        "total": first.get("total", 0),
        "passed": first.get("passed", 0),
        "failed": first.get("failed", 0),
        "skipped": first.get("skipped", 0),
        "pass_rate": first.get("pass_rate", 0.0),
        # 检索层：答错时先分清是「没检索到」还是「检索到了没用好」
        "recall_at_k": retrieval.get("recall_at_k"),
        "recall_k": retrieval.get("k"),
        "recall_scored": retrieval.get("scored", 0),
        "recall_hit": retrieval.get("hit", 0),
        # 顶部「指标汇总」四项（由 rag_eval 算好写进结果 JSON；老结果没有这个字段时给空 dict，
        # 页面会显示「需重跑」而不是假数字——没有数就是没有数，不猜）
        "metrics_summary": data.get("metrics_summary") or {},
        "doc_count": kb.get("doc_count"),
        "kb_path": kb.get("path") or "",
        "provenance": kb.get("provenance") or {},
        "report_file": data.get("report_file") or "",
        "notes": data.get("notes") or [],
        "cases": data.get("cases") or [] if with_cases else [],
    }
    return entry


def load_rag_runs() -> list:
    """扫描 rag_eval/results/（含 archive/）下的 result-*.json，按时间倒序。

    目录不存在（部署沙盒里没有 rag_eval/）返回空列表，看板显示空状态而不是崩。
    逐题明细只给最新一次（体积考虑），历史行 cases 为空，页面显示摘要即可。
    """
    if not RAG_RESULTS_DIR.is_dir():
        return []
    paths = [(p, False) for p in RAG_RESULTS_DIR.glob("result-*.json")]
    if RAG_ARCHIVE_DIR.is_dir():
        paths += [(p, True) for p in RAG_ARCHIVE_DIR.glob("result-*.json")]

    pairs = []
    for path, archived in paths:
        entry = _rag_entry(path, archived)
        if entry:
            pairs.append((entry, path, archived))
    pairs.sort(key=lambda x: _time_key(x[0]["time"]), reverse=True)
    runs = [p[0] for p in pairs]
    # 逐题明细跟着「最新一次真实模型跑」走，而不是「最新一次跑」——
    # 否则一次 mock 自检就会把逐题明细顶掉（它 0/13，没有可看的东西）。
    idx = next((i for i, e in enumerate(runs) if not e.get("is_mock")), None)
    if idx is not None:
        entry, path, archived = pairs[idx]
        runs[idx] = _rag_entry(path, archived, with_cases=True) or runs[idx]
    return runs


def pick_latest_real_run(runs: list):
    """「被测对象 / 最新成绩」取最近一次**真实模型**的跑，跳过 mock-* 自检。

    与模型层同款规矩（mock-* 是管道基线，不进通过率排名）：自检跑 13 条全挂是
    **预期行为**（假模型本来答不对），把它当成绩展示等于向访问者宣称「RAG 层全挂」。
    """
    for r in runs:
        if not r.get("is_mock"):
            return r
    return None


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
            self._json({"latest": runs[0] if runs else None, "runs": runs,
                        "data_source": _agent_data_source()})
        elif path in ("/api/rag", "/api/rag.json"):
            runs = load_rag_runs()
            self._json({
                # latest = 最近一次真实模型的跑（mock 自检不进成绩，见 pick_latest_real_run）；
                # latest_any 保留「最近一次跑」原样，供页面说明「还有一次更新的自检跑」
                "latest": pick_latest_real_run(runs),
                "runs": runs,
                "latest_any": runs[0] if runs else None,
            })
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
