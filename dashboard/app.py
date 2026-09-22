# -*- coding: utf-8 -*-
"""评测报告看板 —— 零依赖（仅 Python 标准库）。

用法：
    python dashboard/app.py            # 默认 0.0.0.0:8080
    PORT=9000 python dashboard/app.py  # 自定义端口

数据源：../reports/ 目录下 report-*.json（runner 产物）。
"""
import json
import os
import re
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
# 优先使用看板自带数据（部署沙盒用），本地开发则回退到项目 reports/
REPORTS_DIR = Path(__file__).resolve().parent / "data"
if not REPORTS_DIR.is_dir():
    REPORTS_DIR = ROOT / "reports"
PORT = int(os.environ.get("PORT", "8080"))

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


def diff_cases(curr_cases, base_cases):
    """跨次对比：按 case_id 字符串排序后比较 passed 状态变化。

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
    curr_map = {c.get('case_id'): c for c in curr if isinstance(c, dict) and c.get('case_id') is not None}
    base_map = {c.get('case_id'): c for c in base if isinstance(c, dict) and c.get('case_id') is not None}
    all_ids = sorted(set(curr_map) | set(base_map))

    def _find(arr, cid):
        for x in arr:
            if isinstance(x, dict) and x.get('case_id') == cid:
                return x
        return None

    regressed, fixed, still_failing = [], [], []
    only_in_curr, only_in_base = [], []
    for cid in all_ids:
        cur = _find(curr, cid)
        b = _find(base, cid)
        if cur is not None and b is None:
            only_in_curr.append(cur)
            continue
        if b is not None and cur is None:
            only_in_base.append(b)
            continue
        cs, bs = case_status(cur), case_status(b)
        if bs == 'pass' and cs != 'pass':
            regressed.append({'curr': cur, 'base': b})
        elif bs != 'pass' and cs == 'pass':
            fixed.append({'curr': cur, 'base': b})
        elif cs != 'pass':
            still_failing.append({'curr': cur, 'base': b})
    return {
        'regressed': regressed,
        'fixed': fixed,
        'stillFailing': still_failing,
        'onlyInCurr': only_in_curr,
        'onlyInBase': only_in_base,
    }


# ============================== 顶部 banner 字段计算 ============================== #
# 首页最顶部一行大字结论：「最新报告 <model> XX.X%，较上次 +/-X.X pt；
# 回归 n 条 / 修复 m 条；最弱维度：<label> YY.Y%」。
# 数据全部来自 api/reports.json + 已有的 diff_cases 逻辑，不加新接口。

_BANNER_DIM_LABELS = {
    'correctness': '正确性', 'instruction_following': '指令遵循', 'format': '格式合规',
    'safety': '安全', 'robustness': '鲁棒性', 'knowledge': '知识时效', 'untagged': '未标注',
}


def _pick_representative_model(entry):
    """从 model_rows 选 banner 代表模型：优先真实模型（名字不以 ``mock-`` 开头）。

    横向评测的报告里通常有 ``mock-baseline`` 等对照组（可能多个 mock-*），
    把对照组推到 banner 会让人误以为「真实模型怎么才 100%」，所以必须挑
    真实模型。全 mock 时退到 ``entry['model']``；单模型报告 / 残缺数据
    时同样退到 ``entry['model']``。
    """
    rows = entry.get('model_rows') or []
    for row in rows:
        if isinstance(row, dict) and row.get('model') and not row['model'].startswith('mock-'):
            return row['model']
    return entry.get('model', '?')


def _representative_rate(entry, model):
    """取代表模型在 entry 里的 pass_rate；找不到则降级到 entry.pass_rate。"""
    for row in (entry.get('model_rows') or []):
        if isinstance(row, dict) and row.get('model') == model:
            return row.get('pass_rate', 0.0)
    return entry.get('pass_rate', 0.0)


def _weakest_dimension(doc, model):
    """从报告里选「指定模型」的最弱 dimension / category。返回 ``(label, rate)`` 或 ``None``。

    数据结构兼容（按优先级）：
    1. ``dimensions``: ``dict[model] -> list[{dimension, label, ..., pass_rate}]``（runner 新输出）
    2. ``categories``: ``dict[model] -> list[{category, total, passed, failed, pass_rate}]``（老格式 / dashboard/data）

    两层都兼容：找不到时返回 ``None``，前端 banner 显示 "—" 而不崩。
    ``untagged`` 不参与最弱选择（用户看不到意义），但全 ``untagged`` 时仍返回自身。
    """
    if not isinstance(doc, dict):
        return None

    # 1) dimensions 优先（runner 新输出）
    dims = doc.get('dimensions')
    candidates = None
    if isinstance(dims, dict):
        if model in dims and isinstance(dims[model], list):
            candidates = dims[model]
        else:
            for v in dims.values():
                if isinstance(v, list):
                    candidates = v
                    break
    elif isinstance(dims, list):
        candidates = dims

    if candidates:
        valid = [d for d in candidates if isinstance(d, dict) and d.get('dimension')]
        if valid:
            pool = [d for d in valid if d.get('dimension') != 'untagged'] or valid
            weakest = min(pool, key=lambda d: d.get('pass_rate', 1.0))
            label = _BANNER_DIM_LABELS.get(
                weakest.get('dimension'),
                weakest.get('label') or weakest.get('dimension'),
            )
            return (label, weakest.get('pass_rate', 0.0))

    # 2) categories fallback（dashboard/data 老数据，runner 老版本）
    cats = doc.get('categories')
    candidates = None
    if isinstance(cats, dict):
        if model in cats and isinstance(cats[model], list):
            candidates = cats[model]
        else:
            for v in cats.values():
                if isinstance(v, list):
                    candidates = v
                    break
    elif isinstance(cats, list):
        candidates = cats

    if candidates:
        # 过滤 total=0 的空桶，避免被"尚未跑该 category"的零分拉低
        valid = [
            d for d in candidates
            if isinstance(d, dict) and d.get('category') and d.get('total', 0) > 0
        ]
        if valid:
            weakest = min(valid, key=lambda d: d.get('pass_rate', 1.0))
            cat = weakest.get('category')
            label = _BANNER_DIM_LABELS.get(cat, cat)
            return (label, weakest.get('pass_rate', 0.0))

    return None


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
        if isinstance(r, dict) and not str(r.get('model', '')).startswith('mock-')
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
    rank = sorted(
        [{'model': r.get('model', '?'), 'pass_rate': r.get('pass_rate', 0.0)} for r in rows],
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


# 模板里的 __METRIC_LABELS__ 在导入时替换成真实的中文标签映射，
# 保证「Python 单测可见的 METRIC_LABELS」与「前端渲染用的 METRIC_LABELS」是同一份数据。
PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM 评测报告看板</title>
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
// 与后端 src/datasets/schema.py 的 DIMENSION_LABELS 保持一致
const DIM_LABELS = {correctness:'正确性', instruction_following:'指令遵循', format:'格式合规',
                    safety:'安全', robustness:'鲁棒性', knowledge:'知识时效', untagged:'未标注'};
const dimLabel = d => d ? (DIM_LABELS[d] || d) : '未标注';
// 指标中文标签：由后端 METRIC_LABELS 注入（与 Python 侧同一份数据），未收录的原样显示
const METRIC_LABELS = __METRIC_LABELS__;
const metricLabel = m => METRIC_LABELS[m] || m;
const rateColor = x => x >= 0.8 ? 'var(--ok)' : x >= 0.5 ? 'var(--warn)' : 'var(--bad)';
const rateInner = x => `<div style="display:flex;align-items:center;gap:8px"><span class="rate" style="color:${rateColor(x)}">${pct(x)}</span><span class="bar"><i style="width:${(x*100).toFixed(1)}%;background:${rateColor(x)}"></i></span></div>`;
const rateCell = x => `<td>${rateInner(x)}</td>`;
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

// 跨次对比：按 case_id 字符串排序
function diffCases(currCases, baseCases) {
  const byId = (arr, id) => (arr || []).find(x => x.case_id === id);
  const ids = new Set();
  (currCases || []).forEach(c => ids.add(c.case_id));
  (baseCases || []).forEach(c => ids.add(c.case_id));
  const allIds = Array.from(ids).sort();
  const regressed = [], fixed = [], stillFailing = [], onlyInCurr = [], onlyInBase = [];
  for (const id of allIds) {
    const cur = byId(currCases, id);
    const base = byId(baseCases, id);
    if (cur && !base) { onlyInCurr.push(cur); continue; }
    if (base && !cur) { onlyInBase.push(base); continue; }
    const cs = statusOf(cur), bs = statusOf(base);
    if (bs === 'pass' && cs !== 'pass') regressed.push({curr: cur, base});
    else if (bs !== 'pass' && cs === 'pass') fixed.push({curr: cur, base});
    else if (cs !== 'pass') stillFailing.push({curr: cur, base});
  }
  return {regressed, fixed, stillFailing, onlyInCurr, onlyInBase};
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

  // 最弱维度：dimensions 优先；dashboard/data 老报告只有 categories，兼容 fallback
  let weakest = null;
  if (currDoc) {
    const findCandidates = (group) => {
      if (!group) return null;
      if (Array.isArray(group)) return group;
      if (group[representative] && Array.isArray(group[representative])) return group[representative];
      for (const v of Object.values(group)) {
        if (Array.isArray(v)) return v;
      }
      return null;
    };

    // 1) dimensions 优先（runner 新输出）
    const dimCands = findCandidates(currDoc.dimensions);
    if (dimCands && dimCands.length) {
      const valid = dimCands.filter(d => d && d.dimension);
      const noUntagged = valid.filter(d => d.dimension !== 'untagged');
      const pool = noUntagged.length ? noUntagged : valid;
      if (pool.length) {
        const w = pool.reduce((a, b) => (a.pass_rate <= b.pass_rate ? a : b));
        weakest = { label: DIM_LABELS[w.dimension] || w.label || w.dimension, rate: w.pass_rate };
      }
    }

    // 2) categories fallback（dashboard/data 老数据）
    if (!weakest) {
      const catCands = findCandidates(currDoc.categories);
      if (catCands && catCands.length) {
        // 过滤 total=0 的空桶
        const valid = catCands.filter(d => d && d.category && (d.total || 0) > 0);
        if (valid.length) {
          const w = valid.reduce((a, b) => (a.pass_rate <= b.pass_rate ? a : b));
          // DIM_LABELS 翻译优先（safety→安全 等），命中不了就原样输出 category 名
          weakest = { label: DIM_LABELS[w.category] || w.category, rate: w.pass_rate };
        }
      }
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

// 渲染报告列表（首页）
async function renderList(reports) {
  const tests = await (await fetch('api/tests.json')).json();
  const hasTests = tests && tests.total;
  const ciAllPass = hasTests && tests.failed === 0 && tests.errors === 0;
  // 顶部 banner：复用现有 api 接口；数据计算是异步的但已与 tests.json 并行 fetch
  const bannerData = await computeBannerData(reports);
  $app.innerHTML = `
    ${bannerData ? renderBanner(bannerData) : ''}
    <h1>LLM 评测报告看板</h1>
    <div class="sub">共 ${reports.length} 份报告 · ${reports.reduce((a,r)=>a+r.case_count,0)} 个用例 · 最新 ${esc(reports[0].time)}</div>
    <div class="grid" style="margin-bottom:16px">
      <div class="stat"><div class="k">报告数</div><div class="v">${reports.length}</div></div>
      <div class="stat"><div class="k">单次最多用例</div><div class="v">${Math.max(...reports.map(r=>r.case_count))}</div></div>
      <div class="stat"><div class="k">最新通过率</div><div class="v" style="color:${rateColor(reports[0].pass_rate)}">${isMultiModel(reports[0]) ? `${pct(Math.min(...reports[0].model_rows.map(x=>x.pass_rate)))} ~ ${pct(Math.max(...reports[0].model_rows.map(x=>x.pass_rate)))}` : pct(reports[0].pass_rate)}</div></div>
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
      <tr><th>报告</th><th>时间</th><th>模型</th><th>用例</th><th>通过 / 失败</th><th>通过率</th><th>P95 延迟</th></tr>
      ${reports.map((r, i) => {
        const multi = isMultiModel(r);
        const modelCell = multi ? multiCell(r, x => `<div>${esc(x.model)}</div>`) : esc(r.model);
        const totalCell = multi ? multiCell(r, x => `<div>${x.total}</div>`) : `${r.case_count}`;
        const pfCell = multi ? multiCell(r, x => `<div>${x.passed} / ${x.failed}</div>`) : `${r.passed} / ${r.failed}`;
        const rateCellHtml = multi
          ? `<td>${multiCell(r, x => rateInner(x.pass_rate))}</td>`
          : rateCell(r.pass_rate);
        const p95Cell = multi
          ? multiCell(r, x => `<div>${x.p95 != null ? x.p95.toFixed(2) + ' ms' : '—'}</div>`)
          : (r.p95 != null ? r.p95.toFixed(2) + ' ms' : '—');
        return `
        <tr class="rowlink" data-go="${encodeURIComponent(r.file)}/summary">
          <td>#${reports.length - i} <span class="badge">${r.tag || 'run'}</span>${multi ? ` <span class="badge">${r.model_rows.length} 模型</span>` : ''}</td>
          <td class="catname">${esc(r.time)}</td>
          <td>${modelCell}</td>
          <td>${totalCell}</td>
          <td>${pfCell}</td>
          ${rateCellHtml}
          <td>${p95Cell}</td>
        </tr>`;
      }).join('')}
    </table></div>`;
}

// 渲染详情页框架（标题 + tabs + 当前 tab 内容）
async function renderDetail(file, tab) {
  const r = await (await fetch('api/report/' + encodeURIComponent(file))).json();
  $app.innerHTML = `
    <a class="back" href="#" onclick="location.hash='';return false">← 返回报告列表</a>
    <h1>${esc(file)}</h1>
    <div class="sub">${esc(r.started_at || '')} · ${r.case_count} 用例 · ${((r.duration_s ?? 0)).toFixed(2)}s
      · 数据集：${esc((r.datasets || []).join(', '))}</div>
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
function conclusionCard(c) {
  if (!c || !c.text) return '';
  return `
    <div class="card conclusion">
      <div style="font-weight:600;margin-bottom:6px">本报告结论</div>
      <div style="font-size:14px;line-height:1.7">${esc(c.text)}</div>
      ${(c.anomalies || []).length ? `
        <ul class="anomaly-list">${c.anomalies.map(a => `<li>${esc(a)}</li>`).join('')}</ul>` : ''}
    </div>`;
}

// 维度（主行）→ 分类（子行）两层分组：维度是能力面，分类是具体任务类型。
// 直接按 cases 现算，避免 dimensions / categories 两份聚合口径不一致（老报告只有其一）。
function buildDimCatGroups(cases) {
  const byModel = new Map();
  (cases || []).forEach(c => {
    const model = c.model || '（无）';
    const dim = c.dimension || 'untagged';
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

function dimCatCard(groups) {
  if (!groups.length) return '';
  const rows = groups.map(g => {
    const untagged = g.dimension === 'untagged';
    const mainRow = `
      <tr${untagged ? ' style="opacity:.75"' : ''}>
        <td>${esc(g.model)}</td>
        <td><span class="badge">${esc(dimLabel(g.dimension))}</span>
          ${untagged ? '<span class="catname" style="font-size:12px"> 未打标签</span>' : ''}</td>
        <td>${g.passed} / ${g.total}${g.skipped ? ` <span class="catname">(跳过 ${g.skipped})</span>` : ''}</td>
        ${rateCell(g.pass_rate)}
      </tr>`;
    const subRows = g.cats.map(c => `
      <tr style="background:var(--panel2)">
        <td></td>
        <td style="padding-left:22px"><span class="catname">↳ ${esc(c.category)}</span></td>
        <td>${c.passed} / ${c.total}${c.skipped ? ` <span class="catname">(跳过 ${c.skipped})</span>` : ''}</td>
        ${rateCell(c.pass_rate)}
      </tr>`).join('');
    return mainRow + subRows;
  }).join('');
  return `
    <div class="card">
      <div style="font-weight:600;margin-bottom:4px">维度 / 分类通过率</div>
      <div class="hint">模型在哪类任务上强弱：维度是能力面（主行），分类是具体任务类型（子行）。</div>
      <table>
        <tr><th>模型</th><th>维度 / 分类</th><th>通过 / 总数</th><th>通过率</th></tr>
        ${rows}
      </table>
    </div>`;
}

// 汇总 tab：结论 → 模型总览 → 维度/分类 → 指标均值（折叠）
function renderSummaryTab(r) {
  const s = r.summary || [];
  const groups = buildDimCatGroups(r.cases || []);
  document.getElementById('tabBody').innerHTML = `
    ${conclusionCard(r._conclusion)}
    <div class="card">
      <div style="font-weight:600;margin-bottom:4px">模型总览</div>
      <div class="hint">各模型整体表现对比。</div>
      <table>
        <tr><th>模型</th><th>通过 / 总数</th><th>通过率</th><th>avg 延迟</th><th>P95</th><th>tokens</th><th>成本</th></tr>
        ${s.map(m => `
          <tr>
            <td>${esc(m.model)}</td>
            <td>${m.passed} / ${m.total}</td>
            ${rateCell(m.pass_rate)}
            <td>${(m.avg_latency_ms ?? 0).toFixed(2)} ms</td>
            <td>${(m.p95_latency_ms ?? 0).toFixed(2)} ms</td>
            <td>${m.total_tokens ?? '—'}</td>
            <td>${m.cost ?? 0}</td>
          </tr>`).join('')}
      </table>
    </div>
    ${dimCatCard(groups)}
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
  let state = {status: 'all', category: 'all', q: ''};
  // 单行预览：压平换行 + 截断（promptfoo 式列表直接可见输入/输出摘要）
  const prev = (s, n) => {
    if (s == null) return '';
    const t = String(s).replace(/\s+/g, ' ').trim();
    return t.length > n ? t.slice(0, n) + '…' : t;
  };

  const draw = () => {
    const filtered = cases.filter(c => {
      if (state.status !== 'all' && statusOf(c) !== state.status) return false;
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
        每行展示<b>输入 / 模型输出</b>摘要，点行看完整 prompt、期望输出、评分明细。
      </div>
      <div class="toolbar">
        <select id="fltStatus">
          <option value="all" ${state.status==='all'?'selected':''}>全部状态</option>
          <option value="pass" ${state.status==='pass'?'selected':''}>通过</option>
          <option value="fail" ${state.status==='fail'?'selected':''}>失败</option>
          <option value="skip" ${state.status==='skip'?'selected':''}>跳过</option>
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
          <tr><th style="width:56px">状态</th><th style="width:110px">case_id</th><th>输入（prompt）</th><th>模型输出（response）</th><th style="width:90px">分类</th><th style="width:80px">延迟</th></tr>
          ${filtered.map(c => {
            const st = statusOf(c);
            const lat = c.latency_ms != null ? c.latency_ms.toFixed(2) + ' ms' : '—';
            const failMetric = (c.metrics || []).find(m => m.passed === false);
            return `
            <tr class="rowlink" data-case="${esc(c.case_id)}">
              <td><span class="badge ${st}">${st==='pass'?'✓':st==='fail'?'✗':'-'}</span></td>
              <td><code>${esc(c.case_id)}</code></td>
              <td class="catname" style="white-space:normal;max-width:280px">${esc(prev(c.prompt, 80))}</td>
              <td class="catname" style="white-space:normal;max-width:280px">${esc(prev(c.response, 80))}${st==='fail' && failMetric ? `<div class="err" style="font-size:12px;margin-top:3px">${esc(prev(failMetric.detail || failMetric.name, 60))}</div>` : ''}</td>
              <td><span class="badge">${esc(c.category || '（无）')}</span></td>
              <td>${lat}</td>
            </tr>`;
          }).join('')}
        </table>
      </div>`;
    document.getElementById('fltStatus').onchange = e => { state.status = e.target.value; draw(); };
    document.getElementById('fltCat').onchange = e => { state.category = e.target.value; draw(); };
    document.getElementById('fltQ').oninput = e => { state.q = e.target.value; draw(); };
  };
  // 用例 id → case 映射（用于行点击打开 drawer）
  window.casesById = new Map(cases.map(c => [c.case_id, c]));
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
    window.casesById = new Map((r.cases || []).map(c => [c.case_id, c]));
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
              <tr class="rowlink ${color}" data-case="${esc(cur.case_id)}">
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
      <a class="back" href="#" onclick="location.hash='';return false">← 返回报告列表</a>
      <h1>测试详情</h1>
      <div class="card"><div class="sub" style="margin:0">暂无测试详情（api/tests-detail.json 不存在）</div></div>`;
    return;
  }

  // 文件（classname 前缀）去重 + 排序
  const files = Array.from(new Set(tests.map(t => (t.classname || '').split('.')[0] || '(其他)'))).sort();
  const summaryInfo = summary || {};

  _testsState = { tests, files, summary: summaryInfo, filter: 'all', file: '', q: '', latestReport };

  $app.innerHTML = `
    <a class="back" href="#" onclick="location.hash='';return false">← 返回报告列表</a>
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
  if (trCase) { openDrawer(window.casesById.get(trCase.dataset.case)); return; }
});

// 路由入口：hash 形如 #<file>/<tab>，旧 #<file> 默认汇总；#tests 看 pytest 用例详情
async function main() {
  const hash = location.hash.slice(1);
  try {
    if (!hash) {
      const reports = await (await fetch('api/reports.json')).json();
      if (!reports.length) { $app.innerHTML = '<div class="empty">reports/ 目录下暂无报告，先跑一次评测 runner 吧。</div>'; return; }
      return renderList(reports);
    }
    const [file, tab = 'summary'] = hash.split('/');
    if (!file) return renderList(await (await fetch('api/reports.json')).json());
    if (file === 'tests') return renderTests();
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

PAGE = PAGE_TEMPLATE.replace(
    "__METRIC_LABELS__", json.dumps(METRIC_LABELS, ensure_ascii=False)
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
            "p95": summary.get("p95_latency_ms"),
        }]
        items.append({
            "file": p.name,
            "time": (m.group(0) if m else data.get("started_at", "?")),
            "tag": p.stem.split("-")[0] if "-" in p.stem else "run",
            "model": summary.get("model", "?"),
            "models": [row["model"] for row in model_rows],
            "model_rows": model_rows,
            "case_count": data.get("case_count", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "pass_rate": summary.get("pass_rate", 0.0),
            "p95": summary.get("p95_latency_ms"),
        })
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
        elif path.startswith("/api/report/"):
            name = os.path.basename(path[len("/api/report/"):])
            fp = REPORTS_DIR / name
            if fp.suffix == ".json" and fp.is_file():
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    # 汇总 tab 顶部结论由后端算好，前端只渲染（与静态导出保持一致）
                    if isinstance(data, dict):
                        data["_conclusion"] = compute_conclusion(data)
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
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"评测报告看板已启动: http://0.0.0.0:{PORT}  (数据源: {REPORTS_DIR})")
    server.serve_forever()


if __name__ == "__main__":
    main()
