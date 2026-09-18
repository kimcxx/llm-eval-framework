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

PAGE = """<!DOCTYPE html>
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
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: var(--muted); font-weight: 500; padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  td { padding: 8px 10px; border-bottom: 1px solid rgba(42,54,80,.5); white-space: nowrap; }
  tr:last-child td { border-bottom: none; }
  tr.rowlink { cursor: pointer; transition: background .15s; }
  tr.rowlink:hover { background: var(--panel2); }
  .rate { font-weight: 600; }
  .bar { position: relative; background: #232d47; border-radius: 5px; height: 8px; width: 130px; overflow: hidden; }
  .bar > i { position: absolute; inset: 0 auto 0 0; border-radius: 5px; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; background: var(--panel2); color: var(--muted); }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 12px; }
  .stat { background: var(--panel2); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .stat .k { color: var(--muted); font-size: 12px; }
  .stat .v { font-size: 20px; font-weight: 600; margin-top: 2px; }
  .back { color: var(--accent); text-decoration: none; display: inline-block; margin-bottom: 14px; font-size: 13px; }
  .back:hover { text-decoration: underline; }
  .catname { color: var(--muted); }
  .empty { color: var(--muted); padding: 40px; text-align: center; }
</style>
</head>
<body>
<div class="wrap" id="app"><div class="empty">加载中…</div></div>
<script>
const $app = document.getElementById('app');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = x => (x * 100).toFixed(1) + '%';
const rateColor = x => x >= 0.8 ? 'var(--ok)' : x >= 0.5 ? 'var(--warn)' : 'var(--bad)';
const rateCell = x => `<td><div style="display:flex;align-items:center;gap:8px"><span class="rate" style="color:${rateColor(x)}">${pct(x)}</span><span class="bar"><i style="width:${(x*100).toFixed(1)}%;background:${rateColor(x)}"></i></span></div></td>`;

async function main() {
  const hash = location.hash.slice(1);
  if (hash) return showDetail(decodeURIComponent(hash));
  const reports = await (await fetch('api/reports.json')).json();
  if (!reports.length) { $app.innerHTML = '<div class="empty">reports/ 目录下暂无报告，先跑一次评测 runner 吧。</div>'; return; }
  const best = Math.max(...reports.map(r => r.case_count));
  const totalCases = reports.reduce((a, r) => a + r.case_count, 0);
  $app.innerHTML = `
    <h1>LLM 评测报告看板</h1>
    <div class="sub">共 ${reports.length} 份报告 · ${totalCases} 个用例 · 最新 ${esc(reports[0].time)}</div>
    <div class="grid" style="margin-bottom:16px">
      <div class="stat"><div class="k">报告数</div><div class="v">${reports.length}</div></div>
      <div class="stat"><div class="k">单次最多用例</div><div class="v">${best}</div></div>
      <div class="stat"><div class="k">最新通过率</div><div class="v" style="color:${rateColor(reports[0].pass_rate)}">${pct(reports[0].pass_rate)}</div></div>
      <div class="stat"><div class="k">最新模型</div><div class="v" style="font-size:15px">${esc(reports[0].model)}</div></div>
    </div>
    <div class="card"><table>
      <tr><th>报告</th><th>时间</th><th>模型</th><th>用例</th><th>通过 / 失败</th><th>通过率</th><th>P95 延迟</th></tr>
      ${reports.map((r, i) => `
        <tr class="rowlink" onclick="location.hash='${encodeURIComponent(r.file)}'">
          <td>#${reports.length - i} <span class="badge">${r.tag || 'run'}</span></td>
          <td class="catname">${esc(r.time)}</td>
          <td>${esc(r.model)}</td>
          <td>${r.case_count}</td>
          <td>${r.passed} / ${r.failed}</td>
          ${rateCell(r.pass_rate)}
          <td>${r.p95 != null ? r.p95.toFixed(2) + ' ms' : '—'}</td>
        </tr>`).join('')}
    </table></div>`;
}

async function showDetail(file) {
  const r = await (await fetch('/api/report/' + encodeURIComponent(file))).json();
  const s = r.summary || [];
  $app.innerHTML = `
    <a class="back" href="#" onclick="location.hash='';return false">← 返回报告列表</a>
    <h1>${esc(file)}</h1>
    <div class="sub">${esc(r.started_at || '')} · ${r.case_count} 用例 · ${((r.duration_s ?? 0)).toFixed(2)}s
      · 数据集：${esc((r.datasets || []).join(', '))}</div>
    ${r.notes && r.notes.length ? `<div class="card" style="color:var(--muted);font-size:13px">备注：${esc(r.notes.join('；'))}</div>` : ''}
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">模型总览</div>
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
    ${(r.categories || {}).length ? `
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">分类通过率</div>
      <table>
        <tr><th>模型</th><th>分类</th><th>通过 / 总数</th><th>通过率</th></tr>
        ${(r.categories || {}).flatMap ? (r.categories || {}).flatMap((rows, model) =>
          rows.map(c => `
          <tr>
            <td>${esc(model)}</td>
            <td><span class="badge">${esc(c.category)}</span></td>
            <td>${c.passed} / ${c.total}</td>
            ${rateCell(c.pass_rate)}
          </tr>`)).join('') : ''}
      </table>
    </div>` : ''}
    ${(s[0] && s[0].metric_avg) ? `
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">指标均值（首个模型）</div>
      <table>
        ${Object.entries(s[0].metric_avg).map(([k, v]) => `
          <tr><td><span class="badge">${esc(k)}</span></td><td>${pct(v)}</td>
          <td><span class="bar" style="width:220px"><i style="width:${(v*100).toFixed(1)}%;background:${rateColor(v)}"></i></span></td></tr>`).join('')}
      </table>
    </div>` : ''}`;
}
main();
addEventListener('hashchange', () => main());
</script>
</body>
</html>"""


def _load_reports():
    """扫描 reports/*.json，返回按时间倒序的元数据列表。"""
    items = []
    for p in sorted(REPORTS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = (data.get("summary") or [{}])[0]
        m = re.search(r"\d{8}-\d{6}", p.stem)
        items.append({
            "file": p.name,
            "time": (m.group(0) if m else data.get("started_at", "?")),
            "tag": p.stem.split("-")[0] if "-" in p.stem else "run",
            "model": summary.get("model", "?"),
            "case_count": data.get("case_count", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "pass_rate": summary.get("pass_rate", 0.0),
            "p95": summary.get("p95_latency_ms"),
        })
    items.sort(key=lambda x: x["time"], reverse=True)
    return items


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
        if path == "/" or path == "/index.html":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/api/reports", "/api/reports.json"):
            self._json(_load_reports())
        elif path.startswith("/api/report/"):
            name = os.path.basename(path[len("/api/report/"):])
            fp = REPORTS_DIR / name
            if fp.suffix == ".json" and fp.is_file():
                try:
                    self._json(json.loads(fp.read_text(encoding="utf-8")))
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
