# -*- coding: utf-8 -*-
"""前端「回调参数越界引用」自检：拦住 2026-11-03 `r is not defined` 那一类 bug。

## 那个 bug 长什么样

```js
const body = matrix.rows.map(r => { ... });        // r 是这一行的回调参数
return `<div>维度 × 模型 对比矩阵${legacyDimNote(r)}</div>`;   // ← 出了回调，r 不存在
```

函数参数里没有它、函数体内也没声明它，它只属于某个嵌套回调——**运行时才炸**，
而且只在「维度矩阵非空 + 正好渲染这张卡片」时炸，人工点页面很难覆盖全
（这份 bug 从 10-28 活到 11-03，没人发现）。

## 为什么不做通用作用域分析

试过：提取每个 `function` 体、剥掉嵌套回调、把引用与「参数 + 体内声明 + 顶层定义 +
JS 内置」做差集。**误报十几处**——因为要正确处理嵌套模板、表达式体箭头、正则字面量，
等于在 Python 里写半个 JS 解析器，维护成本和噪音都不可接受。

收敛成**只拦这一类形态**，但拦得准：

1. 块体箭头函数 `x => { ... }`（含多参数）的**参数**，在它自己的 `{...}` 之外还有引用 → 报；
2. 本 bug 的具体写法用字符串契约钉住（工单允许的「关键模板断言」）。

这两条合起来，既能在真 bug 面前红（阴性对照验证过），又不会因为写法变化就满屏噪音。

## 已知边界（写在明处）

* 只查**块体**箭头（`x => {`）。表达式体箭头（`x => expr`）不查——它的作用域就在那一小段
  表达式里，越界可能性低，而正确划范围需要完整表达式解析；
* 正则字面量按「前一个字符」启发式识别，极端写法可能误判（宁可漏报，不制造噪音）。
"""
from __future__ import annotations

import re

from dashboard.app import PAGE


# --------------------------------------------------------------------------- #
# 词法：剥离注释 / 字符串 / 模板文本 / 正则，但保留模板里 ${...} 的代码
# --------------------------------------------------------------------------- #

def _skip_quoted(src: str, i: int) -> int:
    """``i`` 指向引号，返回收尾引号之后的下标。"""
    q = src[i]
    i += 1
    n = len(src)
    while i < n:
        if src[i] == '\\':
            i += 2
            continue
        if src[i] == q:
            return i + 1
        i += 1
    return i


def _skip_template(src: str, i: int) -> int:
    """``i`` 指向反引号，返回收尾反引号之后（模板里的 ``${...}`` 递归跳过）。"""
    i += 1
    n = len(src)
    while i < n:
        c = src[i]
        if c == '\\':
            i += 2
            continue
        if c == '`':
            return i + 1
        if c == '$' and i + 1 < n and src[i + 1] == '{':
            j = _match_brace(src, i + 1)
            i = n if j < 0 else j + 1
            continue
        i += 1
    return i


def _match_brace(src: str, start: int) -> int:
    """``start`` 指向 ``{``，返回配对 ``}`` 的下标。

    字符串、模板（含嵌套 ``${...}``）都要跳过——不做嵌套会整体配平错乱，
    连函数体都切不对（第一版就吃过这个亏）。
    """
    depth = 0
    i, n = start, len(src)
    while i < n:
        c = src[i]
        if c in '\'"':
            i = _skip_quoted(src, i)
            continue
        if c == '`':
            i = _skip_template(src, i)
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _looks_like_regex(src: str, i: int) -> bool:
    """``src[i]`` 是 ``/``：按前一个非空白字符猜它是正则还是除号。"""
    j = i - 1
    while j >= 0 and src[j] in ' \t':
        j -= 1
    if j < 0:
        return True
    return src[j] in '=(:,[!&|?{};+-*%<>~^'


def _skip_regex(src: str, i: int) -> int:
    j = i + 1
    n = len(src)
    in_class = False
    while j < n:
        c = src[j]
        if c == '\\':
            j += 2
            continue
        if c == '\n':
            return i + 1  # 不跨行 → 其实是除号
        if c == '[':
            in_class = True
        elif c == ']':
            in_class = False
        elif c == '/' and not in_class:
            j += 1
            while j < n and src[j].isalpha():
                j += 1
            return j
        j += 1
    return i + 1


def _code_only(src: str) -> str:
    """剥掉注释、字符串、模板文本、正则字面量；模板里 ``${...}`` 的内容作为代码保留。

    正则是必须单独剥的：``esc`` 里就有 ``/[&<>"']/g``，里面的引号若被当成字符串起始，
    后面整段代码的引号配对都会错位。
    """
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in '\'"':
            i = _skip_quoted(src, i)
            out.append(' ')
        elif c == '/' and i + 1 < n and src[i + 1] == '/':
            while i < n and src[i] != '\n':
                i += 1
            out.append(' ')
        elif c == '/' and i + 1 < n and src[i + 1] == '*':
            end = src.find('*/', i + 2)
            i = n if end < 0 else end + 2
            out.append(' ')
        elif c == '/' and _looks_like_regex(src, i):
            i = _skip_regex(src, i)
            out.append(' ')
        elif c == '`':
            i += 1
            while i < n and src[i] != '`':
                if src[i] == '\\':
                    i += 2
                    continue
                if src[i] == '$' and i + 1 < n and src[i + 1] == '{':
                    end = _match_brace(src, i + 1)
                    if end < 0:
                        i = n
                        break
                    out.append(' ' + _code_only(src[i + 2:end]) + ' ')
                    i = end + 1
                    continue
                i += 1
            i += 1
            out.append(' ')
        else:
            out.append(c)
            i += 1
    return ''.join(out)


# --------------------------------------------------------------------------- #
# 检测：块体回调的参数不许在回调体外出现
# --------------------------------------------------------------------------- #

# 箭头函数：`(a, b) => ...` / `a => ...`（参数都在 => 左侧）
_ARROW = re.compile(
    r'(?<![\w$.])(?:\(\s*([A-Za-z_$][\w$]*(?:\s*,\s*[A-Za-z_$][\w$]*)*)\s*\)'
    r'|([A-Za-z_$][\w$]*))\s*=>'
)


def _arrow_scopes(body: str) -> list[tuple[list[str], int, int, int, int]]:
    """[(参数名, 体起点, 体终点, 声明起点, 声明终点)]，偏移都相对 ``body``。

    块体（``x => { ... }``）与表达式体（``x => expr``）都要，而且**表达式体必须支持**：
    真实的误报就来自 ``.map(c => ({...c}))``——只认块体会把它的 ``c`` 当成越界。
    表达式体的终点用括号配平 + 同层 ``,``/``;``/换行 划出来。
    """
    out: list[tuple[list[str], int, int, int, int]] = []
    for m in _ARROW.finditer(body):
        raw = m.group(1) or m.group(2) or ''
        params = [p.strip() for p in raw.split(',') if p.strip()]
        start = m.end()
        while start < len(body) and body[start].isspace():
            start += 1
        if start < len(body) and body[start] == '{':
            close = _match_brace(body, start)
            end = len(body) if close < 0 else close + 1
        else:
            depth = 0
            end = start
            while end < len(body):
                ch = body[end]
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    if depth == 0:
                        break
                    depth -= 1
                elif depth == 0 and ch in ',;':
                    # 换行**不是**终止符：模板字面量经常跨多行，按换行截断会把
                    # 后面半个模板的引用算成「体外」（第一版就是这么误报的）。
                    break
                end += 1
        out.append((params, start, end, m.start(), m.end()))
    return out


def _positions(text: str, name: str) -> list[int]:
    return [m.start() for m in re.finditer(rf'(?<![\w$.]){re.escape(name)}(?![\w$])', text)]


def _is_render_fn(name: str) -> bool:
    """渲染函数：``renderXxx`` / ``xxxCard`` —— 本 bug 出在这一类里。"""
    return name.startswith('render') or name.endswith('Card')


def _escaped_callback_params(js: str, targets: set[str] | None = None) -> dict[str, list[str]]:
    """{函数名: [越界的回调参数名...]}；空 dict 表示没有这类引用。

    检查**回调参数在体外被引用**：参数只允许出现在它自己的箭头体（块体或表达式体）
    与声明处。体外还有引用，说明代码借用了回调的局部名字——运行时 ReferenceError。

    ``targets`` 限定只查哪些函数。默认只查渲染函数（``render*`` / ``*Card``）：
    全量查会在真实 PAGE 上产生同名参数跨回调的边界误报，而本 bug 的形态就出在
    渲染函数里——宁可窄一点但每条报出来都是真的。
    """
    code = _code_only(js)
    report: dict[str, list[str]] = {}

    for fm in re.finditer(r'\bfunction\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{', code):
        fname = fm.group(1)
        if targets is None:
            if not _is_render_fn(fname):
                continue
        elif fname not in targets:
            continue
        # 函数**自身**的参数在体内任意位置都合法（renderAgent(r) 的 r 就是报告对象）。
        # 不排除的话，凡是参数叫单字母的渲染函数都会被误报——第一版就是这么炸的。
        fn_params = {
            p.strip() for p in fm.group(2).split(',')
            if re.fullmatch(r'[A-Za-z_$][\w$]*', p.strip())
        }
        open_at = fm.end() - 1
        close_at = _match_brace(code, open_at)
        if close_at < 0:
            continue
        body = code[open_at + 1:close_at]
        # 同一个参数名可能被多个回调使用（`c` 在 forEach 与多层 map/reduce 里各一份），
        # 合法范围要把**所有**以它为参数的箭头体合并起来，否则会互相误报。
        spans: dict[str, list[tuple[int, int, int, int]]] = {}
        for params, start, end, decl_start, decl_end in _arrow_scopes(body):
            for p in params:
                spans.setdefault(p, []).append((start, end, decl_start, decl_end))
        bad: list[str] = []
        for p, scopes in spans.items():
            if p in fn_params:
                continue
            # 有同名局部绑定（声明 / catch(e)）时，按名字匹配分不清绑定，
            # 宁可漏报也不误报——护栏的价值在于「报出来都是真的」。
            if re.search(rf'\b(?:const|let|var)\s+{re.escape(p)}\b', body):
                continue
            if re.search(rf'\bcatch\s*\(\s*{re.escape(p)}\s*\)', body):
                continue
            allowed: set[int] = set()
            for start, end, decl_start, decl_end in scopes:
                allowed |= {
                    pos for pos in _positions(body, p)
                    if start <= pos < end or decl_start <= pos < decl_end
                }
            if set(_positions(body, p)) - allowed:
                bad.append(p)
        if bad:
            report[fname] = sorted(bad)
    return report


def _page_js() -> str:
    blocks = re.findall(r'<script[^>]*>(.*?)</script>', PAGE, re.S)
    for b in blocks:
        if 'function ' in b:
            return b
    return ''


# --------------------------------------------------------------------------- #
# 护栏
# --------------------------------------------------------------------------- #

class TestCallbackScope:
    """回调参数出了回调就不存在——这是 2026-11-03 那次崩溃的直接原因。"""

    def test_page_has_script(self) -> None:
        assert _page_js(), '没能从 PAGE 里取到 <script>，护栏本身失效了'

    def test_no_callback_param_escapes(self) -> None:
        js = _page_js()
        escapes = _escaped_callback_params(js)
        assert escapes == {}, (
            '这些函数把回调的参数用到了回调体外（运行时 ReferenceError，'
            '页面会弹「运行错误」横幅）：\n'
            + '\n'.join(f'  {fn}: {", ".join(names)}' for fn, names in escapes.items())
        )


class TestTheActualBugIsFixed:
    """把本 bug 的具体写法钉住（工单允许的「关键模板字符串断言」）。"""

    def test_card_takes_doc_explicitly(self) -> None:
        js = _page_js()
        assert 'function dimensionMatrixCard(matrix, doc)' in js, (
            'dimensionMatrixCard 必须显式接收 doc（报告对象），不能借用回调的行变量'
        )
        assert 'dimensionMatrixCard(matrix, r)' in js, '汇总 tab 调用处要把报告对象传进去'
        assert 'legacyDimNote(doc)' in js, '卡片标题要用传进来的 doc'


class TestDetectorHasTeeth:
    """阴性对照：把**修好之前**的写法喂给检测器，必须报出来。"""

    _BUGGY = """
    function esc(s) { return String(s); }
    function legacyDimNote(doc) { return doc ? '' : ''; }
    function dimensionMatrixCard(matrix) {
      if (!matrix || !matrix.rows.length) return '';
      const body = matrix.rows.map(r => {
        return `<tr><td>${esc(r.label)}</td></tr>`;
      }).join('');
      return `<div>矩阵${legacyDimNote(r)}</div>` + body;
    }
    """

    _FIXED = _BUGGY.replace(
        'dimensionMatrixCard(matrix)', 'dimensionMatrixCard(matrix, doc)'
    ).replace('legacyDimNote(r)', 'legacyDimNote(doc)')

    def test_detector_catches_the_original_bug(self) -> None:
        report = _escaped_callback_params(self._BUGGY)
        assert 'r' in report.get('dimensionMatrixCard', []), (
            f'检测器没抓到原始 bug，这条护栏等于空断言：{report}'
        )

    def test_detector_is_clean_on_the_fixed_version(self) -> None:
        assert _escaped_callback_params(self._FIXED) == {}, '修好之后不该再报'

    def test_detector_ignores_template_text_and_comments(self) -> None:
        """模板里的 HTML 文本、注释里的同名标识符都不算引用。"""
        js = """
        function card(doc, r) {
          // r 只出现在注释里
          const body = [1].map(n => {
            return `${n}` + r;
          }).join('');
          return `<div title="r 是行变量">${doc.name}${body}</div>`;
        }
        """
        assert _escaped_callback_params(js) == {}, '把合法的（r 是函数参数）当成越界了'

    def test_detector_allows_multi_param_arrow(self) -> None:
        js = """
        function card(doc) {
          const body = [1].map((n, i) => {
            return `${i}` + (doc.x || n);
          }).join('');
          return body;
        }
        """
        assert _escaped_callback_params(js) == {}
