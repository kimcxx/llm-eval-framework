# -*- coding: utf-8 -*-
"""看板门户页（空 hash）契约测试。

背景：
    首页标题是「LLM 评测报告看板」、内容全是模型层，Agent 评测像附属品，
    访问者会一头雾水。改造后：空 hash 是门户页（两层各一张入口卡），
    老首页内容整体挪到 ``#llm``。

契约要点（都是「改错了会立刻被发现」的那几条）：
    * 站点标题改名为「LLM & Agent 评测实验室」，旧标题全局无残留；
    * 门户页有两个入口卡，分别指向 ``#llm`` 与 ``#agent``；
    * 路由认得 ``#llm``，空 hash 走门户而不是报告列表；
    * 各页「返回」一律回 ``#llm``，不再回空 hash（否则门户页永远回不去）。
"""

from __future__ import annotations

import dashboard.app as app

OLD_TITLE = "LLM 评测报告看板"
NEW_TITLE = "LLM &amp; Agent 评测实验室"


class TestSiteTitle:
    """两层实验室：标题必须改名，且旧名不留残留。"""

    def test_page_title_is_two_layer_lab(self) -> None:
        assert f"<title>{NEW_TITLE}</title>" in app.PAGE

    def test_old_title_has_no_residue(self) -> None:
        assert OLD_TITLE not in app.PAGE, f"页面里还有旧标题残留：{OLD_TITLE}"


class TestPortalCards:
    """门户页两张入口卡：等大并排，各带最新成绩与跳转。"""

    def test_two_entry_cards_exist(self) -> None:
        assert 'class="portal"' in app.PAGE, "门户容器 .portal 不存在"
        assert 'href="#llm"' in app.PAGE, "缺少模型层入口卡（#llm）"
        assert 'href="#agent"' in app.PAGE, "缺少 Agent 层入口卡（#agent）"

    def test_cards_are_equal_width_grid(self) -> None:
        """.portal 必须是等分网格，否则两张卡会一宽一窄。"""
        css = app.PAGE[app.PAGE.index(".portal {"):app.PAGE.index(".pcard {")]
        assert "repeat(auto-fit, minmax(320px, 1fr))" in css


class TestPortalCardContent:
    """卡片 v2：三段式（被测对象 / 怎么测 / 最新成绩），去掉序号与孤立「进入 →」。"""

    def test_no_serial_number_and_no_bare_enter(self) -> None:
        assert "①" not in app.PAGE and "②" not in app.PAGE, "卡片不该带 ①② 序号"
        assert "进入 →" not in app.PAGE, "孤立的「进入 →」不知所云，已统一为「查看 XX 详情 →」"

    def test_cards_end_with_detail_links(self) -> None:
        assert "查看模型层详情 →" in app.PAGE
        assert "查看 Agent 层详情 →" in app.PAGE

    def test_three_sections_per_card(self) -> None:
        """三段是运行时拼的，源码里表现为 sec('被测对象'...) 这样的调用。"""
        # 每张卡一段；「最新成绩」有「有数据 / 空态」两个分支，所以出现 4 次
        for label in ("被测对象", "怎么测", "最新成绩"):
            assert app.PAGE.count(f"sec('{label}'") >= 2, f"{label} 段应在两张卡上都出现"

    def test_intro_names_the_subjects(self) -> None:
        """简介必须点名：deepseek-chat / deepseek-pro / smolagents CodeAgent。"""
        for name in ("deepseek-chat", "deepseek-pro", "smolagents ToolCallingAgent"):
            assert name in app.PAGE, f"简介段缺少被测对象名：{name}"


class TestScoringSource:
    """模型层成绩来源：跳过 mock-only 报告（CD 冒烟的假模型不能当成绩）。"""

    def test_scoring_report_skips_mock(self) -> None:
        assert "function pickScoringReport(reports)" in app.PAGE
        body = app.PAGE[app.PAGE.index("function pickScoringReport"):]
        body = body[:body.index("\n}")]
        assert "isMockModel" in body, "选成绩报告时必须排除 mock-* 对照组"

    def test_score_rows_filter_mock(self) -> None:
        body = app.PAGE[app.PAGE.index("async function renderPortal"):]
        body = body[:body.index("$app.innerHTML")]
        assert "!isMockModel(x.model)" in body, "成绩里的模型行必须过滤掉 mock-*"


    """路由：空 hash → 门户，#llm → 老首页，#agent → Agent 页。"""

    def test_empty_hash_renders_portal(self) -> None:
        assert "const isPortal = !hash || hash === '/';" in app.PAGE
        assert "if (isPortal) {" in app.PAGE

    def test_llm_route_keeps_old_home(self) -> None:
        assert "if (file === 'llm') return renderList(reports);" in app.PAGE

    def test_agent_route_unchanged(self) -> None:
        assert "if (file === 'agent') return renderAgent();" in app.PAGE


class TestBackLinks:
    """各页返回一律回 #llm（门户页不再被空 hash 顶掉）。"""

    def test_no_back_link_to_empty_hash(self) -> None:
        assert "location.hash=''" not in app.PAGE

    def test_every_back_link_points_to_llm(self) -> None:
        total = app.PAGE.count("← 返回报告列表")
        to_llm = app.PAGE.count('href="#llm">← 返回报告列表</a>')
        assert total > 0, "页面里一个返回链接都没有，回归失去意义"
        assert total == to_llm, f"{total - to_llm} 个返回链接没指向 #llm"

    def test_layer_pages_have_back_to_portal(self) -> None:
        """#llm / #agent 两个层级页各一个「返回实验室首页」（指向空 hash 门户）。"""
        # 详情页与 #tests 是三层深，仍回 #llm；层级页回门户，两者并存
        assert app.PAGE.count('href="#">← 返回实验室首页</a>') == 2
        assert "← 返回实验室首页" in app.PAGE
