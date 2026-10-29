#!/usr/bin/env python
"""RAG 层评测入口。

用法：
    python rag_eval/run_eval.py                                          # 真实跑（默认全部可用模型）
    python rag_eval/run_eval.py --models deepseek-chat --tag rag-v1
    python rag_eval/run_eval.py --models mock-baseline --no-embedding --tag ci-rag   # CI 冒烟，零成本
    python rag_eval/run_eval.py --retrieval-only                         # 只测检索层 命中率@k / 召回率@k，不调模型

与模型层入口（`scripts/run_eval.py`）的关系：判分、执行引擎、报告生成全部复用
`src/`，这里只做两件特有的事——把被测对象换成 mini-RAG 管道，并额外统计检索层指标。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_eval.common import ensure_utf8_stdio  # noqa: E402
from src.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from src.datasets import load_datasets  # noqa: E402
from src.llm import build_client, build_clients  # noqa: E402
from src.metrics import MetricFactory  # noqa: E402
from src.report import write_reports  # noqa: E402
from src.runner import EvalRunner  # noqa: E402

from rag_eval.knowledge import (  # noqa: E402
    check_provenance,
    load_knowledge_base,
    load_provenance,
)
from rag_eval.pipeline import DEFAULT_TOP_K, RAGLLM  # noqa: E402
from rag_eval.retrieval import evaluate_retrieval, format_retrieval, gold_doc_ids  # noqa: E402
from rag_eval.retriever import Retriever  # noqa: E402

DATASET_NAME = "rag_cases"
# RAG 用例集放在 rag_eval/datasets/ 而不是仓库根的 datasets/：
# 模型层入口 scripts/run_eval.py 会扫描 datasets/ 整个目录，放那里会被扫进
# 模型层全量回归（rag_cases 曾被当成模型层题跑，混出一份 111 条的脏报告）。
# 各层的用例各归各位，模型层扫 datasets/ 时自然扫不到 RAG 题。
RAG_DATASETS_DIR = ROOT / "rag_eval" / "datasets"
RESULTS_DIR = ROOT / "rag_eval" / "results"
ARCHIVE_DIR = RESULTS_DIR / "archive"

# 考点（想测什么能力）→ 判定方式（怎么判）→ 指标（结果数字）是三个层次，别混为一谈：
# 逐题明细把它们分成三列，报告顶部只展示指标（数字）。focus 写在用例集里，
# 由加载器塞进 EvalCase.meta（src/ 的 schema 不动，判据口径一个字不改）。
FOCUS_LABELS = {
    "end_to_end": "端到端命中",
    "faithfulness": "忠实度",
    "honesty": "无答案诚实",
}
FOCUS_WHITELIST = tuple(FOCUS_LABELS)
# judge 缺席时忠实度必须显式显示「跳过」，不许留空、不许显示 0/100%
# （CD 报告那次的教训：跳过要写得跟通过率一样醒目）
JUDGE_MISSING_TEXT = "跳过（未配置裁判模型）"


def case_focus(case: Any) -> str:
    """取用例的考点取值；老数据没写 focus 时给空串（页面显示「—」）。"""
    return str((getattr(case, "meta", None) or {}).get("focus") or "").strip()


def rag_focus_summary(rows: list[dict[str, Any]], retrieval: dict[str, Any]) -> dict[str, Any]:
    """五项指标汇总（写在结果 JSON 的 metrics_summary 里，供看板直接渲染）。

    * 检索层 命中率@k：**沿用 retrieval 的既有口径**（命中 ÷ 可判），字段名与数值不动，
      只改显示名——它算的是 Hit Rate@k（只要有一段金标进 top-k 就算），不是 Recall@k；
    * 检索层 召回率@k：逐题 `|top-k ∩ gold| ÷ |gold|` 的算术平均。**老结果没有这个字段，
      value 为 None，页面显示「需重跑」**——没有数就写没有，不猜；
    * 端到端准确率：通过数 ÷ 应判数（应判 = 通过 + 失败，**跳过不计入分母**）；
    * 拒答率：考点为 honesty 那几条的通过比例；
    * 忠实度：考点为 faithfulness 那几条由 judge 裁的通过比例；
      **judge 缺席时整项标记 skipped**，reason 用 JUDGE_MISSING_TEXT——不给数字。
    """

    def rate(sub: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [r for r in sub if r.get("status") in ("通过", "失败")]
        passed = [r for r in scored if r.get("status") == "通过"]
        return {
            "value": (len(passed) / len(scored)) if scored else None,
            "passed": len(passed),
            "scored": len(scored),
        }

    # 召回率的两个原始计数（捞回的金标段数 / 金标总段数），给卡片第三行用。
    # 注意它俩的分母与「可判题数」不同——金标可能一题多段（rag-017 是 2 段）。
    scored_cases = [c for c in (retrieval.get("per_case") or []) if c.get("scored")]
    gold_got = sum(int(c.get("gold_hit_count") or 0) for c in scored_cases)
    gold_all = sum(int(c.get("gold_total") or 0) for c in scored_cases)
    honesty_rows = [r for r in rows if r.get("focus") == "honesty"]
    faith_rows = [r for r in rows if r.get("focus") == "faithfulness"]
    # judge 有没有真的判过：缺席时 judge 根本不会出现在 result.metrics 里（被跳过不计失败）
    judge_seen = any(m.get("name") == "judge" for r in faith_rows for m in (r.get("metrics") or []))
    faith = rate(faith_rows)
    return {
        # 名字写准：这是**命中率**（Hit Rate@k），字段名 recall_at_k 为兼容历史保留
        "recall_at_k": {
            "label": f"检索层 命中率@{retrieval.get('k', 2)}",
            "value": retrieval.get("recall_at_k"),
            "passed": retrieval.get("hit", 0),
            "scored": retrieval.get("scored", 0),
            "skipped": False,
            "reason": "",
        },
        # 新增：真正的**召回率**（逐题 |top-k ∩ gold| ÷ |gold| 的算术平均）。
        # 老结果没有这个字段 → value 为 None，页面显示「需重跑」，不许拿命中率顶替。
        "recall_ratio_at_k": {
            "label": f"检索层 召回率@{retrieval.get('k', 2)}",
            "value": retrieval.get("recall_ratio_at_k"),
            "passed": gold_got,
            "scored": gold_all,
            "unit": "段",
            # reason 只在**真没有数**时给：有值却写着「需重跑」会让读 JSON 的人误判
            "reason": "" if retrieval.get("recall_ratio_at_k") is not None else "需重跑（这份结果没有召回率字段）",
        },
        "end_to_end": {
            "label": "端到端准确率",
            **rate(rows),
            "skipped": False,
            "reason": "",
        },
        "honesty": {
            "label": "拒答率",
            **rate(honesty_rows),
            "skipped": False,
            "reason": "",
        },
        "faithfulness": {
            "label": "忠实度",
            "value": faith["value"] if judge_seen else None,
            "passed": faith["passed"] if judge_seen else 0,
            "scored": faith["scored"] if judge_seen else 0,
            "skipped": not judge_seen,
            "reason": "" if judge_seen else JUDGE_MISSING_TEXT,
        },
    }
DEFAULT_TAG = "rag"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG 层评测：mini-RAG 固定管道（检索 top-k + 生成）走现有判分与报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="模型配置文件")
    parser.add_argument("--models", nargs="*", help="只评测指定模型，默认评测全部可用模型")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="每问检索多少段资料")
    parser.add_argument("--limit", type=int, default=None, help="最多取多少条用例")
    parser.add_argument("--workers", type=int, default=None, help="并发线程数")
    parser.add_argument("--tag", default=None, help="报告文件名前缀，默认 rag")
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="检索与判分都走字符级（不加载向量模型，CI 零成本）",
    )
    parser.add_argument("--strict", action="store_true", help="缺少 API Key 的模型直接报错")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="声明这次是**对照跑**（实验，不是成绩）：结果 JSON 写 run_kind=comparison，"
        "看板不拿它当「最新成绩」。例如人为用 --no-embedding 强制降级做后端对比。",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只跑检索层 命中率@k / 召回率@k，不调用任何模型（零成本，CI 可跑）",
    )
    return parser.parse_args(argv)


def run_kind_of(args: Any, models: list[str]) -> str:
    """这次跑的性质：`real`（正式）/ `comparison`（对照）/ `mock`（自检）。

    `--compare` 是**显式声明**：对照跑必须自己说清楚，不许靠 tag 命名去猜——
    tag 是给人看的，不是给程序判的（同一个 tag 既能跑正式也能跑对照）。
    没声明时按模型名认自检跑（mock-* 是管道基线，不是被测对象）。
    """

    if getattr(args, "compare", False):
        return "comparison"
    if models and all(str(m).removeprefix("rag-").startswith("mock-") for m in models):
        return "mock"
    return "real"


def rel_or_abs(path: Path) -> str:
    """结果文件里写相对路径：仓库可以整体搬家，路径不会失效。"""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def archive_previous(current: Path) -> None:
    """把 results/ 里的旧结果移进 archive/（与 Agent 层同款约定）。

    在**新跑开始时**调用（不是跑完才归档）：上一份结果必须先归位，新结果才能落盘——
    顺序反了的话，一次失败/中断的跑就会把历史顶掉（"跑完再归档"看着更稳，
    但中途崩溃时新文件已写一半，旧的却还在原位等着被下一次覆盖）。
    归档 = 移动不是删除，历史永远找得回来。
    """
    # 注意：不要在开头加 `if not current.exists(): return`。归档是在**新跑开始时**
    # 调的，那一刻新文件必然还不存在——加了这句等于永远不归档（历史被静静覆盖）。
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for old in sorted(RESULTS_DIR.glob("result-*.json")):
        if old.name == current.name:
            continue
        target = ARCHIVE_DIR / old.name
        if target.exists():
            target = ARCHIVE_DIR / f"{old.stem}-{datetime.now().strftime('%H%M%S')}{old.suffix}"
        old.replace(target)


def case_rows(report: Any, cases: list[Any], rag_clients: dict[str, RAGLLM]) -> list[dict[str, Any]]:
    """把每条用例的答案、依据（检索到的文档）、命中情况与判定拼成可核对的明细。"""
    gold_by_id = {c.id: gold_doc_ids(c) for c in cases}
    focus_by_id = {c.id: case_focus(c) for c in cases}
    rows: list[dict[str, Any]] = []
    for result in report.cases:
        client = rag_clients.get(result.model)
        hits = client.hits_by_question.get(result.prompt) if client else None
        retrieved = [h.doc_id for h in hits] if hits else []
        gold = gold_by_id.get(result.case_id, [])
        status = "通过" if result.passed is True else ("跳过" if result.passed is None else "失败")
        rows.append(
            {
                "id": result.case_id,
                "dimension": result.dimension,
                # 考点（想测什么能力）——与「判定方式」「结果」分开显示，三个层次不许混
                "focus": focus_by_id.get(result.case_id, ""),
                "prompt": result.prompt,
                "answer": result.response_text,
                "status": status,
                "retrieved": retrieved,
                "gold": gold,
                "gold_hit": bool(gold) and any(g in retrieved for g in gold),
                "metrics": [
                    {"name": m.name, "score": round(m.score, 4), "passed": m.passed, "detail": m.detail}
                    for m in result.metrics
                ],
                "skipped_metrics": list(result.skipped_metrics),
                "error": result.error,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_utf8_stdio()

    config = load_config(args.config)
    top_k = max(1, int(args.top_k))

    cases = load_datasets(RAG_DATASETS_DIR, [DATASET_NAME])
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        print(f"没有加载到 {DATASET_NAME} 用例，请检查 rag_eval/datasets/{DATASET_NAME}.jsonl")
        return 1

    docs = load_knowledge_base()
    meta = load_provenance()
    retriever = Retriever(docs, prefer_embedding=not args.no_embedding)

    retrieval = evaluate_retrieval(cases, retriever, top_k)
    print()
    print(format_retrieval(retrieval))
    print()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.retrieval_only:
        target = RESULTS_DIR / f"retrieval-{stamp}.json"
        target.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "top_k": top_k,
                    "doc_count": len(docs),
                    "case_count": len(cases),
                    "retrieval": retrieval,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"检索层结果已写入 {rel_or_abs(target)}（未调用任何模型）")
        return 0

    # 新跑开始前先给上一份结果归位：不许覆盖、不许删除（见 archive_previous 的说明）。
    archive_previous(RESULTS_DIR / f"result-{stamp}.json")

    raw_clients = build_clients(config, args.models, strict=args.strict)
    if not raw_clients:
        print("没有可用模型（缺 API Key 或配置为空）；CI 冒烟请用 --models mock-baseline")
        return 1

    # 被测对象的名字必须带 rag- 前缀：报告与看板要能一眼看出这是「接了 RAG 管道的
    # 模型」，而不是裸模型（否则同一份报告里 rag-mock-baseline 与 mock-baseline 会混）。
    rag_clients: dict[str, RAGLLM] = {}
    rag_model_configs: dict[str, Any] = {}
    model_configs = {cfg.name: cfg for cfg in config.models}
    for name, client in raw_clients.items():
        rag = RAGLLM(client, retriever=retriever, top_k=top_k)
        rag_clients[rag.name] = rag
        if name in model_configs:
            rag_model_configs[rag.name] = model_configs[name]

    judge_client = None
    judge_cfg = config.get_judge()
    if judge_cfg is not None and judge_cfg.is_available:
        judge_client = build_client(judge_cfg, config.run)
        print(f"裁判模型 {judge_cfg.name} 已加载（用于忠实度判定，不计入被测对象）")
    else:
        reason = getattr(judge_cfg, "unavailable_reason", "未配置") if judge_cfg else "未配置"
        print(f"裁判模型不可用（{reason}），judge 指标将被跳过（记为跳过，不计入失败）")

    factory = MetricFactory(
        similarity_threshold=config.run.similarity_threshold,
        judge_threshold=config.run.judge_threshold,
        schema_match_threshold=config.run.schema_match_threshold,
        judge_client=judge_client,
        prefer_embedding=not args.no_embedding,
        judge_only_categories=config.run.judge_only_categories,
    )

    provenance_note = check_provenance(docs, meta)
    extra_notes = [
        "被测对象是项目内自建的 mini-RAG 固定管道：每问无条件检索 top-"
        f"{top_k} 再生成，不做「要不要检索」的决策、不产生多步轨迹，"
        "因此与 Agent 层区分，单独成层。",
        f"检索后端：{retriever.describe()}；知识库："
        f"{rel_or_abs(Path('rag_eval/fixtures/knowledge_base.jsonl'))}（{len(docs)} 段，冻结快照）。",
        # 降级原因的异常原文**只留在报告 JSON 的 notes 里**：卡片上放原文等于糊一堆报错，
        # 排查时要看原文的人自然会翻 notes。
        *(
            [f"检索后端降级原因（异常原文）：{retriever.init_error}"]
            if retriever.init_error
            else []
        ),
        f"检索层 命中率@{top_k} = {retrieval['recall_at_k']:.1%}"
        f"（可判 {retrieval['scored']} 条，命中 {retrieval['hit']} 条）；"
        + (
            f"召回率@{top_k} = {retrieval['recall_ratio_at_k']:.1%}"
            f"（逐题捞回金标段数的算术平均，可判 {retrieval['scored']} 题）。"
            if retrieval.get("recall_ratio_at_k") is not None
            else ""
        )
        + "答错时可据此区分「没检索到」与「检索到了没用好」；"
        "两个指标在「一题对一段」时数值相同，一题多段（如 rag-017）时才看得出差别。",
        "知识库与题目目前是占位版（取材项目自身可核对的事实），正式内容由出题方逐条拍板后替换。",
    ]
    if provenance_note:
        extra_notes.append(provenance_note)

    runner = EvalRunner(
        clients=rag_clients,  # type: ignore[arg-type]
        factory=factory,
        run=config.run,
        model_configs=rag_model_configs,
        config_path=str(args.config),
        workers=args.workers,
        extra_notes=extra_notes,
    )

    report = runner.run_cases(cases)

    print("\n" + "=" * 74)
    print(f"{'被测对象':<24}{'用例':>6}{'通过':>6}{'失败':>6}{'跳过':>6}{'通过率':>10}")
    print("-" * 74)
    for stats in report.overall():
        print(
            f"{stats.key:<24}{stats.total:>6}{stats.passed:>6}{stats.failed:>6}"
            f"{stats.skipped:>6}{stats.pass_rate:>9.1%}"
        )
    print("=" * 74)

    written = write_reports(report, config.paths.output_dir, tag=args.tag or DEFAULT_TAG)
    report_file = Path(written["json"])

    result_file = RESULTS_DIR / f"result-{stamp}.json"
    detail_rows = case_rows(report, cases, rag_clients)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        # 这次跑的性质（real / comparison / mock）：看板据此决定它能不能进「最新成绩」。
        # 对照跑（如强制降级做后端对比）是实验不是成绩，不许顶掉正式跑的位置。
        "run_kind": run_kind_of(args, list(rag_clients.keys())),
        "models": list(rag_clients.keys()),
        "top_k": top_k,
        "retriever": retriever.describe(),
        "knowledge_base": {
            "path": rel_or_abs(Path("rag_eval/fixtures/knowledge_base.jsonl")),
            "doc_count": len(docs),
            "provenance": meta,
        },
        "report_file": rel_or_abs(report_file),
        "summary": [
            {
                "model": s.key,
                "total": s.total,
                "passed": s.passed,
                "failed": s.failed,
                "skipped": s.skipped,
                "pass_rate": s.pass_rate,
            }
            for s in report.overall()
        ],
        "retrieval": retrieval,
        "cases": detail_rows,
        # 顶部「指标汇总」四项（数字），逐题明细里则是考点 / 判定方式 / 结果三列
        "metrics_summary": rag_focus_summary(detail_rows, retrieval),
        "notes": report.notes,
    }
    result_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nRAG 层结果：{rel_or_abs(result_file)}")
    print(f"模型层同款报告：{rel_or_abs(report_file)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
