#!/usr/bin/env python
"""RAG 层评测入口。

用法：
    python rag_eval/run_eval.py                                          # 真实跑（默认全部可用模型）
    python rag_eval/run_eval.py --models deepseek-chat --tag rag-v1
    python rag_eval/run_eval.py --models mock-baseline --no-embedding --tag ci-rag   # CI 冒烟，零成本
    python rag_eval/run_eval.py --retrieval-only                         # 只测检索层 recall@k，不调模型

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
RESULTS_DIR = ROOT / "rag_eval" / "results"
ARCHIVE_DIR = RESULTS_DIR / "archive"
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
        "--retrieval-only",
        action="store_true",
        help="只跑检索层 recall@k，不调用任何模型（零成本，CI 可跑）",
    )
    return parser.parse_args(argv)


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

    cases = load_datasets(Path(config.paths.datasets_dir), [DATASET_NAME])
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        print(f"没有加载到 {DATASET_NAME} 用例，请检查 datasets/{DATASET_NAME}.jsonl")
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
        f"检索层 recall@{top_k} = {retrieval['recall_at_k']:.1%}"
        f"（可判 {retrieval['scored']} 条，命中 {retrieval['hit']} 条）；"
        "答错时可据此区分「没检索到」与「检索到了没用好」。",
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
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
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
        "cases": case_rows(report, cases, rag_clients),
        "notes": report.notes,
    }
    result_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nRAG 层结果：{rel_or_abs(result_file)}")
    print(f"模型层同款报告：{rel_or_abs(report_file)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
