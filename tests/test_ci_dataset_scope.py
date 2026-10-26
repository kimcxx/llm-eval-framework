"""CI 数据集范围的守护测试（Q6 裁决①：模型层自检不含 ``rag_cases``）。

Q6 裁决时补的一条要求：**口径不能只写在 yml 注释里，要写成会自动响的护栏**。
以后谁往 ``datasets/`` 加了新 jsonl 却没显式登记进 CI 命令，这里立刻变红，
不靠人发现——呼应 Q5 的教训（护栏要能自动响，不是写在文档里等人读）。

解析用纯文本正则，不引 pyyaml：项目零依赖优先，yml 里就一行命令行，
没必要为一个正则拉一个依赖。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
DATASETS_DIR = ROOT / "datasets"

# 显式豁免（理由写清）：RAG 层用例走自己的入口 rag_eval/run_eval.py。
# 裸模型（不接检索管道）做 RAG 题，成绩既不代表模型能力也不代表 RAG 能力，
# 只会污染模型层通过率——所以模型层自检把它排除在外。
EXEMPT = {"rag_cases"}


def _ci_datasets() -> dict[str, set[str]]:
    """从各 workflow 里抽出 ``--datasets`` 后面跟的数据集清单。"""
    out: dict[str, set[str]] = {}
    for yml in sorted(WORKFLOWS.glob("*.yml")):
        text = yml.read_text(encoding="utf-8")
        found: set[str] = set()
        for m in re.finditer(r"--datasets\s+([A-Za-z0-9_ .]+)", text):
            found.update(tok for tok in m.group(1).split())
        out[yml.name] = found
    return out


def test_rag_cases_excluded_from_model_layer() -> None:
    """模型层自检不该跑 rag_cases（Q6 裁决①）。"""
    ci = _ci_datasets()
    assert ci, "没从任何 workflow 里解析到 --datasets（正则失效还是命令改了？）"
    for name, ds in ci.items():
        assert "rag_cases" not in ds, f"{name} 的模型层自检不该跑 rag_cases（Q6 裁决①）"


def test_every_dataset_declared_or_exempt() -> None:
    """新数据集落进 datasets/ 就必须显式登记，否则 CI 会静默不带它跑。"""
    ci = _ci_datasets()
    declared: set[str] = set()
    for ds in ci.values():
        declared |= ds
    on_disk = {p.stem for p in DATASETS_DIR.glob("*.jsonl")}
    undeclared = on_disk - declared - EXEMPT
    assert not undeclared, f"这些数据集没登记进任何 CI 命令：{sorted(undeclared)}"


def test_declared_datasets_exist_on_disk() -> None:
    """反过来也钉住：CI 写了不存在的数据集名会静默跑空，不报错也不出数。"""
    on_disk = {p.stem for p in DATASETS_DIR.glob("*.jsonl")}
    for name, ds in _ci_datasets().items():
        missing = ds - on_disk
        assert not missing, f"{name} 声明了磁盘上不存在的数据集：{sorted(missing)}"
