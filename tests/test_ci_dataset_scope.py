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

# RAG 层用例已挪出 datasets/（现在在 rag_eval/datasets/rag_cases.jsonl），
# 所以「显式豁免」撤掉了——这一撤反而更强：谁再把 rag_cases 放回 datasets/，
# 下面「未登记就红」那条会立刻抓到它。
EXEMPT: set[str] = set()

# 造最小可用数据集用的样例行（resolve 阶段只看文件名，但保持 JSON 合法）
SAMPLE_CASE = (
    '{"id": "x-1", "category": "qa_zh", "metrics": ["contains"], '
    '"prompt": "p", "expected": "e", "keywords": ["e"]}\n'
)
RAG_CASE = (
    '{"id": "rag-001", "category": "rag_cases", "metrics": ["contains"], '
    '"prompt": "p", "expected": "e", "keywords": ["e"]}\n'
)


def _load_model_entry():
    """加载模型层入口 scripts/run_eval.py（不在包里，按文件路径加载）。

    只用到它的常量与 resolve_model_datasets，模块级没有副作用，导入即安全。
    """
    import importlib.util

    path = ROOT / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("model_run_eval", path)
    assert spec and spec.loader, "加载 scripts/run_eval.py 失败"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_dataset_dir(tmp_path, files: dict) -> Path:
    d = tmp_path / "datasets"
    d.mkdir()
    for name, content in files.items():
        (d / name).write_text(content, encoding="utf-8")
    return d


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


def test_real_datasets_dir_default_scan_has_no_rag_cases() -> None:
    """真实仓库默认扫描（不带 --datasets）的结果里不许再出现 rag_cases。

    2026-10-27 那次污染：rag_cases 放在 datasets/ 下，手跑一次不带 --datasets 的
    全量就被扫进去，混出 111 条的报告。根治是把它挪出 datasets/，这里守住结果。
    """
    mod = _load_model_entry()
    kept, dropped = mod.resolve_model_datasets(DATASETS_DIR, None)
    assert "rag_cases" not in kept, "RAG 用例又回到 datasets/ 目录下了"
    assert kept, "模型层数据集不该是空的"


def test_default_scan_excludes_rag_cases_and_warns(tmp_path, capsys) -> None:
    """兜底那条：即便 rag_cases 被放回扫描目录，默认扫描也会排除它并显式告警。

    模拟「有人又把它放回来」的情形——只靠调用方记得带 --datasets 是不行的。
    """
    mod = _load_model_entry()
    d = _make_dataset_dir(
        tmp_path, {"json_extract.jsonl": SAMPLE_CASE, "rag_cases.jsonl": RAG_CASE}
    )
    kept, dropped = mod.resolve_model_datasets(d, None)
    assert "rag_cases" not in kept
    assert dropped == ["rag_cases"]
    out = capsys.readouterr().out
    assert "rag_cases" in out and "已排除" in out, "排除必须显式告警，不许静默吞掉"


def test_only_rag_cases_requested_raises(tmp_path) -> None:
    """显式只指定 rag_cases 时要报错，不许跑空生成一份没人知道跑了什么的报告。"""
    import pytest

    mod = _load_model_entry()
    d = _make_dataset_dir(tmp_path, {"rag_cases.jsonl": RAG_CASE})
    with pytest.raises(Exception) as exc:
        mod.resolve_model_datasets(d, ["rag_cases"])
    assert "rag_cases" in str(exc.value)
