"""知识库加载。

沿用 Agent 层 fixture 的原则：**评测时禁止读外部可变数据**。
知识库是一份冻结快照（`fixtures/knowledge_base.jsonl`），附 provenance 说明来源与
升级策略；升级 = 一次显式决策（改文档 + 重标金标 + 留痕），永不追新。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
KB_FILE = FIXTURES_DIR / "knowledge_base.jsonl"
KB_META_FILE = FIXTURES_DIR / "knowledge_base.meta.json"


class KnowledgeBaseError(ValueError):
    """知识库文件缺失或格式非法。"""


@dataclass(frozen=True)
class Document:
    """一段知识库文档。"""

    doc_id: str
    title: str
    text: str

    def render(self) -> str:
        """拼进 prompt 的上下文片段：带上 id，便于回答里出现「依据 kb-00x」时可核对。"""
        return f"[{self.doc_id}] {self.title}\n{self.text}"


def load_knowledge_base(path: Path | str | None = None) -> list[Document]:
    path = Path(path) if path is not None else KB_FILE
    if not path.exists():
        raise KnowledgeBaseError(f"知识库文件不存在: {path}")

    docs: list[Document] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(f"{path.name} 第 {lineno} 行不是合法 JSON: {exc}") from exc
        missing = [k for k in ("id", "title", "text") if not raw.get(k)]
        if missing:
            raise KnowledgeBaseError(f"{path.name} 第 {lineno} 行缺少字段 {missing}")
        doc_id = str(raw["id"])
        if doc_id in seen:
            raise KnowledgeBaseError(f"{path.name} 第 {lineno} 行文档 id 重复: {doc_id}")
        seen.add(doc_id)
        docs.append(Document(doc_id=doc_id, title=str(raw["title"]), text=str(raw["text"])))

    if not docs:
        raise KnowledgeBaseError(f"知识库为空: {path}")
    return docs


def load_provenance(path: Path | str | None = None) -> dict:
    """读出 provenance（来源、冻结日期、占位状态、升级策略）。"""
    path = Path(path) if path is not None else KB_META_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def check_provenance(docs: list[Document], meta: dict) -> str:
    """核对 provenance 里声明的文档条数与实际是否一致。

    返回空字符串表示一致；不一致时返回说明文本（由调用方写进报告 notes，
    不抛异常——知识库少了文档不该让整轮评测炸掉）。
    """
    declared = meta.get("doc_count")
    if declared is None:
        return ""
    if int(declared) != len(docs):
        return (
            f"知识库 provenance 声明 {declared} 段文档，实际加载 {len(docs)} 段，"
            f"请一并更新 {KB_META_FILE.name}"
        )
    return ""
