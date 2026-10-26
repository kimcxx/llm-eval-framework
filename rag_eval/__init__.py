"""RAG 评测层。

与模型层、Agent 层并列的第三层：

- **模型层**考「模型答得对不对」；
- **Agent 层**考「agent 有没有调对工具、能不能多步走通」；
- **RAG 层**考「给定资料后，检索能不能找对、生成能不能忠于资料、没有资料时会不会编」。

被测对象是项目内自建的 mini-RAG **固定管道**（无条件检索 top-k 再生成），
不做「要不要检索」的决策、不产生多步轨迹，所以它不是 Agent 层的子集。

判分与报告全部复用 `src/`（用例集 → runner → registry 判分 → 报告），
框架骨架不改：这里只提供「被测对象」与「入口」。
"""

from __future__ import annotations

__all__ = ["knowledge", "pipeline", "retrieval", "retriever"]
