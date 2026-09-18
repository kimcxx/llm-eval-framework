"""SimilarityMetric._ensure_backend 的并发回归测试。

背景：
    src/metrics/similarity.py 中 SimilarityMetric._ensure_backend() 是惰性初始化后端的方法。
    原实现是无锁的 check-then-act，被 src/runner/runner.py 的 ThreadPoolExecutor 并发调用时，
    多个线程会同时通过 ``if self._backend is not None: return`` 检查，导致初始化逻辑
    （含降级警告打印、向量模型加载）被执行多次。

测试策略：
    用 ``monkeypatch`` 向 ``sys.modules`` 注入假的 ``sentence_transformers`` 模块，
    让父类 ``prefer_embedding=True`` 的初始化路径能真实执行（developer 加的双检锁会在
    假 ``SentenceTransformer.__init__`` 的 ``time.sleep(0.05)`` 期间暴露竞态窗口）。
    假类的 ``__init__`` 内部用线程安全的计数器记录实例化次数。

验证要点：
    - 修复前（无锁）：8 线程都通过 fast-pass check、都进入 ``try`` 块、都实例化 SentenceTransformer，
      实例化计数 == 8，测试失败 —— 证明 bug 真实存在；
    - 修复后（双检锁）：只有第一个线程进锁后实例化（计数 == 1），其余线程在锁内第二检 return，
      测试通过。

不修改 ``src/`` 下的任何源码。
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

from src.metrics.similarity import SimilarityMetric


# ============================== 假 sentence_transformers ============================== #


def _install_fake_sentence_transformers(monkeypatch, sleep_seconds: float = 0.05) -> dict:
    """向 ``sys.modules`` 注入假的 ``sentence_transformers`` 模块。

    假 ``SentenceTransformer.__init__`` 包含 ``time.sleep(sleep_seconds)`` 模拟真实加载耗时，
    用线程安全的 ``threading.Lock`` 保护计数器记录实例化次数。

    Args:
        monkeypatch: pytest 提供的 monkeypatch fixture。
        sleep_seconds: 假 ``__init__`` 中 sleep 的秒数，用于放大竞态窗口。

    Returns:
        一个 dict，包含：
            - ``count``: ``int``，实例化计数器（线程安全自增）；
            - ``sleep_seconds``: ``float``，注入的 sleep 秒数。
    """
    state: dict = {"count": 0, "sleep_seconds": sleep_seconds, "lock": threading.Lock()}

    class FakeSentenceTransformer:
        """模拟真实 SentenceTransformer 的加载耗时（sleep + 计数）。"""

        def __init__(self, model_name: str) -> None:
            # 模拟真实 SentenceTransformer 的模型加载耗时，让多个线程有足够时间
            # 在锁内 / 锁外进入初始化路径。
            time.sleep(state["sleep_seconds"])
            with state["lock"]:
                state["count"] += 1
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings: bool = False) -> list[list[float]]:
            """最小占位实现：返回固定假向量，让父类 _embedding_score 能完成计算。
            测试只关心 _ensure_backend 的初始化次数，不关心向量值是否真实。"""
            n = len(texts)
            return [[0.1, 0.2, 0.3, 0.4] for _ in range(n)]

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    # 伪造 numpy：compute() 的 embedding 打分路径会 ``import numpy as np``，
    # 而 numpy 与 sentence-transformers 一样是可选依赖（CI 不安装）。
    # 这里只实现 ``np.dot``（本测试唯一用到的函数），避免测试对未声明依赖产生隐性依赖。
    fake_numpy = ModuleType("numpy")

    def _dot(a, b) -> float:
        return sum(float(x) * float(y) for x, y in zip(list(a), list(b)))

    fake_numpy.dot = _dot
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    return state


# ============================== 并发回归测试 ============================== #


class TestSimilarityConcurrency:
    """SimilarityMetric._ensure_backend 的并发回归测试。"""

    def test_concurrent_backend_init_instantiates_encoder_once(
        self, monkeypatch
    ) -> None:
        """8 线程并发访问 ``backend`` 属性时，向量编码器应只实例化 1 次。

        修复前（无锁）：8 线程都通过 fast-pass check、都进入 ``try`` 块、
        都执行 ``self._encoder = SentenceTransformer(self.model_name)``，
        ``state["count"] == 8``，测试失败 —— 证明 bug 真实存在。

        修复后（双检锁）：只有第一个线程进锁后完成实例化（计数 == 1），
        其余线程在锁内第二检 ``self._backend is not None`` 后直接 return，
        测试通过。
        """
        state = _install_fake_sentence_transformers(monkeypatch)

        metric = SimilarityMetric(prefer_embedding=True)

        def call_backend() -> str:
            return metric.backend

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: call_backend(), range(8)))

        assert state["count"] == 1, (
            f"SentenceTransformer 应只实例化 1 次，实际 {state['count']} 次。"
            "这表明并发场景下 _ensure_backend 的 check-then-act 出现了竞态，"
            "初始化逻辑（含向量模型加载）被多次执行，应使用 double-checked locking 修复。"
        )
        assert metric._backend == "embedding", (
            f"修复后期望 _backend == 'embedding'，实际 {metric._backend!r}。"
        )

    def test_concurrent_compute_init_instantiates_encoder_once(
        self, monkeypatch, make_case, make_response
    ) -> None:
        """8 线程并发调用 ``compute()`` 时，向量编码器应只实例化 1 次。

        这是 ``test_concurrent_backend_init_instantiates_encoder_once`` 的补充：
        ``compute()`` 是 ``runner.py`` 中 ``ThreadPoolExecutor`` 实际调用的入口，
        验证在真实调用链路上同样只发生一次初始化。
        """
        state = _install_fake_sentence_transformers(monkeypatch)

        metric = SimilarityMetric(prefer_embedding=True)
        case = make_case(expected="北京", metrics=["similarity"])
        response = make_response("北京是中国的首都")

        def call_compute() -> object:
            return metric.compute(case, response)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: call_compute(), range(8)))

        assert state["count"] == 1, (
            f"通过 compute() 触发时 SentenceTransformer 应只实例化 1 次，"
            f"实际 {state['count']} 次。"
        )
        assert metric._backend == "embedding"

    def test_serial_backend_init_instantiates_encoder_once(
        self, monkeypatch
    ) -> None:
        """串行调用 8 次 ``backend`` 时，向量编码器应只实例化 1 次（基线对照）。

        这个用例不是回归测试，而是提供"非并发场景"的对照：
        多次串行调用时，fast-pass 应当保证初始化只发生一次。如果此用例失败，
        说明 fast-path 本身有 bug，与并发无关。
        """
        state = _install_fake_sentence_transformers(monkeypatch)

        metric = SimilarityMetric(prefer_embedding=True)

        for _ in range(8):
            metric.backend

        assert state["count"] == 1, (
            f"串行调用时 SentenceTransformer 应只实例化 1 次，实际 {state['count']} 次。"
            "如果此用例失败，说明 fast-path 本身有 bug，与并发无关。"
        )
        assert metric._backend == "embedding"

    def test_concurrent_backend_init_under_low_contention(
        self, monkeypatch
    ) -> None:
        """4 线程并发（低竞争场景）：向量编码器应只实例化 1 次。

        作为高竞争场景（8 线程）的补充，验证双检锁在低竞争下也只初始化一次。
        """
        state = _install_fake_sentence_transformers(monkeypatch, sleep_seconds=0.01)

        metric = SimilarityMetric(prefer_embedding=True)

        def call_backend() -> str:
            return metric.backend

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: call_backend(), range(4)))

        assert state["count"] == 1
        assert metric._backend == "embedding"