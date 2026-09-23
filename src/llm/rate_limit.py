"""全局请求节流。

背景：模型供应商普遍按 RPM/QPS 限流，而评测是「短时间打一大批发请求」的典型场景，
并发开到 workers=N 时很容易触发 429。退避重试只能事后补救，更好的做法是**事前限速**。

实现取舍：用「最小间隔 + 单调时钟」而不是令牌桶，因为配置项只需要一个
`request_interval_s`（60 / RPM 即可换算），语义直观、无额外状态；
等待放在锁外，避免持锁 sleep 把并发完全串行化。
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """保证任意两次 acquire 之间至少间隔 min_interval_s 秒（进程内全局生效）。"""

    def __init__(self, min_interval_s: float = 0.0) -> None:
        self.min_interval_s = max(0.0, float(min_interval_s or 0.0))
        self._lock = threading.Lock()
        self._next_free = 0.0

    @property
    def enabled(self) -> bool:
        return self.min_interval_s > 0

    def acquire(self) -> float:
        """阻塞到允许发起下一次调用为止，返回实际等待秒数。"""
        if not self.enabled:
            return 0.0

        with self._lock:
            now = time.monotonic()
            wait = self._next_free - now
            # 用 max(now, _next_free) 而不是 now：前一轮排队过长时不积压负债
            self._next_free = max(now, self._next_free) + self.min_interval_s

        if wait > 0:
            time.sleep(wait)
            return wait
        return 0.0
