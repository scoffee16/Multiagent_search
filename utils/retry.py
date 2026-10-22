"""
工具调用重试与超时工具模块。

提供指数退避重试装饰器，用于包装对外部 API 的调用（如 Tavily 搜索、数据库查询），
在网络抖动或临时故障时自动重试，避免单次失败导致整个 Agent 任务中断。

使用示例:
    from utils.retry import retry_with_backoff

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def call_external_api():
        ...
"""

import time
import functools
import threading
from typing import Type, Tuple, Callable, Any


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
):
    """
    指数退避重试装饰器。

    Args:
        max_retries: 最大重试次数（不含首次调用）
        base_delay: 首次重试等待秒数
        max_delay: 重试等待上限秒数
        backoff_factor: 退避倍数（每次重试等待 = 上次等待 × backoff_factor）
        exceptions: 触发重试的异常类型元组

    Returns:
        装饰后的函数

    重试时间线示例（base_delay=1, factor=2, max=30）:
        首次失败 → 等 1s → 重试1 → 等 2s → 重试2 → 等 4s → 重试3 → 仍失败则抛出
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # 打印重试信息（生产环境可改为 logging.warning）
                        print(
                            f"[Retry] {func.__name__} 调用失败 "
                            f"(第 {attempt + 1}/{max_retries + 1} 次): {e}. "
                            f"{delay:.1f}s 后重试..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        print(
                            f"[Retry] {func.__name__} 重试 {max_retries} 次后仍失败: "
                            f"{last_exception}"
                        )

            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


# ── 线程安全版本（用于可能在不同线程调用的工具） ──

class RetryState(threading.local):
    """线程本地重试计数器，避免多线程并发时互相干扰。"""
    def __init__(self):
        self.attempt = 0


_retry_state = RetryState()
