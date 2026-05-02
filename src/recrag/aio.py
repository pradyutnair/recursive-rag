"""Shared asyncio helpers backed by per-thread thread pools.

The trick: each Python thread gets its OWN ThreadPoolExecutor (cached in a
threading.local). This avoids the "cannot schedule new futures after shutdown"
issue that bites a single shared global executor when interleaved with
dspy.GEPA's worker threads (some of which trigger early shutdown of the
process-global executor).
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_local = threading.local()


def _get_executor() -> ThreadPoolExecutor:
    ex = getattr(_local, "executor", None)
    if ex is None:
        # _shutdown_called isn't always public; check by attempting a noop.
        ex = ThreadPoolExecutor(max_workers=16, thread_name_prefix=f"recrag-{threading.get_ident()}")
        _local.executor = ex
        return ex
    # Detect a broken/shutdown executor and rebuild
    if getattr(ex, "_shutdown", False) or getattr(ex, "_broken", False):
        ex = ThreadPoolExecutor(max_workers=16, thread_name_prefix=f"recrag-{threading.get_ident()}")
        _local.executor = ex
    return ex


async def to_thread(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Drop-in replacement for asyncio.to_thread using a per-thread executor."""
    loop = asyncio.get_event_loop()
    if kwargs:
        def _call() -> T:
            return fn(*args, **kwargs)
        return await loop.run_in_executor(_get_executor(), _call)
    return await loop.run_in_executor(_get_executor(), fn, *args)


def get_executor() -> ThreadPoolExecutor:
    return _get_executor()
