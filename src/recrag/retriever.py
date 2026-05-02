"""Thin client for the wiki18 retriever at node408:8003.

Uses synchronous httpx (httpx.Client) wrapped in a class-owned ThreadPoolExecutor
for the async API. The shared executor is never shut down, so the retriever
survives even when dspy.GEPA spawns thread-based concurrency that creates and
discards transient asyncio loops.
"""

from __future__ import annotations

from typing import Any

import httpx

from .aio import to_thread
from .contracts import RetrievedChunk

_FIXED_TOPK = 5


class Retriever:
    def __init__(self, base_url: str = "http://node408:8003", timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def _retrieve_batch_sync(self, queries: list[str], k: int) -> list[list[RetrievedChunk]]:
        if not queries:
            return []
        url = f"{self.base_url}/retrieve"
        payload: dict[str, Any] = {"queries": queries, "topk": k, "mode": "text"}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        out: list[list[RetrievedChunk]] = []
        for batch in data.get("results", []):
            out.append([
                RetrievedChunk(
                    chunk_id=str(h.get("chunk_id", "")),
                    text=str(h.get("text", "")),
                    score=float(h.get("score", 0.0)),
                ) for h in batch
            ])
        return out

    async def retrieve(self, query: str, k: int = _FIXED_TOPK) -> list[RetrievedChunk]:
        results = await self.retrieve_batch([query], k=k)
        return results[0] if results else []

    async def retrieve_batch(self, queries: list[str], k: int = _FIXED_TOPK) -> list[list[RetrievedChunk]]:
        return await to_thread(self._retrieve_batch_sync, list(queries), k)
