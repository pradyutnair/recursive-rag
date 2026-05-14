"""Dense retriever client for the wiki18 server at node408:8003."""
from __future__ import annotations

from typing import Any

import httpx


class DenseRetriever:
    def __init__(self, base_url: str = "http://node408:8003", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    async def retrieve_batch(self, client: httpx.AsyncClient, queries: list[str], topk: int = 6) -> list[list[dict[str, Any]]]:
        if not queries:
            return []
        url = f"{self.base_url}/retrieve"
        payload = {"queries": queries, "topk": topk, "mode": "text"}
        resp = await client.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        out: list[list[dict[str, Any]]] = []
        for batch in data.get("results", []):
            out.append([
                {
                    "chunk_id": str(h.get("chunk_id", "")),
                    "text": str(h.get("text", "")),
                    "score": float(h.get("score", 0.0)),
                }
                for h in batch
            ])
        return out

    async def retrieve(self, client: httpx.AsyncClient, query: str, topk: int = 6) -> list[dict[str, Any]]:
        results = await self.retrieve_batch(client, [query], topk=topk)
        return results[0] if results else []


def format_passages(passages: list[dict[str, Any]]) -> str:
    if not passages:
        return "(no passages retrieved)"
    parts = []
    for i, p in enumerate(passages, 1):
        text = p.get("text", "").strip()
        parts.append(f"[Passage {i}] {text}")
    return "\n\n".join(parts)
