"""OpenAI-compatible vLLM client with round-robin across replicas.

Servers:
  - http://localhost:8001/v1  (Qwen/Qwen3-14B)
  - http://localhost:8002/v1
  - http://localhost:8003/v1
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class LMConfig:
    base_urls: tuple[str, ...] = (
        "http://localhost:8001/v1",
        "http://localhost:8002/v1",
        "http://localhost:8003/v1",
    )
    model: str = "Qwen/Qwen3-14B"
    timeout: float = 600.0
    enable_thinking: bool = False
    api_key: str = "EMPTY"


class VLLMClient:
    """Async OpenAI chat-completion client over the three vLLM replicas."""

    def __init__(self, cfg: LMConfig | None = None) -> None:
        self.cfg = cfg or LMConfig()
        self._cycle = itertools.cycle(range(len(self.cfg.base_urls)))
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def calls(self) -> int:
        return self._calls

    def reset_counters(self) -> None:
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def _next_url(self, replica: int | None) -> str:
        if replica is not None:
            return self.cfg.base_urls[replica % len(self.cfg.base_urls)]
        return self.cfg.base_urls[next(self._cycle)]

    async def chat(
        self,
        client: httpx.AsyncClient,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.5,
        top_p: float = 1.0,
        max_tokens: int = 600,
        replica: int | None = None,
        seed: int | None = 42,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        base = self._next_url(replica)
        url = f"{base}/chat/completions"
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": {"enable_thinking": bool(self.cfg.enable_thinking)},
        }
        if seed is not None:
            body["seed"] = int(seed)
        if stop:
            body["stop"] = stop
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        resp = await client.post(url, json=body, headers=headers, timeout=self.cfg.timeout)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        self._calls += 1
        self._prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self._completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        text = ""
        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            text = ""
        return {
            "text": text,
            "raw": data,
            "usage": usage,
            "url": base,
        }
