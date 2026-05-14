"""Five SPARC-RAG agents on top of the shared VLLM client + dense retriever.

Generators run at temperature 0.5 / top_p 1.0 / max_tokens 600 (paper App. A.4).
The Answer Evaluator runs deterministically (temperature=0).
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from .lm import VLLMClient
from .prompts import (
    ANSWER_EVALUATOR_PROMPT,
    ANSWER_GENERATOR_PROMPT,
    ANSWER_SELECTION_PROMPT,
    CONTEXT_MANAGER_UPDATE_PROMPT,
    CONTEXT_MERGING_PROMPT,
    MULTI_PATH_DISPATCH_PROMPT,
)
from .retriever import format_passages


GEN_TEMP = 0.5
GEN_TOP_P = 1.0
GEN_MAX_TOKENS = 600
EVAL_TEMP = 0.0
EVAL_MAX_TOKENS = 256
SEED = 42
STOP_TOKENS = ["[END]"]


_QUERY_ITEM_RE = re.compile(r"<item[^>]*>\s*<query>(.*?)</query>\s*</item>", re.DOTALL | re.IGNORECASE)
_ANY_QUERY_TAG_RE = re.compile(r"<query>(.*?)</query>", re.DOTALL | re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_DECISION_TAG_RE = re.compile(r"<decision>(.*?)</decision>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip(s: str) -> str:
    return (s or "").strip()


def _strip_end_marker(s: str) -> str:
    s = s or ""
    s = s.split("[END]", 1)[0]
    return s.strip()


def _strip_think(s: str) -> str:
    return _THINK_TAG_RE.sub("", s or "")


def _extract_queries(text: str, n: int) -> list[str]:
    body = _strip_think(text)
    matches = [_strip(m.group(1)) for m in _QUERY_ITEM_RE.finditer(body)]
    if not matches:
        matches = [_strip(m.group(1)) for m in _ANY_QUERY_TAG_RE.finditer(body)]
    matches = [m for m in matches if m]
    if not matches:
        # last-ditch: numbered lines
        lines = [l.strip(" -*0123456789.").strip() for l in body.splitlines()]
        matches = [l for l in lines if l and len(l.split()) >= 2]
    return matches[:n] if matches else []


def _extract_answer_from_generator(text: str) -> str:
    body = _strip_end_marker(_strip_think(text))
    # The Answer Generator prompt asks for the bare answer after "Answer:".
    # The model often complies and emits exactly that, so we just trim.
    body = body.strip()
    # If the model echoed "Answer:" or "<answer>...</answer>", strip those.
    m = _ANSWER_TAG_RE.search(body)
    if m:
        return _strip(m.group(1))
    if body.lower().startswith("answer:"):
        body = body[len("answer:"):].strip()
    return body.splitlines()[0].strip() if body else ""


def _extract_answer_tag(text: str) -> str:
    body = _strip_end_marker(text)
    m = _ANSWER_TAG_RE.search(body)
    if m:
        return _strip(m.group(1))
    return _strip(body)


def _extract_decision(text: str) -> str:
    body = _strip_end_marker(text).upper()
    m = _DECISION_TAG_RE.search(body)
    token = _strip(m.group(1)).upper() if m else body
    if "STOP" in token and "CONTINUE" not in token:
        return "STOP"
    return "CONTINUE"


def _normalize_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def _match_branch_by_answer(picked: str, branches: list[dict[str, Any]]) -> int:
    p = _normalize_for_match(picked)
    if not p:
        return 0
    for i, b in enumerate(branches):
        if _normalize_for_match(b.get("answer", "")) == p:
            return i
    for i, b in enumerate(branches):
        ba = _normalize_for_match(b.get("answer", ""))
        if ba and (p in ba or ba in p):
            return i
    return 0


async def query_rewriter(
    lm: VLLMClient,
    client: httpx.AsyncClient,
    *,
    question: str,
    current_query: str,
    memory: str,
    n: int,
    replica: int | None = None,
) -> tuple[list[str], str]:
    prompt = MULTI_PATH_DISPATCH_PROMPT.format(
        N=n,
        query=question,
        current_query=current_query,
        context=memory or "(empty)",
    )
    out = await lm.chat(
        client, [{"role": "user", "content": prompt}],
        temperature=GEN_TEMP, top_p=GEN_TOP_P, max_tokens=GEN_MAX_TOKENS,
        replica=replica, seed=SEED, stop=STOP_TOKENS,
    )
    qs = _extract_queries(out["text"], n)
    if not qs:
        qs = [current_query] * n
    if len(qs) < n:
        qs = (qs + [current_query] * n)[:n]
    return qs, out["text"]


async def memory_update(
    lm: VLLMClient,
    client: httpx.AsyncClient,
    *,
    question: str,
    memory: str,
    passages: list[dict[str, Any]],
    replica: int | None = None,
) -> tuple[str, str]:
    prompt = CONTEXT_MANAGER_UPDATE_PROMPT.format(
        query=question,
        note=memory or "(empty)",
        new_context=format_passages(passages),
    )
    out = await lm.chat(
        client, [{"role": "user", "content": prompt}],
        temperature=GEN_TEMP, top_p=GEN_TOP_P, max_tokens=GEN_MAX_TOKENS,
        replica=replica, seed=SEED, stop=STOP_TOKENS,
    )
    return _strip_end_marker(_strip_think(out["text"])).strip(), out["text"]


async def answer_generator(
    lm: VLLMClient,
    client: httpx.AsyncClient,
    *,
    question: str,
    memory: str,
    replica: int | None = None,
) -> tuple[str, str]:
    prompt = ANSWER_GENERATOR_PROMPT.format(
        note=memory or "(empty)",
        query=question,
    )
    out = await lm.chat(
        client, [{"role": "user", "content": prompt}],
        temperature=GEN_TEMP, top_p=GEN_TOP_P, max_tokens=128,
        replica=replica, seed=SEED, stop=STOP_TOKENS,
    )
    return _extract_answer_from_generator(out["text"]), out["text"]


async def answer_evaluator(
    lm: VLLMClient,
    client: httpx.AsyncClient,
    *,
    question: str,
    current_query: str,
    memory: str,
    answer: str,
    replica: int | None = None,
) -> tuple[str, str]:
    prompt = ANSWER_EVALUATOR_PROMPT.format(
        question=question,
        current_query=current_query,
        note=memory or "(empty)",
        answer=answer or "(empty)",
    )
    out = await lm.chat(
        client, [{"role": "user", "content": prompt}],
        temperature=EVAL_TEMP, top_p=GEN_TOP_P, max_tokens=EVAL_MAX_TOKENS,
        replica=replica, seed=SEED, stop=STOP_TOKENS,
    )
    return _extract_decision(out["text"]), out["text"]


async def select_best_answer(
    lm: VLLMClient,
    client: httpx.AsyncClient,
    *,
    question: str,
    branches: list[dict[str, Any]],
    replica: int | None = None,
) -> tuple[int, str, str]:
    """Returns (branch_index, picked_answer_text, raw_text)."""
    if len(branches) <= 1:
        return 0, branches[0]["answer"] if branches else "", ""
    blocks = []
    for i, b in enumerate(branches, 1):
        blocks.append(
            f"--- Candidate {i} ---\n"
            f"Sub-query: {b.get('query','')}\n"
            f"Reasoning note: {b.get('memory','')}\n"
            f"Answer: {b.get('answer','')}\n"
        )
    prompt = ANSWER_SELECTION_PROMPT.format(
        question=question,
        answer_blocks="\n".join(blocks),
    )
    out = await lm.chat(
        client, [{"role": "user", "content": prompt}],
        temperature=EVAL_TEMP, top_p=GEN_TOP_P, max_tokens=EVAL_MAX_TOKENS,
        replica=replica, seed=SEED, stop=STOP_TOKENS,
    )
    picked = _extract_answer_tag(out["text"])
    idx = _match_branch_by_answer(picked, branches)
    return idx, picked, out["text"]


async def context_merge(
    lm: VLLMClient,
    client: httpx.AsyncClient,
    *,
    question: str,
    notes: list[str],
    replica: int | None = None,
) -> tuple[str, str]:
    notes = [n for n in notes if n and n.strip()]
    if not notes:
        return "", ""
    if len(notes) == 1:
        return notes[0], ""
    reasoning_list = "\n\n".join(f"[Note {i+1}] {n}" for i, n in enumerate(notes))
    prompt = CONTEXT_MERGING_PROMPT.format(
        question=question,
        reasoning_list=reasoning_list,
    )
    out = await lm.chat(
        client, [{"role": "user", "content": prompt}],
        temperature=GEN_TEMP, top_p=GEN_TOP_P, max_tokens=GEN_MAX_TOKENS,
        replica=replica, seed=SEED, stop=STOP_TOKENS,
    )
    return _strip_end_marker(_strip_think(out["text"])).strip(), out["text"]
