"""Algorithm 1 (SPARC-RAG inference) executed with explicit width W and depth D."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .agents import (
    answer_evaluator,
    answer_generator,
    context_merge,
    memory_update,
    query_rewriter,
    select_best_answer,
)
from .lm import VLLMClient
from .retriever import DenseRetriever


@dataclass
class SparcConfig:
    width: int = 2
    max_depth: int = 8
    topk: int = 6
    enable_thinking: bool = False
    record_traces: bool = False


async def _execute_branch(
    lm: VLLMClient,
    retriever: DenseRetriever,
    client: httpx.AsyncClient,
    *,
    question: str,
    sub_query: str,
    prior_memory: str,
    topk: int,
    replica: int | None,
) -> dict[str, Any]:
    passages = await retriever.retrieve(client, sub_query, topk=topk)
    new_memory, _ = await memory_update(
        lm, client,
        question=question,
        memory=prior_memory, passages=passages, replica=replica,
    )
    answer, _ = await answer_generator(
        lm, client, question=question, memory=new_memory, replica=replica,
    )
    decision, _ = await answer_evaluator(
        lm, client,
        question=question, current_query=sub_query,
        memory=new_memory, answer=answer, replica=replica,
    )
    return {
        "query": sub_query,
        "memory": new_memory,
        "answer": answer,
        "decision": decision,
        "n_passages": len(passages),
    }


async def run_sparc(
    lm: VLLMClient,
    retriever: DenseRetriever,
    client: httpx.AsyncClient,
    *,
    question: str,
    cfg: SparcConfig,
) -> dict[str, Any]:
    t0 = time.time()
    lm.reset_counters()
    memory = ""
    final_answer = ""
    rounds: list[dict[str, Any]] = []
    stop_reason = "depth_cap"
    current_query = question
    n_replicas = len(lm.cfg.base_urls)

    for t in range(1, cfg.max_depth + 1):
        if cfg.width == 1:
            sub_queries = [current_query]
        else:
            sub_queries, _ = await query_rewriter(
                lm, client,
                question=question, current_query=current_query, memory=memory,
                n=cfg.width, replica=t % n_replicas,
            )

        branch_tasks = [
            _execute_branch(
                lm, retriever, client,
                question=question,
                sub_query=q,
                prior_memory=memory,
                topk=cfg.topk,
                replica=k % n_replicas,
            )
            for k, q in enumerate(sub_queries)
        ]
        branches = await asyncio.gather(*branch_tasks)

        if cfg.width == 1:
            best_idx = 0
            picked_answer = branches[0]["answer"]
        else:
            best_idx, picked_answer, _ = await select_best_answer(
                lm, client, question=question, branches=branches,
                replica=t % n_replicas,
            )

        best = branches[best_idx]
        if cfg.width > 1:
            ordered = [best["memory"]] + [b["memory"] for i, b in enumerate(branches) if i != best_idx]
            merged, _ = await context_merge(
                lm, client, question=question, notes=ordered,
                replica=t % n_replicas,
            )
            memory = merged
        else:
            memory = best["memory"]

        round_log: dict[str, Any] = {
            "t": t,
            "sub_queries": sub_queries,
            "best_index": best_idx,
            "best_answer": best["answer"],
            "best_decision": best["decision"],
            "branches_meta": [
                {
                    "query": b["query"],
                    "answer": b["answer"],
                    "decision": b["decision"],
                    "n_passages": b["n_passages"],
                }
                for b in branches
            ],
            "memory_chars": len(memory),
        }
        if cfg.record_traces:
            round_log["branches"] = branches
            round_log["memory"] = memory
        rounds.append(round_log)
        # Algorithm 1 line 16: return a_{k*} (the selected branch's answer).
        final_answer = best["answer"] or picked_answer
        current_query = best["query"]

        if best["decision"] == "STOP":
            stop_reason = "evaluator_stop"
            break

    return {
        "answer": final_answer,
        "memory": memory,
        "rounds": rounds,
        "stop_reason": stop_reason,
        "n_rounds": len(rounds),
        "n_llm_calls": lm.calls,
        "prompt_tokens": lm.prompt_tokens,
        "completion_tokens": lm.completion_tokens,
        "total_tokens": lm.total_tokens,
        "wall_clock_s": round(time.time() - t0, 3),
        "config": {
            "width": cfg.width,
            "max_depth": cfg.max_depth,
            "topk": cfg.topk,
        },
    }
