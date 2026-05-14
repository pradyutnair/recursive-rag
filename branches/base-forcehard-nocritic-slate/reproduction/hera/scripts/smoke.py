#!/usr/bin/env python
"""Smoke test: 1 query through orchestrator + executor with seed prompts."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hera.agents import load_seed_prompts
from hera.config import HERAConfig, load_env
from hera.library import ExperienceLibrary, profile_question
from hera.lm import OpenAIClient, VLLMClient
from hera.orchestrator import Orchestrator
from hera.retriever import RetrieverClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


async def main():
    load_env()
    cfg = HERAConfig()
    vllm = VLLMClient(endpoints=list(cfg.resolved_endpoints()), model=cfg.vllm_model,
                       max_tokens=600, concurrency=4)
    openai_client = OpenAIClient(model=cfg.openai_model, max_tokens=512, concurrency=8)
    retriever = RetrieverClient(cfg.resolved_retriever_url(), topk=5, concurrency=4)
    library = ExperienceLibrary()
    prompts = load_seed_prompts()
    orch = Orchestrator(vllm, openai_client, retriever, library, prompts)

    q = "Who is the spouse of the actor who played the title role in the movie Forrest Gump?"
    gold = "Rita Wilson"
    profile = profile_question(q)
    print(f"Query: {q}")
    print(f"Profile: {profile}")
    topo, ids, res = await orch.sample_topology(q, profile, temperature=0.7)
    print(f"Topology: {json.dumps([s for s in topo['execution_order']], indent=2)}")
    traj = await orch.execute(q, gold, topo, ids, orch_tokens=res.prompt_tokens + res.completion_tokens)
    print(f"\nFinal answer: {traj.answer!r}")
    print(f"Gold: {gold}")
    print(f"F1: {traj.f1:.3f} EM: {traj.em} Tokens: {traj.total_tokens}")
    for inv in traj.invocations:
        print(f"  - {inv.name}: out={json.dumps(inv.output)[:150]} err={inv.error}")
    await vllm.aclose()
    await openai_client.aclose()
    await retriever.aclose()


if __name__ == "__main__":
    asyncio.run(main())
