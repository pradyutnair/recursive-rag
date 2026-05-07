#!/usr/bin/env python
"""Evaluate frozen HERA on a test set. Loads library + prompts, runs orchestrator + executor."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hera.agents import load_prompts, load_seed_prompts
from hera.config import HERAConfig, load_env
from hera.data import (
    QAExample,
    load_bamboogle_125,
    load_test_json,
)
from hera.library import ExperienceLibrary, profile_question
from hera.lm import OpenAIClient, VLLMClient
from hera.metric import accuracy, contain, exact_match, f1_score
from hera.orchestrator import Orchestrator
from hera.retriever import RetrieverClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hera.eval")


_TEST_DIR = os.getenv("HERA_TEST_DIR", "/local/yzheng/pnair/data")
if os.path.exists(os.path.join(_TEST_DIR, "test_1000")):
    # Snellius layout: hera/data/test_1000/{ds}.json
    _T = lambda ds: os.path.join(_TEST_DIR, "test_1000", f"{ds}.json")
    TEST_PATHS = {
        "musique": _T("musique"),
        "2wikimultihop": _T("2wikimultihop"),
        "hotpotqa": _T("hotpotqa"),
    }
else:
    # node409 layout
    TEST_PATHS = {
        "musique": "/local/yzheng/pnair/data/musique/questions_1000_seedfull_combined.json",
        "2wikimultihop": "/local/yzheng/pnair/data/2wikimultihop/questions_1000_seed42.json",
        "hotpotqa": "/local/yzheng/pnair/data/hotpotqa/questions_1000_seed42.json",
    }


def load_test(name: str, limit: int = 0) -> list[QAExample]:
    if name == "bamboogle":
        ex = load_bamboogle_125()
    elif name in TEST_PATHS:
        ex = load_test_json(TEST_PATHS[name], source=name)
    else:
        raise ValueError(name)
    if limit and limit < len(ex):
        ex = ex[:limit]
    return ex


async def eval_one(orchestrator: Orchestrator, ex: QAExample, temperature: float
                    ) -> tuple[QAExample, dict]:
    profile = profile_question(ex.question)
    topo, ids, orch_res = await orchestrator.sample_topology(ex.question, profile, temperature=temperature)
    traj = await orchestrator.execute(ex.question, ex.answer, topo, ids,
                                        orch_tokens=orch_res.prompt_tokens + orch_res.completion_tokens)
    return ex, {
        "id": ex.id,
        "source": ex.source,
        "question": ex.question,
        "gold": ex.answer,
        "pred": traj.answer,
        "em": traj.em,
        "f1": traj.f1,
        "contain": traj.contain,
        "acc": traj.acc,
        "tokens": traj.total_tokens,
        "elapsed_s": traj.elapsed_s,
        "topology": [s["agent"] for s in traj.topology["execution_order"]],
        "profile": profile,
        "used_insight_ids": ids,
    }


async def run_async(args: argparse.Namespace) -> None:
    load_env()
    cfg = HERAConfig()
    examples = load_test(args.dataset, limit=args.limit)
    logger.info("Eval %s: %d questions", args.dataset, len(examples))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_{args.dataset}.jsonl"
    summary_path = out_dir / f"summary_{args.dataset}.json"

    library = ExperienceLibrary.load(args.library) if args.library else ExperienceLibrary()
    prompts = load_prompts(args.prompts) if args.prompts else load_seed_prompts()
    logger.info("Library entries: %d, prompts: %d", len(library.entries), len(prompts))

    vllm = VLLMClient(
        endpoints=list(cfg.resolved_endpoints()),
        model=cfg.vllm_model,
        max_tokens=cfg.vllm_max_tokens,
        temperature=cfg.vllm_temperature,
        concurrency=args.vllm_concurrency,
    )
    openai_client = OpenAIClient(
        model=cfg.openai_model,
        max_tokens=cfg.openai_max_tokens,
        temperature=cfg.openai_temperature,
        concurrency=args.openai_concurrency,
    )
    retriever = RetrieverClient(
        url=cfg.resolved_retriever_url(),
        topk=cfg.retriever_topk,
        concurrency=cfg.retriever_concurrency,
    )
    orchestrator = Orchestrator(
        vllm=vllm,
        openai_client=openai_client,
        retriever=retriever,
        library=library,
        prompts=prompts,
        retriever_topk=cfg.retriever_topk,
        library_top_k=cfg.library_top_k_retrieve,
    )

    if args.wandb:
        try:
            import wandb
            wandb.init(project=cfg.wandb_project, name=args.run_name or f"eval_{args.dataset}",
                       config={"dataset": args.dataset, "limit": args.limit, "temperature": args.temperature},
                       tags=["hera", "eval", args.dataset])
        except Exception as e:
            logger.warning("wandb init failed: %s", e)
            args.wandb = False

    sem = asyncio.Semaphore(args.parallel)
    pred_f = open(pred_path, "w")

    metrics_acc = defaultdict(float)
    n = 0

    async def _runner(ex: QAExample):
        async with sem:
            return await eval_one(orchestrator, ex, args.temperature)

    tasks = [_runner(ex) for ex in examples]
    t0 = time.time()
    for fut in asyncio.as_completed(tasks):
        ex, rec = await fut
        pred_f.write(json.dumps(rec) + "\n")
        pred_f.flush()
        metrics_acc["em"] += rec["em"]
        metrics_acc["f1"] += rec["f1"]
        metrics_acc["contain"] += rec["contain"]
        metrics_acc["acc"] += rec["acc"]
        metrics_acc["tokens"] += rec["tokens"]
        n += 1
        if args.wandb:
            try:
                import wandb
                wandb.log({
                    "eval/running_em": metrics_acc["em"] / n,
                    "eval/running_f1": metrics_acc["f1"] / n,
                    "eval/running_acc": metrics_acc["acc"] / n,
                    "eval/running_tokens": metrics_acc["tokens"] / n,
                    "eval/n": n,
                })
            except Exception:
                pass
        if n % 10 == 0:
            logger.info("[%d/%d] em=%.3f f1=%.3f acc=%.3f tok=%.0f elapsed=%.0fs",
                        n, len(examples),
                        metrics_acc["em"] / n, metrics_acc["f1"] / n,
                        metrics_acc["acc"] / n, metrics_acc["tokens"] / n,
                        time.time() - t0)

    pred_f.close()
    summary = {
        "dataset": args.dataset,
        "n": n,
        "em": metrics_acc["em"] / max(1, n),
        "f1": metrics_acc["f1"] / max(1, n),
        "contain": metrics_acc["contain"] / max(1, n),
        "acc": metrics_acc["acc"] / max(1, n),
        "tokens": metrics_acc["tokens"] / max(1, n),
        "elapsed_s": time.time() - t0,
        "library": str(args.library) if args.library else None,
        "prompts": str(args.prompts) if args.prompts else None,
    }
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    logger.info("Final: %s", summary)
    print(json.dumps(summary, indent=2))

    if args.wandb:
        try:
            import wandb
            wandb.log({f"eval/final_{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
            wandb.finish()
        except Exception:
            pass

    await vllm.aclose()
    await openai_client.aclose()
    await retriever.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["musique", "2wikimultihop", "hotpotqa", "bamboogle"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--library", type=str, default=None)
    ap.add_argument("--prompts", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--vllm-concurrency", type=int, default=12)
    ap.add_argument("--openai-concurrency", type=int, default=16)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
