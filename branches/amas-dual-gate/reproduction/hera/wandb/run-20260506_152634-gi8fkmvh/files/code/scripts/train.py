#!/usr/bin/env python
"""HERA training: GRPO group rollouts + experience library + full RoPE."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hera.agents import AGENT_NAMES, AgentPrompt, load_seed_prompts, save_prompts
from hera.config import HERAConfig, load_env
from hera.data import load_qa_jsonl
from hera.grpo import grpo_step
from hera.library import ExperienceLibrary
from hera.lm import OpenAIClient, VLLMClient
from hera.orchestrator import Orchestrator
from hera.retriever import RetrieverClient
from hera.rope import FailureBuffer, add_traj_to_buffer, rope_update_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hera.train")


def init_wandb(cfg: HERAConfig, args: argparse.Namespace) -> "wandb.run | None":
    if not args.wandb:
        return None
    try:
        import wandb
        return wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=args.run_name,
            config={
                "group_size": args.group_size,
                "rollout_temperature": args.rollout_temperature,
                "rope_update_every": args.rope_update_every,
                "rope_variants": args.rope_variants,
                "epochs": args.epochs,
                "train_size": args.train_size,
                "model_orchestrator": cfg.vllm_model,
                "model_subagent": cfg.openai_model,
            },
            tags=["hera", "train"],
        )
    except Exception as e:
        logger.warning("wandb init failed: %s", e)
        return None


async def run_async(args: argparse.Namespace) -> None:
    load_env()
    cfg = HERAConfig()
    out_dir = Path(args.out_dir or cfg.exp_lib_dir / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = out_dir / "prompts.json"
    library_path = out_dir / "library.json"
    log_path = out_dir / "train_log.jsonl"

    train = load_qa_jsonl(args.train_path)
    if args.train_size and args.train_size < len(train):
        train = train[: args.train_size]
    logger.info("Training on %d examples", len(train))

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
        url=cfg.retriever_url,
        topk=cfg.retriever_topk,
        concurrency=cfg.retriever_concurrency,
    )

    prompts = load_seed_prompts()
    if args.resume and prompts_path.exists():
        from hera.agents import load_prompts as _lp
        prompts = _lp(prompts_path)
        logger.info("Resumed prompts from %s", prompts_path)
    library = ExperienceLibrary.load(library_path) if (args.resume and library_path.exists()) else ExperienceLibrary(max_entries=args.library_max)
    buffer = FailureBuffer(capacity=args.failure_buffer)

    orchestrator = Orchestrator(
        vllm=vllm,
        openai_client=openai_client,
        retriever=retriever,
        library=library,
        prompts=prompts,
        retriever_topk=cfg.retriever_topk,
        library_top_k=cfg.library_top_k_retrieve,
    )

    run = init_wandb(cfg, args)

    log_f = open(log_path, "a", buffering=1)
    step_lock = asyncio.Lock()
    state = {
        "step": 0, "rope_steps": 0,
        "cum_f1": 0.0, "cum_tokens": 0, "successes": 0, "total": 0,
    }

    async def process_one(ex, epoch_idx):
        t_start = time.time()
        try:
            gr = await grpo_step(
                orchestrator=orchestrator,
                vllm=vllm,
                library=library,
                query=ex.question,
                gold=ex.answer,
                group_size=args.group_size,
                temperature=args.rollout_temperature,
            )
        except Exception as e:
            logger.warning("grpo_step failed for q=%r: %s", ex.question[:60], str(e)[:200])
            return
        async with step_lock:
            best = gr.trajectories[0]
            for t in gr.trajectories:
                add_traj_to_buffer(buffer, t)
            state["cum_f1"] += best.f1
            state["cum_tokens"] += best.total_tokens
            state["total"] += 1
            if best.f1 > 0:
                state["successes"] += 1
            cur = state["step"]
            state["step"] += 1
            log_record = {
                "step": cur, "epoch": epoch_idx,
                "qid": ex.id, "source": ex.source, "profile": gr.profile,
                "best_f1": best.f1, "best_em": best.em, "best_tokens": best.total_tokens,
                "group_f1": [t.f1 for t in gr.trajectories],
                "group_tokens": [t.total_tokens for t in gr.trajectories],
                "library_size": len(library.entries),
                "insights_added": len(gr.insights),
                "elapsed_s": time.time() - t_start,
            }
            log_f.write(json.dumps(log_record) + "\n")
            if run is not None:
                run.log({
                    "train/best_f1": best.f1,
                    "train/best_em": best.em,
                    "train/best_tokens": best.total_tokens,
                    "train/group_mean_f1": sum(t.f1 for t in gr.trajectories) / len(gr.trajectories),
                    "train/group_max_f1": max(t.f1 for t in gr.trajectories),
                    "train/library_size": len(library.entries),
                    "train/insights_added": len(gr.insights),
                    "train/cumulative_acc": state["successes"] / max(1, state["total"]),
                    "train/cumulative_tokens": state["cum_tokens"] / max(1, state["total"]),
                    "train/step": cur,
                })
            if cur % 5 == 0:
                logger.info("step=%d acc=%.3f mean_tok=%.0f lib=%d rope=%d",
                            cur, state["successes"] / max(1, state["total"]),
                            state["cum_tokens"] / max(1, state["total"]),
                            len(library.entries), state["rope_steps"])

    async def rope_round():
        for agent_name in AGENT_NAMES:
            if buffer.size(agent_name) < 2:
                continue
            try:
                new_p = await rope_update_agent(
                    orchestrator, vllm, prompts, buffer, agent_name,
                    num_variants=args.rope_variants, max_failures=args.rope_max_failures,
                )
            except Exception as e:
                logger.warning("RoPE update failed for %s: %s", agent_name, str(e)[:200])
                continue
            if new_p:
                prompts[agent_name] = new_p
                state["rope_steps"] += 1
                logger.info("RoPE: updated %s (rules=%d, principles=%d)",
                            agent_name, len(new_p.operational_rules), len(new_p.behavioral_principles))
                if run is not None:
                    run.log({
                        "rope/agent_updated": agent_name,
                        "rope/op_rules": len(new_p.operational_rules),
                        "rope/principles": len(new_p.behavioral_principles),
                        "rope/total_updates": state["rope_steps"],
                    })
        save_prompts(prompts, prompts_path)
        library.save(library_path)

    batch_size = args.batch_size
    rope_every = args.rope_update_every
    for epoch in range(args.epochs):
        for i in range(0, len(train), batch_size):
            batch = train[i: i + batch_size]
            await asyncio.gather(*[process_one(ex, epoch) for ex in batch])
            # Trigger RoPE if we've crossed an update boundary.
            if state["step"] // rope_every > (state["step"] - len(batch)) // rope_every:
                await rope_round()
        save_prompts(prompts, prompts_path)
        library.save(library_path)

    save_prompts(prompts, prompts_path)
    library.save(library_path)
    log_f.close()
    if run is not None:
        run.finish()

    await vllm.aclose()
    await openai_client.aclose()
    await retriever.aclose()
    logger.info("Training complete. Results at %s", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--run-name", type=str, default=time.strftime("hera_train_%Y%m%d_%H%M%S"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train-size", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--rollout-temperature", type=float, default=0.9)
    ap.add_argument("--rope-update-every", type=int, default=20)
    ap.add_argument("--rope-variants", type=int, default=3)
    ap.add_argument("--rope-max-failures", type=int, default=4)
    ap.add_argument("--failure-buffer", type=int, default=8)
    ap.add_argument("--library-max", type=int, default=30)
    ap.add_argument("--vllm-concurrency", type=int, default=12)
    ap.add_argument("--openai-concurrency", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
