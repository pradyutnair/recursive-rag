"""TF-GRPO over the AdaptiveProgram, growing a Profile-Insight-Utility experience library.

For each training question:
  - Roll out G times at different temperatures.
  - Score each with the composite Pareto reward (quality * exp(-tokens/T)^alpha).
  - If the group has both clear winners and losers, ask the reflection LM to extract a
    natural-language semantic advantage (per Training-Free GRPO).
  - Convert the advantage into Add/Modify/Delete/Keep ops on the structured experience
    library, conditioned on the question's profile.

Across a batch of questions we then ask the reflection LM to consolidate the proposed
library snapshots into one merged library (deduping and pruning).

The library is saved as both JSON (structured) and TXT (legacy, for backward-compat with
the previous ReactRagPipeline experience prepending).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("DSPY_CACHEDIR", str(Path.cwd() / ".dspy_cache"))

import dspy

from recrag.adaptive_pipeline import AdaptiveConfig, AdaptiveRecursivePipeline
from recrag.grpo.library import ExperienceLibrary
from recrag.grpo.signatures import ExtractGroupOps, OptimizeBatch, SummarizeAdaptiveRollout
from recrag.lm import make_lm
from recrag.metric import composite_reward, oracle_bonus
from recrag.oracle import OracleLookup
from recrag.profile import classify
from recrag.retriever import Retriever
from recrag.wandb_utils import artifact as wandb_artifact
from recrag.wandb_utils import init_wandb, log as wandb_log


def _loads_ops(text: str) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        return []
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []
    except Exception:
        # Best effort: extract first JSON array
        import re

        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, list) else []
        except Exception:
            return []


def _load_program(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return {}
    raw_prompts = obj.get("prompts")
    if isinstance(raw_prompts, dict):
        return {str(k): str(v) for k, v in raw_prompts.items()}
    prompts: dict[str, str] = {}
    for name, value in obj.items():
        if isinstance(value, str):
            prompts[str(name)] = value
        elif isinstance(value, dict):
            sig = value.get("signature")
            if isinstance(sig, dict) and isinstance(sig.get("instructions"), str):
                prompts[str(name)] = sig["instructions"]
    return prompts


def _load_oracle(args: argparse.Namespace) -> OracleLookup | None:
    if not args.oracle_naive_dir:
        return None
    base = Path(args.oracle_naive_dir)
    paths: list[Path] = []
    for ds in (args.oracle_datasets or "").split(","):
        ds = ds.strip()
        if not ds:
            continue
        for candidate in (base / f"{ds}_fresh_naive" / "predictions.jsonl",
                          base / f"naive_{ds}" / "predictions.jsonl",
                          base / f"naive_{ {'2wikimultihop': '2wiki'}.get(ds, ds) }" / "predictions.jsonl"):
            if candidate.exists():
                paths.append(candidate)
                break
    if not paths:
        return None
    return OracleLookup.from_paths(paths)


async def compile_grpo(args: argparse.Namespace) -> ExperienceLibrary:
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.n_train and args.n_train > 0:
        questions = questions[: args.n_train]
    random.Random(args.seed).shuffle(questions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "compile_trace.jsonl"
    lib = ExperienceLibrary.load(args.seed_library) if args.seed_library else ExperienceLibrary()
    lib.max_entries = args.library_cap
    reflection_lm = make_lm(args.reflection_lm, temperature=0.0, max_tokens=args.reflection_max_tokens)
    wandb_run = init_wandb(
        project=args.wandb_project,
        name=args.run_name or Path(args.out_json).stem,
        config=vars(args),
        enabled=not args.no_wandb,
        mode=args.wandb_mode or None,
    )
    oracle = _load_oracle(args)
    program_prompts = _load_program(args.program)
    if oracle:
        print(f"[oracle] loaded {len(oracle)} entries; stats={oracle.stats()}")

    with trace_path.open("w", encoding="utf-8") as trace_f:
        for epoch in range(args.epochs):
            for start in range(0, len(questions), args.batch_size):
                batch = questions[start : start + args.batch_size]
                proposal_texts: list[str] = []

                async def rollout_one(q: dict[str, Any], g: int) -> dict[str, Any]:
                    root_lm = make_lm(args.root_lm, replica_idx=g, temperature=args.temperature, max_tokens=args.root_max_tokens)
                    sub_lm = make_lm(args.sub_lm, replica_idx=g, temperature=0.0, max_tokens=args.sub_max_tokens)
                    pipeline = AdaptiveRecursivePipeline(
                        root_lm, sub_lm, Retriever(args.retriever_url),
                        AdaptiveConfig(
                            max_nodes=args.max_nodes,
                            max_recursion_depth=args.max_recursion,
                            tau_recurse=args.tau_recurse,
                            experience_library=None,  # use the in-memory library directly
                            router_instructions=program_prompts.get("router", AdaptiveConfig.router_instructions),
                            planner_instructions=program_prompts.get("planner", AdaptiveConfig.planner_instructions),
                            synth_instructions=program_prompts.get("synthesizer", AdaptiveConfig.synth_instructions),
                            critic_instructions=program_prompts.get("critic", AdaptiveConfig.critic_instructions),
                            budget_hint=args.budget_hint,
                            max_searches=args.max_searches,
                        ),
                    )
                    # Inject current library snapshot (profile-keyed)
                    pipeline.library = lib
                    pred = await pipeline.run(str(q.get("question", "")))
                    gold = str(q.get("answer", ""))
                    rb = composite_reward(
                        pred.get("answer", ""), gold, pred.get("metadata", {}).get("findings", []),
                        pred.get("metadata", {}).get("total_tokens", 0),
                        pred.get("metadata", {}).get("expected_type", "auto"),
                        token_T=args.token_T, alpha=args.alpha,
                    )
                    return {"q": q, "pred": pred, "rb": rb, "g": g}

                for q in batch:
                    profile = classify(str(q.get("question", "")))
                    qid = str(q.get("id", ""))
                    oracle_entry = oracle.get(qid) if oracle else None
                    oracle_easy = (oracle_entry.em == 1) if oracle_entry else None
                    naive_tokens = oracle_entry.tokens if oracle_entry else 0
                    difficulty_tag = (
                        "easy" if oracle_easy is True
                        else "hard" if oracle_easy is False
                        else "unknown"
                    )
                    rollouts = await asyncio.gather(*[rollout_one(q, g) for g in range(args.group_size)])
                    summaries: list[str] = []
                    for r in rollouts:
                        rb = r["rb"]
                        topology = r["pred"].get("metadata", {}).get("topology", "")
                        bonus, reason = (0.0, "")
                        if oracle_easy is not None:
                            bonus, reason = oracle_bonus(rb.em, topology, rb.tokens, oracle_easy=oracle_easy, naive_tokens=naive_tokens)
                        oracle_score = rb.composite + bonus
                        with dspy.context(lm=reflection_lm):
                            summ = dspy.Predict(SummarizeAdaptiveRollout)(
                                question=str(q.get("question", "")),
                                profile=profile,
                                gold_answer=str(q.get("answer", "")),
                                trajectory=r["pred"].get("readable_trace", ""),
                                reward_breakdown=json.dumps({**rb.as_dict(), "oracle_bonus": bonus, "oracle_reason": reason, "difficulty": difficulty_tag, "naive_tokens": naive_tokens}),
                            ).summary
                        summaries.append(f"[g={r['g']} oracle_score={oracle_score:.3f} em={rb.em} tokens={rb.tokens} topology={topology} difficulty={difficulty_tag}] {str(summ).strip()}")
                        trace_f.write(json.dumps({
                            "epoch": epoch, "id": qid, "dataset": q.get("dataset", ""), "g": r["g"], "profile": profile, "difficulty": difficulty_tag,
                            "reward": rb.as_dict(), "oracle_bonus": bonus, "oracle_reason": reason,
                            "trace": r["pred"].get("readable_trace", "")[:1500],
                        }, ensure_ascii=False) + "\n")
                        trace_f.flush()
                        # store augmented score into the rollout for group-spread check
                        r["oracle_score"] = oracle_score
                    rewards = [rr["oracle_score"] for rr in rollouts]
                    spread = max(rewards) - min(rewards)
                    wandb_log(wandb_run, {
                        "question_id": qid,
                        "profile": profile,
                        "difficulty": difficulty_tag,
                        "group_size": args.group_size,
                        "rewards": rewards,
                        "best_reward": max(rewards),
                        "worst_reward": min(rewards),
                        "spread": spread,
                        "has_mixed_outcomes": spread >= 0.2,
                    })
                    if spread < 0.2:
                        proposal_texts.append(lib.to_text())
                        continue
                    with dspy.context(lm=reflection_lm):
                        ops_raw = dspy.Predict(ExtractGroupOps)(
                            summaries="\n\n".join(summaries),
                            profile=f"{profile}|{difficulty_tag}",
                            current_library=lib.to_text(),
                        ).ops_json
                    ops = _loads_ops(str(ops_raw))
                    # Tag any ADDs lacking a profile with profile|difficulty
                    for op in ops:
                        if str(op.get("op", "")).upper() == "ADD" and not op.get("profile"):
                            op["profile"] = f"{profile}|{difficulty_tag}" if oracle_easy is not None else profile
                    lib.apply_ops(ops)
                    proposal_texts.append(lib.to_text())
                # Batch consolidation
                if proposal_texts:
                    with dspy.context(lm=reflection_lm):
                        merged = dspy.Predict(OptimizeBatch)(
                            batch_proposals="\n\n".join(proposal_texts),
                            current_library=lib.to_text(),
                        ).merged_library
                    lib.merge_text(str(merged))
                lib.max_entries = args.library_cap
                lib.prune()
                lib.save_text(args.out_txt)
                lib.save_json(args.out_json)
                ckpt = checkpoint_dir / f"epoch_{epoch}_batch_{start // args.batch_size}.json"
                lib.save_json(ckpt)
                wandb_log(wandb_run, {
                    "epoch": epoch,
                    "batch_idx": start // args.batch_size,
                    "library_size": len(lib.entries),
                    "ops_applied": len(proposal_texts),
                })
                wandb_artifact(wandb_run, ckpt, name=f"{Path(args.out_json).stem}-e{epoch}-b{start // args.batch_size}", type_="grpo-library")
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
    return lib


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="data/multidataset/train_v3.json")
    p.add_argument("--n-train", type=int, default=0, help="0 = use all rows")
    p.add_argument("--oracle-naive-dir", default="compiled/oracle")
    p.add_argument("--oracle-datasets", default="musique,2wikimultihop,hotpotqa")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--root-lm", default="qwen14b-nothink")
    p.add_argument("--sub-lm", default="qwen14b-nothink")
    p.add_argument("--root-max-tokens", type=int, default=768)
    p.add_argument("--sub-max-tokens", type=int, default=512)
    p.add_argument("--reflection-lm", default="openai/gpt-5")
    p.add_argument("--reflection-max-tokens", type=int, default=16000)
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--max-nodes", type=int, default=6)
    p.add_argument("--max-recursion", type=int, default=0)
    p.add_argument("--tau-recurse", type=float, default=0.5)
    p.add_argument("--max-searches", type=int, default=5)
    p.add_argument("--budget-hint", choices=["tight", "normal", "rich"], default="normal")
    p.add_argument("--token-T", type=float, default=8000.0)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--seed-library")
    p.add_argument("--program", help="JSON file with GEPA/compiled prompts")
    p.add_argument("--library-cap", type=int, default=30)
    p.add_argument("--checkpoint-dir", default="compiled/grpo_v4_E")
    p.add_argument("--wandb-project", default="recrag-grpo")
    p.add_argument("--wandb-mode", default="")
    p.add_argument("--run-name", default="")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--out-dir", default="results/grpo_logs/v4")
    p.add_argument("--out-txt", default="compiled/grpo_v4_E.txt")
    p.add_argument("--out-json", default="compiled/grpo_v4_E.json")
    return p


def main() -> None:
    asyncio.run(compile_grpo(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
