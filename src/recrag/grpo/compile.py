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
import random
from pathlib import Path
from typing import Any

import dspy

from recrag.adaptive_pipeline import AdaptiveConfig, AdaptiveRecursivePipeline
from recrag.grpo.library import ExperienceLibrary
from recrag.grpo.signatures import ExtractGroupOps, OptimizeBatch, SummarizeAdaptiveRollout
from recrag.lm import make_lm
from recrag.metric import composite_reward, oracle_bonus
from recrag.oracle import OracleLookup
from recrag.profile import classify
from recrag.retriever import Retriever


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


def _load_oracle(args: argparse.Namespace) -> OracleLookup | None:
    if not args.oracle_naive_dir:
        return None
    base = Path(args.oracle_naive_dir)
    paths: list[Path] = []
    for ds in (args.oracle_datasets or "").split(","):
        ds = ds.strip()
        if not ds:
            continue
        alias = {"2wikimultihop": "2wiki"}.get(ds, ds)
        p = base / f"naive_{alias}" / "predictions.jsonl"
        if p.exists():
            paths.append(p)
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
    trace_path = out_dir / "compile_trace.jsonl"
    lib = ExperienceLibrary.load(args.seed_library) if args.seed_library else ExperienceLibrary()
    reflection_lm = make_lm(args.reflection_lm, temperature=0.0, max_tokens=args.reflection_max_tokens)
    oracle = _load_oracle(args)
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
                    if max(rewards) - min(rewards) < 0.2:
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
                lib.save_text(args.out_txt)
                lib.save_json(args.out_json)
    return lib


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="data/multidataset/train_v1.json")
    p.add_argument("--n-train", type=int, default=0, help="0 = use all rows")
    p.add_argument("--oracle-naive-dir", default="results/baselines/wiki18-corpus/qwen3-14b-no-think/qwen3_14b_nothink_top5_node408")
    p.add_argument("--oracle-datasets", default="musique,2wikimultihop,hotpotqa")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--root-lm", default="qwen14b-think")
    p.add_argument("--sub-lm", default="qwen14b-nothink")
    p.add_argument("--root-max-tokens", type=int, default=4096)
    p.add_argument("--sub-max-tokens", type=int, default=512)
    p.add_argument("--reflection-lm", default="qwen14b-think")
    p.add_argument("--reflection-max-tokens", type=int, default=2048)
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--max-iters", type=int, default=15)
    p.add_argument("--max-nodes", type=int, default=6)
    p.add_argument("--max-recursion", type=int, default=1)
    p.add_argument("--tau-recurse", type=float, default=0.5)
    p.add_argument("--token-T", type=float, default=8000.0)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--seed-library")
    p.add_argument("--out-dir", default="results/grpo_logs/v2")
    p.add_argument("--out-txt", default="compiled/grpo_v2_E.txt")
    p.add_argument("--out-json", default="compiled/grpo_v2_E.json")
    return p


def main() -> None:
    asyncio.run(compile_grpo(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
