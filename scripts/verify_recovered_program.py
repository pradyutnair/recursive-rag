"""Run a recovered GEPA program on val_v3 to confirm the reconstructed prompts
reproduce yesterday's logged val_aggregate score.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recrag.adaptive_pipeline import AdaptiveConfig, AdaptiveRecursivePipeline
from recrag.lm import make_lm
from recrag.metric import composite_reward_with_oracle
from recrag.oracle import OracleLookup
from recrag.retriever import Retriever


async def run_one(q, args, idx, prompts):
    root_lm = make_lm(args.root_lm, replica_idx=idx, max_tokens=args.root_max_tokens)
    sub_lm = make_lm(args.sub_lm, replica_idx=idx, max_tokens=args.sub_max_tokens)
    pipeline = AdaptiveRecursivePipeline(
        root_lm, sub_lm, Retriever(args.retriever_url),
        AdaptiveConfig(
            planner_instructions=prompts["planner"],
            synth_instructions=prompts["synthesizer"],
            critic_instructions=prompts["critic"],
            max_nodes=args.max_nodes, max_recursion_depth=args.max_recursion,
            tau_recurse=args.tau_recurse, use_dag=True, use_critic=True,
            max_critic_retries=args.max_critic_retries,
        ),
    )
    pred = await pipeline.run(str(q.get("question", "")))
    gold = str(q.get("answer", ""))
    return pred, gold


async def run_all(args):
    prompts = json.loads(Path(args.program).read_text())["prompts"]
    val = json.loads(Path(args.valset).read_text())
    if args.n and args.n > 0:
        val = val[: args.n]
    oracle_paths = []
    base = Path(args.oracle_dir)
    for ds in args.oracle_datasets.split(","):
        for cand in (base / f"{ds}_fresh_naive" / "predictions.jsonl",
                     base / f"naive_{ds}" / "predictions.jsonl",
                     base / f"naive_{ {'2wikimultihop': '2wiki'}.get(ds, ds) }" / "predictions.jsonl"):
            if cand.exists():
                oracle_paths.append(cand)
                break
    oracle = OracleLookup.from_paths(oracle_paths) if oracle_paths else None
    sem = asyncio.Semaphore(args.concurrency)
    rows = [None] * len(val)
    t0 = time.time()

    async def guarded(i, q):
        async with sem:
            try:
                pred, gold = await run_one(q, args, i, prompts)
            except Exception as exc:
                rows[i] = {"id": q.get("id"), "em": 0, "tokens": 0, "score": 0.0, "err": str(exc)}
                return
            qid = str(q.get("id", ""))
            entry = oracle.get(qid) if oracle else None
            oe = (entry.em == 1) if entry else None
            nt = entry.tokens if entry else 0
            topo = pred.get("metadata", {}).get("topology", "")
            rb, score, _ = composite_reward_with_oracle(
                pred.get("answer", ""), gold, pred.get("metadata", {}).get("findings", []),
                pred.get("metadata", {}).get("total_tokens", 0),
                pred.get("metadata", {}).get("expected_type", "auto"),
                topology=topo, oracle_easy=oe, naive_tokens=nt,
            )
            rows[i] = {"id": qid, "em": rb.em, "f1": rb.f1, "tokens": rb.tokens, "score": score, "topo": topo, "answer": pred.get("answer",""), "gold": gold}
            if (i + 1) % 5 == 0 or i == len(val) - 1:
                done = sum(1 for r in rows if r is not None)
                em_running = sum(r["em"] for r in rows if r is not None) / done
                score_avg = sum(r["score"] for r in rows if r is not None) / done
                print(f"done {done}/{len(val)} em_running={em_running:.4f} score_avg={score_avg:.4f} wall_s={time.time()-t0:.1f}", flush=True)

    await asyncio.gather(*[guarded(i, q) for i, q in enumerate(val)])
    n = len(rows)
    em = sum(r["em"] for r in rows) / n
    f1 = sum(r["f1"] for r in rows) / n
    scores = sum(r["score"] for r in rows) / n
    tokens = sum(r["tokens"] for r in rows) / n
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": n, "norm_em": round(em, 4), "token_f1": round(f1, 4),
        "mean_composite_with_oracle": round(scores, 4),
        "mean_tokens": round(tokens, 1),
        "wall_clock_s": round(time.time() - t0, 1),
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    print(f"\nFinal: em={em:.4f}, score_with_oracle={scores:.4f}, tokens={tokens:.1f}\nSaved {args.out}")


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--program", required=True)
    p.add_argument("--valset", default="data/multidataset/val_v3.json")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--root-lm", default="qwen14b-nothink")
    p.add_argument("--sub-lm", default="qwen14b-nothink")
    p.add_argument("--root-max-tokens", type=int, default=768)
    p.add_argument("--sub-max-tokens", type=int, default=512)
    p.add_argument("--max-nodes", type=int, default=6)
    p.add_argument("--max-recursion", type=int, default=0)
    p.add_argument("--tau-recurse", type=float, default=0.5)
    p.add_argument("--max-critic-retries", type=int, default=1)
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--oracle-dir", default="compiled/oracle")
    p.add_argument("--oracle-datasets", default="musique,2wikimultihop,hotpotqa")
    p.add_argument("--concurrency", type=int, default=6)
    return p


def main():
    asyncio.run(run_all(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
