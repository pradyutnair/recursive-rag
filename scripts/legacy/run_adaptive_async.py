"""Run the AdaptiveRecursivePipeline (DAG planner + recursive sub-agents +
synthesizer + citation gate) on a question file. Saves predictions.jsonl,
config.json, summary.json, and a small RUN_NOTE.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recrag.adaptive_pipeline import AdaptiveConfig, AdaptiveRecursivePipeline
from recrag.lm import make_lm
from recrag.metric import composite_reward
from recrag.pipeline import PipelineConfig, ReactRagPipeline
from recrag.retriever import Retriever


def load_questions(path: str, n: int | None) -> list[dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return rows[:n] if n else rows


async def run_one(q: dict[str, Any], args: argparse.Namespace, idx: int) -> dict[str, Any]:
    root_lm = make_lm(args.root_lm, replica_idx=idx, max_tokens=args.root_max_tokens)
    sub_lm = make_lm(args.sub_lm, replica_idx=idx, max_tokens=args.sub_max_tokens)
    if args.mode == "react":
        pipeline = ReactRagPipeline(
            root_lm, sub_lm, Retriever(args.retriever_url),
            PipelineConfig(max_iters=args.max_iters, experience_library=args.experience_library, citation_gate=True),
        )
    else:
        pipeline = AdaptiveRecursivePipeline(
            root_lm, sub_lm, Retriever(args.retriever_url),
            AdaptiveConfig(
                max_nodes=args.max_nodes,
                max_recursion_depth=args.max_recursion,
                tau_recurse=args.tau_recurse,
                experience_library=args.experience_library,
                use_dag=args.mode != "sas",
            ),
        )
    pred = await pipeline.run(str(q.get("question", "")))
    gold = str(q.get("answer", ""))
    rb = composite_reward(
        pred.get("answer", ""), gold, pred.get("metadata", {}).get("findings", []),
        pred.get("metadata", {}).get("total_tokens", 0),
        pred.get("metadata", {}).get("expected_type", "auto"),
        token_T=args.token_T, alpha=args.alpha,
    )
    return {
        "id": str(q.get("id", idx)),
        "question": str(q.get("question", "")),
        "answer": pred.get("answer", ""),
        "predicted_answer": pred.get("answer", ""),
        "gold": gold,
        "metadata": pred.get("metadata", {}),
        "trajectory": pred.get("trajectory", {}),
        "readable_trace": pred.get("readable_trace", ""),
        "reward": rb.as_dict(),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    n = len(rows)
    em = sum(1 for r in rows if r["reward"]["em"] == 1.0)
    f1 = sum(r["reward"]["f1"] for r in rows) / n
    contain = sum(r["reward"]["contain"] for r in rows) / n
    quality = sum(r["reward"]["quality"] for r in rows) / n
    eff = sum(r["reward"]["efficiency"] for r in rows) / n
    comp = sum(r["reward"]["composite"] for r in rows) / n
    tokens = sum(r["metadata"].get("total_tokens", 0) for r in rows) / n
    elapsed = sum(r["metadata"].get("elapsed_s", 0.0) for r in rows) / n
    n_nodes = sum(r["metadata"].get("n_nodes", 0) for r in rows) / n
    profiles: dict[str, int] = {}
    topologies: dict[str, int] = {}
    for r in rows:
        profiles[r["metadata"].get("profile", "?")] = profiles.get(r["metadata"].get("profile", "?"), 0) + 1
        topologies[r["metadata"].get("topology", "?")] = topologies.get(r["metadata"].get("topology", "?"), 0) + 1
    return {
        "n": n,
        "norm_em": round(em / n, 4),
        "token_f1": round(f1, 4),
        "contain": round(contain, 4),
        "mean_quality": round(quality, 4),
        "mean_efficiency": round(eff, 4),
        "mean_composite": round(comp, 4),
        "mean_tokens": round(tokens, 1),
        "mean_elapsed_s": round(elapsed, 2),
        "mean_nodes": round(n_nodes, 2),
        "profiles": profiles,
        "topologies": topologies,
    }


async def run_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    if pred_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {pred_path}; pass --overwrite")
    if pred_path.exists():
        pred_path.unlink()
    questions = load_questions(args.questions, args.n)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict[str, Any] | None] = [None] * len(questions)
    t0 = time.time()

    async def guarded(i: int, q: dict[str, Any]) -> None:
        async with sem:
            try:
                r = await run_one(q, args, i)
            except Exception as exc:
                r = {
                    "id": str(q.get("id", i)),
                    "question": str(q.get("question", "")),
                    "answer": "",
                    "predicted_answer": "",
                    "gold": str(q.get("answer", "")),
                    "metadata": {"total_tokens": 0, "elapsed_s": 0, "tool_errors": [str(exc)], "n_nodes": 0, "profile": "error"},
                    "trajectory": {},
                    "readable_trace": f"ERROR: {exc}",
                    "reward": {"em": 0.0, "f1": 0.0, "contain": 0.0, "grounded": 0.0, "shape": 0.0, "quality": 0.0, "efficiency": 0.0, "composite": 0.0, "tokens": 0},
                }
            rows[i] = r
            with pred_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(json.dumps({
                "done": i + 1, "id": r["id"], "answer": r["answer"], "gold": r["gold"][:60],
                "em": r["reward"]["em"], "tokens": r["metadata"].get("total_tokens", 0),
                "nodes": r["metadata"].get("n_nodes", 0),
                "wall_s": round(time.time() - t0, 1),
            }, ensure_ascii=False), flush=True)

    await asyncio.gather(*[guarded(i, q) for i, q in enumerate(questions)])
    final_rows = [r for r in rows if r is not None]
    summary = summarize(final_rows)
    summary["wall_clock_s"] = round(time.time() - t0, 2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "RUN_NOTE.md").write_text("# Run Note\n\n" + json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nFINAL:", json.dumps(summary, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--n", type=int)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["adaptive", "react", "sas"], default="adaptive")
    p.add_argument("--root-lm", default="qwen14b-nothink")
    p.add_argument("--sub-lm", default="qwen14b-nothink")
    p.add_argument("--root-max-tokens", type=int, default=2048)
    p.add_argument("--sub-max-tokens", type=int, default=512)
    p.add_argument("--max-iters", type=int, default=15)  # ReAct only
    p.add_argument("--max-nodes", type=int, default=6)
    p.add_argument("--max-recursion", type=int, default=2)
    p.add_argument("--tau-recurse", type=float, default=0.5)
    p.add_argument("--experience-library")
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--token-T", type=float, default=8000.0)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    asyncio.run(run_all(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
