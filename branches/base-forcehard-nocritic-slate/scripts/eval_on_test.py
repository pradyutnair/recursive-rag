"""Run a configured AdaptiveRecursivePipeline on the fixed 1000q test sets and
the OOD Bamboogle 125q. Outputs predictions.jsonl and a Pareto summary that
can be compared apples-to-apples with the existing FlashRAG baselines.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recrag.adaptive_pipeline import AdaptiveConfig, AdaptiveRecursivePipeline
from recrag.lm import make_lm
from recrag.metric import composite_reward
from recrag.retriever import Retriever

TEST_SETS = {
    "musique": "/local/yzheng/pnair/workspace/adaptive-mas/data/musique/questions_1000_seedfull_combined.json",
    "2wikimultihop": "/local/yzheng/pnair/workspace/adaptive-mas/data/2wikimultihop/questions_1000_seed42.json",
    "hotpotqa": "/local/yzheng/pnair/workspace/adaptive-mas/data/hotpotqa/questions_1000_seed42.json",
    "bamboogle": "/local/yzheng/pnair/workspace/adaptive-mas/data/bamboogle/questions_125.json",
}


def _load_program(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return None
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
    return prompts or None


def make_pipeline(args: argparse.Namespace, idx: int) -> AdaptiveRecursivePipeline:
    root_lm = make_lm(args.root_lm, replica_idx=idx, max_tokens=args.root_max_tokens)
    sub_lm = make_lm(args.sub_lm, replica_idx=idx, max_tokens=args.sub_max_tokens)
    force_route = None if args.force_route == "auto" else args.force_route
    if force_route is None and args.random_easy_rate >= 0:
        rng = random.Random(args.random_route_seed + idx)
        force_route = "easy" if rng.random() < args.random_easy_rate else "hard"
    cfg_kwargs: dict[str, Any] = dict(
        max_nodes=args.max_nodes,
        max_recursion_depth=args.max_recursion,
        tau_recurse=args.tau_recurse,
        experience_library=args.experience_library,
        use_dag=args.use_dag,
        use_critic=args.use_critic,
        use_router=args.use_router,
        force_route=force_route,
        max_critic_retries=args.max_critic_retries,
        budget_hint=args.budget_hint,
        max_searches=args.max_searches,
        share_mode=args.share_mode,
        worker_width=args.worker_width,
        blackboard_top_k=args.blackboard_top_k,
        repair_budget=args.repair_budget,
        use_escalation=args.use_escalation,
        tau_escalate=args.tau_escalate,
        easy_max_attempts=args.easy_max_attempts,
    )
    prompts = _load_program(args.program)
    if prompts:
        if "router" in prompts:
            cfg_kwargs["router_instructions"] = prompts["router"]
        if "planner" in prompts:
            cfg_kwargs["planner_instructions"] = prompts["planner"]
        if "synthesizer" in prompts:
            cfg_kwargs["synth_instructions"] = prompts["synthesizer"]
        if "critic" in prompts:
            cfg_kwargs["critic_instructions"] = prompts["critic"]
    return AdaptiveRecursivePipeline(root_lm, sub_lm, Retriever(args.retriever_url), AdaptiveConfig(**cfg_kwargs))


async def run_one(q: dict[str, Any], args: argparse.Namespace, idx: int) -> dict[str, Any]:
    pipeline = make_pipeline(args, idx)
    pred = await pipeline.run(str(q.get("question", "")), budget_hint=args.budget_hint)
    gold = str(q.get("answer", ""))
    rb = composite_reward(
        pred.get("answer", ""), gold, pred.get("metadata", {}).get("findings", []),
        pred.get("metadata", {}).get("total_tokens", 0),
        pred.get("metadata", {}).get("expected_type", "auto"),
    )
    return {
        "id": str(q.get("id", idx)),
        "dataset": str(q.get("dataset", "")),
        "source_profile": q.get("profile"),
        "oracle_easy": q.get("oracle_easy"),
        "naive_tokens": q.get("naive_tokens"),
        "question": str(q.get("question", "")),
        "answer": pred.get("answer", ""),
        "gold": gold,
        "metadata": pred.get("metadata", {}),
        "trajectory": pred.get("trajectory", {}),
        "readable_trace": pred.get("readable_trace", ""),
        "reward": rb.as_dict(),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    em = sum(1 for r in rows if r["reward"]["em"] == 1.0)
    easy_rows = [r for r in rows if r["metadata"].get("route") == "easy"]
    summary = {
        "n": n,
        "norm_em": round(em / n, 4),
        "token_f1": round(sum(r["reward"]["f1"] for r in rows) / n, 4),
        "contain": round(sum(r["reward"]["contain"] for r in rows) / n, 4),
        "mean_quality": round(sum(r["reward"]["quality"] for r in rows) / n, 4),
        "mean_efficiency": round(sum(r["reward"]["efficiency"] for r in rows) / n, 4),
        "mean_composite": round(sum(r["reward"]["composite"] for r in rows) / n, 4),
        "mean_tokens": round(sum(r["metadata"].get("total_tokens", 0) for r in rows) / n, 1),
        "mean_elapsed_s": round(sum(r["metadata"].get("elapsed_s", 0.0) for r in rows) / n, 2),
        "mean_nodes": round(sum(r["metadata"].get("n_nodes", 0) for r in rows) / n, 2),
        "route_dist": _hist(rows, "route"),
        "topology_dist": _hist(rows, "topology"),
        "profile_dist": _hist(rows, "profile"),
        "topology_mutated_rate": round(sum(1 for r in rows if r["metadata"].get("topology_mutated")) / n, 4),
        "escalation_rate": round(sum(1 for r in rows if r["metadata"].get("escalated")) / n, 4),
        "mean_wall_clock": round(sum(r["metadata"].get("elapsed_s", 0.0) for r in rows) / n, 2),
        "mean_retrievals": round(sum(r["metadata"].get("hops", 0) for r in rows) / n, 2),
        "answer_rate": round(sum(1 for r in rows if str(r.get("answer", "")).strip()) / n, 4),
        "share_mode_dist": _hist(rows, "share_mode"),
        "worker_width_dist": _hist(rows, "worker_width"),
        "blackboard_top_k_dist": _hist(rows, "blackboard_top_k"),
    }
    if easy_rows:
        summary["easy_route_fraction"] = round(len(easy_rows) / n, 4)
        summary["easy_route_norm_em"] = round(sum(1 for r in easy_rows if r["reward"]["em"] == 1.0) / len(easy_rows), 4)
        summary["easy_route_mean_tokens"] = round(sum(r["metadata"].get("total_tokens", 0) for r in easy_rows) / len(easy_rows), 1)
    else:
        summary["easy_route_fraction"] = 0.0
        summary["easy_route_norm_em"] = 0.0
        summary["easy_route_mean_tokens"] = 0.0
    return summary


def _hist(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        v = r["metadata"].get(key, "?")
        out[str(v)] = out.get(str(v), 0) + 1
    return out


async def run_dataset(args: argparse.Namespace, dataset: str, src_path: Path, out_dir: Path) -> dict:
    questions = json.loads(src_path.read_text(encoding="utf-8"))
    if args.n and args.n > 0:
        questions = questions[: args.n]
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    if pred_path.exists():
        pred_path.unlink()
    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict | None] = [None] * len(questions)
    t0 = time.time()
    em_so_far = 0

    async def guarded(i: int, q: dict) -> None:
        nonlocal em_so_far
        async with sem:
            try:
                r = await run_one(q, args, i)
            except Exception as exc:
                r = {
                    "id": str(q.get("id", i)),
                    "dataset": dataset,
                    "question": str(q.get("question", "")),
                    "answer": "",
                    "gold": str(q.get("answer", "")),
                    "metadata": {"total_tokens": 0, "elapsed_s": 0, "tool_errors": [str(exc)], "n_nodes": 0, "profile": "error"},
                    "trajectory": {},
                    "readable_trace": f"ERROR: {exc}",
                    "reward": {"em": 0.0, "f1": 0.0, "contain": 0.0, "grounded": 0.0, "shape": 0.0, "quality": 0.0, "efficiency": 0.0, "composite": 0.0, "tokens": 0},
                }
            em_so_far += int(r["reward"]["em"])
            rows[i] = r
            with pred_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0 or i == len(questions) - 1:
                print(json.dumps({
                    "ds": dataset, "done": i + 1, "n": len(questions),
                    "em_running": round(em_so_far / (i + 1), 4),
                    "wall_s": round(time.time() - t0, 1),
                }), flush=True)

    await asyncio.gather(*[guarded(i, q) for i, q in enumerate(questions)])
    final_rows = [r for r in rows if r is not None]
    summary = summarize(final_rows)
    summary["wall_clock_s"] = round(time.time() - t0, 2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


async def run_all(args: argparse.Namespace) -> None:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}
    if args.questions_file:
        src = Path(args.questions_file)
        ds = args.dataset_name or src.stem
        summaries[ds] = await run_dataset(args, ds, src, out_root / ds)
        (out_root / "summary.json").write_text(json.dumps(summaries, indent=2))
        print("\nALL:", json.dumps(summaries, indent=2))
        return
    for ds in args.datasets.split(","):
        ds = ds.strip()
        if not ds or ds not in TEST_SETS:
            continue
        src = ROOT / TEST_SETS[ds]
        out_dir = out_root / ds
        s = await run_dataset(args, ds, src, out_dir)
        summaries[ds] = s
        print(f"[done] {ds}: {json.dumps(s)}")
    (out_root / "summary.json").write_text(json.dumps(summaries, indent=2))
    print("\nALL:", json.dumps(summaries, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--questions-file", default="", help="Run an arbitrary JSON question file instead of built-in test sets")
    p.add_argument("--dataset-name", default="", help="Dataset label for --questions-file output")
    p.add_argument("--datasets", default="musique,2wikimultihop,hotpotqa,bamboogle")
    p.add_argument("--n", type=int, default=0, help="0 = all")
    p.add_argument("--root-lm", default="qwen14b-nothink")
    p.add_argument("--sub-lm", default="qwen14b-nothink")
    p.add_argument("--root-max-tokens", type=int, default=768)
    p.add_argument("--sub-max-tokens", type=int, default=512)
    p.add_argument("--max-nodes", type=int, default=6)
    p.add_argument("--max-recursion", type=int, default=0)
    p.add_argument("--tau-recurse", type=float, default=0.5)
    p.add_argument("--use-dag", action="store_true", default=True)
    p.add_argument("--no-dag", dest="use_dag", action="store_false")
    p.add_argument("--use-critic", action="store_true", default=False)
    p.add_argument("--no-critic", dest="use_critic", action="store_false")
    p.add_argument("--use-escalation", action="store_true", default=True)
    p.add_argument("--no-escalation", dest="use_escalation", action="store_false")
    p.add_argument("--tau-escalate", type=float, default=0.7)
    p.add_argument("--easy-max-attempts", type=int, default=2)
    p.add_argument("--use-router", action="store_true", default=True)
    p.add_argument("--no-router", dest="use_router", action="store_false")
    p.add_argument("--force-route", choices=["auto", "easy", "hard"], default="auto")
    p.add_argument("--random-easy-rate", type=float, default=-1.0, help="Ablation only: force random easy route with this probability when >=0")
    p.add_argument("--random-route-seed", type=int, default=17)
    p.add_argument("--max-critic-retries", type=int, default=0)
    p.add_argument("--max-searches", type=int, default=5)
    p.add_argument("--share-mode", choices=["full_share", "parents_only", "blind_workers"], default="full_share")
    p.add_argument("--worker-width", type=int, default=999)
    p.add_argument("--blackboard-top-k", type=int, default=3)
    p.add_argument("--repair-budget", type=int, default=0)
    p.add_argument("--budget-hint", choices=["tight", "normal", "rich"], default="normal")
    p.add_argument("--experience-library")
    p.add_argument("--program", help="JSON file with recovered/compiled prompts {planner, synthesizer, critic}")
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--concurrency", type=int, default=8)
    return p


def main() -> None:
    asyncio.run(run_all(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
