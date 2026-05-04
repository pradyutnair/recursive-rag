"""Evaluate the true single-agent search lane."""
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

from recrag.lm import make_lm
from recrag.metric import composite_reward
from recrag.retriever import Retriever
from recrag.sas_pipeline import SASConfig, SingleAgentSearchPipeline

TEST_SETS = {
    "musique": "/local/yzheng/pnair/workspace/adaptive-mas/data/musique/questions_1000_seedfull_combined.json",
    "2wikimultihop": "/local/yzheng/pnair/workspace/adaptive-mas/data/2wikimultihop/questions_1000_seed42.json",
    "hotpotqa": "/local/yzheng/pnair/workspace/adaptive-mas/data/hotpotqa/questions_1000_seed42.json",
    "bamboogle": "results/runs/base_forcehard_nocritic_bamboogle_20260504/bamboogle/predictions.jsonl",
}


def make_pipeline(args: argparse.Namespace, idx: int) -> SingleAgentSearchPipeline:
    lm = make_lm(args.lm, replica_idx=idx, max_tokens=args.max_tokens)
    return SingleAgentSearchPipeline(
        lm,
        Retriever(args.retriever_url),
        SASConfig(max_searches=args.max_searches, retrieve_topk=args.retrieve_topk, excerpt_chars=args.excerpt_chars),
    )


async def run_one(q: dict[str, Any], args: argparse.Namespace, idx: int) -> dict[str, Any]:
    pipeline = make_pipeline(args, idx)
    pred = await pipeline.run(str(q.get("question", "")))
    gold = str(q.get("gold", q.get("answer", "")))
    rb = composite_reward(
        pred.get("answer", ""),
        gold,
        [],
        pred.get("metadata", {}).get("total_tokens", 0),
        "auto",
    )
    return {
        "id": str(q.get("id", idx)),
        "dataset": str(q.get("dataset", "")),
        "source_profile": q.get("profile"),
        "question": str(q.get("question", "")),
        "answer": pred.get("answer", ""),
        "gold": gold,
        "metadata": pred.get("metadata", {}),
        "trajectory": pred.get("trajectory", {}),
        "readable_trace": pred.get("readable_trace", ""),
        "reward": rb.as_dict(),
    }


def _hist(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        v = r["metadata"].get(key, "?")
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    em = sum(1 for r in rows if r["reward"]["em"] == 1.0)
    return {
        "n": n,
        "norm_em": round(em / n, 4),
        "token_f1": round(sum(r["reward"]["f1"] for r in rows) / n, 4),
        "contain": round(sum(r["reward"]["contain"] for r in rows) / n, 4),
        "mean_quality": round(sum(r["reward"]["quality"] for r in rows) / n, 4),
        "mean_efficiency": round(sum(r["reward"]["efficiency"] for r in rows) / n, 4),
        "mean_composite": round(sum(r["reward"]["composite"] for r in rows) / n, 4),
        "mean_tokens": round(sum(r["metadata"].get("total_tokens", 0) for r in rows) / n, 1),
        "mean_elapsed_s": round(sum(r["metadata"].get("elapsed_s", 0.0) for r in rows) / n, 2),
        "mean_search_calls": round(sum(r["metadata"].get("search_calls", 0) for r in rows) / n, 2),
        "search_call_dist": _hist(rows, "search_calls"),
        "profile_dist": _hist(rows, "profile"),
        "method": "true_sas_research_plan",
    }


async def run_dataset(args: argparse.Namespace, dataset: str, src_path: Path, out_dir: Path) -> dict:
    raw = src_path.read_text(encoding="utf-8").strip()
    if src_path.suffix == ".jsonl":
        questions = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        questions = json.loads(raw)
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
                    "metadata": {"total_tokens": 0, "elapsed_s": 0, "tool_errors": [str(exc)], "profile": "error"},
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
                    "ds": dataset,
                    "done": i + 1,
                    "n": len(questions),
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
        if not ds:
            continue
        summaries[ds] = await run_dataset(args, ds, Path(TEST_SETS[ds]), out_root / ds)
        print(f"[done] {ds}: {json.dumps(summaries[ds])}")
    (out_root / "summary.json").write_text(json.dumps(summaries, indent=2))
    print("\nALL:", json.dumps(summaries, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--questions-file", default="")
    p.add_argument("--dataset-name", default="")
    p.add_argument("--datasets", default="musique,2wikimultihop,hotpotqa,bamboogle")
    p.add_argument("--n", type=int, default=0, help="0 = all")
    p.add_argument("--lm", default="qwen14b-nothink")
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--max-searches", type=int, default=5)
    p.add_argument("--retrieve-topk", type=int, default=5)
    p.add_argument("--excerpt-chars", type=int, default=700)
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--concurrency", type=int, default=8)
    return p


def main() -> None:
    asyncio.run(run_all(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
