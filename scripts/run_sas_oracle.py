"""Run our SAS-mode pipeline on a fresh question pool to produce the oracle.

For each fresh-pool dataset (musique, 2wikimultihop, hotpotqa) we run the
single-agent lane (max_nodes=1, no DAG, no recursion, no critic) on every
question and write predictions.jsonl. The oracle for GEPA / TF-GRPO is then
"did our SAS lane solve this question (em=1)?" plus the SAS-lane token cost.

Input:  data/multidataset/fresh_pool/{dataset}.json
Output: compiled/oracle/{dataset}_fresh_naive/predictions.jsonl + summary.json
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
from recrag.contracts import normalize_answer
from recrag.lm import make_lm
from recrag.retriever import Retriever


def _em(p: str, g: str) -> int:
    return 1 if normalize_answer(p) == normalize_answer(g) else 0


async def run_one(q: dict[str, Any], args: argparse.Namespace, idx: int) -> dict[str, Any]:
    root_lm = make_lm(args.root_lm, replica_idx=idx, max_tokens=args.root_max_tokens)
    sub_lm = make_lm(args.sub_lm, replica_idx=idx, max_tokens=args.sub_max_tokens)
    pipeline = AdaptiveRecursivePipeline(
        root_lm, sub_lm, Retriever(args.retriever_url),
        AdaptiveConfig(
            max_nodes=1,
            max_recursion_depth=0,
            tau_recurse=0.5,
            experience_library=None,
            use_dag=False,
            use_critic=False,
            max_critic_retries=0,
        ),
    )
    pred = await pipeline.run(str(q.get("question", "")))
    pred_ans = pred.get("answer", "")
    gold = str(q.get("answer", ""))
    return {
        "id": str(q.get("id", idx)),
        "dataset": str(q.get("dataset", "")),
        "question": str(q.get("question", "")),
        "answer": pred_ans,
        "gold_answer": gold,
        "metadata": {
            "total_tokens": pred.get("metadata", {}).get("total_tokens", 0),
            "elapsed_s": pred.get("metadata", {}).get("elapsed_s", 0.0),
            "topology": pred.get("metadata", {}).get("topology", "single_hop"),
            "hops": pred.get("metadata", {}).get("hops", 0),
            "retries": pred.get("metadata", {}).get("retries", 0),
        },
    }


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
                    "gold_answer": str(q.get("answer", "")),
                    "metadata": {"total_tokens": 0, "elapsed_s": 0, "error": str(exc)},
                }
            r["em"] = _em(r["answer"], r["gold_answer"])
            em_so_far += r["em"]
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
    em_rate = sum(r["em"] for r in final_rows) / len(final_rows) if final_rows else 0.0
    mean_tokens = sum(r["metadata"].get("total_tokens", 0) for r in final_rows) / len(final_rows) if final_rows else 0.0
    summary = {
        "dataset": dataset,
        "n": len(final_rows),
        "em": round(em_rate, 4),
        "mean_tokens": round(mean_tokens, 1),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


async def run_all(args: argparse.Namespace) -> None:
    summaries = {}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        src = ROOT / args.fresh_pool / f"{ds}.json"
        out_dir = ROOT / args.out_dir / f"{ds}_fresh_naive"
        s = await run_dataset(args, ds, src, out_dir)
        summaries[ds] = s
        print(f"[done] {ds}: {json.dumps(s)}")
    overall = {
        "datasets": summaries,
        "total_em": round(sum(s["em"] * s["n"] for s in summaries.values()) / sum(s["n"] for s in summaries.values()), 4),
        "total_mean_tokens": round(sum(s["mean_tokens"] * s["n"] for s in summaries.values()) / sum(s["n"] for s in summaries.values()), 1),
    }
    (ROOT / args.out_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2))
    print("\nOVERALL:", json.dumps(overall, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--fresh-pool", default="data/multidataset/fresh_pool")
    p.add_argument("--out-dir", default="compiled/oracle")
    p.add_argument("--datasets", default="musique,2wikimultihop,hotpotqa")
    p.add_argument("--n", type=int, default=0, help="0 = all")
    p.add_argument("--root-lm", default="qwen14b-nothink")
    p.add_argument("--sub-lm", default="qwen14b-nothink")
    p.add_argument("--root-max-tokens", type=int, default=512)
    p.add_argument("--sub-max-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--retriever-url", default="http://node408:8003")
    return p


def main() -> None:
    asyncio.run(run_all(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
