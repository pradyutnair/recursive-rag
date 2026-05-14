#!/usr/bin/env python
"""Annotate queries with (reasoning_type, complexity) per HERA paper §4.

Paper: "we first use GPT-4o to annotate queries from 2WikiQA, HotpotQA, AmbigQA, and
MuSiQue along reasoning type (bridge, intersection, comparison, temporal multi-hop,
ambiguous) and complexity (easy, medium, hard). Based on these annotations, we perform
stratified, difficulty-aware sampling..."

Output: JSONL with id -> {reasoning_type, complexity}. Cached so re-runs skip annotated.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hera.config import load_env
from hera.data import load_qa_jsonl, load_test_json, load_bamboogle_125, load_musique_train, load_2wiki_train, load_hotpotqa_train
from hera.lm import OpenAIClient, parse_json_lenient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("hera.annotate")


# Paper §D.1: 6 reasoning types (bridge multi-hop, intersection multi-hop, comparison multi-hop,
# temporal multi-hop, causal multi-hop, ambiguous).
REASONING_TYPES = ("bridge", "intersection", "comparison", "temporal", "causal", "ambiguous")
COMPLEXITIES = ("easy", "medium", "hard")


SYSTEM = (
    "You annotate multi-hop QA queries with (reasoning_type, complexity). "
    "Always respond with valid JSON."
)


def build_user(query: str) -> str:
    return f"""Classify this question along two axes (per HERA paper §D.1):

1. reasoning_type: one of
   - bridge: sequential dependency through an intermediate entity (e.g., "Which university did the author of The Old Man and the Sea attend?")
   - intersection: requires answers satisfying multiple independent constraints (e.g., "Which scientists won a Nobel Prize and later served as a university president?")
   - comparison: compares attributes across entities after retrieval (e.g., "Who was born earlier, Marie Curie or Albert Einstein?")
   - temporal: reasoning over time ordering or temporal containment (e.g., "Who was president when the Berlin Wall fell?")
   - causal: explaining cause-effect chains across events (e.g., "Why did the 2008 financial crisis lead to increased banking regulation?")
   - ambiguous: multiple plausible interpretations (e.g., "When did the Manhattan Project begin and end?")

2. complexity: easy, medium, or hard, based on number of reasoning hops and ambiguity.

Question: {query}

Respond:
{{"reasoning_type": "...", "complexity": "..."}}
"""


async def annotate_one(client: OpenAIClient, qid: str, question: str) -> dict:
    res = await client.chat(SYSTEM, build_user(question), json_mode=True, max_tokens=80, temperature=0.0)
    parsed = parse_json_lenient(res.text)
    rt = parsed.get("reasoning_type", "bridge") if isinstance(parsed, dict) else "bridge"
    cx = parsed.get("complexity", "medium") if isinstance(parsed, dict) else "medium"
    if rt not in REASONING_TYPES:
        rt = "bridge"
    if cx not in COMPLEXITIES:
        cx = "medium"
    return {"id": qid, "question": question, "reasoning_type": rt, "complexity": cx}


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                out[str(d["id"])] = d
            except Exception:
                continue
    return out


async def annotate_set(examples: list, out_path: Path, model: str, concurrency: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_existing(out_path)
    todo = [e for e in examples if str(e.id) not in cache]
    logger.info("Annotate %d (cached %d) -> %s", len(todo), len(cache), out_path)
    if not todo:
        return
    client = OpenAIClient(model=model, max_tokens=80, temperature=0.0, concurrency=concurrency)
    sem = asyncio.Semaphore(concurrency)

    async def _runner(ex):
        async with sem:
            try:
                return await annotate_one(client, ex.id, ex.question)
            except Exception as e:
                logger.warning("annotate failed id=%s: %s", ex.id, str(e)[:100])
                return {"id": ex.id, "question": ex.question, "reasoning_type": "bridge", "complexity": "medium"}

    tasks = [_runner(ex) for ex in todo]
    with open(out_path, "a") as f:
        n = 0
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                logger.info("annotated %d/%d", n, len(todo))
    await client.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["train", "test", "all"], default="all")
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "data" / "annotations"))
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--train-limit", type=int, default=2000,
                    help="Max examples per dataset for train annotation pool")
    args = ap.parse_args()
    load_env()

    out_dir = Path(args.out_dir)

    async def run():
        if args.target in ("train", "all"):
            await annotate_set(
                load_musique_train(limit=args.train_limit),
                out_dir / "annot_train_musique.jsonl",
                args.model, args.concurrency,
            )
            await annotate_set(
                load_2wiki_train(limit=args.train_limit),
                out_dir / "annot_train_2wikimultihop.jsonl",
                args.model, args.concurrency,
            )
            await annotate_set(
                load_hotpotqa_train(limit=args.train_limit),
                out_dir / "annot_train_hotpotqa.jsonl",
                args.model, args.concurrency,
            )
        if args.target in ("test", "all"):
            tests = [
                ("/local/yzheng/pnair/data/musique/questions_1000_seedfull_combined.json", "musique"),
                ("/local/yzheng/pnair/data/2wikimultihop/questions_1000_seed42.json", "2wikimultihop"),
                ("/local/yzheng/pnair/data/hotpotqa/questions_1000_seed42.json", "hotpotqa"),
            ]
            for path, src in tests:
                await annotate_set(
                    load_test_json(path, source=src),
                    out_dir / f"annot_test_{src}.jsonl",
                    args.model, args.concurrency,
                )
            await annotate_set(
                load_bamboogle_125(),
                out_dir / "annot_test_bamboogle.jsonl",
                args.model, args.concurrency,
            )

    asyncio.run(run())
    logger.info("Done. Annotations at %s", out_dir)


if __name__ == "__main__":
    main()
