#!/usr/bin/env python
"""Build stratified HERA training set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hera.data import save_qa_jsonl, stratified_train, load_qa_jsonl


def _load_annot_index(annot_dir: str) -> dict[str, dict[str, str]]:
    import glob
    import json as _json
    out: dict[str, dict[str, str]] = {}
    for f in glob.glob(f"{annot_dir}/annot_*.jsonl"):
        with open(f) as fh:
            for line in fh:
                try:
                    d = _json.loads(line)
                    qid = str(d.get("id", ""))
                    if qid:
                        out[qid] = {
                            "reasoning_type": str(d.get("reasoning_type", "")),
                            "complexity": str(d.get("complexity", "")),
                        }
                except Exception:
                    continue
    return out


def stratified_difficulty_aware(annot: dict, per_dataset: int, seed: int):
    """Paper §4: stratified, difficulty-aware sampling.

    Bucket by (reasoning_type, complexity) within each source dataset; sample to balance
    coverage across reasoning categories while preserving distributional diversity.
    """
    import random
    from collections import defaultdict
    from hera.data import (
        load_musique_train, load_2wiki_train, load_hotpotqa_train,
    )
    rng = random.Random(seed)
    pool: dict[str, list] = {
        "musique": load_musique_train(limit=10000),
        "2wikimultihop": load_2wiki_train(limit=10000),
        "hotpotqa": load_hotpotqa_train(limit=10000),
    }
    selected: list = []
    for src, exs in pool.items():
        buckets: dict[tuple, list] = defaultdict(list)
        for e in exs:
            a = annot.get(e.id)
            key = ((a.get("reasoning_type") if a else (e.question_type or "bridge")) or "bridge",
                   (a.get("complexity") if a else "medium") or "medium")
            buckets[key].append(e)
        keys = list(buckets.keys())
        rng.shuffle(keys)
        per_bucket = max(1, per_dataset // max(1, len(keys)))
        chosen: list = []
        for k in keys:
            rng.shuffle(buckets[k])
            chosen.extend(buckets[k][:per_bucket])
        rng.shuffle(chosen)
        selected.extend(chosen[:per_dataset])
    rng.shuffle(selected)
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "train_stratified.jsonl"))
    ap.add_argument("--annotations-dir", type=str,
                    default=str(ROOT / "data" / "annotations"),
                    help="GPT-4o annotation dir; if present, paper-style stratified sampling is used.")
    args = ap.parse_args()
    annot = _load_annot_index(args.annotations_dir)
    if annot:
        print(f"Using {len(annot)} GPT-4o annotations for difficulty-aware sampling")
        examples = stratified_difficulty_aware(annot, per_dataset=args.per_dataset, seed=args.seed)
    else:
        print("No annotations found; falling back to question_type heuristic stratification")
        examples = stratified_train(per_dataset=args.per_dataset, seed=args.seed)
    save_qa_jsonl(examples, args.out)
    print(f"Wrote {len(examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
