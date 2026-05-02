"""Build a HERA-style multi-dataset stratified trainset for GEPA / TF-GRPO.

Strata are (dataset x oracle_easy x profile). Within each stratum we sample
proportionally to the stratum size (capped per stratum). Bamboogle is held out
as OOD eval and is NOT included in the training pool.

Output: data/multidataset/<name>.json — list of dicts with
  {id, dataset, question, answer, profile, oracle_easy, naive_tokens}

The `dataset` and `oracle_easy` fields are propagated through the pipeline so
GEPA / TF-GRPO see them as auxiliary metadata for routing-aware feedback.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recrag.oracle import OracleLookup
from recrag.profile import classify

NAIVE_BASE = ROOT / "results/baselines/wiki18-corpus/qwen3-14b-no-think/qwen3_14b_nothink_top5_node408"

DATASETS = {
    "musique": {
        "questions": "data/musique/stratified_100.json",
        "naive": "naive_musique",
    },
    "2wikimultihop": {
        "questions": "data/2wikimultihop/questions_1000_seed42.json",
        "naive": "naive_2wiki",
    },
    "hotpotqa": {
        "questions": "data/hotpotqa/questions_1000_seed42.json",
        "naive": "naive_hotpotqa",
    },
}


def stratified_sample(rows: list[dict], oracle: OracleLookup, n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_stratum: dict[tuple[bool, str], list[dict]] = defaultdict(list)
    for r in rows:
        qid = str(r.get("id", ""))
        entry = oracle.get(qid)
        easy = bool(entry and entry.em == 1)
        prof = classify(str(r.get("question", "")))
        r["_oracle_easy"] = easy
        r["_oracle_tokens"] = int(entry.tokens) if entry else 0
        r["_profile"] = prof
        by_stratum[(easy, prof)].append(r)
    # Allocate per-stratum proportional to size, with min 1 if stratum non-empty
    total_avail = sum(len(v) for v in by_stratum.values())
    if total_avail == 0:
        return []
    per_stratum: dict[tuple[bool, str], int] = {}
    leftover = n
    for k, v in by_stratum.items():
        share = max(1, round(n * len(v) / total_avail))
        per_stratum[k] = min(share, len(v))
        leftover -= per_stratum[k]
    # Greedy fill / trim leftover
    keys = list(by_stratum.keys())
    while leftover > 0 and keys:
        rng.shuffle(keys)
        progress = False
        for k in keys:
            if per_stratum[k] < len(by_stratum[k]):
                per_stratum[k] += 1
                leftover -= 1
                progress = True
                if leftover <= 0:
                    break
        if not progress:
            break
    while leftover < 0 and keys:
        rng.shuffle(keys)
        for k in keys:
            if per_stratum[k] > 1:
                per_stratum[k] -= 1
                leftover += 1
                if leftover >= 0:
                    break
    out: list[dict] = []
    for k, want in per_stratum.items():
        pool = by_stratum[k]
        rng.shuffle(pool)
        out.extend(pool[:want])
    return out


def build(args) -> None:
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train: list[dict] = []
    val: list[dict] = []
    per_ds = max(1, args.train_per_dataset)
    val_share = max(0.0, min(0.5, args.val_share))
    stats = {}
    for ds, cfg in DATASETS.items():
        rows = json.loads((ROOT / cfg["questions"]).read_text())
        oracle = OracleLookup.from_paths([NAIVE_BASE / cfg["naive"] / "predictions.jsonl"])
        n_target = int(per_ds * (1 + val_share))
        sampled = stratified_sample(rows, oracle, n=n_target, seed=args.seed + hash(ds) % 1000)
        rng.shuffle(sampled)
        n_val = int(round(len(sampled) * val_share))
        ds_val = sampled[:n_val]
        ds_train = sampled[n_val:]
        for r in ds_train:
            train.append({
                "id": r["id"], "dataset": ds, "question": r["question"], "answer": r["answer"],
                "profile": r["_profile"], "oracle_easy": r["_oracle_easy"],
                "naive_tokens": r["_oracle_tokens"],
            })
        for r in ds_val:
            val.append({
                "id": r["id"], "dataset": ds, "question": r["question"], "answer": r["answer"],
                "profile": r["_profile"], "oracle_easy": r["_oracle_easy"],
                "naive_tokens": r["_oracle_tokens"],
            })
        easy_t = sum(1 for r in ds_train if r["_oracle_easy"])
        easy_v = sum(1 for r in ds_val if r["_oracle_easy"])
        prof_dist = {}
        for r in ds_train + ds_val:
            prof_dist[r["_profile"]] = prof_dist.get(r["_profile"], 0) + 1
        stats[ds] = {"train": len(ds_train), "val": len(ds_val), "easy_train": easy_t, "easy_val": easy_v, "profiles": prof_dist}
    train_path = out_dir / f"train_{args.tag}.json"
    val_path = out_dir / f"val_{args.tag}.json"
    train_path.write_text(json.dumps(train, indent=2, ensure_ascii=False))
    val_path.write_text(json.dumps(val, indent=2, ensure_ascii=False))
    summary = {
        "train_path": str(train_path.relative_to(ROOT)),
        "val_path": str(val_path.relative_to(ROOT)),
        "n_train": len(train), "n_val": len(val),
        "stats_per_dataset": stats,
    }
    (out_dir / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/multidataset")
    p.add_argument("--train-per-dataset", type=int, default=20)
    p.add_argument("--val-share", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="v1")
    build(p.parse_args())


if __name__ == "__main__":
    main()
