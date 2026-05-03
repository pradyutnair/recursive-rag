"""Recompute EM/F1/contain + per-question token cost for the existing FlashRAG
baselines, on the exact same 1000q test ids we evaluate ours on. Outputs a
single CSV-like JSON for plotting Pareto curves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recrag.metric import composite_reward

BASE = ROOT / "results/baselines/wiki18-corpus/qwen3-14b-no-think/qwen3_14b_nothink_top5_node408"

DATASETS = {
    "musique": {"test": "data/musique/questions_1000_seedfull_combined.json", "baseline_alias": "musique"},
    "2wikimultihop": {"test": "data/2wikimultihop/questions_1000_seed42.json", "baseline_alias": "2wiki"},
    "hotpotqa": {"test": "data/hotpotqa/questions_1000_seed42.json", "baseline_alias": "hotpotqa"},
    "bamboogle": {"test": "data/bamboogle/questions_125.json", "baseline_alias": "bamboogle"},
}
METHODS = ["naive", "ircot", "opera", "ma-rag"]


def rescore_one(method: str, ds: str, alias: str, test_path: Path) -> dict:
    pred_path = BASE / f"{method}_{alias}" / "predictions.jsonl"
    if not pred_path.exists():
        return {}
    by_id: dict[str, dict] = {}
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_id[str(r.get("id", ""))] = r
    test_rows = json.loads(test_path.read_text(encoding="utf-8"))
    scored: list[dict] = []
    for q in test_rows:
        qid = str(q.get("id", ""))
        if qid not in by_id:
            continue
        r = by_id[qid]
        gold = str(q.get("answer", ""))
        pred = str(r.get("answer", ""))
        toks = int(r.get("metadata", {}).get("total_tokens", 0))
        rb = composite_reward(pred, gold, [], toks, "auto")
        scored.append({"id": qid, "em": rb.em, "f1": rb.f1, "contain": rb.contain, "tokens": toks})
    if not scored:
        return {}
    n = len(scored)
    return {
        "method": method, "dataset": ds, "n": n,
        "norm_em": round(sum(s["em"] for s in scored) / n, 4),
        "token_f1": round(sum(s["f1"] for s in scored) / n, 4),
        "contain": round(sum(s["contain"] for s in scored) / n, 4),
        "mean_tokens": round(sum(s["tokens"] for s in scored) / n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/diagnostics/baselines_rescored.json")
    args = ap.parse_args()
    results = []
    for ds, cfg in DATASETS.items():
        for method in METHODS:
            r = rescore_one(method, ds, cfg["baseline_alias"], ROOT / cfg["test"])
            if r:
                results.append(r)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    cols = ["method", "dataset", "n", "norm_em", "token_f1", "contain", "mean_tokens"]
    print("\t".join(cols))
    for r in results:
        print("\t".join(str(r.get(c, "-")) for c in cols))


if __name__ == "__main__":
    main()
