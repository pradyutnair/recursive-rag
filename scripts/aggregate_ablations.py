"""Aggregate Pareto data (norm_em vs avg tokens) across ablation runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_run(run_dir: Path) -> dict:
    s = run_dir / "summary.json"
    p = run_dir / "predictions.jsonl"
    if not s.exists() or not p.exists():
        return {}
    summary = json.loads(s.read_text())
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not rows:
        return summary
    em = mean(r["reward"]["em"] for r in rows)
    f1 = mean(r["reward"]["f1"] for r in rows)
    contain = mean(r["reward"]["contain"] for r in rows)
    composite = mean(r["reward"]["composite"] for r in rows)
    eff = mean(r["reward"]["efficiency"] for r in rows)
    tokens = mean(r["metadata"].get("total_tokens", 0) for r in rows)
    elapsed = mean(r["metadata"].get("elapsed_s", 0.0) for r in rows)
    nodes = mean(r["metadata"].get("n_nodes", 0) for r in rows)
    citation_acc = mean(1.0 if r["metadata"].get("citation_accepted") else 0.0 for r in rows)
    return {
        **summary,
        "em": round(em, 4), "f1": round(f1, 4), "contain": round(contain, 4),
        "composite": round(composite, 4), "efficiency": round(eff, 4),
        "mean_tokens": round(tokens, 1), "mean_elapsed_s": round(elapsed, 2),
        "mean_nodes": round(nodes, 2), "citation_accept_rate": round(citation_acc, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="space-separated run dirs")
    ap.add_argument("--labels", nargs="+", help="optional labels matching --runs")
    ap.add_argument("--out", default="results/diagnostics/ablation_pareto.json")
    args = ap.parse_args()
    labels = args.labels if args.labels else [Path(r).name for r in args.runs]
    summaries = []
    for label, run in zip(labels, args.runs):
        s = load_run(Path(run))
        if s:
            summaries.append({"run": label, "path": run, **s})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summaries, indent=2))
    cols = ["run", "n", "em", "f1", "contain", "composite", "efficiency", "mean_tokens", "mean_nodes", "mean_elapsed_s", "citation_accept_rate"]
    print("\t".join(cols))
    for s in summaries:
        print("\t".join(str(s.get(c, "-")) for c in cols))


if __name__ == "__main__":
    main()
