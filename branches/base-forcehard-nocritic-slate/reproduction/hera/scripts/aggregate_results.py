#!/usr/bin/env python
"""Aggregate per-dataset summaries into a single report table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    summaries = sorted(out.glob("summary_*.json"))
    rows = []
    for p in summaries:
        d = json.loads(p.read_text())
        rows.append({
            "dataset": d.get("dataset", p.stem),
            "n": d.get("n", 0),
            "em": d.get("em", 0.0),
            "f1": d.get("f1", 0.0),
            "acc": d.get("acc", 0.0),
            "tokens": d.get("tokens", 0.0),
            "elapsed_s": d.get("elapsed_s", 0.0),
        })
    print(f"{'dataset':<16} {'n':>5} {'EM':>7} {'F1':>7} {'Acc':>7} {'tokens':>8} {'time(s)':>8}")
    for r in rows:
        print(f"{r['dataset']:<16} {r['n']:>5} {r['em']:>7.3f} {r['f1']:>7.3f} {r['acc']:>7.3f} {r['tokens']:>8.0f} {r['elapsed_s']:>8.0f}")
    Path(out / "aggregate.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
