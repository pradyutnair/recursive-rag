"""Analyze adaptive RAG runs without re-running models.

Inputs are prediction JSONL files from scripts/eval_on_test.py. The script
computes per-run summaries, per-profile slices, route/oracle summaries when
oracle_easy is present, and paired bootstrap deltas between runs sharing ids.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _em(row: dict[str, Any]) -> float:
    return float(row.get("reward", {}).get("em", 0.0))


def _f1(row: dict[str, Any]) -> float:
    return float(row.get("reward", {}).get("f1", 0.0))


def _tokens(row: dict[str, Any]) -> float:
    return float(row.get("metadata", {}).get("total_tokens", 0.0))


def _profile(row: dict[str, Any]) -> str:
    return str(row.get("source_profile") or row.get("metadata", {}).get("profile") or "unknown")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    easy = [r for r in rows if r.get("metadata", {}).get("route") == "easy"]
    out = {
        "n": n,
        "em": round(mean(_em(r) for r in rows), 4),
        "f1": round(mean(_f1(r) for r in rows), 4),
        "mean_tokens": round(mean(_tokens(r) for r in rows), 1),
        "easy_route_fraction": round(len(easy) / n, 4),
        "easy_route_em": round(mean(_em(r) for r in easy), 4) if easy else 0.0,
        "easy_route_mean_tokens": round(mean(_tokens(r) for r in easy), 1) if easy else 0.0,
    }
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_profile[_profile(r)].append(r)
    out["profiles"] = {
        p: {
            "n": len(rs),
            "em": round(mean(_em(r) for r in rs), 4),
            "mean_tokens": round(mean(_tokens(r) for r in rs), 1),
            "easy_route_fraction": round(sum(1 for r in rs if r.get("metadata", {}).get("route") == "easy") / len(rs), 4),
        }
        for p, rs in sorted(by_profile.items())
    }
    if any(r.get("oracle_easy") is not None for r in rows):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            route = str(r.get("metadata", {}).get("route", "unknown"))
            oracle = "easy" if r.get("oracle_easy") is True else "hard" if r.get("oracle_easy") is False else "unknown"
            buckets[f"oracle_{oracle}__route_{route}"].append(r)
        out["routing_oracle"] = {
            k: {
                "n": len(rs),
                "em": round(mean(_em(r) for r in rs), 4),
                "mean_tokens": round(mean(_tokens(r) for r in rs), 1),
            }
            for k, rs in sorted(buckets.items())
        }
    return out


def paired_bootstrap(a: list[dict[str, Any]], b: list[dict[str, Any]], *, n_boot: int, seed: int) -> dict[str, Any]:
    by_a = {str(r.get("id")): r for r in a}
    by_b = {str(r.get("id")): r for r in b}
    ids = sorted(set(by_a) & set(by_b))
    n = len(ids)
    if n == 0:
        return {"n": 0}
    base_delta = mean(_em(by_a[i]) - _em(by_b[i]) for i in ids)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        draw = [ids[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(_em(by_a[i]) - _em(by_b[i]) for i in draw))
    samples.sort()
    lo = samples[int(0.025 * (n_boot - 1))]
    hi = samples[int(0.975 * (n_boot - 1))]
    if base_delta >= 0:
        p = sum(1 for x in samples if x <= 0.0) / n_boot
    else:
        p = sum(1 for x in samples if x >= 0.0) / n_boot
    return {
        "n": n,
        "delta_em": round(base_delta, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "p_two_sided_approx": round(min(1.0, 2 * p), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="prediction JSONL files")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"), help="Label pair for paired bootstrap A-B")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels = args.labels or [Path(p).parent.name for p in args.runs]
    data = {label: _read_jsonl(Path(path)) for label, path in zip(labels, args.runs)}
    out = {"runs": {label: summarize(rows) for label, rows in data.items()}, "comparisons": {}}
    for a, b in args.compare:
        if a in data and b in data:
            out["comparisons"][f"{a}_minus_{b}"] = paired_bootstrap(data[a], data[b], n_boot=args.n_bootstrap, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
