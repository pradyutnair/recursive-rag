"""Check whether gold answers appear in retrieved passages for run queries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recrag.contracts import normalize_answer
from recrag.retriever import Retriever


def _queries(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for f in row.get("metadata", {}).get("findings", []):
        for q in f.get("queries_used", []) or []:
            q = str(q).strip()
            if q and q not in out:
                out.append(q)
    return out


def _hit_rank(chunks: list[Any], gold: str) -> int | None:
    needle = normalize_answer(gold)
    if not needle:
        return None
    for i, c in enumerate(chunks, start=1):
        if needle in normalize_answer(c.text):
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retriever-url", default="http://node408:8003")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.predictions).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]
    retriever = Retriever(args.retriever_url, timeout_seconds=120)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        gold = str(row.get("gold") or row.get("answer") or "")
        queries = _queries(row) or [str(row.get("question", ""))]
        batch = retriever._retrieve_batch_sync(queries, args.topk)
        ranks = [_hit_rank(chunks, gold) for chunks in batch]
        best = min([r for r in ranks if r is not None], default=None)
        out_rows.append({
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "em": row.get("reward", {}).get("em"),
            "f1": row.get("reward", {}).get("f1"),
            "tokens": row.get("metadata", {}).get("total_tokens"),
            "gold": gold,
            "prediction": row.get("answer"),
            "n_queries": len(queries),
            "gold_hit_topk": best is not None,
            "best_gold_rank": best,
            "query_hits": [
                {"query": q, "gold_rank": rank}
                for q, rank in zip(queries, ranks)
            ],
        })
        print(json.dumps({"done": len(out_rows), "id": row.get("id"), "best_rank": best}), flush=True)
    summary = {
        "n": len(out_rows),
        "topk": args.topk,
        "gold_hit_rate": round(sum(1 for r in out_rows if r["gold_hit_topk"]) / max(1, len(out_rows)), 4),
        "misses": sum(1 for r in out_rows if not r["gold_hit_topk"]),
        "rows": out_rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
