"""Quick forensics: bucket failure modes from the best MuSiQue strat100 run."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recrag.contracts import normalize_answer

PRED = ROOT / "results/runs/musique_stratified100/stratified100_topology_span_qwen14b_nothink_20260501_1235/predictions/predictions.jsonl"
GOLD = ROOT / "data/musique/stratified_100.json"

BRIDGE = re.compile(r"\b(of|that|by|where|when|which|who)\b", re.I)


def categorize(row: dict, gold_by_id: dict[str, str]) -> tuple[str, dict]:
    qid = str(row.get("id", ""))
    pred = str(row.get("predicted_answer", row.get("answer", "")))
    gold = gold_by_id.get(qid, "")
    npred, ngold = normalize_answer(pred), normalize_answer(gold)
    if npred == ngold:
        return "correct", {}
    findings = row.get("metadata", {}).get("findings", [])
    confs = [float(f.get("confidence", 0.0)) for f in findings]
    n_findings = len(findings)
    low_conf = sum(1 for c in confs if c < 0.65)
    answers = [normalize_answer(f.get("answer", "")) for f in findings]
    saw_gold_in_findings = any(ngold and (ngold in a or a in ngold) for a in answers)
    saw_gold_in_pred_substring = bool(npred and ngold and (ngold in npred or npred in ngold))
    n_unique_chunks = len({f.get("evidence_chunk_id", "") for f in findings if f.get("evidence_chunk_id")})
    if not pred.strip():
        return "blank_answer", {"findings": n_findings}
    if saw_gold_in_pred_substring:
        return "surface_form", {"pred": pred, "gold": gold}
    if saw_gold_in_findings and not saw_gold_in_pred_substring:
        return "synthesis_dropped_gold", {"pred": pred, "gold": gold, "answers": answers}
    if n_findings == 0:
        return "no_hops_emitted", {}
    if low_conf == n_findings:
        return "all_low_conf_retrieval_miss", {"confs": confs}
    bridge_count = len(BRIDGE.findall(row.get("question", "").lower()))
    if n_findings <= 1 and bridge_count >= 2:
        return "planner_underdecomposed", {"q": row.get("question", ""), "n": n_findings}
    if n_findings >= 4:
        return "planner_overdecomposed", {"n": n_findings}
    if low_conf > 0:
        return "extractor_low_conf_partial", {"confs": confs}
    return "extractor_wrong_high_conf", {"pred": pred, "gold": gold, "answers": answers, "confs": confs}


def main() -> None:
    gold = json.loads(GOLD.read_text())
    gold_by_id = {str(g["id"]): str(g["answer"]) for g in gold}
    rows = [json.loads(l) for l in PRED.read_text().splitlines() if l.strip()]
    cats: Counter[str] = Counter()
    examples: dict[str, list] = {}
    tokens_total = 0
    for r in rows:
        cat, info = categorize(r, gold_by_id)
        cats[cat] += 1
        examples.setdefault(cat, []).append({"id": r.get("id"), "q": r.get("question", "")[:120], **info})
        tokens_total += int(r.get("metadata", {}).get("total_tokens", 0))

    out = {
        "n": len(rows),
        "mean_tokens": round(tokens_total / max(1, len(rows)), 1),
        "buckets": dict(cats.most_common()),
        "examples": {k: v[:3] for k, v in examples.items()},
    }
    out_path = ROOT / "results/diagnostics/failure_categorization_20260501.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"n": out["n"], "buckets": out["buckets"]}, indent=2))


if __name__ == "__main__":
    main()
