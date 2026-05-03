"""Reconstruct a specific GEPA candidate program from a crashed run's log file.

Parses /tmp/run_gepa_v3.log (or any GEPA log) to extract the proposed prompt
texts at each iteration, follows lineage (single-mutation parent + merge rules),
and writes a JSON file with (planner, synthesizer, critic) for the requested
candidate index.

Usage:
  python scripts/recover_gepa_program.py --log /tmp/run_gepa_v3.log \
      --candidate 13 --out compiled/gepa_v3_recovered_cand13.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recrag.adaptive_pipeline import (
    DEFAULT_CRITIC_INSTRUCTIONS,
    DEFAULT_PLANNER_INSTRUCTIONS,
    DEFAULT_SYNTH_INSTRUCTIONS,
)


PROPOSE_RE = re.compile(
    r"Iteration (\d+): Proposed new text for (planner|synthesizer|critic): (.*?)(?=\nGEPA Optimization|\n\d{4}/\d{2}/\d{2})",
    re.DOTALL,
)
SELECT_RE = re.compile(r"Iteration (\d+): Selected program (\d+) score: ([-\d\.]+)")
MERGE_RE = re.compile(r"Iteration (\d+): Merged programs (\d+) and (\d+) via ancestor (\d+)")
NEW_CAND_RE = re.compile(r"Iteration (\d+): New program candidate index: (\d+)")
VALSET_SCORE_RE = re.compile(r"Iteration (\d+): Val aggregate for new program: ([-\d\.]+)")


def parse_log(log_text: str):
    """Walk the log line-by-line, building per-iteration events ordered."""
    events_per_iter: dict[int, dict] = {}
    proposed = list(PROPOSE_RE.finditer(log_text))
    for m in proposed:
        it = int(m.group(1))
        events_per_iter.setdefault(it, {})[f"propose_{m.group(2)}"] = m.group(3).strip()
    for m in SELECT_RE.finditer(log_text):
        it = int(m.group(1))
        events_per_iter.setdefault(it, {})["selected_parent"] = int(m.group(2))
        events_per_iter[it]["selected_parent_score"] = float(m.group(3))
    for m in MERGE_RE.finditer(log_text):
        it = int(m.group(1))
        events_per_iter.setdefault(it, {})["merged"] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    for m in NEW_CAND_RE.finditer(log_text):
        it = int(m.group(1))
        events_per_iter.setdefault(it, {})["candidate_index"] = int(m.group(2))
    for m in VALSET_SCORE_RE.finditer(log_text):
        it = int(m.group(1))
        events_per_iter.setdefault(it, {})["val_aggregate"] = float(m.group(2))
    return events_per_iter


def build_candidates(events_per_iter, seed_prompts):
    """Reconstruct each accepted candidate's full (planner, synth, critic)."""
    candidates: dict[int, dict[str, str]] = {0: dict(seed_prompts)}
    agg_scores: dict[int, float] = {0: 0.0}  # seed score; will be updated if available
    for it in sorted(events_per_iter):
        ev = events_per_iter[it]
        idx = ev.get("candidate_index")
        if idx is None:
            continue  # iteration didn't produce a candidate
        if "merged" in ev:
            id1, id2, anc = ev["merged"]
            if id1 not in candidates or id2 not in candidates or anc not in candidates:
                print(f"WARN: iter {it} merge of {id1},{id2} via {anc}; missing parents")
                continue
            new = dict(candidates[anc])  # start from ancestor
            for pred in ("planner", "synthesizer", "critic"):
                pred_anc = candidates[anc][pred]
                pred_1 = candidates[id1][pred]
                pred_2 = candidates[id2][pred]
                if (pred_anc == pred_1 or pred_anc == pred_2) and pred_1 != pred_2:
                    take_from = id2 if pred_anc == pred_1 else id1
                    new[pred] = candidates[take_from][pred]
                elif pred_anc != pred_1 and pred_anc != pred_2:
                    s1 = agg_scores.get(id1, 0.0)
                    s2 = agg_scores.get(id2, 0.0)
                    take_from = id1 if s1 > s2 else (id2 if s2 > s1 else id1)
                    new[pred] = candidates[take_from][pred]
                elif pred_1 == pred_2:
                    new[pred] = pred_1
            candidates[idx] = new
        else:
            parent = ev.get("selected_parent", 0)
            new = dict(candidates.get(parent, candidates[0]))
            for key in ("planner", "synthesizer", "critic"):
                proposed_text = ev.get(f"propose_{key}")
                if proposed_text is not None:
                    new[key] = proposed_text
            candidates[idx] = new
        if "val_aggregate" in ev:
            agg_scores[idx] = ev["val_aggregate"]
    return candidates, agg_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--candidate", type=int, default=13)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list-all", action="store_true")
    args = ap.parse_args()
    log = Path(args.log).read_text(encoding="utf-8")
    events = parse_log(log)
    seed = {
        "planner": DEFAULT_PLANNER_INSTRUCTIONS,
        "synthesizer": DEFAULT_SYNTH_INSTRUCTIONS,
        "critic": DEFAULT_CRITIC_INSTRUCTIONS,
    }
    candidates, agg_scores = build_candidates(events, seed)
    if args.list_all:
        rows = []
        for k in sorted(candidates):
            rows.append({"idx": k, "agg": agg_scores.get(k, 0.0), "planner_len": len(candidates[k]["planner"]), "synth_len": len(candidates[k]["synthesizer"]), "critic_len": len(candidates[k]["critic"])})
        print(json.dumps(rows, indent=2))
    if args.candidate not in candidates:
        sys.exit(f"Candidate {args.candidate} not found. Available: {sorted(candidates)}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidate": args.candidate,
        "val_aggregate": agg_scores.get(args.candidate),
        "prompts": candidates[args.candidate],
        "all_aggregate_scores": agg_scores,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote candidate {args.candidate} (val_aggregate={payload['val_aggregate']}) to {out}")
    print(f"Prompt sizes: planner={len(payload['prompts']['planner'])}, synth={len(payload['prompts']['synthesizer'])}, critic={len(payload['prompts']['critic'])}")


if __name__ == "__main__":
    main()
