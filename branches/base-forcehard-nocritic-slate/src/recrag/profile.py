"""Lightweight, deterministic question-profile classifier.

Used as the key into the topology-conditioned experience library. Profiles:
  - one_hop: single direct fact, no nested bridge
  - parallel_compare: comparison/intersection across N entities
  - bridge_2hop: one bridge phrase, sequential
  - bridge_3hop_plus: nested bridge phrases, 3+ hops
  - temporal: when/year/date with bridge
  - numeric: how many / count / rank with bridge
  - yes_no: boolean question

Heuristic-only on purpose: cheap, fast, and stable as a key. A learned
classifier can replace this later if/when SFT happens.
"""
from __future__ import annotations

import re

_BRIDGE_TOKENS = (
    " of ", " who ", " that ", " which ", " where ", " when ", " by ", " from ",
    " whose ", " in which ", " for the ", " for which ",
)
_COMPARE = re.compile(r"\b(both|either|each|whose|who is older|who was first|same|different|or which|or who)\b", re.I)
_INTERSECT = re.compile(r"\b(both|all of|each of|in common|share|shared)\b", re.I)
_TEMPORAL = re.compile(r"\b(when|what year|what date|how old|since|before|after|during)\b", re.I)
_NUMERIC = re.compile(r"\b(how many|how much|number of|rank|count of|tallest|largest|smallest|highest|lowest)\b", re.I)
_YESNO = re.compile(r"^(is|was|were|are|do|does|did|has|have|had|can|could|should|will|would)\b", re.I)


def classify(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return "one_hop"
    ql = q.lower()
    # Yes/no first because it dominates surface form
    if _YESNO.match(ql) and (" or " not in ql or _COMPARE.search(ql) is None):
        # Distinguish wh-leading short questions; simple yes/no heuristic
        if not any(w in ql for w in [" who ", " what ", " when ", " where ", " how "]):
            return "yes_no"
    # Count bridges
    bridge_count = sum(ql.count(t) for t in _BRIDGE_TOKENS)
    has_compare = bool(_COMPARE.search(ql)) or bool(_INTERSECT.search(ql))
    has_temporal = bool(_TEMPORAL.search(ql))
    has_numeric = bool(_NUMERIC.search(ql))
    if has_compare and bridge_count <= 2:
        return "parallel_compare"
    if has_numeric and bridge_count >= 1:
        return "numeric"
    if has_temporal and bridge_count >= 1:
        return "temporal"
    if bridge_count >= 3:
        return "bridge_3hop_plus"
    if bridge_count >= 1:
        return "bridge_2hop"
    return "one_hop"


def expected_hops(profile: str) -> int:
    return {
        "one_hop": 1,
        "yes_no": 1,
        "bridge_2hop": 2,
        "parallel_compare": 2,
        "temporal": 2,
        "numeric": 2,
        "bridge_3hop_plus": 3,
    }.get(profile, 2)
