"""Evaluation metrics for the force-hard no-critic base."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .contracts import normalize_answer


def _f1(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    from collections import Counter

    pc, gc = Counter(pt), Counter(gt)
    n = sum((pc & gc).values())
    p = n / len(pt)
    r = n / len(gt)
    return (2 * p * r) / (p + r) if (p + r) else 0.0


def _contain(pred: str, gold: str) -> float:
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    return 1.0 if (np_ and ng and ng in np_) else 0.0


def _grounding(pred: str, findings: list[dict[str, Any]]) -> float:
    """1.0 if predicted answer is found in any cited finding's answer span,
    0.5 if only token overlap >= 0.5, else 0.0.
    """
    if not pred or not findings:
        return 0.0
    np_ = normalize_answer(pred)
    if not np_:
        return 0.0
    p_tokens = set(np_.split())
    best = 0.0
    for f in findings:
        fa = normalize_answer(str(f.get("answer", "")))
        if not fa:
            continue
        if np_ in fa or fa in np_:
            return 1.0
        ft = set(fa.split())
        if p_tokens and ft:
            overlap = len(p_tokens & ft) / len(p_tokens)
            best = max(best, overlap)
    return 1.0 if best >= 0.8 else (0.5 if best >= 0.5 else 0.0)


def _shape_match(pred: str, expected_type: str) -> float:
    if not pred:
        return 0.0
    p = pred.strip().lower()
    if expected_type in ("date",):
        import re

        return 1.0 if re.search(r"\b(?:1[6-9]\d{2}|20\d{2})\b|\bjanuary|february|march|april|may|june|july|august|september|october|november|december\b", p) else 0.4
    if expected_type in ("number",):
        import re

        return 1.0 if re.search(r"\b\d", p) or any(w in p.split() for w in ["one","two","three","four","five","six","seven","eight","nine","ten","hundred","thousand","million","billion"]) else 0.4
    if expected_type in ("yes_no",):
        return 1.0 if p.split()[0] in ("yes", "no", "true", "false") else 0.3
    return 1.0


@dataclass
class RewardBreakdown:
    em: float
    f1: float
    contain: float
    grounded: float
    shape: float
    quality: float
    efficiency: float
    composite: float
    tokens: int

    def as_dict(self) -> dict[str, float]:
        return {
            "em": self.em,
            "f1": round(self.f1, 4),
            "contain": self.contain,
            "grounded": self.grounded,
            "shape": self.shape,
            "quality": round(self.quality, 4),
            "efficiency": round(self.efficiency, 4),
            "composite": round(self.composite, 4),
            "tokens": self.tokens,
        }


def composite_reward(
    pred: str,
    gold: str,
    findings: list[dict[str, Any]],
    total_tokens: int,
    expected_type: str = "auto",
    *,
    token_T: float = 8000.0,
    alpha: float = 0.3,
) -> RewardBreakdown:
    em = 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0
    f1 = _f1(pred, gold)
    contain = _contain(pred, gold)
    grounded = _grounding(pred, findings)
    shape = _shape_match(pred, expected_type)
    quality = 1.5 * em + 1.0 * f1 + 0.3 * contain + 0.4 * grounded + 0.2 * shape
    efficiency = math.exp(-max(0, total_tokens) / token_T)
    composite = quality * (efficiency ** alpha)
    return RewardBreakdown(em, f1, contain, grounded, shape, quality, efficiency, composite, int(total_tokens))
