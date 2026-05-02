"""Composite Pareto reward: quality * exp(-tokens/T)^alpha.

Used by both GEPA (returns dspy.Prediction-like score+feedback) and TF-GRPO
(used as the rollout reward inside group-relative semantic-advantage extraction).

The metric is intentionally smooth and bounded so it survives small perturbations
in tokens/length without cliffs that destabilise reflection LMs.
"""
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
    quality = 1.0 * em + 0.5 * f1 + 0.3 * contain + 0.4 * grounded + 0.2 * shape
    efficiency = math.exp(-max(0, total_tokens) / token_T)
    composite = quality * (efficiency ** alpha)
    return RewardBreakdown(em, f1, contain, grounded, shape, quality, efficiency, composite, int(total_tokens))


def oracle_bonus(
    em: float,
    topology: str,
    total_tokens: int,
    *,
    oracle_easy: bool,
    naive_tokens: int,
) -> tuple[float, str]:
    """Routing-aware bonus on top of the composite reward.

    Returns (bonus, reason) where reason is a short string for GEPA feedback.
    """
    is_single = topology == "single_hop"
    if oracle_easy:
        if is_single and em == 1.0:
            return 0.5, f"oracle=easy: cheap single_hop recovery (+0.5)"
        if em == 0.0:
            return -0.5, f"oracle=easy: regression vs SAS (which solved this in {naive_tokens} tokens) (-0.5)"
        if naive_tokens > 0 and total_tokens > 3 * naive_tokens:
            return -0.3, f"oracle=easy: correct but {total_tokens} tokens vs SAS's {naive_tokens} (-0.3)"
        return 0.0, "oracle=easy: neutral"
    # oracle_hard: SAS failed
    if em == 1.0:
        return 0.8, f"oracle=hard: MAS recovery of an SAS failure (+0.8)"
    if is_single and em == 0.0:
        return -0.4, f"oracle=hard: chose single_hop and copied SAS's failure (-0.4)"
    return 0.0, "oracle=hard: tried but missed (neutral)"


def composite_reward_with_oracle(
    pred: str,
    gold: str,
    findings: list[dict[str, Any]],
    total_tokens: int,
    expected_type: str = "auto",
    *,
    topology: str = "",
    oracle_easy: bool | None = None,
    naive_tokens: int = 0,
    token_T: float = 8000.0,
    alpha: float = 0.3,
) -> tuple[RewardBreakdown, float, str]:
    rb = composite_reward(pred, gold, findings, total_tokens, expected_type, token_T=token_T, alpha=alpha)
    if oracle_easy is None:
        return rb, rb.composite, ""
    bonus, reason = oracle_bonus(rb.em, topology, total_tokens, oracle_easy=bool(oracle_easy), naive_tokens=int(naive_tokens or 0))
    return rb, rb.composite + bonus, reason


def feedback_text(rb: RewardBreakdown, traj_summary: str = "") -> str:
    """Natural-language feedback for GEPA reflection."""
    parts: list[str] = []
    if rb.em == 1.0:
        parts.append("Correct.")
    elif rb.contain == 1.0:
        parts.append("Answer contains gold but is verbose; tighten the final span.")
    elif rb.f1 >= 0.5:
        parts.append("Partial overlap with gold; refine the target span and answer-type alignment.")
    else:
        parts.append("Wrong answer; revisit decomposition and grounding.")
    if rb.grounded < 0.5:
        parts.append("Final answer is not grounded in cited hop findings; cite-and-align before submit.")
    if rb.shape < 0.5:
        parts.append("Answer shape does not match the expected answer type (date/number/yes_no).")
    if rb.efficiency < 0.5:
        parts.append(f"Token cost too high ({rb.tokens}); collapse to fewer hops or use hop_batch for parallel facts.")
    if rb.efficiency > 1.2 and rb.em == 0.0:
        parts.append("Cheap but wrong; allocate more hops on bridge phrases before submitting.")
    if traj_summary:
        parts.append(traj_summary)
    return " ".join(parts)
