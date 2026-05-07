"""Standard QA metrics: normalized exact match, token F1, accuracy (contains)."""
from __future__ import annotations

import re
import string
from collections import Counter


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNC = set(string.punctuation)
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNC)
    s = _ARTICLES.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    pc, gc = Counter(pt), Counter(gt)
    common = pc & gc
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pt)
    r = n / len(gt)
    return (2 * p * r) / (p + r)


def accuracy(pred: str, gold: str) -> float:
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    return 1.0 if (np_ and ng and ng in np_) else 0.0


def best_over_aliases(fn, pred: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    return max(fn(pred, g) for g in golds)


def score_row(pred: str, golds: list[str]) -> dict[str, float]:
    if isinstance(golds, str):
        golds = [golds]
    return {
        "em": best_over_aliases(exact_match, pred, golds),
        "f1": best_over_aliases(token_f1, pred, golds),
        "acc": best_over_aliases(accuracy, pred, golds),
    }
