"""Oracle routing signal extracted from a naive_rag (true SAS lane) baseline.

For each training question we know whether the same backbone running plain
naive RAG already solved it. That single bit (plus the SAS token cost) is a
sharp routing signal:

  oracle_easy[qid] = (naive_rag_em == 1)
  naive_tokens[qid] = total_tokens spent by SAS

Used by GEPA + TF-GRPO to shape the composite reward and to tag experience
library entries by difficulty.
"""
from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_ART = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = set(string.punctuation)


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = _ART.sub("", s)
    s = "".join(c for c in s if c not in _PUNCT)
    return " ".join(s.split()).strip()


def _em(pred: str, gold: str) -> int:
    return 1 if _norm(pred) == _norm(gold) else 0


@dataclass
class OracleEntry:
    em: int
    tokens: int
    answer: str = ""
    gold: str = ""


@dataclass
class OracleLookup:
    """Per-qid SAS oracle. Built from naive_<dataset>/predictions.jsonl."""

    entries: dict[str, OracleEntry]

    def get(self, qid: str) -> OracleEntry | None:
        return self.entries.get(str(qid))

    def __len__(self) -> int:
        return len(self.entries)

    def stats(self) -> dict[str, float]:
        if not self.entries:
            return {"n": 0, "em": 0.0, "mean_tokens": 0.0, "easy_share": 0.0}
        n = len(self.entries)
        em = sum(e.em for e in self.entries.values())
        toks = sum(e.tokens for e in self.entries.values())
        return {"n": n, "em": round(em / n, 4), "mean_tokens": round(toks / n, 1), "easy_share": round(em / n, 4)}

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "OracleLookup":
        entries: dict[str, OracleEntry] = {}
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = str(r.get("id", "")).strip()
                if not qid:
                    continue
                pred = str(r.get("answer", r.get("predicted_answer", "")))
                gold = str(r.get("gold_answer", r.get("gold", r.get("answer_gold", ""))))
                tokens = int(r.get("metadata", {}).get("total_tokens", 0))
                entries[qid] = OracleEntry(em=_em(pred, gold), tokens=tokens, answer=pred, gold=gold)
        return cls(entries=entries)

    @classmethod
    def from_naive_dir(cls, base_dir: str | Path, datasets: Iterable[str]) -> "OracleLookup":
        """Convention: <base_dir>/naive_<dataset>/predictions.jsonl"""
        base = Path(base_dir)
        paths = [base / f"naive_{d}" / "predictions.jsonl" for d in datasets]
        return cls.from_paths(paths)
