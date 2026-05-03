"""Profile-Insight-Utility experience library (HERA-inspired, structured).

Stores entries as (profile, insight, rationale, utility). Retrieval at runtime
returns top-K by (profile match boost + utility), with diversity filtering to
avoid near-duplicate insights. Utility increments on successful application.

Backwards-compatible with the previous flat text format: load() understands
both legacy text-only entries and the new structured entries.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BASE_PROFILES = (
    "any",
    "one_hop",
    "yes_no",
    "bridge_2hop",
    "bridge_3hop_plus",
    "parallel_compare",
    "temporal",
    "numeric",
)
VALID_DIFFICULTIES = ("easy", "hard", "unknown")
VALID_BUDGETS = ("tight", "normal", "rich")


@dataclass
class ExperienceEntry:
    id: str
    text: str  # the insight (canonical short rule)
    rationale: str = ""
    profile: str = "any"
    utility: int = 0
    uses: int = 0


@dataclass
class ExperienceLibrary:
    entries: list[ExperienceEntry] = field(default_factory=list)
    next_num: int = 1
    max_entries: int = 30

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceLibrary":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        entries: list[ExperienceEntry] = []
        for x in data.get("entries", []):
            if "profile" not in x:
                x["profile"] = "any"
            if "utility" not in x:
                x["utility"] = 0
            if "uses" not in x:
                x["uses"] = 0
            entries.append(ExperienceEntry(**{k: v for k, v in x.items() if k in ExperienceEntry.__dataclass_fields__}))
        lib = cls(entries=entries, next_num=int(data.get("next_num", 1)), max_entries=int(data.get("max_entries", 30)))
        if entries:
            try:
                lib.next_num = max(lib.next_num, max(int(e.id.split("-")[-1]) for e in entries) + 1)
            except Exception:
                pass
        return lib

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.prune()
        p.write_text(json.dumps({"next_num": self.next_num, "max_entries": self.max_entries, "entries": [asdict(e) for e in self.entries]}, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_text(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_text(), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {"next_num": self.next_num, "max_entries": self.max_entries, "entries": [asdict(e) for e in self.entries]}

    def to_text(self, profile: str | None = None, top_k: int | None = None) -> str:
        rows = self.retrieve(profile, top_k=top_k) if profile else list(self.entries)
        return "\n".join(f"{e.id} [{e.profile}|u={e.utility}]: {e.text}" for e in rows)

    def add(self, text: str, rationale: str = "", profile: str = "any") -> str:
        text = text.strip()
        if not text:
            return ""
        prof = _normalize_profile(profile)
        eid = f"E-{self.next_num:03d}"
        self.next_num += 1
        self.entries.append(ExperienceEntry(id=eid, text=text, rationale=rationale.strip(), profile=prof))
        self.prune()
        return eid

    def modify(self, eid: str, text: str = "", rationale: str = "", profile: str | None = None) -> None:
        for e in self.entries:
            if e.id == eid:
                if text.strip():
                    e.text = text.strip()
                if rationale.strip():
                    e.rationale = rationale.strip()
                if profile:
                    e.profile = _normalize_profile(profile)
                self.prune()
                return

    def delete(self, eid: str) -> None:
        self.entries = [e for e in self.entries if e.id != eid]

    def reward(self, eid: str, hit: bool = True) -> None:
        for e in self.entries:
            if e.id == eid:
                e.uses += 1
                if hit:
                    e.utility += 1
                return

    def retrieve(self, profile: str | None, top_k: int | None = 4) -> list[ExperienceEntry]:
        """Return top-K entries: profile match boost + utility, with simple
        token-overlap diversity to avoid near-duplicates."""
        if not self.entries:
            return []
        rows = list(self.entries)
        prof = profile or "any"

        def score(e: ExperienceEntry) -> float:
            match_boost = _profile_match_score(e.profile, prof)
            util = e.utility / max(1, e.uses or 1)
            return match_boost + util

        rows.sort(key=score, reverse=True)
        if top_k is None or top_k <= 0:
            return rows
        # Diversity filter
        out: list[ExperienceEntry] = []
        seen_tokens: list[set[str]] = []
        for e in rows:
            toks = set(re.findall(r"[a-zA-Z]+", e.text.lower()))
            if any(len(toks & s) / max(1, len(toks)) > 0.7 for s in seen_tokens):
                continue
            out.append(e)
            seen_tokens.append(toks)
            if len(out) >= top_k:
                break
        return out

    def merge_text(self, text: str) -> None:
        """Best-effort merge from a free-form text dump (for backward compat)."""
        seen = {self._canon(e.text) for e in self.entries}
        for line in text.splitlines():
            clean = re.sub(r"^E-\d{3}\s*(\[[^\]]+\])?:?\s*", "", line).strip(" -\t")
            key = self._canon(clean)
            if key and key not in seen:
                self.add(clean)
                seen.add(key)

    def apply_ops(self, ops: list[dict[str, Any]]) -> None:
        for op in ops or []:
            kind = str(op.get("op", "KEEP")).upper()
            if kind == "ADD":
                self.add(str(op.get("text", "")), str(op.get("rationale", "")), str(op.get("profile", "any")))
            elif kind == "MODIFY":
                self.modify(str(op.get("id", "")), str(op.get("text", "")), str(op.get("rationale", "")), str(op.get("profile", "")) or None)
            elif kind == "DELETE":
                self.delete(str(op.get("id", "")))
            elif kind == "MERGE":
                ids = [str(x) for x in op.get("ids", []) if str(x)]
                text = str(op.get("text", "")).strip()
                rationale = str(op.get("rationale", "")).strip()
                profile = str(op.get("profile", "any"))
                if len(ids) >= 2 and text:
                    for eid in ids:
                        self.delete(eid)
                    self.add(text, rationale, profile)
            elif kind == "PRUNE":
                self.prune()
        self.prune()

    def prune(self) -> None:
        if self.max_entries <= 0 or len(self.entries) <= self.max_entries:
            return
        self.entries.sort(key=lambda e: (e.utility / max(1, e.uses or 1), e.utility, -e.uses), reverse=True)
        self.entries = self.entries[: self.max_entries]

    @staticmethod
    def _canon(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()


def _normalize_profile(profile: str | None) -> str:
    parts = [p.strip() for p in str(profile or "any").split("|") if p.strip()]
    if not parts:
        return "any"
    base = parts[0] if parts[0] in BASE_PROFILES else "any"
    tags = []
    for p in parts[1:]:
        if p in VALID_DIFFICULTIES or p in VALID_BUDGETS:
            tags.append(p)
    return "|".join([base, *tags]) if tags else base


def _profile_match_score(entry_profile: str, query_profile: str | None) -> float:
    ep = _normalize_profile(entry_profile).split("|")
    qp = _normalize_profile(query_profile).split("|")
    if ep[0] == "any":
        return 1.0
    if ep[0] != qp[0]:
        return 0.0
    score = 2.0
    if len(ep) > 1:
        score += sum(0.5 for tag in ep[1:] if tag in qp[1:])
        score -= sum(0.25 for tag in ep[1:] if tag not in qp[1:])
    return score
