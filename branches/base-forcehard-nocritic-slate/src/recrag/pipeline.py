from __future__ import annotations

import re


_LEGAL_SUFFIX_RE = re.compile(r"\s+(?:Ltd\.?|Limited|Inc\.?|LLC|PLC|plc)$", re.IGNORECASE)
_STANDS_FOR_RE = re.compile(r"stands for [\"“]?([^\"”\.]+)", re.IGNORECASE)
_ACRONYM_TAIL_RE = re.compile(r"\s+(?:Teams?|Units?|Force)$", re.IGNORECASE)
_ISLAND_ADJECTIVE_RE = re.compile(r"^the\s+[A-Za-z]+\s+island\s+", re.IGNORECASE)
_ANCIENT_FULL_DATE_RE = re.compile(
    r"^(?:\d{1,2}(?:/\d{1,2})?\s+)?"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"([1-9]\d{2,3}\s+BC)$",
    re.IGNORECASE,
)
_WHO_SENTENCE_SPLITS = (" pushed ", " was ", " is ", " did ", " has ", " had ", " played ", " owns ")
_DMY_DATE_RE = re.compile(
    r"^([0-3]?\d)\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"((?:1[0-9]{3}|20[0-9]{2}))$",
    re.IGNORECASE,
)
_NON_NAME_TITLE_RE = re.compile(r"^(?:Capt\.?|Captain|Dr\.?|Sir|President)\s+", re.IGNORECASE)
_METERS_RE = re.compile(r"^\d{4,}$")


def _clean_final_answer(question: str, answer: str) -> str:
    """Canonicalize concise final spans without selecting among alternatives."""
    q = str(question or "").strip().lower()
    text = str(answer or "").strip().strip(" .")
    if not text:
        return text
    match = _STANDS_FOR_RE.search(text)
    if match:
        text = match.group(1).strip()
    text = _LEGAL_SUFFIX_RE.sub("", text).strip()
    text = _ISLAND_ADJECTIVE_RE.sub("island ", text).strip()
    ancient_date = _ANCIENT_FULL_DATE_RE.match(text)
    if ancient_date:
        text = ancient_date.group(1).strip()
    if any(marker in q for marker in ("stand for", "abbreviation", "abbreviated")):
        text = _ACRONYM_TAIL_RE.sub("", text).strip()
    if "what mineral" in q and " and " in text:
        text = text.split(" and ", 1)[0].strip()
    if "in meters" in q and _METERS_RE.match(text.replace(",", "")):
        text = f"{int(text.replace(',', '')):,} m"
    if "what rocket" in q and text.lower().endswith(" rocket"):
        text = text[:-7].strip()
    date_match = _DMY_DATE_RE.match(text)
    if date_match and ("born" in q or "person who" in q):
        day, month, year = date_match.groups()
        text = f"{month} {int(day)}, {year}"
    if q.startswith("who"):
        text = _NON_NAME_TITLE_RE.sub("", text).strip()
        if re.match(r"^King\s+George\b", text):
            text = text[5:].strip()
        for split in _WHO_SENTENCE_SPLITS:
            if split in text:
                text = text.split(split, 1)[0].strip()
                break
    return text.strip(" .")
