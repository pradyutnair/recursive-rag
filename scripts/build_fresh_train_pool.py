"""Build fresh train/val candidate pools outside the fixed 1000q test sets.

Outputs one JSON file per dataset:
  data/multidataset/fresh_pool/{musique,2wikimultihop,hotpotqa}.json

MuSiQue uses the local train sample when available. 2Wiki and Hotpot are pulled
from the FlashRAG Hugging Face dataset train splits.
"""
from __future__ import annotations

import argparse
import json
import random
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HF_DATASET = "RUC-NLPIR/FlashRAG_datasets"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_SIZE_URL = "https://datasets-server.huggingface.co/size"

DATASETS = {
    "musique": {
        "local_sources": ["data/musique/musique_train_500.json"],
        "test": "data/musique/questions_1000_seedfull_combined.json",
        "hf_config": "musique",
    },
    "2wikimultihop": {
        "local_sources": [],
        "test": "data/2wikimultihop/questions_1000_seed42.json",
        "hf_config": "2wikimultihopqa",
    },
    "hotpotqa": {
        "local_sources": [],
        "test": "data/hotpotqa/questions_1000_seed42.json",
        "hf_config": "hotpotqa",
    },
}


def _norm_text(value: str) -> str:
    return " ".join(str(value or "").lower().split()).strip()


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return rows if isinstance(rows, list) else []


def _answer(row: dict[str, Any]) -> str:
    if row.get("answer") is not None:
        return str(row.get("answer", ""))
    answers = row.get("golden_answers") or row.get("answers") or []
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(answers or "")


def _normalize_row(row: dict[str, Any], dataset: str, source: str) -> dict[str, Any] | None:
    qid = str(row.get("id", "")).strip()
    question = str(row.get("question", "")).strip()
    answer = _answer(row).strip()
    if not qid or not question or not answer:
        return None
    out = {
        "id": qid,
        "dataset": dataset,
        "question": question,
        "answer": answer,
        "source": source,
    }
    if isinstance(row.get("metadata"), dict):
        out["question_type"] = row["metadata"].get("type", "")
    elif row.get("question_type") is not None:
        out["question_type"] = row.get("question_type", "")
    return out


def _url_json(url: str, timeout: int = 60) -> dict[str, Any]:
    context = None
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, timeout=timeout, context=context) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _hf_num_rows(config: str, split: str) -> int:
    url = HF_SIZE_URL + "?" + urllib.parse.urlencode({"dataset": HF_DATASET})
    obj = _url_json(url)
    split_rows = obj.get("size", {}).get("splits", []) or obj.get("splits", [])
    for item in split_rows:
        if item.get("config") == config and item.get("split") == split:
            return int(item.get("num_rows", 0))
    raise RuntimeError(f"Could not resolve HF row count for {config}/{split}")


def _hf_rows(config: str, split: str, offset: int, length: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({
        "dataset": HF_DATASET,
        "config": config,
        "split": split,
        "offset": offset,
        "length": min(length, 100),
    })
    obj = _url_json(HF_ROWS_URL + "?" + query)
    return [r.get("row", {}) for r in obj.get("rows", [])]


def _collect_local(dataset: str, cfg: dict[str, Any], test_ids: set[str], test_questions: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_questions: set[str] = set()
    for rel in cfg["local_sources"]:
        for row in _load_json(ROOT / rel):
            norm = _normalize_row(row, dataset, rel)
            if not norm:
                continue
            qnorm = _norm_text(norm["question"])
            if norm["id"] in test_ids or qnorm in test_questions:
                continue
            if norm["id"] in seen or qnorm in seen_questions:
                continue
            seen.add(norm["id"])
            seen_questions.add(qnorm)
            rows.append(norm)
    return rows


def _collect_hf(
    dataset: str,
    cfg: dict[str, Any],
    test_ids: set[str],
    test_questions: set[str],
    existing_ids: set[str],
    existing_questions: set[str],
    want: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    total = _hf_num_rows(cfg["hf_config"], "train")
    offsets = list(range(0, total, 100))
    rng.shuffle(offsets)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set(existing_ids)
    seen_questions: set[str] = set(existing_questions)
    for offset in offsets:
        for row in _hf_rows(cfg["hf_config"], "train", offset, 100):
            norm = _normalize_row(row, dataset, f"hf:{HF_DATASET}/{cfg['hf_config']}/train")
            if not norm:
                continue
            qnorm = _norm_text(norm["question"])
            if norm["id"] in test_ids or qnorm in test_questions:
                continue
            if norm["id"] in seen or qnorm in seen_questions:
                continue
            seen.add(norm["id"])
            seen_questions.add(qnorm)
            rows.append(norm)
            if len(rows) >= want:
                return rows
        time.sleep(0.05)
    return rows


def build(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for i, (dataset, cfg) in enumerate(DATASETS.items()):
        test_rows = _load_json(ROOT / cfg["test"])
        test_ids = {str(r.get("id", "")).strip() for r in test_rows}
        test_questions = {_norm_text(str(r.get("question", ""))) for r in test_rows}
        rows = _collect_local(dataset, cfg, test_ids, test_questions)
        if len(rows) < args.per_dataset:
            rows.extend(_collect_hf(
                dataset,
                cfg,
                test_ids,
                test_questions,
                {r["id"] for r in rows},
                {_norm_text(r["question"]) for r in rows},
                args.per_dataset - len(rows),
                args.seed + i * 1009,
            ))
        rows = rows[: args.per_dataset]
        out = out_dir / f"{dataset}.json"
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        row_questions = {_norm_text(r["question"]) for r in rows}
        summary[dataset] = {
            "path": str(out.relative_to(ROOT)),
            "n": len(rows),
            "test_ids": len(test_ids),
            "id_overlap": len({r["id"] for r in rows} & test_ids),
            "question_overlap": len(row_questions & test_questions),
            "sources": sorted({r["source"] for r in rows}),
        }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/multidataset/fresh_pool")
    p.add_argument("--per-dataset", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260502)
    build(p.parse_args())


if __name__ == "__main__":
    main()
