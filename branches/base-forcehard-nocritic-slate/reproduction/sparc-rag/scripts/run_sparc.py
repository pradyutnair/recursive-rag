"""Run SPARC-RAG over a dataset shard.

Reads questions from a JSON file (list of {id, question, answer, ...}). Writes:
  <out_dir>/predictions.jsonl
  <out_dir>/summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparc.eval import score_row
from sparc.lm import LMConfig, VLLMClient
from sparc.loop import SparcConfig, run_sparc
from sparc.retriever import DenseRetriever


def _load_questions(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("questions", "data", "items"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    raise ValueError(f"Unrecognized question file shape: {path}")


def _golds(q: dict[str, Any]) -> list[str]:
    out: list[str] = []
    a = q.get("answer")
    if isinstance(a, str) and a:
        out.append(a)
    elif isinstance(a, list):
        out.extend(str(x) for x in a if isinstance(x, str) and x)
    aliases = q.get("answer_aliases") or q.get("aliases") or []
    if isinstance(aliases, list):
        out.extend(str(x) for x in aliases if isinstance(x, str) and x)
    return out or [""]


async def _run_one(
    *,
    lm: VLLMClient,
    retriever: DenseRetriever,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    q: dict[str, Any],
    cfg: SparcConfig,
    pred_path: Path,
    pred_lock: asyncio.Lock,
    progress: dict[str, Any],
) -> dict[str, Any]:
    qid = str(q.get("id", q.get("qid", "")))
    question = str(q.get("question", q.get("question_text", "")))
    golds = _golds(q)
    async with sem:
        try:
            local_lm = VLLMClient(lm.cfg)
            result = await run_sparc(local_lm, retriever, client, question=question, cfg=cfg)
            answer = result.get("answer", "")
            scores = score_row(answer, golds)
            row = {
                "id": qid,
                "question": question,
                "gold": golds,
                "answer": answer,
                "scores": scores,
                "metadata": {
                    "n_rounds": result["n_rounds"],
                    "n_llm_calls": result["n_llm_calls"],
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                    "wall_clock_s": result["wall_clock_s"],
                    "stop_reason": result["stop_reason"],
                    "config": result["config"],
                },
                "rounds": result.get("rounds", []),
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "id": qid,
                "question": question,
                "gold": golds,
                "answer": "",
                "scores": {"em": 0.0, "f1": 0.0, "acc": 0.0},
                "metadata": {"error": f"{type(exc).__name__}: {exc}"},
                "rounds": [],
            }
        async with pred_lock:
            with pred_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress["done"] += 1
            progress["em"] += int(row["scores"]["em"])
            progress["f1_sum"] += float(row["scores"]["f1"])
            progress["tok"] += int(row["metadata"].get("total_tokens", 0) or 0)
            d = progress["done"]
            n = progress["n"]
            if d % max(1, n // 20) == 0 or d == n:
                print(json.dumps({
                    "done": d, "n": n,
                    "em": round(progress["em"] / d, 4),
                    "f1": round(progress["f1_sum"] / d, 4),
                    "avg_tokens": round(progress["tok"] / d, 1),
                    "wall_s": round(time.time() - progress["t0"], 1),
                }), flush=True)
        return row


def _summarize(rows: list[dict[str, Any]], wall_clock_s: float) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    em = sum(r["scores"]["em"] for r in rows) / n
    f1 = sum(r["scores"]["f1"] for r in rows) / n
    acc = sum(r["scores"]["acc"] for r in rows) / n
    tok = sum(int(r["metadata"].get("total_tokens", 0) or 0) for r in rows) / n
    rounds = sum(int(r["metadata"].get("n_rounds", 0) or 0) for r in rows) / n
    wall = sum(float(r["metadata"].get("wall_clock_s", 0.0) or 0.0) for r in rows) / n
    err = sum(1 for r in rows if r["metadata"].get("error"))
    return {
        "n": n,
        "em": round(em, 4),
        "f1": round(f1, 4),
        "acc": round(acc, 4),
        "avg_tokens": round(tok, 1),
        "avg_rounds": round(rounds, 2),
        "avg_wall_s": round(wall, 2),
        "wall_clock_s": round(wall_clock_s, 2),
        "error_count": err,
    }


async def _amain(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    if pred_path.exists():
        pred_path.unlink()

    questions = _load_questions(Path(args.questions))
    if args.n and args.n > 0:
        questions = questions[: args.n]

    cfg = SparcConfig(
        width=args.width,
        max_depth=args.max_depth,
        topk=args.topk,
        enable_thinking=False,
        record_traces=bool(args.record_traces),
    )
    lm_cfg = LMConfig(enable_thinking=False)
    lm = VLLMClient(lm_cfg)
    retriever = DenseRetriever(args.retriever_url)
    sem = asyncio.Semaphore(args.concurrency)
    pred_lock = asyncio.Lock()
    progress = {"done": 0, "em": 0, "f1_sum": 0.0, "tok": 0, "n": len(questions), "t0": time.time()}

    limits = httpx.Limits(max_connections=args.concurrency * 8, max_keepalive_connections=args.concurrency * 4)
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=60.0)
    t0 = time.time()
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        rows = await asyncio.gather(*[
            _run_one(lm=lm, retriever=retriever, client=client, sem=sem,
                     q=q, cfg=cfg, pred_path=pred_path, pred_lock=pred_lock,
                     progress=progress)
            for q in questions
        ])
    summary = _summarize(rows, time.time() - t0)
    summary["config"] = {
        "width": cfg.width, "max_depth": cfg.max_depth, "topk": cfg.topk,
        "concurrency": args.concurrency,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nSUMMARY:", json.dumps(summary, indent=2))


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True, help="Path to JSON or JSONL with question records.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--width", type=int, default=2)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--retriever-url", default="http://node408:8003")
    p.add_argument("--record-traces", action="store_true", default=False)
    return p


def main() -> None:
    asyncio.run(_amain(_argparser().parse_args()))


if __name__ == "__main__":
    main()
