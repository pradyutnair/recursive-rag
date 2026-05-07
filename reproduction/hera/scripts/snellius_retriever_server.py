#!/usr/bin/env python
"""Standalone wiki18 retriever server for Snellius.

Loads:
  - intfloat/e5-base-v2 (encoder)
  - FAISS Flat index of wiki18 100w embeddings
  - wiki18 corpus JSONL

Serves POST /retrieve {queries:[str], topk:int, mode:"text"} → results:[[Passage,...]].
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional, Union

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("retriever")

app = FastAPI()
state = {}


class Req(BaseModel):
    queries: Union[list[str], list[list[float]]]
    topk: Optional[int] = 5
    mode: Optional[str] = "text"


@app.get("/health")
def health():
    return {"ok": True, "n_passages": state.get("n_passages", 0)}


@app.post("/retrieve")
def retrieve(req: Req):
    topk = req.topk or 5
    qs = req.queries
    if isinstance(qs[0], str):
        # Encode with e5: prefix "query: "
        prefixed = [f"query: {q}" for q in qs]
        embs = state["encoder"].encode(prefixed, batch_size=32, convert_to_numpy=True,
                                          normalize_embeddings=True)
    else:
        embs = np.asarray(qs, dtype=np.float32)
    embs = embs.astype("float32")
    scores, idx = state["index"].search(embs, topk)
    results = []
    corpus = state["corpus"]
    for q_idx in range(len(qs)):
        per_q = []
        for k in range(topk):
            chunk_idx = int(idx[q_idx][k])
            if chunk_idx < 0 or chunk_idx >= len(corpus):
                continue
            entry = corpus[chunk_idx]
            per_q.append({
                "chunk_id": str(entry.get("id", chunk_idx)),
                "text": entry.get("contents") or entry.get("text") or "",
                "score": float(scores[q_idx][k]),
            })
        results.append(per_q)
    return {"results": results}


def load_corpus(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-path", required=True)
    ap.add_argument("--corpus-path", required=True)
    ap.add_argument("--encoder", default="intfloat/e5-base-v2")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logger.info("Loading FAISS index from %s ...", args.index_path)
    index = faiss.read_index(args.index_path)
    logger.info("Index ntotal = %d", index.ntotal)

    logger.info("Loading corpus from %s ...", args.corpus_path)
    corpus = load_corpus(args.corpus_path)
    logger.info("Corpus loaded: %d entries", len(corpus))

    logger.info("Loading encoder %s on %s ...", args.encoder, args.device)
    enc = SentenceTransformer(args.encoder, device=args.device)
    enc.eval()

    state["index"] = index
    state["corpus"] = corpus
    state["encoder"] = enc
    state["n_passages"] = index.ntotal
    logger.info("Server ready on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
