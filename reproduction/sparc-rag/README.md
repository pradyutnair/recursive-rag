# SPARC-RAG repro (Qwen3-14B, dense-only wiki18)

Faithful reproduction of Yang et al. 2026 "SPARC-RAG: Adaptive Sequential-Parallel
Scaling with Context Management for Retrieval-Augmented Generation"
(arXiv:2602.00083), restricted to:

- **Model**: `Qwen/Qwen3-14B` (no-think) served via three vLLM replicas at
  `localhost:{8001,8002,8003}` on `node409`.
- **Datasets**: HotpotQA, 2WikiMultiHopQA, MuSiQue (1000q each), Bamboogle (125q).
  Same canonical 1000q test sets used by the recursive-rag baselines.
- **Retriever**: dense-only via the wiki18 retriever at `http://node408:8003`
  (`POST /retrieve`, top-k = 6).
- **Omitted vs paper**: Natural Questions, Elasticsearch BM25, the agentic
  retrieval-method tag, the DPO fine-tuning stage.

## Layout

```
src/sparc/
  lm.py           # OpenAI-compat vLLM client, round-robin over 3 replicas
  retriever.py    # Async POST /retrieve client
  prompts.py      # Prompts adapted from paper Appendix C.1
  agents.py       # QueryRewriter / MemUpdate / Generator / Evaluator / SelectBest / Merge
  loop.py         # Algorithm 1 with explicit width W and depth D
  eval.py         # norm_em / token_f1 / accuracy
scripts/
  run_sparc.py    # Single-config runner over a question file
  run_grid.py     # WxD grid driver
```

## Quickstart on node409

```bash
cd /local/yzheng/pnair/workspace/sparc-rag
# 25q pilot
python scripts/run_sparc.py \
  --questions /local/yzheng/pnair/workspace/tmp/04-sage-autonomous/data/node408_shards/hotpotqa/questions_shard_0.json \
  --out-dir results/pilot/hotpotqa_W2D4 --n 25 --width 2 --max-depth 4 --concurrency 6
```
