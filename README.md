# Recursive RAG

Adaptive multi-hop QA pipeline: profile-aware Plan*RAG-style DAG planner → per-node retrieval/extraction sub-agent → synthesizer → critic with topology mutation, on top of a homogeneous Qwen3-14B backbone with no API dependency at runtime. Compile-time uses GEPA (gpt-5 reflection) and TF-GRPO with an SAS-lane oracle.

## Repository layout

```
recursive-rag/
├── src/recrag/                Pipeline + GEPA + TF-GRPO + oracle + experience library
│   ├── adaptive_pipeline.py     Async DAG executor + critic + topology mutation
│   ├── sync_pipeline.py         Sync mirror used by GEPA threading
│   ├── dag.py, profile.py       Plan*RAG-style DAG schema + heuristic profile classifier
│   ├── tools.py, retriever.py   Per-node retrieve + extract + rewrite loop, node408 client
│   ├── metric.py                Composite Pareto reward + oracle-routing bonus
│   ├── oracle.py                SAS-lane oracle lookup
│   ├── grpo/                    TF-GRPO prompts + library + compile loop
│   └── gepa/                    GEPA metric + compile loop
├── scripts/                   Active v3 workflow (see scripts/legacy/ for older flow)
├── compiled/                  GEPA-evolved programs and SAS oracle predictions
│   └── legacy/                  Hand-written experience libraries (pre-v3)
├── data/                      Training pools and test sets per dataset
│   └── multidataset/            Stratified train/val/fresh-pool across MuSiQue/2Wiki/HotpotQA
├── results/
│   ├── baselines/               FlashRAG naive_rag / IRCoT / OPERA / MA-RAG predictions
│   ├── runs/test_v3_cand13/     Current v3 evaluation outputs
│   ├── runs/_archive_pre_v3/    Earlier exploratory runs
│   └── diagnostics/             Forensics, baseline rescore, val verifications
└── recursive_mas/RESEARCH.md   Stage notes
```

## Active scripts (v3 workflow)

| Script | Purpose |
|---|---|
| `scripts/build_fresh_train_pool.py` | Sample fresh train candidates outside the test 1000q sets |
| `scripts/run_sas_oracle.py` | Run our SAS-mode pipeline on the fresh pool to label oracle_easy bits |
| `scripts/build_stratified_trainset.py` | Stratify (dataset × oracle_easy × profile) and emit train/val splits |
| `scripts/forensics.py` | Bucket failure modes from a predictions.jsonl |
| `scripts/rescore_baselines.py` | Rescore FlashRAG baselines on the exact 1000q test ids |
| `scripts/recover_gepa_program.py` | Reconstruct a GEPA candidate program from a crashed run's log |
| `scripts/verify_recovered_program.py` | Sanity-check a recovered/compiled program on val |
| `scripts/eval_on_test.py` | Run the configured pipeline on the 1000q test sets |
| `scripts/aggregate_ablations.py` | Combine multiple run summaries into a Pareto table |
| `python -m recrag.gepa.compile ...` | Run GEPA on the stratified trainset |
| `python -m recrag.grpo.compile ...` | Run TF-GRPO to grow the experience library |

## Quick reproduce (current state)

```bash
# 1. Build fresh train pool (already done)
python scripts/build_fresh_train_pool.py

# 2. Run SAS oracle (already done)
python scripts/run_sas_oracle.py --datasets musique,2wikimultihop,hotpotqa --concurrency 12

# 3. Build stratified trainset (already done)
python scripts/build_stratified_trainset.py --train-per-dataset 80 --val-per-dataset 10 --tag v3

# 4. Run GEPA (with checkpointing + recovery)
python -m recrag.gepa.compile \
    --questions data/multidataset/train_v3.json --valset data/multidataset/val_v3.json \
    --reflection-lm openai/gpt-5 --max-metric-calls 800 --num-threads 6 \
    --log-dir compiled/gepa_v3_logs --out compiled/gepa_v3.json \
    --oracle-naive-dir compiled/oracle

# 5. Eval on test
python scripts/eval_on_test.py \
    --program compiled/gepa_v3_recovered_cand13.json \
    --datasets musique,2wikimultihop,hotpotqa,bamboogle \
    --out-dir results/runs/test_v3_cand13 --concurrency 60
```

## Compute

- Generators: 3× vLLM Qwen3-14B at `localhost:8001/8002/8003`
- Retriever: `node408:8003` (wiki18 FAISS index, top-5)
- Reflection LM (GEPA / TF-GRPO only): `openai/gpt-5`
