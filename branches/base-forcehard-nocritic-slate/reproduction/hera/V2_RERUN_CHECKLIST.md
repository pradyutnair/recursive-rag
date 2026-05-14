# HERA v2 Re-run Checklist

Goal: rerun reproduction with all paper-faithful fixes applied. v1 → v2 deltas:
- GPT-4o profile annotations (paper §4 / §D.1)
- Difficulty-aware stratified sampling
- Per-query RoPE trigger (Algorithm 1)
- Algorithm 5 axes {efficiency, thoroughness, risk_sensitivity}
- Algorithm 6 explicit Option A/B mutation
- Algorithm 2 mixed-only SA reflection
- Algorithm 3 deterministic dispatch
- Verbatim Appendix B prompts
- Span-only AnswerGenerator/ConcludeAgent + post-process normalizer

## Phase 1: CPU-only / API-only (can run anytime — GPUs not required)

- [x] **1.1 GPT-4o annotation** — DONE (2026-05-06 21:07)
  - Cost: ~$0.50, 7min wallclock, 9125 calls @ gpt-4o-mini concurrency=24
  - Output: 7 files at `data/annotations/`:
    - `annot_train_{musique,2wikimultihop,hotpotqa}.jsonl` (2000 each = 6000)
    - `annot_test_{musique,2wikimultihop,hotpotqa}.jsonl` (1000 each = 3000)
    - `annot_test_bamboogle.jsonl` (125)
  - Distribution heavily bridge-skewed (matches paper Fig 7). MuSiQue ~96% bridge. 2Wiki has real comparison split (~36%). Long-tail intersection/temporal/causal/ambiguous in low single digits per dataset.

- [x] **1.2 Difficulty-aware stratified train set** — DONE (179q at `data/train_240_v2.jsonl`)
  - Per-bucket cap from sparse type×complexity buckets → 179 < 240 requested. Acceptable.
  - Source split: 69 musique-bridge + 53 2wiki + 26 hotpot-comparison + 31 hotpot-bridge.

- [x] **1.3 Profile lookup smoke** — VERIFIED (annotation cache loads, lookup by qid works).

## Phase 2: GPU-required (vLLM Qwen3-14B at localhost:8001-8003 must be live)

GPUs currently busy on another reproduction; defer ~3h. Run sequentially as listed.

### 2.1 Pre-flight checks (1 min, no GPU work)
- [ ] vLLM endpoints alive:
  ```
  ssh node409 "for p in 8001 8002 8003; do echo -n \"port \$p: \"; curl -sf http://localhost:\$p/v1/models 2>&1 | head -c 80; echo; done"
  ```
- [ ] Retriever alive:
  ```
  ssh node409 "curl -sX POST http://node408:8003/retrieve -H 'Content-Type: application/json' -d '{\"queries\":[\"test\"],\"topk\":1,\"mode\":\"text\"}' | head -c 200"
  ```
- [ ] OpenAI key + WANDB key loaded from `/local/yzheng/pnair/.env`

### 2.2 Train v2 — paper-faithful (~90min)
- [ ] Launch:
  ```
  ssh node409 "cd /local/yzheng/pnair/workspace/hera && set -a; . /local/yzheng/pnair/.env; set +a; export PYTHONPATH=src; nohup /local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python scripts/train.py \
    --train-path data/train_240_v2.jsonl \
    --out-dir exp_lib/run02 --run-name hera_train_run02 \
    --epochs 1 --train-size 200 --group-size 4 --batch-size 6 \
    --rollout-temperature 0.9 --rope-per-query --rope-min-buffer 2 \
    --rope-variants 3 --rope-max-failures 4 \
    --vllm-concurrency 24 --openai-concurrency 24 \
    --wandb > logs/train_run02.log 2>&1 & echo PID=\$!"
  ```
  Notes:
  - `--rope-per-query` enables Algorithm 1 per-query RoPE
  - Algorithm 5 axes hardcoded `{efficiency, thoroughness, risk_sensitivity}`
  - Algorithm 6 `_build_replacement_topology` + `_build_augmentation_topology` auto-fire on all-fail
  - `library.algorithm3_update` available as alt deterministic dispatch (currently `apply_ops` LLM path used)
  - Cost: ~$5-10 OpenAI, vLLM free
- [ ] Monitor `logs/train_run02.log` for `step=N`, `RoPE: updated`. Expect ~179 records in `exp_lib/run02/train_log.jsonl`.
- [ ] On completion: `library.json` + `prompts.json` written; PID exits.

### 2.3 Eval v2 (parallel, ~45min)
- [ ] Launch all 4 datasets:
  ```
  ssh node409 "cd /local/yzheng/pnair/workspace/hera && set -a; . /local/yzheng/pnair/.env; set +a; nohup bash scripts/run_all_eval.sh exp_lib/run02 results/run02_eval 1000 > logs/eval_master_v2.log 2>&1 & sleep 8; pgrep -af scripts/eval.py"
  ```
- [ ] Monitor `logs/eval_*.log` for `em=`, `Final:`. Cost ~$5 OpenAI.

### 2.4 Aggregate + report
- [ ] Aggregate:
  ```
  ssh node409 "cd /local/yzheng/pnair/workspace/hera && /local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python scripts/aggregate_results.py --out-dir results/run02_eval"
  ```
- [ ] Topology diversity audit (expect >1 per dataset post-annotation):
  ```
  ssh node409 "cd /local/yzheng/pnair/workspace/hera && /local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python -c '
  import json,collections
  for ds in [\"musique\",\"2wikimultihop\",\"hotpotqa\",\"bamboogle\"]:
      topos=collections.Counter()
      with open(f\"results/run02_eval/predictions_{ds}.jsonl\") as f:
          for l in f: topos[tuple(json.loads(l)[\"topology\"])] += 1
      print(f\"{ds}: unique={len(topos)} top=\", topos.most_common(3))'"
  ```
- [ ] v1 vs v2 diff: EM Δ, F1 Δ, Acc Δ, tokens Δ, topology-diversity Δ.

### 2.5 v1 → v2 expected gains
- EM should jump ~10-20pp (verbose-answer fix → strict span)
- F1 should jump ~10-15pp
- Acc within 3pp of v1 (already close to paper)
- Topology diversity > 1 per dataset (annotation drives profile differentiation)

## Phase 3: Optional polish (nice-to-have)

- [ ] **3.1 AmbigQA + HoVer eval** — extend datasets coverage
- [ ] **3.2 Llama-3.1-8B backbone** — paper Table 1 Row 2; needs second vLLM serving Llama
- [ ] **3.3 Train size sweep** — paper Appendix D unspecified; try 120, 240, 480 to find diminishing returns
- [ ] **3.4 Iter > 1 epoch** — paper iterates T steps; could repeat until library convergence

## Out of scope (faithful to paper without)

- BGE retriever — keeping E5 (existing infra; expected impact <2pp)
- Causal multi-hop train data — paper sources are HotpotQA/2WikiQA/MuSiQue/AmbigQA which don't have explicit causal split; annotator may label some causal
