# Adaptive Multi-Agent RAG: Effort-Conditioned Collaborative Search for Multi-Hop QA

## Mission

University of Amsterdam MSc thesis by Pradyut Nair, supervised by Yijia Zheng (Multimedia Analytics Lab).

**Title:** "Retrieval-Augmented Generation with Multi-Agent Collaborative Search"

**Research questions (from project description):**

- Q1. Can multi-agent collaborative search increase the parallelism of iterative RAG, improving both efficiency and overall performance?
- Q2. What collaboration strategies maximize the utilization of collective intelligence in multi-agent RAG systems?
- Q3. Can inference-time scaling laws be observed in multi-agent RAG systems?

## Thesis story (reframed 2026-05-05)

The architecture blends **PlanRAG** (DAG-based decomposition with parallel sub-query execution) and **HERA** (token-aware orchestration with experience libraries and topology mutation). The central empirical question, motivated by Tran & Kiela (2604.02460) and SPARC-RAG (2602.00083), is: **how does multi-agent collaborative search scale with inference-time compute, and which collaboration dimensions matter most?**

Three contributions:

1. **Inference-time scaling-law analysis for multi-agent multi-hop QA (WxD framework).** Systematic Width × Depth scaling sweep on RecRAG: W = per-hop retrieval diversity (MAX_ATTEMPTS), D = system-level iterative refinement (max_critic_retries), plus evidence density (top_k). Map the full Pareto frontier across 36 configs, fit scaling curves, identify optimal allocation. Compare against SAGE, AMAS-PRO, MA-RAG, IRCoT, naive_rag on the same test IDs. This is the first WxD scaling analysis for DAG-based multi-agent RAG under a homogeneous LLM setup. Directly answers Q3 and partly Q2.
2. **Component ablation of multi-agent collaboration mechanisms.** Six targeted ablations (context-aware retrieval, tag substitution, parallel execution, synthesizer, critic, single-agent at matched budget) reveal which collaboration strategies contribute to quality and on which question profiles. The single-agent matched-budget comparison directly addresses the Tran & Kiela question. This answers Q1 and Q2.
3. **Gradient-free prompt optimization for multi-agent orchestration.** GEPA (reflective prompt evolution) optimizes planner, synthesizer, and investigator prompts jointly. TF-GRPO builds an experience library of reusable strategy insights. Both are training-time only. The contribution: showing that gradient-free optimization meaningfully improves multi-agent RAG over hand-written prompts.

**Key framing:** This is an empirical analysis thesis, not a SOTA-chasing thesis. Mixed or negative results (e.g., "multi-agent helps on bridge questions but hurts on simple ones", "W>2 has diminishing returns") are valid contributions if cleanly analyzed. HERA uses GPT-4o-mini for orchestration; we use homogeneous Qwen3-14B throughout, which is a fairer test of whether multi-agent collaboration itself helps vs just using a stronger model.

## Current status (as of 2026-05-05)

**Phase 0 complete: RecRAG force-hard-no-critic selected as thesis base system.** Next step: Phase 1 WxD scaling sweep on 200q MuSiQue. See "System Selection" section below for the decision rationale and "Thesis Experiment Plan" for the full 6-phase roadmap.

Prior work (GEPA v4, GRPO v4, cascade v5, adaptive routing) remains available as reference and potential contributions, but the thesis now restarts from the force-hard-no-critic base to build a clean experimental narrative: scaling → ablation → optimization → fine-tuning.

Key recent code changes (already applied):
- `MAX_ATTEMPTS` bumped from 3 to 5 in `tools.py` and eval/GEPA defaults
- Context-aware retrieval for child nodes in `dag.py` and `sync_pipeline.py`: child node retrieval queries include parent question/answer context for disambiguation
- `_coerce_granularity()` in `tools.py`: extracts year from full dates when question asks "what year"
- Router instructions synced between `adaptive_pipeline.py` (async, used by eval) and `sync_pipeline.py` (sync, used by GEPA)

## Hard constraints (non-negotiable)

- **Homogeneous Qwen3-14B at runtime.** No auxiliary models. No cross-encoder reranker. No different embedder. No LoRA fine-tune. No RL training. The retriever stays at E5-base on wiki18 100w-chunk corpus served at `node408:8003`.
- **No ensembling, pooling, majority voting, best-of-N selection across independent generations.**
- **No use of gold answers, baseline predictions, or any benchmark output as a runtime feature.** The oracle (SAS-correctness label) is training-time supervision only.
- **Full open-domain wiki18 corpus** (~21M chunks). NOT closed gold+distractor settings.
- All test-set comparisons must be on the exact same question IDs as the FlashRAG baselines we already have rescored.

## Compute environment

- Active node: `node409`. `module load cuda12.6/toolkit/12.6` before any GPU command.
- 3x vLLM serving `Qwen/Qwen3-14B` at `localhost:8001`, `localhost:8002`, `localhost:8003`. `max_model_len=16384`, `max_num_seqs=32`, prefix caching enabled.
- Retriever: `node408:8003` POST `/retrieve` with body `{"queries":[...],"topk":N,"mode":"text"}`. Backed by E5-base FAISS over wiki18.
- Python env: `/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python` (dspy 3.2.0 with `dspy.GEPA`).
- API keys at `/local/yzheng/pnair/.env`. `OPENAI_API_KEY` is set; use `openai/gpt-5` for GEPA reflection (~$0.50-1 per full run).
- DSPy cache: set `DSPY_CACHEDIR=/local/yzheng/pnair/workspace/recursive-rag/.dspy_cache`.

## Existing assets

**Stable (reuse as-is):**

- **Pipeline scaffold** at `src/recrag/`: profile classifier (7 buckets), Plan*RAG-style DAG planner with `<A.I.J>` tag substitution, async executor with parallel within-layer execution, per-node investigator with retrieve+extract+rewrite (`_hop_async`, `MAX_ATTEMPTS=5`), synthesizer, critic with topology mutation, citation gate. Sync mirror for GEPA threading. Context-aware retrieval: child nodes inherit parent question/answer context in their retrieval query for entity disambiguation. Granularity coercion: `_coerce_granularity` narrows answers to match question granularity (e.g. extracts year from full date when question asks "what year").
- **Oracle on 1500 fresh questions** in `compiled/oracle/{musique,2wikimultihop,hotpotqa}_fresh_naive/predictions.jsonl`. Per-dataset SAS EM: musique 0.07, 2wiki 0.354, hotpot 0.362. Mean tokens 2325.
- **Stratified train/val** in `data/multidataset/train_v3.json` (240q) and `val_v3.json` (30q).
- **Baselines rescored on exact 1000q test ids**: `results/diagnostics/baselines_rescored.json` for {naive_rag, IRCoT, OPERA, MA-RAG} x {musique, 2wikimultihop, hotpotqa, bamboogle}.

**First-draft components (open for redesign):**

- **GEPA driver** in `src/recrag/gepa/compile.py` + `src/recrag/gepa/metric.py`. Patched dspy `gepa.proposer.merge.find_common_ancestor_pair` for the zero-weight crash bug; preserve this patch.
- **TF-GRPO driver** in `src/recrag/grpo/compile.py` + `src/recrag/grpo/library.py` + `src/recrag/grpo/signatures.py`. HERA-style ADD/MERGE/PRUNE/KEEP experience library.
- **Recovered GEPA program** at `compiled/gepa_v3_recovered_cand13.json`, val EM 0.433 on val_v3 (30q).

## Test sets (fixed; never train on these)

- MuSiQue 1000q: `data/musique/questions_1000_seedfull_combined.json`
- 2WikiMultiHopQA 1000q: `data/2wikimultihop/questions_1000_seed42.json`
- HotpotQA 1000q: `data/hotpotqa/questions_1000_seed42.json`
- Bamboogle 125q OOD: `data/bamboogle/questions_125.json`

## Numeric baseline reference

Open-domain full-wiki18, Qwen3-14B no-think, top-5 retrieval, exact 1000q test IDs:


| method    | MuSiQue EM / F1 / contain / tokens | 2Wiki EM / F1 / contain / tokens | HotpotQA EM / F1 / contain / tokens | Bamboogle EM / F1 / contain / tokens |
| --------- | ---------------------------------- | -------------------------------- | ----------------------------------- | ------------------------------------ |
| naive_rag | 0.053 / 0.130 / 0.080 / 877        | 0.209 / 0.282 / 0.295 / 922      | 0.323 / 0.443 / 0.407 / 885         | 0.160 / 0.309 / 0.192 / 854          |
| IRCoT     | 0.075 / 0.128 / 0.143 / 7896       | 0.159 / 0.256 / 0.559 / 7579     | 0.311 / 0.410 / 0.485 / 6457        | 0.208 / 0.311 / 0.320 / 6775         |
| OPERA     | 0.106 / 0.168 / 0.168 / 3445       | 0.082 / 0.144 / 0.193 / 3673     | 0.143 / 0.231 / 0.293 / 3070        | 0.264 / 0.382 / 0.328 / 2720         |
| MA-RAG    | 0.124 / 0.224 / 0.215 / 9667       | 0.218 / 0.325 / 0.473 / 9416     | 0.244 / 0.364 / 0.426 / 8830        | 0.384 / 0.505 / 0.504 / 6828         |


HERA paper headline (Qwen3-14B + GPT-4o-mini, not directly comparable due to mixed models):


| dataset         | HERA EM | HERA F1 | HERA approx tokens |
| --------------- | ------- | ------- | ------------------ |
| MuSiQue         | 0.272   | 0.358   | ~12-15k            |
| 2WikiMultiHopQA | 0.595   | 0.648   | ~12-15k            |
| HotpotQA        | 0.525   | 0.630   | ~12-15k            |
| Bamboogle       | 0.465   | 0.605   | ~10-12k            |


Our current val_v3 results (30q stratified):


| config                     | EM    | mean tokens | wall clock |
| -------------------------- | ----- | ----------- | ---------- |
| force-hard (always MAS)    | 0.333 | 6,843       | 101s       |
| force-easy (always SAS)    | 0.300 | 2,251       | 39s        |
| recovered cand13 (GEPA v3) | 0.433 | 12,363      | -          |


Our current 100q pilot results (matched IDs, pre-optimization, Qwen3-14B no-think):


| dataset   | RecRAG EM | MA-RAG EM | Opera EM | RecRAG Tok | MA-RAG Tok | Opera Tok | Pareto vs MA-RAG     |
| --------- | --------- | --------- | -------- | ---------- | ---------- | --------- | -------------------- |
| MuSiQue   | 0.250     | 0.150     | 0.120    | 7,356      | 9,978      | 3,550     | RecRAG dominates     |
| HotpotQA  | 0.340     | 0.210     | 0.180    | 6,824      | 9,259      | 3,119     | RecRAG dominates     |
| 2Wiki     | 0.310     | 0.250     | 0.090    | 7,898      | 9,877      | 3,883     | RecRAG dominates     |
| Bamboogle  | 0.320     | 0.360     | 0.260    | 6,702      | 6,902      | 2,732     | On Pareto frontier   |

Pre-optimization, RecRAG Pareto-dominates MA-RAG on 3/4 datasets. Bamboogle gap is -0.04 EM at slightly fewer tokens. Note: Bamboogle results have ~0.04-0.08 run-to-run variance without DSPy cache due to LLM stochasticity.

Pilot prediction files: `results/runs/pilot_v4_nothink_{musique,hotpot,2wiki,bamboogle}/`


## Targets (relative, not absolute)

The thesis claims are relative comparisons, not absolute SOTA numbers.

**Primary targets (must demonstrate on 1000q test sets):**


| claim                                    | what to show                                                                   | on how many datasets |
| ---------------------------------------- | ------------------------------------------------------------------------------ | -------------------- |
| MAS > SAS on hard questions              | force-hard EM > force-easy EM on questions the oracle labels hard, p < 0.05    | >= 3/4               |
| Effort-adaptive Pareto-dominates uniform | routed system achieves comparable EM to force-hard at >= 20% fewer mean tokens | >= 3/4               |
| Effort-adaptive > force-easy             | routed system EM > force-easy EM by >= 3 points                                | >= 3/4               |
| Optimization helps                       | GEPA-optimized > unoptimized seed on val and test, at matched or lower tokens  | >= 3/4               |
| System beats baselines                   | system EM > best FlashRAG baseline (MA-RAG) at comparable or fewer tokens      | >= 2/4               |


**Secondary targets (efficiency analysis):**


| metric                                                              | target                            |
| ------------------------------------------------------------------- | --------------------------------- |
| Mean tokens on easy-routed questions                                | <= 4,000                          |
| EM on easy-routed questions                                         | >= naive_rag EM on same questions |
| Easy-route fraction roughly matches oracle distribution per dataset | within 10pp                       |


**Ablation targets (paired bootstrap p < 0.05):**


| contrast                                   | what to show                                               |
| ------------------------------------------ | ---------------------------------------------------------- |
| oracle-supervised router vs random routing | oracle routing achieves higher EM at matched token budget  |
| oracle-supervised router vs no-oracle GEPA | oracle signal is load-bearing (higher EM or better Pareto) |
| with experience library vs without         | library contributes positively on at least 2/4 datasets    |
| with critic vs without critic              | critic contributes positively on hard questions            |


## Reward design (training-time only)

```
quality   = 1.0*EM + 0.5*F1 + 0.3*contain + 0.4*grounded + 0.2*shape_match
efficiency = exp(-total_tokens / 8000)
composite = quality * efficiency^alpha    with alpha = 0.3

oracle_bonus(easy, em, topology, tokens, naive_tokens):
  if oracle_easy:
      if topology == easy_lane and em == 1: +0.5
      if em == 0:                          : -0.5
      if em == 1 and tokens > 3*naive_tokens: -0.3
  else (oracle_hard):
      if em == 1                           : +0.8
      if topology == easy_lane and em == 0 : -0.4

final_reward = composite + oracle_bonus
```

Weights are starting guesses; sweep if needed.

## Architecture

The pipeline exists at `src/recrag/adaptive_pipeline.py` (async) and `src/recrag/sync_pipeline.py` (sync for GEPA). Flow: profile classify -> router -> (easy: single investigator | hard: planner -> DAG executor -> synthesizer -> critic) -> citation gate.

The router is the effort-conditioning mechanism. It takes `{question, profile, experience}`, returns `{"route": "easy|hard", "reason": "..."}`. Easy lane runs a single investigator with `MAX_ATTEMPTS=5`. Hard lane runs the full DAG pipeline.

GEPA optimizes four named predictors: `router`, `planner`, `synthesizer`, `critic`.

## Optimization layer: GEPA + TF-GRPO with W&B logging

### W&B integration (mandatory for all optimization runs)

Every GEPA and TF-GRPO run must log to Weights & Biases for checkpointing and analysis.

**GEPA W&B logging:**

- Project: `recrag-gepa`
- Log per metric call: `{call_idx, question_id, dataset, profile, route, topology, em, f1, contain, total_tokens, composite_reward, oracle_bonus, final_reward}`
- Log per candidate evaluation: `{candidate_idx, val_em, val_mean_tokens, val_composite, prompts_hash, prompt_lengths}`
- Log per generation: `{gen_idx, best_val_em, best_val_tokens, pareto_size, num_candidates}`
- Artifact: save the best program as a W&B artifact after every generation, so no progress is lost on crashes
- Config: record all hyperparameters (alpha, token_T, oracle bonus weights, reflection LM, seed program path)

**TF-GRPO W&B logging:**

- Project: `recrag-grpo`
- Log per question rollout group: `{question_id, profile, group_size, rewards, best_reward, worst_reward, spread, has_mixed_outcomes}`
- Log per library update: `{epoch, batch_idx, library_size, ops_applied, new_entries, modified_entries, deleted_entries}`
- Artifact: save the experience library JSON after every batch, so any crash preserves the latest library state
- Config: record group_size, epochs, batch_size, reflection LM, oracle bonus weights

**Checkpoint discipline:**

- GEPA: `compiled/gepa_<run>/gen_<N>.json` saved locally after every generation, plus W&B artifact
- TF-GRPO: `compiled/grpo_<run>/epoch_<E>_batch_<B>.json` saved locally after every batch, plus W&B artifact
- Always use `--log-dir` for local checkpoints AND W&B for remote backup
- On crash recovery: load the latest W&B artifact or local checkpoint, whichever is newer

### GEPA driver

Current driver at `src/recrag/gepa/compile.py`. Key decisions to make:

- **Reflection LM**: use `openai/gpt-5` for stronger prompt rewrites
- **Module independence vs co-evolution**: consider co-evolving router + planner since their decisions are coupled, or evolving in stages (router first, then planner, then synth+critic)
- **Seed diversity**: seed from `compiled/gepa_v3_recovered_cand13.json` for planner/synth/critic; hand-write router seed based on HERA's chain-disambiguation heuristics
- **Metric calls**: start with 800, plot val score vs calls used, stop at plateau

### TF-GRPO driver

Current driver at `src/recrag/grpo/compile.py`. Key decisions:

- **Group size**: start with G=4
- **Library cap**: max 30 entries, force pruning
- **Tagging**: entries tagged `[easy|hard]` x `[profile]`
- **Update frequency**: after every batch of 10 questions

### Optimization invariants

- Reward must balance quality and token cost
- Oracle bonus must be present (otherwise contribution 1 cannot be ablated)
- The effort-conditioning signal (router's decision) must drive topology selection
- Modules to optimize: at minimum router, planner, synthesizer, critic

## Lessons from prior work (avoid these)

1. Do not run synth or critic in think mode. No-think Qwen3-14B is 5-10x faster and sufficient.
2. Do not skip critic based purely on planner-emitted topology. Use the heuristic profile classifier as hard floor.
3. Always launch optimization with `--log-dir` AND W&B logging. The previous GEPA run lost 7h of compute by crashing without checkpoints.
4. Preserve the patched `gepa.proposer.merge.find_common_ancestor_pair` (zero-weight bug fix).
5. Do not use `asyncio.to_thread` directly. Use `recrag.aio.to_thread`.
6. Do not run 1000q test sets without first passing val_v3 (30q) and a 100q stratified pilot.

## System selection decision (2026-05-05)

**Decision: RecRAG force-hard-no-critic** is the thesis base system.

Three candidate systems evaluated on wiki18 corpus, Qwen3-14B no-think, same retriever:

| system | MuSiQue EM / tok | 2Wiki EM / tok | HotpotQA EM / tok | Bamboogle EM / tok |
| --- | --- | --- | --- | --- |
| RecRAG force-hard-no-critic | 0.170 / 6989 | 0.308 / 8138 | 0.323 / 6051 | 0.312 / 4701 |
| AMAS-PRO strict verified-SAS | 0.193 / ~10.5k | 0.414 / ~12k | 0.429 / ~10k | 0.400 / ~8k |
| SAGE static_multi | 0.154 / 7013 | 0.240 / 6988 | 0.370 / 6408 | 0.416 / 11443 |
| MA-RAG (FlashRAG baseline) | 0.124 / 9667 | 0.218 / 9416 | 0.244 / 8830 | 0.384 / 6828 |

**Why RecRAG:**

1. **Richest WxD scaling surface.** MAX_ATTEMPTS per node (W=1,2,3,5) gives per-hop retrieval diversity — the real SPARC-RAG W analog, where each attempt is a full retrieve→extract→rewrite cycle producing diverse query reformulations. max_critic_retries (D=0,1,2) gives system-level iterative refinement. top_k (3,5,10) gives evidence density. SAGE's `max_concurrent_investigators` is a latency knob (same total work, executed faster), not a scaling axis. SAGE's `max_queries_per_entity` is a weaker W analog (currently fixed at 1; increasing it adds retrieval queries but without the rewrite diversity of RecRAG's attempts). AMAS-PRO is a linear chain with no width dimension.
2. **Best Pareto efficiency.** RecRAG achieves higher or comparable EM to SAGE on 2/4 datasets (MuSiQue +0.016, 2Wiki +0.068) and is cheaper on 3/4 (only HotpotQA: 6051 vs 6408 tokens, comparable). SAGE's Bamboogle advantage (+0.104 EM) comes at 2.4× token cost (11443 vs 4701). AMAS-PRO has the highest EM across the board but at 1.5-1.7× token cost.
3. **Genuine multi-agent collaboration to study.** DAG topology with parallel within-layer execution, context-aware child retrieval via `<A.I.J>` tag substitution, synthesizer aggregation — all individually ablatable for Q2. AMAS-PRO's "collaboration" is 92-98% linear MAS with no structural diversity. SAGE has parallel investigators but no inter-agent info flow within a wave (blackboard is read between waves, not within).
4. **Supervisor endorsement.** Supervisor feedback explicitly recommended RecRAG as thesis scaffold.

**Note:** GEPA and TF-GRPO can be integrated into any of the three systems; portability is not a differentiator.

SAGE and AMAS-PRO results are included as baselines in the comparison table.

**Remote workspace paths (node409):**
- RecRAG: `/local/yzheng/pnair/workspace/recursive-rag`
- AMAS-PRO: `/local/yzheng/pnair/workspace/adaptive-mas`
- SAGE: `/local/yzheng/pnair/workspace/tmp/04-sage-autonomous`


## Thesis experiment plan (revised 2026-05-05)

All phases build on the RecRAG force-hard-no-critic base. Validate on 200q MuSiQue before scaling to full 1000q×3+125q. Thesis submission deadline: June 19, 2026.


### Phase 0: Base system selection — DONE

- [x] Evaluated RecRAG force-hard-no-critic, AMAS-PRO, SAGE on wiki18 with Qwen3-14B
- [x] Confirmed RecRAG Pareto-dominates MA-RAG on 3/4 datasets (100q pilot)
- [x] Full 1000q force-hard-no-critic results collected (see system selection table)
- [x] Supervisor feedback endorses RecRAG as thesis scaffold
- [x] GEPA and TF-GRPO already integrated in RecRAG codebase
- [x] vLLM servers live on node409 (ports 8001-8003), retriever on node408:8003


### Phase 1: WxD scaling sweep — SPARC-RAG style (~2-3 days compute)

Motivated by SPARC-RAG (2602.00083): systematic Width × Depth scaling analysis for multi-agent RAG.

**Scaling grid:**

| axis | parameter | values | interpretation |
| --- | --- | --- | --- |
| W (width) | MAX_ATTEMPTS per node | {1, 2, 3, 5} | Per-hop retrieval diversity: each attempt is an independent retrieve→extract→rewrite cycle |
| D (depth) | max_critic_retries | {0, 1, 2} | System-level iterative refinement: critic rejects → planner retries with feedback |
| Evidence | top_k | {3, 5, 10} | Retrieval density per search |

Full grid: 4 × 3 × 3 = 36 configs.

**Execution plan:**

- [ ] Run all 36 configs on 200q MuSiQue (stratified subset)
- [ ] Plot EM vs mean_tokens for each config → identify Pareto frontier
- [ ] Identify best W, best D, best top_k via marginal contribution analysis
- [ ] Select top 3-5 Pareto-efficient configs for full evaluation
- [ ] Run selected configs on full 1000q × {MuSiQue, 2Wiki, HotpotQA} + 125q Bamboogle
- [ ] Fit scaling curves: EM = f(tokens) per dataset, report slopes
- [ ] Statistical tests: paired bootstrap for adjacent configs on Pareto frontier

**Gate:** At least one config beats force-hard-no-critic baseline (W=5, D=0, top_k=5) on EM or Pareto-efficiency on 200q MuSiQue. If no config improves, the scaling analysis itself is the contribution (negative results are valid).

**Contribution:** Answers Q3 (inference-time scaling laws in multi-agent RAG). Reports the first WxD scaling analysis for DAG-based multi-agent RAG under a homogeneous LLM setup.


### Phase 2: Component ablations on 200q MuSiQue (~1-2 days compute)

Starting from the best WxD config identified in Phase 1, ablate individual components:

| ablation | what changes | expected effect |
| --- | --- | --- |
| No context-aware retrieval | Child nodes don't inherit parent Q/A in retrieval queries | Entity disambiguation degrades on bridge questions |
| No tag substitution | Remove `<A.I.J>` parent→child info flow | Sub-queries lose grounding, hallucinate entities |
| Sequential execution | Disable parallel within-layer execution | Same EM, higher latency (measures collaboration overhead) |
| No synthesizer | Direct concatenation of node answers instead of LLM synthesis | Coherence drops on complex DAG topologies |
| Critic on (D>0 only) | Re-enable critic at best D setting | Measures critic's marginal quality contribution |
| Single-agent matched budget | Single investigator with token budget = MAS mean tokens | The Tran & Kiela (2604.02460) question: is MAS better than SAS at matched compute? |

**Execution plan:**

- [ ] Run each ablation on 200q MuSiQue with best WxD config
- [ ] Compute paired EM deltas + 95% bootstrap CIs for each ablation
- [ ] Per-profile breakdown: 7 profiles × 6 ablations marginal contribution table
- [ ] Write up component contribution analysis

**Gate:** At least 3/6 ablations show statistically significant EM change (p < 0.05).

**Contribution:** Answers Q1 (where does parallel collaboration help?) and Q2 (which collaboration strategies contribute most?).


### Phase 3: GEPA prompt optimization (~6-10h compute)

Gradient-free prompt evolution on the best WxD config from Phase 1.

**Execution plan:**

- [ ] Validate GEPA on 200q MuSiQue: 400 metric calls, seed from best Phase 1 config prompts
- [ ] If 200q val shows improvement (EM +3pp or token savings ≥15%), scale to full:
  ```
  python -m recrag.gepa.compile \
    --questions data/multidataset/train_v3.json \
    --valset data/multidataset/val_v3.json \
    --reflection-lm openai/gpt-5 \
    --reflection-max-tokens 16000 \
    --max-metric-calls 800 \
    --num-threads 6 \
    --log-dir compiled/gepa_v6_logs \
    --out compiled/gepa_v6.json \
    --wandb-project recrag-gepa
  ```
- [ ] Evaluate GEPA-optimized program on val_v3 (30q), then 100q pilot, then full 1000q×3+125q
- [ ] Compare GEPA-optimized vs unoptimized seed: EM, F1, mean_tokens on all datasets
- [ ] Analyze prompt evolution: length drift, semantic changes across generations

**Gate:** GEPA-optimized > unoptimized on val EM at matched or lower tokens.

**Fallback:** If GEPA doesn't converge, freeze planner/synth from seed and evolve only retrieval/extraction prompts. Report optimization dynamics even if final EM is flat (characterizing the search landscape is itself a contribution).

**Contribution:** Shows gradient-free optimization meaningfully improves multi-agent RAG prompts.


### Phase 4: TF-GRPO experience library (~5-8h compute)

Build an experience library on the GEPA-optimized program from Phase 3.

**Execution plan:**

- [ ] Validate TF-GRPO on 200q MuSiQue: 2 epochs, group_size=4, library_cap=20
- [ ] If 200q shows improvement, scale to full:
  ```
  python -m recrag.grpo.compile \
    --questions data/multidataset/train_v3.json \
    --program compiled/gepa_v6.json \
    --reflection-lm openai/gpt-5 \
    --group-size 4 \
    --epochs 3 \
    --batch-size 10 \
    --library-cap 30 \
    --out-json compiled/grpo_v6_library.json \
    --out-txt compiled/grpo_v6_library.txt \
    --checkpoint-dir compiled/grpo_v6_checkpoints \
    --wandb-project recrag-grpo
  ```
- [ ] Evaluate GEPA + library vs GEPA-only on full test sets
- [ ] Analyze library: entry distribution by profile/difficulty, usage frequency, quality impact

**Gate:** GEPA + library ≥ GEPA-only on val EM, or matches at fewer tokens.

**Fallback:** If library hurts, note as negative result. Experience libraries may not help under homogeneous-model constraint (unlike HERA which uses GPT-4o-mini for orchestration).


### Phase 5: Optional DPO/GRPO fine-tuning (stretch goal, ~1 week)

Only if Phases 1-4 complete with time remaining before thesis deadline (June 19, 2026).

- [ ] Collect preference pairs from GEPA rollouts: (winning prompt, losing prompt) per question
- [ ] DPO fine-tune Qwen3-14B on preference pairs (LoRA, 1 epoch)
- [ ] Evaluate fine-tuned model on full test sets
- [ ] Compare: fine-tuned + GEPA vs inference-only GEPA

**Gate:** Fine-tuned model shows ≥5pp EM improvement over inference-only on at least 2/4 datasets.


### Phase 6: Thesis writing (~2-3 weeks, overlapping with Phases 3-5)

- [ ] Related Work: multi-agent RAG, inference-time scaling, gradient-free optimization
- [ ] Method: RecRAG architecture, WxD scaling framework, GEPA/TF-GRPO integration
- [ ] Experimental Setup: datasets, baselines, evaluation protocol, compute
- [ ] Results: scaling curves, ablation tables, optimization trajectories
- [ ] Analysis: per-profile breakdowns, qualitative examples, comparison with SAGE/AMAS-PRO
- [ ] Conclusion: answers to Q1-Q3, limitations, future work


## Prior work archive (experiments from 2026-05-03 to 2026-05-04)

These experiments were run under an earlier adaptive-routing framing. Key results preserved for reference and potential reuse.

<details>
<summary>Click to expand prior experiment results</summary>

**Force-hard-no-critic 1000q (base system):**

| dataset | EM | tokens |
| --- | --- | --- |
| MuSiQue | 0.170 | 6989 |
| 2Wiki | 0.308 | 8138 |
| HotpotQA | 0.323 | 6051 |
| Bamboogle | 0.312 | 4701 |

Run: `results/runs/test_forcehard_nocritic_20260504`

**Adaptive routing 1000q:**

| dataset | EM | tokens |
| --- | --- | --- |
| MuSiQue | 0.158 | 8004 |
| 2Wiki | 0.268 | 7313 |
| HotpotQA | 0.332 | 5481 |
| Bamboogle | 0.264 | 5163 |

Run: `results/runs/test_v4_default_20260504`

**Force-easy (SAS) 1000q:**

| dataset | EM | tokens |
| --- | --- | --- |
| MuSiQue | 0.062 | 2380 |
| 2Wiki | 0.239 | 2300 |
| HotpotQA | 0.297 | 1810 |
| Bamboogle | 0.112 | 1743 |

Run: `results/runs/test_forceeasy_20260504`

**Force-hard (with critic) 1000q:**

| dataset | EM | tokens |
| --- | --- | --- |
| MuSiQue | 0.171 | 7624 |
| 2Wiki | 0.305 | 8852 |
| HotpotQA | 0.328 | 6663 |
| Bamboogle | 0.312 | 5273 |

Run: `results/runs/test_forcehard_20260504`

**GEPA v4:** Best optimizer val composite 0.2891 (780 metric calls). Val EM 0.3333 (missed gate). Cand13 recovery val EM 0.3667. W&B run `38a33711`.

**GRPO v4:** Stopped at epoch_0_batch_3 (18 entries). Val EM 0.3667 at higher token cost than cand13 alone.

**Random routing:** MuSiQue 0.162, 2Wiki 0.275, HotpotQA 0.306, Bamboogle 0.272. Adaptive beats random significantly only on HotpotQA (p=0.018).

**Cascade v5:** GEPA v5 W&B run `icq3rthv`. Early signal: iteration 9 val score 0.4344. Status unknown.

**Max-searches=3 sweep:** Val EM 0.3333 with fewer tokens than ms=4/5. Full run launched in tmux `forcehard_ms3_20260504`.

</details>

## Reference papers

- **SPARC-RAG** (Width × Depth scaling for RAG, DPO fine-tuning): [https://arxiv.org/abs/2602.00083](https://arxiv.org/abs/2602.00083)
- **PlanRAG** (DAG decomposition, parallel sub-query execution): [https://arxiv.org/abs/2410.20753](https://arxiv.org/abs/2410.20753)
- **HERA** (token-aware orchestration, experience library, topology mutation): [https://arxiv.org/abs/2604.00901](https://arxiv.org/abs/2604.00901)
- **Tran and Kiela.** (single-agent vs multi-agent at equal budgets, the efficiency question): [https://arxiv.org/abs/2604.02460](https://arxiv.org/abs/2604.02460)
- TF-GRPO (gradient-free GRPO, semantic advantage): [https://arxiv.org/abs/2510.08191](https://arxiv.org/abs/2510.08191)
- GEPA (reflective prompt evolution with Pareto frontier): [https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)
- MA-RAG (multi-agent CoT for RAG): [https://arxiv.org/abs/2505.20096](https://arxiv.org/abs/2505.20096)

## What success looks like

The thesis is complete when ALL of these are true:

- **WxD scaling analysis published** (Phase 1): Pareto frontier plots per dataset showing EM vs tokens across the W×D×top_k grid. Fitted scaling curves with slopes. At least one non-trivial finding (e.g., W=2 is optimal, D>0 has diminishing returns, top_k saturates at 5).
- **Component ablation table** (Phase 2): 6 ablations × 200q MuSiQue with paired EM deltas and 95% bootstrap CIs. At least 3/6 show statistically significant effect (p < 0.05).
- **Statistical significance**: paired bootstrap p < 0.05 vs naive_rag and MA-RAG on EM, on at least 3/4 datasets with the best WxD config.
- **Optimization contribution shown** (Phase 3): GEPA-optimized > unoptimized seed on val and test, at matched or lower tokens.
- **Per-profile breakdown**: 7 profiles × key ablations, showing where multi-agent collaboration helps most.
- **Inference-time scaling-law plot** published with fitted slopes per system (RecRAG configs, SAGE, AMAS-PRO, MA-RAG, IRCoT, naive_rag).
- **Q1 answered** via component ablation — where does parallel within-layer execution and context-aware retrieval help?
- **Q2 answered** via WxD analysis — which scaling dimension (width/diversity vs depth/refinement vs evidence density) contributes most to quality?
- **Q3 answered** via scaling-law plot — is there a predictable EM-vs-tokens relationship in multi-agent RAG?

If HERA-level absolute numbers are not reached, the thesis acknowledges this as a model-gap effect (homogeneous Qwen3-14B vs HERA's GPT-4o-mini orchestration) and positions the work as a controlled study of multi-agent collaboration under a strictly homogeneous setup.