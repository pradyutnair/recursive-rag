# Adaptive Multi-Agent RAG: Effort-Conditioned Collaborative Search for Multi-Hop QA

## Mission

University of Amsterdam MSc thesis by Pradyut Nair, supervised by Yijia Zheng (Multimedia Analytics Lab).

**Title:** "Retrieval-Augmented Generation with Multi-Agent Collaborative Search"

**Research questions (from project description):**

- Q1. Can multi-agent collaborative search increase the parallelism of iterative RAG, improving both efficiency and overall performance?
- Q2. What collaboration strategies maximize the utilization of collective intelligence in multi-agent RAG systems?
- Q3. Can inference-time scaling laws be observed in multi-agent RAG systems?

## Thesis story

The architecture blends **PlanRAG** (DAG-based decomposition with parallel sub-query execution) and **HERA** (token-aware orchestration with experience libraries and topology mutation). The central empirical question, motivated by Tran & Kiela (2604.02460), is: **when does multi-agent collaborative search actually help over a single agent at matched compute?**

Three contributions:

1. **Effort-conditioned routing.** An oracle-supervised router predicts question difficulty and allocates compute accordingly: easy questions get a single investigator (cheap, fast); hard questions get the full DAG planner with parallel multi-agent execution, synthesizer, and critic (expensive, thorough). The claim: adaptive effort allocation is Pareto-superior to uniform allocation. Always-SAS wastes accuracy on hard questions; always-MAS wastes tokens on easy ones. The oracle signal (SAS-correctness on held-out questions) is training-time supervision only, never used at inference.
2. **Gradient-free prompt optimization for multi-agent orchestration.** GEPA (reflective prompt evolution) optimizes router, planner, synthesizer, and critic prompts jointly using a composite reward that balances quality (EM/F1/contain) against token cost. TF-GRPO builds an experience library of reusable strategy insights, tagged by question profile and difficulty. Both are training-time only. The contribution: showing that gradient-free optimization meaningfully improves multi-agent RAG over hand-written prompts, and characterizing the optimization dynamics (convergence, prompt drift, library growth).
3. **Inference-time scaling-law analysis for multi-hop QA.** Systematic comparison of five+ systems (naive_rag, IRCoT, OPERA, MA-RAG, our force-easy, our force-hard, our effort-adaptive) on four datasets, all sharing the same retriever, backbone, and test IDs. Plot EM vs mean tokens, fit scaling curves, identify the regime where adaptive multi-agent is optimal. This directly answers Q3 and, combined with the per-question-type ablation, answers Q1 and Q2.

**Key framing:** This is an empirical analysis thesis, not a SOTA-chasing thesis. Mixed or negative results (e.g., "multi-agent helps on bridge questions but hurts on simple ones") are valid contributions if cleanly analyzed. HERA uses GPT-4o-mini for orchestration; we use homogeneous Qwen3-14B throughout, which is a fairer test of whether multi-agent collaboration itself helps vs just using a stronger model.

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

- **Pipeline scaffold** at `src/recrag/`: profile classifier (7 buckets), Plan*RAG-style DAG planner with `<A.I.J>` tag substitution, async executor with parallel within-layer execution, per-node investigator with retrieve+extract+rewrite (`_hop_async`, `MAX_ATTEMPTS=3`), synthesizer, critic with topology mutation, citation gate. Sync mirror for GEPA threading.
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

The router is the effort-conditioning mechanism. It takes `{question, profile, experience}`, returns `{"route": "easy|hard", "reason": "..."}`. Easy lane runs a single investigator with `MAX_ATTEMPTS=3-4`. Hard lane runs the full DAG pipeline.

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

## Experiment plan with detailed checkpoints

### Phase 0: Orient and verify (1h)

- Confirm vLLM servers are live on node409 (ports 8001, 8002, 8003)
- Confirm retriever is live on node408:8003
- Run recovered cand13 against val_v3, confirm EM ~0.43 reproduces
- Run force-hard on val_v3, confirm EM ~0.33 reproduces
- Run force-easy on val_v3, confirm EM ~0.30 reproduces
- **Gate:** all three reproduce within 2pp. If not, debug before proceeding.

### Phase 1: W&B integration + GEPA/GRPO instrumentation (3-4h)

- `pip install wandb` in the venv, `wandb login`
- Add W&B logging to `src/recrag/gepa/compile.py`: per-call metrics, per-candidate metrics, per-generation summary, artifact saving
- Add W&B logging to `src/recrag/grpo/compile.py`: per-group metrics, per-library-update metrics, artifact saving
- Smoke test: run GEPA with `--max-metric-calls 20` and verify W&B dashboard shows metrics and artifacts
- Smoke test: run TF-GRPO with `--epochs 1 --batch-size 5` and verify W&B dashboard shows library evolution
- **Gate:** W&B dashboards show clean logging, artifacts downloadable, no crashes.

### Phase 2: GEPA optimization (6-10h compute, monitor via W&B)

- Launch GEPA run with full config:
  ```
  python -m recrag.gepa.compile \
    --questions data/multidataset/train_v3.json \
    --valset data/multidataset/val_v3.json \
    --reflection-lm openai/gpt-5 \
    --reflection-max-tokens 16000 \
    --max-metric-calls 800 \
    --num-threads 6 \
    --log-dir compiled/gepa_v4_logs \
    --out compiled/gepa_v4.json \
    --oracle-naive-dir compiled/oracle \
    --oracle-datasets musique,2wikimultihop,hotpotqa \
    --wandb-project recrag-gepa \
    --track-stats
  ```
- Monitor W&B: check val_em curve every ~50 metric calls. If plateaued by call 400, stop early and save.
- After completion: evaluate best candidate on val_v3. Record EM, F1, contain, mean_tokens.
- If val_em < 0.35 (worse than force-hard), diagnose:
  - Check if router is collapsing to always-easy or always-hard
  - Check if planner is collapsing to single_hop on bridge questions
  - Check prompt lengths for bloat
  - Consider staged evolution (fix router, evolve planner only)
- **Gate:** GEPA-optimized program achieves val_em >= 0.35 at mean_tokens <= 10,000.
- **Fallback:** If GEPA does not converge after 2 attempts, freeze planner/synth/critic from cand13 and only GEPA-evolve the router. This still gives contribution 1 (routing) even without contribution 2 (full optimization).

### Phase 3: TF-GRPO experience library (5-8h compute, monitor via W&B)

- Launch TF-GRPO using the GEPA-optimized program as the base:
  ```
  python -m recrag.grpo.compile \
    --questions data/multidataset/train_v3.json \
    --program compiled/gepa_v4.json \
    --reflection-lm openai/gpt-5 \
    --oracle-naive-dir compiled/oracle \
    --group-size 4 \
    --epochs 3 \
    --batch-size 10 \
    --library-cap 30 \
    --out-json compiled/grpo_v4_library.json \
    --wandb-project recrag-grpo
  ```
- Monitor W&B: check library size growth, mean reward per epoch, ops distribution
- After completion: evaluate program + library on val_v3
- Compare val results: (seed) vs (GEPA-only) vs (GEPA + library)
- **Gate:** GEPA + library >= GEPA-only on val_em, or at least matches at fewer tokens.
- **Fallback:** If library hurts, run test evals without it and note as negative result in thesis.

### Phase 4: Pilot eval on 100q stratified per dataset (2h)

- Build 100q stratified pilot per dataset (25q each from musique, 2wiki, hotpot, bamboogle)
- Run effort-adaptive system on each pilot
- Compute EM, F1, contain, mean_tokens, route_distribution per dataset
- Compare against baselines on same 100q IDs
- **Gate:** system EM > MA-RAG on at least 2/4 pilot datasets at comparable or fewer tokens. If not, diagnose per-profile failures before proceeding.

### Phase 5: Full eval on 1000q x 4 datasets (6-8h)

- Run effort-adaptive on MuSiQue 1000q -> `results/runs/test_v4/musique/`
- Run effort-adaptive on 2Wiki 1000q -> `results/runs/test_v4/2wikimultihop/`
- Run effort-adaptive on HotpotQA 1000q -> `results/runs/test_v4/hotpotqa/`
- Run effort-adaptive on Bamboogle 125q -> `results/runs/test_v4/bamboogle/`
- Compute all metrics. Log to W&B as a summary table.
- Run force-hard on all datasets (if not already done) -> `results/runs/test_forcehard/`
- Run force-easy on all datasets (if not already done) -> `results/runs/test_forceeasy/`

### Phase 6: Ablations on same 1000q test IDs (8-12h)

- **No-critic**: run system with critic disabled
- **No-experience-library**: run system without library
- **Random routing**: route easy/hard at observed oracle mix rate, randomly
- **No-oracle GEPA**: re-run GEPA without oracle bonus in reward, evaluate that program
- **Force-easy (SAS baseline)**: already done in Phase 5
- **Force-hard (MAS upper bound)**: already done in Phase 5
- All ablations on same question IDs, same retriever, same model
- Compute paired bootstrap CIs for all pairwise comparisons
- Per-profile breakdown: 7 profiles x each ablation condition

### Phase 7: Analysis and writeup (5-7 days)

- **Pareto plot per dataset**: EM vs mean tokens for all baselines + all system variants. One figure per dataset.
- **Inference-time scaling-law plot**: log-log EM vs tokens across all systems. Fit slopes. Identify crossover points where more compute stops helping.
- **Per-question-type ablation table**: 7 profiles x {router, planner, parallel exec, critic, library} = 35 cells with marginal EM contribution and 95% CI.
- **Routing analysis**: confusion matrix (router prediction vs oracle), EM as function of router confidence.
- **Effort-conditioning analysis**: EM and tokens broken down by route (easy vs hard), compared against force-easy and force-hard.
- **Optimization trajectory**: GEPA val_em vs metric calls (from W&B), library size vs epoch, prompt length evolution.
- **Qualitative examples**: 3-5 success cases (routing helped), 3-5 failure cases (routing hurt), with full trace.
- **Write thesis chapters**: Related Work, Method, Experimental Setup, Results, Analysis, Conclusion.

## Reference papers

- **PlanRAG** (DAG decomposition, parallel sub-query execution): [https://arxiv.org/abs/2410.20753](https://arxiv.org/abs/2410.20753)
- **HERA** (token-aware orchestration, experience library, topology mutation): [https://arxiv.org/abs/2604.00901](https://arxiv.org/abs/2604.00901)
- **Tran and Kiela.** (single-agent vs multi-agent at equal budgets, the efficiency question): [https://arxiv.org/abs/2604.02460](https://arxiv.org/abs/2604.02460)
- TF-GRPO (gradient-free GRPO, semantic advantage): [https://arxiv.org/abs/2510.08191](https://arxiv.org/abs/2510.08191)
- GEPA (reflective prompt evolution with Pareto frontier): [https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)
- MA-RAG (multi-agent CoT for RAG): [https://arxiv.org/abs/2505.20096](https://arxiv.org/abs/2505.20096)

## What success looks like

The thesis is complete when ALL of these are true:

- **Pareto plot** per dataset published showing effort-adaptive system Pareto-dominates at least one baseline (better EM at fewer tokens) on at least 3/4 datasets.
- **Statistical significance**: paired bootstrap p < 0.05 vs naive_rag and MA-RAG on EM, on at least 3/4 datasets.
- **Effort-conditioning demonstrated**: routed system uses significantly fewer tokens than force-hard at comparable EM on at least 3/4 datasets, and significantly higher EM than force-easy on at least 3/4 datasets.
- **Optimization contribution shown**: GEPA-optimized > unoptimized seed, demonstrated on val and test.
- **Oracle ablation**: oracle-supervised router > random routing, p < 0.05 on at least 2/3 in-distribution datasets.
- **Per-profile ablation table** complete: 7 profiles x 5 components, each with marginal EM contribution and 95% CI.
- **Inference-time scaling-law plot** published with fitted slopes per system.
- **Q1 answered** via per-profile ablation (where does parallelism help?).
- **Q2 answered** via collaboration strategy comparison (which components contribute most, for which question types?).
- **Q3 answered** via scaling-law plot (is there a predictable EM-vs-tokens relationship?).

If HERA-level absolute numbers are not reached, the thesis acknowledges this as a model-gap effect (homogeneous Qwen3-14B vs HERA's GPT-4o-mini orchestration) and positions the work as a controlled study of multi-agent collaboration under a strictly homogeneous setup.