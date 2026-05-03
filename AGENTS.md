# Handoff Prompt: Adaptive Multi-Agent RAG with Oracle Routing, Budget Conditioning, and Inference-Time Scaling Laws

## Mission

You are taking over a University of Amsterdam MSc thesis project on multi-hop retrieval-augmented question answering. The thesis title and questions (from the official project description) are:

**"Retrieval-Augmented Generation with Multi-Agent Collaborative Search"**

- Q1. Can multi-agent collaborative search increase the parallelism of iterative RAG, improving both efficiency and overall performance?
- Q2. What collaboration strategies maximize the utilization of collective intelligence in multi-agent RAG systems?
- Q3. Can inference-time scaling laws be observed in multi-agent RAG systems?

The thesis is **not** a "new architecture" thesis. The architecture is intentionally a faithful synthesis of two existing methods, **Plan*RAG (https://arxiv.org/abs/2410.20753)** and **HERA (https://arxiv.org/abs/2604.00901)**. The novelty is in three empirical contributions on top:

1. **Oracle-supervised routing** — using SAS-correctness as a per-question routing signal during gradient-free optimization. HERA's orchestrator learns from group-relative `(F1, tokens)` rewards; nobody uses an explicit SAS-easy/SAS-hard label as supervision. Show this signal is load-bearing.

2. **Budget-conditioned (resource-aware) orchestration** — at inference time, the orchestrator receives a runtime budget hint (`tight`, `normal`, `rich`) and selects topology accordingly. This realizes HERA's explicit future-work line: *"meta-optimization of agent roles, adaptive multi-turn orchestration, and resource-aware autonomous evolution, paving the way for hierarchical role structures, conditional agent sequencing, and efficient reasoning under token or latency constraints"*. No published paper has implemented controllable budget conditioning for multi-hop RAG.

3. **The first systematic inference-time scaling-law study for multi-hop QA in the open-domain full-wiki18 regime** — directly answers Q3. Five systems sharing scaffold, retriever, backbone, and test ids: naive_rag (SAS), IRCoT, MA-RAG, HERA replication, our system. Plot EM vs mean tokens per system per dataset, with paired bootstrap CIs.

Plus a per-question-type ablation answering Q1 and Q2: which components (router, DAG planner, parallel execution, critic, experience library) contribute how much on which question profiles (one_hop, bridge_2hop, bridge_3hop_plus, parallel_compare, temporal, numeric, yes_no).

## Hard constraints (non-negotiable)

- **Homogeneous Qwen3-14B at runtime.** No auxiliary models. No cross-encoder reranker. No different embedder. No LoRA fine-tune. No RL training. The retriever stays at E5-base on wiki18 100w-chunk corpus served at `node408:8003`.
- **No ensembling, pooling, majority voting, best-of-N selection across independent generations.**
- **No use of gold answers, baseline predictions, or any benchmark output as a runtime feature.** The oracle (SAS-correctness label) is **training-time supervision only** and never read at inference.
- **Full open-domain wiki18 corpus** (~21M chunks). NOT closed gold+distractor settings (which is what HippoRAG2 / GFM-RAG / BridgeRAG / PropRAG papers use). Numbers in those papers (40+ EM on MuSiQue) are not comparable.
- All test-set comparisons must be on the exact same question IDs as the FlashRAG baselines we already have rescored.

## Compute environment

- Active node: `node409`. `module load cuda12.6/toolkit/12.6` before any GPU command.
- 3× vLLM serving `Qwen/Qwen3-14B` at `localhost:8001`, `localhost:8002`, `localhost:8003`. `max_model_len=16384`, `max_num_seqs=32`, prefix caching enabled.
- Retriever: `node408:8003` POST `/retrieve` with body `{"queries":[...],"topk":N,"mode":"text"}`. Backed by E5-base FAISS over wiki18.
- Python env: `/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python` (dspy 3.2.0 with `dspy.GEPA`).
- API keys at `/local/yzheng/pnair/.env`. `OPENAI_API_KEY` is set; use `openai/gpt-5` for GEPA reflection (~$0.50-1 per full run, ~10x faster than local think reflection).
- DSPy cache: set `DSPY_CACHEDIR=/local/yzheng/pnair/workspace/recursive-rag/.dspy_cache`.

## Existing assets (do not redo)

- **Pipeline scaffold** at `/local/yzheng/pnair/workspace/recursive-rag/src/recrag/`: profile classifier (heuristic, 7 buckets), Plan*RAG-style DAG planner with `<A.I.J>` tag substitution, async executor with parallel within-layer execution, per-node investigator with retrieve+extract+rewrite (existing `_hop_async`, `MAX_ATTEMPTS=3`), synthesizer, critic, topology mutation, citation gate. Sync mirror for GEPA threading.
- **GEPA driver** in `src/recrag/gepa/compile.py`: supports `--reflection-lm openai/gpt-5`, `--max-metric-calls`, `--log-dir` checkpointing, `--oracle-naive-dir`, `--oracle-datasets`. Patched dspy `gepa.proposer.merge.find_common_ancestor_pair` for the zero-weight crash bug; preserve this patch.
- **TF-GRPO driver** in `src/recrag/grpo/compile.py` with HERA-style ADD/MERGE/PRUNE/KEEP experience library at `src/recrag/grpo/library.py`, profile|difficulty tagged.
- **Oracle on 1500 fresh questions** in `compiled/oracle/{musique,2wikimultihop,hotpotqa}_fresh_naive/predictions.jsonl`. Per-dataset SAS EM: musique 0.07, 2wiki 0.354, hotpot 0.362. Mean tokens 2325. These are SAS-mode runs (single agent, max 3 retrieve-extract-rewrite cycles) on fresh questions verified to have zero ID overlap with test sets.
- **Stratified train/val** in `data/multidataset/train_v3.json` (240q, 80 per dataset, balanced across dataset × oracle_easy × profile) and `val_v3.json` (30q).
- **Recovered GEPA-evolved program** at `compiled/gepa_v3_recovered_cand13.json` — yesterday's GEPA winner, val EM 0.433 on val_v3. Use as seed for planner/synthesizer/critic prompts.
- **Baselines rescored on exact 1000q test ids**: `results/diagnostics/baselines_rescored.json` with `(method, dataset, norm_em, token_f1, contain, mean_tokens)` for {naive_rag, IRCoT, OPERA, MA-RAG} × {musique, 2wikimultihop, hotpotqa, bamboogle}.

## Test sets (fixed; never train on these)

- MuSiQue 1000q: `data/musique/questions_1000_seedfull_combined.json`
- 2WikiMultiHopQA 1000q: `data/2wikimultihop/questions_1000_seed42.json`
- HotpotQA 1000q: `data/hotpotqa/questions_1000_seed42.json`
- Bamboogle 125q OOD: `data/bamboogle/questions_125.json`

All four use `node408:8003` retriever, top-5 by default.

## Numeric baseline reference

Open-domain full-wiki18, Qwen3-14B no-think, top-5 retrieval, exact 1000q test IDs (rescored from FlashRAG predictions):

| method | MuSiQue EM / F1 / contain / tokens | 2Wiki EM / F1 / contain / tokens | HotpotQA EM / F1 / contain / tokens | Bamboogle EM / F1 / contain / tokens |
|---|---|---|---|---|
| naive_rag | 0.053 / 0.130 / 0.080 / 877 | 0.209 / 0.282 / 0.295 / 922 | 0.323 / 0.443 / 0.407 / 885 | 0.160 / 0.309 / 0.192 / 854 |
| IRCoT | 0.075 / 0.128 / 0.143 / 7896 | 0.159 / 0.256 / 0.559 / 7579 | 0.311 / 0.410 / 0.485 / 6457 | 0.208 / 0.311 / 0.320 / 6775 |
| OPERA | 0.106 / 0.168 / 0.168 / 3445 | 0.082 / 0.144 / 0.193 / 3673 | 0.143 / 0.231 / 0.293 / 3070 | 0.264 / 0.382 / 0.328 / 2720 |
| MA-RAG | 0.124 / 0.224 / 0.215 / 9667 | 0.218 / 0.325 / 0.473 / 9416 | 0.244 / 0.364 / 0.426 / 8830 | 0.384 / 0.505 / 0.504 / 6828 |

HERA paper headline (https://arxiv.org/abs/2604.00901) on the same datasets in equivalent regime, Qwen3-14B + GPT-4o-mini agents:

| dataset | HERA EM | HERA F1 | HERA approx tokens |
|---|---|---|---|
| MuSiQue | 0.272 | 0.358 | ~12-15k |
| 2WikiMultiHopQA | 0.595 | 0.648 | ~12-15k |
| HotpotQA | 0.525 | 0.630 | ~12-15k |
| Bamboogle | 0.465 | 0.605 | ~10-12k |

AMAS-PRO existing baseline (homogeneous Qwen3-14B, no GPT-4o-mini), MuSiQue 1000q: 0.226 EM, 21,704 tokens.

Recovered cand13 verification on val_v3 (30q stratified, after GEPA crash recovery): 0.433 EM, 12,363 tokens.

## Numeric targets for the thesis

**Primary (Pareto-frontier targets, must hit at least 2/4):**

| dataset | target EM | target mean tokens | notes |
|---|---|---|---|
| MuSiQue 1000q | ≥ 0.27 | ≤ 9,000 | match HERA's EM at ≤ 60% of HERA tokens; +14 EM over best baseline (MA-RAG) |
| 2WikiMultiHopQA 1000q | ≥ 0.55 | ≤ 9,000 | match HERA on EM at ≤ 60% tokens; +33 EM over best baseline |
| HotpotQA 1000q | ≥ 0.50 | ≤ 8,000 | within HERA range on EM at ≤ 50% tokens |
| Bamboogle 125q (OOD) | ≥ 0.45 | ≤ 8,000 | match HERA OOD; +7 EM over best baseline |

**Secondary (efficiency on easy-routed questions, must hit):**

| metric | target |
|---|---|
| Mean tokens on questions where the router decides "easy" | ≤ 4,000 |
| EM on those easy-routed questions | ≥ baseline naive_rag's EM on the same questions |
| Easy-route fraction on MuSiQue | 8-15% (matches oracle stat 7%) |
| Easy-route fraction on 2Wiki | 30-40% (matches oracle stat 35%) |
| Easy-route fraction on HotpotQA | 30-40% (matches oracle stat 36%) |

**Ablation targets (must show statistically significant differences, paired bootstrap CI p < 0.05):**

| ablation contrast | minimum effect to claim novelty |
|---|---|
| oracle-trained router vs no-oracle GEPA | ≥ +2 EM at matched token budget on MuSiQue |
| oracle-trained router vs random routing at observed mix | ≥ +3 EM |
| budget="rich" vs budget="tight" on same system | ≥ +3 EM gain when allowed more budget |
| budget="tight" vs naive_rag at matched tokens | ≥ +2 EM (the system Pareto-dominates SAS at SAS budget) |

## Reward design (training-time only; never used at inference)

```
quality   = 1.0·EM + 0.5·F1 + 0.3·contain + 0.4·grounded + 0.2·shape_match
efficiency = exp(-total_tokens / 8000)
composite = quality · efficiency^alpha    with alpha = 0.3

oracle_bonus(easy, em, topology, tokens, naive_tokens):
  if oracle_easy:
      if topology == easy_lane and em == 1: +0.5     # cheap recovery, target behavior
      if em == 0:                          : -0.5    # regression vs SAS
      if em == 1 and tokens > 3·naive_tokens: -0.3   # right but wasteful
  else (oracle_hard):
      if em == 1                           : +0.8    # MAS recovery, the thesis claim
      if topology == easy_lane and em == 0 : -0.4    # router under-routed

final_reward = composite + oracle_bonus
```

For the budget-conditioned variant of the metric, also include `+0.3 · (1 if mean_tokens <= budget_target_for_hint else 0)` as a hint-respecting bonus.

GEPA metric returns `dspy.Prediction(score=final_reward, feedback=...)`. Feedback string includes the oracle context per question (e.g., `[ORACLE] naive_rag (Qwen3-14B no-think SAS lane) solved this in 840 tokens. Yours: topology=dag_n3, tokens=6200, em=0. oracle=easy: regression vs SAS (-0.5). [budget=normal]`). Module-targeted feedback for `router`, `planner`, `synthesizer`, `critic` is added based on `pred_name` argument.

## Architecture you are extending

The runtime already exists at `src/recrag/adaptive_pipeline.py` (async) and `src/recrag/sync_pipeline.py` (sync mirror for GEPA). It already does: profile classify → planner think → DAG executor (parallel within layer with `<A.I.J>` tag substitution) → per-node investigator with up to 3 retrieve-extract-rewrite cycles → synthesizer → critic with topology mutation → citation gate.

What it is missing:

1. **A Router module before the planner.** Adds two named predictors: `router` (no-think, ~200 token budget, takes `{question, profile, experience, budget_hint}`, returns strict JSON `{"route":"easy|hard","reason":"..."}`). When `route=easy`, run a single Investigator using existing `tools._hop_async` semantics with `MAX_ATTEMPTS=3 or 4` (the easy lane is multi-shot retrieval inside one agent, NOT one-retrieve-one-read). When `route=hard`, run the existing planner→DAG→synth→critic path.

2. **A `budget_hint` input field** wired through the router, planner, and synthesizer prompts. Three values: `tight`, `normal`, `rich`. At inference time the user passes one of these. The orchestrator prompt evolves (via GEPA) to honor the budget hint.

3. **GEPA wired to optimize four named predictors**, not three: `router`, `planner`, `synthesizer`, `critic`. Use `compiled/gepa_v3_recovered_cand13.json` as seed for planner/synth/critic; hand-write the seed router prompt in the spirit of HERA's chain-disambiguation rules (named entity vs unnamed bridge, nested-of chains, comparisons → hard; single named subject + one missing attribute → easy).

4. **HERA-style token-aware reflection** — the GEPA reflection prompt for the orchestrator/router must include both `F1` and `tokens` per trajectory, sorted by `(F1↓, tokens↑)`. Already partly done in `src/recrag/grpo/signatures.py`; extend to GEPA reflection too.

## Things that wasted prior time — avoid these

1. Do not run synth or critic in think mode. They are decision tasks; no-think Qwen3-14B suffices and is 5-10× faster.
2. Do not skip critic based purely on planner-emitted topology. Use the heuristic profile classifier as the hard floor (`profile_classifier == one_hop` AND `topology == easy_lane` AND `extractor_confidence >= 0.7`).
3. Always launch GEPA with `--log-dir compiled/<run>_logs --reflection-lm openai/gpt-5 --reflection-max-tokens 16000 --max-metric-calls 800 --num-threads 6 --track-stats`. Yesterday's run lost 7h of compute by crashing without checkpointing.
4. Preserve the patched `gepa.proposer.merge.find_common_ancestor_pair` (zero-weight bug fix). If the package is reinstalled, re-apply.
5. Do not use `asyncio.to_thread` directly. Use `recrag.aio.to_thread` (per-thread executor with broken-executor detection). dspy's threadpool conflicts with the default loop.
6. Do not run on the 1000q test sets without first hitting the targets on val_v3 (30q) and a 100q stratified pilot.

## Concrete experiment plan (what produces the thesis)

### Phase 0 — orient (~1h)
Read the pipeline files. Run `compiled/gepa_v3_recovered_cand13.json` against val_v3 to confirm seed quality reproduces.

### Phase 1 — add Router and budget hint (~4-6h)
Add `router` and `route_lm` (no-think) to the pipeline. Add `budget_hint` field. Plumb easy lane through `_hop_async` with MAX_ATTEMPTS=3-4. Validate on a 30q smoke that easy lane costs ≤ 4k tokens.

### Phase 2 — GEPA over all four predictors (~6h, ~$1-3)
`python -m recrag.gepa.compile --questions data/multidataset/train_v3.json --valset data/multidataset/val_v3.json --reflection-lm openai/gpt-5 --reflection-max-tokens 16000 --max-metric-calls 800 --num-threads 6 --log-dir compiled/gepa_v4_logs --out compiled/gepa_v4.json --oracle-naive-dir compiled/oracle --oracle-datasets musique,2wikimultihop,hotpotqa`. Seed planner/synth/critic from `gepa_v3_recovered_cand13.json`; hand-write router seed.

### Phase 3 — TF-GRPO over experience library (~5h, ~$3-8)
`python -m recrag.grpo.compile --questions data/multidataset/train_v3.json --reflection-lm openai/gpt-5 --oracle-naive-dir compiled/oracle --group-size 4 --epochs 3 --batch-size 10 --out-json compiled/grpo_v4_E.json`. Library entries get tagged `[easy|profile]` or `[hard|profile]`.

### Phase 4 — pilot eval on 100q stratified per dataset (~1.5h)
Decision gate: only proceed to 1000q if pilot beats best FlashRAG baseline at lower or equal mean tokens on at least 3/4 datasets.

### Phase 5 — full eval on 1000q × 4 datasets at three budget hints (~10-15h)
Three runs per dataset: `budget_hint=tight`, `normal`, `rich`. Output `results/runs/test_v4_<budget>/<dataset>/predictions.jsonl + summary.json`.

### Phase 6 — ablations on the same 1000q test ids (~6-10h)
- `--no-critic`: critic contribution
- Force easy: SAS-only lower bound (must match naive_rag-style numbers)
- Force hard: unconditional MAS upper bound
- Random routing at observed easy/hard mix rate
- No-oracle GEPA (re-run GEPA without oracle bonus, eval that program)
- HERA replication: build orchestrator that mirrors HERA Algorithm 1-6 on our scaffold, run it

### Phase 7 — analysis and writeup (~3-5d)
- **Pareto plot per dataset** (EM vs mean tokens): all baselines as points + each system as line across budget hints. Paired bootstrap CIs and McNemar significance.
- **Inference-time scaling-law plot**: log-log EM vs tokens with fitted slopes per system, per dataset. Identify regimes where adaptive < SAS, adaptive < MAS, adaptive optimal.
- **Per-question-type ablation table**: EM, mean tokens, marginal contribution of each component, broken down by 7 profiles.
- **Routing analysis**: confusion matrix of router prediction vs oracle, and EM as function of router confidence.
- **Budget-conditioning curve**: EM per dataset as function of budget hint, with confidence intervals.

## Reference papers (must read before starting)

- Plan*RAG (external DAG, atomic subqueries, parallel execution): https://arxiv.org/abs/2410.20753
- HERA (token-aware orchestrator, Profile-Insight-Utility library, RoPE, topology mutation): https://arxiv.org/abs/2604.00901
- TF-GRPO (gradient-free GRPO, semantic advantage): https://arxiv.org/abs/2510.08191
- GEPA (reflective prompt evolution with Pareto frontier): https://arxiv.org/abs/2507.19457
- MA-RAG (multi-agent CoT for RAG): https://arxiv.org/abs/2505.20096
- RecursiveMAS (latent recursive multi-agent): https://arxiv.org/abs/2604.25917
- Single-agent vs multi-agent at equal budgets (the threat model to address): https://arxiv.org/abs/2604.02460
- BridgeRAG (bridge-conditioned reranking, closed corpus): https://arxiv.org/abs/2604.03384
- Search-R1 (RL-trained search agent, NOT in our regime): https://arxiv.org/abs/2503.09516
- R1-Searcher (RL-trained, NOT in our regime): https://arxiv.org/abs/2503.05592
- HippoRAG2 (graph KG retrieval, closed corpus): https://arxiv.org/abs/2502.14802
- MuSiQue dataset paper: https://arxiv.org/abs/2108.00573
- 2WikiMultiHopQA dataset paper: https://aclanthology.org/2020.coling-main.580/
- HotpotQA dataset paper: https://arxiv.org/abs/1809.09600

## What success looks like

You can claim the thesis is done when ALL of these are true:

- Pareto curve per dataset published with our system at three budget points dominating at least one operating point of HERA on at least 3/4 datasets.
- Statistical significance: paired bootstrap p < 0.05 vs naive_rag and MA-RAG on EM, on at least 3/4 datasets.
- Oracle ablation: oracle-trained router > no-oracle GEPA winner > random routing, p < 0.05 on at least 2/3 in-distribution datasets, evaluated on the 1000q test sets (not val_v3).
- Budget-conditioning ablation: EM(`rich`) > EM(`normal`) > EM(`tight`) by at least +3 EM each step on MuSiQue, with mean tokens monotonically increasing.
- Per-profile ablation table is complete: 7 profiles × {router, planner, parallel exec, critic, experience library} = 35 cells, each with marginal EM contribution and 95% CI.
- Inference-time scaling-law plot is published with fitted slopes per system.
- The thesis writeup explicitly addresses Q1 (parallelism contribution from per-profile ablation), Q2 (best collaboration strategy as identified by experience library analysis), Q3 (scaling-law plot).

If you hit those criteria, the thesis is publishable. Architectural novelty is not required; empirical contributions and clean analysis are sufficient for the MSc.

The infrastructure is solid. The architecture is one router-lane addition away from being able to deliver the thesis claims. The hard work remaining is empirical and analytical, not architectural. Run the experiments, plot the curves, write the analysis.