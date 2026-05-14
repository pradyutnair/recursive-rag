# Thesis Results Snapshot - 2026-05-04

## Bottom Line
- Ideal 1000q EM targets are not met tonight; the defensible story is relative and empirical, not SOTA-chasing.
- The full 1000q/125q MAS system beats MA-RAG on 3/4 datasets with fewer tokens: MuSiQue, 2WikiMultiHopQA, HotpotQA.
- Collaboration is clearly load-bearing: blackboard/DAG handoff beats blind workers by large paired F1 margins on MuSiQue 200q and Bamboogle 125q.
- Test-time scaling is dataset-dependent: MAS helps MuSiQue and 2Wiki, is neutral/negative on HotpotQA, and small/noisy on Bamboogle. This supports effort-conditioned routing.
- A simple profile-adaptive blend saves tokens at near-MAS EM, especially on 2Wiki, but it is not yet a strong accuracy win.

## Full Fixed-ID Results
| dataset | MAS EM | MAS F1 | MAS tok | SAS EM | SAS F1 | SAS tok | adaptive EM | adaptive tok | MA-RAG EM | MA-RAG tok | result vs MA-RAG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| musique | 0.170 | 0.241 | 6989 | 0.079 | 0.153 | 3252 | 0.166 | 6824 | 0.124 | 9667 | wins EM and tokens |
| 2wikimultihop | 0.308 | 0.370 | 8138 | 0.281 | 0.333 | 3198 | 0.304 | 6016 | 0.218 | 9416 | wins EM and tokens |
| hotpotqa | 0.323 | 0.421 | 6051 | 0.331 | 0.437 | 3192 | 0.316 | 5307 | 0.244 | 8830 | wins EM and tokens |
| bamboogle | 0.312 | 0.478 | 4701 | 0.304 | 0.421 | 3003 | 0.312 | 4518 | 0.384 | 6828 | fewer tokens, lower EM |

## Collaboration Ablation
| dataset | full EM/F1/tok | parents EM/F1/tok | blind EM/F1/tok | full-blind F1 delta | paired CI |
|---|---:|---:|---:|---:|---|
| musique | 0.190/0.278/6930 | 0.190/0.273/7090 | 0.040/0.089/6491 | +0.189 | [0.129, 0.253], p=0.0005 |
| bamboogle | 0.368/0.471/4685 | 0.376/0.480/4692 | 0.008/0.024/5125 | +0.447 | [0.365, 0.531], p=0.0005 |

Interpretation: parent handoff is the critical communication channel. Sibling sharing is not consistently better than parents-only on these datasets, so the clean claim should be Blackboard-mediated DAG fact handoff, not broad sibling gossip.

## Search-Depth Scaling
| dataset | max1 EM/F1/tok | max5 EM/F1/tok | F1 delta | EM delta |
|---|---:|---:|---:|---:|
| musique | 0.170/0.259/5064 | 0.190/0.278/6930 | +0.019 | +0.020 |
| bamboogle | 0.312/0.411/3990 | 0.368/0.471/4685 | +0.060 | +0.056 |

Interpretation: extra per-node search improves Bamboogle clearly and MuSiQue modestly, with diminishing returns. This is a useful scaling curve, not a monotonic miracle.

## Worker Width
| dataset | width1 wall | width4 wall | width1 F1 | width4 F1 | note |
|---|---:|---:|---:|---:|---|
| musique | 495.9s | 483.3s | 0.281 | 0.278 | little speedup because most plans are sequential bridge chains |
| bamboogle | 207.2s | 208.3s | 0.471 | 0.471 | little speedup because most plans are sequential bridge chains |

## Thesis Framing
1. Q1: Multi-agent DAG search improves over single-agent on MuSiQue and 2Wiki at full scale, but not uniformly; this motivates adaptive compute rather than always-MAS.
2. Q2: Communication matters. Removing Blackboard handoff collapses performance on MuSiQue and Bamboogle. Parent dependency facts are the dominant useful communication strategy.
3. Q3: Scaling laws are observable but dataset-dependent: search budget has positive diminishing returns; worker width only helps when the plan has parallel layers; uniform scaling wastes tokens on HotpotQA/simple cases.

## Artifacts
- results/runs/test_forcehard_nocritic_20260504/summary.json
- results/runs/test_true_sas_research_plan_20260504/summary.json
- results/runs/adaptive_profile_blend_20260504/summary.json
- results/runs/sage_blackboard_ablate_20260504/full_share/summary.json
- results/runs/sage_blackboard_ablate_20260504/parents_only/summary.json
- results/runs/sage_blackboard_ablate_20260504/blind_workers/summary.json
- results/runs/sage_blackboard_scaling_20260504/maxsearch1_width4/summary.json
- results/runs/sage_blackboard_scaling_20260504/maxsearch5_width1/summary.json
- results/analysis/sage_blackboard_stats_20260504.json
