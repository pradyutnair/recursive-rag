# Repo Instructions

Be concise. This branch is the clean base for future experiments.

## Base Method

Use only the force-hard no-critic RecRAG-MAS path:

`question -> profile -> DAG planner -> DAG executor -> synthesizer -> citation gate -> answer cleanup`

Runtime constraints:

- Homogeneous `Qwen/Qwen3-14B` no-think for all agents.
- Retriever remains E5-base over wiki18 served from `node408:8003`.
- No router, easy lane, adaptive cascade, critic, GEPA, GRPO, oracle labels, W&B, ensembling, pooling, majority voting, best-of-N, reranking, LoRA, RL, or auxiliary models.
- All future methods should branch from this base and add one controlled change at a time.

## Reference

- Frozen artifact: `compiled/base_method_forcehard_nocritic_20260504.json`
- Architecture diagram: `docs/architecture.md`
- Eval entrypoint: `scripts/eval_on_test.py`
