# RecRAG-MAS Base

Bare force-hard no-critic recursive RAG baseline for thesis experiments.

## Method

Runtime path:

`question -> profile -> DAG planner -> parallel DAG executor -> synthesizer -> citation gate -> answer cleanup`

Removed from this branch: router, easy lane, critic, GEPA, GRPO, oracle training assets, W&B artifacts, old analysis scripts, and previous experiment outputs.

## Frozen Reference

Frozen base artifact:

`compiled/base_method_forcehard_nocritic_20260504.json`

Retained full result files:

- `results/runs/test_forcehard_nocritic_20260504/` for MuSiQue 1000q, 2Wiki 1000q, HotpotQA 1000q, and the original Bamboogle 125q run.
- `results/runs/base_forcehard_nocritic_bamboogle_20260504/` for the frozen Bamboogle 125q run after generic answer cleanup.

Reference scores:

| dataset | EM | F1 | contain | mean tokens |
| --- | ---: | ---: | ---: | ---: |
| MuSiQue 1000q | 0.170 | 0.241 | 0.193 | 6,989.1 |
| 2Wiki 1000q | 0.308 | 0.370 | 0.376 | 8,137.9 |
| HotpotQA 1000q | 0.323 | 0.421 | 0.388 | 6,051.1 |
| Bamboogle 125q original | 0.312 | 0.478 | 0.352 | 4,700.7 |
| Bamboogle | 0.392 | 0.494 | 0.400 | 4,760.6 |

## Run

```bash
cd /local/yzheng/pnair/workspace/recursive-rag
module load cuda12.6/toolkit/12.6
export DSPY_CACHEDIR=/local/yzheng/pnair/workspace/recursive-rag/.dspy_cache

/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python scripts/eval_on_test.py \
  --out-dir results/runs/base_forcehard_nocritic \
  --questions-file /local/yzheng/pnair/workspace/adaptive-mas/data/bamboogle/questions_125.json \
  --dataset-name bamboogle \
  --root-lm qwen14b-nothink \
  --sub-lm qwen14b-nothink \
  --root-max-tokens 768 \
  --sub-max-tokens 512 \
  --max-searches 5 \
  --max-recursion 0 \
  --concurrency 24
```

Architecture reference: `docs/architecture.md`.
