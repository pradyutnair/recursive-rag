# Results

This folder contains Recursive RAG experiment outputs and wiki18-corpus baseline summaries. Metrics are produced with `scripts/eval_offline.py`.

## Layout

- `runs/<dataset>/<run>/config/`: run config.
- `runs/<dataset>/<run>/predictions/`: model predictions and traces.
- `runs/<dataset>/<run>/metrics/`: eval, token, topology, and postprocess summaries.
- `runs/<dataset>/<run>/notes/`: run notes.
- `baselines/wiki18-corpus/`: external wiki18 baseline outputs only.
- `summaries/`: compact JSON summaries across runs and wiki18 baselines.
- `scratch/`: one-off diagnostics.

## Recursive RAG


| Method                                   | Backbone                                  | Dataset                | EM    | F1    | Contain | Mean Tokens  | Mean Latency (s) | Mean Subagents |
| ---------------------------------------- | ----------------------------------------- | ---------------------- | ----- | ----- | ------- | ------------ | ---------------- | -------------- |
| Recursive RAG                            | Qwen3-14B homogeneous no-think            | 2Wiki                  | 0.348 | 0.435 | 0.545   | 14025.355    | not recorded     | 2.928          |
| Recursive RAG                            | Qwen3-8B homogeneous no-think             | 2Wiki                  | 0.3   | 0.395 | 0.502   | 11950.731    | not recorded     | 2.635          |
| Recursive RAG                            | Qwen3-14B homogeneous no-think            | Bamboogle              | 0.376 | 0.503 | 0.432   | 9634.192     | not recorded     | 2.128          |
| Recursive RAG                            | Qwen3-8B homogeneous no-think             | Bamboogle              | 0.336 | 0.474 | 0.416   | 10075.552    | not recorded     | 2.256          |
| Recursive RAG                            | Qwen3-14B homogeneous no-think            | HotpotQA               | 0.329 | 0.456 | 0.485   | 12177.202    | not recorded     | 2.477          |
| Recursive RAG                            | Qwen3-8B homogeneous no-think             | HotpotQA               | 0.351 | 0.471 | 0.498   | 11848.022    | not recorded     | 2.421          |
| Recursive RAG + diagnostic cleanup       | Qwen3-14B homogeneous no-think            | MuSiQue                | 0.24  | 0.313 | 0.25    | not recorded | not recorded     | not recorded   |
| Recursive RAG                            | Qwen3-14B homogeneous no-think            | MuSiQue                | 0.208 | 0.303 | 0.246   | not recorded | not recorded     | not recorded   |
| Recursive RAG                            | Qwen3-14B planner + GPT-4o-mini subagents | MuSiQue                | 0.209 | 0.304 | 0.254   | 13554.887    | not recorded     | 2.953          |
| Recursive RAG                            | Qwen3-8B homogeneous no-think             | MuSiQue                | 0.167 | 0.263 | 0.215   | 11965.083    | not recorded     | 2.673          |
| Recursive RAG stratified pilot + cleanup | Qwen3-14B homogeneous no-think            | MuSiQue stratified 100 | 0.3   | 0.386 | 0.3     | not recorded | not recorded     | not recorded   |
| Recursive RAG stratified pilot           | Qwen3-14B homogeneous no-think            | MuSiQue stratified 100 | 0.28  | 0.382 | 0.3     | not recorded | not recorded     | not recorded   |


MuSiQue `diagnostic cleanup` rows are answer-format cleanup checks, not separate runtime methods. Recursive RAG latency was not recorded in these runs; future runs should persist wall-clock latency per question.

## Wiki18 Baselines


| Method              | Backbone                                  | Dataset   | EM    | F1    | Contain | Mean Tokens | Mean Latency (s) | Mean Subagents |
| ------------------- | ----------------------------------------- | --------- | ----- | ----- | ------- | ----------- | ---------------- | -------------- |
| IRCoT               | Qwen3-8B no-think, top5 node408           | 2Wiki     | 0.243 | 0.332 | 0.474   | 7106.4      | not recorded     | not applicable |
| MA-RAG              | Qwen3-8B planner-think 4096, top5 node408 | 2Wiki     | 0.206 | 0.286 | 0.362   | 11481.418   | not recorded     | not applicable |
| Naive RAG           | Qwen3-8B think 4096, top5 node408         | 2Wiki     | 0.143 | 0.219 | 0.365   | 1803.045    | not recorded     | not applicable |
| OPERA               | Qwen3-8B no-think, top5 node408           | 2Wiki     | 0.065 | 0.127 | 0.181   | 3527.011    | 23.028           | not applicable |
| OPERA planner-think | Qwen3-8B planner-think 4096, top5 node408 | 2Wiki     | 0.084 | 0.156 | 0.226   | 4437.64     | 75.328           | not applicable |
| IRCoT               | Qwen3-8B no-think, top5 node408           | Bamboogle | 0.296 | 0.389 | 0.352   | 6558.32     | not recorded     | not applicable |
| MA-RAG              | Qwen3-8B planner-think 4096, top5 node408 | Bamboogle | 0.424 | 0.539 | 0.472   | 8673.632    | not recorded     | not applicable |
| Naive RAG           | Qwen3-8B think 4096, top5 node408         | Bamboogle | 0.216 | 0.305 | 0.272   | 1936.992    | not recorded     | not applicable |
| OPERA               | Qwen3-8B no-think, top5 node408           | Bamboogle | 0.224 | 0.342 | 0.312   | 2281.808    | 15.134           | not applicable |
| OPERA planner-think | Qwen3-8B planner-think 4096, top5 node408 | Bamboogle | 0.24  | 0.36  | 0.376   | 2965.976    | 38.327           | not applicable |
| IRCoT               | Qwen3-8B no-think, top5 node408           | HotpotQA  | 0.347 | 0.464 | 0.448   | 6046.448    | not recorded     | not applicable |
| MA-RAG              | Qwen3-8B planner-think 4096, top5 node408 | HotpotQA  | 0.278 | 0.379 | 0.384   | 10880.213   | not recorded     | not applicable |
| Naive RAG           | Qwen3-8B think 4096, top5 node408         | HotpotQA  | 0.304 | 0.415 | 0.415   | 1743.07     | not recorded     | not applicable |
| OPERA               | Qwen3-8B no-think, top5 node408           | HotpotQA  | 0.131 | 0.217 | 0.323   | 2775.649    | 16.593           | not applicable |
| OPERA planner-think | Qwen3-8B planner-think 4096, top5 node408 | HotpotQA  | 0.108 | 0.197 | 0.308   | 3999.321    | 64.538           | not applicable |
| IRCoT               | Qwen3-8B no-think, top5 node408           | MuSiQue   | 0.081 | 0.156 | 0.127   | 7409.18     | not recorded     | not applicable |
| MA-RAG              | Qwen3-8B planner-think 4096, top5 node408 | MuSiQue   | 0.124 | 0.214 | 0.191   | 12068.031   | not recorded     | not applicable |
| Naive RAG           | Qwen3-8B think 4096, top5 node408         | MuSiQue   | 0.054 | 0.125 | 0.083   | 2088.36     | not recorded     | not applicable |
| OPERA               | Qwen3-8B no-think, top5 node408           | MuSiQue   | 0.069 | 0.13  | 0.146   | 3202.486    | 22.156           | not applicable |
| OPERA planner-think | Qwen3-8B planner-think 4096, top5 node408 | MuSiQue   | 0.069 | 0.128 | 0.15    | 4626.552    | 73.665           | not applicable |


## Current Takeaways

- Recursive RAG beats the listed wiki18 MuSiQue baselines on EM/F1/contain, but remains below the target range.
- Qwen3-14B improves 2Wiki and Bamboogle over Qwen3-8B, but Qwen3-8B is better on HotpotQA in the current setup.
- Token tails are still high on 14B, especially 2Wiki; this is the main efficiency issue to address before GEPA/TF-GRPO.

