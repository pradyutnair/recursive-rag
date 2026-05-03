# Legacy scripts

Pre-v3 scripts kept for reproducibility but NOT part of the current SOTA workflow. The v3 workflow is in `scripts/` (one level up).

| File | Original purpose | Replaced by |
|---|---|---|
| `run_react.py` | Old DSPy ReAct runner (single-agent loop) | `scripts/eval_on_test.py` with `--no-dag` flag (force-SAS ablation) |
| `run_adaptive_async.py` | Earlier async version of the adaptive runner | `scripts/eval_on_test.py` (uses sync pipeline for thread-safety with GEPA) |
| `eval_offline.py` | Standalone offline eval against a predictions.jsonl + gold | `scripts/eval_on_test.py` writes both predictions and per-dataset summaries in one pass |
| `headtohead.py` | Two-method comparison utility | `scripts/aggregate_ablations.py` |
| `compile_gepa_wrapper.py` | Trivial wrapper around `recrag.gepa.compile.main` | use `python -m recrag.gepa.compile` |
| `compile_grpo_wrapper.py` | Trivial wrapper around `recrag.grpo.compile.main` | use `python -m recrag.grpo.compile` |

Don't delete; some thesis ablations may still want to invoke `run_react.py` for the pre-DAG baseline.
