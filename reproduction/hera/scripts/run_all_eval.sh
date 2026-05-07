#!/usr/bin/env bash
# Run HERA eval on all 4 datasets in parallel processes.
# Usage: bash run_all_eval.sh <run_dir> <out_dir> [limit]
set -euo pipefail

RUN_DIR="${1:?run_dir required}"
OUT_DIR="${2:?out_dir required}"
LIMIT="${3:-1000}"

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
# Resolve python: env override > Snellius FlashRAG > node409 adaptive-mas
PY="${HERA_PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in /projects/prjs1800/venvs/FlashRAG-venv/bin/python /local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
fi
LIB="$RUN_DIR/library.json"
PR="$RUN_DIR/prompts.json"

mkdir -p "$OUT_DIR" logs

set -a
for env_path in /projects/prjs1800/.env /local/yzheng/pnair/.env "$HOME/.env"; do
  if [ -f "$env_path" ]; then . "$env_path"; break; fi
done
set +a
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENAI_API_KEY_PRIVATE:-}}"
export PYTHONPATH="$PROJECT_ROOT/src"

run_eval() {
  local ds="$1"
  local lim="$2"
  local temp="$3"
  echo "[launch] $ds limit=$lim temp=$temp"
  $PY scripts/eval.py \
    --dataset "$ds" \
    --out-dir "$OUT_DIR" \
    --library "$LIB" \
    --prompts "$PR" \
    --limit "$lim" \
    --temperature "$temp" \
    --parallel 10 \
    --vllm-concurrency 12 \
    --openai-concurrency 24 \
    --wandb \
    --run-name "hera_eval_${ds}" \
    > "logs/eval_${ds}.log" 2>&1 &
  echo "  pid=$!"
}

# Indistribution: temp 0.0
run_eval musique "$LIMIT" 0.0
run_eval 2wikimultihop "$LIMIT" 0.0
run_eval hotpotqa "$LIMIT" 0.0
# OOD: temp 0.3 (paper § 4)
run_eval bamboogle 125 0.3

wait
echo "=== ALL EVAL COMPLETE ==="
ls "$OUT_DIR"/summary_*.json
for f in "$OUT_DIR"/summary_*.json; do
  echo "--- $f ---"
  cat "$f"
done
