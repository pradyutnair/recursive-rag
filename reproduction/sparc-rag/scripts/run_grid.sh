#!/usr/bin/env bash
# WxD grid driver. Runs every (dataset, W, D) combination on the first N
# questions of each dataset. Output goes to results/grid/<dataset>/W<W>_D<D>/.
#
# Usage:
#   bash scripts/run_grid.sh <N> <CONCURRENCY> [datasets...]
# Example:
#   bash scripts/run_grid.sh 100 10 hotpotqa 2wikimultihop musique bamboogle

set -u

N=${1:-100}
CONC=${2:-10}
shift 2 || true
DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(hotpotqa 2wikimultihop musique bamboogle)
fi

PYTHON=/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python
ROOT=/local/yzheng/pnair/workspace/sparc-rag
DATA_ROOT=/local/yzheng/pnair/workspace/tmp/04-sage-autonomous/data/node408_shards

declare -A QFILE=(
    [hotpotqa]="$DATA_ROOT/hotpotqa/questions_shard_0.json"
    [2wikimultihop]="$DATA_ROOT/2wikimultihop/questions_shard_0.json"
    [musique]="$DATA_ROOT/musique/questions_shard_0.json"
    [bamboogle]="$DATA_ROOT/bamboogle/questions_shard_0.json"
)

WIDTHS=(1 2 4)
DEPTHS=(2 4 6 8)

cd "$ROOT"
mkdir -p results/grid

for DS in "${DATASETS[@]}"; do
    for W in "${WIDTHS[@]}"; do
        for D in "${DEPTHS[@]}"; do
            OUT="results/grid/${DS}/W${W}_D${D}"
            mkdir -p "$OUT"
            if [[ -f "$OUT/summary.json" ]]; then
                echo "[skip] $OUT (summary exists)"
                continue
            fi
            echo "[run] ds=$DS W=$W D=$D N=$N"
            $PYTHON scripts/run_sparc.py \
                --questions "${QFILE[$DS]}" \
                --out-dir "$OUT" \
                --n "$N" --width "$W" --max-depth "$D" --topk 6 \
                --concurrency "$CONC" \
                > "$OUT/run.log" 2>&1
            tail -1 "$OUT/run.log"
        done
    done
done

echo "[done] grid complete"
