#!/usr/bin/env bash
# Run missing grid configs for ONE dataset.
# Usage: bash run_missing_grid_ds.sh <DS> [CONC]
set -u

PYTHON=/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python
ROOT=/local/yzheng/pnair/workspace/sparc-rag
DATA_ROOT=/local/yzheng/pnair/workspace/tmp/04-sage-autonomous/data/node408_shards
DS=$1
CONC=${2:-20}

run() {
    local W=$1 D=$2
    local OUT="$ROOT/results/grid/$DS/W${W}_D${D}"
    mkdir -p "$OUT"
    if [[ -f "$OUT/summary.json" ]]; then
        echo "[skip] $OUT"
        return
    fi
    rm -f "$OUT/predictions.jsonl"
    echo "[run] $DS W=$W D=$D conc=$CONC"
    $PYTHON $ROOT/scripts/run_sparc.py \
        --questions "$DATA_ROOT/$DS/questions_shard_0.json" \
        --out-dir "$OUT" \
        --n 100 --width "$W" --max-depth "$D" --topk 6 \
        --concurrency "$CONC" \
        > "$OUT/run.log" 2>&1
    tail -1 "$OUT/run.log"
}

cd "$ROOT"
for WD in "2 8" "4 2" "4 4" "4 6" "4 8"; do
    run $WD
done
echo "[done] $DS missing grid complete"
