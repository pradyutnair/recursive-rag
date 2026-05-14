#!/usr/bin/env bash
# Re-launch the missing 2wiki + musique grid points at high concurrency.
set -u

PYTHON=/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python
ROOT=/local/yzheng/pnair/workspace/sparc-rag
DATA_ROOT=/local/yzheng/pnair/workspace/tmp/04-sage-autonomous/data/node408_shards
CONC=${1:-24}

run() {
    local DS=$1 W=$2 D=$3
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
for CFG in "2wikimultihop 2 8" "2wikimultihop 4 2" "2wikimultihop 4 4" "2wikimultihop 4 6" "2wikimultihop 4 8" \
           "musique 2 8" "musique 4 2" "musique 4 4" "musique 4 6" "musique 4 8"; do
    run $CFG
done
echo "[done] missing grid complete"
