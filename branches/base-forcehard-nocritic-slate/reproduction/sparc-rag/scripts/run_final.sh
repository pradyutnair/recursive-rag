#!/usr/bin/env bash
# Run final 1000q (per dataset) and 125q bamboogle at the chosen (W,D) per
# dataset. Reads results/grid/best.json from aggregate_grid.py.
#
# Usage:
#   bash scripts/run_final.sh <CONCURRENCY>

set -u

CONC=${1:-12}
PYTHON=/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python
ROOT=/local/yzheng/pnair/workspace/sparc-rag
DATA_ROOT=/local/yzheng/pnair/workspace/tmp/04-sage-autonomous/data/node408_shards
ADAMAS_DATA=/local/yzheng/pnair/workspace/adaptive-mas/data

declare -A QFILE_FULL=(
    [hotpotqa]="$ADAMAS_DATA/hotpotqa/questions_1000_seed42.json"
    [2wikimultihop]="$ADAMAS_DATA/2wikimultihop/questions_1000_seed42.json"
    [musique]="$ADAMAS_DATA/musique/questions_1000_seedfull_combined.json"
    [bamboogle]="$ROOT/data/bamboogle_125.json"
)

cd "$ROOT"
mkdir -p results/final
BEST=results/grid/best.json

if [[ ! -f "$BEST" ]]; then
    echo "missing $BEST -- run aggregate_grid.py first"
    exit 1
fi

for DS in hotpotqa 2wikimultihop musique bamboogle; do
    W=$($PYTHON -c "import json; print(json.load(open('$BEST'))['$DS']['W'])")
    D=$($PYTHON -c "import json; print(json.load(open('$BEST'))['$DS']['D'])")
    QF=${QFILE_FULL[$DS]}
    if [[ ! -f "$QF" ]]; then
        echo "[skip] missing $QF for $DS"
        continue
    fi
    OUT="results/final/${DS}/W${W}_D${D}"
    mkdir -p "$OUT"
    if [[ -f "$OUT/summary.json" ]]; then
        echo "[skip] $OUT (summary exists)"
        continue
    fi
    echo "[run] ds=$DS W=$W D=$D file=$QF"
    $PYTHON scripts/run_sparc.py \
        --questions "$QF" \
        --out-dir "$OUT" \
        --width "$W" --max-depth "$D" --topk 6 \
        --concurrency "$CONC" \
        > "$OUT/run.log" 2>&1
    tail -3 "$OUT/run.log"
done

echo "[done] final runs complete"
