#!/usr/bin/env bash
# Five-fold cross-validation, thresholds from pooled validation folds, and evaluation (mean +/- SD over fold models).
#
#   bash scripts/run_all.sh DEV_MANIFEST [EXTERNAL_MANIFEST:NAME ...]
set -euo pipefail
DEV=$1; shift
RUNS=${RUNS:-runs}
for FOLD in 0 1 2 3 4; do
  bash scripts/run_fold.sh "$FOLD" "$DEV" "$@"
done
python -m protocouncil.council calibrate --predictions "$RUNS"/council/fold*/val.csv --out "$RUNS/council/thresholds.json"
for SPLIT in val "${@##*:}"; do
  [ -z "$SPLIT" ] && continue
  echo "== $SPLIT"
  python -m protocouncil.council evaluate --predictions "$RUNS"/council/fold*/"$SPLIT".csv \
    --thresholds "$RUNS/council/thresholds.json" --out "$RUNS/council/metrics_$SPLIT.csv"
done
