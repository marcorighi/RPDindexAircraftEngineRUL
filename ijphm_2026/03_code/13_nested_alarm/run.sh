#!/usr/bin/env bash
set -Eeuo pipefail
STAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ARTICLE_DIR="$(cd -- "$STAGE_DIR/../.." && pwd -P)"
DATASET="${1:-FD001}"
case "$DATASET" in FD001|FD002|FD003|FD004) ;; *) echo "Uso: $0 FD001|FD002|FD003|FD004" >&2; exit 2 ;; esac
DATAFILE="$ARTICLE_DIR/02_raw_data/PM_train.csv"
[[ -r "$DATAFILE" && -f "$ARTICLE_DIR/03_code/10_failure_proximity/stage10.py" ]] || { echo 'Project Stage10 prerequisito mancante' >&2; exit 3; }
OUTPUT="$ARTICLE_DIR/04_results/13_nested_alarm_${DATASET}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
python3 "$STAGE_DIR/stage13.py" --project-root "$ARTICLE_DIR" --dataset "$DATASET" --output "$OUTPUT" \
   --outer-folds 5 --inner-folds 4 --horizon 30 --min-lead 5 \
   --min-prefix 8 --max-premature 0.20 --bootstrap 1000 --seed 20261103
(cd "$OUTPUT" && sha256sum -c SHA256SUMS)
printf '\nRISULTATI STAGE13: %s\n' "$OUTPUT"
