#!/usr/bin/env bash
set -Eeuo pipefail
STAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ARTICLE_DIR="$(cd -- "$STAGE_DIR/../.." && pwd -P)"
DATASET="${1:-FD001}"
case "$DATASET" in FD001|FD002|FD003|FD004) ;; *) echo "Uso: $0 FD001|FD002|FD003|FD004" >&2; exit 2 ;; esac
if [[ ! -r "$ARTICLE_DIR/02_raw_data/PM_train.csv" ]]; then echo "Project root non valido: $ARTICLE_DIR" >&2; exit 2; fi
OUTPUT="$ARTICLE_DIR/04_results/10_failure_proximity_${DATASET}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
python3 "$STAGE_DIR/stage10.py" --project-root "$ARTICLE_DIR" --dataset "$DATASET" --output "$OUTPUT" --horizon 30 --min-prefix 8 --threshold 0.5 --confirmations 2 --folds 5 --bootstrap 1000
(cd "$OUTPUT" && sha256sum -c SHA256SUMS)
printf '\nRESULTATI: %s\n' "$OUTPUT"
