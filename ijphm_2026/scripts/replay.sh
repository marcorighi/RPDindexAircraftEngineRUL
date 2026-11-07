#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo 'Missing isolated .venv; run setup_environment.sh' >&2; exit 3; }
DS="${1:-FD001}"
MODE="${2:-both}"
case "$DS" in FD001|FD002|FD003|FD004|all) ;; *) echo 'Dataset FD001|FD002|FD003|FD004|all'; exit 2;; esac
case "$MODE" in stage13|stage18|both) ;; *) echo 'Mode stage13|stage18|both'; exit 2;; esac
for ds in $([[ "$DS" == 'all' ]] && echo 'FD001 FD002 FD003 FD004' || echo "$DS"); do
 file="$ROOT/02_raw_data/$([[ "$ds" == FD001 ]] && echo 'PM_train.csv' || echo "PM_train_$ds.csv")"
 [[ -r "$file" ]] || { echo "Missing $file; run prepare_nasa_data.py" >&2; exit 4; }
 stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$_$ds"
 if [[ "$MODE" != stage18 ]]; then
  out="$ROOT/04_results/13_nested_alarm_${ds}_${stamp}"
  "$PY" "$ROOT/03_code/13_nested_alarm/stage13.py" --project-root "$ROOT" --dataset "$ds" --output "$out" --outer-folds 5 --inner-folds 4 --horizon 30 --min-lead 5 --min-prefix 8 --max-premature 0.20 --bootstrap 1000 --seed 20261103
  (cd "$out" && sha256sum -c SHA256SUMS)
  "$PY" "$ROOT/scripts/check_reference.py" stage13 "$ds" "$out"
  "$PY" "$ROOT/scripts/archive_result.py" "$out"
 fi
 if [[ "$MODE" != stage13 ]]; then
  out="$ROOT/04_results/18_rpd_components_${ds}_${stamp}"
  "$PY" "$ROOT/03_code/18_rpd_component_ablation/stage18.py" --project-root "$ROOT" --dataset "$ds" --output "$out" --outer-folds 5 --horizon 30 --min-prefix 8 --bootstrap 500 --seed 20261104
  (cd "$out" && sha256sum -c SHA256SUMS)
  "$PY" "$ROOT/scripts/check_reference.py" stage18 "$ds" "$out"
  "$PY" "$ROOT/scripts/archive_result.py" "$out"
 fi
done
