#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo 'Missing isolated .venv; run scripts/setup_environment.sh' >&2; exit 3; }
mkdir -p "$ROOT/04_results/test_logs"
for entry in 'S10:10_failure_proximity' 'S13:13_nested_alarm' 'S18:18_rpd_component_ablation' 'MATH:20_manuscript_stage18_v04'; do
 name="${entry%%:*}"; dir="${entry#*:}"
 echo "== $name =="
 "$PY" -m unittest discover -s "$ROOT/03_code/$dir/tests" -v > "$ROOT/04_results/test_logs/$name.log" 2>&1 || { tail -40 "$ROOT/04_results/test_logs/$name.log"; exit 1; }
 grep -E 'Ran [0-9]+ tests|^OK$' "$ROOT/04_results/test_logs/$name.log"
done
