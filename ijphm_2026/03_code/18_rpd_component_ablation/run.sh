#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ART="$(cd -- "$HERE/../.." && pwd -P)"
DATASET="${1:-FD001}"
case "$DATASET" in FD001|FD002|FD003|FD004|all) ;; *) echo 'Uso: run.sh FD001|FD002|FD003|FD004|all' >&2; exit 2;; esac
for tool in python3 zip unzip sha256sum; do command -v "$tool" >/dev/null || { echo "Manca: $tool" >&2; exit 3; }; done
mkdir -p "$ART/04_results/ZIP"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
CURRENT=()
run_one () {
 local d="$1" raw filename out archive
 case "$d" in
   FD001) filename='PM_train.csv';;
   FD002) filename='PM_train_FD002.csv';;
   FD003) filename='PM_train_FD003.csv';;
   FD004) filename='PM_train_FD004.csv';;
 esac
 raw="$ART/02_raw_data/$filename"
 [[ -r "$raw" ]] || { echo "Manca dataset: $raw" >&2; return 4; }
 out="18_rpd_components_${d}_${STAMP}"
 archive="$ART/04_results/ZIP/$out.zip"
 [[ ! -e "$ART/04_results/$out" && ! -e "$archive" ]] || { echo "Esiste gia': $out" >&2; return 5; }
 python3 "$HERE/stage18.py" --project-root "$ART" --dataset "$d" \
   --output "$ART/04_results/$out" --outer-folds 5 --horizon 30 --min-prefix 8 --bootstrap 500 --seed 20261104
 (cd "$ART/04_results/$out" && sha256sum -c SHA256SUMS)
 (cd "$ART/04_results" && zip -qr "ZIP/$out.zip" "$out")
 unzip -tq "$archive"
 (cd "$ART/04_results/ZIP" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256" && sha256sum -c "$(basename "$archive").sha256")
 CURRENT+=("$ART/04_results/$out")
 echo "ZIP $d: $archive"
}
if [[ "$DATASET" == all ]]; then
 for D in FD001 FD002 FD003 FD004; do run_one "$D"; done
 summary="18_rpd_components_four_dataset_${STAMP}"
 mkdir "$ART/04_results/$summary"
 python3 "$HERE/aggregate.py" --inputs "${CURRENT[@]}" --out "$ART/04_results/$summary"
 (cd "$ART/04_results/$summary" && sha256sum -c SHA256SUMS)
 (cd "$ART/04_results" && zip -qr "ZIP/$summary.zip" "$summary")
 unzip -tq "$ART/04_results/ZIP/$summary.zip"
 (cd "$ART/04_results/ZIP" && sha256sum "$summary.zip" > "$summary.zip.sha256" && sha256sum -c "$summary.zip.sha256")
 echo "ZIP RIEPILOGO DA ALLEGARE: $ART/04_results/ZIP/$summary.zip"
else
 run_one "$DATASET"
fi
