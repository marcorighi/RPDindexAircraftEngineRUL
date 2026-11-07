#!/usr/bin/env bash
set -Eeuo pipefail
STAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ART="$(cd -- "$STAGE_DIR/../.." && pwd -P)"
DATASET="${1:-FD001}"
case "$DATASET" in FD001|FD002|FD003|FD004);; *) echo "Uso: $0 FD001|FD002|FD003|FD004 [risultati_directory]" >&2; exit 2;; esac
if [[ -n "${2:-}" ]]; then
    SRC="$(realpath -- "$2")"
else
    mapfile -t CANDIDATES < <(find "$ART/04_results" -maxdepth 1 -type d -name "13_nested_alarm_${DATASET}_*" -print | sort)
    if [[ ${#CANDIDATES[@]} -ne 1 ]]; then
        echo "ERRORE: attese esattamente 1 directory Stage13 per $DATASET, trovate ${#CANDIDATES[@]}. Passare la directory esplicita." >&2
        exit 3
    fi
    SRC="${CANDIDATES[0]}"
fi
[[ -f "$SRC/SHA256SUMS" && "$SRC" == "$ART/04_results/13_nested_alarm_${DATASET}_"* ]] || { echo "Directory o SHA256SUMS non validi: $SRC" >&2; exit 4; }
(cd "$SRC" && sha256sum -c SHA256SUMS)
mkdir -p "$ART/04_results/ZIP"
OUT="$ART/04_results/ZIP/$(basename -- "$SRC").zip"
[[ ! -e "$OUT" ]] || { echo "Archivio gia esistente: $OUT" >&2; exit 5; }
python3 - "$SRC" "$OUT" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import os,sys,hashlib
src,out=Path(sys.argv[1]),Path(sys.argv[2])
tmp=out.with_name('.'+out.name+'.partial')
if tmp.exists():raise SystemExit('Temporary archive exists')
try:
    with ZipFile(tmp,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for f in sorted(src.iterdir()):
            if f.is_file():archive.write(f,arcname=f.name)
    os.replace(tmp,out)
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(out.suffix+'.sha256').write_text(f'{digest}  {out}\n')
    print('ZIP:',out,'\nSHA256:',out.with_suffix(out.suffix+'.sha256'))
finally:
    if tmp.exists():tmp.unlink()
PY
