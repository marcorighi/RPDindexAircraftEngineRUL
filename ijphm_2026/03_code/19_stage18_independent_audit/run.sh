#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ART="$(cd "$HERE/../.." && pwd)"
RESULTS="$ART/04_results"
ZIPDIR="$RESULTS/ZIP"
mkdir -p "$ZIPDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
NAME="19_stage18_reaudit_$STAMP"
OUT="$RESULTS/$NAME"
[ ! -e "$OUT" ] || { echo "Output already exists: $OUT" >&2; exit 3; }
[ ! -e "$ZIPDIR/$NAME.zip" ] || { echo "ZIP already exists: $NAME" >&2; exit 3; }
python3 "$HERE/verify_stage18.py" --zip-dir "$ZIPDIR" --output "$OUT"
(
  cd "$RESULTS"
  zip -qr "ZIP/$NAME.zip" "$NAME"
  cd ZIP
  unzip -tq "$NAME.zip"
  sha256sum "$NAME.zip" > "$NAME.zip.sha256"
  sha256sum -c "$NAME.zip.sha256"
)
echo "ZIP: $ZIPDIR/$NAME.zip"
echo "SHA: $ZIPDIR/$NAME.zip.sha256"
