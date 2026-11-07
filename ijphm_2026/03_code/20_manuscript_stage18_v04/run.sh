#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$HERE/verify_manuscript.py"
if [[ -d "$HERE/tests" ]]; then
    python3 -m unittest discover -s "$HERE/tests" -v
else
    python3 -m unittest discover -s "$HERE/../tests" -v
fi
