#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -e "$ROOT/.venv" ]]; then echo 'Refusing to replace existing .venv' >&2; exit 3; fi
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements-observed.txt"
"$ROOT/.venv/bin/python" - <<'PY2'
import json,sys,numpy,pandas,scipy,sklearn
print(json.dumps({'python':sys.version,'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},indent=2))
PY2
