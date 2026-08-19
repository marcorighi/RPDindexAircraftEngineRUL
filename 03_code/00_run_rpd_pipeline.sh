#!/usr/bin/env bash
set -Eeuo pipefail

# RPD pipeline for C-MAPSS / PM_train.csv
# Place this file in 03_code/ and run it from anywhere:
#   bash 03_code/00_run_rpd_pipeline.sh
# Optional:
#   bash 03_code/00_run_rpd_pipeline.sh --clean

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$PROJECT_DIR/02_raw_data"
RESULTS_DIR="$PROJECT_DIR/04_results"
INPUT="$RAW_DIR/PM_train.csv"
COMPUTEINDEX="$SCRIPT_DIR/computeindex_optimized.py"
SOFTWARE_DOI="10.5281/zenodo.21892421"
TECHNICAL_REPORT_DOI="10.5281/zenodo.21898684"

INDEX_DIR="$RESULTS_DIR/01_indices"
BY_ENGINE_DIR="$INDEX_DIR/by_engine"
PLOT_DATA_DIR="$RESULTS_DIR/02_plot_data"
FIG_DIR="$RESULTS_DIR/03_figures"
HIRES_DIR="$RESULTS_DIR/04_figures_hires"
TABLE_DIR="$RESULTS_DIR/05_tables"
LOG_DIR="$RESULTS_DIR/06_logs"

if [[ "${1:-}" == "--clean" ]]; then
    echo "Removing previous calculated results..."
    rm -rf "$INDEX_DIR" "$PLOT_DATA_DIR" "$FIG_DIR" "$HIRES_DIR" "$TABLE_DIR" "$LOG_DIR"
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--clean]" >&2
    exit 2
fi

mkdir -p "$BY_ENGINE_DIR" "$PLOT_DATA_DIR" "$FIG_DIR" "$HIRES_DIR" "$TABLE_DIR" "$LOG_DIR"

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "ERROR: missing $INPUT" >&2; exit 1; }
[[ -f "$COMPUTEINDEX" ]] || { echo "ERROR: missing $COMPUTEINDEX" >&2; exit 1; }

python3 - <<'PY' || {
import pandas, numpy
PY
    echo "ERROR: Python packages pandas and numpy are required." >&2
    echo "Install them with your distribution/package environment before retrying." >&2
    exit 1
}

STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/run_${STAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "RPD C-MAPSS pipeline"
echo "Project: $PROJECT_DIR"
echo "Input:   $INPUT"
echo "Results: $RESULTS_DIR"
echo

export RPD_PROJECT_DIR="$PROJECT_DIR"
export RPD_INPUT="$INPUT"
export RPD_COMPUTEINDEX="$COMPUTEINDEX"
export RPD_SOFTWARE_DOI="$SOFTWARE_DOI"
export RPD_TECHNICAL_REPORT_DOI="$TECHNICAL_REPORT_DOI"
export RPD_INDEX_DIR="$INDEX_DIR"
export RPD_BY_ENGINE_DIR="$BY_ENGINE_DIR"
export RPD_TABLE_DIR="$TABLE_DIR"

python3 <<'PY'
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(os.environ["RPD_PROJECT_DIR"])
INPUT = Path(os.environ["RPD_INPUT"])
COMPUTEINDEX = Path(os.environ["RPD_COMPUTEINDEX"])
INDEX_DIR = Path(os.environ["RPD_INDEX_DIR"])
BY_ENGINE_DIR = Path(os.environ["RPD_BY_ENGINE_DIR"])
TABLE_DIR = Path(os.environ["RPD_TABLE_DIR"])
SOFTWARE_DOI = os.environ["RPD_SOFTWARE_DOI"]
TECHNICAL_REPORT_DOI = os.environ["RPD_TECHNICAL_REPORT_DOI"]

INDEX_NAMES = ["RPDlep", "RPDlea", "RPDlepr", "RPDgep", "RPDgea", "RPDgepr"]
SENSORS = [f"s{i}" for i in range(1, 22)]


def load_computeindex_module(path: Path):
    spec = importlib.util.spec_from_file_location("computeindex_optimized", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


computeindex = load_computeindex_module(COMPUTEINDEX)

missing_api = [
    name for name in INDEX_NAMES + ["progressive_indices"]
    if not hasattr(computeindex, name)
]
if missing_api:
    raise RuntimeError(
        f"{COMPUTEINDEX.name} is missing the required API: {missing_api}"
    )

progressive_indices = computeindex.progressive_indices


def validate_optimized_module() -> None:
    """Fast deterministic sanity checks for the optimized implementation."""
    tests = [
        (
            np.array([1.0, 2.0, 4.0, 6.0]),
            1.0,
            "strictly increasing"
        ),
        (
            np.array([6.0, 4.0, 2.0, 1.0]),
            -1.0,
            "strictly decreasing"
        ),
        (
            np.array([1.0, 1.0, 1.0, 1.0]),
            0.0,
            "constant"
        ),
    ]

    for x, expected, description in tests:
        trajectories = progressive_indices(x)

        for name in INDEX_NAMES:
            values = np.asarray(trajectories[name], dtype=float)

            if values.shape != (len(x) - 1,):
                raise RuntimeError(
                    f"Validation failed for {name} ({description}): "
                    f"expected {len(x)-1} progressive values, got {values.shape}"
                )

            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Validation failed for {name} ({description}): "
                    "non-finite value found"
                )

            if not np.allclose(values, expected, rtol=1e-12, atol=1e-12):
                raise RuntimeError(
                    f"Validation failed for {name} ({description}): "
                    f"expected {expected}, got {values}"
                )

            if np.any(values < -1.0 - 1e-12) or np.any(values > 1.0 + 1e-12):
                raise RuntimeError(
                    f"Validation failed for {name}: value outside [-1,1]"
                )

    # Mixed sequence: check the progressive API against the six public
    # optimized functions themselves for every prefix. This detects divergence
    # between the optimized single-value and progressive implementations.
    x = np.array(
        [1.0, 3.0, 2.0, 5.0, 4.0, 4.0,
         0.0183179687636068, 0.66165806279797, 1.0,
         6.51487788578436e-10],
        dtype=float,
    )
    trajectories = progressive_indices(x)

    for name in INDEX_NAMES:
        func = getattr(computeindex, name)
        direct = np.array(
            [func(x[:k].tolist()) for k in range(2, len(x) + 1)],
            dtype=float,
        )
        progressive = np.asarray(trajectories[name], dtype=float)

        if not np.allclose(
            progressive, direct, rtol=1e-12, atol=1e-12, equal_nan=True
        ):
            raise RuntimeError(
                f"Progressive/direct validation failed for {name}\n"
                f"progressive={progressive}\n"
                f"direct={direct}"
            )

    print(
        "[OK] computeindex_optimized.py passed deterministic "
        "and progressive/direct validation."
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


start = time.time()
validate_optimized_module()

df = pd.read_csv(INPUT)
expected = ["id", "cycle", "setting1", "setting2", "setting3"] + SENSORS
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in {INPUT.name}: {missing}")

if df[["id", "cycle"]].duplicated().any():
    raise ValueError("Duplicate (id, cycle) rows found in training data")

ids = sorted(df["id"].unique().tolist())
print(f"[INFO] {INPUT.name}: {len(df)} rows, {len(ids)} engines, {len(SENSORS)} sensors")
print(f"[INFO] cycles/engine: min={df.groupby('id').size().min()}, "
      f"mean={df.groupby('id').size().mean():.2f}, max={df.groupby('id').size().max()}")

long_path = INDEX_DIR / "PM_train_RPD_progressive.csv.gz"
header = ["id", "cycle", "RUL", "sensor", "sensor_value"] + INDEX_NAMES

summary_rows = []

with gzip.open(long_path, "wt", newline="") as gz:
    writer = csv.writer(gz)
    writer.writerow(header)

    for pos_engine, engine_id in enumerate(ids, start=1):
        eng = df[df["id"] == engine_id].sort_values("cycle")
        cycles = eng["cycle"].to_numpy(dtype=int)
        final_cycle = int(cycles[-1])
        if len(cycles) < 2:
            print(f"[WARN] engine {engine_id}: fewer than 2 cycles; skipped")
            continue

        # Collect matrices for legacy-compatible by-engine files:
        # 21 rows (sensors) x (T-1) progressive prefixes.
        matrices = {name: [] for name in INDEX_NAMES}

        for sensor in SENSORS:
            x = eng[sensor].to_numpy(dtype=float)
            out = progressive_indices(x)
            for name in INDEX_NAMES:
                matrices[name].append(out[name])

            for k in range(1, len(cycles)):
                cycle = int(cycles[k])
                rul = final_cycle - cycle
                writer.writerow([
                    int(engine_id), cycle, rul, sensor, float(x[k]),
                    *(float(out[name][k - 1]) for name in INDEX_NAMES),
                ])

            # Compact endpoint summary, useful for sanity checks and later tables.
            summary_rows.append({
                "id": int(engine_id),
                "sensor": sensor,
                "final_cycle": final_cycle,
                **{name: float(out[name][-1]) for name in INDEX_NAMES},
            })

        for name in INDEX_NAMES:
            arr = np.vstack(matrices[name])
            out_file = BY_ENGINE_DIR / f"{engine_id}_{name}_id.csv"
            pd.DataFrame(arr).to_csv(out_file, index=False, header=False)

        if pos_engine == 1 or pos_engine % 10 == 0 or pos_engine == len(ids):
            print(f"[INFO] processed engine {pos_engine}/{len(ids)} (id={engine_id}, cycles={len(cycles)})")

summary = pd.DataFrame(summary_rows)
summary_path = TABLE_DIR / "PM_train_RPD_endpoint_by_engine_sensor.csv"
summary.to_csv(summary_path, index=False)

# Dataset inventory / reproducibility metadata.
manifest = {
    "input_file": str(INPUT.relative_to(PROJECT)),
    "input_sha256": sha256(INPUT),
    "computeindex_file": str(COMPUTEINDEX.relative_to(PROJECT)),
    "computeindex_sha256": sha256(COMPUTEINDEX),
    "software_doi": SOFTWARE_DOI,
    "technical_report_doi": TECHNICAL_REPORT_DOI,
    "n_rows": int(len(df)),
    "n_engines": int(len(ids)),
    "n_sensors": len(SENSORS),
    "indices": INDEX_NAMES,
    "progressive_definition": "For each engine/sensor, values are computed on prefixes x[0:2], ..., x[0:T].",
    "rul_train_definition": "RUL = final_cycle_of_engine - current_cycle; final training cycle has RUL=0.",
    "long_output": str(long_path.relative_to(PROJECT)),
    "endpoint_output": str(summary_path.relative_to(PROJECT)),
    "legacy_by_engine_output": str(BY_ENGINE_DIR.relative_to(PROJECT)),
    "elapsed_seconds": round(time.time() - start, 3),
}
manifest_path = INDEX_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

print()
print(f"[OK] Progressive dataset: {long_path}")
print(f"[OK] Per-engine index files: {BY_ENGINE_DIR}")
print(f"[OK] Endpoint summary: {summary_path}")
print(f"[OK] Manifest: {manifest_path}")
print(f"[OK] elapsed: {manifest['elapsed_seconds']:.1f} s")
PY

cat > "$LOG_DIR/last_run.txt" <<EOF
log=$LOG_FILE
input=$INPUT
output=$INDEX_DIR/PM_train_RPD_progressive.csv.gz
EOF

cat > "$RESULTS_DIR/README_results.txt" <<'EOF'
04_results/
  01_indices/
    PM_train_RPD_progressive.csv.gz
      Long-format progressive RPD dataset.
      Columns: id, cycle, RUL, sensor, sensor_value,
               RPDlep, RPDlea, RPDlepr, RPDgep, RPDgea, RPDgepr
    by_engine/
      Compatibility files: 21 sensor rows x progressive-cycle columns,
      one file for each engine and RPD index.
    manifest.json
      Input/code hashes, software/report DOI, and execution metadata.

  02_plot_data/
      Reserved for plot-ready tables/data.

  03_figures/
      Figures for inspection/drafting.

  04_figures_hires/
      Final publication-quality figures.

  05_tables/
    PM_train_RPD_endpoint_by_engine_sensor.csv
      Final RPD value for every engine/sensor/index.

  06_logs/
      Pipeline execution logs.
EOF

echo
echo "Done."
echo "Log: $LOG_FILE"
