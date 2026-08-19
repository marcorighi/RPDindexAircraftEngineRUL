#!/usr/bin/env python3
"""
70_predict_pm_test.py

Generate one frozen-model RUL prediction for each engine in PM_test.csv.

This script is intentionally BLIND to PM_truth.csv.

Scientific workflow
-------------------
For each test engine:

    complete observed PM_test trajectory
        -> take raw sensor values at the final observed cycle
        -> compute the three GLOBAL RPD indices on the complete observed
           sensor trajectory
        -> take operating settings at the final observed cycle
        -> assemble the exact feature vector expected by the frozen model
        -> predict ONE RUL value for that engine

The fitted model is loaded from:
    04_results/07_models/RUL_best_model.joblib

No fitting, tuning, feature selection, threshold adjustment, or use of
PM_truth.csv occurs in this script.

Expected frozen model
---------------------
The bundle produced by 60_train_rul_model.py contains:
    estimator
    feature_set
    model_name
    feature_names
    global_indices
    sensors
    settings
    rul_cap
    prediction_min
    prediction_max
    ...

For the currently selected model this should normally be:
    raw_plus_rpd + ridge
with 87 features:
    21 raw sensor endpoint values
    63 global RPD endpoint values
    3 operating settings

Inputs
------
02_raw_data/
    PM_test.csv

03_code/
    computeindex_optimized.py

04_results/07_models/
    RUL_best_model.joblib
    RUL_best_model_metadata.json   (optional provenance check)

Outputs
-------
04_results/02_plot_data/
    70_PM_test_endpoint_features.csv.gz

04_results/05_tables/
    26_PM_test_RUL_predictions.csv
    70_pm_test_prediction_manifest.json

Important
---------
- PM_truth.csv is NEVER opened.
- The optimized RPD implementation is used exclusively.
- The output prediction file contains exactly one row per test engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
except Exception as exc:
    raise SystemExit(
        "This script requires joblib.\n"
        "Install it in the active Python environment, e.g.:\n"
        "    python -m pip install joblib\n"
        f"Original import error: {exc}"
    )

try:
    from computeindex_optimized import RPDgep, RPDgea, RPDgepr
except Exception as exc:
    raise SystemExit(
        "Cannot import RPDgep, RPDgea and RPDgepr from "
        "computeindex_optimized.py.\n"
        "Place this script in the same 03_code directory as the optimized "
        "implementation.\n"
        f"Original import error: {exc}"
    )


GLOBAL_FUNCTIONS = {
    "RPDgep": RPDgep,
    "RPDgea": RPDgea,
    "RPDgepr": RPDgepr,
}

DEFAULT_SENSORS = [f"s{i}" for i in range(1, 22)]
DEFAULT_SETTINGS = ["setting1", "setting2", "setting3"]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    p = argparse.ArgumentParser(
        description=(
            "Apply the frozen PM_train RUL model to PM_test.csv and produce "
            "one blind RUL prediction per engine."
        )
    )
    p.add_argument(
        "--test",
        type=Path,
        default=project_dir / "02_raw_data" / "PM_test.csv",
        help="PM_test.csv.",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=project_dir
        / "04_results"
        / "07_models"
        / "RUL_best_model.joblib",
        help="Frozen model bundle produced by 60_train_rul_model.py.",
    )
    p.add_argument(
        "--model-metadata",
        type=Path,
        default=project_dir
        / "04_results"
        / "07_models"
        / "RUL_best_model_metadata.json",
        help="Optional model metadata JSON for provenance checks.",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=project_dir / "04_results" / "05_tables",
        help="Output directory for the blind prediction table and manifest.",
    )
    p.add_argument(
        "--plot-data-dir",
        type=Path,
        default=project_dir / "04_results" / "02_plot_data",
        help="Output directory for endpoint feature data.",
    )
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_test(path: Path, sensors: list[str], settings: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["id", "cycle", *settings, *sensors]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing expected columns: {missing}"
        )

    if df[["id", "cycle"]].duplicated().any():
        bad = df.loc[
            df[["id", "cycle"]].duplicated(keep=False),
            ["id", "cycle"],
        ].head(10)
        raise ValueError(
            "Duplicate (id, cycle) rows found in PM_test.csv. Example:\n"
            f"{bad.to_string(index=False)}"
        )

    numeric_cols = ["id", "cycle", *settings, *sensors]
    arr = df[numeric_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "PM_test.csv contains non-finite values in required columns."
        )

    df = df.sort_values(["id", "cycle"]).reset_index(drop=True)

    # C-MAPSS trajectories should start at cycle 1 and be contiguous.
    for engine_id, g in df.groupby("id", sort=True):
        cycles = g["cycle"].to_numpy(dtype=int)
        expected = np.arange(cycles[0], cycles[-1] + 1, dtype=int)

        if cycles[0] != 1:
            raise ValueError(
                f"Engine {engine_id} starts at cycle {cycles[0]}, expected 1."
            )

        if not np.array_equal(cycles, expected):
            raise ValueError(
                f"Engine {engine_id} has a non-contiguous cycle sequence."
            )

        if len(cycles) < 2:
            raise ValueError(
                f"Engine {engine_id} has only {len(cycles)} observation(s); "
                "global RPD endpoint features require at least two cycles."
            )

    return df


def load_model_bundle(path: Path) -> dict:
    bundle = joblib.load(path)

    required = [
        "estimator",
        "feature_set",
        "model_name",
        "feature_names",
        "global_indices",
        "sensors",
        "settings",
        "prediction_min",
        "prediction_max",
    ]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise ValueError(
            f"Model bundle is missing required fields: {missing}"
        )

    feature_names = list(bundle["feature_names"])
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Model bundle contains duplicate feature names.")

    return bundle


def validate_model_definition(bundle: dict) -> None:
    global_indices = list(bundle["global_indices"])

    unsupported = [
        idx for idx in global_indices if idx not in GLOBAL_FUNCTIONS
    ]
    if unsupported:
        raise ValueError(
            "The frozen model requests global RPD indices not supported by "
            f"this prediction script: {unsupported}"
        )

    sensors = list(bundle["sensors"])
    if not sensors:
        raise ValueError("Frozen model contains an empty sensor list.")

    feature_set = str(bundle["feature_set"])
    supported_feature_sets = {
        "rpd_global",
        "raw_sensor",
        "raw_plus_rpd",
    }
    if feature_set not in supported_feature_sets:
        raise ValueError(
            f"Unsupported frozen feature set: {feature_set}"
        )


def compute_endpoint_feature_row(
    engine_id: int,
    g: pd.DataFrame,
    bundle: dict,
) -> dict:
    """
    Compute all potentially required endpoint features, then the caller keeps
    only the exact frozen-model feature list.
    """
    sensors = list(bundle["sensors"])
    settings = list(bundle["settings"])
    global_indices = list(bundle["global_indices"])

    g = g.sort_values("cycle")
    endpoint = g.iloc[-1]

    row: dict[str, float | int] = {
        "id": int(engine_id),
        "last_cycle": int(endpoint["cycle"]),
        "observed_length": int(len(g)),
    }

    # Raw endpoint sensor values.
    for sensor in sensors:
        row[f"raw_{sensor}"] = float(endpoint[sensor])

    # Operating settings at endpoint.
    for setting in settings:
        row[setting] = float(endpoint[setting])

    # Global RPD endpoint values from the complete observed trajectory.
    for sensor in sensors:
        values = g[sensor].to_numpy(dtype=float).tolist()

        for index_name in global_indices:
            value = GLOBAL_FUNCTIONS[index_name](values)

            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite {index_name} for engine={engine_id}, "
                    f"sensor={sensor}"
                )

            if value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
                raise ValueError(
                    f"{index_name} outside [-1,1] for engine={engine_id}, "
                    f"sensor={sensor}: {value}"
                )

            # Numerical clipping only at floating-point tolerance.
            value = min(1.0, max(-1.0, float(value)))
            row[f"{index_name}_{sensor}"] = value

    return row


def build_endpoint_features(
    test: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    rows = []

    engines = sorted(test["id"].unique())

    for position, engine_id in enumerate(engines, start=1):
        g = test[test["id"] == engine_id]
        row = compute_endpoint_feature_row(
            int(engine_id),
            g,
            bundle,
        )
        rows.append(row)

        if position % 20 == 0 or position == len(engines):
            print(
                f"[INFO] endpoint features {position}/{len(engines)} "
                f"(engine id={engine_id}, cycles={len(g)})"
            )

    features = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)

    if features["id"].duplicated().any():
        raise RuntimeError("Duplicate engine ids in endpoint feature table.")

    feature_names = list(bundle["feature_names"])
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(
            "Could not construct all frozen-model features. Missing: "
            f"{missing}"
        )

    X = features[feature_names].to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        raise ValueError("Endpoint feature matrix contains non-finite values.")

    return features


def predict_rul(
    features: pd.DataFrame,
    bundle: dict,
) -> np.ndarray:
    estimator = bundle["estimator"]
    feature_names = list(bundle["feature_names"])

    X = features[feature_names]

    pred = np.asarray(estimator.predict(X), dtype=float)

    if pred.shape != (len(features),):
        raise ValueError(
            f"Unexpected prediction shape: {pred.shape}"
        )

    if not np.all(np.isfinite(pred)):
        raise ValueError("Frozen model produced non-finite RUL predictions.")

    pred_min = float(bundle["prediction_min"])
    pred_max = float(bundle["prediction_max"])

    pred = np.clip(pred, pred_min, pred_max)
    return pred


def load_optional_metadata(path: Path) -> dict | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_metadata_against_bundle(
    metadata: dict | None,
    bundle: dict,
) -> None:
    if metadata is None:
        return

    checks = {
        "selected_feature_set": str(bundle["feature_set"]),
        "selected_model": str(bundle["model_name"]),
        "feature_count": len(bundle["feature_names"]),
    }

    for key, expected in checks.items():
        if key in metadata and metadata[key] != expected:
            raise ValueError(
                f"Model metadata mismatch for '{key}': "
                f"metadata={metadata[key]!r}, bundle={expected!r}"
            )


def main() -> int:
    args = parse_args()

    test_path = args.test.resolve()
    model_path = args.model.resolve()
    metadata_path = args.model_metadata.resolve()
    tables_dir = args.tables_dir.resolve()
    plot_data_dir = args.plot_data_dir.resolve()

    for path in [test_path, model_path]:
        if not path.exists():
            print(f"[ERROR] Missing input: {path}", file=sys.stderr)
            return 2

    tables_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "endpoint_features":
            plot_data_dir / "70_PM_test_endpoint_features.csv.gz",
        "predictions":
            tables_dir / "26_PM_test_RUL_predictions.csv",
        "manifest":
            tables_dir / "70_pm_test_prediction_manifest.json",
    }

    print("Blind PM_test RUL prediction")
    print(f"PM_test: {test_path}")
    print(f"Model:   {model_path}")
    print("PM_truth: NOT USED")
    print()

    try:
        bundle = load_model_bundle(model_path)
        validate_model_definition(bundle)

        metadata = load_optional_metadata(metadata_path)
        validate_metadata_against_bundle(metadata, bundle)

        sensors = list(bundle["sensors"])
        settings = list(bundle["settings"])

        test = read_test(
            test_path,
            sensors=sensors,
            settings=settings,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    engine_count = int(test["id"].nunique())
    print(
        f"[INFO] PM_test.csv: rows={len(test)}, "
        f"engines={engine_count}, "
        f"cycles/engine min={test.groupby('id').size().min()}, "
        f"mean={test.groupby('id').size().mean():.2f}, "
        f"max={test.groupby('id').size().max()}"
    )
    print(
        f"[INFO] frozen configuration: "
        f"{bundle['feature_set']} + {bundle['model_name']} "
        f"({len(bundle['feature_names'])} features)"
    )

    try:
        features = build_endpoint_features(
            test,
            bundle,
        )
        pred = predict_rul(
            features,
            bundle,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    predictions = features[
        ["id", "last_cycle", "observed_length"]
    ].copy()
    predictions["RUL_pred"] = pred

    # These are BLIND predictions. Do not add truth-derived columns here.
    predictions = predictions.sort_values("id").reset_index(drop=True)

    if len(predictions) != engine_count:
        print(
            "[ERROR] Prediction count does not equal test engine count.",
            file=sys.stderr,
        )
        return 2

    feature_output_cols = [
        "id",
        "last_cycle",
        "observed_length",
        *bundle["feature_names"],
    ]
    features[feature_output_cols].to_csv(
        files["endpoint_features"],
        index=False,
        compression="gzip",
    )
    predictions.to_csv(
        files["predictions"],
        index=False,
    )

    model_metadata_sha = (
        sha256(metadata_path) if metadata_path.exists() else None
    )

    computeindex_path = (
        Path(__file__).resolve().parent / "computeindex_optimized.py"
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pm_test_file": str(test_path),
        "pm_test_sha256": sha256(test_path),
        "model_file": str(model_path),
        "model_sha256": sha256(model_path),
        "model_metadata_file": (
            str(metadata_path) if metadata_path.exists() else None
        ),
        "model_metadata_sha256": model_metadata_sha,
        "computeindex_file": str(computeindex_path),
        "computeindex_sha256": (
            sha256(computeindex_path)
            if computeindex_path.exists()
            else None
        ),
        "engine_count": engine_count,
        "input_row_count": int(len(test)),
        "one_prediction_per_engine": True,
        "prediction_point":
            "final observed cycle of each PM_test engine trajectory",
        "feature_set": str(bundle["feature_set"]),
        "model_name": str(bundle["model_name"]),
        "feature_count": len(bundle["feature_names"]),
        "feature_names": list(bundle["feature_names"]),
        "global_indices": list(bundle["global_indices"]),
        "sensors": list(bundle["sensors"]),
        "settings": list(bundle["settings"]),
        "rul_cap_used_during_training": bundle.get("rul_cap"),
        "prediction_min": float(bundle["prediction_min"]),
        "prediction_max": float(bundle["prediction_max"]),
        "models_refit": False,
        "model_tuning_performed": False,
        "test_set_used": True,
        "truth_set_used": False,
        "pm_truth_opened": False,
        "software_doi": "10.5281/zenodo.21892421",
        "technical_report_doi": "10.5281/zenodo.21898684",
        "outputs": {
            key: str(value)
            for key, value in files.items()
        },
    }

    files["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"[OK] Endpoint features: {files['endpoint_features']}")
    print(f"[OK] Blind predictions: {files['predictions']}")
    print(f"[OK] Manifest:          {files['manifest']}")
    print()

    print("Blind PM_test predictions:")
    with pd.option_context(
        "display.max_rows", 120,
        "display.width", 120,
        "display.float_format", "{:.6g}".format,
    ):
        print(predictions.to_string(index=False))

    print()
    print(
        "PM_truth.csv has not been read. "
        "Do not modify the frozen model after inspecting these predictions."
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
