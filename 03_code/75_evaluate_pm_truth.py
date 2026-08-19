#!/usr/bin/env python3
"""
75_evaluate_pm_truth.py

Final evaluation of the previously frozen blind PM_test RUL predictions against
PM_truth.csv.

This script MUST be run only after 70_predict_pm_test.py has generated the blind
prediction file. It never modifies or regenerates predictions.

Inputs
------
02_raw_data/
    PM_truth.csv

04_results/05_tables/
    26_PM_test_RUL_predictions.csv
    70_pm_test_prediction_manifest.json   (optional but recommended)

Outputs
-------
04_results/05_tables/
    27_PM_test_RUL_evaluation_by_engine.csv
    28_PM_test_RUL_metrics.csv
    29_PM_test_RUL_error_quantiles.csv
    75_pm_truth_evaluation_manifest.json

Metrics
-------
Let:
    error = predicted RUL - true RUL

The script computes:
- RMSE
- MAE
- R^2
- mean and median signed error (bias)
- PHM08/NASA asymmetric score, total and mean
- fraction of RUL overestimates
- fraction of RUL underestimates
- 95th percentile absolute error
- maximum absolute error
- 95th percentile late/positive error
- maximum late/positive error

PHM08/NASA asymmetric score
---------------------------
    error < 0  : exp(-error / 13) - 1
    error >= 0 : exp( error / 10) - 1

Therefore late predictions (RUL_pred > RUL_true) are penalized more heavily.

Truth-file handling
-------------------
The script supports two common layouts:

1) Explicit ID and truth columns, e.g.
       id,RUL
       1,112
       ...

2) One truth value per row with no engine ID.
   In this case row 1 is mapped to the smallest prediction engine ID, row 2
   to the next ID, etc. This is the standard ordering used by C-MAPSS truth
   files. The script requires the number of rows to match exactly.

Safety/reproducibility
----------------------
- The blind prediction file is read-only.
- SHA-256 hashes are recorded for predictions, truth, and the blind-prediction
  manifest when available.
- Exactly one prediction and one truth value per engine are required.
- No fitting, tuning, clipping, recalibration, or post-hoc correction occurs.
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
    from sklearn.metrics import r2_score
    import sklearn
except Exception as exc:
    raise SystemExit(
        "This script requires scikit-learn.\n"
        "Install it in the active Python environment, e.g.:\n"
        "    python -m pip install scikit-learn\n"
        f"Original import error: {exc}"
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    p = argparse.ArgumentParser(
        description=(
            "Evaluate frozen blind PM_test RUL predictions against PM_truth.csv."
        )
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=project_dir
        / "04_results"
        / "05_tables"
        / "26_PM_test_RUL_predictions.csv",
        help="Blind predictions produced by 70_predict_pm_test.py.",
    )
    p.add_argument(
        "--truth",
        type=Path,
        default=project_dir / "02_raw_data" / "PM_truth.csv",
        help="Ground-truth residual RUL file.",
    )
    p.add_argument(
        "--prediction-manifest",
        type=Path,
        default=project_dir
        / "04_results"
        / "05_tables"
        / "70_pm_test_prediction_manifest.json",
        help="Optional provenance manifest from 70_predict_pm_test.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "04_results" / "05_tables",
        help="Directory for final evaluation outputs.",
    )
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def nasa_penalty(error: np.ndarray) -> np.ndarray:
    """
    error = predicted RUL - true RUL.
    """
    error = np.asarray(error, dtype=float)

    out = np.empty_like(error, dtype=float)
    early = error < 0

    out[early] = np.expm1((-error[early]) / 13.0)
    out[~early] = np.expm1(error[~early] / 10.0)
    return out


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["id", "RUL_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}"
        )

    if df["id"].isna().any() or df["RUL_pred"].isna().any():
        raise ValueError("Prediction file contains missing id/RUL_pred values.")

    if df["id"].duplicated().any():
        bad = df.loc[df["id"].duplicated(keep=False), ["id", "RUL_pred"]]
        raise ValueError(
            "Prediction file contains duplicate engine IDs:\n"
            f"{bad.to_string(index=False)}"
        )

    df["id"] = df["id"].astype(int)
    df["RUL_pred"] = df["RUL_pred"].astype(float)

    if not np.all(np.isfinite(df["RUL_pred"].to_numpy(dtype=float))):
        raise ValueError("Prediction file contains non-finite RUL predictions.")

    if (df["RUL_pred"] < 0).any():
        raise ValueError(
            "Prediction file contains negative RUL values. "
            "The blind prediction stage should already have applied its frozen "
            "prediction clipping."
        )

    return df.sort_values("id").reset_index(drop=True)


def find_column_case_insensitive(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    lookup = {str(c).strip().lower(): c for c in columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]

    return None


def read_truth(
    path: Path,
    prediction_ids: np.ndarray,
) -> tuple[pd.DataFrame, str]:
    """
    Return columns:
        id, RUL_true

    mapping_mode describes how PM_truth was interpreted.
    """
    # First try the normal CSV parser.
    truth = pd.read_csv(path)

    # Remove completely empty columns sometimes created by trailing separators.
    truth = truth.dropna(axis=1, how="all")

    if truth.empty:
        raise ValueError(f"{path.name} contains no truth rows.")

    columns = list(truth.columns)

    id_col = find_column_case_insensitive(
        columns,
        ["id", "engine_id", "engine", "unit", "unit_id"],
    )
    rul_col = find_column_case_insensitive(
        columns,
        [
            "RUL",
            "RUL_true",
            "true_RUL",
            "truth",
            "remaining_useful_life",
            "remaining useful life",
            # In this project's PM_truth.csv the residual RUL column is
            # named "cycle". Accept it explicitly when an engine id column
            # is also present.
            "cycle",
        ],
    )

    # Case A: explicit ID + truth column.
    if id_col is not None and rul_col is not None:
        out = truth[[id_col, rul_col]].copy()
        out.columns = ["id", "RUL_true"]
        mapping_mode = (
            f"explicit columns: id='{id_col}', RUL_true='{rul_col}'"
        )

    # Case B: exactly one parsed column. It may either have a real header or
    # pandas may have interpreted the first numeric truth value as a header.
    elif truth.shape[1] == 1:
        col = truth.columns[0]

        # Detect a numeric-looking column name, which means a headerless CSV
        # was probably read with the first truth value as the header.
        numeric_header = False
        try:
            float(str(col).strip())
            numeric_header = True
        except ValueError:
            numeric_header = False

        if numeric_header:
            # Re-read without header so the first truth value is retained.
            truth_no_header = pd.read_csv(path, header=None)
            truth_no_header = truth_no_header.dropna(axis=1, how="all")

            if truth_no_header.shape[1] != 1:
                raise ValueError(
                    "PM_truth appears headerless but does not contain exactly "
                    "one truth column."
                )

            values = truth_no_header.iloc[:, 0]
            mapping_mode = (
                "headerless single-column truth; rows mapped to sorted "
                "prediction engine IDs"
            )
        else:
            values = truth.iloc[:, 0]
            mapping_mode = (
                f"single-column truth '{col}'; rows mapped to sorted "
                "prediction engine IDs"
            )

        values = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)

        if len(values) != len(prediction_ids):
            raise ValueError(
                "Single-column truth row count does not match prediction "
                f"engine count: truth={len(values)}, "
                f"predictions={len(prediction_ids)}"
            )

        out = pd.DataFrame(
            {
                "id": prediction_ids.astype(int),
                "RUL_true": values,
            }
        )

    # Case C: two or more columns but no recognized names.
    # A frequent layout is simply "id,truth" without meaningful headers.
    # We deliberately do NOT guess here.
    else:
        raise ValueError(
            "Could not unambiguously identify PM_truth columns.\n"
            f"Columns found: {columns}\n"
            "Supported layouts are either explicit id/RUL columns or a single "
            "truth value per row."
        )

    if out["id"].isna().any() or out["RUL_true"].isna().any():
        raise ValueError("PM_truth contains missing id/RUL values.")

    out["id"] = pd.to_numeric(out["id"], errors="raise").astype(int)
    out["RUL_true"] = pd.to_numeric(
        out["RUL_true"],
        errors="raise",
    ).astype(float)

    if out["id"].duplicated().any():
        raise ValueError("PM_truth contains duplicate engine IDs.")

    if not np.all(np.isfinite(out["RUL_true"].to_numpy(dtype=float))):
        raise ValueError("PM_truth contains non-finite RUL values.")

    if (out["RUL_true"] < 0).any():
        raise ValueError("PM_truth contains negative RUL values.")

    out = out.sort_values("id").reset_index(drop=True)

    prediction_id_set = set(map(int, prediction_ids))
    truth_id_set = set(map(int, out["id"].to_numpy()))

    missing_truth = sorted(prediction_id_set - truth_id_set)
    extra_truth = sorted(truth_id_set - prediction_id_set)

    if missing_truth or extra_truth:
        raise ValueError(
            "Prediction/truth engine ID mismatch.\n"
            f"Missing truth IDs: {missing_truth}\n"
            f"Extra truth IDs: {extra_truth}"
        )

    if len(out) != len(prediction_ids):
        raise ValueError(
            f"Truth row count ({len(out)}) does not match prediction "
            f"engine count ({len(prediction_ids)})."
        )

    return out, mapping_mode


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    abs_error = np.abs(error)
    late_error = np.maximum(error, 0.0)
    nasa = nasa_penalty(error)

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(abs_error))

    if np.ptp(y_true) == 0:
        r2 = np.nan
    else:
        r2 = float(r2_score(y_true, y_pred))

    return {
        "engine_count": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_error": float(np.mean(error)),
        "median_error": float(np.median(error)),
        "mean_abs_error": mae,
        "median_abs_error": float(np.median(abs_error)),
        "nasa_score_sum": float(np.sum(nasa)),
        "nasa_score_mean": float(np.mean(nasa)),
        "overestimate_fraction": float(np.mean(error > 0)),
        "underestimate_fraction": float(np.mean(error < 0)),
        "exact_fraction": float(np.mean(error == 0)),
        "p95_abs_error": float(np.quantile(abs_error, 0.95)),
        "max_abs_error": float(np.max(abs_error)),
        "p95_late_error": float(np.quantile(late_error, 0.95)),
        "max_late_error": float(np.max(late_error)),
        "mean_true_rul": float(np.mean(y_true)),
        "median_true_rul": float(np.median(y_true)),
        "mean_pred_rul": float(np.mean(y_pred)),
        "median_pred_rul": float(np.median(y_pred)),
    }


def quantile_table(
    error: np.ndarray,
) -> pd.DataFrame:
    error = np.asarray(error, dtype=float)
    abs_error = np.abs(error)
    late_error = np.maximum(error, 0.0)
    nasa = nasa_penalty(error)

    quantiles = [
        0.00,
        0.025,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.975,
        1.00,
    ]

    rows = []
    for q in quantiles:
        rows.append(
            {
                "quantile": q,
                "signed_error": float(np.quantile(error, q)),
                "abs_error": float(np.quantile(abs_error, q)),
                "late_error": float(np.quantile(late_error, q)),
                "nasa_penalty": float(np.quantile(nasa, q)),
            }
        )

    return pd.DataFrame(rows)


def load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_prediction_manifest(
    manifest: dict | None,
    prediction_path: Path,
) -> None:
    if manifest is None:
        return

    if manifest.get("truth_set_used") is True:
        raise ValueError(
            "Blind prediction manifest unexpectedly states "
            "'truth_set_used: true'."
        )

    if manifest.get("pm_truth_opened") is True:
        raise ValueError(
            "Blind prediction manifest unexpectedly states "
            "'pm_truth_opened: true'."
        )

    expected_prediction_path = (
        manifest.get("outputs", {}).get("predictions")
    )
    if expected_prediction_path:
        expected_name = Path(expected_prediction_path).name
        if expected_name != prediction_path.name:
            raise ValueError(
                "Prediction manifest refers to a different prediction file: "
                f"{expected_name} vs {prediction_path.name}"
            )


def main() -> int:
    args = parse_args()

    prediction_path = args.predictions.resolve()
    truth_path = args.truth.resolve()
    prediction_manifest_path = args.prediction_manifest.resolve()
    output_dir = args.output_dir.resolve()

    for path in [prediction_path, truth_path]:
        if not path.exists():
            print(f"[ERROR] Missing input: {path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "by_engine":
            output_dir / "27_PM_test_RUL_evaluation_by_engine.csv",
        "metrics":
            output_dir / "28_PM_test_RUL_metrics.csv",
        "quantiles":
            output_dir / "29_PM_test_RUL_error_quantiles.csv",
        "manifest":
            output_dir / "75_pm_truth_evaluation_manifest.json",
    }

    print("Final PM_test RUL evaluation against PM_truth")
    print(f"Blind predictions: {prediction_path}")
    print(f"PM_truth:          {truth_path}")
    print()

    try:
        pred_manifest = load_optional_json(
            prediction_manifest_path
        )
        validate_prediction_manifest(
            pred_manifest,
            prediction_path,
        )

        predictions = read_predictions(prediction_path)

        truth, truth_mapping_mode = read_truth(
            truth_path,
            predictions["id"].to_numpy(dtype=int),
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    evaluation = predictions.merge(
        truth,
        on="id",
        how="inner",
        validate="one_to_one",
    ).sort_values("id").reset_index(drop=True)

    if len(evaluation) != len(predictions):
        print(
            "[ERROR] Evaluation merge changed the number of engines.",
            file=sys.stderr,
        )
        return 2

    evaluation["error"] = (
        evaluation["RUL_pred"] - evaluation["RUL_true"]
    )
    evaluation["abs_error"] = np.abs(evaluation["error"])
    evaluation["squared_error"] = evaluation["error"] ** 2
    evaluation["late_error"] = np.maximum(
        evaluation["error"],
        0.0,
    )
    evaluation["overestimate"] = evaluation["error"] > 0
    evaluation["underestimate"] = evaluation["error"] < 0
    evaluation["nasa_penalty"] = nasa_penalty(
        evaluation["error"].to_numpy(dtype=float)
    )

    y_true = evaluation["RUL_true"].to_numpy(dtype=float)
    y_pred = evaluation["RUL_pred"].to_numpy(dtype=float)

    metrics = compute_metrics(y_true, y_pred)
    metrics_row = {
        "evaluation": "PM_test_vs_PM_truth",
        **metrics,
    }
    metrics_df = pd.DataFrame([metrics_row])

    quantiles_df = quantile_table(
        evaluation["error"].to_numpy(dtype=float)
    )

    evaluation.to_csv(files["by_engine"], index=False)
    metrics_df.to_csv(files["metrics"], index=False)
    quantiles_df.to_csv(files["quantiles"], index=False)

    prediction_manifest_sha = (
        sha256(prediction_manifest_path)
        if prediction_manifest_path.exists()
        else None
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "blind_predictions_file": str(prediction_path),
        "blind_predictions_sha256": sha256(prediction_path),
        "pm_truth_file": str(truth_path),
        "pm_truth_sha256": sha256(truth_path),
        "blind_prediction_manifest_file": (
            str(prediction_manifest_path)
            if prediction_manifest_path.exists()
            else None
        ),
        "blind_prediction_manifest_sha256": prediction_manifest_sha,
        "truth_mapping_mode": truth_mapping_mode,
        "engine_count": int(len(evaluation)),
        "one_prediction_per_engine": True,
        "one_truth_value_per_engine": True,
        "predictions_modified": False,
        "models_refit": False,
        "model_tuning_performed": False,
        "posthoc_calibration_performed": False,
        "test_set_used": True,
        "truth_set_used": True,
        "nasa_score_definition":
            "error=predicted-true; error<0: exp(-error/13)-1; "
            "error>=0: exp(error/10)-1",
        "metrics": metrics,
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

    print(
        f"[INFO] truth mapping: {truth_mapping_mode}"
    )
    print(
        f"[INFO] engines evaluated: {len(evaluation)}"
    )
    print()

    print(f"[OK] Per-engine evaluation: {files['by_engine']}")
    print(f"[OK] Final metrics:         {files['metrics']}")
    print(f"[OK] Error quantiles:       {files['quantiles']}")
    print(f"[OK] Manifest:              {files['manifest']}")
    print()

    print("Final independent PM_test metrics:")
    print(f"  RMSE                  = {metrics['rmse']:.6f}")
    print(f"  MAE                   = {metrics['mae']:.6f}")
    print(f"  R2                    = {metrics['r2']:.6f}")
    print(f"  mean error            = {metrics['mean_error']:.6f}")
    print(f"  median error          = {metrics['median_error']:.6f}")
    print(f"  NASA score sum        = {metrics['nasa_score_sum']:.6f}")
    print(f"  NASA score mean       = {metrics['nasa_score_mean']:.6f}")
    print(
        f"  overestimate fraction = "
        f"{metrics['overestimate_fraction']:.6f}"
    )
    print(
        f"  underestimate fraction= "
        f"{metrics['underestimate_fraction']:.6f}"
    )
    print(f"  p95 abs error         = {metrics['p95_abs_error']:.6f}")
    print(f"  max abs error         = {metrics['max_abs_error']:.6f}")
    print(f"  p95 late error        = {metrics['p95_late_error']:.6f}")
    print(f"  max late error        = {metrics['max_late_error']:.6f}")
    print()

    worst = evaluation.nlargest(
        10,
        "abs_error",
    )[
        [
            "id",
            "RUL_true",
            "RUL_pred",
            "error",
            "abs_error",
            "nasa_penalty",
        ]
    ]

    print("10 largest absolute errors:")
    with pd.option_context(
        "display.width", 120,
        "display.float_format", "{:.6g}".format,
    ):
        print(worst.to_string(index=False))

    print()
    print(
        "Evaluation complete. Predictions were not recalibrated or modified "
        "after opening PM_truth.csv."
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
