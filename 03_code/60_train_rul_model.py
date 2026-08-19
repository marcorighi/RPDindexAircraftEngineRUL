#!/usr/bin/env python3
"""
60_train_rul_model.py

Group-wise cross-validation and final RUL model training using progressive
global RPD features derived from PM_train.csv.

Scientific design
-----------------
- Uses PM_train.csv only.
- Does NOT read PM_test.csv or PM_truth.csv.
- Entire engines are kept together by GroupKFold.
- The engine is the independent validation unit.
- Three predefined feature representations are compared under the same folds:
      1) rpd_global
      2) raw_sensor
      3) raw_plus_rpd
  The three operating settings are included as common covariates by default,
  because C-MAPSS sensor values depend on operating condition.
- Two predefined regressors are evaluated without hyperparameter searching:
      Ridge regression
      HistGradientBoostingRegressor
- The best configuration is selected by lowest mean per-engine RMSE from
  out-of-fold predictions, then lowest mean per-engine MAE as tie-breaker.
- After model selection, the selected configuration is fitted once on the
  complete training set and saved for later frozen application to PM_test.csv.

Why group-wise validation?
--------------------------
Rows from the same engine are temporally dependent. Random row-level train/test
splits would leak engine-specific trajectory information. GroupKFold ensures
that every validation engine is completely unseen during fitting.

Feature matrix
--------------
The progressive RPD input is expected in long form:
    id, cycle, RUL, sensor, sensor_value, RPDgep, RPDgea, RPDgepr, ...

It is reshaped to one row per (engine, cycle), with one feature per
sensor/index combination. The first usable observation is cycle 2 because the
progressive RPD trajectory begins with the prefix (x1, x2).

Target
------
True training RUL:
    RUL = final_cycle(engine) - cycle

By default the target is NOT capped. An optional --rul-cap is provided for a
separate, explicitly declared piecewise-RUL experiment, but should not be used
silently.

Metrics
-------
Out-of-fold:
- RMSE
- MAE
- R^2
- NASA/PHM08-style asymmetric score:
      d = prediction - true RUL
      d < 0  : exp(-d / 13) - 1   (early prediction)
      d >= 0 : exp( d / 10) - 1   (late prediction)
  Late predictions are therefore penalized more heavily.
- The same errors are summarized per engine so long trajectories do not
  dominate the principal comparison.

Pseudo-test horizons
--------------------
For each held-out engine, the script also selects the row whose true RUL is
closest to each requested horizon (default: 20, 50, 100, 150 cycles). These
checkpoint metrics approximate evaluation at different truncation depths while
retaining strict engine-level separation.

Expected project layout
-----------------------
02_raw_data/
    PM_train.csv

04_results/01_indices/
    PM_train_RPD_progressive.csv.gz

Outputs
-------
04_results/02_plot_data/
    60_RUL_groupcv_predictions.csv.gz

04_results/05_tables/
    15_RUL_groupcv_fold_metrics.csv
    16_RUL_groupcv_engine_metrics.csv
    17_RUL_groupcv_summary.csv
    18_RUL_model_selection.csv
    19_RUL_horizon_metrics.csv
    20_RUL_feature_sets.csv
    60_rul_training_manifest.json

04_results/07_models/
    RUL_best_model.joblib
    RUL_best_model_metadata.json
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
    import sklearn
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:
    raise SystemExit(
        "This script requires scikit-learn and joblib.\n"
        "Install them in the active Python environment, e.g.:\n"
        "    python -m pip install scikit-learn joblib\n"
        f"Original import error: {exc}"
    )


GLOBAL_INDICES = ["RPDgep", "RPDgea", "RPDgepr"]
SENSORS = [f"s{i}" for i in range(1, 22)]
SETTINGS = ["setting1", "setting2", "setting3"]

DEFAULT_FEATURE_SETS = [
    "rpd_global",
    "raw_sensor",
    "raw_plus_rpd",
]

DEFAULT_MODELS = [
    "ridge",
    "hgb",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    p = argparse.ArgumentParser(
        description=(
            "Train/evaluate RUL regressors using engine-wise GroupKFold "
            "on PM_train-derived global RPD features."
        )
    )
    p.add_argument(
        "--progressive",
        type=Path,
        default=project_dir
        / "04_results"
        / "01_indices"
        / "PM_train_RPD_progressive.csv.gz",
        help="Progressive long-form RPD dataset.",
    )
    p.add_argument(
        "--raw-train",
        type=Path,
        default=project_dir / "02_raw_data" / "PM_train.csv",
        help="Raw PM_train.csv, used for operating settings and checks.",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=project_dir / "04_results" / "05_tables",
        help="Output directory for result tables.",
    )
    p.add_argument(
        "--plot-data-dir",
        type=Path,
        default=project_dir / "04_results" / "02_plot_data",
        help="Output directory for out-of-fold prediction data.",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=project_dir / "04_results" / "07_models",
        help="Output directory for final fitted model.",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of GroupKFold splits (default: 5).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260812,
        help="Random seed used by stochastic regressors (default: 20260812).",
    )
    p.add_argument(
        "--feature-sets",
        default=",".join(DEFAULT_FEATURE_SETS),
        help=(
            "Comma-separated feature sets. Supported: "
            "rpd_global, raw_sensor, raw_plus_rpd "
            "(default: all three)."
        ),
    )
    p.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated models. Supported: ridge,hgb.",
    )
    p.add_argument(
        "--no-settings",
        action="store_true",
        help=(
            "Do not include setting1-setting3 as common operating-condition "
            "covariates. By default they are included in every feature set."
        ),
    )
    p.add_argument(
        "--rul-cap",
        type=float,
        default=None,
        help=(
            "Optional training-target cap. Default: no cap. "
            "Validation metrics are always computed against the uncapped true RUL."
        ),
    )
    p.add_argument(
        "--horizons",
        default="20,50,100,150",
        help="Comma-separated pseudo-test RUL horizons (default: 20,50,100,150).",
    )
    p.add_argument(
        "--prediction-min",
        type=float,
        default=0.0,
        help="Lower bound applied to predictions before scoring (default: 0).",
    )
    p.add_argument(
        "--prediction-max",
        type=float,
        default=None,
        help=(
            "Optional upper bound applied to predictions. "
            "Default: maximum true RUL in PM_train."
        ),
    )
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_sensor_sort_key(name: str) -> int:
    return int(name[1:])


def nasa_score_components(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    PHM08-style asymmetric penalty.

    d = predicted - true
    early (d < 0)  -> exp(-d / 13) - 1
    late  (d >= 0) -> exp( d / 10) - 1
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true

    out = np.empty_like(d, dtype=float)
    early = d < 0

    out[early] = np.expm1((-d[early]) / 13.0)
    out[~early] = np.expm1(d[~early] / 10.0)
    return out


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    nasa = nasa_score_components(y_true, y_pred)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "nasa_score_sum": float(np.sum(nasa)),
        "nasa_score_mean": float(np.mean(nasa)),
        "mean_error": float(np.mean(y_pred - y_true)),
        "median_error": float(np.median(y_pred - y_true)),
    }


def read_and_validate_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["id", "cycle", *SETTINGS, *SENSORS]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing expected columns: {missing}"
        )

    if df[["id", "cycle"]].duplicated().any():
        raise ValueError("Duplicate (id, cycle) rows found in PM_train.csv")

    return df.sort_values(["id", "cycle"]).reset_index(drop=True)


def read_and_validate_progressive(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    expected = [
        "id",
        "cycle",
        "RUL",
        "sensor",
        "sensor_value",
        *GLOBAL_INDICES,
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing expected columns: {missing}"
        )

    if df[["id", "cycle", "sensor"]].duplicated().any():
        raise ValueError(
            "Duplicate (id, cycle, sensor) rows found in progressive dataset"
        )

    return df.sort_values(["id", "cycle", "sensor"]).reset_index(drop=True)


def build_feature_matrix(
    progressive: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Return one row per (id, cycle) and a dictionary defining the available
    feature families.
    """
    # Meta target table.
    meta = (
        progressive[["id", "cycle", "RUL"]]
        .drop_duplicates()
        .sort_values(["id", "cycle"])
        .reset_index(drop=True)
    )

    # Raw current sensor values from the progressive file.
    raw_wide = progressive.pivot(
        index=["id", "cycle"],
        columns="sensor",
        values="sensor_value",
    )
    raw_wide = raw_wide.reindex(
        columns=sorted(raw_wide.columns, key=numeric_sensor_sort_key)
    )
    raw_wide.columns = [f"raw_{c}" for c in raw_wide.columns]
    raw_wide = raw_wide.reset_index()

    # Global RPD features.
    rpd_tables = []
    for index_name in GLOBAL_INDICES:
        w = progressive.pivot(
            index=["id", "cycle"],
            columns="sensor",
            values=index_name,
        )
        w = w.reindex(
            columns=sorted(w.columns, key=numeric_sensor_sort_key)
        )
        w.columns = [f"{index_name}_{c}" for c in w.columns]
        rpd_tables.append(w)

    rpd_wide = pd.concat(rpd_tables, axis=1).reset_index()

    # Operating settings from the original training table.
    settings = raw[["id", "cycle", *SETTINGS]].copy()

    feature_df = (
        meta.merge(raw_wide, on=["id", "cycle"], how="inner", validate="one_to_one")
        .merge(rpd_wide, on=["id", "cycle"], how="inner", validate="one_to_one")
        .merge(settings, on=["id", "cycle"], how="left", validate="one_to_one")
        .sort_values(["id", "cycle"])
        .reset_index(drop=True)
    )

    if feature_df[SETTINGS].isna().any().any():
        raise ValueError(
            "Operating settings could not be matched for all progressive rows."
        )

    raw_cols = [f"raw_{s}" for s in SENSORS]
    rpd_cols = [
        f"{idx}_{sensor}"
        for idx in GLOBAL_INDICES
        for sensor in SENSORS
    ]

    feature_groups = {
        "raw": raw_cols,
        "rpd": rpd_cols,
        "settings": SETTINGS.copy(),
    }

    # Hard consistency checks.
    expected_rows = len(meta)
    if len(feature_df) != expected_rows:
        raise ValueError(
            f"Feature reshape changed row count: {expected_rows} -> "
            f"{len(feature_df)}"
        )

    feature_cols = raw_cols + rpd_cols + SETTINGS
    non_finite = ~np.isfinite(
        feature_df[feature_cols].to_numpy(dtype=float)
    )
    if np.any(non_finite):
        count = int(np.sum(non_finite))
        raise ValueError(
            f"Feature matrix contains {count} non-finite values."
        )

    return feature_df, feature_groups


def feature_names_for_set(
    feature_set: str,
    groups: dict[str, list[str]],
    include_settings: bool,
) -> list[str]:
    if feature_set == "rpd_global":
        cols = groups["rpd"].copy()
    elif feature_set == "raw_sensor":
        cols = groups["raw"].copy()
    elif feature_set == "raw_plus_rpd":
        cols = groups["raw"] + groups["rpd"]
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    if include_settings:
        cols += groups["settings"]

    return cols


def make_model(name: str, seed: int):
    if name == "ridge":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    if name == "hgb":
        return Pipeline(
            steps=[
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=seed,
                    ),
                )
            ]
        )

    raise ValueError(f"Unsupported model: {name}")


def fit_model(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
):
    """
    Both predefined estimators accept sample_weight at their final estimator.
    Passing it through the Pipeline avoids weighting rows from long engines
    more heavily in the regression loss.
    """
    estimator.fit(
        X,
        y,
        model__sample_weight=sample_weight,
    )
    return estimator


def equal_engine_sample_weights(groups: np.ndarray) -> np.ndarray:
    """
    Each engine contributes total weight 1, then weights are normalized to
    mean 1 for numerical convenience.
    """
    group_series = pd.Series(groups)
    counts = group_series.value_counts()
    w = group_series.map(lambda g: 1.0 / counts[g]).to_numpy(dtype=float)
    w *= len(w) / np.sum(w)
    return w


def per_engine_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouping = [
        "feature_set",
        "model",
        "fold",
        "id",
    ]

    for keys, g in predictions.groupby(grouping, sort=False):
        feature_set, model_name, fold, engine_id = keys
        m = regression_metrics(
            g["RUL_true"].to_numpy(dtype=float),
            g["RUL_pred"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "fold": int(fold),
                "id": int(engine_id),
                "n_points": int(len(g)),
                **m,
            }
        )

    return pd.DataFrame(rows)


def horizon_metrics(
    predictions: pd.DataFrame,
    horizons: list[float],
) -> pd.DataFrame:
    """
    For every held-out engine and requested true-RUL horizon, retain one row:
    the available prediction whose true RUL is closest to the horizon.

    Then aggregate across engines for each configuration/horizon.
    """
    rows = []

    for (feature_set, model_name), config in predictions.groupby(
        ["feature_set", "model"],
        sort=False,
    ):
        for horizon in horizons:
            selected_rows = []

            for engine_id, g in config.groupby("id", sort=False):
                g = g.copy()

                # Require the engine trajectory to span the requested horizon.
                min_rul = float(g["RUL_true"].min())
                max_rul = float(g["RUL_true"].max())
                if horizon < min_rul or horizon > max_rul:
                    continue

                idx = (g["RUL_true"] - horizon).abs().idxmin()
                selected_rows.append(g.loc[idx])

            if not selected_rows:
                continue

            selected = pd.DataFrame(selected_rows)

            m = regression_metrics(
                selected["RUL_true"].to_numpy(dtype=float),
                selected["RUL_pred"].to_numpy(dtype=float),
            )

            rows.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "target_horizon": float(horizon),
                    "engine_count": int(selected["id"].nunique()),
                    **m,
                }
            )

    return pd.DataFrame(rows)


def aggregate_summary(
    predictions: pd.DataFrame,
    engine_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (feature_set, model_name), g in predictions.groupby(
        ["feature_set", "model"],
        sort=False,
    ):
        overall = regression_metrics(
            g["RUL_true"].to_numpy(dtype=float),
            g["RUL_pred"].to_numpy(dtype=float),
        )

        eg = engine_metrics[
            (engine_metrics["feature_set"] == feature_set)
            & (engine_metrics["model"] == model_name)
        ]

        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "engine_count": int(g["id"].nunique()),
                "sample_count": int(len(g)),
                "oof_rmse_all_rows": overall["rmse"],
                "oof_mae_all_rows": overall["mae"],
                "oof_r2_all_rows": overall["r2"],
                "oof_nasa_score_mean_all_rows": overall["nasa_score_mean"],
                "oof_mean_error_all_rows": overall["mean_error"],
                "mean_engine_rmse": float(eg["rmse"].mean()),
                "median_engine_rmse": float(eg["rmse"].median()),
                "mean_engine_mae": float(eg["mae"].mean()),
                "median_engine_mae": float(eg["mae"].median()),
                "mean_engine_r2": float(eg["r2"].mean()),
                "median_engine_r2": float(eg["r2"].median()),
                "mean_engine_nasa_score_mean": float(
                    eg["nasa_score_mean"].mean()
                ),
                "median_engine_nasa_score_mean": float(
                    eg["nasa_score_mean"].median()
                ),
                "mean_engine_bias": float(eg["mean_error"].mean()),
                "median_engine_bias": float(eg["mean_error"].median()),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["mean_engine_rmse", "mean_engine_mae"],
        ascending=[True, True],
    ).reset_index(drop=True)


def main() -> int:
    args = parse_args()

    progressive_path = args.progressive.resolve()
    raw_path = args.raw_train.resolve()
    tables_dir = args.tables_dir.resolve()
    plot_data_dir = args.plot_data_dir.resolve()
    models_dir = args.models_dir.resolve()

    for path in [progressive_path, raw_path]:
        if not path.exists():
            print(f"[ERROR] Missing input: {path}", file=sys.stderr)
            return 2

    if args.folds < 2:
        print("[ERROR] --folds must be >= 2.", file=sys.stderr)
        return 2

    if args.rul_cap is not None and args.rul_cap <= 0:
        print("[ERROR] --rul-cap must be positive.", file=sys.stderr)
        return 2

    feature_sets = [
        x.strip()
        for x in args.feature_sets.split(",")
        if x.strip()
    ]
    supported_feature_sets = set(DEFAULT_FEATURE_SETS)
    unknown_feature_sets = sorted(
        set(feature_sets).difference(supported_feature_sets)
    )
    if unknown_feature_sets:
        print(
            f"[ERROR] Unsupported feature sets: {unknown_feature_sets}",
            file=sys.stderr,
        )
        return 2

    models = [
        x.strip()
        for x in args.models.split(",")
        if x.strip()
    ]
    supported_models = set(DEFAULT_MODELS)
    unknown_models = sorted(set(models).difference(supported_models))
    if unknown_models:
        print(
            f"[ERROR] Unsupported models: {unknown_models}",
            file=sys.stderr,
        )
        return 2

    try:
        horizons = [
            float(x.strip())
            for x in args.horizons.split(",")
            if x.strip()
        ]
    except ValueError:
        print("[ERROR] Invalid --horizons list.", file=sys.stderr)
        return 2

    tables_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "predictions":
            plot_data_dir / "60_RUL_groupcv_predictions.csv.gz",
        "fold_metrics":
            tables_dir / "15_RUL_groupcv_fold_metrics.csv",
        "engine_metrics":
            tables_dir / "16_RUL_groupcv_engine_metrics.csv",
        "summary":
            tables_dir / "17_RUL_groupcv_summary.csv",
        "selection":
            tables_dir / "18_RUL_model_selection.csv",
        "horizon_metrics":
            tables_dir / "19_RUL_horizon_metrics.csv",
        "feature_sets":
            tables_dir / "20_RUL_feature_sets.csv",
        "manifest":
            tables_dir / "60_rul_training_manifest.json",
        "best_model":
            models_dir / "RUL_best_model.joblib",
        "best_model_metadata":
            models_dir / "RUL_best_model_metadata.json",
    }

    print("RPD RUL model training / engine-wise cross-validation")
    print(f"Progressive: {progressive_path}")
    print(f"Raw train:   {raw_path}")
    print(f"Folds:       {args.folds}")
    print(f"Feature sets:{feature_sets}")
    print(f"Models:      {models}")
    print(f"Settings:    {'excluded' if args.no_settings else 'included'}")
    print(f"RUL cap:     {args.rul_cap}")
    print()

    raw = read_and_validate_raw(raw_path)
    progressive = read_and_validate_progressive(progressive_path)

    feature_df, feature_groups = build_feature_matrix(
        progressive,
        raw,
    )

    engine_count = int(feature_df["id"].nunique())
    if args.folds > engine_count:
        print(
            f"[ERROR] --folds={args.folds} exceeds engine count={engine_count}",
            file=sys.stderr,
        )
        return 2

    include_settings = not args.no_settings

    feature_map = {
        feature_set: feature_names_for_set(
            feature_set,
            feature_groups,
            include_settings,
        )
        for feature_set in feature_sets
    }

    # Feature-set metadata.
    feature_rows = []
    for feature_set, cols in feature_map.items():
        feature_rows.append(
            {
                "feature_set": feature_set,
                "feature_count": len(cols),
                "settings_included": include_settings,
                "raw_feature_count": sum(c.startswith("raw_") for c in cols),
                "rpd_feature_count": sum(
                    c.startswith(tuple(GLOBAL_INDICES)) for c in cols
                ),
                "setting_feature_count": sum(c in SETTINGS for c in cols),
                "features": "|".join(cols),
            }
        )
    feature_set_table = pd.DataFrame(feature_rows)
    feature_set_table.to_csv(files["feature_sets"], index=False)

    y_true_all = feature_df["RUL"].to_numpy(dtype=float)
    groups_all = feature_df["id"].to_numpy(dtype=int)

    if args.rul_cap is None:
        y_model_all = y_true_all.copy()
    else:
        y_model_all = np.minimum(y_true_all, args.rul_cap)

    pred_max = (
        float(np.max(y_true_all))
        if args.prediction_max is None
        else float(args.prediction_max)
    )
    pred_min = float(args.prediction_min)

    if pred_max <= pred_min:
        print(
            "[ERROR] prediction maximum must exceed prediction minimum.",
            file=sys.stderr,
        )
        return 2

    splitter = GroupKFold(n_splits=args.folds)
    # Create folds once and reuse exactly the same engine splits for all
    # feature/model configurations.
    dummy_X = np.zeros((len(feature_df), 1), dtype=float)
    folds = list(
        splitter.split(
            dummy_X,
            y_true_all,
            groups=groups_all,
        )
    )

    # Fold assignment table for reproducibility.
    fold_for_row = np.full(len(feature_df), -1, dtype=int)
    for fold_no, (_, valid_idx) in enumerate(folds, start=1):
        fold_for_row[valid_idx] = fold_no

    if np.any(fold_for_row < 0):
        raise RuntimeError("Some rows were not assigned to a validation fold.")

    predictions_all = []
    fold_metric_rows = []

    total_configs = len(feature_sets) * len(models)
    config_no = 0

    for feature_set in feature_sets:
        cols = feature_map[feature_set]
        X_all = feature_df[cols]

        for model_name in models:
            config_no += 1
            print(
                f"[INFO] configuration {config_no}/{total_configs}: "
                f"{feature_set} + {model_name} "
                f"({len(cols)} features)"
            )

            for fold_no, (train_idx, valid_idx) in enumerate(folds, start=1):
                train_engines = np.unique(groups_all[train_idx])
                valid_engines = np.unique(groups_all[valid_idx])

                overlap = np.intersect1d(train_engines, valid_engines)
                if len(overlap):
                    raise RuntimeError(
                        f"Engine leakage in fold {fold_no}: {overlap.tolist()}"
                    )

                estimator = make_model(model_name, args.seed)

                weights = equal_engine_sample_weights(
                    groups_all[train_idx]
                )

                fit_model(
                    estimator,
                    X_all.iloc[train_idx],
                    y_model_all[train_idx],
                    weights,
                )

                pred = estimator.predict(X_all.iloc[valid_idx])
                pred = np.clip(pred, pred_min, pred_max)

                y_valid_true = y_true_all[valid_idx]

                m = regression_metrics(
                    y_valid_true,
                    pred,
                )

                fold_metric_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold_no,
                        "train_engine_count": int(len(train_engines)),
                        "validation_engine_count": int(len(valid_engines)),
                        "train_row_count": int(len(train_idx)),
                        "validation_row_count": int(len(valid_idx)),
                        **m,
                    }
                )

                fold_pred = feature_df.loc[
                    valid_idx,
                    ["id", "cycle", "RUL"],
                ].copy()
                fold_pred = fold_pred.rename(
                    columns={"RUL": "RUL_true"}
                )
                fold_pred["RUL_pred"] = pred
                fold_pred["error"] = pred - y_valid_true
                fold_pred["abs_error"] = np.abs(
                    pred - y_valid_true
                )
                fold_pred["nasa_penalty"] = nasa_score_components(
                    y_valid_true,
                    pred,
                )
                fold_pred["fold"] = fold_no
                fold_pred["feature_set"] = feature_set
                fold_pred["model"] = model_name

                predictions_all.append(fold_pred)

                print(
                    f"  fold {fold_no}/{args.folds}: "
                    f"RMSE={m['rmse']:.3f}, "
                    f"MAE={m['mae']:.3f}, "
                    f"R2={m['r2']:.3f}"
                )

    predictions = pd.concat(
        predictions_all,
        ignore_index=True,
    )
    fold_metrics = pd.DataFrame(fold_metric_rows)

    # Keep a stable column order.
    predictions = predictions[
        [
            "feature_set",
            "model",
            "fold",
            "id",
            "cycle",
            "RUL_true",
            "RUL_pred",
            "error",
            "abs_error",
            "nasa_penalty",
        ]
    ].sort_values(
        ["feature_set", "model", "id", "cycle"]
    ).reset_index(drop=True)

    engine_metrics = per_engine_metrics(predictions)
    summary = aggregate_summary(
        predictions,
        engine_metrics,
    )
    horizons_table = horizon_metrics(
        predictions,
        horizons,
    )

    # Selection is entirely based on OOF training-engine performance.
    selection = summary.copy()
    selection.insert(
        0,
        "selection_rank",
        np.arange(1, len(selection) + 1),
    )
    selection["selected"] = False
    selection.loc[0, "selected"] = True
    selection["selection_criterion"] = (
        "lowest mean_engine_rmse; mean_engine_mae tie-breaker"
    )

    best_feature_set = str(selection.loc[0, "feature_set"])
    best_model_name = str(selection.loc[0, "model"])
    best_features = feature_map[best_feature_set]

    # Fit the frozen final model once on all PM_train-derived rows.
    final_model = make_model(
        best_model_name,
        args.seed,
    )
    final_weights = equal_engine_sample_weights(groups_all)
    fit_model(
        final_model,
        feature_df[best_features],
        y_model_all,
        final_weights,
    )

    bundle = {
        "estimator": final_model,
        "feature_set": best_feature_set,
        "model_name": best_model_name,
        "feature_names": best_features,
        "global_indices": GLOBAL_INDICES,
        "sensors": SENSORS,
        "settings": SETTINGS if include_settings else [],
        "rul_cap": args.rul_cap,
        "prediction_min": pred_min,
        "prediction_max": pred_max,
        "training_engine_count": engine_count,
        "training_row_count": int(len(feature_df)),
        "seed": args.seed,
    }
    joblib.dump(bundle, files["best_model"])

    model_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_feature_set": best_feature_set,
        "selected_model": best_model_name,
        "feature_count": len(best_features),
        "feature_names": best_features,
        "selection_criterion":
            "lowest mean_engine_rmse; mean_engine_mae tie-breaker",
        "selected_oof_metrics": {
            k: (
                float(selection.loc[0, k])
                if isinstance(selection.loc[0, k], (float, np.floating))
                else int(selection.loc[0, k])
                if isinstance(selection.loc[0, k], (int, np.integer))
                else selection.loc[0, k]
            )
            for k in selection.columns
            if k not in {"selected", "selection_criterion"}
        },
        "rul_cap": args.rul_cap,
        "prediction_min": pred_min,
        "prediction_max": pred_max,
        "test_set_used": False,
        "truth_set_used": False,
    }
    files["best_model_metadata"].write_text(
        json.dumps(model_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    predictions.to_csv(
        files["predictions"],
        index=False,
        compression="gzip",
    )
    fold_metrics.to_csv(files["fold_metrics"], index=False)
    engine_metrics.to_csv(files["engine_metrics"], index=False)
    summary.to_csv(files["summary"], index=False)
    selection.to_csv(files["selection"], index=False)
    horizons_table.to_csv(files["horizon_metrics"], index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "progressive_file": str(progressive_path),
        "progressive_sha256": sha256(progressive_path),
        "raw_train_file": str(raw_path),
        "raw_train_sha256": sha256(raw_path),
        "engine_count": engine_count,
        "row_count": int(len(feature_df)),
        "folds": args.folds,
        "grouping_unit": "engine id",
        "feature_sets": feature_sets,
        "models": models,
        "operating_settings_included": include_settings,
        "rul_cap": args.rul_cap,
        "prediction_clip": [pred_min, pred_max],
        "horizons": horizons,
        "global_indices": GLOBAL_INDICES,
        "software_doi": "10.5281/zenodo.21892421",
        "technical_report_doi": "10.5281/zenodo.21898684",
        "sample_weighting":
            "Each training engine has equal total regression weight.",
        "model_selection":
            "Lowest mean per-engine OOF RMSE; mean per-engine OOF MAE "
            "used as tie-breaker.",
        "nasa_score_definition":
            "d=prediction-true; d<0: exp(-d/13)-1; "
            "d>=0: exp(d/10)-1",
        "test_set_used": False,
        "truth_set_used": False,
        "selected_feature_set": best_feature_set,
        "selected_model": best_model_name,
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
    print("[OK] Cross-validation complete.")
    print(f"[OK] OOF predictions: {files['predictions']}")
    print(f"[OK] Summary: {files['summary']}")
    print(f"[OK] Selection: {files['selection']}")
    print(f"[OK] Horizon metrics: {files['horizon_metrics']}")
    print(f"[OK] Frozen model: {files['best_model']}")
    print(f"[OK] Manifest: {files['manifest']}")
    print()

    print("Model selection summary:")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 180,
        "display.float_format", "{:.6g}".format,
    ):
        print(
            selection[
                [
                    "selection_rank",
                    "feature_set",
                    "model",
                    "mean_engine_rmse",
                    "mean_engine_mae",
                    "mean_engine_r2",
                    "mean_engine_nasa_score_mean",
                    "selected",
                ]
            ].to_string(index=False)
        )

    print()
    print(
        f"Selected: {best_feature_set} + {best_model_name} "
        f"({len(best_features)} features)"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
