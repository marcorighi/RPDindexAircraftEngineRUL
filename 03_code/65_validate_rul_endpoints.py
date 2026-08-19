#!/usr/bin/env python3
"""
65_validate_rul_endpoints.py

Repeated pseudo-test endpoint validation using ONLY out-of-fold predictions
generated from PM_train.csv by 60_train_rul_model.py.

Scientific purpose
------------------
The official C-MAPSS test problem provides one truncated trajectory per engine
and requires one RUL prediction at the final observed cycle. The row-wise
GroupKFold analysis in 60_train_rul_model.py is useful for model development,
but it evaluates every available cycle of each held-out engine.

This script creates a stricter endpoint-style validation:

    held-out engine trajectory
        -> choose one truncation point
        -> use the OOF prediction at that endpoint only
        -> compare with the true training RUL

For every Monte Carlo repeat, each held-out engine contributes EXACTLY ONE
endpoint. Therefore long engines cannot dominate the endpoint metrics.

No model is refit here. The script reuses the already-generated OOF predictions,
so the validation remains strictly engine-wise and leakage-free.

Default truncation experiment
-----------------------------
For each engine and repeat, choose an integer true RUL uniformly from 20 to 120
cycles, inclusive. The same endpoint draw is used for every feature/model
configuration, enabling paired comparisons.

The default upper limit of 120 was deliberately chosen to be below the shortest
available progressive training trajectory. The script checks this assumption
and aborts if the requested range is not fully available for every engine.

This uniform distribution is a pre-specified stress-test distribution; it is
NOT intended to reproduce PM_truth.csv and PM_truth.csv is never read.

Metrics per 100-engine pseudo-test
----------------------------------
- RMSE
- MAE
- PHM08/NASA asymmetric score, sum and mean
- mean/median signed error
- fraction of RUL overestimates (late/risky predictions)
- 95th percentile absolute error
- maximum absolute error
- 95th percentile positive error (late-prediction error)
- maximum positive error

NASA/PHM08-style score
----------------------
Let:
    d = predicted RUL - true RUL

Then:
    d < 0  : exp(-d / 13) - 1     # early/conservative prediction
    d >= 0 : exp( d / 10) - 1     # late/unsafe prediction

Late predictions are penalized more heavily.

Inputs
------
04_results/02_plot_data/
    60_RUL_groupcv_predictions.csv.gz

Optional provenance input:
04_results/05_tables/
    60_rul_training_manifest.json

Outputs
-------
04_results/02_plot_data/
    65_RUL_endpoint_repeat_metrics.csv.gz

04_results/05_tables/
    21_RUL_endpoint_summary.csv
    22_RUL_endpoint_pairwise.csv
    23_RUL_endpoint_engine_summary.csv
    24_RUL_endpoint_rankings.csv
    25_RUL_endpoint_range_check.csv
    65_endpoint_validation_manifest.json

Important
---------
- PM_test.csv is NOT read.
- PM_truth.csv is NOT read.
- No model is fitted or tuned.
- All configurations are evaluated using exactly the same sampled endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "feature_set",
    "model",
    "fold",
    "id",
    "cycle",
    "RUL_true",
    "RUL_pred",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    p = argparse.ArgumentParser(
        description=(
            "Repeated one-endpoint-per-engine pseudo-test validation "
            "from GroupKFold OOF RUL predictions."
        )
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=project_dir
        / "04_results"
        / "02_plot_data"
        / "60_RUL_groupcv_predictions.csv.gz",
        help="OOF prediction file produced by 60_train_rul_model.py.",
    )
    p.add_argument(
        "--training-manifest",
        type=Path,
        default=project_dir
        / "04_results"
        / "05_tables"
        / "60_rul_training_manifest.json",
        help="Optional provenance manifest from 60_train_rul_model.py.",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=project_dir / "04_results" / "05_tables",
        help="Output directory for endpoint-validation tables.",
    )
    p.add_argument(
        "--plot-data-dir",
        type=Path,
        default=project_dir / "04_results" / "02_plot_data",
        help="Output directory for repeat-level metrics.",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=5000,
        help=(
            "Number of independent 100-engine pseudo-test repeats "
            "(default: 5000)."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260812,
        help="Random seed (default: 20260812).",
    )
    p.add_argument(
        "--min-rul",
        type=int,
        default=20,
        help="Minimum sampled endpoint true RUL, inclusive (default: 20).",
    )
    p.add_argument(
        "--max-rul",
        type=int,
        default=120,
        help="Maximum sampled endpoint true RUL, inclusive (default: 120).",
    )
    p.add_argument(
        "--configs",
        default="all",
        help=(
            "Configurations to evaluate. Default 'all'. "
            "Alternatively provide comma-separated FEATURE_SET:MODEL pairs, "
            "e.g. raw_plus_rpd:ridge,rpd_global:hgb,raw_plus_rpd:hgb"
        ),
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
    error = predicted RUL - true RUL
    """
    error = np.asarray(error, dtype=float)
    out = np.empty_like(error, dtype=float)

    early = error < 0
    out[early] = np.expm1((-error[early]) / 13.0)
    out[~early] = np.expm1(error[~early] / 10.0)
    return out


def parse_config_filter(
    value: str,
    available: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    if value.strip().lower() == "all":
        return available

    requested: list[tuple[str, str]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid configuration '{item}'. "
                "Expected FEATURE_SET:MODEL."
            )
        feature_set, model = item.split(":", 1)
        requested.append((feature_set.strip(), model.strip()))

    missing = [cfg for cfg in requested if cfg not in available]
    if missing:
        raise ValueError(
            f"Requested configurations not found in predictions: {missing}. "
            f"Available: {available}"
        )

    return requested


def validate_predictions(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Prediction file is missing required columns: {missing}"
        )

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Prediction file contains missing required values.")

    key = ["feature_set", "model", "id", "RUL_true"]
    if df[key].duplicated().any():
        duplicated = df.loc[df[key].duplicated(keep=False), key].head(10)
        raise ValueError(
            "Duplicate configuration/engine/RUL rows found. Example:\n"
            f"{duplicated}"
        )

    # Every engine must belong to one and only one validation fold for each
    # configuration.
    fold_counts = (
        df.groupby(["feature_set", "model", "id"])["fold"]
        .nunique()
    )
    if (fold_counts != 1).any():
        bad = fold_counts[fold_counts != 1].head(10)
        raise ValueError(
            "Some engine/configuration appears in more than one OOF fold:\n"
            f"{bad}"
        )


def configuration_label(feature_set: str, model: str) -> str:
    return f"{feature_set} + {model}"


def build_prediction_cubes(
    df: pd.DataFrame,
    configs: list[tuple[str, str]],
    min_rul: int,
    max_rul: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[tuple[str, str], np.ndarray],
    pd.DataFrame,
]:
    """
    Build one prediction matrix per configuration:
        rows    = engines
        columns = integer RUL values min_rul ... max_rul

    Every cell must exist. This guarantees identical endpoint draws for all
    configurations and all engines.
    """
    engines = np.array(sorted(df["id"].unique()), dtype=int)
    rul_values = np.arange(min_rul, max_rul + 1, dtype=int)

    range_rows = []
    cubes: dict[tuple[str, str], np.ndarray] = {}

    # First document available RUL ranges.
    for feature_set, model in configs:
        g = df[
            (df["feature_set"] == feature_set)
            & (df["model"] == model)
        ]

        for engine_id, eg in g.groupby("id", sort=True):
            available = np.sort(eg["RUL_true"].astype(int).unique())
            range_rows.append(
                {
                    "feature_set": feature_set,
                    "model": model,
                    "id": int(engine_id),
                    "available_min_rul": int(available.min()),
                    "available_max_rul": int(available.max()),
                    "available_rul_count": int(len(available)),
                    "requested_min_rul": min_rul,
                    "requested_max_rul": max_rul,
                    "covers_requested_range": bool(
                        available.min() <= min_rul
                        and available.max() >= max_rul
                        and set(rul_values).issubset(set(available))
                    ),
                }
            )

    range_check = pd.DataFrame(range_rows)

    bad = range_check[~range_check["covers_requested_range"]]
    if not bad.empty:
        common_max = int(
            range_check.groupby(["feature_set", "model", "id"])[
                "available_max_rul"
            ]
            .first()
            .min()
        )
        raise ValueError(
            "Requested RUL sampling range is not fully available for every "
            "engine/configuration.\n"
            f"Requested: [{min_rul}, {max_rul}]\n"
            f"Maximum common upper RUL appears to be {common_max}.\n"
            "Choose a smaller --max-rul.\n"
            "First failing rows:\n"
            f"{bad.head(10).to_string(index=False)}"
        )

    # Then create exact engine x RUL prediction matrices.
    for feature_set, model in configs:
        g = df[
            (df["feature_set"] == feature_set)
            & (df["model"] == model)
            & (df["RUL_true"] >= min_rul)
            & (df["RUL_true"] <= max_rul)
        ].copy()

        g["RUL_true"] = g["RUL_true"].astype(int)

        pivot = g.pivot(
            index="id",
            columns="RUL_true",
            values="RUL_pred",
        )
        pivot = pivot.reindex(index=engines, columns=rul_values)

        if pivot.isna().any().any():
            where = np.argwhere(pivot.isna().to_numpy())
            i, j = where[0]
            raise ValueError(
                f"Missing prediction for config {(feature_set, model)}, "
                f"engine={pivot.index[i]}, RUL={pivot.columns[j]}"
            )

        cubes[(feature_set, model)] = pivot.to_numpy(dtype=float)

    return engines, rul_values, cubes, range_check


def generate_endpoint_draws(
    repeats: int,
    engine_count: int,
    min_rul: int,
    max_rul: int,
    seed: int,
) -> np.ndarray:
    """
    Shape = (repeats, engine_count).
    Each row is one pseudo-test dataset; each column is one engine.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(
        low=min_rul,
        high=max_rul + 1,
        size=(repeats, engine_count),
        dtype=np.int16,
    )


def gather_predictions(
    cube: np.ndarray,
    draws: np.ndarray,
    min_rul: int,
) -> np.ndarray:
    """
    cube:  engines x RUL-grid
    draws: repeats x engines

    Return: repeats x engines.
    """
    indices = draws.astype(np.int64) - int(min_rul)
    cube_for_repeats = cube.T  # RUL-grid x engines

    engine_indices = np.arange(cube.shape[0], dtype=np.int64)[None, :]
    return cube_for_repeats[indices, engine_indices]


def repeat_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Arrays are repeats x engines.
    """
    err = y_pred - y_true
    abs_err = np.abs(err)
    late_err = np.maximum(err, 0.0)
    nasa = nasa_penalty(err)

    return {
        "rmse": np.sqrt(np.mean(err ** 2, axis=1)),
        "mae": np.mean(abs_err, axis=1),
        "nasa_score_sum": np.sum(nasa, axis=1),
        "nasa_score_mean": np.mean(nasa, axis=1),
        "mean_error": np.mean(err, axis=1),
        "median_error": np.median(err, axis=1),
        "overestimate_fraction": np.mean(err > 0, axis=1),
        "p95_abs_error": np.quantile(abs_err, 0.95, axis=1),
        "max_abs_error": np.max(abs_err, axis=1),
        "p95_late_error": np.quantile(late_err, 0.95, axis=1),
        "max_late_error": np.max(late_err, axis=1),
    }


def summary_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q975": float(np.quantile(values, 0.975)),
        "q95": float(np.quantile(values, 0.95)),
    }


def create_summary(repeat_df: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "rmse",
        "mae",
        "nasa_score_sum",
        "nasa_score_mean",
        "mean_error",
        "median_error",
        "overestimate_fraction",
        "p95_abs_error",
        "max_abs_error",
        "p95_late_error",
        "max_late_error",
    ]

    rows = []

    for (feature_set, model), g in repeat_df.groupby(
        ["feature_set", "model"],
        sort=False,
    ):
        row = {
            "feature_set": feature_set,
            "model": model,
            "configuration": configuration_label(feature_set, model),
            "repeat_count": int(len(g)),
            "engine_count_per_repeat": int(g["engine_count"].iloc[0]),
            "sampled_min_rul": int(g["sampled_min_rul"].iloc[0]),
            "sampled_max_rul": int(g["sampled_max_rul"].iloc[0]),
        }

        for metric in metric_names:
            s = summary_stats(g[metric].to_numpy(dtype=float))
            for stat_name, value in s.items():
                row[f"{stat_name}_{metric}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def create_pairwise(repeat_df: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "rmse",
        "mae",
        "nasa_score_sum",
        "p95_abs_error",
        "max_abs_error",
        "p95_late_error",
        "max_late_error",
    ]

    configs = (
        repeat_df[["feature_set", "model"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    configs = list(configs)

    indexed = {
        cfg: repeat_df[
            (repeat_df["feature_set"] == cfg[0])
            & (repeat_df["model"] == cfg[1])
        ].sort_values("repeat")
        for cfg in configs
    }

    rows = []

    for cfg_a, cfg_b in itertools.combinations(configs, 2):
        a = indexed[cfg_a]
        b = indexed[cfg_b]

        if not np.array_equal(
            a["repeat"].to_numpy(),
            b["repeat"].to_numpy(),
        ):
            raise RuntimeError("Repeat alignment failure in pairwise analysis.")

        for metric in metric_names:
            va = a[metric].to_numpy(dtype=float)
            vb = b[metric].to_numpy(dtype=float)

            # Lower is better for all pairwise metrics listed here.
            diff_b_minus_a = vb - va

            rows.append(
                {
                    "feature_set_A": cfg_a[0],
                    "model_A": cfg_a[1],
                    "configuration_A": configuration_label(*cfg_a),
                    "feature_set_B": cfg_b[0],
                    "model_B": cfg_b[1],
                    "configuration_B": configuration_label(*cfg_b),
                    "metric": metric,
                    "repeat_count": len(va),
                    "median_A": float(np.median(va)),
                    "median_B": float(np.median(vb)),
                    "median_B_minus_A": float(
                        np.median(diff_b_minus_a)
                    ),
                    "A_win_fraction": float(np.mean(va < vb)),
                    "B_win_fraction": float(np.mean(vb < va)),
                    "tie_fraction": float(np.mean(va == vb)),
                }
            )

    return pd.DataFrame(rows)


def create_engine_summary(
    configs: list[tuple[str, str]],
    engines: np.ndarray,
    draws: np.ndarray,
    prediction_matrices: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    rows = []
    y_true = draws.astype(float)

    for cfg in configs:
        y_pred = prediction_matrices[cfg]
        err = y_pred - y_true
        abs_err = np.abs(err)
        late = np.maximum(err, 0.0)
        nasa = nasa_penalty(err)

        for j, engine_id in enumerate(engines):
            e = err[:, j]
            ae = abs_err[:, j]
            le = late[:, j]
            ns = nasa[:, j]

            rows.append(
                {
                    "feature_set": cfg[0],
                    "model": cfg[1],
                    "configuration": configuration_label(*cfg),
                    "id": int(engine_id),
                    "repeat_count": int(len(e)),
                    "mean_error": float(np.mean(e)),
                    "median_error": float(np.median(e)),
                    "mae": float(np.mean(ae)),
                    "rmse": float(np.sqrt(np.mean(e ** 2))),
                    "overestimate_fraction": float(np.mean(e > 0)),
                    "p95_abs_error": float(np.quantile(ae, 0.95)),
                    "max_abs_error": float(np.max(ae)),
                    "p95_late_error": float(np.quantile(le, 0.95)),
                    "max_late_error": float(np.max(le)),
                    "mean_nasa_penalty": float(np.mean(ns)),
                    "median_nasa_penalty": float(np.median(ns)),
                }
            )

    return pd.DataFrame(rows)


def create_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Two transparent rankings are reported; no single winner is silently chosen.

    Accuracy ranking:
        median RMSE, then median MAE.

    PHM08/safety ranking:
        median NASA score sum, then q95 NASA score sum,
        then median RMSE.
    """
    accuracy = summary.sort_values(
        ["median_rmse", "median_mae"],
        ascending=[True, True],
    ).copy()
    accuracy["accuracy_rank"] = np.arange(1, len(accuracy) + 1)

    phm = summary.sort_values(
        [
            "median_nasa_score_sum",
            "q95_nasa_score_sum",
            "median_rmse",
        ],
        ascending=[True, True, True],
    ).copy()
    phm["phm08_rank"] = np.arange(1, len(phm) + 1)

    cols = ["feature_set", "model", "accuracy_rank"]
    out = summary.merge(
        accuracy[cols],
        on=["feature_set", "model"],
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        phm[["feature_set", "model", "phm08_rank"]],
        on=["feature_set", "model"],
        how="left",
        validate="one_to_one",
    )

    return out.sort_values(
        ["phm08_rank", "accuracy_rank"]
    ).reset_index(drop=True)


def main() -> int:
    args = parse_args()

    predictions_path = args.predictions.resolve()
    manifest_path = args.training_manifest.resolve()
    tables_dir = args.tables_dir.resolve()
    plot_data_dir = args.plot_data_dir.resolve()

    if not predictions_path.exists():
        print(
            f"[ERROR] OOF predictions not found: {predictions_path}",
            file=sys.stderr,
        )
        return 2

    if args.repeats < 1:
        print("[ERROR] --repeats must be >= 1.", file=sys.stderr)
        return 2

    if args.min_rul < 0 or args.max_rul <= args.min_rul:
        print(
            "[ERROR] Require 0 <= --min-rul < --max-rul.",
            file=sys.stderr,
        )
        return 2

    tables_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "repeat_metrics":
            plot_data_dir / "65_RUL_endpoint_repeat_metrics.csv.gz",
        "summary":
            tables_dir / "21_RUL_endpoint_summary.csv",
        "pairwise":
            tables_dir / "22_RUL_endpoint_pairwise.csv",
        "engine_summary":
            tables_dir / "23_RUL_endpoint_engine_summary.csv",
        "rankings":
            tables_dir / "24_RUL_endpoint_rankings.csv",
        "range_check":
            tables_dir / "25_RUL_endpoint_range_check.csv",
        "manifest":
            tables_dir / "65_endpoint_validation_manifest.json",
    }

    print("RPD RUL endpoint-style pseudo-test validation")
    print(f"OOF predictions: {predictions_path}")
    print(f"Repeats:         {args.repeats}")
    print(f"Seed:            {args.seed}")
    print(f"RUL sampling:    uniform integer [{args.min_rul}, {args.max_rul}]")
    print()

    predictions = pd.read_csv(predictions_path)
    validate_predictions(predictions)

    available_configs = sorted(
        predictions[["feature_set", "model"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    try:
        configs = parse_config_filter(
            args.configs,
            available_configs,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    predictions = predictions.merge(
        pd.DataFrame(configs, columns=["feature_set", "model"]),
        on=["feature_set", "model"],
        how="inner",
    )

    print(f"[INFO] configurations ({len(configs)}):")
    for cfg in configs:
        print(f"  - {configuration_label(*cfg)}")

    try:
        engines, rul_values, cubes, range_check = build_prediction_cubes(
            predictions,
            configs,
            args.min_rul,
            args.max_rul,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    range_check.to_csv(files["range_check"], index=False)

    engine_count = len(engines)
    print(
        f"[INFO] engines={engine_count}; common exact RUL grid="
        f"{rul_values[0]}..{rul_values[-1]} ({len(rul_values)} values)"
    )

    draws = generate_endpoint_draws(
        repeats=args.repeats,
        engine_count=engine_count,
        min_rul=args.min_rul,
        max_rul=args.max_rul,
        seed=args.seed,
    )
    y_true = draws.astype(float)

    repeat_rows = []
    prediction_matrices: dict[tuple[str, str], np.ndarray] = {}

    for cfg in configs:
        pred_matrix = gather_predictions(
            cubes[cfg],
            draws,
            args.min_rul,
        )
        prediction_matrices[cfg] = pred_matrix

        m = repeat_metrics(
            y_true,
            pred_matrix,
        )

        config_df = pd.DataFrame(
            {
                "repeat": np.arange(1, args.repeats + 1, dtype=int),
                "feature_set": cfg[0],
                "model": cfg[1],
                "configuration": configuration_label(*cfg),
                "engine_count": engine_count,
                "sampled_min_rul": args.min_rul,
                "sampled_max_rul": args.max_rul,
                **m,
            }
        )
        repeat_rows.append(config_df)

        print(
            f"[INFO] {configuration_label(*cfg)}: "
            f"median RMSE={np.median(m['rmse']):.3f}, "
            f"median MAE={np.median(m['mae']):.3f}, "
            f"median NASA sum={np.median(m['nasa_score_sum']):.3f}"
        )

    repeat_df = pd.concat(
        repeat_rows,
        ignore_index=True,
    )

    summary = create_summary(repeat_df)
    pairwise = create_pairwise(repeat_df)
    engine_summary = create_engine_summary(
        configs,
        engines,
        draws,
        prediction_matrices,
    )
    rankings = create_rankings(summary)

    repeat_df.to_csv(
        files["repeat_metrics"],
        index=False,
        compression="gzip",
    )
    summary.to_csv(files["summary"], index=False)
    pairwise.to_csv(files["pairwise"], index=False)
    engine_summary.to_csv(files["engine_summary"], index=False)
    rankings.to_csv(files["rankings"], index=False)

    provenance = {
        "training_manifest_present": manifest_path.exists(),
        "training_manifest_path": (
            str(manifest_path) if manifest_path.exists() else None
        ),
        "training_manifest_sha256": (
            sha256(manifest_path) if manifest_path.exists() else None
        ),
    }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "oof_predictions_file": str(predictions_path),
        "oof_predictions_sha256": sha256(predictions_path),
        **provenance,
        "engine_count": engine_count,
        "configuration_count": len(configs),
        "configurations": [
            {"feature_set": cfg[0], "model": cfg[1]}
            for cfg in configs
        ],
        "repeats": args.repeats,
        "seed": args.seed,
        "sampling_distribution":
            "independent discrete uniform true RUL per engine and repeat",
        "sampled_min_rul": args.min_rul,
        "sampled_max_rul": args.max_rul,
        "same_endpoint_draws_for_all_configurations": True,
        "one_endpoint_per_engine_per_repeat": True,
        "models_refit": False,
        "model_tuning_performed": False,
        "test_set_used": False,
        "truth_set_used": False,
        "nasa_score_definition":
            "d=predicted-true; d<0: exp(-d/13)-1; "
            "d>=0: exp(d/10)-1",
        "rankings":
            {
                "accuracy":
                    "median RMSE, then median MAE",
                "phm08":
                    "median NASA score sum, then q95 NASA score sum, "
                    "then median RMSE",
            },
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
    for key, path in files.items():
        print(f"[OK] {key}: {path}")

    print()
    print("Endpoint validation rankings:")
    display_cols = [
        "feature_set",
        "model",
        "accuracy_rank",
        "phm08_rank",
        "median_rmse",
        "median_mae",
        "median_nasa_score_sum",
        "q95_nasa_score_sum",
        "median_overestimate_fraction",
        "median_p95_late_error",
        "median_max_late_error",
    ]

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 190,
        "display.float_format", "{:.6g}".format,
    ):
        print(rankings[display_cols].to_string(index=False))

    print()
    print(
        "No final model was selected or modified by this script. "
        "Review the accuracy and PHM08 rankings before freezing the "
        "test-set model."
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
