#!/usr/bin/env python3
"""
50_analyze_train.py

Statistical analysis of progressive RPD trajectories computed from PM_train.csv.

Expected input
--------------
04_results/01_indices/PM_train_RPD_progressive.csv.gz

Required columns
----------------
id, cycle, RUL, sensor, sensor_value,
RPDlep, RPDlea, RPDlepr, RPDgep, RPDgea, RPDgepr

Outputs
-------
04_results/05_tables/
    01_RPD_metrics_by_engine_sensor.csv
    02_RPD_summary_by_index_sensor.csv
    03_RPD_summary_by_index.csv
    04_RPD_local_vs_global.csv
    05_RPD_paired_local_global.csv
    06_RPD_sensor_ranking.csv
    07_sensor_characteristics.csv
    50_analysis_manifest.json

Purpose
-------
This script is intentionally descriptive/diagnostic. It does not use PM_test.csv
or PM_truth.csv and therefore does not leak test-set ground truth into method
development.

Metrics
-------
For each (engine, sensor, RPD index) trajectory:

- Pearson correlation with RUL and cycle.
- Spearman rank correlation with RUL and cycle.
- Monotonicity:
      |N_inc - N_dec| / (N - 1)
  with ties contributing to the denominator. Range [0, 1].
- Signed monotonicity:
      (N_inc - N_dec) / (N - 1)
  Range [-1, 1].
- Tie fraction.
- Linear slope versus normalized life fraction in [0, 1].
- Robust early/late medians (first/last 20% of the trajectory).
- Late-minus-early change and absolute change.
- Endpoint value.

The principal comparison between local and global RPD families is based on
paired engine/sensor observations, restricted to sensor trajectories with
non-zero raw sensor variability.

No model fitting or RUL prediction is performed here.
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
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


INDEX_NAMES = [
    "RPDlep",
    "RPDlea",
    "RPDlepr",
    "RPDgep",
    "RPDgea",
    "RPDgepr",
]

LOCAL_INDICES = ["RPDlep", "RPDlea", "RPDlepr"]
GLOBAL_INDICES = ["RPDgep", "RPDgea", "RPDgepr"]

PAIRS = [
    ("RPDlep", "RPDgep", "p"),
    ("RPDlea", "RPDgea", "a"),
    ("RPDlepr", "RPDgepr", "pr"),
]

REQUIRED_COLUMNS = [
    "id",
    "cycle",
    "RUL",
    "sensor",
    "sensor_value",
    *INDEX_NAMES,
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Analyze progressive RPD trajectories from PM_train.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir
        / "04_results"
        / "01_indices"
        / "PM_train_RPD_progressive.csv.gz",
        help="Progressive RPD dataset (default: project 04_results/01_indices).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "04_results" / "05_tables",
        help="Directory for statistical result tables.",
    )
    parser.add_argument(
        "--early-late-fraction",
        type=float,
        default=0.20,
        help="Fraction of each trajectory used for early/late robust medians "
             "(default: 0.20).",
    )
    parser.add_argument(
        "--sensor-std-eps",
        type=float,
        default=1e-12,
        help="Raw-sensor standard-deviation threshold below which an "
             "engine/sensor trajectory is treated as constant.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan

    # Average ranks for ties, equivalent to the usual Spearman definition.
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return safe_corr(xr, yr)


def trajectory_metrics(
    y: np.ndarray,
    cycle: np.ndarray,
    rul: np.ndarray,
    early_late_fraction: float,
) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(cycle) & np.isfinite(rul)
    y = y[mask]
    cycle = cycle[mask]
    rul = rul[mask]

    n = len(y)
    if n == 0:
        return {
            "n_points": 0,
            "pearson_rul": np.nan,
            "abs_pearson_rul": np.nan,
            "spearman_rul": np.nan,
            "abs_spearman_rul": np.nan,
            "pearson_cycle": np.nan,
            "abs_pearson_cycle": np.nan,
            "spearman_cycle": np.nan,
            "abs_spearman_cycle": np.nan,
            "monotonicity_signed": np.nan,
            "monotonicity": np.nan,
            "tie_fraction": np.nan,
            "slope_life_fraction": np.nan,
            "early_median": np.nan,
            "late_median": np.nan,
            "delta_late_early": np.nan,
            "abs_delta_late_early": np.nan,
            "endpoint": np.nan,
            "rpd_min": np.nan,
            "rpd_max": np.nan,
            "rpd_range": np.nan,
        }

    pearson_rul = safe_corr(y, rul)
    spearman_rul = spearman_corr(y, rul)
    pearson_cycle = safe_corr(y, cycle)
    spearman_cycle = spearman_corr(y, cycle)

    if n >= 2:
        dy = np.diff(y)
        inc = int(np.sum(dy > 0))
        dec = int(np.sum(dy < 0))
        ties = int(np.sum(dy == 0))
        den = n - 1

        monotonicity_signed = (inc - dec) / den
        monotonicity = abs(monotonicity_signed)
        tie_fraction = ties / den

        life_fraction = np.linspace(0.0, 1.0, n)
        if np.ptp(y) == 0:
            slope = 0.0
        else:
            slope = float(np.polyfit(life_fraction, y, 1)[0])
    else:
        monotonicity_signed = np.nan
        monotonicity = np.nan
        tie_fraction = np.nan
        slope = np.nan

    k = max(1, int(math.ceil(n * early_late_fraction)))
    early_median = float(np.median(y[:k]))
    late_median = float(np.median(y[-k:]))
    delta_late_early = late_median - early_median

    endpoint = float(y[-1])
    rpd_min = float(np.min(y))
    rpd_max = float(np.max(y))

    return {
        "n_points": n,
        "pearson_rul": pearson_rul,
        "abs_pearson_rul": abs(pearson_rul) if np.isfinite(pearson_rul) else np.nan,
        "spearman_rul": spearman_rul,
        "abs_spearman_rul": abs(spearman_rul) if np.isfinite(spearman_rul) else np.nan,
        "pearson_cycle": pearson_cycle,
        "abs_pearson_cycle": abs(pearson_cycle) if np.isfinite(pearson_cycle) else np.nan,
        "spearman_cycle": spearman_cycle,
        "abs_spearman_cycle": abs(spearman_cycle) if np.isfinite(spearman_cycle) else np.nan,
        "monotonicity_signed": monotonicity_signed,
        "monotonicity": monotonicity,
        "tie_fraction": tie_fraction,
        "slope_life_fraction": slope,
        "early_median": early_median,
        "late_median": late_median,
        "delta_late_early": delta_late_early,
        "abs_delta_late_early": abs(delta_late_early),
        "endpoint": endpoint,
        "rpd_min": rpd_min,
        "rpd_max": rpd_max,
        "rpd_range": rpd_max - rpd_min,
    }


def q25(s: pd.Series) -> float:
    return float(s.quantile(0.25))


def q75(s: pd.Series) -> float:
    return float(s.quantile(0.75))


def sign_consistency(s: pd.Series) -> float:
    """
    Fraction of finite, non-zero values having the majority sign.
    1.0 means all engines have the same sign; 0.5 is evenly split.
    """
    a = s.to_numpy(dtype=float)
    a = a[np.isfinite(a) & (a != 0)]
    if len(a) == 0:
        return np.nan

    p = np.sum(a > 0)
    n = np.sum(a < 0)
    return float(max(p, n) / (p + n))


def summarize_index_sensor(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    subset = metrics[metrics["informative_sensor_trajectory"]].copy()

    for (index_name, sensor), g in subset.groupby(
        ["index", "sensor"], sort=True
    ):
        rows.append(
            {
                "index": index_name,
                "family": g["family"].iloc[0],
                "sensor": sensor,
                "engine_count": g["id"].nunique(),
                "median_abs_spearman_rul": g["abs_spearman_rul"].median(),
                "q25_abs_spearman_rul": q25(g["abs_spearman_rul"]),
                "q75_abs_spearman_rul": q75(g["abs_spearman_rul"]),
                "median_abs_pearson_rul": g["abs_pearson_rul"].median(),
                "median_monotonicity": g["monotonicity"].median(),
                "q25_monotonicity": q25(g["monotonicity"]),
                "q75_monotonicity": q75(g["monotonicity"]),
                "median_tie_fraction": g["tie_fraction"].median(),
                "median_abs_delta_late_early": g["abs_delta_late_early"].median(),
                "median_rpd_range": g["rpd_range"].median(),
                "spearman_sign_consistency": sign_consistency(g["spearman_rul"]),
                "slope_sign_consistency": sign_consistency(g["slope_life_fraction"]),
                "median_endpoint": g["endpoint"].median(),
                "q25_endpoint": q25(g["endpoint"]),
                "q75_endpoint": q75(g["endpoint"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_index(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    subset = metrics[metrics["informative_sensor_trajectory"]].copy()

    for index_name, g in subset.groupby("index", sort=False):
        rows.append(
            {
                "index": index_name,
                "family": g["family"].iloc[0],
                "engine_sensor_count": len(g),
                "engine_count": g["id"].nunique(),
                "sensor_count": g["sensor"].nunique(),
                "median_abs_spearman_rul": g["abs_spearman_rul"].median(),
                "q25_abs_spearman_rul": q25(g["abs_spearman_rul"]),
                "q75_abs_spearman_rul": q75(g["abs_spearman_rul"]),
                "median_abs_pearson_rul": g["abs_pearson_rul"].median(),
                "median_monotonicity": g["monotonicity"].median(),
                "q25_monotonicity": q25(g["monotonicity"]),
                "q75_monotonicity": q75(g["monotonicity"]),
                "median_tie_fraction": g["tie_fraction"].median(),
                "median_abs_delta_late_early": g["abs_delta_late_early"].median(),
                "median_rpd_range": g["rpd_range"].median(),
                "spearman_sign_consistency": sign_consistency(g["spearman_rul"]),
                "slope_sign_consistency": sign_consistency(g["slope_life_fraction"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_family(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = metrics[metrics["informative_sensor_trajectory"]].copy()

    for family, g in subset.groupby("family", sort=False):
        rows.append(
            {
                "family": family,
                "index_count": g["index"].nunique(),
                "engine_sensor_index_count": len(g),
                "engine_count": g["id"].nunique(),
                "sensor_count": g["sensor"].nunique(),
                "median_abs_spearman_rul": g["abs_spearman_rul"].median(),
                "q25_abs_spearman_rul": q25(g["abs_spearman_rul"]),
                "q75_abs_spearman_rul": q75(g["abs_spearman_rul"]),
                "median_abs_pearson_rul": g["abs_pearson_rul"].median(),
                "median_monotonicity": g["monotonicity"].median(),
                "q25_monotonicity": q25(g["monotonicity"]),
                "q75_monotonicity": q75(g["monotonicity"]),
                "median_tie_fraction": g["tie_fraction"].median(),
                "median_abs_delta_late_early": g["abs_delta_late_early"].median(),
                "median_rpd_range": g["rpd_range"].median(),
            }
        )

    return pd.DataFrame(rows)


def paired_local_global(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Pair each local/global member using the same engine and sensor.
    Produces effect summaries and, if SciPy is available, two-sided paired
    Wilcoxon signed-rank p-values.
    """
    rows = []

    base = metrics[metrics["informative_sensor_trajectory"]].copy()

    for local_name, global_name, variant in PAIRS:
        local = base[base["index"] == local_name].set_index(["id", "sensor"])
        glob = base[base["index"] == global_name].set_index(["id", "sensor"])

        common = local.index.intersection(glob.index)
        local = local.loc[common]
        glob = glob.loc[common]

        row = {
            "variant": variant,
            "local_index": local_name,
            "global_index": global_name,
            "paired_engine_sensor_count": len(common),
        }

        for metric in [
            "abs_spearman_rul",
            "abs_pearson_rul",
            "monotonicity",
            "abs_delta_late_early",
            "rpd_range",
        ]:
            a = local[metric].to_numpy(dtype=float)
            b = glob[metric].to_numpy(dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            a = a[mask]
            b = b[mask]

            diff = b - a

            row[f"local_median_{metric}"] = (
                float(np.median(a)) if len(a) else np.nan
            )
            row[f"global_median_{metric}"] = (
                float(np.median(b)) if len(b) else np.nan
            )
            row[f"median_global_minus_local_{metric}"] = (
                float(np.median(diff)) if len(diff) else np.nan
            )
            row[f"global_win_fraction_{metric}"] = (
                float(np.mean(diff > 0)) if len(diff) else np.nan
            )
            row[f"tie_fraction_{metric}"] = (
                float(np.mean(diff == 0)) if len(diff) else np.nan
            )

            p_value = np.nan
            if wilcoxon is not None and len(diff) >= 2 and np.any(diff != 0):
                try:
                    p_value = float(
                        wilcoxon(
                            b,
                            a,
                            alternative="two-sided",
                            zero_method="wilcox",
                        ).pvalue
                    )
                except Exception:
                    p_value = np.nan

            row[f"wilcoxon_p_{metric}"] = p_value

        rows.append(row)

    return pd.DataFrame(rows)


def sensor_characteristics(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    rows = []

    for sensor, g in df.groupby("sensor", sort=True):
        raw = g["sensor_value"].to_numpy(dtype=float)
        raw = raw[np.isfinite(raw)]

        if len(raw):
            overall_std = float(np.std(raw, ddof=0))
            n_unique = int(pd.Series(raw).nunique(dropna=True))
            raw_min = float(np.min(raw))
            raw_max = float(np.max(raw))
        else:
            overall_std = np.nan
            n_unique = 0
            raw_min = np.nan
            raw_max = np.nan

        engine_stds = (
            g.groupby("id", sort=False)["sensor_value"]
            .std(ddof=0)
            .fillna(0.0)
        )

        informative_engines = int(np.sum(engine_stds > eps))
        engine_count = int(len(engine_stds))

        rows.append(
            {
                "sensor": sensor,
                "engine_count": engine_count,
                "raw_n_unique": n_unique,
                "raw_min": raw_min,
                "raw_max": raw_max,
                "raw_std_all_rows": overall_std,
                "informative_engine_count": informative_engines,
                "informative_engine_fraction": (
                    informative_engines / engine_count if engine_count else np.nan
                ),
                "globally_constant": bool(
                    n_unique <= 1 or (np.isfinite(overall_std) and overall_std <= eps)
                ),
            }
        )

    return pd.DataFrame(rows)


def sensor_ranking(summary_index_sensor: pd.DataFrame) -> pd.DataFrame:
    """
    Ranking is deliberately transparent: no arbitrary weighted composite score.

    For each RPD index:
      1. descending median |Spearman(RPD, RUL)|
      2. descending median monotonicity
      3. descending sign consistency

    The ranking is descriptive and should not itself be treated as a fitted
    model-selection criterion.
    """
    parts = []

    for index_name, g in summary_index_sensor.groupby("index", sort=False):
        ranked = g.sort_values(
            [
                "median_abs_spearman_rul",
                "median_monotonicity",
                "spearman_sign_consistency",
            ],
            ascending=[False, False, False],
            na_position="last",
        ).copy()

        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        parts.append(ranked)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def main() -> int:
    args = parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 2

    if not (0 < args.early_late_fraction <= 0.5):
        print(
            "[ERROR] --early-late-fraction must be in (0, 0.5].",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    print("RPD training-set statistical analysis")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print()

    df = pd.read_csv(input_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(
            f"[ERROR] Missing required columns: {missing}",
            file=sys.stderr,
        )
        return 2

    # Stable ordering is useful both for interpretation and reproducibility.
    df = df.sort_values(["id", "sensor", "cycle"]).reset_index(drop=True)

    print(
        f"[INFO] rows={len(df)}, engines={df['id'].nunique()}, "
        f"sensors={df['sensor'].nunique()}"
    )

    # Sensor characteristics at dataset level.
    sensor_chars = sensor_characteristics(df, args.sensor_std_eps)

    # Per-engine/sensor raw variability, used to exclude constant raw sensor
    # trajectories from comparative summaries while retaining them in the
    # detailed output.
    raw_std = (
        df.groupby(["id", "sensor"], sort=False)["sensor_value"]
        .std(ddof=0)
        .fillna(0.0)
        .rename("sensor_value_std")
    )

    rows = []
    grouped = df.groupby(["id", "sensor"], sort=False)

    total_groups = grouped.ngroups
    for group_no, ((engine_id, sensor), g) in enumerate(grouped, start=1):
        g = g.sort_values("cycle")

        cycle = g["cycle"].to_numpy(dtype=float)
        rul = g["RUL"].to_numpy(dtype=float)
        sensor_std = float(raw_std.loc[(engine_id, sensor)])
        informative = bool(sensor_std > args.sensor_std_eps)

        for index_name in INDEX_NAMES:
            y = g[index_name].to_numpy(dtype=float)
            m = trajectory_metrics(
                y,
                cycle,
                rul,
                args.early_late_fraction,
            )

            rows.append(
                {
                    "id": engine_id,
                    "sensor": sensor,
                    "index": index_name,
                    "family": (
                        "local" if index_name in LOCAL_INDICES else "global"
                    ),
                    "sensor_value_std": sensor_std,
                    "informative_sensor_trajectory": informative,
                    "cycle_first": int(g["cycle"].iloc[0]),
                    "cycle_last": int(g["cycle"].iloc[-1]),
                    "rul_first": int(g["RUL"].iloc[0]),
                    "rul_last": int(g["RUL"].iloc[-1]),
                    **m,
                }
            )

        if group_no % 250 == 0 or group_no == total_groups:
            print(
                f"[INFO] analyzed engine/sensor groups "
                f"{group_no}/{total_groups}"
            )

    metrics = pd.DataFrame(rows)

    summary_index_sensor = summarize_index_sensor(metrics)
    summary_index = summarize_index(metrics)
    family_summary = summarize_family(metrics)
    paired_summary = paired_local_global(metrics)
    ranking = sensor_ranking(summary_index_sensor)

    files = {
        "metrics_by_engine_sensor":
            output_dir / "01_RPD_metrics_by_engine_sensor.csv",
        "summary_by_index_sensor":
            output_dir / "02_RPD_summary_by_index_sensor.csv",
        "summary_by_index":
            output_dir / "03_RPD_summary_by_index.csv",
        "local_vs_global":
            output_dir / "04_RPD_local_vs_global.csv",
        "paired_local_global":
            output_dir / "05_RPD_paired_local_global.csv",
        "sensor_ranking":
            output_dir / "06_RPD_sensor_ranking.csv",
        "sensor_characteristics":
            output_dir / "07_sensor_characteristics.csv",
        "manifest":
            output_dir / "50_analysis_manifest.json",
    }

    metrics.to_csv(files["metrics_by_engine_sensor"], index=False)
    summary_index_sensor.to_csv(files["summary_by_index_sensor"], index=False)
    summary_index.to_csv(files["summary_by_index"], index=False)
    family_summary.to_csv(files["local_vs_global"], index=False)
    paired_summary.to_csv(files["paired_local_global"], index=False)
    ranking.to_csv(files["sensor_ranking"], index=False)
    sensor_chars.to_csv(files["sensor_characteristics"], index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_wilcoxon_available": wilcoxon is not None,
        "input_file": str(input_path),
        "input_sha256": sha256(input_path),
        "rows": int(len(df)),
        "engine_count": int(df["id"].nunique()),
        "sensor_count": int(df["sensor"].nunique()),
        "index_names": INDEX_NAMES,
        "local_indices": LOCAL_INDICES,
        "global_indices": GLOBAL_INDICES,
        "early_late_fraction": args.early_late_fraction,
        "sensor_std_eps": args.sensor_std_eps,
        "monotonicity_definition":
            "abs(N_inc-N_dec)/(N-1), ties included in denominator",
        "signed_monotonicity_definition":
            "(N_inc-N_dec)/(N-1), ties included in denominator",
        "sensor_ranking_definition":
            "descending median abs Spearman(RPD,RUL), then median "
            "monotonicity, then Spearman sign consistency; no weighted "
            "composite score",
        "test_set_used": False,
        "truth_set_used": False,
        "outputs": {k: str(v) for k, v in files.items() if k != "manifest"},
    }

    files["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    for label, path in files.items():
        print(f"[OK] {label}: {path}")

    print()
    print("Local/global family summary:")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 160,
        "display.float_format", "{:.6f}".format,
    ):
        print(family_summary.to_string(index=False))

    print()
    if wilcoxon is None:
        print(
            "[WARN] SciPy not available: paired Wilcoxon p-values were "
            "left as NaN. All other analyses are complete."
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
