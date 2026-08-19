#!/usr/bin/env python3
"""
Create an illustrative local-vs-global progressive RPD figure.

The representative engine-sensor trajectory is selected automatically
from the experimental results to avoid manual cherry-picking.

Selection criterion
-------------------
1. Only informative engine-sensor trajectories are considered.
2. Sensors informative in at least --min-engines engines are retained.
3. RPDlep and RPDgep are paired for each engine-sensor trajectory.
4. For each trajectory, a profile is constructed from:
   - global absolute Spearman correlation with RUL
   - global absolute Pearson correlation with RUL
   - global monotonicity
   - global absolute early-late change
   - global RPD range
   and the corresponding global-minus-local differences.
5. Each variable is centered on its median and scaled by its IQR.
6. The engine-sensor trajectory with the smallest multivariate distance
   from the median profile is selected.

The selection can be overridden explicitly with --engine and --sensor.

Expected project layout
-----------------------
RESULTS_DIR/
    01_indices/
        PM_train_RPD_progressive.csv.gz
    02_plot_data/
    03_figures/
    04_figures_hires/
    05_tables/
        01_RPD_metrics_by_engine_sensor.csv

Outputs
-------
02_plot_data/fig_rpd_local_global_example.csv
02_plot_data/fig_rpd_local_global_example_selection.json
03_figures/fig_rpd_local_global_example.pdf
04_figures_hires/fig_rpd_local_global_example.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LOCAL_INDEX = "RPDlep"
GLOBAL_INDEX = "RPDgep"

SELECTION_METRICS = [
    "abs_spearman_rul",
    "abs_pearson_rul",
    "monotonicity",
    "abs_delta_late_early",
    "rpd_range",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a representative raw/local/global progressive RPD trajectory."
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("."),
        help="Root results directory containing 01_indices, 02_plot_data, etc. "
             "(default: current directory)",
    )

    parser.add_argument(
        "--engine",
        type=int,
        default=None,
        help="Override automatic selection and use this engine id.",
    )

    parser.add_argument(
        "--sensor",
        type=str,
        default=None,
        help="Override automatic selection and use this sensor, e.g. s4.",
    )

    parser.add_argument(
        "--min-engines",
        type=int,
        default=90,
        help="Minimum number of informative engines required for a sensor "
             "during automatic selection (default: 90).",
    )

    parser.add_argument(
        "--basename",
        type=str,
        default="fig_rpd_local_global_example",
        help="Base name for generated files.",
    )

    parser.add_argument(
        "--png-dpi",
        type=int,
        default=600,
        help="Resolution of high-resolution PNG output (default: 600).",
    )

    return parser.parse_args()


def find_existing(root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path

    formatted = "\n".join(f"  - {root / c}" for c in candidates)
    raise FileNotFoundError(
        "None of the expected input files was found:\n" + formatted
    )


def load_metrics(results_dir: Path) -> tuple[pd.DataFrame, Path]:
    path = find_existing(
        results_dir,
        [
            "05_tables/01_RPD_metrics_by_engine_sensor.csv",
            "05_tables_/01_RPD_metrics_by_engine_sensor.csv",
        ],
    )

    df = pd.read_csv(path)

    required = {
        "id",
        "sensor",
        "index",
        "informative_sensor_trajectory",
        *SELECTION_METRICS,
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    return df, path


def representative_case(
    metrics: pd.DataFrame,
    min_engines: int,
) -> tuple[int, str, dict]:
    """
    Select the engine-sensor pair closest to the multivariate median
    local/global behavior.
    """

    df = metrics[
        metrics["informative_sensor_trajectory"].astype(bool)
        & metrics["index"].isin([LOCAL_INDEX, GLOBAL_INDEX])
    ].copy()

    # Number of informative engines for each sensor.
    sensor_counts = (
        df[df["index"] == GLOBAL_INDEX]
        .groupby("sensor")["id"]
        .nunique()
    )

    eligible_sensors = sensor_counts[
        sensor_counts >= min_engines
    ].index

    df = df[df["sensor"].isin(eligible_sensors)].copy()

    if df.empty:
        raise RuntimeError(
            f"No sensors satisfy min_engines={min_engines}."
        )

    # Convert local/global rows into one row per engine-sensor pair.
    pivot = df.pivot(
        index=["id", "sensor"],
        columns="index",
        values=SELECTION_METRICS,
    )

    pivot.columns = [
        f"{metric}_{index}"
        for metric, index in pivot.columns
    ]

    pivot = pivot.dropna().copy()

    # Global metrics themselves plus global-minus-local improvements.
    selection_columns = []

    for metric in SELECTION_METRICS:
        global_col = f"{metric}_{GLOBAL_INDEX}"
        local_col = f"{metric}_{LOCAL_INDEX}"
        delta_col = f"delta_{metric}"

        pivot[delta_col] = pivot[global_col] - pivot[local_col]

        selection_columns.append(global_col)
        selection_columns.append(delta_col)

    # Robust distance to the multivariate median.
    medians = pivot[selection_columns].median()
    q25 = pivot[selection_columns].quantile(0.25)
    q75 = pivot[selection_columns].quantile(0.75)
    iqr = q75 - q25

    # If any variable has zero IQR, it should not affect the distance.
    iqr = iqr.replace(0.0, np.nan)

    standardized = (
        (pivot[selection_columns] - medians) / iqr
    ).fillna(0.0)

    pivot["representative_distance"] = np.sqrt(
        np.mean(np.square(standardized), axis=1)
    )

    selected_index = pivot["representative_distance"].idxmin()
    engine, sensor = selected_index

    row = pivot.loc[selected_index]

    info = {
        "selection_method": (
            "minimum robust multivariate distance from the median "
            "RPDlep/RPDgep engine-sensor profile"
        ),
        "local_index": LOCAL_INDEX,
        "global_index": GLOBAL_INDEX,
        "minimum_informative_engines_per_sensor": int(min_engines),
        "eligible_sensor_count": int(len(eligible_sensors)),
        "eligible_sensors": sorted(
            eligible_sensors.tolist(),
            key=lambda s: int(s[1:]) if s.startswith("s") else s,
        ),
        "selected_engine": int(engine),
        "selected_sensor": str(sensor),
        "representative_distance": float(
            row["representative_distance"]
        ),
        "metrics": {},
    }

    for metric in SELECTION_METRICS:
        local_value = float(row[f"{metric}_{LOCAL_INDEX}"])
        global_value = float(row[f"{metric}_{GLOBAL_INDEX}"])

        info["metrics"][metric] = {
            LOCAL_INDEX: local_value,
            GLOBAL_INDEX: global_value,
            "global_minus_local": global_value - local_value,
        }

    return int(engine), str(sensor), info


def explicit_case_info(
    metrics: pd.DataFrame,
    engine: int,
    sensor: str,
) -> dict:
    subset = metrics[
        (metrics["id"] == engine)
        & (metrics["sensor"] == sensor)
        & metrics["index"].isin([LOCAL_INDEX, GLOBAL_INDEX])
    ]

    info = {
        "selection_method": "explicit command-line selection",
        "local_index": LOCAL_INDEX,
        "global_index": GLOBAL_INDEX,
        "selected_engine": int(engine),
        "selected_sensor": str(sensor),
        "metrics": {},
    }

    for metric in SELECTION_METRICS:
        values = {}

        for index in [LOCAL_INDEX, GLOBAL_INDEX]:
            row = subset[subset["index"] == index]
            if len(row) == 1 and pd.notna(row.iloc[0][metric]):
                values[index] = float(row.iloc[0][metric])

        if LOCAL_INDEX in values and GLOBAL_INDEX in values:
            values["global_minus_local"] = (
                values[GLOBAL_INDEX] - values[LOCAL_INDEX]
            )

        info["metrics"][metric] = values

    return info


def load_progressive_case(
    results_dir: Path,
    engine: int,
    sensor: str,
) -> tuple[pd.DataFrame, Path]:
    path = find_existing(
        results_dir,
        [
            "01_indices/PM_train_RPD_progressive.csv.gz",
            "05_tables/PM_train_RPD_progressive.csv.gz",
            "05_tables/PM_train_RPD_progressive.csv",
        ],
    )

    required_columns = [
        "id",
        "cycle",
        "RUL",
        "sensor",
        "sensor_value",
        LOCAL_INDEX,
        GLOBAL_INDEX,
    ]

    # The dataset is modest enough (~431k rows) to load directly.
    df = pd.read_csv(
        path,
        usecols=required_columns,
    )

    case = df[
        (df["id"] == engine)
        & (df["sensor"] == sensor)
    ].copy()

    if case.empty:
        raise RuntimeError(
            f"No progressive data found for engine={engine}, "
            f"sensor={sensor} in {path}"
        )

    case = case.sort_values("cycle").reset_index(drop=True)

    if case["cycle"].duplicated().any():
        raise RuntimeError(
            "Duplicate cycles found in selected engine-sensor trajectory."
        )

    return case, path


def make_figure(
    case: pd.DataFrame,
    engine: int,
    sensor: str,
    pdf_path: Path,
    png_path: Path,
    png_dpi: int,
) -> None:
    """
    Two-panel publication-oriented figure:
      (a) raw sensor trajectory
      (b) local and global progressive RPD trajectories
    """

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )

    ax_raw, ax_rpd = axes

    # Panel (a): raw sensor.
    ax_raw.plot(
        case["cycle"],
        case["sensor_value"],
        linewidth=1.0,
    )

    ax_raw.set_ylabel("Sensor value")
    ax_raw.text(
        0.01,
        0.96,
        "(a)",
        transform=ax_raw.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # Panel (b): local/global progressive RPD.
    ax_rpd.plot(
        case["cycle"],
        case[LOCAL_INDEX],
        linewidth=0.9,
        label=LOCAL_INDEX,
    )

    ax_rpd.plot(
        case["cycle"],
        case[GLOBAL_INDEX],
        linewidth=1.4,
        linestyle="--",
        label=GLOBAL_INDEX,
    )

    ax_rpd.set_xlabel("Cycle")
    ax_rpd.set_ylabel("Progressive RPD")
    ax_rpd.set_ylim(-1.05, 1.05)
    ax_rpd.legend(
        frameon=False,
        loc="best",
    )

    ax_rpd.text(
        0.01,
        0.96,
        "(b)",
        transform=ax_rpd.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # Keep the figure itself clean; engine/sensor are documented in
    # metadata and should preferably appear in the manuscript caption.
    for ax in axes:
        ax.tick_params(direction="out")

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=png_dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir.resolve()

    plot_data_dir = results_dir / "02_plot_data"
    figures_dir = results_dir / "03_figures"
    hires_dir = results_dir / "04_figures_hires"

    for directory in [
        plot_data_dir,
        figures_dir,
        hires_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    metrics, metrics_path = load_metrics(results_dir)

    if (args.engine is None) != (args.sensor is None):
        raise SystemExit(
            "--engine and --sensor must either both be specified "
            "or both be omitted."
        )

    if args.engine is None:
        engine, sensor, selection_info = representative_case(
            metrics,
            min_engines=args.min_engines,
        )
    else:
        engine = args.engine
        sensor = args.sensor
        selection_info = explicit_case_info(
            metrics,
            engine,
            sensor,
        )

    case, progressive_path = load_progressive_case(
        results_dir,
        engine,
        sensor,
    )

    data_output = (
        plot_data_dir / f"{args.basename}.csv"
    )

    metadata_output = (
        plot_data_dir / f"{args.basename}_selection.json"
    )

    pdf_output = (
        figures_dir / f"{args.basename}.pdf"
    )

    png_output = (
        hires_dir / f"{args.basename}.png"
    )

    case.to_csv(data_output, index=False)

    selection_info.update(
        {
            "progressive_input": str(progressive_path),
            "metrics_input": str(metrics_path),
            "number_of_cycles_plotted": int(len(case)),
            "first_cycle": int(case["cycle"].min()),
            "last_cycle": int(case["cycle"].max()),
            "first_RUL": float(case["RUL"].iloc[0]),
            "last_RUL": float(case["RUL"].iloc[-1]),
            "plot_data_file": str(data_output),
            "pdf_file": str(pdf_output),
            "png_file": str(png_output),
        }
    )

    with metadata_output.open("w", encoding="utf-8") as f:
        json.dump(
            selection_info,
            f,
            indent=2,
            ensure_ascii=False,
        )

    make_figure(
        case=case,
        engine=engine,
        sensor=sensor,
        pdf_path=pdf_output,
        png_path=png_output,
        png_dpi=args.png_dpi,
    )

    print()
    print("Representative RPD figure generated")
    print("-----------------------------------")
    print(f"Results directory : {results_dir}")
    print(f"Engine            : {engine}")
    print(f"Sensor            : {sensor}")
    print(f"Cycles            : {len(case)}")
    print(f"Progressive input : {progressive_path}")
    print(f"Metrics input     : {metrics_path}")
    print()
    print("Selected metrics:")

    for metric, values in selection_info["metrics"].items():
        if (
            LOCAL_INDEX in values
            and GLOBAL_INDEX in values
        ):
            print(
                f"  {metric:26s} "
                f"{LOCAL_INDEX}={values[LOCAL_INDEX]:.6f}  "
                f"{GLOBAL_INDEX}={values[GLOBAL_INDEX]:.6f}  "
                f"delta={values['global_minus_local']:+.6f}"
            )

    print()
    print("Outputs:")
    print(f"  {data_output}")
    print(f"  {metadata_output}")
    print(f"  {pdf_output}")
    print(f"  {png_output}")


if __name__ == "__main__":
    main()
    
    
