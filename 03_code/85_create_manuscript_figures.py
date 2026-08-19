#!/usr/bin/env python3
"""Create the three main quantitative figures for the RPD/C-MAPSS manuscript.

Outputs
-------
Figure 2: temporal-shuffle validation (observed vs permutation null)
Figure 3: engine-wise GroupKFold RMSE distributions for all six configurations
Figure 4: independent-test predicted vs true RUL

The script also writes the exact plotting data and a small JSON manifest so that
all manuscript figures are traceable and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FIG2_INPUT = "14_RPD_temporal_shuffle_dataset_level.csv"
FIG3_INPUT = "16_RUL_groupcv_engine_metrics.csv"
FIG4_INPUT = "27_PM_test_RUL_evaluation_by_engine.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_default_tables() -> Path:
    candidates = [
        Path("04_results/05_tables"),
        Path("05_tables"),
        Path("."),
    ]
    for c in candidates:
        if all((c / f).exists() for f in (FIG2_INPUT, FIG3_INPUT, FIG4_INPUT)):
            return c
    return Path("04_results/05_tables")


def save_figure(fig, pdf_path: Path, png_path: Path) -> None:
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_temporal_shuffle(df: pd.DataFrame, plot_data: Path, figures: Path, hires: Path):
    required = {
        "index", "metric", "engine_count", "permutations",
        "observed_engine_median", "null_median", "null_q025", "null_q975",
        "empirical_p_upper",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{FIG2_INPUT}: missing columns: {sorted(missing)}")

    index_order = ["RPDgep", "RPDgea", "RPDgepr"]
    metric_order = [
        "abs_spearman_rul",
        "abs_pearson_rul",
        "monotonicity",
        "abs_delta_late_early",
    ]
    metric_labels = {
        "abs_spearman_rul": r"Absolute Spearman correlation $|\rho_S|$",
        "abs_pearson_rul": r"Absolute Pearson correlation $|r|$",
        "monotonicity": "Monotonicity",
        "abs_delta_late_early": r"Absolute early--late change $|\Delta_{EL}|$",
    }

    out = df[df["index"].isin(index_order) & df["metric"].isin(metric_order)].copy()
    out["index"] = pd.Categorical(out["index"], index_order, ordered=True)
    out["metric"] = pd.Categorical(out["metric"], metric_order, ordered=True)
    out = out.sort_values(["metric", "index"]).reset_index(drop=True)
    out.to_csv(plot_data / "fig_temporal_shuffle_validation.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.09, top=0.87, wspace=0.23, hspace=0.28)
    axes = axes.ravel()
    x = np.arange(len(index_order), dtype=float)

    for panel, (ax, metric) in enumerate(zip(axes, metric_order)):
        s = out[out["metric"] == metric].set_index("index").reindex(index_order)
        obs = s["observed_engine_median"].to_numpy(float)
        null = s["null_median"].to_numpy(float)
        q025 = s["null_q025"].to_numpy(float)
        q975 = s["null_q975"].to_numpy(float)
        yerr = np.vstack([null - q025, q975 - null])

        # Null distribution: open squares and its central 95% interval.
        ax.errorbar(
            x - 0.07, null, yerr=yerr,
            fmt="s", markersize=5.0, markerfacecolor="white",
            markeredgecolor="0.25", ecolor="0.30", elinewidth=1.5,
        capsize=5.0, capthick=1.4, linestyle="none", zorder=2,
        )
        # Observed ordering: filled circles.
        ax.plot(x + 0.07, obs, "o", markersize=5.5, color="black", zorder=3)

        ax.set_xticks(x, index_order)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric_labels[metric])
        ax.grid(axis="y", linewidth=0.5, alpha=0.25)
        ax.text(0.02, 0.96, f"({chr(97 + panel)})", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="black",
               markersize=5.5, label="Observed temporal order"),
        Line2D([0], [0], marker="s", linestyle="none", color="0.35",
               markerfacecolor="white", markersize=5.0,
               label="Temporal-shuffle null: median and 95% interval"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.985), columnspacing=1.8)

    pdf = figures / "fig_temporal_shuffle_validation.pdf"
    png = hires / "fig_temporal_shuffle_validation.png"
    save_figure(fig, pdf, png)
    return pdf, png


def figure_groupcv_rmse(df: pd.DataFrame, plot_data: Path, figures: Path, hires: Path):
    required = {"feature_set", "model", "id", "rmse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{FIG3_INPUT}: missing columns: {sorted(missing)}")

    order = [
        ("raw_plus_rpd", "ridge"),
        ("rpd_global", "hgb"),
        ("raw_plus_rpd", "hgb"),
        ("rpd_global", "ridge"),
        ("raw_sensor", "hgb"),
        ("raw_sensor", "ridge"),
    ]
    labels = {
        ("raw_plus_rpd", "ridge"): "Raw + global RPD / Ridge",
        ("rpd_global", "hgb"): "Global RPD / HGB",
        ("raw_plus_rpd", "hgb"): "Raw + global RPD / HGB",
        ("rpd_global", "ridge"): "Global RPD / Ridge",
        ("raw_sensor", "hgb"): "Raw sensors / HGB",
        ("raw_sensor", "ridge"): "Raw sensors / Ridge",
    }

    out = df[["feature_set", "model", "fold", "id", "n_points", "rmse", "mae", "r2"]].copy()
    out["configuration"] = [labels.get((f, m), f"{f} / {m}") for f, m in zip(out.feature_set, out.model)]
    rank = {labels[k]: i + 1 for i, k in enumerate(order)}
    out["display_order"] = out["configuration"].map(rank)
    out = out.sort_values(["display_order", "id"]).reset_index(drop=True)
    out.to_csv(plot_data / "fig_groupcv_engine_rmse.csv", index=False)

    arrays = []
    display_labels = []
    means = []
    for k in order:
        s = df[(df["feature_set"] == k[0]) & (df["model"] == k[1])]["rmse"].to_numpy(float)
        if len(s) == 0:
            raise ValueError(f"No rows found for configuration {k}")
        arrays.append(s)
        display_labels.append(labels[k])
        means.append(float(np.mean(s)))

    # Reverse so the best-ranked configuration is shown at the top.
    arrays_plot = arrays[::-1]
    labels_plot = display_labels[::-1]
    means_plot = means[::-1]
    positions = np.arange(1, len(arrays_plot) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    bp = ax.boxplot(
        arrays_plot,
        vert=False,
        positions=positions,
        widths=0.55,
        patch_artist=False,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3.0, alpha=0.45),
        medianprops=dict(linewidth=1.3, color="black"),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    ax.plot(means_plot, positions, "D", color="black", markersize=4.5,
            linestyle="none", label="Mean per-engine RMSE")

    ax.set_yticks(positions, labels_plot)
    ax.set_xlabel("Per-engine RMSE (cycles)")
    ax.grid(axis="x", linewidth=0.5, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    pdf = figures / "fig_groupcv_engine_rmse.pdf"
    png = hires / "fig_groupcv_engine_rmse.png"
    save_figure(fig, pdf, png)
    return pdf, png


def figure_independent_test(df: pd.DataFrame, plot_data: Path, figures: Path, hires: Path):
    required = {"id", "RUL_pred", "RUL_true", "error", "abs_error"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{FIG4_INPUT}: missing columns: {sorted(missing)}")

    out = df.copy().sort_values("id").reset_index(drop=True)
    out.to_csv(plot_data / "fig_independent_test_rul.csv", index=False)

    y_true = out["RUL_true"].to_numpy(float)
    y_pred = out["RUL_pred"].to_numpy(float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")

    worst_i = int(np.argmax(np.abs(err)))
    worst = out.iloc[worst_i]

    maxv = float(max(y_true.max(), y_pred.max()))
    upper = np.ceil((maxv + 10.0) / 20.0) * 20.0

    fig, ax = plt.subplots(figsize=(5.8, 5.4), constrained_layout=True)
    ax.scatter(y_true, y_pred, s=28, facecolors="none", edgecolors="black", linewidths=0.8)
    ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1.0, color="0.35",
            label="Identity line")

    ax.scatter([worst["RUL_true"]], [worst["RUL_pred"]], s=56,
               facecolors="black", edgecolors="black", zorder=4)
    ax.annotate(
        f"Engine {int(worst['id'])}\nerror = {worst['error']:+.1f} cycles",
        xy=(worst["RUL_true"], worst["RUL_pred"]),
        xytext=(8, -28), textcoords="offset points",
        ha="left", va="top",
        arrowprops=dict(arrowstyle="-", linewidth=0.8),
    )

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True residual RUL (cycles)")
    ax.set_ylabel("Predicted residual RUL (cycles)")
    ax.grid(linewidth=0.5, alpha=0.20)

    metrics_text = f"RMSE = {rmse:.2f}\nMAE = {mae:.2f}\n$R^2$ = {r2:.3f}"
    ax.text(0.04, 0.96, metrics_text, transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.65", linewidth=0.8))

    pdf = figures / "fig_independent_test_rul.pdf"
    png = hires / "fig_independent_test_rul.png"
    save_figure(fig, pdf, png)
    return pdf, png, {"rmse": rmse, "mae": mae, "r2": r2,
                      "worst_engine": int(worst["id"]),
                      "worst_error": float(worst["error"])}


def parse_args():
    default_tables = resolve_default_tables()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables", type=Path, default=default_tables,
                   help=f"Input tables directory (default auto-detected: {default_tables})")
    p.add_argument("--plot-data", type=Path, default=Path("04_results/02_plot_data"),
                   help="Directory for exact plotting-data CSVs")
    p.add_argument("--figures", type=Path, default=Path("04_results/03_figures"),
                   help="Directory for vector PDF figures")
    p.add_argument("--hires", type=Path, default=Path("04_results/04_figures_hires"),
                   help="Directory for 600-dpi PNG figures")
    return p.parse_args()


def main():
    args = parse_args()
    tables = args.tables
    plot_data = ensure_dir(args.plot_data)
    figures = ensure_dir(args.figures)
    hires = ensure_dir(args.hires)

    paths = {name: tables / name for name in (FIG2_INPUT, FIG3_INPUT, FIG4_INPUT)}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    df2 = pd.read_csv(paths[FIG2_INPUT])
    df3 = pd.read_csv(paths[FIG3_INPUT])
    df4 = pd.read_csv(paths[FIG4_INPUT])

    fig2_pdf, fig2_png = figure_temporal_shuffle(df2, plot_data, figures, hires)
    fig3_pdf, fig3_png = figure_groupcv_rmse(df3, plot_data, figures, hires)
    fig4_pdf, fig4_png, test_metrics = figure_independent_test(df4, plot_data, figures, hires)

    manifest = {
        "script": Path(__file__).name,
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "figure_2": {"pdf": str(fig2_pdf), "png": str(fig2_png)},
            "figure_3": {"pdf": str(fig3_pdf), "png": str(fig3_png)},
            "figure_4": {"pdf": str(fig4_pdf), "png": str(fig4_png)},
        },
        "independent_test_metrics_recomputed_from_figure_input": test_metrics,
    }
    manifest_path = plot_data / "80_manuscript_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Created manuscript figures:")
    print(f"  Fig. 2 PDF: {fig2_pdf}")
    print(f"  Fig. 2 PNG: {fig2_png}")
    print(f"  Fig. 3 PDF: {fig3_pdf}")
    print(f"  Fig. 3 PNG: {fig3_png}")
    print(f"  Fig. 4 PDF: {fig4_pdf}")
    print(f"  Fig. 4 PNG: {fig4_png}")
    print(f"  Manifest:   {manifest_path}")
    print("Independent-test check:")
    print(f"  RMSE = {test_metrics['rmse']:.6f}")
    print(f"  MAE  = {test_metrics['mae']:.6f}")
    print(f"  R2   = {test_metrics['r2']:.6f}")
    print(f"  Worst engine = {test_metrics['worst_engine']} ({test_metrics['worst_error']:+.6f} cycles)")


if __name__ == "__main__":
    main()
