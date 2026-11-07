#!/usr/bin/env python3
"""RPDindex IJPHM Stage 10: retrospective engine-wise failure-proximity pilot.

Original RPDgep is unchanged: (c+ S+ - c- S-) / (c+ S+ + c- S-).
The fast implementation uses a Fenwick tree to update pairwise counts/sums
from OBSERVED prefixes only. The full engine's value domain is used only to
construct an exact, order-preserving coordinate map for the data structure;
future values do not enter prefix counts, sums or features.

Scientific status: exploratory grouped-CV on run-to-failure engines; NO new
blinded test and NO classification of failed components.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline

SENSORS = [f"s{i}" for i in range(1, 22)]
SETTINGS = [f"setting{i}" for i in range(1, 4)]
DATASETS = {
    "FD001": "PM_train.csv",
    "FD002": "PM_train_FD002.csv",
    "FD003": "PM_train_FD003.csv",
    "FD004": "PM_train_FD004.csv",
}


class Fenwick:
    """Point increments and prefix sums, 1-indexed."""

    def __init__(self, n: int):
        self.tree = np.zeros(n + 1, dtype=np.float64)

    def add(self, i: int, value: float) -> None:
        while i < len(self.tree):
            self.tree[i] += value
            i += i & -i

    def prefix(self, i: int) -> float:
        total = 0.0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total


def rpdgep_prefix_fast(x: np.ndarray) -> np.ndarray:
    """Exact progressive RPDgep using all available positive/negative pairs.

    O(n log n), O(n) extra memory; computes the UNCHANGED definition.
    """
    from bisect import bisect_left, bisect_right

    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError("Expected one-dimensional finite numeric trajectory")
    if len(a) == 0:
        return np.empty(0, dtype=np.float64)

    # Only a coordinate dictionary, not statistics: prefix results are
    # provably identical to recomputing on the prefix itself (unit tested).
    coords = sorted(set(a.tolist()))
    count_tree, sum_tree = Fenwick(len(coords)), Fenwick(len(coords))
    n_seen, total_sum = 0, 0.0
    count_plus = count_minus = 0
    sum_plus = sum_minus = 0.0
    out = np.zeros(len(a), dtype=np.float64)

    for k, current in enumerate(a):
        lo = bisect_left(coords, current)
        hi = bisect_right(coords, current)
        n_less = int(round(count_tree.prefix(lo)))
        sum_less = sum_tree.prefix(lo)
        n_leq = int(round(count_tree.prefix(hi)))
        sum_leq = sum_tree.prefix(hi)
        n_greater = n_seen - n_leq
        sum_greater = total_sum - sum_leq
        count_plus += n_less
        count_minus += n_greater
        sum_plus += max(0.0, current * n_less - sum_less)
        sum_minus += max(0.0, sum_greater - current * n_greater)
        pos = count_plus * sum_plus
        neg = count_minus * sum_minus
        if pos + neg > 0:
            out[k] = (pos - neg) / (pos + neg)
        count_tree.add(lo + 1, 1)
        sum_tree.add(lo + 1, current)
        n_seen += 1
        total_sum += current
    return out


def rpdgep_bruteforce(x: np.ndarray) -> float:
    """Small reference formula for tests, not intended for long trajectories."""
    a = np.asarray(x, dtype=np.float64)
    pp, nn, sp, sn = 0, 0, 0.0, 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            d = a[j] - a[i]
            if d > 0:
                pp += 1
                sp += d
            elif d < 0:
                nn += 1
                sn -= d
    p, n = pp * sp, nn * sn
    return (p - n) / (p + n) if p + n > 0 else 0.0


def ols_prefix_slope(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Historical OLS slope: no future observations, no learned preprocessing."""
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    n = np.arange(1, len(t) + 1, dtype=float)
    st = np.cumsum(t)
    sx = np.cumsum(v)
    stt = np.cumsum(t * t)
    stx = np.cumsum(t * v)
    den = n * stt - st**2
    ans = np.zeros(len(t), dtype=float)
    valid = den > 1e-12
    ans[valid] = (n[valid] * stx[valid] - st[valid] * sx[valid]) / den[valid]
    return ans


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing data: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    expected = ["id", "cycle", *SETTINGS, *SENSORS]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Input columns absent: {missing}; got {list(df.columns)}")
    df = df.loc[:, expected].copy()
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df.isna().any().any() or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Missing/nonfinite values: stop, do not impute or bridge gaps")
    if df.duplicated(["id", "cycle"]).any():
        raise ValueError("Duplicate (engine, cycle) records")
    if ((df["id"] % 1 != 0) | (df["cycle"] % 1 != 0)).any():
        raise ValueError("Engine IDs and cycles must be integers")
    df = df.sort_values(["id", "cycle"], kind="mergesort").reset_index(drop=True)
    for engine, group in df.groupby("id", sort=False):
        cs = group["cycle"].to_numpy(dtype=int)
        if cs[0] != 1 or not np.all(np.diff(cs) == 1):
            raise ValueError(f"Engine {engine}: expected consecutive cycles starting at 1")
    return df


def construct_prefix_features(df: pd.DataFrame, horizon: int, min_prefix: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only prefixes up to current cycle. Exclude failure cycle (RUL=0)."""
    parts = []
    excluded = []
    for engine, g in df.groupby("id", sort=True):
        cycle = g["cycle"].to_numpy(dtype=int)
        if len(cycle) < min_prefix + horizon + 2:
            excluded.append({"id": int(engine), "reason": "too_short_for_both_classes", "last_cycle": int(cycle[-1])})
            continue
        rul = int(cycle[-1]) - cycle
        x = g[SENSORS].to_numpy(dtype=float)
        rep = {"id": np.repeat(int(engine), len(g)), "cycle": cycle, "rul": rul,
               "event_within_horizon": (rul > 0) & (rul <= horizon)}
        for j, name in enumerate(SETTINGS):
            rep[name] = g[name].to_numpy(dtype=float)
        for j, name in enumerate(SENSORS):
            rep[name] = x[:, j]
            rep[f"rpd_{name}"] = rpdgep_prefix_fast(x[:, j])
            rep[f"slope_{name}"] = ols_prefix_slope(cycle, x[:, j])
        pg = pd.DataFrame(rep)
        pg = pg[(pg["cycle"] >= min_prefix) & (pg["rul"] >= 1)].copy()
        if pg["event_within_horizon"].nunique() != 2:
            excluded.append({"id": int(engine), "reason": "no_two_classes_after_filter", "last_cycle": int(cycle[-1])})
            continue
        parts.append(pg)
    if not parts:
        raise ValueError("No eligible engines after feature construction")
    return pd.concat(parts, ignore_index=True), pd.DataFrame(excluded, columns=["id", "reason", "last_cycle"])


def balanced_engine_weights(engine_ids: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Equal contribution per training engine, then balance classes on train only."""
    _, inverse, counts = np.unique(engine_ids, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    for c in (0, 1):
        mask = y == c
        if not mask.any():
            raise ValueError("Fold training data lacks one of the two event classes")
        w[mask] *= 0.5 / w[mask].sum()
    w *= len(w) / w.sum()
    return w


def stable_alarm_cycles(cycles: np.ndarray, probabilities: np.ndarray, threshold: float, consecutive: int) -> list[int]:
    """Alarm confirmations must occur on consecutive physical cycles."""
    hits = []
    streak = 0
    last = None
    for t, p in zip(cycles, probabilities):
        t = int(t)
        if p >= threshold and (last is not None and t == last + 1):
            streak += 1
        elif p >= threshold:
            streak = 1
        else:
            streak = 0
        if streak >= consecutive:
            hits.append(t)
        last = t
    return hits


def engine_alarms(pred: pd.DataFrame, model: str, horizon: int, threshold: float, consecutive: int) -> pd.DataFrame:
    rows = []
    for engine, g in pred.groupby("id", sort=True):
        g = g.sort_values("cycle")
        c = g["cycle"].to_numpy(dtype=int)
        p = g[model].to_numpy(dtype=float)
        alarm_cycles = stable_alarm_cycles(c, p, threshold, consecutive)
        rul_for_cycle = dict(zip(c, g["rul"].to_numpy(dtype=int)))
        first_rul = rul_for_cycle[alarm_cycles[0]] if alarm_cycles else np.nan
        within = [rul_for_cycle[t] for t in alarm_cycles if rul_for_cycle[t] <= horizon]
        first_within = within[0] if within else np.nan
        rows.append({"id": int(engine), "model": model,
                     "first_alarm_rul": first_rul,
                     "first_alarm_premature": int(bool(alarm_cycles) and first_rul > horizon),
                     "first_alarm_in_window": int(bool(alarm_cycles) and first_rul <= horizon),
                     "ever_alarm_in_window": int(bool(within)),
                     "first_in_window_rul": first_within,
                     "never_alarm": int(not alarm_cycles)})
    return pd.DataFrame(rows)


def paired_bootstrap_mean_ci(diffs: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sims = np.empty(resamples, dtype=float)
    for i in range(resamples):
        sims[i] = rng.choice(diffs, size=len(diffs), replace=True).mean()
    return float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))


def run(args: argparse.Namespace) -> Path:
    project = Path(args.project_root).expanduser().resolve()
    path = project / "02_raw_data" / DATASETS[args.dataset]
    out = Path(args.output).expanduser().resolve()
    if not (project / "03_code").is_dir() or not (project / "04_results").is_dir():
        raise ValueError(f"Not the expected IJPHM article project: {project}")
    if not out.is_relative_to((project / "04_results").resolve()):
        raise ValueError("Output must be inside project/04_results")
    if args.horizon < 2 or args.min_prefix < 3 or args.confirmations < 1:
        raise ValueError("Invalid horizon, prefix, or confirmations")
    if args.threshold < 0 or args.threshold > 1:
        raise ValueError("Threshold must be in [0,1]")

    original = read_dataset(path)
    feat, excluded = construct_prefix_features(original, args.horizon, args.min_prefix)
    eng = feat["id"].to_numpy(dtype=int)
    y = feat["event_within_horizon"].to_numpy(dtype=int)
    feature_sets = {
        "cycle_only": ["cycle"],
        "rpd_only": [*[f"rpd_{s}" for s in SENSORS]],
        "raw_age": ["cycle", *SETTINGS, *SENSORS],
        "raw_age_rpd_gep": ["cycle", *SETTINGS, *SENSORS, *[f"rpd_{s}" for s in SENSORS]],
        "raw_age_ols": ["cycle", *SETTINGS, *SENSORS, *[f"slope_{s}" for s in SENSORS]],
        "raw_age_ols_rpd_gep": ["cycle", *SETTINGS, *SENSORS,
                                *[f"slope_{s}" for s in SENSORS], *[f"rpd_{s}" for s in SENSORS]],
    }
    if feat["id"].nunique() < args.folds:
        raise ValueError(f"Need >= {args.folds} engines for grouped folds")
    outputs = feat[["id", "cycle", "rul", "event_within_horizon"]].copy()
    outputs["fold"] = 0
    fold_split = GroupKFold(n_splits=args.folds)
    fold_desc = []
    for fold, (train, valid) in enumerate(fold_split.split(feat, y, eng), 1):
        tr, va = set(eng[train]), set(eng[valid])
        if tr.intersection(va):
            raise AssertionError("Data leakage: overlapping engines")
        wt = balanced_engine_weights(eng[train], y[train])
        outputs.loc[valid, "fold"] = fold
        for label, cols in feature_sets.items():
            pipe = Pipeline([
                ("standardize", StandardScaler()),
                ("logistic", LogisticRegression(C=1.0, max_iter=1500, solver="lbfgs")),
            ])
            pipe.fit(feat.iloc[train][cols], y[train], logistic__sample_weight=wt)
            outputs.loc[valid, label] = pipe.predict_proba(feat.iloc[valid][cols])[:, 1]
        fold_desc.append({"fold": fold, "train_engines": len(tr), "valid_engines": len(va),
                          "train_records": len(train), "valid_records": len(valid),
                          "train_positives": int(y[train].sum()), "valid_positives": int(y[valid].sum())})
        print(f"{args.dataset} fold {fold}/{args.folds}: {len(tr)} train engines, {len(va)} validation engines", flush=True)

    rows, alarm_frames = [], []
    for engine, g in outputs.groupby("id", sort=True):
        yy = g["event_within_horizon"].to_numpy(dtype=int)
        for name in feature_sets:
            prob = g[name].to_numpy(dtype=float)
            rows.append({"id": int(engine), "model": name,
                         "average_precision": float(average_precision_score(yy, prob)),
                         "roc_auc": float(roc_auc_score(yy, prob)),
                         "brier": float(brier_score_loss(yy, prob)),
                         "positive_prevalence": float(np.mean(yy))})
    by_eng = pd.DataFrame(rows)
    for name in feature_sets:
        alarm_frames.append(engine_alarms(outputs, name, args.horizon, args.threshold, args.confirmations))
    alarms = pd.concat(alarm_frames, ignore_index=True)

    summaries = []
    for name in feature_sets:
        g, a = by_eng[by_eng["model"] == name], alarms[alarms["model"] == name]
        inwindow = a["first_in_window_rul"].dropna()
        summaries.append({"model": name, "n_engines": len(g), "n_records": len(outputs),
                          "mean_engine_average_precision": g["average_precision"].mean(),
                          "mean_engine_roc_auc": g["roc_auc"].mean(),
                          "mean_engine_brier": g["brier"].mean(),
                          "mean_engine_event_prevalence": g["positive_prevalence"].mean(),
                          "premature_first_alarm_engine_fraction": a["first_alarm_premature"].mean(),
                          "ever_alarm_in_window_engine_fraction": a["ever_alarm_in_window"].mean(),
                          "never_alarm_engine_fraction": a["never_alarm"].mean(),
                          "median_first_in_window_rul_among_alerted": inwindow.median() if len(inwindow) else np.nan})
    summary = pd.DataFrame(summaries)
    piv = by_eng.pivot(index="id", columns="model", values="average_precision").sort_index()
    contrasts = []
    for baseline, variant in [("raw_age", "raw_age_rpd_gep"),
                              ("raw_age_ols", "raw_age_ols_rpd_gep"),
                              ("raw_age_ols", "raw_age_rpd_gep")]:
        delta = (piv[variant] - piv[baseline]).to_numpy(dtype=float)
        ci = paired_bootstrap_mean_ci(delta, args.bootstrap, args.seed)
        contrasts.append({"baseline": baseline, "variant": variant, "n_engines": len(delta),
                          "mean_engine_ap_gain": float(np.mean(delta)),
                          "median_engine_ap_gain": float(np.median(delta)),
                          "fraction_engines_better": float(np.mean(delta > 0)),
                          "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1]})
    paired = pd.DataFrame(contrasts)

    # Do not write anything until all models and analyses have finished.
    out.mkdir(parents=True, exist_ok=False)
    outputs.to_csv(out / "oof_predictions.csv.gz", index=False, compression="gzip", float_format="%.12g")
    by_eng.to_csv(out / "engine_metrics.csv", index=False)
    alarms.to_csv(out / "engine_alarms.csv", index=False)
    summary.to_csv(out / "model_summary.csv", index=False)
    paired.to_csv(out / "paired_engine_ap.csv", index=False)
    excluded.to_csv(out / "excluded_engines.csv", index=False)
    pd.DataFrame(fold_desc).to_csv(out / "fold_integrity.csv", index=False)
    if args.save_features:
        feat.to_csv(out / "progressive_prefix_features.csv.gz", index=False, compression="gzip")
    meta = {
        "stage": "10_failure_proximity_exploratory", "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "raw_input": str(path), "raw_sha256": file_sha256(path),
        "horizon_cycles": args.horizon, "minimum_prefix_cycles": args.min_prefix,
        "terminal_cycle_excluded": True, "threshold_fixed": args.threshold,
        "consecutive_cycle_confirmations": args.confirmations, "outer_folds": args.folds,
        "seed_bootstrap": args.seed, "bootstrap_resamples": args.bootstrap,
        "engine_count_all": int(original["id"].nunique()),
        "engine_count_eligible": int(feat["id"].nunique()),
        "engine_count_excluded": int(len(excluded)), "record_count_used": len(feat),
        "rpd_definition": "RPDgep = (c_plus*S_plus - c_minus*S_minus)/(c_plus*S_plus + c_minus*S_minus)",
        "feature_sets": feature_sets,
        "preprocessing": "fold-trained StandardScaler; per-fold class- and engine-balanced sample weights",
        "model": "LogisticRegression(C=1.0, solver=lbfgs, max_iter=1500)",
        "evaluation_status": "Retrospective grouped-CV; no independent blind holdout; no component fault labels",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
    }
    (out / "run_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    report = [f"# IJPHM Stage10 | {args.dataset} | Exploratory failure-proximity pilot", "",
              f"Observed engines: {meta['engine_count_all']} | eligible: {meta['engine_count_eligible']} | excluded: {meta['engine_count_excluded']}",
              f"Cycles used: {len(outputs)} | event = 1 <= RUL <= {args.horizon} | failure cycle removed", "",
              "## Mean engine-level metrics", "", summary.to_string(index=False), "", "## Paired engine AP differences", "",
              paired.to_string(index=False), "",
              "## Interpretation limits", "",
              "All training trajectories end at simulated failure; premature alarms are NOT operational false-alarm rates.",
              "No independent blind holdout is used; FD002/FD004 were previously examined in earlier stages.",
              "The task is binary proximity to end-of-life, NOT diagnosis of a failed component.",
              "Threshold 0.5 and two consecutive cycles are fixed, not chosen on validation engines.",
              "RPDgep is the original index, not a new formula or direct prognostic predictor.", ""]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    filehashes = [(file_sha256(p), p.name) for p in sorted(out.iterdir()) if p.is_file()]
    (out / "SHA256SUMS").write_text("".join(f"{d}  {name}\n" for d, name in filehashes))
    print("\nSTAGE10 COMPLETED:", out, "\n", paired.to_string(index=False))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", required=True)
    p.add_argument("--dataset", choices=DATASETS, default="FD001")
    p.add_argument("--output", required=True)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--min-prefix", type=int, default=8)
    p.add_argument("--confirmations", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--save-features", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
