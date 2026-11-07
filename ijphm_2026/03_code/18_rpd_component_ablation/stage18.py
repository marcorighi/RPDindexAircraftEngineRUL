#!/usr/bin/env python3
"""IJPHM Stage18: fixed-feature ablation, global RPDgep vs its components.

Retrospective exploratory GroupKFold on historical C-MAPSS training engines.
Frozen Stage10's raw data parser, age, OLS prefix and RPD formula are imported unchanged.
Not prospective validation; no threshold tuning or alarm estimands in this stage.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn import __version__ as SK_VERSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

S10DIR = Path(__file__).resolve().parents[1] / '10_failure_proximity'
if not (S10DIR / 'stage10.py').is_file():
    raise RuntimeError('Stage10 required as an unchanged sibling of Stage18')
sys.path.insert(0, str(S10DIR))
import stage10 as s10

SEED = 20261104
WINDOW = 16
BASE = ['cycle', *s10.SETTINGS, *s10.SENSORS,
        *[f'slope_{z}' for z in s10.SENSORS]]

def fcols(prefix: str) -> list[str]:
    return [f'{prefix}_{z}' for z in s10.SENSORS]

FEATURES = {
    'B_raw_age_ols': BASE,
    'B_plus_rpd': BASE + fcols('rpd'),
    'B_plus_gamma': BASE + fcols('gamma'),
    'B_plus_eta': BASE + fcols('eta'),
    'B_plus_gamma_eta': BASE + fcols('gamma') + fcols('eta'),
    'B_plus_gamma_eta_rpd': BASE + fcols('gamma') + fcols('eta') + fcols('rpd'),
    'B_plus_gamma_eta_poly2': BASE + sum((fcols(z) for z in
        ('gamma', 'eta', 'gamma_eta', 'gamma2', 'eta2')), []),
    'B_plus_kendall_tau_b': BASE + fcols('tau_b'),
    'B_plus_theilsen16': BASE + fcols('ts16'),
}
CONTRASTS = [
    ('B_raw_age_ols', 'B_plus_rpd', 'RPD vs baseline'),
    ('B_plus_rpd', 'B_plus_gamma_eta', 'components vs RPD'),
    ('B_plus_gamma_eta', 'B_plus_gamma_eta_rpd', 'RPD after components'),
    ('B_plus_rpd', 'B_plus_gamma_eta_poly2', 'polynomial vs RPD'),
    ('B_plus_kendall_tau_b', 'B_plus_rpd', 'RPD vs Kendall tau-b'),
    ('B_plus_theilsen16', 'B_plus_rpd', 'RPD vs Theil-Sen(16)'),
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def component_prefix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts and sums of *past-to-present* global signed pairs in O(n log n).

    Gamma excludes sensor-tied pairs. Kendall tau-b adjusts for ties, since
    cycle/time itself has no ties. Full-array coordinates are only dictionary
    positions; future measurement values never contribute to prefix aggregates.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('Finite 1D trajectory required')
    coords = sorted(set(a.tolist()))
    counts = s10.Fenwick(len(coords))
    sums = s10.Fenwick(len(coords))
    gamma = np.zeros(len(a), dtype=float)
    eta = np.zeros(len(a), dtype=float)
    tau = np.zeros(len(a), dtype=float)
    pc = nc = 0
    ps = ns = 0.0
    total = 0.0
    for k, val in enumerate(a):
        lo = bisect_left(coords, val)
        hi = bisect_right(coords, val)
        less = int(round(counts.prefix(lo)))
        at_most = int(round(counts.prefix(hi)))
        lower_sum = sums.prefix(lo)
        greater = k - at_most
        greater_sum = total - sums.prefix(hi)
        pc += less
        nc += greater
        ps += max(0.0, val * less - lower_sum)
        ns += max(0.0, greater_sum - val * greater)
        if pc + nc > 0:
            gamma[k] = (pc - nc) / (pc + nc)
            tau[k] = (pc - nc) / np.sqrt((k * (k + 1) / 2) * (pc + nc))
        if ps + ns > 0:
            eta[k] = (ps - ns) / (ps + ns)
        counts.add(lo + 1, 1)
        sums.add(lo + 1, float(val))
        total += float(val)
    return gamma, eta, tau


def theilsen_trailing(x: np.ndarray, w: int = WINDOW) -> np.ndarray:
    """EXACT median of all pair slopes within last w observed cycles.

    This is a fixed-window Theil--Sen comparator, NOT full-history Theil--Sen.
    All cycle spacings are 1 (validated in Stage10 parser).
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all() or w < 2:
        raise ValueError('Finite 1D trajectory and w>=2 required')
    out = np.zeros(len(a), dtype=float)
    for m in range(2, min(len(a), w - 1) + 1):
        i, j = np.triu_indices(m, 1)
        out[m - 1] = np.median((a[j] - a[i]) / (j - i))
    if len(a) >= w:
        windows = sliding_window_view(a, w)
        i, j = np.triu_indices(w, 1)
        slopes = (windows[:, j] - windows[:, i]) / (j - i)[None, :]
        out[w - 1:] = np.median(slopes, axis=1)
    return out


def features_for_dataset(raw: pd.DataFrame, horizon: int = 30, min_prefix: int = 8):
    base, excluded = s10.construct_prefix_features(raw, horizon, min_prefix)
    parts = []
    for eid, g in raw.groupby('id', sort=True):
        cycles = g.cycle.to_numpy(dtype=int)
        names = {'id': np.repeat(int(eid), len(g)), 'cycle': cycles}
        for sensor in s10.SENSORS:
            vals = g[sensor].to_numpy(dtype=float)
            ga, et, tb = component_prefix(vals)
            rpd = s10.rpdgep_prefix_fast(vals)
            den = 1 + ga * et
            reconstructed = np.zeros(len(vals), dtype=float)
            ok = np.abs(den) > 1e-13
            reconstructed[ok] = (ga[ok] + et[ok]) / den[ok]
            if not np.allclose(reconstructed, rpd, atol=1e-9, rtol=1e-9):
                raise AssertionError(f'Frozen RPD formula inconsistent with decomposition for {eid}/{sensor}')
            names[f'gamma_{sensor}'] = ga
            names[f'eta_{sensor}'] = et
            names[f'tau_b_{sensor}'] = tb
            names[f'gamma_eta_{sensor}'] = ga * et
            names[f'gamma2_{sensor}'] = ga * ga
            names[f'eta2_{sensor}'] = et * et
            names[f'ts16_{sensor}'] = theilsen_trailing(vals)
        parts.append(pd.DataFrame(names))
    extra = pd.concat(parts, ignore_index=True)
    merged = base.merge(extra, on=['id', 'cycle'], how='left', validate='one_to_one', sort=False)
    if merged.isna().any().any():
        raise AssertionError('Feature join must contain no missing values')
    if not np.isfinite(merged.select_dtypes(include='number').to_numpy(dtype=float)).all():
        raise AssertionError('All features must be finite')
    return merged, excluded


def train_oof(feat: pd.DataFrame, outer_folds: int = 5) -> pd.DataFrame:
    y = feat.event_within_horizon.to_numpy(dtype=int)
    ids = feat.id.to_numpy(dtype=int)
    splitter = GroupKFold(n_splits=outer_folds)
    out = feat[['id','cycle','rul','event_within_horizon']].copy()
    out['outer_fold'] = 0
    for name in FEATURES:
        out[name] = np.nan
    for fold, (itr, ival) in enumerate(splitter.split(feat, y, ids), 1):
        if set(ids[itr]) & set(ids[ival]):
            raise AssertionError('Engine leakage')
        wt = s10.balanced_engine_weights(ids[itr], y[itr])
        for name, cols in FEATURES.items():
            pipe = Pipeline([
                ('standardize', StandardScaler()),
                ('logistic', LogisticRegression(C=1.0,max_iter=1500,solver='lbfgs')),
            ])
            pipe.fit(feat.iloc[itr][cols],y[itr],logistic__sample_weight=wt)
            out.loc[ival,name] = pipe.predict_proba(feat.iloc[ival][cols])[:, 1]
        out.loc[ival,'outer_fold'] = fold
        print(f'outer {fold}/{outer_folds} complete (9 feature models)',flush=True)
    if out[list(FEATURES)].isna().any().any() or (out.outer_fold == 0).any():
        raise AssertionError('Incomplete out-of-fold predictions')
    return out


def summarize(pred: pd.DataFrame, dataset: str):
    y = pred.event_within_horizon.to_numpy(dtype=int)
    return pd.DataFrame([{'dataset':dataset,'model':name,
                          'n_engines':pred.id.nunique(),'n_records':len(pred),
                          'pooled_AP':average_precision_score(y,pred[name]),
                          'pooled_AUROC':roc_auc_score(y,pred[name]),
                          'pooled_Brier':brier_score_loss(y,pred[name])}
                         for name in FEATURES])


def bootstrap_contrasts(pred: pd.DataFrame, dataset: str, n: int = 500, seed: int = SEED) -> pd.DataFrame:
    """Engine bootstrap for pooled AP with pre-sorted, tie-aware scores.

    Whole-engine resampling is EXACTLY equivalent to integer sample weights
    per original engine, so scores need sorting only once per model.  This
    avoids repeated data copying and re-sorting inside each bootstrap draw.
    No model refitting or training uncertainty is represented in the interval.
    """
    if n < 10:
        raise ValueError('Require at least 10 bootstrap replicates')
    engine, engine_idx = np.unique(pred.id.to_numpy(dtype=int), return_inverse=True)
    y = pred.event_within_horizon.to_numpy(dtype=np.float64)
    if len(engine) < 2:
        raise ValueError('Need at least two independent engines')
    rank_data = {}
    for name in FEATURES:
        score = pred[name].to_numpy(dtype=np.float64)
        order = np.argsort(-score,kind='mergesort')
        ordered_score=score[order]
        # Count positive/negative weights at the end of each tied-score group.
        ends=np.r_[np.flatnonzero(np.diff(ordered_score)!=0),len(order)-1]
        rank_data[name]=(engine_idx[order],y[order],ends)

    def ap_from_engine_counts(name, counts):
        ix, positives, ends = rank_data[name]
        ww = counts[ix]
        total_positive = np.dot(ww, positives)
        if total_positive <= 0:
            raise ValueError('A bootstrap sample contains no positives')
        tp = np.cumsum(ww * positives)[ends]
        tested = np.cumsum(ww)[ends]
        precision = np.divide(tp,tested,out=np.zeros_like(tp),where=tested>0)
        gained = np.diff(np.r_[0.,tp])
        return float(np.dot(gained,precision)/total_positive)

    all_one=np.ones(len(engine),dtype=float)
    observed={name:ap_from_engine_counts(name,all_one) for name in FEATURES}
    for name in FEATURES:
        reference=average_precision_score(y,pred[name].to_numpy(dtype=float))
        if abs(observed[name]-reference)>1e-12:
            raise AssertionError(f'Optimized bootstrap AP disagrees with sklearn: {name}')
    draws={ (a,b):np.empty(n,dtype=float) for a,b,_ in CONTRASTS }
    rng=np.random.default_rng(seed)
    for j in range(n):
        multiplicity=np.bincount(rng.integers(0,len(engine),len(engine)),minlength=len(engine))
        aps={name:ap_from_engine_counts(name,multiplicity) for name in FEATURES}
        for a,b,_ in CONTRASTS:
            draws[(a,b)][j]=aps[b]-aps[a]
        if (j+1)%100==0:
            print(f'bootstrap {j+1}/{n}',flush=True)
    rows=[]
    for a,b,label in CONTRASTS:
        vals=draws[(a,b)]
        rows.append({'dataset':dataset,'contrast':label,'base':a,'alternative':b,
                     'delta_pooled_AP':observed[b]-observed[a],
                     'bootstrap_low':float(np.quantile(vals,.025)),
                     'bootstrap_high':float(np.quantile(vals,.975)),
                     'bootstrap_repetitions':n,
                     'bootstrap_unit':'engine (fixed OOF predictions)'})
    return pd.DataFrame(rows)


def run(args):
    project = Path(args.project_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() or not output.is_relative_to((project/'04_results').resolve()):
        raise ValueError('Output must be a new directory in 04_results')
    if not 2 <= args.outer_folds <= 10:
        raise ValueError('Invalid number of outer folds')
    if args.window != WINDOW:
        raise ValueError('Stage18 preregistered comparison has fixed Theil-Sen window 16')
    raw_path = project/'02_raw_data'/s10.DATASETS[args.dataset]
    raw = s10.read_dataset(raw_path)
    feat,excluded=features_for_dataset(raw,args.horizon,args.min_prefix)
    print(f'{args.dataset}: {feat.id.nunique()} engines, {len(feat)} records',flush=True)
    if feat.id.nunique() < args.outer_folds:
        raise ValueError('Insufficient independent engines')
    oof = train_oof(feat,args.outer_folds)
    summary = summarize(oof,args.dataset)
    contrasts = bootstrap_contrasts(oof,args.dataset,args.bootstrap,args.seed)
    output.mkdir(parents=False,exist_ok=False)
    oof.to_csv(output/'outer_oof_predictions.csv.gz',index=False,compression='gzip',float_format='%.17g')
    summary.to_csv(output/'model_summary.csv',index=False,float_format='%.17g')
    contrasts.to_csv(output/'paired_bootstrap.csv',index=False,float_format='%.17g')
    excluded.to_csv(output/'excluded_engines.csv',index=False)
    reread = pd.read_csv(output/'outer_oof_predictions.csv.gz',float_precision='round_trip')
    check=summarize(reread,args.dataset)
    for name in FEATURES:
        orig=float(summary.loc[summary.model==name,'pooled_AP'].iloc[0])
        replay=float(check.loc[check.model==name,'pooled_AP'].iloc[0])
        if abs(orig-replay)>1e-12:
            raise AssertionError(f'AP cannot be recovered from archived predictions: {name}: {orig}, {replay}')
    manifest = {'stage':'18_rpd_component_ablation','dataset':args.dataset,
       'created_utc':datetime.now(timezone.utc).isoformat(),
       'source_sha256':sha(Path(__file__)),
       'frozen_stage10_sha256':sha(S10DIR/'stage10.py'),
       'raw_sha256':sha(raw_path),'python':sys.version,'sklearn':SK_VERSION,
       'params':{'outer_folds':args.outer_folds,'horizon':args.horizon,
                 'min_prefix':args.min_prefix,'bootstrap':args.bootstrap,
                 'seed':args.seed,'theilsen_trailing_window':WINDOW},
       'model_names':list(FEATURES),'recomputed_AP_from_archived_17digit_probabilities':True,
       'status':'exploratory retrospective re-analysis; previous C-MAPSS data inspected',
       'limits':['no new independent holdout','no operating-regime normalization',
                 'bootstrap conditions on fixed OOF fitted models',
                 'windowed Theil-Sen is not full-prefix Theil-Sen',
                 'no alarm thresholds or real nonfailure controls']}
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'REPORT.md').write_text('# Stage18 '+args.dataset+' RPD component ablation\n\n'+
       '## Out-of-fold global AP (NOT per-engine AP)\n\n'+summary.to_string(index=False)+
       '\n\n## Paired engine bootstrap (fixed fitted models)\n\n'+contrasts.to_string(index=False)+
       '\n\nNo real-world false alarm rates or independent holdout are claimed.\n')
    with (output/'SHA256SUMS').open('w') as f:
        for p in sorted(output.iterdir()):
            if p.name!='SHA256SUMS': f.write(f'{sha(p)}  {p.name}\n')
    print(summary.to_string(index=False),flush=True)
    print(contrasts[['contrast','delta_pooled_AP','bootstrap_low','bootstrap_high']].to_string(index=False),flush=True)
    print('STAGE18 COMPLETED:',output,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',required=True)
    ap.add_argument('--dataset',required=True,choices=sorted(s10.DATASETS))
    ap.add_argument('--output',required=True)
    ap.add_argument('--outer-folds',type=int,default=5)
    ap.add_argument('--horizon',type=int,default=30)
    ap.add_argument('--min-prefix',type=int,default=8)
    ap.add_argument('--bootstrap',type=int,default=500)
    ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--window',type=int,default=WINDOW)
    run(ap.parse_args())

if __name__=='__main__':main()
