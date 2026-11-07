#!/usr/bin/env python3
"""IJPHM Stage13: nested, training-only alarm-threshold calibration.

Retrospective cross-validation of first-alarm utility on C-MAPSS run-to-failure
training engines. RPDgep is imported unchanged from frozen Stage10.

No external holdout, no operating-regime normalization and no nonfailure control
trajectories: results cannot establish real-world false-alarm rates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

STAGE10_DIR = Path(__file__).resolve().parents[1] / '10_failure_proximity'
if not (STAGE10_DIR / 'stage10.py').exists():
    raise RuntimeError(f'Missing original Stage10: {STAGE10_DIR / "stage10.py"}')
sys.path.insert(0, str(STAGE10_DIR))
import stage10 as s10

MODELS = {
  'raw_age_ols': ['cycle', *s10.SETTINGS, *s10.SENSORS,
                  *[f'slope_{x}' for x in s10.SENSORS]],
  'raw_age_ols_rpd_gep': ['cycle', *s10.SETTINGS, *s10.SENSORS,
                         *[f'slope_{x}' for x in s10.SENSORS],
                         *[f'rpd_{x}' for x in s10.SENSORS]],
}
THRESHOLDS = np.unique(np.r_[np.linspace(0.50, 0.999, 101),
                               [0.9995, 0.9999, 0.99999, 0.999999, 1.000001]])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for buf in iter(lambda: f.read(1 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def fit_and_predict(df, train_ix, valid_ix, model):
    y = df['event_within_horizon'].to_numpy(dtype=int)
    ids = df['id'].to_numpy(dtype=int)
    weight = s10.balanced_engine_weights(ids[train_ix], y[train_ix])
    pipe = Pipeline([
        ('standardize', StandardScaler()),
        ('logistic', LogisticRegression(C=1.0, max_iter=1500, solver='lbfgs')),
    ])
    pipe.fit(df.iloc[train_ix][MODELS[model]], y[train_ix], logistic__sample_weight=weight)
    return pipe.predict_proba(df.iloc[valid_ix][MODELS[model]])[:, 1]


def grouped_first_alarm(df, probability_col, threshold, horizon, min_lead):
    """Per-engine first alarm after two consecutive *physical* cycle hits."""
    rows=[]
    for id_, g in df.groupby('id', sort=True):
        g=g.sort_values('cycle')
        cs=g['cycle'].to_numpy(dtype=int)
        ru=g['rul'].to_numpy(dtype=int)
        pp=g[probability_col].to_numpy(dtype=float)
        t=s10.stable_alarm_cycles(cs,pp,threshold,2)
        first=int(ru[np.searchsorted(cs,t[0])]) if t else None
        in_window=any(1 <= int(ru[np.searchsorted(cs,x)]) <= horizon for x in t)
        rows.append({
          'id':int(id_), 'first_alarm_rul': first,
          'premature':int(first is not None and first>horizon),
          'actionable':int(first is not None and min_lead<=first<=horizon),
          'late':int(first is not None and 1<=first<min_lead),
          'never':int(first is None),
          'ever_in_window':int(in_window),
        })
    return pd.DataFrame(rows)


def first_alarm_grid(df, probability_col, horizon, min_lead, thresholds=THRESHOLDS):
    """Optimized threshold table; equivalent to grouped_first_alarm for two hits.

    Pair minima at consecutive physical cycles make the first qualifying pair
    the first confirmed alarm; future observations never enter a prior alarm.
    """
    buffers=[]
    for id_,g in df.groupby('id',sort=True):
        gg=g.sort_values('cycle')
        c=gg['cycle'].to_numpy(dtype=int)
        p=gg[probability_col].to_numpy(dtype=float)
        r=gg['rul'].to_numpy(dtype=int)
        valid=(np.diff(c)==1)
        pair=np.minimum(p[:-1],p[1:])[valid]
        pair_rul=r[1:][valid]
        buffers.append((int(id_),pair,pair_rul))
    values=[]
    for th in thresholds:
        pre=act=late=never=0
        leads=[]
        for _,pair,rr in buffers:
            loc=np.flatnonzero(pair>=th)
            if len(loc)==0:
                never+=1; continue
            first=int(rr[loc[0]])
            if first>horizon: pre+=1
            elif first>=min_lead:
                act+=1; leads.append(first)
            else: late+=1
        n=len(buffers)
        values.append({'threshold':float(th), 'premature_fraction':pre/n,
                       'actionable_fraction':act/n,'late_fraction':late/n,
                       'never_fraction':never/n,
                       'median_actionable_lead':float(np.median(leads)) if leads else 0.0})
    return pd.DataFrame(values)


def choose_threshold(grid, max_premature):
    """Fixed lexicographic rule, all inputs from inner OOF train engines.

    Of thresholds meeting early-alarm cap, prefer most actionable first alarms
    (first alarm 5--30 cycles prior); tie -> higher median lead, lower early
    alarm rate, lower threshold. The >1 threshold encodes never alert.
    """
    eligible=grid[grid.premature_fraction <= max_premature + 1e-12].copy()
    if eligible.empty:
        raise AssertionError('No feasible threshold (the >1 fallback must qualify)')
    ranked=eligible.sort_values(['actionable_fraction','median_actionable_lead',
                                 'premature_fraction','threshold'],
                                ascending=[False,False,True,True],kind='mergesort')
    return ranked.iloc[0]


def bootstrap_binary_gain(base, extra, resamples, seed):
    a=np.asarray(extra,dtype=float)-np.asarray(base,dtype=float)
    rng=np.random.default_rng(seed)
    draws=rng.choice(a,(resamples,len(a)),replace=True).mean(axis=1)
    return float(a.mean()), float(np.quantile(draws,0.025)), float(np.quantile(draws,0.975))


def run(args):
    project=Path(args.project_root).expanduser().resolve()
    output=Path(args.output).expanduser().resolve()
    results=(project/'04_results').resolve()
    if not (project/'03_code'/'10_failure_proximity'/'stage10.py').is_file():
        raise ValueError('Stage10 source is missing')
    if output.exists() or not output.is_relative_to(results):
        raise ValueError('Output must be a NEW path inside 04_results')
    if args.outer_folds<2 or args.inner_folds<2 or args.horizon<5 or args.min_lead<1 or args.min_lead>args.horizon:
        raise ValueError('Invalid experiment parameters')
    if not (0<=args.max_premature<=1): raise ValueError('Invalid early alarm cap')
    raw=project/'02_raw_data'/s10.DATASETS[args.dataset]
    df=s10.read_dataset(raw)
    feat, excluded=s10.construct_prefix_features(df,args.horizon,args.min_prefix)
    engines=feat['id'].to_numpy(dtype=int)
    y=feat['event_within_horizon'].to_numpy(dtype=int)
    if len(np.unique(engines))<args.outer_folds*(args.inner_folds+1):
        raise ValueError('Insufficient engines for inner and outer grouped folds')
    outer=GroupKFold(n_splits=args.outer_folds)
    thresholds=[]
    pred=feat[['id','cycle','rul','event_within_horizon']].copy()
    pred['outer_fold']=0
    for model in MODELS: pred[model]=np.nan
    for i,(train,valid) in enumerate(outer.split(feat,y,engines),1):
        inner=GroupKFold(n_splits=args.inner_folds)
        train_df=feat.iloc[train].reset_index(drop=True)
        train_ids=train_df['id'].to_numpy(dtype=int)
        train_y=train_df['event_within_horizon'].to_numpy(dtype=int)
        if set(engines[train]) & set(engines[valid]): raise AssertionError('Group leakage')
        inner_oof=train_df[['id','cycle','rul']].copy()
        for model in MODELS: inner_oof[model]=np.nan
        for j,(tr,iv) in enumerate(inner.split(train_df,train_y,train_ids),1):
            if set(train_ids[tr]) & set(train_ids[iv]): raise AssertionError('Inner group leakage')
            for model in MODELS:
                inner_oof.loc[iv,model]=fit_and_predict(train_df,tr,iv,model)
            print(f'{args.dataset} outer {i}/{args.outer_folds} inner {j}/{args.inner_folds}',flush=True)
        if inner_oof[list(MODELS)].isna().any().any(): raise AssertionError('Incomplete inner OOF')
        for model in MODELS:
            grid=first_alarm_grid(inner_oof,model,args.horizon,args.min_lead)
            picked=choose_threshold(grid,args.max_premature)
            thresholds.append({'dataset':args.dataset,'outer_fold':i,'model':model,
                'threshold':float(picked['threshold']),
                'inner_premature_fraction':float(picked['premature_fraction']),
                'inner_actionable_fraction':float(picked['actionable_fraction']),
                'inner_late_fraction':float(picked['late_fraction']),
                'inner_never_fraction':float(picked['never_fraction']),
                'inner_train_engines':int(len(np.unique(train_ids))),
                'outer_valid_engines':int(len(np.unique(engines[valid])))})
            pred.loc[valid,model]=fit_and_predict(feat,train,valid,model)
        pred.loc[valid,'outer_fold']=i
        print(f'{args.dataset} outer {i}/{args.outer_folds} complete',flush=True)
    if pred[list(MODELS)].isna().any().any() or (pred['outer_fold']==0).any(): raise AssertionError('Outer OOF incomplete')
    chosen=pd.DataFrame(thresholds)
    alerts=[]
    for i in sorted(pred.outer_fold.unique()):
        chunk=pred[pred.outer_fold==i]
        for model in MODELS:
            th=float(chosen.loc[(chosen.outer_fold==i)&(chosen.model==model),'threshold'].iloc[0])
            a=grouped_first_alarm(chunk,model,th,args.horizon,args.min_lead)
            a['model']=model; a['outer_fold']=int(i);a['threshold']=th
            alerts.append(a)
    alarm=pd.concat(alerts,ignore_index=True)
    ss=[]
    for model in MODELS:
        a=alarm[alarm.model==model]
        prob=pred[model].to_numpy(dtype=float)
        y=pred['event_within_horizon'].to_numpy(dtype=int)
        ss.append({'dataset':args.dataset,'model':model,'engines':len(a),'records':len(pred),
                  'pooled_AP':average_precision_score(y,prob),'pooled_AUROC':roc_auc_score(y,prob),
                  'premature_first_fraction':a.premature.mean(),
                  'actionable_first_fraction':a.actionable.mean(),
                  'late_first_fraction':a.late.mean(),'never_first_fraction':a.never.mean(),
                  'ever_in_window_fraction':a.ever_in_window.mean(),
                  'median_first_alarm_rul':a.first_alarm_rul.median(),
                  'selected_thresholds':','.join(str(x) for x in chosen[chosen.model==model].threshold.to_list())})
    summary=pd.DataFrame(ss)
    base=alarm[alarm.model=='raw_age_ols'].set_index('id').sort_index()
    extra=alarm[alarm.model=='raw_age_ols_rpd_gep'].set_index('id').sort_index()
    if not base.index.equals(extra.index):raise AssertionError('Mismatched engine pairing')
    out=[]
    for field in ('premature','actionable','late','never','ever_in_window'):
        delta,low,high=bootstrap_binary_gain(base[field],extra[field],args.bootstrap,args.seed)
        out.append({'metric':field,'with_minus_without':delta,'95_low':low,'95_high':high,
                    'engines':len(base)})
    comparisons=pd.DataFrame(out)
    # All computations complete before output directory creation.
    output.mkdir(parents=False,exist_ok=False)
    pred.to_csv(output/'outer_oof_predictions.csv.gz',index=False,compression='gzip',float_format='%.17g')
    chosen.to_csv(output/'thresholds_selected_inner_oof.csv',index=False)
    alarm.to_csv(output/'first_alarms_outer_oof.csv',index=False)
    summary.to_csv(output/'model_summary.csv',index=False)
    comparisons.to_csv(output/'paired_alarm_differences.csv',index=False)
    excluded.to_csv(output/'excluded_engines.csv',index=False)
    manifest={
       'stage':'13_nested_alarm_thresholds',
       'dataset':args.dataset,
       'raw_sha256':sha(raw),
       'stage10_source_sha256':sha(STAGE10_DIR/'stage10.py'),
       'stage13_source_sha256':sha(Path(__file__)),
       'created_utc':datetime.now(timezone.utc).isoformat(),
       'outer_folds':args.outer_folds,'inner_folds':args.inner_folds,
       'horizon':args.horizon,'min_lead':args.min_lead,'min_prefix':args.min_prefix,
       'max_premature_training':args.max_premature,
       'confirmations':2,
       'decision_rule':'choose threshold solely from inner grouped OOF of outer-training engines; maximize actionable first alerts subject to training premature rate <= target',
       'threshold_grid':[float(x) for x in THRESHOLDS],
       'selection_predeclared_after_stage12':True,
       'original_RPD_formula_unchanged':True,
       'no_regime_normalization':True,
       'status':'retrospective nested CV; not independent prospective validation',
       'limitations':['all engines fail in training','no nonfailure controls',
           'thresholds chosen on inner OOF but protocol designed after exploratory Stage12',
           'no new blind external test','no component localization','model probabilities not calibrated'],
    }
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'REPORT.md').write_text('# Stage13 nested alarm threshold study - '+args.dataset+'\n\n'+
        '## Model summary\n\n'+summary.to_string(index=False)+'\n\n'+
        '## Paired differences (RPD minus baseline)\n\n'+comparisons.to_string(index=False)+
        '\n\nThresholds chosen on inner OOF of outer-training engines only.\n'
        'Results are retrospective. RPD formula unchanged.\n')
    with (output/'SHA256SUMS').open('w') as f:
        for p in sorted(output.iterdir()):
            if p.is_file() and p.name!='SHA256SUMS': f.write(f'{sha(p)}  {p.name}\n')
    print('STAGE13 COMPLETE:', output, '\n', summary.to_string(index=False), '\n', comparisons.to_string(index=False),flush=True)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',required=True)
    ap.add_argument('--dataset',choices=s10.DATASETS,required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--outer-folds',type=int,default=5)
    ap.add_argument('--inner-folds',type=int,default=4)
    ap.add_argument('--horizon',type=int,default=30)
    ap.add_argument('--min-lead',type=int,default=5)
    ap.add_argument('--min-prefix',type=int,default=8)
    ap.add_argument('--max-premature',type=float,default=0.20)
    ap.add_argument('--bootstrap',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=20261103)
    run(ap.parse_args())

if __name__=='__main__': main()
