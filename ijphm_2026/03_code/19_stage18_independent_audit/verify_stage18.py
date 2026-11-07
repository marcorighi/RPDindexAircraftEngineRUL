#!/usr/bin/env python3
"""Read-only, reproducible verification of the frozen IJPHM Stage18 results.
Input: six preexisting Stage18 ZIP/.sha256 files, optional four Stage13 archives.
Output: new directory containing independently recomputed metrics and checks.
Do not modify any original research result.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import zipfile, hashlib, json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

STAMP = '20261104T062939Z_1336864'
DATASETS = ('FD001','FD002','FD003','FD004')
STAGE13 = {
 'FD001':'13_nested_alarm_FD001_20261103T215658Z_1079390.zip',
 'FD002':'13_nested_alarm_FD002_20261103T221003Z_1085863.zip',
 'FD003':'13_nested_alarm_FD003_20261103T221100Z_1086099.zip',
 'FD004':'13_nested_alarm_FD004_20261103T221118Z_1086181.zip',
}
MODELS = (
 'B_raw_age_ols','B_plus_rpd','B_plus_gamma','B_plus_eta',
 'B_plus_gamma_eta','B_plus_gamma_eta_rpd','B_plus_gamma_eta_poly2',
 'B_plus_kendall_tau_b','B_plus_theilsen16',
)

def hashfile(path: Path) -> str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):
   h.update(chunk)
 return h.hexdigest()

def verify_zip(p:Path, expected_prefix: str):
 sha=p.with_suffix(p.suffix+'.sha256')
 if not p.is_file() or not sha.is_file():
  raise FileNotFoundError(f'Missing {p} and/or .zip.sha256 (do not recompute)')
 words=sha.read_text().split()
 if len(words)<2 or words[1]!=p.name or words[0]!=hashfile(p):
  raise AssertionError(f'External SHA mismatch: {p.name}')
 z=zipfile.ZipFile(p)
 if z.testzip() is not None: raise AssertionError(f'Invalid ZIP: {p.name}')
 prefix=expected_prefix+'/'
 manifest=z.read(prefix+'SHA256SUMS').decode()
 for line in manifest.splitlines():
  digest, name=line.split(maxsplit=1)
  data=z.read(prefix+name.lstrip('*'))
  if hashlib.sha256(data).hexdigest()!=digest:
   raise AssertionError(f'Internal SHA mismatch: {name}')
 return z

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--zip-dir',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args()
 zipdir=args.zip_dir.expanduser().resolve()
 out=args.output.expanduser().resolve()
 if out.exists(): raise FileExistsError(f'Refusing overwrite: {out}')
 if not zipdir.is_dir():raise NotADirectoryError(zipdir)
 mets=[]; contrasts=[]; controls=[]; stage13=[]
 for ds in DATASETS:
  name=f'18_rpd_components_{ds}_{STAMP}'
  p=zipdir/(name+'.zip')
  with verify_zip(p,name) as z:
   base=name+'/'
   summary=pd.read_csv(z.open(base+'model_summary.csv'))
   boot=pd.read_csv(z.open(base+'paired_bootstrap.csv'))
   # MANDATORY: the default pandas parser loses binary64 distinctions near 1.
   df=pd.read_csv(z.open(base+'outer_oof_predictions.csv.gz'), compression='gzip',
                  float_precision='round_trip')
   manifest=json.loads(z.read(base+'run_manifest.json'))
   if manifest['dataset']!=ds:raise AssertionError(f'{ds}: manifest dataset')
   if list(df.columns)!=['id','cycle','rul','event_within_horizon','outer_fold',*MODELS]:
    raise AssertionError(f'{ds}: unexpected OOF columns')
   if df.duplicated(['id','cycle']).any():raise AssertionError(f'{ds}: duplicates')
   if not df.cycle.ge(8).all() or not df.rul.ge(1).all():
    raise AssertionError(f'{ds}: unsupported cycles/RUL')
   if not (df.event_within_horizon == df.rul.le(30)).all():
    raise AssertionError(f'{ds}: inconsistent labels')
   byengine=df.groupby('id')
   if not byengine.outer_fold.nunique().eq(1).all():
    raise AssertionError(f'{ds}: engine in multiple folds')
   if not byengine.event_within_horizon.sum().eq(30).all():
    raise AssertionError(f'{ds}: not 30 positives per engine')
   if not byengine.cycle.min().eq(8).all() or not byengine.rul.min().eq(1).all():
    raise AssertionError(f'{ds}: truncated engine')
   if not byengine.cycle.diff().dropna().eq(1).all() or not byengine.rul.diff().dropna().eq(-1).all():
    raise AssertionError(f'{ds}: noncontiguous engine cycles')
   if set(df.outer_fold.unique())!={1,2,3,4,5}:
    raise AssertionError(f'{ds}: not five folds')
   y=df.event_within_horizon.to_numpy(dtype=int)
   for model in MODELS:
    pp=df[model].to_numpy(dtype=float)
    if not np.isfinite(pp).all() or not ((pp>=0)&(pp<=1)).all():
     raise AssertionError(f'{ds}/{model}: non-finite or outside [0,1]')
    r=summary.loc[summary.model==model]
    if len(r)!=1:raise AssertionError(f'{ds}/{model}: missing summary')
    r=r.iloc[0]
    computed=(average_precision_score(y,pp),roc_auc_score(y,pp),brier_score_loss(y,pp))
    if max(abs(computed[i]-float(r[field])) for i,field in enumerate(('pooled_AP','pooled_AUROC','pooled_Brier')))>1e-10:
     raise AssertionError(f'{ds}/{model}: recomputed metrics mismatch')
    mets.append(dict(dataset=ds,model=model,engines=df.id.nunique(),records=len(df),
                     positives=int(y.sum()),AP=computed[0],AUROC=computed[1],Brier=computed[2]))
   if len(boot)!=6: raise AssertionError(f'{ds}: unexpected contrast count')
   for row in boot.itertuples():
    a=summary.loc[summary.model==row.alternative,'pooled_AP'].iloc[0]
    b=summary.loc[summary.model==row.base,'pooled_AP'].iloc[0]
    if abs((a-b)-row.delta_pooled_AP)>1e-12:raise AssertionError(f'{ds}: contrast mismatch')
    contrasts.append(dict(dataset=ds,contrast=row.contrast,base=row.base,alternative=row.alternative,
                          delta_AP=row.delta_pooled_AP,CI_low=row.bootstrap_low,
                          CI_high=row.bootstrap_high,bootstrap_repetitions=row.bootstrap_repetitions))
   controls.append(dict(dataset=ds,engines=df.id.nunique(),records=len(df),positive_records=int(y.sum()),
                   external_SHA=hashfile(p),raw_SHA=manifest['raw_sha256'],
                   source_SHA=manifest['source_sha256'],integrity='PASS',folds='PASS'))
   q=zipdir/STAGE13[ds]
   if q.is_file():
    with zipfile.ZipFile(q) as z2:
     nn=[x for x in z2.namelist() if x.endswith('outer_oof_predictions.csv.gz')]
     if len(nn)!=1:raise AssertionError(f'{ds}: Stage13 OOF missing')
     older=pd.read_csv(z2.open(nn[0]),compression='gzip',float_precision='round_trip')
    merged=df[['id','cycle','rul','outer_fold','B_raw_age_ols','B_plus_rpd']].merge(
      older[['id','cycle','rul','outer_fold','raw_age_ols','raw_age_ols_rpd_gep']],
      on=['id','cycle','rul','outer_fold'],indicator=True,how='outer')
    if not (merged._merge=='both').all() or len(merged)!=len(df):
     raise AssertionError(f'{ds}: Stage13 OOF alignment mismatch')
    stage13.append(dict(dataset=ds,matched=len(merged),
       max_abs_delta_base=np.max(np.abs(merged.B_raw_age_ols-merged.raw_age_ols)),
       max_abs_delta_rpd=np.max(np.abs(merged.B_plus_rpd-merged.raw_age_ols_rpd_gep)),
       fold_match='PASS'))
  print(f'{ds}: {len(df):,} records, {df.id.nunique()} engines, metrics validated',flush=True)
 out.mkdir(parents=True,exist_ok=False)
 pd.DataFrame(mets).to_csv(out/'stage18_metrics_recomputed.csv',index=False,float_format='%.17g')
 pd.DataFrame(contrasts).to_csv(out/'stage18_contrasts_checked.csv',index=False,float_format='%.17g')
 pd.DataFrame(controls).to_csv(out/'stage18_integrity_checked.csv',index=False)
 pd.DataFrame(stage13).to_csv(out/'stage18_stage13_alignment.csv',index=False,float_format='%.17g')
 note={'status':'ALL_METRICS_VERIFIED','method':'OOF metrics re-evaluated from fixed archived predictions',
       'pandas_float_precision':'round_trip', 'not_recomputed':'fitting, feature extraction, bootstrap replicates',
       'stage13_check':'alignment only, not a new independent sample',
       'inputs':[f'18_rpd_components_{ds}_{STAMP}.zip' for ds in DATASETS]}
 (out/'AUDIT_STATUS.json').write_text(json.dumps(note,indent=2)+'\n')
 print('ALL STAGE18 ARCHIVE CHECKS PASSED',flush=True)
 print('Audit output:',out,flush=True)
if __name__=='__main__':main()
