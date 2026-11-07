#!/usr/bin/env python3
"""Consolidate summaries only, never refit or recompute pooled cross-dataset AP."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(inputs,out):
    out=Path(out)
    files=[]
    models=[]
    gains=[]
    for folder in map(Path,inputs):
        assert folder.is_dir()
        sums=(folder/'SHA256SUMS').read_text().splitlines()
        for line in sums:
            digest,filename=line.split(None,1)
            p=folder/filename.strip()
            if sha(p)!=digest:raise ValueError(f'Checksum failure: {p}')
        manifest=json.loads((folder/'run_manifest.json').read_text())
        files.append({'dataset':manifest['dataset'], 'result_directory':folder.name,
                      'raw_sha256':manifest['raw_sha256'], 'source_sha256':manifest['source_sha256']})
        models.append(pd.read_csv(folder/'model_summary.csv'))
        gains.append(pd.read_csv(folder/'paired_bootstrap.csv'))
    datasets=[f['dataset'] for f in files]
    if len(datasets)!=len(set(datasets)):
        raise ValueError('Repeated dataset in aggregate')
    pd.concat(models,ignore_index=True).to_csv(out/'all_models.csv',index=False,float_format='%.17g')
    pd.concat(gains,ignore_index=True).to_csv(out/'all_contrasts.csv',index=False,float_format='%.17g')
    pd.DataFrame(files).to_csv(out/'source_runs.csv',index=False)
    (out/'README.md').write_text('# Stage18: existing-fold exploratory ablation\n'
        'Per-dataset out-of-fold metrics, NOT independently validated test results.\n'
        'Do NOT pool AP values by averaging across datasets.\n'
        'Send full dataset ZIPs if reproduction of predictions is needed.\n')
    with (out/'SHA256SUMS').open('w') as f:
        for p in sorted(out.iterdir()):
            if p.name!='SHA256SUMS':f.write(f'{sha(p)}  {p.name}\n')
    print(pd.concat(models,ignore_index=True)[['dataset','model','pooled_AP']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.inputs,a.out)
