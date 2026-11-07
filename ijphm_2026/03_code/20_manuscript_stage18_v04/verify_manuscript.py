#!/usr/bin/env python3
"""IJPHM_015: check archived tables and formula provenance; no refitting."""
import csv, json, hashlib
from pathlib import Path

CODEDIR=Path(__file__).resolve().parent
if (CODEDIR.parent/'paper'/'data').is_dir():
    PAPER=CODEDIR.parent/'paper'  # source/bundle layout
else:
    ART=CODEDIR.parents[1]  # installed under 01_articolo/03_code/<version>
    PAPER=ART/'01_paper'/'02_scientific_revision'/'20261104_IJPHM_manuscript_v04'
DATA=PAPER/'data'

def readcsv(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))
def close(x,y,atol=1e-11):
    if abs(float(x)-float(y))>atol:
        raise AssertionError(f'numerical mismatch: {x} vs {y} (tol={atol})')

expected={'FD001':(100,19831,3000),'FD002':(260,51679,7800),
          'FD003':(100,23920,3000),'FD004':(249,59257,7470)}
models={'B_raw_age_ols','B_plus_rpd','B_plus_gamma','B_plus_eta',
        'B_plus_gamma_eta','B_plus_gamma_eta_rpd','B_plus_gamma_eta_poly2',
        'B_plus_kendall_tau_b','B_plus_theilsen16'}
metrics=readcsv(DATA/'stage18_metrics_recomputed.csv')
assert len(metrics)==36, f'expected 36 OOF model rows, got {len(metrics)}'
lookup={}
for r in metrics:
    k=(r['dataset'],r['model'])
    assert k not in lookup, f'duplicate metric row {k}'
    assert r['model'] in models and r['dataset'] in expected
    close(r['engines'],expected[r['dataset']][0],atol=0)
    close(r['records'],expected[r['dataset']][1],atol=0)
    close(r['positives'],expected[r['dataset']][2],atol=0)
    for v in ('AP','AUROC','Brier'):
        assert 0<=float(r[v])<=1, (k,v,r[v])
    lookup[k]=r
assert len(lookup)==36
assert sum(v[0] for v in expected.values())==709
assert sum(v[1] for v in expected.values())==154687

original=readcsv(DATA/'dataset_summary.csv')
assert len(original)==4
for r in original:
    d=r['dataset']
    close(r['baseline_ap_reported'],lookup[d,'B_raw_age_ols']['AP'],atol=6e-13)
    close(r['rpd_ap_reported'],lookup[d,'B_plus_rpd']['AP'],atol=6e-13)

checks=readcsv(DATA/'stage18_contrasts_checked.csv')
assert len(checks)==24
for r in checks:
    lhs=lookup[r['dataset'],r['alternative']]['AP']
    rhs=lookup[r['dataset'],r['base']]['AP']
    close(float(lhs)-float(rhs),r['delta_AP'],atol=2e-14)
    assert int(r['bootstrap_repetitions'])==500
    assert float(r['CI_low'])<=float(r['CI_high'])

state=json.loads((DATA/'AUDIT_STATUS.json').read_text())
assert state['status']=='ALL_METRICS_VERIFIED'
assert state['pandas_float_precision']=='round_trip'
align=readcsv(DATA/'stage18_stage13_alignment.csv')
assert len(align)==4
for r in align:
    assert r['fold_match']=='PASS'
    assert int(r['matched'])==expected[r['dataset']][1]
    assert float(r['max_abs_delta_base'])<5.1e-13
    assert float(r['max_abs_delta_rpd'])<5.1e-13
integrity=readcsv(DATA/'stage18_integrity_checked.csv')
assert len(integrity)==4
for r in integrity:
    assert r['integrity']=='PASS' and r['folds']=='PASS'

formsha=hashlib.sha256((PAPER/'sections'/'02_definitions.tex').read_bytes()).hexdigest()
EXPECTED_FORMULA_SHA='306ecd50208dc4e0edb975c270666faa294da8cc5b192e78adeaf833e09e0977'
assert formsha==EXPECTED_FORMULA_SHA, 'original six-index formula text changed'

# Presence of 9x4 printed values, rounded to 5 decimal places, in the LaTeX table.
results=(PAPER/'sections'/'04_results.tex').read_text()
for name in models:
    for d in expected:
        assert f'{float(lookup[d,name]["AP"]):.5f}' in results, f'missing rounded AP {d} {name}'

for d in expected:
    delta=float(lookup[d,'B_plus_rpd']['AP'])-float(lookup[d,'B_raw_age_ols']['AP'])
    gamma_delta=float(lookup[d,'B_plus_gamma_eta']['AP'])-float(lookup[d,'B_plus_rpd']['AP'])
    print(f'{d}: engines={expected[d][0]:3d}, records={expected[d][1]:5d}, '
          f'RPD-baseline={delta:+.6f}, gamma_eta-RPD={gamma_delta:+.6f}')
print('IJPHM_015 VALIDATION PASSED: 36 model rows; 24 contrasts; stage13 alignment; formula SHA')
print('Interpretation: verified fixed OOF predictions; neither refit nor independent cohort.')
