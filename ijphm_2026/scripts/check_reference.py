#!/usr/bin/env python3
"""Guard against gross regression; not an independent blind validation.

Exact numeric matches depend on frozen library versions and floating-point runtime.
These approximate checks do not assert bitwise equality across machines.
"""
from pathlib import Path
import csv,json,sys
REF13={
 'FD001':{'raw_age_ols':.983700,'raw_age_ols_rpd_gep':.990504},
 'FD002':{'raw_age_ols':.919485,'raw_age_ols_rpd_gep':.973646},
 'FD003':{'raw_age_ols':.962059,'raw_age_ols_rpd_gep':.972324},
 'FD004':{'raw_age_ols':.882149,'raw_age_ols_rpd_gep':.924120},
}
REF18={
 'FD001':{'B_raw_age_ols':.983700,'B_plus_rpd':.990504,'B_plus_gamma_eta':.989819},
 'FD002':{'B_raw_age_ols':.919485,'B_plus_rpd':.973646,'B_plus_gamma_eta':.975519},
 'FD003':{'B_raw_age_ols':.962059,'B_plus_rpd':.972324,'B_plus_gamma_eta':.980118},
 'FD004':{'B_raw_age_ols':.882149,'B_plus_rpd':.924120,'B_plus_gamma_eta':.928913},
}
if len(sys.argv)!=4:sys.exit('Usage: check_reference.py stage13|stage18 FD001|FD002|FD003|FD004 RESULT_DIR')
stage,ds,folder=sys.argv[1:]
if stage not in ('stage13','stage18'):sys.exit('Invalid stage')
ref=(REF13 if stage=='stage13' else REF18)[ds]
path=Path(folder)/'model_summary.csv'
with path.open(newline='') as f:rows=list(csv.DictReader(f))
metrics={r['model']:float(r['pooled_AP']) for r in rows}
report=[]
for model,expected in ref.items():
 observed=metrics.get(model)
 if observed is None:raise ValueError(f'Missing {model}')
 report.append({'model':model,'original_AP':expected,'reproduced_AP':observed,'abs_difference':abs(observed-expected)})
status='WITHIN_EXPLORATORY_TOLERANCE' if all(r['abs_difference']<=.001 for r in report) else 'REVIEW_REQUIRED'
(Path(folder)/'replication_reference_check.json').write_text(json.dumps({'status':status,'comparison_scope':'AP only; original published rounded values; not provenance proof','entries':report},indent=2)+'\n')
print(ds,stage,status)
if status!='WITHIN_EXPLORATORY_TOLERANCE':sys.exit(7)
