#!/usr/bin/env python3
"""Recheck published Stage13 table aggregates from the frozen IJPHM_010 audit CSV.

This is a consistency check against archived summary rows; it does NOT
recalculate models or undo the original probability serialization issue.
"""
from pathlib import Path
import csv
import argparse

p = argparse.ArgumentParser()
p.add_argument("--data-dir", type=Path, required=True)
args=p.parse_args()
base = args.data_dir.expanduser().resolve()
with (base / 'dataset_summary.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
assert {r['dataset'] for r in rows} == {'FD001','FD002','FD003','FD004'}
assert len(rows)==4
nums = lambda col: sum(int(r[col]) for r in rows)
assert nums('engines') == 709
assert nums('observations') == 154687
assert nums('positive_observations') == 21270
for model in ['baseline','rpd']:
    totals = [nums(f'{x}_{model}') for x in ['premature','actionable','late','never']]
    assert sum(totals) == 709, (model,totals)
assert [nums('premature_baseline'),nums('premature_rpd')]==[58,27]
assert [nums('actionable_baseline'),nums('actionable_rpd')]==[640,668]
assert [nums('never_baseline'),nums('never_rpd')]==[1,5]
assert [nums('ever_in_window_baseline'),nums('ever_in_window_rpd')]==[708,702]
for r in rows:
    delta = float(r['rpd_ap_reported']) - float(r['baseline_ap_reported'])
    assert abs(delta - float(r['delta_ap_reported'])) < 1e-12
    assert delta > 0
    print(f"{r['dataset']} engines={r['engines']} AP_gain={delta:.6f} ",
          f"premature {r['premature_baseline']}->{r['premature_rpd']}")
print('CHECKS PASSED (archived summary, not independent re-fitting)')
