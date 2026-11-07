#!/usr/bin/env python3
"""Check the 17 frozen source hashes against the distributable candidate."""
import csv,hashlib,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/'metadata/source_lineage.csv').open(newline='')))
assert len(rows)==17
changed=0
for row in rows:
    p=root/'03_code'/row['path']
    b=p.read_bytes()
    got=hashlib.sha256(b).hexdigest()
    assert got==row['candidate_sha256'],row['path']
    if row['change']=='unchanged':
        assert got==row['original_sha256'],row['path']
    else:
        assert row['path']=='13_nested_alarm/stage13.py'
        assert row['change']=='csv_float_format_12g_to_17g'
        old=b.replace(b"float_format='%.17g'",b"float_format='%.12g'")
        assert old!=b
        assert hashlib.sha256(old).hexdigest()==row['original_sha256']
        changed+=1
assert changed==1,changed
print('SOURCE LINEAGE: 16 unchanged; 1 Stage13 formatting-only change; PASS')
