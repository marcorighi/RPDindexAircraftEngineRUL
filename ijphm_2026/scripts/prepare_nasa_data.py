#!/usr/bin/env python3
"""Convert original NASA double-zip training members to the exact Stage10 CSV schema.
Do not distribute generated CSVs without confirming the original provider's terms.
"""
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
from datetime import datetime, timezone
import hashlib, io, csv, json, os, sys

EXPECTED_SHA='c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2'
EXPECTED={'FD001':(20631,100),'FD002':(53759,260),'FD003':(24720,100),'FD004':(61249,249)}
NAMES={'FD001':'PM_train.csv',**{k:f'PM_train_{k}.csv' for k in ('FD002','FD003','FD004')}}
HEADER=['id','cycle',*[f'setting{i}' for i in range(1,4)],*[f's{i}' for i in range(1,22)]]

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 if len(sys.argv)!=2:
  sys.exit('Usage: python scripts/prepare_nasa_data.py /path/to/NASA_turbofan_original.zip')
 input=Path(sys.argv[1]).expanduser().resolve(strict=True)
 root=Path(__file__).resolve().parents[1]
 data=root/'02_raw_data'
 if sha(input)!=EXPECTED_SHA:
  sys.exit('STOP: NASA zip SHA-256 mismatch. Source version must be audited first.')
 if data.exists() and any(data.iterdir()):
  sys.exit('STOP: output 02_raw_data is not empty; no overwrite permitted')
 data.mkdir(exist_ok=True)
 outputs=[]
 try:
  with ZipFile(input) as outer:
   n=[name for name in outer.namelist() if Path(name).name=='CMAPSSData.zip']
   if len(n)!=1:raise ValueError('CMAPSSData.zip missing/ambiguous')
   inner_bytes=outer.read(n[0])
  with ZipFile(io.BytesIO(inner_bytes)) as inner:
   for ds,(expected_rows,expected_engines) in EXPECTED.items():
    member=[n for n in inner.namelist() if Path(n).name==f'train_{ds}.txt']
    if len(member)!=1:raise ValueError(f'NASA member missing/ambiguous: {ds}')
    path=data/NAMES[ds]
    if path.exists():raise ValueError(f'Already exists: {path}')
    tmp=path.with_suffix('.csv.partial')
    engines=defaultdict(list); count=0
    with inner.open(member[0]) as origin, tmp.open('w',newline='',encoding='utf-8') as dest:
     w=csv.writer(dest,lineterminator='\n');w.writerow(HEADER)
     for line in origin:
      fields=line.decode('ascii').split()
      if not fields:continue
      if len(fields)!=26:raise ValueError(f'{ds}: invalid column count')
      eng,cycle=int(fields[0]),int(fields[1])
      engines[eng].append(cycle)
      count+=1;w.writerow(fields)
    if count!=expected_rows or len(engines)!=expected_engines or any(c!=list(range(1,len(c)+1)) for c in engines.values()):
     raise ValueError(f'{ds}: unexpected rows, engines or cycles')
    tmp.rename(path);outputs.append(path)
    print(f'{ds}: rows={count}, engines={len(engines)}, SHA256={sha(path)}',flush=True)
  record={'nasa_zip_sha256':EXPECTED_SHA,'nested_zip_sha256':hashlib.sha256(inner_bytes).hexdigest(),
   'date_utc':datetime.now(timezone.utc).isoformat(),
   'note':'Local CSV conversion. Source numerical identity should be independently checked; no redistribution license granted.',
   'datasets':{p.name:sha(p) for p in outputs}}
  (data/'data_provenance.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
 except Exception:
  for path in outputs:path.unlink(missing_ok=True)
  for path in data.glob('*.partial'):path.unlink(missing_ok=True)
  raise

if __name__=='__main__':main()
