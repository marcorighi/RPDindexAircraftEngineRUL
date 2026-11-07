#!/usr/bin/env python3
"""Archive only a contained output directory; never embed NASA raw data."""
import hashlib,sys,zipfile
from pathlib import Path
if len(sys.argv)!=2:sys.exit('Usage: archive_result.py /path/to/generated_result_dir')
p=Path(sys.argv[1]).resolve(strict=True)
root=Path(__file__).resolve().parents[1]
results=(root/'04_results').resolve()
if p.parent!=results:sys.exit('Refusing path outside result root')
zdir=results/'ZIP';zdir.mkdir(exist_ok=True)
zpath=zdir/(p.name+'.zip')
if zpath.exists():sys.exit('Refusing to overwrite ZIP')
with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for f in sorted(p.rglob('*')):
  if f.is_file():z.write(f,arcname=f'{p.name}/{f.relative_to(p)}')
with zipfile.ZipFile(zpath) as z:
 if z.testzip():sys.exit('ZIP integrity error')
h=hashlib.sha256(zpath.read_bytes()).hexdigest()
(zdir/(zpath.name+'.sha256')).write_text(f'{h}  {zpath.name}\n')
print('ARCHIVE:',zpath)
