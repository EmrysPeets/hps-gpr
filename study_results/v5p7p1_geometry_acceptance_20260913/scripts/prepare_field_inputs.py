"""Restore the three active ASCII maps from bundled, pinned archives."""
from pathlib import Path
import tarfile,json,hashlib,shutil
B=Path(__file__).resolve().parents[1]
c=json.loads((B/'inputs/geometry/fieldmaps/active_field_config.json').read_text())
for y,v in c['years'].items():
 p=B/v['input_relative_path']
 if not p.exists():
  archive=p.with_suffix('.tar.gz')
  with tarfile.open(archive) as tf:
   member=next(m for m in tf.getmembers() if Path(m.name).name==p.name and m.isfile())
   with tf.extractfile(member) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],f'{y}: active field hash mismatch'
 print(y,p.name,'verified')
