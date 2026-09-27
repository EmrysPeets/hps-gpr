"""Stream reviewed read-only helper through an existing authenticated connection."""
from pathlib import Path
import subprocess,json,numpy as np,hashlib,time
B=Path(__file__).resolve().parents[1]
cmd=['ssh','-S','/tmp/hps-v61-s3df.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','epeets@s3dfdtn.slac.stanford.edu',"ssh -o BatchMode=yes -o ConnectTimeout=8 iana '/sdf/home/e/epeets/src/hps-gpr-main/venv/bin/python -'"]
start=time.time();rows=[]
with (B/'scripts/extract_v16_remote.py').open('rb') as script,(B/'qa/extraction.stderr').open('w') as log:
 p=subprocess.Popen(cmd,stdin=script,stdout=subprocess.PIPE,stderr=log,text=True)
 for line in p.stdout:
  row=json.loads(line);family=row['family'];m=row['mass_MeV'];out=B/'inputs'/family;out.mkdir(exist_ok=True,parents=True);meta=row['metadata']
  np.savez_compressed(out/f'm{m:03d}.npz',edges_GeV=row['edges_GeV'],sumw=row['sumw'],sumw2=row['sumw2'],metadata=json.dumps(meta))
  (out/f'm{m:03d}.json').write_text(json.dumps(meta,indent=2)+'\n');rows.append({'family':family,'mass_MeV':m,'selected':meta['stats']['selected'],'sha256':hashlib.sha256((out/f'm{m:03d}.npz').read_bytes()).hexdigest()})
  print(f'Saved {family} {m} MeV: {meta["stats"]["selected"]} candidates',flush=True)
 rc=p.wait()
 (B/'qa/extraction_summary.json').write_text(json.dumps({'passed':rc==0 and len(rows)==22,'returncode':rc,'seconds':time.time()-start,'samples':rows},indent=2)+'\n')
 raise SystemExit(rc)
