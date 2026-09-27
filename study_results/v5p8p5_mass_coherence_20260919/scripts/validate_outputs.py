"""Validate stored numeric products, pinned inputs, and untouched parent release."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
B=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for item in json.loads((B/'inputs/copy_manifest.json').read_text()):
 assert sha(B/item['copy'])==item['sha256'],item['copy']
 checks.append('input '+item['copy'])
f=pd.read_csv(B/'results/free_curves.csv');p=pd.read_csv(B/'results/free_peaks.csv')
for w in [2.25,2.4,2.5,2.6]:
 a=f[f.width_sigma==w].sort_values('mass_MeV');assert np.array_equal(a.mass_MeV,np.arange(19,251));q=np.zeros(232)
 for y in ['2015','2016','2021']:
  d=np.load(B/f'inputs/parent_fields/w{w:.2f}_{y}.npz');idx=(d['masses']-19).astype(int);q[idx]+=np.maximum(d['observed_r'],0)**2
 assert np.max(abs(q-a.q_R))<1e-12
 assert ((a.local_p>0)&(a.local_p<=1)).all()
 assert np.all(a[a.q_R==0].local_p==1)
 assert (a.global_p+1e-8>=a.local_p-5e-3).all()
 z=np.load(B/f'fields/free_w{w:.2f}.npz');assert np.allclose(z['validation_score'].max(axis=1),z['direct_maximum'])
 pp=p[p.width_sigma==w].iloc[0];assert pp.mass_MeV==a.iloc[np.argmax(a.score)].mass_MeV
 j=int(pp.mass_MeV-19);k=int((z['direct_maximum']>=z['observed_score'][j]).sum());assert k==pp.direct_global_k
 for name in ['direct','gaussian']:
  arr=z[name+'_maximum'];calc=np.array([(arr>=x).sum() for x in z['observed_score']]);assert np.array_equal(calc,a[name+'_global_k'])
 checks.append(f'full scan exact statistic, atoms, extrema and exceedances width {w}')
s=json.loads((B/'results/stability_coherence_summary.json').read_text());z=np.load(B/'fields/stability_null.npz')
for source in ['direct','gaussian']:
 for name in ['T','W']:
  v=z[source+'_'+name];assert s[source+'_'+name]['k']==int((v<=s['observed_'+name]).sum())
checks.append('paired coherence/stability inclusive lower tails')
repo=B.parents[1];parent=json.loads((B/'qa/parent_before.json').read_text());existing=[repo/k for k in parent]
parent_checked=all(p.exists() for p in existing)
if parent_checked:
 for name,value in parent.items():assert sha(repo/name)==value,name
 checks.append(f'all {len(parent)} parent study and release files unchanged')
# Absence of a parent checkout is valid for an unpacked portable bundle.
result=dict(complete=True,checks=checks,parent_release_checked=parent_checked,parent_files=len(parent),free_rows=len(f),free_width_peaks=len(p),source_pin_count=len(json.loads((B/'inputs/copy_manifest.json').read_text())))
(B/'qa/final_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
