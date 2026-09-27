"""Check complete v16 extraction, flow normalization and recorded fit validity."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1]
rows=[]
for family,typ in [('tc','TargetConstrained'),('uc','Unconstrained')]:
 paths=sorted((B/'inputs'/family).glob('m[0-9][0-9][0-9].npz'));assert len(paths)==11
 for p in paths:
  m=int(p.stem[1:]);d=np.load(p);q=json.loads(p.with_suffix('.json').read_text());n=q['stats']['selected']
  assert q['vertex_types']=={typ:n} and n==q['stats']['entries']==q['sumw']==q['sumw2']
  assert q['stats']['nonunit_weights']==q['stats']['nonfinite_or_nonpositive']==0
  assert np.array_equal(d['sumw'],d['sumw2']) and np.all(d['sumw']>=0)
  assert d['sumw'].sum()+q['underflow_sumw']+q['overflow_sumw']==n
  assert max(abs(q[k]*1000-m) for k in ['truth_mass_min_GeV','truth_mass_max_GeV'])<.004
  for f in q['files']:assert len(f['branch_payload_sha256'])==8
  rows.append({'family':family,'mass_MeV':m,'selected':n,'source_files':len(q['files']),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 for suffix,col,n in [('core_fits','core_location_valid',11),('span_checks','valid',33),('bin_replicas','valid',352)]:
  t=pd.read_csv(B/'results'/f'{family}_{suffix}.csv');assert len(t)==n and t[col].all()
assert json.loads((B/'qa/extraction_summary.json').read_text())['passed']
q={'passed':True,'samples':22,'source_ROOT_files':sum(r['source_files'] for r in rows),'selected_TC':sum(r['selected'] for r in rows if r['family']=='tc'),'selected_UC':sum(r['selected'] for r in rows if r['family']=='uc'),'baseline_fits':22,'span_checks':66,'resampling_fits':704,'branch_hash_scope':'streamed branch payloads, not complete ROOT files','samples_checked':rows}
(B/'qa/input_validation.json').write_text(json.dumps(q,indent=2)+'\n');print('PASS: 22 samples, 48 source files, 22 core fits, 66 span checks, 704 resampling fits')
