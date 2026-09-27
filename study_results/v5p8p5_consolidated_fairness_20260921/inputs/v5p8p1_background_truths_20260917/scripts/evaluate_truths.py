#!/usr/bin/env python3
"""Fixed inference policy, coherent alternative truths; no truth selection."""
from pathlib import Path
import os,sys,time,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1]
sys.path.insert(0,str(ROOT/'study_results/v5p0p5_analysis_note_20260916/scripts'))
import common as C
import numpy as np,pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import norm
P=json.loads((B/'protocol.json').read_text());start=time.monotonic()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def fit(mass,counts):
 p=C.moving_context('2016',mass,counts=counts);s=p['S'][:,0];mod=C.OneSignalProfile(p['b'],p['L'],s);f=mod.fit(p['n']);z=mod.fit(p['n'],0.)
 sig=float(1/np.sqrt(s@cho_solve(cho_factor(np.diag(p['b'])+p['L']@p['L'].T,lower=True),s)))
 return dict(signed_r=float(np.sign(f['A'])*np.sqrt(max(0,2*(z['nll']-f['nll'])))),A_hat=float(f['A']),sigma_fisher=sig,max_score=float(max(f['score'],z['score'])),min_lambda=float(min(f['min_lambda'],z['min_lambda'])),load=p['diagnostic']['load'])
x=dict(np.load(B/'truths/backgrounds.npz'));extra=dict(np.load(B/'truths/supplementary.npz'));x.update({k:v for k,v in extra.items() if k not in ['edges_GeV','observed','x_MeV']});names=[k for k in x if k not in ['edges_GeV','observed','x_MeV']]
assert np.allclose(x['observed'],C.DATA['2016']['n'],rtol=0,atol=0)
assert np.allclose(x['edges_GeV'],C.DATA['2016']['edges'],rtol=0,atol=1e-14)
rows=[];injections=[];refs={}
for mass in P['anchors_MeV']:
 refs[mass]=fit(mass,x['observed'])
for mass in P['scan_masses_MeV']:
 for name in names:
  f=fit(mass,x[name]);rows.append(dict(mass_MeV=mass,truth=name,**f))
  if mass in refs:
   for strength in [2.,5.]:
    sig=refs[mass]['sigma_fisher'];fi=fit(mass,x[name]+strength*sig*C.signal('2016',mass))
    injections.append(dict(mass_MeV=mass,truth=name,strength_original_sigma=strength,sigma_reference=sig,recovered_fraction=(fi['A_hat']-f['A_hat'])/(strength*sig),delta_r=fi['signed_r']-f['signed_r'],baseline_r=f['signed_r'],**fi))
 if mass%10==0 or mass==180:
  pd.DataFrame(rows).to_csv(B/'results/asimov_scan.csv',index=False);pd.DataFrame(injections).to_csv(B/'results/injections.csv',index=False)
  print('mass',mass,'elapsed',round(time.monotonic()-start,1),flush=True)
d=pd.DataFrame(rows);inj=pd.DataFrame(injections);summ=[]
for name,g in d.groupby('truth',sort=False):
 j=g.signed_r.abs().idxmax();summ.append(dict(truth=name,masses=len(g),rms_root=float(np.sqrt(np.mean(g.signed_r**2))),max_abs_root=float(g.signed_r.abs().max()),max_abs_mass_MeV=float(d.loc[j,'mass_MeV']),minimum_root=float(g.signed_r.min()),maximum_root=float(g.signed_r.max()),fraction_abs_gt1=float(np.mean(g.signed_r.abs()>1)),fraction_abs_gt2=float(np.mean(g.signed_r.abs()>2))))
pd.DataFrame(summ).to_csv(B/'results/asimov_summary.csv',index=False)
pd.DataFrame([dict(mass_MeV=m,**r) for m,r in refs.items()]).to_csv(B/'results/observed_anchors.csv',index=False)
qa={'passed':bool((d.max_score<2e-7).all() and (inj.max_score<2e-7).all() and (d.min_lambda>0).all() and (inj.min_lambda>0).all()),'truths':names,'scan_rows':len(d),'injection_rows':len(inj),'minimum_injection_recovered_fraction':float(inj.recovered_fraction.min()),'maximum_injection_recovered_fraction':float(inj.recovered_fraction.max()),'maximum_score':float(max(d.max_score.max(),inj.max_score.max())),'elapsed_seconds':time.monotonic()-start,'global_calibration':False}
write(B/'qa/asimov_validation.json',qa)
paths=[Path(__file__),B/'protocol.json',B/'truths/backgrounds.npz',B/'truths/supplementary.npz',Path(C.__file__),ROOT/'study_results/v5p0p5_analysis_note_20260916/scripts/parent_core.py',ROOT/'study_results/v5p0p5_analysis_note_20260916/scripts/limit_solver.py',ROOT/'study_results/v5p0p5_analysis_note_20260916/inputs/spectrum_2016.npz']
write(B/'inputs/asimov_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths});print(json.dumps(qa,indent=2));assert len(d)==len(names)*len(P['scan_masses_MeV']) and (d.min_lambda>0).all()
