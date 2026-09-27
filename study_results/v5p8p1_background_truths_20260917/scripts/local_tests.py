#!/usr/bin/env python3
"""Conditional local tests of frozen coherent alternative backgrounds."""
from pathlib import Path
import os,sys,time,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1]
LEGACY=Path('/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p2-20260909')
sys.path.insert(0,str(LEGACY/'study_results/v4p9p13_calibration_20260905'))
import calibration_core as core
c=core.c
from gp_refit_pilot import CachedCholeskyPredictor
sys.path.insert(0,str(ROOT/'study_results/v5p8p0_local_significance_mapping_20260917/gpr'))
from fast_profile import FastBatchProfile as BatchProfile
from hps_gpr.template import build_window_template_from_full
from hps_gpr.statistics import _chol_with_jitter
import numpy as np,pandas as pd
from scipy.stats import norm,beta
P=json.loads((B/'protocol.json').read_text());N=P['new_poisson_per_anchor_truth'];start=time.monotonic()
cfg=c.production.load_config(c.production.DEFAULT_CARD);datasets=c.production.make_datasets(cfg);states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
x=dict(np.load(B/'truths/backgrounds.npz'));extra=dict(np.load(B/'truths/supplementary.npz'));x.update({k:v for k,v in extra.items() if k not in ['edges_GeV','observed','x_MeV']});names=[k for k in x if k not in ['edges_GeV','observed','x_MeV']]
rows=[];checks=[];failures=[]
def write(p,o):Path(p).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
for mass in P['anchors_MeV']:
 st=states['2016',mass];kernel=c.make_fixed_kernel(st['const_opt'],st['ls_opt'])
 p=c.production.estimate_background_for_dataset(datasets['2016'],mass/1000,cfg,rebin=5,restarts=0,kernel=kernel,optimize=False)
 assert np.allclose(p.y_full,x['observed'],rtol=0,atol=0)
 mask=p.blind_mask;keep=~mask;w,S=build_window_template_from_full(p.edges_full,mask,mass/1000,p.sigma_val,config=cfg)
 pred=CachedCholeskyPredictor(p.x_full[keep],p.x_full[mask],kernel,cfg)
 def evaluate(spectra):
  roots=[]
  for i in range(0,len(spectra),32):
   block=spectra[i:i+32];bs=[];Ls=[]
   for n in block:
    b,C=pred.predict(n[keep]);C,_=c.production.condition_covariance_block(C,b);bs.append(b);Ls.append(_chol_with_jitter(C))
   counts=block[:,mask];bs=np.array(bs);Ls=np.array(Ls);model=BatchProfile(counts,bs,Ls,w)
   scalar=c.Profile(bs[0],Ls[0],w,'linear');f=scalar.fit(counts[0]);z=scalar.fit(counts[0],0.);r=float(np.sign(f['A'])*np.sqrt(max(0,2*(z['nll']-f['nll']))));err=float(abs(r-model.r[0]));assert err<2e-5 and model.max_score<2e-7,(err,model.max_score)
   checks.append(dict(mass_MeV=mass,scalar_error=err,max_score=float(model.max_score),size=len(block)));roots.extend(model.r)
  return np.asarray(roots)
 robs=float(evaluate(np.array([x['observed']]))[0]);saved={'observed_r':robs}
 for ti,name in enumerate(names):
  seed=[58120260917,mass,ti];rng=np.random.default_rng(np.random.SeedSequence(seed));truth=x[name];
  try:
   a=float(evaluate(np.array([truth]))[0])
   rr=evaluate(rng.poisson(truth,size=(N,len(truth))).astype(float))
  except (RuntimeError,AssertionError) as exc:
   failures.append(dict(mass_MeV=mass,truth=name,seed=seed,requested_experiments=N,error=str(exc),probability_reported=False))
   write(B/'qa/local_failures.json',failures);print('FAILED',mass,name,str(exc),flush=True);continue
  saved[name]=rr;saved[name+'_seed']=seed
  atom=robs<=0;k=N if atom else int(np.sum(rr>=robs));lo=1. if atom else (0. if k==0 else float(beta.ppf(.025,k,N-k+1)));hi=1. if k==N else float(beta.ppf(.975,k+1,N-k))
  rows.append(dict(mass_MeV=mass,truth=name,N=N,observed_r=robs,nominal_p0=float(norm.sf(max(0,robs))),asimov_r=a,mean_r=float(rr.mean()),sd_r=float(rr.std(ddof=1)),exceedances=k,p_mle=k/N,p_addone=(k+1)/(N+1),p_lo95=lo,p_hi95=hi,bounded_atom=atom,one_sided_upper95=1. if k==N else float(beta.ppf(.95,k+1,N-k)),raw_signed_exceedances=int(np.sum(rr>=robs)),raw_nominal05_rejections=int(np.sum(rr>norm.isf(.05)))))
  print('local',mass,name,'a',round(a,3),'k',k,'elapsed',round(time.monotonic()-start,1),flush=True)
  pd.DataFrame(rows).to_csv(B/'results/local_tail_tests.csv',index=False)
 np.savez_compressed(B/f'results/local_{mass}.npz',**saved)
q=pd.DataFrame(rows);summary={'complete':True,'passed':True,'new_poisson_experiments':len(q)*N,'scenarios':len(q),'maximum_scalar_error':max(v['scalar_error'] for v in checks),'maximum_optimizer_score':max(v['max_score'] for v in checks),'checks':checks,'elapsed_seconds':time.monotonic()-start,'global_calibration':False,'failed_scenarios':failures,'all_scenarios_passed':not failures}
write(B/'qa/local_validation.json',summary)
files=[Path(__file__),Path(core.__file__),Path(c.__file__),Path(sys.modules['gp_refit_pilot'].__file__),Path(sys.modules['fast_profile'].__file__),Path(sys.modules['batch_profile'].__file__),B/'protocol.json',B/'truths/backgrounds.npz',B/'truths/supplementary.npz',c.production.DEFAULT_STATES,c.production.DEFAULT_CARD]
write(B/'inputs/local_manifest.json',{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
print('COMPLETE',len(q)*N,time.monotonic()-start,flush=True)
