#!/usr/bin/env python3
"""New bounded direct local-tail experiments under two explicit plug-in truths."""
from pathlib import Path
import os,sys,time,json,hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1]; REPO=B.parents[1]
LEGACY=Path('/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p2-20260909')
sys.path.insert(0,str(LEGACY/'study_results/v4p9p13_calibration_20260905'))
import calibration_core as core
c=core.c
core.STRESS={k:(REPO/p.relative_to(LEGACY),key) for k,(p,key) in core.STRESS.items()}
from gp_refit_pilot import CachedCholeskyPredictor
from batch_profile import BatchProfile
from hps_gpr.template import build_window_template_from_full
from hps_gpr.statistics import _chol_with_jitter
import numpy as np,pandas as pd
from scipy.stats import norm,beta,kstest
from datetime import datetime,timezone
P=json.loads((B/'protocol.json').read_text()); N=P['new_null_experiments_per_anchor_truth']
deadline=datetime.fromisoformat(P['fit_deadline_utc'].replace('Z','+00:00')).timestamp()
cfg=c.production.load_config(c.production.DEFAULT_CARD); datasets=c.production.make_datasets(cfg)
states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
start=time.monotonic();rows=[];manifest={}
previous=json.loads((B/'qa/local_checks.json').read_text()) if (B/'qa/local_checks.json').exists() else {}
checks=previous.get('checks',[])
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for module in [core,c,sys.modules['gp_refit_pilot'],sys.modules['batch_profile'],c.production,sys.modules['hps_gpr.template'],sys.modules['hps_gpr.statistics']]:
    p=Path(module.__file__);manifest[str(p)]=sha(p)
for p in [Path(__file__),B/'protocol.json',c.production.DEFAULT_CARD,c.production.DEFAULT_STATES]+[p for p,key in core.STRESS.values()]:manifest[str(p)]=sha(p)
write(B/'inputs/local_runtime_manifest.json',manifest)
for year,masses in P['local_anchors'].items():
  for mass in masses:
    checkpoint=B/f'local/{year}_{mass}.npz'
    if checkpoint.exists():
      rr=json.loads((B/f'local/{year}_{mass}.json').read_text())
      for saved in rr:
        if saved['observed_r']<=0: saved['p_lo95']=saved['p_hi95']=1.
      rows.extend(rr); continue
    st=states[year,mass];kernel=c.make_fixed_kernel(st['const_opt'],st['ls_opt'])
    p=c.production.estimate_background_for_dataset(datasets[year],mass/1000,cfg,rebin=5,restarts=0,kernel=kernel,optimize=False)
    mask=p.blind_mask;keep=~mask
    w,S=build_window_template_from_full(p.edges_full,mask,mass/1000,p.sigma_val,config=cfg)
    pred=CachedCholeskyPredictor(p.x_full[keep],p.x_full[mask],kernel,cfg)
    full=CachedCholeskyPredictor(p.x_full[keep],p.x_full,kernel,cfg)
    control,_=full.predict(p.y_full[keep]);stress=core.stress_truth(year,p)
    def evaluate(spectra):
      roots=[]
      for startrow in range(0,len(spectra),32):
        if time.time()>deadline:raise RuntimeError('Predeclared fit deadline reached')
        block=spectra[startrow:startrow+32]; bs=[]; Ls=[]
        for n in block:
          b,C=pred.predict(n[keep]);C,_=c.production.condition_covariance_block(C,b)
          bs.append(b);Ls.append(_chol_with_jitter(C))
        counts=block[:,mask];bs=np.array(bs);Ls=np.array(Ls)
        model=BatchProfile(counts,bs,Ls,w)
        scalar=c.Profile(bs[0],Ls[0],w,'linear');f=scalar.fit(counts[0]);z=scalar.fit(counts[0],0.)
        r=float(np.sign(f['A'])*np.sqrt(max(0,2*(z['nll']-f['nll']))));err=abs(r-model.r[0])
        assert err<2e-5 and model.max_score<2e-7,(err,model.max_score)
        checks.append(dict(year=year,mass=mass,scalar_error=err,max_score=model.max_score,n=len(block)))
        roots.extend(model.r)
      return np.array(roots)
    baseline=evaluate(np.array([p.y_full,stress,control])); robs=float(baseline[0])
    arrays=dict(observed=p.y_full,stress=stress,control=control,edges_GeV=p.edges_full,baseline_roots=baseline,mass=mass)
    localrows=[]
    for tid,(truthname,truth) in enumerate([('archived_stress',stress),('local_GP_control',control)]):
      seed=[58020260917,int(year),mass,tid];rng=np.random.default_rng(np.random.SeedSequence(seed))
      samples=rng.poisson(truth,size=(N,len(truth))).astype(float);r=evaluate(samples)
      arrays[truthname+'_roots']=r;arrays[truthname+'_seed']=seed
      k=int(np.sum(r>=robs)) if robs>0 else N
      lo=0. if k==0 else float(beta.ppf(.025,k,N-k+1));hi=1. if k==N else float(beta.ppf(.975,k+1,N-k))
      if robs<=0:lo=hi=1.
      upper95=float(1-.05**(1/N)) if k==0 else None
      # Split-sample Gaussian diagnostic: fitted moments from the first half, tails from the second.
      mu=float(r[:N//2].mean());sd=float(r[:N//2].std(ddof=1));hold=(r[N//2:]-mu)/sd
      rr=dict(dataset=year,mass_MeV=mass,truth=truthname,N=N,exceedances=k,observed_r=robs,
        nominal_p0=float(norm.sf(max(0,robs))),asimov_r=float(baseline[tid+1]),mean_r=float(r.mean()),sd_r=float(r.std(ddof=1)),
        p_mle=k/N,p_addone=(k+1)/(N+1),p_lo95=lo,p_hi95=hi,zero_count_upper95=upper95,
        conditional_Z_display=max(0,float(norm.isf((k+1)/(N+1)))) if k<N else 0.,
        heldout_n=len(hold),heldout_mean=float(hold.mean()),heldout_sd=float(hold.std(ddof=1)),
        heldout_r_gt_1p645=int(np.sum(hold>norm.isf(.05))),heldout_r_gt_2p326=int(np.sum(hold>norm.isf(.01))),
        heldout_KS_p=float(kstest(hold,'norm').pvalue),bounded_atom=bool(robs<=0),
        raw_signed_root_exceedances=int(np.sum(r>=robs)),
        interval_kind='exact inclusive q0 atom; no Monte Carlo uncertainty' if robs<=0 else 'two-sided95 Clopper-Pearson conditional binomial interval')
      rows.append(rr);localrows.append(rr)
    np.savez_compressed(checkpoint,**arrays);write(B/f'local/{year}_{mass}.json',localrows)
    pd.DataFrame(rows).to_csv(B/'local/local_tail_tests.csv',index=False,float_format='%.17g')
    write(B/'qa/local_checks.json',dict(complete=False,checks=checks))
    print(year,mass,'r',round(robs,4),'p',[(x['truth'],x['exceedances']) for x in localrows],'elapsed',round(time.monotonic()-start,1),flush=True)
q=pd.DataFrame(rows)
q.to_csv(B/'local/local_tail_tests.csv',index=False,float_format='%.17g')
env=[]
for (year,mass),g in q.groupby(['dataset','mass_MeV'],sort=False):
  best=g.loc[g.p_addone.idxmax()]
  env.append(dict(dataset=year,mass_MeV=mass,observed_r=best.observed_r,nominal_p0=best.nominal_p0,
    finite_family_p_addone=best.p_addone,controlling_truth=best.truth,
    conditional_Z_display=best.conditional_Z_display,physical_significance_validated=False))
pd.DataFrame(env).to_csv(B/'local/finite_family_envelope.csv',index=False,float_format='%.17g')
write(B/'qa/local_checks.json',dict(complete=True,new_experiments=len(rows)*N,pointwise_scenarios=len(rows),
  elapsed_seconds=time.monotonic()-start+previous.get('elapsed_seconds',0.),checks=checks,maximum_scalar_error=max(x['scalar_error'] for x in checks),
  maximum_score=max(x['max_score'] for x in checks),no_scan_global_inference=True))
print('COMPLETE',len(rows)*N,time.monotonic()-start,flush=True)
