#!/usr/bin/env python3
from pathlib import Path
import os,sys,json,time,hashlib
from datetime import datetime, timezone
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
from scipy.stats import norm,beta,kstest
B=Path(__file__).resolve().parent
DATA=B.parents[2]
R=Path('/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p2-20260909')
OLD=DATA/'study_results/v5p1p1_significance_covariance_injection_20260910'
sys.path.insert(0,str(R/'study_results/v4p9p13_calibration_20260905'))
import calibration_core as core
c=core.c
from gp_refit_pilot import CachedCholeskyPredictor
from fast_profile import FastBatchProfile as BatchProfile
from hps_gpr.template import build_window_template_from_full
from hps_gpr.statistics import _chol_with_jitter
cfg=c.production.load_config(c.production.DEFAULT_CARD)
datasets=c.production.make_datasets(cfg)
states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
G={'2015':R/'study_results/v4p9p14_interpretation_global_20260906/global/2015','2016':R/'study_results/v4p9p15_global_2016_2021_20260906/global_fast/2016','2021':R/'study_results/v4p9p15_global_2016_2021_20260906/global_fast/2021'}
START=time.monotonic();CHECKS=[];INPUTS={}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def register(p):
 p=Path(p);INPUTS[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size};return p
for p in [__file__,B/'rank_one.py',B/'fast_profile.py',B/'protocol.json',core.__file__,c.__file__,c.production.__file__,sys.modules['gp_refit_pilot'].__file__,sys.modules['batch_profile'].__file__,c.production.DEFAULT_CARD,c.production.DEFAULT_STATES,c.production.DEFAULT_INPUT_PROVENANCE]:register(p)
for line in c.production.DEFAULT_CARD.read_text().splitlines():
 if line.startswith(('path_2015:','path_2016:','path_2021:')):register(line.split(': ',1)[1])
for name in c.production.REQUIRED_RUNTIME_MODULES:register(sys.modules[name].__file__)
def setup(year,mass):
 m=min(float(mass),90.) if year=='2015' else float(mass)
 lo,hi=int(np.floor(m)),int(np.ceil(m));u=m-lo
 const=np.exp((1-u)*np.log(states[year,lo]['const_opt'])+u*np.log(states[year,hi]['const_opt']))
 ls=np.exp((1-u)*np.log(states[year,lo]['ls_opt'])+u*np.log(states[year,hi]['ls_opt']))
 kernel=c.make_fixed_kernel(const,ls)
 p=c.production.estimate_background_for_dataset(datasets[year],mass/1000,cfg,rebin=5,restarts=0,kernel=kernel,optimize=False)
 keep=~p.blind_mask;w,S=build_window_template_from_full(p.edges_full,p.blind_mask,mass/1000,p.sigma_val,config=cfg)
 pred=CachedCholeskyPredictor(p.x_full[keep],p.x_full[p.blind_mask],kernel,cfg)
 return dict(p=p,keep=keep,w=w,S=S,pred=pred,kernel=kernel,const=float(const),ls=float(ls))
def evaluate(ctx,spectra,moment_provider=None):
 roots=[]
 for start in range(0,len(spectra),32):
  spec=spectra[start:start+32];bs=[];Ls=[]
  for offset,n in enumerate(spec):
   b,C=ctx['pred'].predict(n[ctx['keep']]) if moment_provider is None else moment_provider(start+offset)
   C,_=c.production.condition_covariance_block(C,b);bs.append(b);Ls.append(_chol_with_jitter(C))
  bs=np.array(bs);Ls=np.array(Ls);n=spec[:,ctx['p'].blind_mask];model=BatchProfile(n,bs,Ls,ctx['w'])
  scalar=c.Profile(bs[0],Ls[0],ctx['w'],'linear');f=scalar.fit(n[0]);z=scalar.fit(n[0],0.)
  r=np.sign(f['A'])*np.sqrt(max(0,2*(z['nll']-f['nll'])));err=float(abs(r-model.r[0]))
  assert err<2e-5 and model.max_score<2e-7
  CHECKS.append(dict(n=len(spec),scalar_error=err,max_score=float(model.max_score),fallbacks=int(model.fallbacks)))
  roots.extend(model.r)
 return np.array(roots)
def observed(ctx):
 p=ctx['p'];C,_=c.production.condition_covariance_block(p.cov,p.mu);mod=c.Profile(p.mu,_chol_with_jitter(C),ctx['w'],'linear');n=p.y_full[p.blind_mask]
 f=mod.fit(n);z=mod.fit(n,0.);r=float(np.sign(f['A'])*np.sqrt(max(0,2*(z['nll']-f['nll']))))
 return dict(observed_r=r,Ahat=f['A'],sigma_A=f['sigma'],const=ctx['const'],ls=ctx['ls'],signal_sigma_MeV=float(p.sigma_val*1000),n_signal_bins=int(p.blind_mask.sum()),n_training_bins=int(ctx['keep'].sum()))
def main():
 assert json.loads((B/'rank_one_validation.json').read_text())['passed']
 rows=[];aggregates=[]
 for year in ['2016','2015','2021']:
  old=np.load(register(OLD/f'derived/field_{year}/field.npz'))
  oldm=old['masses'];truth=old['truth'];Vold=old['validation'];Dold=old['D'];aold=old['a']
  observed_table=pd.read_csv(register(G[year]/'analysis/pvalue_curves.csv'));observed_table=observed_table[observed_table.method=='profiled'].set_index('mass_MeV')
  masses=oldm.copy()
  if year=='2016':masses=np.unique(np.r_[oldm,np.arange(39.5,180,.5),np.arange(74.25,79,.25)])
  folder=B/f'field_{year}';folder.mkdir(exist_ok=True)
  spec=np.load(register(G[year]/'validation1000/spectra.npz'))
  assert np.array_equal(spec['truth'],truth)
  spectra=spec['counts'][:256];asimov=np.broadcast_to(truth,(len(truth)+1,len(truth))).copy();ii=np.arange(len(truth));asimov[ii+1,ii]+=np.sqrt(truth)
  cols=[];vals=[];obs=[];detail=[]
  for k,mass in enumerate(masses):
   inherited=np.flatnonzero(oldm==mass);cp=folder/f'm{mass:07.2f}.npz';meta=folder/f'm{mass:07.2f}.json'
   if len(inherited):
    j=int(inherited[0]);a=float(aold[j]);D=Dold[:,j];v=Vold[:,j]
    if mass in observed_table.index:o=dict(observed_r=float(observed_table.loc[mass,'observed_r']))
    else:o=observed(setup(year,float(mass)))
   elif cp.exists():
    x=np.load(cp);a=float(x['a']);D=x['D'];v=x['validation'];o=json.loads(meta.read_text())
   else:
    assert datetime.now(timezone.utc)<datetime(2026,9,17,17,43,tzinfo=timezone.utc), 'Declared fit deadline reached; per-mass checkpoints preserved'
    ctx=setup(year,float(mass))
    from rank_one import RankOneAsimov
    try:
     fast=RankOneAsimov(ctx,truth,cfg);gp_qa=fast.qa
    except (AssertionError,np.linalg.LinAlgError) as error:
     fast=None;gp_qa=[{'exact_fallback':True,'reason':str(error)}]
    rs=evaluate(ctx,asimov,moment_provider=fast);v=evaluate(ctx,spectra);a=float(rs[0]);D=rs[1:]-rs[0];o=observed(ctx)
    np.savez_compressed(cp,a=a,D=D,validation=v);o['rank_one_gp_checks']=gp_qa;write(meta,o)
   sd=float(np.linalg.norm(D));r=o['observed_r'];z=(r-a)/sd;vc=(v-a)/sd;kexc=int(np.count_nonzero(v>=r))
   train=v[:128];hold=v[128:];zh=(hold-train.mean())/train.std(ddof=1)
   low=0. if kexc==0 else float(beta.ppf(.025,kexc,257-kexc));high=1. if kexc==256 else float(beta.ppf(.975,kexc+1,256-kexc))
   row=dict(dataset=year,mass_MeV=float(mass),inherited_coordinate=bool(len(inherited)),observed_r=r,stress_offset=a,response_sd=sd,z_stress_conditional=float(z),p_raw_gaussian=float(norm.sf(max(0,r))),p_stress_gaussian_ungated=float(norm.sf(z)),p_stress_gated=float(norm.sf(z)) if r>0 else 1.,empirical_exceedances=kexc,empirical_addone=(kexc+1)/257,empirical_low=low,empirical_high=high,stress_toy_mean=float(v.mean()),stress_toy_sd=float(v.std(ddof=1)),centered_toy_mean=float(vc.mean()),centered_toy_sd=float(vc.std(ddof=1)),holdout_mean=float(zh.mean()),holdout_sd=float(zh.std(ddof=1)),holdout_absZ_lt1p96=float(np.mean(abs(zh)<1.96)),holdout_normal_KS_p=float(kstest(zh,'norm').pvalue),**{a:b for a,b in o.items() if a not in ['observed_r','rank_one_gp_checks']})
   rows.append(row);detail.append(row);cols.append(np.r_[a,D]);vals.append(v);obs.append(r)
   if k%20==0 or not len(inherited):print(year,k+1,len(masses),mass,'new',not bool(len(inherited)),'sec',round(time.monotonic()-START,1),flush=True)
  X=np.column_stack(cols);a=X[0];D=X[1:];C=D.T@D;sd=np.sqrt(np.diag(C));K=C/np.outer(sd,sd);V=np.column_stack(vals);Z=(V-a)/sd
  np.savez_compressed(folder/'field.npz',masses=masses,a=a,D=D,C=C,s=sd,K=K,validation=V,observed_r=obs,truth=truth,edges_GeV=old['edges_GeV'])
  aggregate=dict(dataset=year,n_coordinates=len(masses),new_coordinates=len(masses)-len(oldm),max_abs_offset=float(np.max(abs(a))),median_abs_offset=float(np.median(abs(a))),rms_offset=float(np.sqrt(np.mean(a*a))),response_sd_min=float(sd.min()),response_sd_max=float(sd.max()),mean_abs_centered_toy_bias=float(np.mean(abs(Z.mean(axis=0)))),mean_abs_sd_error=float(np.mean(abs(Z.std(axis=0,ddof=1)-1))),corr_RMSE=float(np.sqrt(np.mean((np.corrcoef(V,rowvar=False)-K)**2))),min_K_eigenvalue=float(np.linalg.eigvalsh(K).min()),holdout_coverage_average=float(np.mean([r['holdout_absZ_lt1p96'] for r in detail])))
  aggregates.append(aggregate);pd.DataFrame(rows).to_csv(B/'local_mapping.csv',index=False,float_format='%.17g');write(B/'dataset_summary.json',aggregates);write(B/'input_manifest.json',INPUTS)
  print('COMPLETE',year,aggregate,flush=True)
 write(B/'execution.json',dict(complete=True,seconds=time.monotonic()-START,workers=1,BLAS_threads=1,new_coordinates=sum(r['new_coordinates'] for r in aggregates),scalar_checks=len(CHECKS),max_scalar_error=max(x['scalar_error'] for x in CHECKS),max_score=max(x['max_score'] for x in CHECKS),checks=CHECKS))
if __name__=='__main__':main()
