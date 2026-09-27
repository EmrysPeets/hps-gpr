#!/usr/bin/env python3
"""Statistics from frozen 2021 MC-generated/Gaussian-fit cohorts; no new fits."""
from pathlib import Path
import argparse,json,hashlib,warnings
import numpy as np
import pandas as pd
from scipy.stats import beta
B=Path(__file__).resolve().parents[1]
MASSES=list(range(60,241,20));POLICIES=['pole','logshift'];SOURCES=['nominal','functional'];LEVELS=[0,1,3,5]
GRID=[0,1,2,3,4,5,6,8,10,12,16,20,24];N=100;MASTER=63520260924

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
 if isinstance(v,np.generic):v=v.item()
 return None if isinstance(v,float) and not np.isfinite(v) else v

def boolean(x):
 y=x.astype(str).str.lower()
 if not y.isin(['true','false','1','0','1.0','0.0']).all():raise ValueError('Invalid boolean '+x.name)
 return y.isin(['true','1','1.0'])
def stat(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)];n=len(x);s=np.std(x,ddof=1) if n>1 else np.nan
 return dict(n=n,mean=np.mean(x) if n else np.nan,sd=s,se=s/np.sqrt(n) if n else np.nan)
def cp(k,n):return (0 if k==0 else beta.ppf(.025,k,n-k+1),1 if k==n else beta.ppf(.975,k+1,n-k)) if n else (np.nan,np.nan)
def boot(x,ix,kind='mean'):
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  return np.nanstd(np.asarray(x)[ix],axis=1,ddof=1) if kind=='sd' else np.nanmean(np.asarray(x)[ix],axis=1)
def interval(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return dict(se=np.std(x,ddof=1),ci95_lo=np.quantile(x,.025),ci95_hi=np.quantile(x,.975)) if len(x)>1 else dict(se=np.nan,ci95_lo=np.nan,ci95_hi=np.nan)
def moments(row,name,x,ix=None):
 row.update({name+'_'+k:v for k,v in stat(x).items()})
 if ix is not None:row.update({name+'_sd_bootstrap_'+k:v for k,v in interval(boot(x,ix,'sd')).items()})
def binary(row,name,x):
 x=np.asarray(x,float);x=x[np.isfinite(x)];n=len(x);k=int(x.sum());lo,hi=cp(k,n)
 row.update({name+'_k':k,name+'_n':n,name+'_fraction':k/n if n else np.nan,name+'_cp95_lo':lo,name+'_cp95_hi':hi,
             name+'_all_attempt_lo':k/N,name+'_all_attempt_hi':(k+N-n)/N})
def load(p,cohort):
 d=pd.read_csv(p)
 for c in ['fit_valid','profile_valid','native_cls90_valid']:
  if c in d:d[c]=boolean(d[c].fillna(False))
 for c in ['mass_MeV','toy','z']:d[c]=d[c].astype(int)
 keys=['source','policy','mass_MeV','shape','z','toy']
 if d.duplicated(keys).any():raise ValueError('Duplicate '+cohort+' result')
 expected={(s,p,m,'mc',z,t) for s in (['nominal'] if cohort=='pilot' else SOURCES) for p in POLICIES for m in MASSES for z in ([0] if cohort=='pilot' else GRID if cohort=='calibration' else LEVELS) for t in range(N)}
 if cohort=='evaluation':expected|={('nominal',p,m,'gaussian_log',z,t) for p in POLICIES for m in MASSES for z in [1,3,5] for t in range(N)}
 found=set(d[keys].itertuples(index=False,name=None))
 if expected!=found:raise ValueError(f'{cohort}: missing{len(expected-found)}, unexpected{len(found-expected)}')
 good=d.fit_valid
 if not np.isfinite(d.loc[good,['Ahat','sigma_postfit']]).all().all() or not d.loc[good,'sigma_postfit'].gt(0).all():raise ValueError('Invalid free fit marked valid')
 if cohort!='pilot':
  for key,q in d.groupby(['mass_MeV','z']):
   if q.A_expected.nunique()!=1 or q.s0.nunique()!=1 or not np.allclose(q.A_expected,q.z*q.s0):raise ValueError('Yield not common fixed z*s0')
 return d

def rank_values(y,grid,amplitudes):
 p=(1+(grid<=y).sum(axis=1))/(N+1);cls=np.minimum(1,p/p[0]);p0k=int((grid[0]>=y).sum());plo,phi=cp(p0k,N)
 out=dict(p0_k=p0k,p0_n=N,p0_rank=(1+p0k)/(N+1),p0_binomial_cp95_lo=plo,p0_binomial_cp95_hi=phi,
          p_grid_json=json.dumps(p.tolist()),cls_grid_json=json.dumps(cls.tolist()))
 for name,probs in [('rank',p),('toy_cls',cls)]:
  accept=probs>.1;idx=np.flatnonzero(accept);upper=amplitudes[idx[-1]] if len(idx) else 0.
  out.update({name+'_U_grid':upper,name+'_empty':len(idx)==0,name+'_right_censored':bool(accept[-1]),
     name+'_holey':bool(len(idx) and not accept[idx[0]:idx[-1]+1].all()),name+'_accepted_z':json.dumps([GRID[i] for i in idx])})
 return out,p,cls

def main(base=B,reps=2000,include_observed=True):
 r=base/'results';pilot=load(r/'pilot_rows.csv','pilot');cal=load(r/'calibration_rows.csv','calibration');ev=load(r/'evaluation_rows.csv','evaluation')
 ref=json.loads((base/'pilot_reference.json').read_text());assert (base/'calibration_freeze.json').is_file()
 for m in MASSES:
  q=pilot[pilot.mass_MeV.eq(m)&pilot.policy.eq('logshift')]
  assert len(q)==N and q.fit_valid.all() and np.isclose(q.sigma_postfit.mean(),ref['masses'][str(m)]['s0'])
 ix={s:np.stack([np.random.default_rng(np.random.SeedSequence([MASTER,4,si,j])).integers(0,N,N) for j in range(reps)]) for si,s in enumerate(SOURCES)}
 cells={k:q.set_index('toy').reindex(range(N)) for k,q in cal.groupby(['source','policy','mass_MeV','z'])}
 pars={};calrows=[];quant=[]
 for s in SOURCES:
  for p in POLICIES:
   for m in MASSES:
    n=cells[s,p,m,0];q=cells[s,p,m,3];y=n.Ahat.where(n.fit_valid).to_numpy();err=n.sigma_postfit.where(n.fit_valid).to_numpy();pull=y/err
    response=(q.Ahat.where(q.fit_valid).to_numpy()-y)/q.A_expected.iloc[0]
    row=dict(source=s,policy=p,mass_MeV=m,s0=float(q.s0.iloc[0]),n_null=int(n.fit_valid.sum()),n_response=int(np.isfinite(response).sum()),
       delta=stat(y)['mean'],delta_se=stat(y)['se'],mu0=stat(pull)['mean'],mu0_se=stat(pull)['se'],k0=stat((y-stat(y)['mean'])/err)['sd'],null_pull_sd=stat(pull)['sd'],R=stat(response)['mean'],R_se=stat(response)['se'],mean_sigma=stat(err)['mean'])
    for name,v in [('delta',y),('mu0',pull),('R',response)]:row.update({name+'_bootstrap_'+k:x for k,x in interval(boot(v,ix[s])).items()})
    db=boot(y,ix[s]);rb=boot(response,ix[s]);cov=np.cov(db,rb,ddof=1)
    row.update(delta_bootstrap_variance=cov[0,0],R_bootstrap_variance=cov[1,1],delta_R_bootstrap_covariance=cov[0,1])
    kb=np.nanstd((y[ix[s]]-db[:,None])/err[ix[s]],axis=1,ddof=1)
    row.update({'k0_bootstrap_'+k:x for k,x in interval(kb).items()})
    row['valid']=row['n_null']==N and row['n_response']==N and row['R']>0
    pars[s,p,m]=row;calrows.append(row)
    lo,hi=beta.ppf([.025,.975],10,91)
    for z in GRID:
     q=cells[s,p,m,z];ys=np.sort(q.loc[q.fit_valid,'Ahat'].to_numpy())
     quant.append(dict(source=s,policy=p,mass_MeV=m,z=z,A_expected=float(q.A_expected.iloc[0]),n=len(ys),critical90=ys[9] if len(ys)==N else np.nan,
                       acceptance_beta95_lo=1-hi,acceptance_beta95_hi=1-lo,rank_marginal_acceptance_lower=91/101))
 evalcells={k:q.set_index('toy').reindex(range(N)) for k,q in ev.groupby(['source','policy','mass_MeV','shape','z'])}
 summaries=[];arrays={};pointrows=[];limits=[]
 for key,q in evalcells.items():
  s,p,m,shape,z=key;null=evalcells[s,p,m,'mc',0];yn=null.Ahat.where(null.fit_valid).to_numpy();sn=null.sigma_postfit.where(null.fit_valid).to_numpy()
  y=q.Ahat.where(q.fit_valid).to_numpy();sig=q.sigma_postfit.where(q.fit_valid).to_numpy();A=float(q.A_expected.iloc[0]);s0=float(q.s0.iloc[0]);rawpull=(y-A)/sig
  for cs in [s]+(['nominal'] if s=='functional' else []):
   par=pars[cs,p,m];mu=par['mu0'];delta=par['delta'];R=par['R']
   values=dict(Ahat=y,bias=y-A,sigma=sig,error_ratio=sig/s0,pull=rawpull,centered_pull=rawpull-mu,
     offset_pull=(y-delta-A)/sig,response_residual_pull=(y-delta-R*A)/sig,
     meanpull_centered_yield=y-mu*sig,yield_offset_centered=y-delta)
   if shape=='mc':
    ac=(y-delta)/R
    vc=(par['k0']**2*sig**2+par['delta_bootstrap_variance']+ac**2*par['R_bootstrap_variance']+2*ac*par['delta_R_bootstrap_covariance'])/R**2
    values.update(affine_yield=ac,affine_sigma_approx=np.sqrt(np.maximum(0,vc)),affine_bias=ac-A,affine_pull_approx=(ac-A)/np.sqrt(np.maximum(0,vc)))
   else:values['response_residual_pull']=np.full(N,np.nan)
   if A>0:
    values.update(raw_recovery=y/A,paired_response=(y-yn)/A,meanpull_centered_response=((y-mu*sig)-(yn-mu*sn))/A)
   row=dict(source=s,calibration_source=cs,policy=p,mass_MeV=m,shape=shape,z=z,A_expected=A,s0=s0,attempted=N,fit_valid=int(q.fit_valid.sum()),profile_valid=int(q.profile_valid.sum()))
   for name,v in values.items():moments(row,name,v,ix[s] if name in ['pull','centered_pull','response_residual_pull'] else None)
   for name,thr in [('contain68',1.),('contain95',3.841459)]:binary(row,name,np.where(q.profile_valid,q.q_true<=thr,np.nan))
   summaries.append(row);arrays[s,cs,p,m,shape,z]=values
   if shape!='mc':continue
   grid=np.stack([cells[cs,p,m,g].Ahat.where(cells[cs,p,m,g].fit_valid).to_numpy() for g in GRID]);gridA=np.array([float(cells[cs,p,m,g].A_expected.iloc[0]) for g in GRID]);gridvalid=np.isfinite(grid).all()
   for t in range(N):
    valid=bool(np.isfinite(y[t]) and gridvalid)
    out=dict(source=s,calibration_source=cs,policy=p,mass_MeV=m,z=z,toy=t,A_expected=A,s0=s0,valid=valid)
    if valid:
     vals,pv,cv=rank_values(y[t],grid,gridA);out.update(vals)
     for method,prefix,probs in [('rank_neyman90','rank',pv),('rank_cls90','toy_cls',cv)]:
      limits.append(dict(source=s,calibration_source=cs,policy=p,mass_MeV=m,z=z,toy=t,method=method,valid=True,
         acceptance=float(probs[GRID.index(z)]>.1),upper_coverage=float(vals[prefix+'_U_grid']>=A-1e-8),
         U=vals[prefix+'_U_grid'],U_over_s0=vals[prefix+'_U_grid']/s0,empty=vals[prefix+'_empty'],right_censored=vals[prefix+'_right_censored'],holey=vals[prefix+'_holey']))
    else:
     for method in ['rank_neyman90','rank_cls90']:
      limits.append(dict(source=s,calibration_source=cs,policy=p,mass_MeV=m,z=z,toy=t,method=method,valid=False,acceptance=np.nan,upper_coverage=np.nan,U=np.nan,U_over_s0=np.nan,empty=False,right_censored=False,holey=False))
    pointrows.append(out)
  if shape=='mc':
   for t in range(N):
    valid=bool(q.native_cls90_valid.iloc[t]);u=float(q.native_cls90.iloc[t]) if valid else np.nan
    limits.append(dict(source=s,calibration_source='none',policy=p,mass_MeV=m,z=z,toy=t,method='native_cls90',valid=valid,
      acceptance=float(u>=A) if valid else np.nan,upper_coverage=float(u>=A) if valid else np.nan,U=u,U_over_s0=u/s0,empty=False,right_censored=False,holey=False))
 comparison=[]
 for s in SOURCES:
  shapes=['mc','gaussian_log'] if s=='nominal' else ['mc']
  for shape in shapes:
   for m in MASSES:
    for z in (LEVELS if shape=='mc' else [1,3,5]):
     a=arrays[s,s,'pole',m,shape,z];b=arrays[s,s,'logshift',m,shape,z]
     for metric in ['bias','pull','centered_pull','sigma']+(['raw_recovery','paired_response'] if z else []):
      v=b[metric]-a[metric];row=dict(source=s,shape=shape,mass_MeV=m,z=z,metric=metric,direction='logshift_minus_pole')
      row.update({'difference_'+k:x for k,x in stat(v).items()});row.update({'bootstrap_'+k:x for k,x in interval(boot(v,ix[s])).items()});comparison.append(row)
 ld=pd.DataFrame(limits);ls=[]
 for key,d in ld.groupby(['source','calibration_source','policy','mass_MeV','z','method']):
  out=dict(zip(['source','calibration_source','policy','mass_MeV','z','method'],key));out.update(attempted=len(d),valid=int(d.valid.sum()),empty_sets=int(d['empty'].sum()),right_censored=int(d.right_censored.sum()),holey_sets=int(d.holey.sum()),median_U_over_s0=d.U_over_s0.median())
  binary(out,'acceptance',d.acceptance);binary(out,'upper_coverage',d.upper_coverage);ls.append(out)
 outputs={'calibration_summary.csv':pd.DataFrame(calrows),'calibration_quantiles.csv':pd.DataFrame(quant),'heldout_summary.csv':pd.DataFrame(summaries),
          'policy_comparisons.csv':pd.DataFrame(comparison),'pointwise_rows.csv':pd.DataFrame(pointrows),'limit_rows.csv':ld,'limit_summary.csv':pd.DataFrame(ls)}
 fails=[]
 for name,d in [('pilot',pilot),('calibration',cal),('evaluation',ev)]:
  bad=~d.fit_valid
  if name=='evaluation':bad|=~d.profile_valid
  f=d[bad].copy();f['failure_stage']=name;fails.append(f)
 native=ev[ev['shape'].eq('mc')];f=native[~native.native_cls90_valid].copy();f['failure_stage']='native_cls90';fails.append(f)
 outputs['failure_ledger.csv']=pd.concat(fails,ignore_index=True)
 for name,d in outputs.items():d.to_csv(r/name,index=False,float_format='%.17g')
 # Dense observed scans are evaluated against empirical tables only at native calibration masses.
 op=r/'observed_dense.csv';observed=[]
 def branch(m):
  ratio=(105.6583745/np.asarray(m,float))**2
  return np.where(np.asarray(m)>211.316749,1+np.sqrt(np.maximum(0,1-4*ratio))*(1+2*ratio),1.)
 if include_observed and op.is_file():
  from observed import conversion
  od=pd.read_csv(op);od['visible_branch_factor']=branch(od.mass_MeV)
  od['epsilon2_90_ee_proxy']=od.epsilon2_90
  od['epsilon2_90_visible_legacy']=od.epsilon2_90*od.visible_branch_factor
  od.to_csv(r/'observed_display.csv',index=False,float_format='%.17g')
  for _,o in od[od.scope.eq('2021')].iterrows():
   m=float(o.mass_MeV);p=o.policy
   if m not in MASSES or p not in POLICIES:continue
   count_per_eps2=conversion('2021',m);y=float(o.psi_hat)*count_per_eps2*1e-8
   for s in SOURCES:
    grid=np.stack([cells[s,p,int(m),g].Ahat.where(cells[s,p,int(m),g].fit_valid).to_numpy() for g in GRID]);ga=np.array([float(cells[s,p,int(m),g].A_expected.iloc[0]) for g in GRID])
    if not np.isfinite(grid).all():continue
    vals,_,_=rank_values(y,grid,ga)
    for prefix in ['rank','toy_cls']:
     vals[prefix+'_epsilon2_ee_proxy']=vals[prefix+'_U_grid']/count_per_eps2
     vals[prefix+'_epsilon2_visible_legacy']=vals[prefix+'_epsilon2_ee_proxy']*float(branch(m))
    observed.append(dict(mass_MeV=int(m),policy=p,calibration_source=s,Ahat=y,s0=pars[s,p,int(m)]['s0'],p0_asymptotic=float(o.p0_asymptotic),**vals))
  pd.DataFrame(observed).to_csv(r/'observed_pointwise.csv',index=False,float_format='%.17g')
 jp=r/'combined_observed_rank.csv'
 if include_observed and jp.is_file():
  jd=pd.read_csv(jp);jd['visible_branch_factor']=branch(jd.mass_MeV)
  jd['epsilon2_90_grid_ee_proxy']=jd.epsilon2_90_grid
  jd['epsilon2_90_grid_visible_legacy']=jd.epsilon2_90_grid*jd.visible_branch_factor
  jd.to_csv(r/'combined_observed_rank_display.csv',index=False,float_format='%.17g')
 paths=[r/(x+'_rows.csv') for x in ['pilot','calibration','evaluation']]+[base/'pilot_reference.json',base/'calibration_freeze.json',Path(__file__).resolve()]
 if include_observed and op.is_file():paths.extend([op,base/'scripts/observed.py'])
 if include_observed and jp.is_file():paths.append(jp)
 meta=dict(schema_version=1,pilot_rows=len(pilot),calibration_rows=len(cal),evaluation_rows=len(ev),pilot_valid=int(pilot.fit_valid.sum()),calibration_valid=int(cal.fit_valid.sum()),evaluation_valid=int(ev.fit_valid.sum()),evaluation_profile_valid=int(ev.profile_valid.sum()),native_cls90_attempted=len(native),native_cls90_valid=int(native.native_cls90_valid.sum()),failure_rows=len(outputs['failure_ledger.csv']),observed_pointwise_rows=len(observed),bootstrap_replicates=reps,master_seed=MASTER,all_planned_ids_present=True,
  source_hashes={str(p.relative_to(base)):sha(p) for p in paths},definitions={'signal':'Native full-selected MC generated; Gaussian full-template fitted. Gaussian-log controls separate.',
   'logshift':'Fit and training-exclusion mask share log-displaced center; nominal resolution retained.',
   'mu0':'Calibration mean null Wald pull Ahat/sigma; mean-pull centering is diagnostic and differs from fixed yield offset.',
   'delta':'Calibration mean null fitted yield in full-selected candidate units.',
   'R':'Calibration paired MC response at z3; not tail acceptance or detector efficiency.',
   'rank':'Raw Ahat lower-tail p=(1+k)/101; reject<=0.10; grid endpoints only, right-censor at24s0.',
   'p0':'Raw Ahat upper-tail add-onep, with separate k/100 binomial95%CP interval; pointwise fixed source.',
   'bootstrap':'Whole IDs across masses/policies/levels; independent source streams; frozen pilot scale.',
   'coverage':'Rank size<=10/101 marginal over calibration and independent evaluation at fixed generator; frozen tables checked separately.'},
  limitations='Conditional fixed-source and fixed-empirical-template diagnostics, not global calibration or physical exclusion.')
 (r/'summary.json').write_text(json.dumps(clean(meta),indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in meta.items() if k.endswith('_rows') or k.endswith('_valid')}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=B);p.add_argument('--bootstrap-replicates',type=int,default=2000);p.add_argument('--skip-observed',action='store_true');a=p.parse_args();main(a.base.resolve(),a.bootstrap_replicates,not a.skip_observed)
