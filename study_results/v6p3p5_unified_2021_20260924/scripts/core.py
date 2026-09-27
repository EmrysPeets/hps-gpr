"""2021 native-MC generation with Gaussian extraction; no extra guard."""
from pathlib import Path
import os,sys,json,hashlib,datetime,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v635-mpl');sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'inputs/v6p1/scripts'))
import common as C
import core_centering as MC
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.linalg import cholesky,cho_solve,solve_triangular
MASTER=63520260924
MASSES=tuple(range(60,241,20));POLICIES=('pole','logshift');SOURCES=('nominal','functional')
GRID=(0,1,2,3,4,5,6,8,10,12,16,20,24);LEVELS=(0,1,3,5);NTOYS=100
D=C.DATA['2021'];NULL=dict(np.load(B/'inputs/null_2021.npz'))
TRUTHS={'nominal':NULL['truth'],'functional':D['stress']}
LOG=json.loads((B/'inputs/v6p1/derived/analytic_shift_models.json').read_text())
COEF=next(m['coefficients'] for m in LOG['models'] if m['model']=='logarithmic')
assert np.array_equal(D['n'],NULL['observed']) and np.array_equal(D['edges'],NULL['edges_GeV'])
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
array_hash=ahash
def write_json(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rng(key):return np.random.default_rng(np.random.SeedSequence(key))
def center(m):
 assert 60<=m<=240,'No log-law extrapolation'
 return float(m+COEF[0]+COEF[1]*np.log(m/150.))
def mc_categories(m):
 cdf=MC.cumulative(m,D['edges']);a=np.r_[cdf[0],np.diff(cdf),1-cdf[-1]]
 assert a.min()>=0 and abs(a.sum()-1)<1e-12
 return a
def gaussian_categories(m):
 cdf=ndtr((D['edges']-center(m)/1000)/C.sigma('2021',m));return np.r_[cdf[0],np.diff(cdf),1-cdf[-1]]
def background_key(cohort,source,toy):return [MASTER,{'pilot':1,'calibration':2,'evaluation':3}[cohort],SOURCES.index(source),int(toy)]
def signal_key(cohort,source,m,toy,z,shape):return [MASTER,{'calibration':20,'evaluation':30,'smoke':90}[cohort],SOURCES.index(source),int(m),int(toy),int(z),0 if shape=='mc' else 1]
def signature():
 parts=[]
 for row in json.loads((B/'provenance/input_hashes.json').read_text()):
  assert sha(B/row['path'])==row['sha256'],'Pinned input changed';parts.append((row['path'],row['sha256']))
 for rel in ('scripts/core.py','scripts/run_study.py','protocol.json','inputs/cohorts.npz','inputs/templates.npz'):parts.append((rel,sha(B/rel)))
 return hashlib.sha256(json.dumps(sorted(parts)).encode()).hexdigest()

class Context:
 def __init__(self,m,policy,year='2021'):
  self.m=m;self.policy=policy;self.year=year;self.D=C.DATA[year];self.sigma=C.sigma(year,m)
  self.center=center(m) if year=='2021' and policy=='logshift' else float(m)
  self.lo=self.center/1000-2.25*self.sigma;self.hi=self.center/1000+2.25*self.sigma
  x=self.D['x'];self.mask=(x>=self.lo)&(x<=self.hi);self.guard=self.mask
  assert (x<self.lo).sum()>=3 and (x>self.hi).sum()>=3
  self.const,self.ls=C.kernel_state(year,m)
  self.K=C.kernel(x[~self.mask],x[~self.mask],self.const,self.ls)
  self.Kqt=C.kernel(x[self.mask],x[~self.mask],self.const,self.ls)
  self.Kqq=C.kernel(x[self.mask],x[self.mask],self.const,self.ls)
  cdf=ndtr((self.D['edges']-self.center/1000)/self.sigma)
  self.categories=np.r_[cdf[0],np.diff(cdf),1-cdf[-1]];self.p=self.categories[1:-1]
 def predict(self,counts):
  n=np.asarray(counts,float)[~self.mask];pos=n>0;target=np.zeros_like(n);target[pos]=np.log(n[pos]);alpha=np.ones_like(n);alpha[pos]=1/n[pos]
  K=self.K.copy();K.flat[::len(K)+1]+=alpha;L=cholesky(K,lower=True,check_finite=False)
  mu=self.Kqt@cho_solve((L,True),target,check_finite=False);v=solve_triangular(L,self.Kqt.T,lower=True,check_finite=False)
  cov=self.Kqq-v.T@v;cov=.5*(cov+cov.T);b=np.exp(mu+.5*np.maximum(np.diag(cov),0))
  covariance=np.outer(b,b)*np.expm1(np.clip(cov,-40,40));factor,diag=C.factor_cov(covariance,b)
  return b,factor,diag

def checked_fit(model,n,fixed=None,initial=None):
 fit=model.fit(n,fixed=fixed,initial=initial)
 assert np.isfinite(fit['nll']) and np.isfinite(fit['score']) and fit['score']<3e-5 and fit['min_lambda']>0
 if fixed is None:
  _,_,H,_=model._objective(fit['z'],n,model.Jfree,model.b,model.penfree)
  unit=np.zeros(len(H));unit[0]=1;variance=float(cho_solve((cholesky(H,lower=True),True),unit)[0]);assert variance>0
  sigma=float(model.scale*np.sqrt(variance));assert np.isfinite(sigma) and sigma>0 and abs(fit['sigma']/sigma-1)<1e-10;fit['sigma']=sigma
 return fit

def fit_pair(b,L,p,n,expected,need_profile):
 attempts=[];frees=[];profiles=[]
 for attempt in range(3):
  tolerance=2e-7 if attempt==0 else 2e-9;model=C.OneSignalProfile(b,L,p,score_tolerance=tolerance)
  initial=None if attempt<2 else np.r_[.5,np.zeros(model.rank)]
  if attempt>0 and profiles:initial=np.r_[expected/model.scale,min(profiles,key=lambda q:q['nll'])['theta']]
  rec=dict(attempt=attempt,tolerance=tolerance);free=None
  try:
   free=checked_fit(model,n,initial=initial);frees.append(free);rec.update(free_valid=True,nll=free['nll'],score=free['score'])
  except Exception as e:rec.update(free_valid=False,error=str(e))
  if free is not None and need_profile:
   try:
    fixed=checked_fit(model,n,fixed=float(expected),initial=free['theta'] if attempt!=1 else np.zeros(model.rank));profiles.append(fixed);rec.update(profile_valid=True,true_nll=fixed['nll'])
   except Exception as e:rec.update(profile_valid=False,profile_error=str(e))
  attempts.append(rec)
  if frees and (not need_profile or (profiles and 2*(min(f['nll'] for f in profiles)-min(f['nll'] for f in frees))>=-2e-6)):break
 free=min(frees,key=lambda f:f['nll']) if frees else None;fixed=min(profiles,key=lambda f:f['nll']) if profiles else None
 valid=bool(free is not None and fixed is not None and 2*(fixed['nll']-free['nll'])>=-2e-6)
 return free,fixed,valid,attempts

def row(ctx,background,draw,A,toy,source,z,cohort,shape,s0=None,prediction=None):
 counts=background+draw[1:-1]
 out=dict(cohort=cohort,source=source,policy=ctx.policy,mass_MeV=ctx.m,toy=toy,z=z,shape=shape,A_expected=float(A),s0=s0,
  Ahat=None,sigma_postfit=None,pull=None,q_true=None,profile_contains68=None,profile_contains95=None,
  fit_valid=False,profile_valid=False,failure_reason='',native_cls90_valid=False,native_cls90=None,native_cls90_failure='',
  actual_full=int(draw.sum()),actual_support=int(draw[1:-1].sum()),actual_window=int(draw[1:-1][ctx.mask].sum()),actual_training=int(draw[1:-1][~ctx.mask].sum()),
  actual_outside_support=int(draw[0]+draw[-1]),background_hash=ahash(background),signal_hash=ahash(draw),counts_hash=ahash(counts),
  fit_mask_hash=ahash(ctx.mask),fit_template_hash=ahash(ctx.categories),center_MeV=ctx.center,nominal_sigma_MeV=ctx.sigma*1000,
  fit_bins=int(ctx.mask.sum()),kernel_const=ctx.const,kernel_ls=ctx.ls,sigma_method='observed_profile_hessian')
 try:
  b,L,diag=ctx.predict(counts) if prediction is None else prediction
  out.update(gp_mean_hash=ahash(b),gp_covariance_hash=ahash(L),nuisance_rank=int(diag['rank']),covariance_load=float(diag['load']))
  free,fixed,valid,attempts=fit_pair(b,L,ctx.p[ctx.mask],counts[ctx.mask],A,cohort in ('evaluation','smoke'))
  out.update(attempts_json=json.dumps(attempts),attempt_count=len(attempts),fit_valid=free is not None,profile_valid=valid)
  if free is not None:out.update(Ahat=free['A'],sigma_postfit=free['sigma'],pull=(free['A']-A)/free['sigma'],free_nll=free['nll'],fit_score=free['score'],min_lambda=free['min_lambda'])
  else:out['failure_reason']='free_fit_failed'
  if fixed is not None:out.update(true_nll=fixed['nll'],profile_score=fixed['score'])
  if valid:
   q=max(0.,2*(fixed['nll']-free['nll']));out.update(q_true=q,profile_contains68=q<=1,profile_contains95=q<=3.841459)
  elif cohort in ('evaluation','smoke'):out['failure_reason']+=';profile_failed'
  if cohort in ('evaluation','smoke') and shape=='mc' and free is not None:
   try:
    ul=C.OneSignalProfile(b,L,ctx.p[ctx.mask]).limit(counts[ctx.mask]);assert ul['ok'] and ul['max_score']<3e-5 and ul['min_lambda']>0 and abs(ul['cls']-.1)<2e-6
    out.update(native_cls90=ul['A90'],native_cls90_valid=True,native_cls90_contains=bool(ul['A90']>=A),signed_r=ul['signed_r'],p0_asymptotic=ul['p0_fixed_mass'])
   except Exception as e:out['native_cls90_failure']=str(e)
 except Exception as e:out['failure_reason']=type(e).__name__+':'+str(e)
 return out

def chunk_base(cohort,source,m,start):return B/f'results/{cohort}/{source}_m{m:03d}_t{start:03d}'
def run_chunk(cohort,source,m,start,stop,sig,reference_sha=None,calibration_sha=None):
 assert signature()==sig
 if reference_sha:assert sha(B/'pilot_reference.json')==reference_sha
 if calibration_sha:assert sha(B/'calibration_freeze.json')==calibration_sha
 base=chunk_base(cohort,source,m,start);marker=base.with_suffix('.json')
 if marker.exists():
  old=json.loads(marker.read_text());assert old['complete'] and old['signature']==sig and old['reference_sha256']==reference_sha and old['calibration_sha256']==calibration_sha
  assert all(sha(B/p)==h for p,h in old['output_hashes'].items());return dict(cohort=cohort,source=source,mass=m,start=start,cached=True)
 began=utc();timer=time.monotonic();contexts=[Context(m,p) for p in POLICIES];backgrounds=np.load(B/'inputs/cohorts.npz')[cohort+'_'+source]
 ref=json.loads((B/'pilot_reference.json').read_text()) if reference_sha else None;s0=ref['masses'][str(m)]['s0'] if ref else None
 cats={'mc':mc_categories(m),'gaussian_log':gaussian_categories(m)};rows=[];draws=[];keys=[]
 for toy in range(start,stop):
  if (B/'STOP').exists():raise RuntimeError('STOP marker; checkpoint retained')
  settings=[('mc',z) for z in ((0,) if cohort=='pilot' else GRID if cohort=='calibration' else LEVELS)]
  if cohort=='evaluation' and source=='nominal':settings += [('gaussian_log',z) for z in (1,3,5)]
  for shape,z in settings:
   A=0. if z==0 else float(z*s0)
   draw=np.zeros(len(D['n'])+2,dtype=np.int64) if z==0 else rng(signal_key(cohort,source,m,toy,z,shape)).poisson(A*cats[shape])
   for ctx in contexts:rows.append(row(ctx,backgrounds[toy],draw,A,toy,source,z,cohort,shape,s0))
   if cohort!='pilot':draws.append(draw);keys.append((toy,z,0 if shape=='mc' else 1))
 out=base.with_suffix('.csv');tmp=out.with_suffix('.csv.tmp');pd.DataFrame(rows).to_csv(tmp,index=False,float_format='%.17g');tmp.replace(out);paths=[out]
 if cohort!='pilot':
  p=base.with_suffix('.npz');np.savez_compressed(p,draws=np.array(draws),keys=np.array(keys));paths.append(p)
 write_json(marker,dict(complete=True,started_utc=began,completed_utc=utc(),signature=sig,reference_sha256=reference_sha,calibration_sha256=calibration_sha,
  cohort=cohort,source=source,mass_MeV=m,start=start,stop=stop,rows=len(rows),output_hashes={str(p.relative_to(B)):sha(p) for p in paths}))
 return dict(cohort=cohort,source=source,mass=m,start=start,cached=False,rows=len(rows),seconds=round(time.monotonic()-timer,2))
