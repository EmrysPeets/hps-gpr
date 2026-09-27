"""Deterministic common-region Poisson fits. This module never draws random numbers."""
from pathlib import Path
import os,sys,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v520-mpl')
import numpy as np,pandas as pd
from scipy.linalg import cholesky,cho_solve,solve_triangular,block_diag
from scipy.stats import chi2
B=Path(__file__).resolve().parents[1]
SCOPES=json.loads((B/'inputs/scopes.json').read_text())
DATA={y:dict(np.load(B/f'inputs/spectrum_{y}.npz')) for y in ('2015','2016','2021')}
for d in DATA.values(): d['idx']={int(m):i for i,m in enumerate(d['masses'])}
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def kernel(x,z,const,ls):return const*np.exp(-.5*((np.log(x)[:,None]-np.log(z)[None,:])/ls)**2)
def predict(x,n,mask,const,ls,query=None):
 """Exact fixed log-GP: alpha=1/n, log targets, unnormalized zero mean."""
 xt=x[~mask];yt=n[~mask];xq=x[mask] if query is None else query
 if np.any(yt<=0):raise ValueError('This pinned high-count lane requires positive training bins')
 K=kernel(xt,xt,const,ls);K.flat[::len(K)+1]+=1/yt
 L=cholesky(K,lower=True,check_finite=False);Kqt=kernel(xq,xt,const,ls)
 latent=Kqt@cho_solve((L,True),np.log(yt),check_finite=False)
 v=solve_triangular(L,Kqt.T,lower=True,check_finite=False)
 C=kernel(xq,xq,const,ls)-v.T@v;C=.5*(C+C.T)
 b=np.exp(latent+.5*np.maximum(np.diag(C),0))
 return b,np.outer(b,b)*np.expm1(np.clip(C,-40,40))
def factor_cov(C,b,full=False):
 C=.5*(C+C.T);scale=max(float(np.diag(C).max()),1.)
 load=None
 for r in (1e-10,1e-9,1e-8,1e-7,1e-6,1e-5):
  try:cholesky(C+r*scale*np.eye(len(C)),lower=True);load=r;break
  except np.linalg.LinAlgError:pass
 if load is None:raise RuntimeError('Covariance loading exceeds 1e-5')
 C=C+load*scale*np.eye(len(C));sd=np.sqrt(b)
 vals,U=np.linalg.eigh(C/sd[:,None]/sd[None,:]);keep=vals>1e-8 if not full else vals>0
 L=sd[:,None]*U[:,keep]*np.sqrt(vals[keep])
 return L,dict(load=load,rank=int(keep.sum()),bins=len(b),max_omitted=float(max(vals[~keep],default=0.)))
def context(year,m1,m2,padding=3.,full=False,truth='observed',region=None,anchor=None):
 d=DATA[year];i=d['idx'][int(m1)];j=d['idx'][int(m2)]
 center=int(np.floor((m1+m2)/2)) if anchor is None else int(anchor)
 ci=d['idx'][center];x=d['x'];n=d['n'] if truth=='observed' else d['stress']
 lo=m1/1000-padding*d['sigma'][i];hi=m2/1000+padding*d['sigma'][j]
 if region is not None:lo,hi=region
 mask=(x>=lo)&(x<=hi)
 if np.sum(x<lo)<3 or np.sum(x>hi)<3:raise ValueError('Fewer than three bins in an external sideband')
 b,C=predict(x,n,mask,d['const'][ci],d['ls'][ci]);L,diagnostic=factor_cov(C,b,full)
 S=np.column_stack([d['templates'][i,mask]*d['conversion'][i]*1e-8,d['templates'][j,mask]*d['conversion'][j]*1e-8])
 return dict(year=year,m1=m1,m2=m2,anchor=center,lo=lo,hi=hi,mask=mask,n=n[mask],b=b,C=C,L=L,S=S,
  conversion=d['conversion'][[i,j]],template_fraction=d['templates'][[i,j]][:,mask].sum(axis=1),diagnostic=diagnostic)
class Model:
 def __init__(self,parts):
  self.parts=parts;self.n=np.concatenate([p['n'] for p in parts]);self.b=np.concatenate([p['b'] for p in parts]);self.L=block_diag(*[p['L'] for p in parts]);self.S=np.vstack([p['S'] for p in parts])
  self.rank=self.L.shape[1];self.scale=1/np.sqrt(np.sum(self.S**2/self.b[:,None],axis=0));self.T=self.S*self.scale
 def fit(self,active=(),fixed=None,initial=None):
  fixed=np.zeros(2) if fixed is None else np.asarray(fixed,float)
  active=list(active);base=self.b+self.S@fixed
  J=np.column_stack((self.T[:,active],self.L));k=len(active);size=J.shape[1]
  z=np.zeros(size) if initial is None else initial.copy();pen=np.r_[np.zeros(k),np.ones(self.rank)]
  def eval(v,hessian=True):
   lam=base+J@v
   if np.any(lam<=0):return np.inf,None,None,lam
   t=(lam-self.n)/self.n
   value=float(np.sum(self.n*(t-np.log1p(t)))+.5*np.sum(pen*v*v))
   g=J.T@((lam-self.n)/lam)+pen*v
   H=(J.T*(self.n/lam**2))@J+np.diag(pen) if hessian else None
   return value,g,H,lam
  for iteration in range(100):
   value,g,H,lam=eval(z);score=float(np.max(abs(g))) if size else 0.
   if score<2e-7:break
   step=np.linalg.solve(H,-g);descent=g@step
   if descent>=0:raise RuntimeError('Non-descent Newton direction')
   a=1.
   for _ in range(50):
    v2=eval(z+a*step,False)[0]
    if v2<=value+1e-4*a*descent+1e-11:z+=a*step;break
    a*=.5
   else:raise RuntimeError('Line search failed')
  else:raise RuntimeError(f'Fit convergence failed {score}')
  amp=fixed.copy();amp[active]=z[:k]*self.scale[active]
  theta=z[k:];bfit=self.b+self.L@theta
  covariance=None
  if k==2:covariance=np.linalg.inv(H)[:2,:2]*np.outer(self.scale,self.scale)
  return dict(nll=value,amp=amp,theta=theta,bfit=bfit,lam=lam,score=score,iterations=iteration,covariance=covariance)
 def hypotheses(self):
  f0=self.fit();fl=self.fit([0]);fh=self.fit([1]);ff=self.fit([0,1])
  hl=fl if fl['amp'][0]>=0 else f0;hh=fh if fh['amp'][1]>=0 else f0
  feasible=[f0,hl,hh]+([ff] if np.min(ff['amp'])>=0 else [])
  f2=min(feasible,key=lambda f:f['nll'])
  if min(hl['nll'],hh['nll'])<f2['nll']-1e-8:raise RuntimeError('Nesting failure')
  # Efficient Fisher information at fitted H0; auxiliary Gaussian constraint included.
  lam=f0['lam'];I=self.S.T@(self.S/lam[:,None]);cross=self.S.T@(self.L/lam[:,None]);H=self.L.T@(self.L/lam[:,None])+np.eye(self.rank)
  I-=cross@np.linalg.solve(H,cross.T);I=.5*(I+I.T)
  rho=float(np.clip(I[0,1]/np.sqrt(I[0,0]*I[1,1]),-1,1))
  Q2=max(0.,2*(f0['nll']-f2['nll']));qadd=max(0.,2*(min(hl['nll'],hh['nll'])-f2['nll']))
  w2=float(np.arccos(rho)/(2*np.pi));p2=float(.5*chi2.sf(Q2,1)+w2*chi2.sf(Q2,2)) if Q2>1e-10 else 1.
  cov=ff['covariance'];rhoamp=float(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]))
  return dict(H0=f0,HL=hl,HH=hh,HLH=f2,free=ff),dict(Q2=Q2,q_add=qadd,p2_fixed_pair_chibar=p2,p_add_Gaussian_bound=float(np.exp(-qadd/2)),
   psi1_1e8=float(f2['amp'][0]),psi2_1e8=float(f2['amp'][1]),unbounded_psi1_1e8=float(ff['amp'][0]),unbounded_psi2_1e8=float(ff['amp'][1]),
   sigma1_1e8=float(np.sqrt(cov[0,0])),sigma2_1e8=float(np.sqrt(cov[1,1])),rho_score=rho,rho_amplitude=rhoamp,information_condition=float(np.linalg.cond(I)),
   chi_bar_w2=w2,chi_bar_w0=.5-w2,max_score=max(f['score'] for f in (f0,fl,fh,ff)),min_lambda=min(float(f['lam'].min()) for f in (f0,hl,hh,f2)),
   min_background=float(f2['bfit'].min()),n_bins=len(self.n),nuisance_rank=self.rank,
   nll_0=f0['nll'],nll_L=hl['nll'],nll_H=hh['nll'],nll_LH=f2['nll'])
def rowfit(scope,m1,m2,**kw):
 parts=[context(y,m1,m2,**kw) for y in scope[2]];model=Model(parts);fits,row=model.hypotheses()
 row.update(scope=scope[0],label=scope[1],m1=int(m1),m2=int(m2),midpoint=(m1+m2)/2,separation=m2-m1,
  maximum_covariance_load=max(p['diagnostic']['load'] for p in parts),max_omitted_mode=max(p['diagnostic']['max_omitted'] for p in parts))
 return row,model,fits

def continuous_signal(year,mass_MeV,mask=None):
 """Physical resolution and exact fractional native-bin density at continuous mass."""
 from scipy.special import ndtr
 d=DATA[year];m=mass_MeV/1000;sig=float(np.polynomial.polynomial.polyval(m,d['sigma_coeffs']))
 edges=d['native_edges'];width=np.diff(edges);lo=m-1.64*sig;hi=m+1.64*sig
 overlap=np.maximum(0.,np.minimum(edges[1:],hi)-np.maximum(edges[:-1],lo))
 density=float(np.sum(d['native_counts']*overlap/width)/(hi-lo))
 conversion=3*np.pi*m*float(d['frad_effective'])*density/(2/137.)
 w=np.diff(ndtr((d['edges']-m)/sig));w/=w.sum()
 if mask is not None:w=w[mask]
 return w*conversion*1e-8

def single_fit(model,mass):
 import copy
 S=np.concatenate([continuous_signal(p['year'],mass,p['mask']) for p in model.parts])
 mod=copy.copy(model);mod.S=np.column_stack([S,S]);mod.scale=1/np.sqrt(np.sum(mod.S**2/mod.b[:,None],axis=0));mod.T=mod.S*mod.scale
 f=mod.fit([0])
 return f if f['amp'][0]>=0 else mod.fit()
