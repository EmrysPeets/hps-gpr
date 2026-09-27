"""Fixed HPS likelihood and analytic Poisson-response propagation."""
from pathlib import Path
import os,sys,json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular,block_diag
import common as C
from fast_profile import FastBatchProfile
B=Path(__file__).resolve().parents[1];P=json.loads((B/'protocol.json').read_text());YEARS=['2015','2016','2021']
OFFSET={};total=0
for y in YEARS:OFFSET[y]=total;total+=len(C.DATA[y]['n'])
TOTAL=total
class Context:
 def __init__(self,year,mass,width=2.25):
  d=C.DATA[year];self.year=year;self.mass=mass;self.x=d['x'];self.n=d['n'];sig=C.sigma(year,mass)
  self.mask=(self.x>=mass/1000-width*sig)&(self.x<=mass/1000+width*sig);self.keep=~self.mask
  self.const,self.ls=C.kernel_state(year,mass);xt=self.x[self.keep];xq=self.x[self.mask]
  self.K=C.kernel(xt,xt,self.const,self.ls);self.Kqt=C.kernel(xq,xt,self.const,self.ls);self.Kqq=C.kernel(xq,xq,self.const,self.ls)
  self.S=C.signal(year,mass,self.mask)
 def predict(self,n,derivatives=False):
  y=n[self.keep];positive=y>0;target=np.zeros_like(y);target[positive]=np.log(y[positive]);alpha=np.ones_like(y);alpha[positive]=1/y[positive]
  M=self.K.copy();M.flat[::len(M)+1]+=alpha;fac=cho_factor_local(M);co=cho_solve(fac,target,check_finite=False)
  v=solve_triangular(fac[0],self.Kqt.T,lower=True,check_finite=False);cl=self.Kqq-v.T@v;cl=.5*(cl+cl.T)
  mu=self.Kqt@co;b=np.exp(mu+.5*np.maximum(np.diag(cl),0));raw=np.outer(b,b)*np.expm1(np.clip(cl,-40,40));L,diag=C.factor_cov(raw,b)
  p={'year':self.year,'n':n[self.mask],'b':b,'L':L,'Craw':raw,'S':self.S,'context':self,'load':diag['load']}
  if derivatives:
   assert positive.all();H=cho_solve(fac,self.Kqt.T,check_finite=False).T
   logder=H*(1/y+co/y**2)[None,:]-.5*H**2/y[None,:]**2
   p.update(H=H,logder=logder,db=b[:,None]*logder,expcl=np.exp(np.clip(cl,-40,40)),training_counts=y)
  return p

def cho_factor_local(M):return (cholesky(M,lower=True,check_finite=False),True)
def fit(parts):
 b=np.concatenate([p['b'] for p in parts]);L=block_diag(*[p['L'] for p in parts]);S=np.concatenate([p['S'] for p in parts]);n=np.concatenate([p['n'] for p in parts]);model=C.OneSignalProfile(b,L,S)
 f=model.fit(n);z=model.fit(n,0.);r=float(np.sign(f['A'])*np.sqrt(max(0,2*(z['nll']-f['nll']))))
 assert max(f['score'],z['score'])<2e-7,(r,f['score'],z['score'])
 return dict(r=r,f=f,z=z,model=model,n=n,b=b,L=L,S=S,score=max(f['score'],z['score']))

def response(parts,result,truths):
 """Envelope derivative in count-space; GP covariance derivative included."""
 D=np.zeros(TOTAL);r=result['r'];off=0
 if abs(r)<1e-4:
  V=np.diag(result['b'])+result['L']@result['L'].T
  v=cho_solve(cho_factor_local(V),result['S']);weight=v/np.sqrt(result['S']@v)
 for p in parts:
  n=len(p['b']);sl=slice(off,off+n);off+=n;ctx=p['context'];year=p['year'];truth=truths[year]
  if abs(r)<1e-4:
   dw=weight[sl];dt=-dw@p['db']
  else:
   lam0=result['z']['lam'][sl];lam1=result['f']['lam'][sl]
   g0=(lam0-p['n'])/lam0;g1=(lam1-p['n'])/lam1
   dw=np.log(lam1/lam0)/r
   def covariance_contraction(g):
    first=2*(g*(p['Craw']@g))@p['logder'];bg=p['b']*g;M=np.outer(bg,bg)*p['expcl']
    return first-np.sum(p['H']*(M@p['H']),axis=0)/p['training_counts']**2
   dt=((g0-g1)@p['db']-.5*(covariance_contraction(g0)-covariance_contraction(g1)))/r
  grad=np.zeros(len(truth));grad[ctx.mask]=dw;grad[ctx.keep]=dt;D[OFFSET[year]:OFFSET[year]+len(truth)]=grad*np.sqrt(truth)
 return D

def batch_fit(part_banks):
 nt=len(part_banks[0]);bs=[];ns=[];ls=[];ss=[];blocks=[];row=col=0
 for bank in part_banks:
  nb=len(bank[0]['b']);rank=max(p['L'].shape[1] for p in bank);L=np.zeros((nt,nb,rank))
  for i,p in enumerate(bank):L[i,:,:p['L'].shape[1]]=p['L']
  bs.append(np.array([p['b'] for p in bank]));ns.append(np.array([p['n'] for p in bank]));ls.append(L);ss.append(bank[0]['S']);blocks.append((row,row+nb,col,col+rank));row+=nb;col+=rank
 factors=np.zeros((nt,row,col))
 for L,(r0,r1,c0,c1) in zip(ls,blocks):factors[:,r0:r1,c0:c1]=L
 S=np.concatenate(ss);S=S/S.sum();model=FastBatchProfile(np.concatenate(ns,axis=1),np.concatenate(bs,axis=1),factors,S,blocks=blocks)
 check=fit([bank[0] for bank in part_banks]);err=float(abs(check['r']-model.r[0]));assert err<2e-5,err
 return model.r,dict(scalar_error=err,max_score=model.max_score,fallbacks=model.fallbacks)
