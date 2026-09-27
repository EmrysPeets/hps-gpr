"""C=0 Poisson signal fit, with exact retrained parent GP count means.

The GP mean retains exp(mu + diag(C_log)/2). Its covariance is omitted ONLY
inside the signal likelihood. No source centering or empirical rescaling.
"""
from pathlib import Path
import os,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
PARENT=BASE/'inputs/parent'
sys.path.insert(0,str(PARENT/'scripts'))
import common as C
import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular
from scipy.optimize import brentq
from scipy.stats import norm,chi2,beta
from math import comb

YEARS=('2015','2016','2021')
LIMITS={'2015':(19.,100.),'2016':(39.,180.),'2021':(50.,250.)}

def check_stop():
    if (BASE/'STOP').exists():raise SystemExit('STOP resource marker')

class Context:
    def __init__(self,year,mass):
        self.year=year;self.mass=float(mass)
        d=C.DATA[year];self.x=d['x'];sig=C.sigma(year,mass)
        self.mask=(self.x>=mass/1000-2.25*sig)&(self.x<=mass/1000+2.25*sig)
        self.keep=~self.mask
        self.const,self.ls=C.kernel_state(year,mass)
        xt=self.x[self.keep];xq=self.x[self.mask]
        self.K=C.kernel(xt,xt,self.const,self.ls)
        self.Kqt=C.kernel(xq,xt,self.const,self.ls)
        self.S=C.signal(year,mass,self.mask)
        self.Sfull=C.signal(year,mass)
    def predict(self,counts):
        y=np.asarray(counts,float)[self.keep];pos=y>0
        target=np.zeros_like(y);target[pos]=np.log(y[pos])
        alpha=np.ones_like(y);alpha[pos]=1/y[pos]
        M=self.K.copy();M.flat[::len(M)+1]+=alpha
        L=cholesky(M,lower=True,check_finite=False)
        co=cho_solve((L,True),target,check_finite=False)
        v=solve_triangular(L,self.Kqt.T,lower=True,check_finite=False)
        latent_var=np.maximum(self.const-np.einsum('ij,ij->j',v,v),0)
        b=np.exp(self.Kqt@co+.5*latent_var)
        assert np.all(np.isfinite(b)) and np.all(b>0)
        return b

def fit(n,b,S):
    """Simultaneous independent scalar solves; S is the original common unit."""
    n=np.atleast_2d(n).astype(float);b=np.atleast_2d(b).astype(float);S=np.asarray(S,float)
    assert n.shape==b.shape and len(S)==n.shape[1] and np.all(b>0) and np.all(S>=0)
    sigma0=1/np.sqrt(np.sum(S[None,:]**2/b,axis=1))
    T=S[None,:]*sigma0[:,None]
    pos=S>0
    lo=np.max(-b[:,pos]/T[:,pos],axis=1)*(1-1e-12)
    hi=np.maximum(np.sum(n,axis=1)/np.sum(T,axis=1),1.)
    z=np.zeros(len(n));done=np.zeros(len(n),bool)
    for iteration in range(80):
        lam=b+z[:,None]*T
        assert np.all(lam>0)
        g=np.sum(T*(n/lam-1),axis=1)
        h=np.sum(n*T*T/lam**2,axis=1)
        done |= np.abs(g)<2e-8
        if np.all(done):break
        lo=np.where((g>0)&~done,z,lo);hi=np.where((g<=0)&~done,z,hi)
        candidate=z+g/h
        bad=(candidate<=lo)|(candidate>=hi)|~np.isfinite(candidate)
        candidate=np.where(bad,(lo+hi)/2,candidate)
        z=np.where(done,z,candidate)
    else:
        for k in np.flatnonzero(~done):
            fun=lambda v:np.sum(T[k]*(n[k]/(b[k]+v*T[k])-1))
            z[k]=brentq(fun,lo[k],hi[k],xtol=1e-12)
    A=z*sigma0;delta=A[:,None]*S
    lam=b+delta
    improvement=np.sum(n*np.log1p(delta/b)-delta,axis=1)
    assert np.min(improvement)>-1e-7,(improvement.min(),A[np.argmin(improvement)])
    r=np.sign(A)*np.sqrt(2*np.maximum(improvement,0))
    q0=np.maximum(r,0)**2
    score=np.abs(np.sum(T*(n/lam-1),axis=1))
    assert np.max(score)<2e-7,(np.max(score),np.argmax(score))
    sigma=1/np.sqrt(np.sum(n*S[None,:]**2/lam**2,axis=1))
    return dict(A=A,sigma_A=sigma,sigma0=sigma0,r=r,q0=q0,
                score=score,minimum_lambda=np.min(lam,axis=1),
                nominal_local_p=np.where(q0>0,norm.sf(r),1.),
                conventional_local_p=norm.sf(np.maximum(r,0)),
                nominal_local_Z=np.maximum(r,0))

def free_p(q,k):
    q=np.asarray(q,float)
    p=sum(comb(k,j)*2.**(-k)*chi2.sf(q,j) for j in range(1,k+1))
    return np.where(q>0,p,1.)

def interval(k,n):
    return (0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k)))
