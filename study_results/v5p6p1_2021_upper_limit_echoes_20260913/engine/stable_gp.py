"""PSD feature-space log-GP conditioning; avoids subtractive posterior covariance.
Positive eigenspace of the same RBF Gram matrix, cutoff 1e-14 relative to max.
Singular-value ridge conditioning preserves all retained prior uncertainty.
"""
import numpy as np
from scipy.linalg import eigh,svd
from functools import lru_cache

@lru_cache(maxsize=600)
def features(x_tuple,const,ls,cut=1e-14):
 x=np.asarray(x_tuple);K=const*np.exp(-.5*((np.log(x)[:,None]-np.log(x)[None,:])/ls)**2)
 val,U=eigh(K,check_finite=False);keep=val>cut*val[-1]
 F=U[:,keep]*np.sqrt(val[keep]);err=np.max(np.abs(F@F.T-K))
 return F,dict(feature_rank=int(keep.sum()),prior_max_abs_error=float(err),prior_relative_error=float(err/const),prior_min_eigenvalue=float(val[0]))

def predict_grid(x,n,train,query_indices,const,ls,cut=1e-14):
 F,diag=features(tuple(x),float(const),float(ls),float(cut));nt=np.asarray(n)[train]
 pos=nt>0;target=np.zeros(len(nt));target[pos]=np.log(nt[pos]);sd=np.ones(len(nt));sd[pos]=1/np.sqrt(nt[pos])
 U,s,Vh=svd(F[train]/sd[:,None],full_matrices=False,check_finite=False);Fq=F[query_indices]
 FV=Fq@Vh.T;mean=FV@((s/(1+s*s))*(U.T@(target/sd)))
 root=FV/np.sqrt(1+s*s);C=root@root.T;b=np.exp(mean+.5*np.diag(C));Ccounts=np.outer(b,b)*np.expm1(C)
 return b,.5*(Ccounts+Ccounts.T),diag
