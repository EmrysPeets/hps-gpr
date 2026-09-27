"""Exact Fisher atom mixture within the fixed-source Gaussian approximation."""
import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm
def chisq_even_tail(x,k):
    h=np.maximum(np.asarray(x),0)/2
    poly=np.ones_like(h)
    if k>=2:poly+=h
    if k>=3:poly+=h*h/2
    return np.exp(-h)*poly
def fisher_tail(T,q):
    """P(sum -2log p_i >= T); p_i=U_i for U_i<q_i and 1 otherwise.

    U_i are independent Uniform(0,1). Conditional on an active subset J,
    T=-2 sum_J log(q_i)+chi2_(2|J|), with subset probability
    prod_J q_i prod_notJ (1-q_i). The all-inactive atom is retained at T=0.
    """
    q=np.clip(np.asarray(q),1e-300,1.);T=np.asarray(T);k=len(q);out=np.zeros_like(T,dtype=float)
    for code in range(1,1<<k):
        sel=[i for i in range(k) if (code>>i)&1];weight=np.ones_like(q[0]);offset=np.zeros_like(q[0])
        for i in range(k):
            if i in sel:weight*=q[i];offset-=2*np.log(q[i])
            else:weight*=1-q[i]
        out+=weight*chisq_even_tail(T-offset,len(sel))
    return np.where(T<=0,1.,np.clip(out,1e-300,1.))
def fisher_stat(z,r):
    return -2*np.sum(np.where(np.asarray(r)>0,log_ndtr(-np.asarray(z)),0.),axis=0)
def fisher_scores(z,r,q):
    T=fisher_stat(z,r);p=fisher_tail(T,q)
    return -np.log(p),p,chisq_even_tail(T,len(q))
