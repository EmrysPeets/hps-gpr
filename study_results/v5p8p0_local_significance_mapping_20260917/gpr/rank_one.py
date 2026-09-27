"""Exact frozen-kernel update for one changed training-count bin.

The GPR latent matrix changes in one diagonal entry and the latent target in
one coordinate. Sherman-Morrison therefore computes the same prediction as a
new Cholesky factorization, up to floating-point error. Production moment and
likelihood conditioning remain untouched. Independent Cholesky checks gate use.
"""
import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular
from gp_refit_pilot import count_moments
from hps_gpr.gpr import preprocess_xy_for_gpr
class RankOneAsimov:
 def __init__(self,ctx,truth,cfg):
  self.ctx=ctx;self.truth=truth;self.cfg=cfg;pred=ctx['pred'];y=truth[ctx['keep']]
  _,target,alpha=preprocess_xy_for_gpr(pred.x_train,y,cfg)
  assert cfg.pre_log and np.allclose(alpha,1/y,rtol=0,atol=0)
  self.y=y;self.alpha=alpha;self.target=target
  M=pred.K.copy();M[pred.diagonal]+=alpha;L=cholesky(M,lower=True,check_finite=False)
  self.coeff=cho_solve((L,True),target,check_finite=False)
  self.invdiag=np.diag(cho_solve((L,True),np.eye(len(y)),check_finite=False))
  self.H=cho_solve((L,True),pred.Kqt.T,check_finite=False).T
  v=solve_triangular(L,pred.Kqt.T,lower=True,check_finite=False)
  self.mu=pred.Kqt@self.coeff;self.C=pred.Kqq-v.T@v
  self.index={int(i):j for j,i in enumerate(np.flatnonzero(ctx['keep']))}
  self.reference=count_moments(self.mu,self.C,cfg)
  self.qa=[]
  for j in [0,len(y)//2,len(y)-1]:
   n=y.copy();n[j]+=np.sqrt(n[j]);direct=pred.predict(n);fast=self._one(j)
   mean_abs=float(np.max(abs(direct[0]-fast[0])));mean_rel=float(np.max(abs(direct[0]-fast[0])/np.maximum(1,direct[0])))
   covariance_relative=float(np.linalg.norm(direct[1]-fast[1])/max(1,np.linalg.norm(direct[1])))
   self.qa.append(dict(training_index=j,max_mean_abs=mean_abs,max_mean_rel=mean_rel,covariance_relative=covariance_relative))
   assert mean_rel<2e-8 and covariance_relative<2e-5,self.qa[-1]
 def _one(self,j):
  y=self.y[j];d=np.sqrt(y);da=1/(y+d)-self.alpha[j];dt=np.log1p(d/y)
  den=1+da*self.invdiag[j];h=self.H[:,j]
  mu=self.mu+h*(dt-da*self.coeff[j])/den
  C=self.C+(da/den)*np.outer(h,h)
  return count_moments(mu,C,self.cfg)
 def __call__(self,spectrum_index):
  if spectrum_index==0:return self.reference
  j=self.index.get(spectrum_index-1)
  return self.reference if j is None else self._one(j)
