"""Algebraically identical BatchProfile objective using batched BLAS products."""
import numpy as np
from batch_profile import BatchProfile
class FastBatchProfile(BatchProfile):
 def objective(self,z,fixed):
  free=fixed is None;theta=z[:,int(free):]
  a=z[:,0]*self.scale if free else np.full(self.nt,fixed)
  lam=self.b+np.matmul(self.L,theta[...,None])[...,0]+a[:,None]*self.w
  positive=self.n>0;t=(lam-self.n)/np.where(positive,self.n,1.)
  value=np.sum(np.where(positive,self.n*(t-np.log1p(t)),lam),axis=1)+.5*np.sum(theta**2,axis=1)
  blocks=[(0,len(self.w),0,self.npar)] if self.blocks is None else self.blocks
  offset=int(free);dim=self.npar+offset
  gradient=np.zeros((self.nt,dim));H=np.zeros((self.nt,dim,dim));r=(lam-self.n)/lam;v=self.n/lam**2
  if free:
   gradient[:,0]=self.scale*np.sum(self.w*r,axis=1);H[:,0,0]=self.scale**2*np.sum(self.w**2*v,axis=1)
  for r0,r1,c0,c1 in blocks:
   l=self.L[:,r0:r1,c0:c1];lt=np.swapaxes(l,1,2);ci=slice(c0+offset,c1+offset)
   gradient[:,ci]=np.matmul(lt,r[:,r0:r1,None])[...,0]+theta[:,c0:c1]
   H[:,ci,ci]=np.matmul(lt,v[:,r0:r1,None]*l)
   if free:
    cross=self.scale[:,None]*np.matmul(lt,(v[:,r0:r1]*self.w[r0:r1])[...,None])[...,0]
    H[:,0,ci]=cross;H[:,ci,0]=cross
  ind=np.arange(self.npar)+offset;H[:,ind,ind]+=1
  return value,gradient,H,lam
