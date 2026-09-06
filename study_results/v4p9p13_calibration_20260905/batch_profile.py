"""Vectorized exact centered-Poisson/Gaussian profiling; scalar reference fallback."""
import numpy as np
from run_comparison import Profile

class BatchProfile:
    def __init__(self, counts, background, factors, template, blocks=None):
        self.n=np.asarray(counts,float);self.b=np.asarray(background,float)
        self.L=np.asarray(factors,float);self.w=np.asarray(template,float)
        self.scale=np.sqrt(self.b.sum(axis=1));self.nt=len(self.b)
        self.blocks=blocks
        self.npar=self.L.shape[-1];self.max_score=0.;self.fallbacks=0
        self.free=self.fit();self.null=self.fit(0.)
        delta=2*(self.null['nll']-self.free['nll'])
        if np.min(delta)<-1e-6:raise RuntimeError('Free/null nesting')
        self.r=np.sign(self.free['A'])*np.sqrt(np.maximum(delta,0))
        self.den=np.where(self.free['A']>=0,self.free['nll'],self.null['nll'])

    def objective(self,z,fixed):
        free=fixed is None;theta=z[:,int(free):]
        a=z[:,0]*self.scale if free else np.full(self.nt,fixed)
        lam=self.b+np.einsum('tij,tj->ti',self.L,theta)+a[:,None]*self.w
        positive=self.n>0
        t=(lam-self.n)/np.where(positive,self.n,1.)
        value=np.sum(np.where(positive,self.n*(t-np.log1p(t)),lam),axis=1)+.5*np.sum(theta**2,axis=1)
        if self.blocks is None:
            blocks=[(0,len(self.w),0,self.npar)]
        else:blocks=self.blocks
        offset=int(free);dim=self.npar+offset
        gradient=np.zeros((self.nt,dim));H=np.zeros((self.nt,dim,dim))
        r=(lam-self.n)/lam;v=self.n/lam**2
        if free:
            gradient[:,0]=self.scale*np.sum(self.w*r,axis=1)
            H[:,0,0]=self.scale**2*np.sum(self.w**2*v,axis=1)
        for r0,r1,c0,c1 in blocks:
            l=self.L[:,r0:r1,c0:c1];ci=slice(c0+offset,c1+offset)
            gradient[:,ci]=np.einsum('tij,ti->tj',l,r[:,r0:r1])+theta[:,c0:c1]
            H[:,ci,ci]=np.einsum('tij,ti,tik->tjk',l,v[:,r0:r1],l)
            if free:
                cross=self.scale[:,None]*np.einsum('tij,ti->tj',l,v[:,r0:r1]*self.w[r0:r1])
                H[:,0,ci]=cross;H[:,ci,0]=cross
        ind=np.arange(self.npar)+offset;H[:,ind,ind]+=1
        return value,gradient,H,lam

    def solve(self,H,g,free):
        if self.blocks is None:return np.linalg.solve(H,-g[:,:,None])[...,0]
        step=np.zeros_like(g);offset=int(free);crosses=[];parts=[]
        denominator=H[:,0,0].copy() if free else None
        numerator=-g[:,0].copy() if free else None
        for _,_,c0,c1 in self.blocks:
            ci=slice(c0+offset,c1+offset);block=H[:,ci,ci]
            rhs=np.stack((g[:,ci],H[:,ci,0]),axis=-1) if free else g[:,ci,None]
            inv=np.linalg.solve(block,rhs)
            if free:
                numerator+=np.sum(H[:,0,ci]*inv[:,:,0],axis=1)
                denominator-=np.sum(H[:,0,ci]*inv[:,:,1],axis=1)
                parts.append((ci,inv))
            else:step[:,ci]=-inv[:,:,0]
        if free:
            if np.any(denominator<=0):raise np.linalg.LinAlgError('Nonpositive Schur curvature')
            step[:,0]=numerator/denominator
            for ci,inv in parts:step[:,ci]=-inv[:,:,0]-inv[:,:,1]*step[:,0,None]
        return step

    def fit(self,fixed=None):
        dim=self.npar+int(fixed is None);z=np.zeros((self.nt,dim))
        if fixed is not None and hasattr(self,'free'):z[:]=self.free['z'][:,1:]
        if dim==0:
            v,g,h,lam=self.objective(z,fixed)
            return dict(nll=v,A=np.full(self.nt,fixed),z=z)
        active=np.ones(self.nt,bool)
        for iteration in range(12):
            v,g,H,lam=self.objective(z,fixed)
            scores=np.max(np.abs(g),axis=1);active=scores>=2e-7
            if not active.any():break
            try: step=self.solve(H[active],g[active],fixed is None)
            except np.linalg.LinAlgError:break
            # Check full Newton steps; uncommon hard rows use scalar line search.
            proposal=z.copy();proposal[active]+=step
            vn,_,_,ln=self.objective(proposal,fixed)
            good=(np.min(ln,axis=1)>0)&np.isfinite(vn)&(vn<=v+1e-10)
            z[active&good]=proposal[active&good]
            if not good[active].all():break
        v,g,H,lam=self.objective(z,fixed)
        scores=np.max(np.abs(g),axis=1)
        bad=(scores>=2e-7)|~np.isfinite(v)|(np.min(lam,axis=1)<=0)
        for i in np.flatnonzero(bad):
            scalar=Profile(self.b[i],self.L[i],self.w,'linear')
            fit=scalar.fit(self.n[i],fixed=fixed)
            v[i]=fit['nll'];z[i]=fit['z'];scores[i]=fit['score'];self.fallbacks+=1
        self.max_score=max(self.max_score,float(scores.max()))
        if np.any(scores>=2e-7):raise RuntimeError('Unconverged batch fit')
        return dict(nll=v,A=z[:,0]*self.scale if fixed is None else np.full(self.nt,fixed),z=z)

    def q(self,A):
        fit=self.fit(A);raw=2*(fit['nll']-self.den)
        take=self.free['A']<=A
        if np.min(raw[take],initial=0)<-1e-6:raise RuntimeError('Fixed/free nesting')
        return np.where(take,np.maximum(raw,0),0)
