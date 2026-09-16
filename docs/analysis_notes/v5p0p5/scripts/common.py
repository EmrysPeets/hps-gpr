"""Pinned spectra, declared continuous states, and common-region count models."""
from parent_core import *
from limit_solver import poisson_deviance_half, OneSignalProfile
from scipy.linalg import cho_factor
from itertools import combinations
import copy
# Retain inputs/scopes.json as the unchanged parent record.
SCOPES=copy.deepcopy(SCOPES)
for s in SCOPES:
 if '2015' in s[2]:s[4]=100
LIMITS={y:(float(d['masses'][0]),100. if y=='2015' else float(d['masses'][-1])) for y,d in DATA.items()}
SHORT={'individual_2015_full':'2015','individual_2016_full':'2016','individual_2021_10pct':'2021 (10%)','pair_2015_2016':'2015 + 2016','pair_2015_2021':'2015 + 2021','pair_2016_2021':'2016 + 2021','all_2015_2016_2021':'All three'}
def sigma(year,mass):return float(np.polynomial.polynomial.polyval(mass/1000.,DATA[year]['sigma_coeffs']))
def signal(year,mass,mask=None):return continuous_signal(year,mass,mask)
def kernel_state(year,mass,policy='interpolate',anchor=None):
 d=DATA[year];m=float(mass if anchor is None else anchor);m=min(max(m,float(d['masses'][0])),float(d['masses'][-1]))
 if policy=='nearest':m=float(np.floor(m+.5))
 if m.is_integer():
  i=d['idx'][int(m)];return float(d['const'][i]),float(d['ls'][i])
 # Positive log-linear interpolation is declared before the half-grid scan.
 return tuple(float(np.exp(np.interp(m,d['masses'],np.log(d[k])))) for k in ('const','ls'))
def predict(x,n,mask,const,ls,query=None):
 # Same archived log preprocessing, including zero-target=0, zero alpha=1.
 xt=x[~mask];yt=np.asarray(n)[~mask];xq=x[mask] if query is None else query
 if np.any(yt<0):raise ValueError('Negative training count')
 pos=yt>0;target=np.zeros_like(yt,dtype=float);target[pos]=np.log(yt[pos]);alpha=np.ones_like(target);alpha[pos]=1/yt[pos]
 K=kernel(xt,xt,const,ls);K.flat[::len(K)+1]+=alpha
 L=cholesky(K,lower=True,check_finite=False);Kqt=kernel(xq,xt,const,ls)
 latent=Kqt@cho_solve((L,True),target,check_finite=False)
 v=solve_triangular(L,Kqt.T,lower=True,check_finite=False)
 C=kernel(xq,xq,const,ls)-v.T@v;C=.5*(C+C.T)
 b=np.exp(latent+.5*np.maximum(np.diag(C),0))
 return b,np.outer(b,b)*np.expm1(np.clip(C,-40,40))
def context(year,masses,padding=3.,counts=None,anchor=None,region=None,full=False,policy='interpolate'):
 d=DATA[year];masses=np.atleast_1d(masses).astype(float);m1,m2=float(min(masses)),float(max(masses));x=d['x']
 if m1<LIMITS[year][0] or m2>LIMITS[year][1]:raise ValueError(f'Unsupported mass for {year}: {masses}')
 n=d['n'] if counts is None else np.asarray(counts,float)
 anchor=float(np.floor((m1+m2)/2)) if anchor is None else float(anchor)
 lo,hi=(m1/1000-padding*sigma(year,m1),m2/1000+padding*sigma(year,m2)) if region is None else region
 mask=(x>=lo)&(x<=hi)
 if np.sum(x<lo)<3 or np.sum(x>hi)<3:raise ValueError('Fewer than three bins in an external sideband')
 const,ls=kernel_state(year,anchor,policy=policy);b,C=predict(x,n,mask,const,ls);L,diag=factor_cov(C,b,full)
 return dict(year=year,masses=masses,anchor=anchor,lo=lo,hi=hi,mask=mask,n=n[mask],counts=n,b=b,C=C,L=L,
  S=np.column_stack([signal(year,m,mask) for m in masses]),const=const,ls=ls,diagnostic=diag)
def moving_context(year,mass,counts=None,policy='interpolate',anchor=None):
 return context(year,[mass],padding=2.25,counts=counts,anchor=mass if anchor is None else anchor,policy=policy)
class MultiModel:
 def __init__(self,parts):
  self.parts=parts;self.n=np.concatenate([p['n'] for p in parts]);self.b=np.concatenate([p['b'] for p in parts]);self.L=block_diag(*[p['L'] for p in parts]);self.S=np.vstack([p['S'] for p in parts]);self.k=self.S.shape[1]
  self.rank=self.L.shape[1];self.scale=1/np.sqrt(np.sum(self.S**2/self.b[:,None],axis=0));self.T=self.S*self.scale
 def fit(self,active=(),fixed=None,initial=None):
  fixed=np.zeros(self.k) if fixed is None else np.asarray(fixed,float);active=list(active);base=self.b+self.S@fixed
  J=np.column_stack((self.T[:,active],self.L));k=len(active);z=np.zeros(J.shape[1]) if initial is None else initial.copy();pen=np.r_[np.zeros(k),np.ones(self.rank)]
  def ev(v,hessian=True):
   lam=base+J@v
   if np.any(lam<=0):return np.inf,None,None,lam
   value=poisson_deviance_half(self.n,lam)+.5*float(np.sum(pen*v*v));g=J.T@(1-self.n/lam)+pen*v
   H=(J.T*(self.n/lam**2))@J if hessian else None
   if H is not None:H.flat[::len(H)+1]+=pen
   return value,g,H,lam
  for iteration in range(100):
   value,g,H,lam=ev(z);score=float(np.max(abs(g),initial=0))
   if score<2e-7:break
   step=cho_solve(cho_factor(H,lower=True,check_finite=False),-g,check_finite=False);descent=g@step
   if descent>=0:raise RuntimeError('Non-descent Newton direction')
   a=1.
   for _ in range(50):
    if ev(z+a*step,False)[0]<=value+1e-4*a*descent+1e-11:z+=a*step;break
    a*=.5
   else:raise RuntimeError('Line search failed')
  else:raise RuntimeError(f'Fit convergence failed {score}')
  amp=fixed.copy();amp[active]+=z[:k]*self.scale[active];theta=z[k:];cov=None
  if k:cov=np.linalg.inv(H)[:k,:k]*np.outer(self.scale[active],self.scale[active])
  return dict(nll=float(value),amp=amp,theta=theta,bfit=self.b+self.L@theta,lam=lam,score=score,iterations=iteration,covariance=cov,active=active)
 def efficient_information(self,lam=None):
  lam=self.b if lam is None else lam;I=self.S.T@(self.S/lam[:,None]);cross=self.S.T@(self.L/lam[:,None]);H=self.L.T@(self.L/lam[:,None])+np.eye(self.rank)
  I-=cross@cho_solve(cho_factor(H,lower=True,check_finite=False),cross.T,check_finite=False);return .5*(I+I.T)
 def hypotheses(self):
  raw={a:self.fit(a) for k in range(self.k+1) for a in combinations(range(self.k),k)};fits={}
  for a in raw:
   feasible=[f for s,f in raw.items() if set(s).issubset(a) and np.min(f['amp'])>=0]
   fits[a]=min(feasible,key=lambda f:f['nll'])
  total=tuple(range(self.k));f0=fits[()];fk=fits[total];best_prev=min(f['nll'] for a,f in fits.items() if len(a)==self.k-1)
  summary=dict(Q=max(0.,2*(f0['nll']-fk['nll'])),q_add=max(0.,2*(best_prev-fk['nll'])),max_score=max(f['score'] for f in raw.values()),min_background=float(fk['bfit'].min()),min_lambda=float(fk['lam'].min()))
  if self.k==2:
   I=self.efficient_information(f0['lam']);rho=float(np.clip(I[0,1]/np.sqrt(I[0,0]*I[1,1]),-1,1));w2=float(np.arccos(rho)/(2*np.pi));Q=summary['Q'];cov=raw[total]['covariance']
   summary.update(Q2=Q,p2_fixed_pair_chibar=float(.5*chi2.sf(Q,1)+w2*chi2.sf(Q,2)) if Q>1e-10 else 1.,rho_score=rho,rho_amplitude=float(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])),psi1_1e8=float(fk['amp'][0]),psi2_1e8=float(fk['amp'][1]),nll_LH=fk['nll'],nll_L=fits[(0,)]['nll'],nll_H=fits[(1,)]['nll'],nll_0=f0['nll'])
  return fits,summary

def full_expectation(part,fit,offset=0):
 d=DATA[part['year']];mask=part['mask'];fullmean,Cfull=predict(d['x'],part['counts'],mask,part['const'],part['ls'],query=d['x']);sl=slice(offset,offset+len(part['n']));sd=np.sqrt(part['b'])
 U,sing,_=np.linalg.svd(part['L']/sd[:,None],full_matrices=False);delta=fit['bfit'][sl]-part['b'];coef=(U@((U.T@(delta/sd))/sing**2))/sd
 residual=part['L']@(part['L'].T@coef)-delta
 if np.max(abs(residual)/sd)>1e-6:raise RuntimeError('Full-support extension residual')
 continuum=fullmean+Cfull[:,mask]@coef;continuum[mask]=fit['bfit'][sl]
 sig=np.column_stack([signal(part['year'],m) for m in part['masses']])@fit['amp'];mean=continuum+sig
 if np.any(mean<=0) or np.any(continuum<=0):raise RuntimeError('Nonpositive full-support truth')
 return dict(mean=mean,background=continuum,signal=sig,external_gp_mean=fullmean,extension_residual=float(np.max(abs(residual)/sd)))

def scan_spectrum(counts,masses,scope_keys=None,policy='interpolate',anchor=None):
 keys={s[0] for s in SCOPES} if scope_keys is None else set(scope_keys);rows=[]
 for m in masses:
  parts={y:moving_context(y,float(m),n,policy,anchor) for y,n in counts.items() if LIMITS[y][0]<=m<=LIMITS[y][1]}
  for scope in SCOPES:
   key,label,ys,lo,hi=scope
   if key not in keys or not all(y in parts for y in ys) or not lo<=m<=hi:continue
   ps=[parts[y] for y in ys];b=np.concatenate([p['b'] for p in ps]);L=block_diag(*[p['L'] for p in ps]);S=np.concatenate([p['S'][:,0] for p in ps]);n=np.concatenate([p['n'] for p in ps])
   r=OneSignalProfile(b,L,S).limit(n);r.update(scope=key,mass_MeV=float(m),epsilon2_90=r['A90']*1e-8,kernel_policy=policy,extension_2015=bool('2015' in ys and m>90),maximum_covariance_load=max(p['diagnostic']['load'] for p in ps));rows.append(r)
 return rows
