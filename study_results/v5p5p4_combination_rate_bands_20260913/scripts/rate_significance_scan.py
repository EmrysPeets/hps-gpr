"""Exact fixed-shape rate profiles and deterministic three-score Gaussian tail integrals."""
from pathlib import Path
import sys,os,json
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'engine'))
import experiment as X
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2,norm
from numpy.polynomial.legendre import leggauss
F=np.load(B/'derived/score_fields.npz');means=F['means_MeV'];tindex=-1

def maximize(fun,lo,hi,n=41):
 grid=np.linspace(lo,hi,n);vals=np.array([fun(v) for v in grid]);candidates=[(vals[0],lo),(vals[-1],hi)]
 for i in range(1,n-1):
  if vals[i]>=vals[i-1] and vals[i]>=vals[i+1]:
   r=minimize_scalar(lambda z:-fun(z),bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-10});candidates.append((-r.fun,r.x))
 return max(candidates)

def sphere(n):
 z,w=leggauss(n);phi=np.arange(4*n)*2*np.pi/(4*n);r=np.sqrt(1-z*z)
 xyz=np.stack([r[:,None]*np.cos(phi)[None,:],r[:,None]*np.sin(phi)[None,:],np.broadcast_to(z[:,None],(n,len(phi)))],axis=-1).reshape(-1,3)
 return xyz,np.repeat(w/(2*len(phi)),len(phi))

def probability(q,dirs,weight,vectors):
 if q<=1e-12:return 1.
 out=0
 for k in range(0,len(dirs),2048):
  c=np.max(dirs[k:k+2048]@vectors.T,axis=1);positive=c>0;tail=np.zeros_like(c);tail[positive]=chi2.sf(q/c[positive]**2,3);out+=float(tail@weight[k:k+2048])
 return out

spheres={n:sphere(n) for n in [32,64]};rows=[];qa=[];zrows=[]
for j,m in enumerate(means):
 templates=[X.shape(i,float(m),1)[0] for i in range(3)]
 sqrtI=np.array([F['sqrt_information_'+y][j,tindex] for y in X.YS]);z=np.array([F['unit_vectors_'+y][j,tindex]@F['whitened_residual_'+y] for y in X.YS])
 for i,y in enumerate(X.YS):zrows.append(dict(mass_MeV=m,dataset=y,gaussian_score=z[i],sqrt_information=sqrtI[i]))
 for kind,feature,bounds in [('power',np.log(X.E/2.3),(-6.,6.)),('exponential',-(X.E-2.3),(0.,6.))]:
  cache={}
  def exact(v):
   if float(v) not in cache:
    R=np.exp(feature*v);S=np.concatenate([templates[i]*R[i] for i in range(3)]);f=X.C.OneSignalProfile(X.b,X.L,S).fit(X.n)
    cache[float(v)]=f
   f=cache[float(v)];return max(0,2*(X.nullnll-f['nll'])) if f['A']>0 else 0.
  q,slope=maximize(exact,*bounds);fit=cache[float(slope)]
  def gaussian(v):
   w=sqrtI*np.exp(feature*v);w/=np.linalg.norm(w);return max(0,float(w@z))**2
  qg,sg=maximize(gaussian,*bounds)
  ps={}
  for ng in [257,513]:
   slopes=np.linspace(*bounds,ng);vectors=sqrtI[None,:]*np.exp(slopes[:,None]*feature);vectors/=np.linalg.norm(vectors,axis=1)[:,None]
   for n in ([32,64] if ng==513 else [64]):
    dirs,w=spheres[n];ps[(n,ng)]=probability(q,dirs,w,vectors)
  dirs,w=spheres[64];pg=probability(qg,dirs,w,vectors)
  rows.append(dict(mass_MeV=m,model=kind,Q_exact=q,sqrtQ_exact=np.sqrt(q),slope=slope,amplitude_2p3_epsilon2=max(0,fit['A'])*1e-8,Q_gaussian=qg,gaussian_slope=sg,p_reference_at_exact_Q=ps[(64,513)],Z_reference_at_exact_Q=norm.isf(ps[(64,513)]),p_gaussian_statistic=pg,Z_gaussian_statistic=norm.isf(pg),sphere_fractional_change=(ps[(64,513)]-ps[(32,513)])/ps[(64,513)],slope_grid_fractional_change=(ps[(64,513)]-ps[(64,257)])/ps[(64,513)],max_inner_score=max(f['score'] for f in cache.values())))
 # Benchmarks exercise the radial integral using known single-direction and orthant tails.
 dirs,w=spheres[64]
 for qtest in [6.,18.]:
  vec=sqrtI/np.linalg.norm(sqrtI);pv=probability(qtest,dirs,w,vec[None,:]);target=norm.sf(np.sqrt(qtest))
  c=np.linalg.norm(np.maximum(dirs,0),axis=1);tail=np.zeros(len(c));mask=c>0;tail[mask]=chi2.sf(qtest/c[mask]**2,3);po=float(tail@w);to=3/8*chi2.sf(qtest,1)+3/8*chi2.sf(qtest,2)+1/8*chi2.sf(qtest,3)
  qa.append(dict(mass_MeV=m,Q=qtest,direction_fractional_error=(pv-target)/target,orthant_fractional_error=(po-to)/to))
pd.DataFrame(rows).to_csv(B/'derived/rate_significance_scan.csv',index=False);pd.DataFrame(zrows).to_csv(B/'derived/signed_gaussian_scores.csv',index=False)
r=pd.DataFrame(rows);a=pd.DataFrame(qa);checks=dict(all_positive=bool((r.p_reference_at_exact_Q>0).all()),sphere_refinement=bool(abs(r.sphere_fractional_change).max()<.002),slope_refinement=bool(abs(r.slope_grid_fractional_change).max()<.002),direction_benchmark=bool(abs(a.direction_fractional_error).max()<.002),orthant_benchmark=bool(abs(a.orthant_fractional_error).max()<.002),inner_convergence=bool(r.max_inner_score.max()<2e-7))
summary=dict(passed=all(checks.values()),checks=checks,rows=len(r),method='For z~N(0,I3), integrate chi3 radial tail over sphere of max positive standardized rate-direction projection; no simulation',sphere_polar_nodes=64,sphere_azimuth_nodes=256,slope_nodes=513,max_sphere_fractional_change=float(abs(r.sphere_fractional_change).max()),max_slope_fractional_change=float(abs(r.slope_grid_fractional_change).max()),max_single_direction_relative_error=float(abs(a.direction_fractional_error).max()),max_orthant_relative_error=float(abs(a.orthant_fractional_error).max()),gaussian_covariance='diag(null-fit lambda)+L L^T',conditional_fixed_mass_and_scaled_width=True,global_calibration=False)
(B/'qa/rate_scan_validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));print(r[r.mass_MeV==92].to_string(index=False));assert summary['passed']
