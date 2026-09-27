#!/usr/bin/env python3
"""Exact deterministic coupling decomposition; no new toys and no discovery calibration."""
from pathlib import Path
import os,sys,json,hashlib,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
ENG=ROOT/'study_results/v5p0p5_analysis_note_20260916/scripts'
sys.path.insert(0,str(ENG))
import common as C
import numpy as np,pandas as pd
from scipy.linalg import block_diag,cho_factor,cho_solve
from scipy.stats import norm,chi2
START=time.monotonic()
YEARS=('2015','2016','2021')
ANCHORS=(42.,51.,66.,71.,75.,76.,77.,90.,91.,92.,95.,117.)
# Retrospective anchors include inherited stress excursions and highlighted scan peaks.
MASS=np.arange(39.,101.)

def fitpart(p):
 s=p['S'][:,0];model=C.OneSignalProfile(p['b'],p['L'],s)
 f=model.fit(p['n']);h=model.fit(p['n'],0.)
 r=float(np.sign(f['A'])*np.sqrt(max(0.,2*(h['nll']-f['nll']))))
 V=np.diag(h['lam'])+p['L']@p['L'].T
 I=float(s@cho_solve(cho_factor(V,lower=True),s))
 U=float(s@(p['n']/h['lam']-1))
 return dict(r=r,A=float(f['A']),I=I,score_z=U/np.sqrt(I),nll=float(f['nll']),null_nll=float(h['nll']),max_score=max(f['score'],h['score']),min_lambda=min(f['min_lambda'],h['min_lambda']))

def joint(parts):
 p=dict(n=np.concatenate([x['n'] for x in parts]),b=np.concatenate([x['b'] for x in parts]),L=block_diag(*[x['L'] for x in parts]),S=np.concatenate([x['S'][:,0] for x in parts])[:,None])
 return fitpart(p)

single=[];combined=[]
for m in np.unique(np.r_[MASS,ANCHORS]):
 ys=[y for y in YEARS if C.LIMITS[y][0]<=m<=C.LIMITS[y][1]]
 for truth in ('observed','stress'):
  parts=[C.moving_context(y,float(m),counts=None if truth=='observed' else C.DATA[y]['stress']) for y in ys]
  fits=[fitpart(p) for p in parts];j=joint(parts)
  I=np.array([f['I'] for f in fits]);w=np.sqrt(I/I.sum());r=np.array([f['r'] for f in fits]);zs=np.array([f['score_z'] for f in fits])
  weighted=float(w@r);score=float(w@zs);qfree=float(r@r);qcommon=j['r']**2
  qfreeplus=float(np.maximum(r,0)@np.maximum(r,0));qcommonplus=max(0.,j['r'])**2
  for y,f,ww in zip(ys,fits,w):
   single.append(dict(mass_MeV=m,truth=truth,dataset=y,signed_r=f['r'],epsilon2_hat=f['A']*1e-8,information_per_1e8=f['I'],score_z=f['score_z'],weight=ww,information_fraction=ww**2,weighted_root=ww*f['r'],max_score=f['max_score'],min_lambda=f['min_lambda']))
  combined.append(dict(mass_MeV=m,truth=truth,datasets='+'.join(ys),signed_r=j['r'],epsilon2_hat=j['A']*1e-8,weighted_individual_roots=weighted,weighted_individual_scores=score,weighted_root_error=j['r']-weighted,score_root_error=j['r']-score,Q_independent_unconstrained=qfree,Q_common_unconstrained=qcommon,Q_rate_incompatibility=qfree-qcommon,Q_independent_nonnegative=qfreeplus,Q_common_nonnegative=qcommonplus,independent_amplitudes=len(ys),max_score=j['max_score'],min_lambda=j['min_lambda']))
 print('mass',m,'seconds',round(time.monotonic()-START,2),flush=True)
S=pd.DataFrame(single);J=pd.DataFrame(combined)
S.to_csv(OUT/'individual_information.csv',index=False,float_format='%.17g');J.to_csv(OUT/'coupling_decomposition.csv',index=False,float_format='%.17g')

# Shape-held exposure scaling is a deterministic mechanism control, not historical 10% data.
exposure=[]
for m in (42.,76.,92.,117.):
 for scale in (.1,.25,.5,1.):
  p=C.moving_context('2016',m,counts=C.DATA['2016']['stress']*scale);p['S']*=scale
  f=fitpart(p);exposure.append(dict(dataset='2016',mass_MeV=m,relative_exposure=scale,signed_stress_r=f['r'],r_div_sqrt_exposure=f['r']/np.sqrt(scale),epsilon2_hat=f['A']*1e-8,max_score=f['max_score'],source='2016 full coherent stress shape scaled; fixed kernel; not historical 10 percent source'))
E=pd.DataFrame(exposure);E.to_csv(OUT/'fixed_shape_exposure_control.csv',index=False,float_format='%.17g')
# Gaussian score identity: an algebraic model check, not a physical-model validation.
z=np.array([1.3,-2.8,2.1]);I=np.array([.5,2.,4.]);w=np.sqrt(I/I.sum());common=float(w@z);perp=float(np.dot(z-w*common,z-w*common))
qa=dict(passed=bool((S.max_score<2e-7).all() and (J.max_score<2e-7).all() and (J.Q_rate_incompatibility>-1e-7).all() and (J.Q_independent_nonnegative+1e-7>=J.Q_common_nonnegative).all()),max_optimizer_score=float(max(S.max_score.max(),J.max_score.max())),min_rate_incompatibility=float(J.Q_rate_incompatibility.min()),gaussian_orthogonal_identity_error=float(abs(z@z-common**2-perp)),max_weighted_root_error=float(abs(J.weighted_root_error).max()),max_score_root_error=float(abs(J.score_root_error).max()),seconds=time.monotonic()-START,unique_masses=int(J.mass_MeV.nunique()),total_individual_fits=len(S),total_combined_fits=len(J),new_random_experiments=0,global_p_computed=False,physical_local_calibration=False)
(OUT/'validation.json').write_text(json.dumps(qa,indent=2)+'\n')
files=[ENG/'common.py',ENG/'parent_core.py',ENG/'limit_solver.py']+[ENG.parent/f'inputs/spectrum_{y}.npz' for y in YEARS]+[Path(__file__)]
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(qa,indent=2));assert qa['passed']
