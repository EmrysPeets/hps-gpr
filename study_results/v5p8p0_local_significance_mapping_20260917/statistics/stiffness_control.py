#!/usr/bin/env python3
"""Predeclared length-scale ladder; no policy selected from observed significance."""
from pathlib import Path
import os,sys,json,hashlib,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
ENG=ROOT/'study_results/v5p0p5_analysis_note_20260916/scripts';sys.path.insert(0,str(ENG))
import common as C
import numpy as np,pandas as pd
from scipy.linalg import cho_factor,cho_solve
START=time.monotonic();YEAR='2016';MASSES=(42.,66.,74.,76.,90.,92.,117.);FACTORS=(.5,.75,1.,1.25)
base_kernel=C.kernel_state;rows=[];base_records=[]

def lml(part):
 d=C.DATA[YEAR];mask=part['mask'];x=d['x'][~mask];n=part['counts'][~mask]
 pos=n>0;y=np.zeros(len(n));y[pos]=np.log(n[pos]);alpha=np.ones(len(n));alpha[pos]=1/n[pos]
 K=C.kernel(x,x,part['const'],part['ls']);K.flat[::len(K)+1]+=alpha;L=cho_factor(K,lower=True)
 return float(-.5*y@cho_solve(L,y)-np.log(np.diag(L[0])).sum()-.5*len(y)*np.log(2*np.pi))

def fit(p):
 S=p['S'][:,0];model=C.OneSignalProfile(p['b'],p['L'],S);f=model.fit(p['n']);n=model.fit(p['n'],0.)
 V=np.diag(p['b'])+p['L']@p['L'].T;sig=1/np.sqrt(S@cho_solve(cho_factor(V,lower=True),S))
 r=float(np.sign(f['A'])*np.sqrt(max(0.,2*(n['nll']-f['nll']))))
 return dict(signed_r=r,A_hat=f['A'],sigma_fisher=float(sig),max_score=max(f['score'],n['score']),min_lambda=min(f['min_lambda'],n['min_lambda']),sideband_log_marginal_likelihood=lml(p))

for mass in MASSES:
 C.kernel_state=base_kernel
 nominal=C.moving_context(YEAR,mass);fbase=fit(nominal);sigma_ref=fbase['sigma_fisher']
 d=C.DATA[YEAR];control,_=C.predict(d['x'],d['n'],nominal['mask'],nominal['const'],nominal['ls'],query=d['x'])
 full_signal=C.signal(YEAR,mass)
 truths={'stress':d['stress'],'nominal_local_GP_control':control}
 base_records.append(dict(mass_MeV=mass,sigma_ref_epsilon2=sigma_ref*1e-8,nominal_const=nominal['const'],nominal_ls=nominal['ls'],control_is_mass_specific=True))
 for factor in FACTORS:
  def scaled_kernel(year,mass,policy='interpolate',anchor=None):
   const,ls=base_kernel(year,mass,policy,anchor);return const,ls*factor
  C.kernel_state=scaled_kernel
  p=C.moving_context(YEAR,mass);f=fit(p)
  rows.append(dict(mass_MeV=mass,ls_factor=factor,truth='observed',strength=0.,sigma_ref_epsilon2=sigma_ref*1e-8,kernel_ls=p['ls'],kernel_const=p['const'],**f))
  for truth_name,truth in truths.items():
   for strength in (0.,2.,5.):
    counts=truth+strength*sigma_ref*full_signal
    p=C.moving_context(YEAR,mass,counts=counts);f=fit(p)
    rows.append(dict(mass_MeV=mass,ls_factor=factor,truth=truth_name,strength=strength,sigma_ref_epsilon2=sigma_ref*1e-8,kernel_ls=p['ls'],kernel_const=p['const'],**f))
 C.kernel_state=base_kernel
 print('stiffness',mass,'seconds',round(time.monotonic()-START,2),flush=True)
D=pd.DataFrame(rows)
D['A_hat_over_sigma_ref']=D.A_hat/(D.sigma_ref_epsilon2/1e-8)
D['sigma_fisher_over_ref']=D.sigma_fisher/(D.sigma_ref_epsilon2/1e-8)
# Paired deterministic difference tracks net extraction of the same injected yield.
zero=D[D.strength==0][['mass_MeV','ls_factor','truth','A_hat','signed_r']].rename(columns={'A_hat':'baseline_A','signed_r':'baseline_r'})
D=D.merge(zero,on=['mass_MeV','ls_factor','truth'])
D['delta_r']=D.signed_r-D.baseline_r
D['recovered_signal_fraction']=np.where(D.strength>0,(D.A_hat-D.baseline_A)/(D.strength*D.sigma_ref_epsilon2/1e-8),np.nan)
D.to_csv(OUT/'stiffness_scan.csv',index=False,float_format='%.17g')
protocol=dict(masses_MeV=MASSES,lengthscale_multipliers=FACTORS,kernel_amplitude='Held at nominal reviewed value',kernel_policy='Multiply archived nominal length scale; no optimization and no selection based on observed roots',signal_strengths_in_nominal_errors=[0,2,5],truths=list(truths),local_control='At each mass, full nominal GP mean from observed sidebands with nominal signal window excluded; mass-specific plug-in control, no global null',uncertainty='Fisher sigma from GP count covariance plus Poisson diagonal of GP mean; descriptive, not coverage',observed_lml='Exact sideband lognormal-GP Gaussian log marginal likelihood with unchanged zero mean and count-derived errors; training-set fit score, not predictive closure or discovery calibration',signal_tails='Full-bin integrated Gaussian injected across whole spectrum before GP sideband retraining',baseline=base_records,new_random_experiments=0,discovery_calibrated=False)
(OUT/'stiffness_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
qa=dict(passed=bool((D.max_score<2e-7).all() and (D.min_lambda>0).all()),rows=len(D),seconds=time.monotonic()-START,max_score=float(D.max_score.max()),min_lambda=float(D.min_lambda.min()),all_factors_executed=True,no_policy_selected=True)
(OUT/'stiffness_validation.json').write_text(json.dumps(qa,indent=2)+'\n');print(json.dumps(qa,indent=2));assert qa['passed']
