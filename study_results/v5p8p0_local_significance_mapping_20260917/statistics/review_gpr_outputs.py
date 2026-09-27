#!/usr/bin/env python3
"""Independent saved-array review; no numerical fits or new Monte Carlo draws."""
from pathlib import Path
import os,json,hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];G=B/'gpr';OUT=Path(__file__).resolve().parent
f=np.load(G/'field_2016/field.npz');old=np.load(G/'inputs/v511_2016_field.npz');m=f['masses'];checks={};detail={}
def check(k,v):checks[k]=bool(v)
expect=np.unique(np.r_[np.arange(39,180.01,.5),np.arange(74,79.01,.25)])
check('293_expected_nodes',np.array_equal(m,expect) and len(m)==293)
j=np.searchsorted(m,old['masses']);check('old_masses_nested',np.array_equal(m[j],old['masses']))
inv={k:float(np.max(abs(f[k][...,j]-old[k]))) for k in ['a','D','s','validation']};check('same_coordinate_invariance',max(inv.values())<1e-12);detail['same_node_maximum_errors']=inv
C=f['D'].T@f['D'];s=np.sqrt(np.diag(C));R=C/np.outer(s,s)
check('covariance_from_responses',np.max(abs(C-f['C']))<1e-12 and np.max(abs(s-f['s']))<1e-12 and np.max(abs(R-f['K']))<1e-12)
e=np.linalg.eigvalsh(R);check('positive_definite_correlation',e.min()>0);detail['minimum_correlation_eigenvalue']=float(e.min())
check('coherent256_rows',f['validation'].shape==(256,293))
uniform=[]
for step,count in [(1.,142),(.5,283)]:
 mask=np.isclose(m/step,np.round(m/step));obs=f['observed_r'][mask];a=f['a'][mask]
 check(f'uniform_{step}_count',mask.sum()==count)
 uniform.append(dict(step_MeV=step,hypotheses=int(mask.sum()),stress_RMS=float(np.sqrt(np.mean(a*a))),maximum_observed_root=float(obs.max()),maximum_observed_mass=float(m[mask][np.argmax(obs)])))
detail['uniform_subsets']=uniform;detail['mixed293_node_stress_RMS']=float(np.sqrt(np.mean(f['a']**2)))
summary=pd.read_csv(G/'uniform_grid_summary.csv')
for r in uniform:
 x=summary[(summary.dataset==2016)&(summary.step_MeV==r['step_MeV'])].iloc[0]
 check(f'uniform_{r["step_MeV"]}_reported_RMS',abs(x.rms_stress_offset-r['stress_RMS'])<1e-12)
regions={'low':(39,50),'stress':(70,81),'quarter':(74,79),'high':(85,100),'full':(39,180)}
arrays={'ungated_centered':np.load(G/'gaussian_grid_maxima.npz'),'gated_centered':np.load(G/'gaussian_gated_maxima.npz'),'raw_positive_root':np.load(G/'gaussian_raw_maxima.npz')}
Z=(f['validation']-f['a'])/f['s'];zo=(f['observed_r']-f['a'])/f['s'];obs_table=pd.read_csv(G/'ordering_observed_maxima.csv');tails=pd.read_csv(G/'ordering_fixed_thresholds.csv')
ordering_results=[]
for ordering,A in arrays.items():
 for key in A.files:
  region,ss=key.rsplit('_',1);step=float(ss);mask=(m>=regions[region][0])&(m<=regions[region][1])&np.isclose(m/step,np.round(m/step));fine=A[key];coarse=A[f'{region}_1.0']
  check(f'{ordering}_{key}_nested_maxima',len(fine)==100000 and np.all(fine>=coarse))
  if ordering=='ungated_centered':vm=Z[:,mask].max(axis=1);threshold=float(zo[mask].max())
  elif ordering=='gated_centered':
   vm=np.where(f['validation'][:,mask]>0,Z[:,mask],-np.inf).max(axis=1);threshold=float(np.max(np.where(f['observed_r'][mask]>0,zo[mask],-np.inf)))
  else:vm=np.maximum(0,f['validation'][:,mask].max(axis=1));threshold=float(max(0,f['observed_r'][mask].max()))
  row=obs_table[(obs_table.region==region)&(obs_table.step_MeV==step)&(obs_table.ordering==ordering)].iloc[0]
  check(f'{ordering}_{key}_observed_counts',np.count_nonzero(fine>=threshold)==row.gaussian_exceedances and np.count_nonzero(vm>=threshold)==row.direct_exceedances)
  for t in (2.,3.,4.):
   row=tails[(tails.region==region)&(tails.step_MeV==step)&(tails.ordering==ordering)&(tails.threshold==t)].iloc[0]
   check(f'{ordering}_{key}_tail{t}',np.count_nonzero(fine>=t)==row.gaussian_exceedances and np.count_nonzero(vm>=t)==row.direct_exceedances)
  if region=='full':ordering_results.append(dict(ordering=ordering,step_MeV=step,observed_threshold=threshold,gaussian_exceedances=int(np.count_nonzero(fine>=threshold)),direct_exceedances=int(np.count_nonzero(vm>=threshold))))
detail['full_scan_orderings']=ordering_results
qa=json.loads((G/'rank_one_validation.json').read_text());check('recorded_rankone_reference_pass',qa['passed'] and max(x['max_abs_root_difference'] for x in qa['results'])<2e-5)
detail['rankone_complete_response_comparison_masses']=[x['mass_MeV'] for x in qa['results']];detail['rankone_maximum_root_error']=max(x['max_abs_root_difference'] for x in qa['results'])
newmask=~np.isin(m,old['masses']);moment=[];fallbacks=0;direct_nodes=[]
for mass in m[newmask]:
 q=json.loads((G/f'field_2016/m{mass:07.2f}.json').read_text())
 if 'rank_one_gp_checks' not in q:direct_nodes.append(float(mass))
 for r in q.get('rank_one_gp_checks',[]):
  if r.get('exact_fallback'):fallbacks+=1
  else:moment.append(r)
check('per_coordinate_moment_checks',all(q['max_mean_rel']<2e-8 and q['covariance_relative']<2e-5 for q in moment));detail['rankone_moment_checks']=len(moment);detail['rankone_moment_fallbacks']=fallbacks;detail['direct_Cholesky_first_pass_nodes']=direct_nodes
execution=json.loads((G/'execution.json').read_text());check('scalar_batch_likelihood_equivalence',execution['max_scalar_error']<2e-5 and execution['max_score']<2e-7);detail['scalar_checks']=execution['scalar_checks'];detail['maximum_scalar_error']=execution['max_scalar_error']
files=[G/'rank_one.py',G/'fast_profile.py',G/'summarize_grid.py',G/'field_2016/field.npz',G/'rank_one_validation.json',G/'ordering_fixed_thresholds.csv',G/'ordering_observed_maxima.csv']
result=dict(passed=all(checks.values()),checks=checks,detail=detail,input_sha256={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},claim='Validates numerical covariance propagation and conditional finite-grid comparisons; no independently calibrated local or global particle significance.',notes=['The140 added nodes comprise130 additional half-MeV coordinates and10 quarter-step coordinates relative to153-node v5.1.1. Uniform1/0.5 comparisons use142/283 coordinates, excluding quarter extras.','Negative infinity is the correct gated maximum when no raw fitted root is positive. All paired maxima remain monotone for the nested grids.','local_mapping.csv empirical signed-root exceedances are ungated; they must not be substituted for its gated-probability column or the independent local/q0 tables.','The held-out shape metrics include uncertainty from estimated training moments and are descriptive.','The rank-one update changes both log-count target and alpha=1/count noise consistently. Exact scalar likelihood comparisons use the same propagated moments, while independent Cholesky comparisons verify moments.'])
(OUT/'gpr_independent_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':result['passed'],'checks':len(checks),'detail':detail},indent=2));assert result['passed']
