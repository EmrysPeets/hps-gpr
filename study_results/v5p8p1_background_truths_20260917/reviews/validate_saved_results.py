#!/usr/bin/env python3
"""Independent saved-array checks; no fits, new toys, or truth selection."""
from pathlib import Path
import os,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1];OUT=B/'reviews';checks={};details={}
def check(name,test):checks[name]=bool(test)
P=json.loads((B/'protocol.json').read_text());T=dict(np.load(B/'truths/backgrounds.npz'));meta={'edges_GeV','observed','x_MeV'}
if (B/'truths/supplementary.npz').exists():T.update({k:v for k,v in dict(np.load(B/'truths/supplementary.npz')).items() if k not in meta})
names=[k for k in T if k not in meta]
reference=np.load(ROOT/'study_results/v5p0p5_analysis_note_20260916/inputs/spectrum_2016.npz')
check('observed_source_identical',np.array_equal(T['observed'],reference['n']))
check('native_edges_identical',np.allclose(T['edges_GeV'],reference['edges'],rtol=0,atol=1e-14))
check('edges_strictly_increasing',np.all(np.diff(T['edges_GeV'])>0))
truth_metrics=[]
for name in names:
 t=T[name];check(name+'_positive_finite_full_support',t.shape==T['observed'].shape and np.all(np.isfinite(t)) and np.all(t>0))
 curvature=np.diff(np.log(t),n=2);j=int(np.argmax(abs(curvature)))
 truth_metrics.append(dict(truth=name,expected_counts=float(t.sum()),normalization_relative_observed=float(t.sum()/T['observed'].sum()),minimum_bin_mean=float(t.min()),maximum_absolute_second_log_difference=float(abs(curvature).max()),maximum_curvature_center_MeV=float(500*(T['edges_GeV'][j+1]+T['edges_GeV'][j+2]))))
details['truth_metrics']=truth_metrics
D=pd.read_csv(B/'results/asimov_scan.csv');I=pd.read_csv(B/'results/injections.csv');O=pd.read_csv(B/'results/observed_anchors.csv');S=pd.read_csv(B/'results/asimov_summary.csv')
check('all_frozen_truths_in_scan',set(D.truth)==set(names))
check('all_142_uniform_masses_per_truth',all(np.array_equal(np.sort(g.mass_MeV.to_numpy()),P['scan_masses_MeV']) for _,g in D.groupby('truth')))
check('finite_profile_outputs',np.all(np.isfinite(D[['signed_r','A_hat','sigma_fisher','max_score','min_lambda']].to_numpy())))
check('converged_positive_profiles',D.max_score.max()<2e-7 and D.min_lambda.min()>0 and I.max_score.max()<2e-7 and I.min_lambda.min()>0)
for name,g in D.groupby('truth'):
 q=S[S.truth==name].iloc[0]
 check(name+'_summary_RMS',abs(q.rms_root-np.sqrt(np.mean(g.signed_r**2)))<1e-11)
 check(name+'_summary_extrema',abs(q.minimum_root-g.signed_r.min())<1e-11 and abs(q.maximum_root-g.signed_r.max())<1e-11)
check('all_injection_scenarios',len(I)==len(names)*len(P['anchors_MeV'])*2 and set(I.strength_original_sigma)=={2.,5.})
base=D[['truth','mass_MeV','A_hat','signed_r']].rename(columns={'A_hat':'baseline_A_check','signed_r':'baseline_r_check'});Q=I.merge(base,on=['truth','mass_MeV']).merge(O[['mass_MeV','sigma_fisher']],on='mass_MeV',suffixes=('','_reference_check'))
check('same_original_sigma_across_truths',np.max(abs(Q.sigma_reference-Q.sigma_fisher_reference_check))<1e-12)
recovered=(Q.A_hat-Q.baseline_A_check)/(Q.strength_original_sigma*Q.sigma_reference)
check('injected_yield_recovery_recounted',np.max(abs(recovered-Q.recovered_fraction))<1e-11)
check('injected_root_increment_recounted',np.max(abs(Q.signed_r-Q.baseline_r_check-Q.delta_r))<1e-11)
old=pd.read_csv(ROOT/'study_results/v5p8p0_local_significance_mapping_20260917/statistics/individual_information.csv');old=old[(old.dataset==2016)&(old.truth=='observed')]
R=O.merge(old[['mass_MeV','signed_r']],on='mass_MeV',suffixes=('','_v580'))
check('unchanged_observed_signed_roots',len(R)==len(P['anchors_MeV']) and np.max(abs(R.signed_r-R.signed_r_v580))<2e-5)
details['observed_root_maximum_replay_error']=float(np.max(abs(R.signed_r-R.signed_r_v580)))
details['truth_count']=len(names);details['scan_rows']=len(D);details['injection_rows']=len(I);details['minimum_recovered_fraction']=float(I.recovered_fraction.min());details['maximum_recovered_fraction']=float(I.recovered_fraction.max())
files=[B/'truths/backgrounds.npz',B/'truths/supplementary.npz',B/'results/asimov_scan.csv',B/'results/asimov_summary.csv',B/'results/injections.csv',B/'results/observed_anchors.csv']
details['strict_optimizer_failure_rows']=D.loc[D.max_score>=2e-7,['mass_MeV','truth','max_score','signed_r']].to_dict('records')
result=dict(passed=all(checks.values()),execution_complete=len(D)==len(names)*len(P['scan_masses_MeV']) and len(I)==len(names)*len(P['anchors_MeV'])*2,checks=checks,details=details,input_sha256={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},statistical_scope='Implementation and conditional deterministic response validated; physical null adequacy and independent local calibration not established.',curvature_diagnostic='Finite log curvature is descriptive; no threshold here declares a smooth or physically adequate join.')
(OUT/'statistics_deterministic_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':result['passed'],'checks':len(checks),'details':details},indent=2));print('Strict optimizer failures remain flagged; no tolerance was loosened.') if not result['passed'] else None
