"""Independent arithmetic/identity checks of completed specialist outputs."""
from pathlib import Path
import os, json, csv, hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
import numpy as np
from scipy.stats import norm, beta
H=Path(__file__).resolve().parent;B=H.parent
R=B/'residual_diagnostic/results';G=B/'scan_maximum/results'
def read(p):
    out=list(csv.DictReader(p.open()))
    for row in out:
        for k,v in row.items():
            try:row[k]=float(v)
            except ValueError:pass
    return out
a=read(R/'analytic_scan.csv');c=read(R/'poisson_controls.csv');p=read(R/'paired_contrasts.csv');e=read(R/'empirical_decomposition.csv')
f=dict(np.load(B/'scan_maximum/inputs/2021.npz'));summary=json.loads((G/'summary.json').read_text())
curves=list(csv.DictReader((B/'residual_diagnostic/inputs/slide14_original_scan.csv').open()))
arrays=np.load(R/'paired_Q_arrays.npz');keys=sorted(arrays.files)
means={(r['mass_MeV'],r['control']):r['mean'] for r in c}
identityerr=max(abs(r['empirical_fixedV_Q_identity']-means[(r['mass_MeV'],'exact_refit_fixedV')]) for r in e)
components=max(abs(r['expected_refit_Q_per_bin']-r['expected_refit_noise_per_bin']-r['deterministic_bias_Q_per_bin']) for r in a)
replay=max(abs(r['observed_Q_per_bin']-float(s['Q_per_bin'])) for r,s in zip(a,curves))
meanerr=0.
for r in c:
    v=arrays[f'm{int(r["mass_MeV"])}_{r["control"]}'];meanerr=max(meanerr,abs(v.mean()-r['mean']))
maxima=np.column_stack([arrays[f'm{int(r["mass_MeV"])}_exact_refit_adaptiveV'] for r in a]).max(1)
obs=max(r['observed_Q_per_bin'] for r in a);k=int(np.count_nonzero(maxima>=obs));n=len(maxima)
T=float(max(f['observed_r'].max(),0));u=(T-f['a'])/f['s']
exact={'independent_mass_nodes':float(-np.expm1(norm.logcdf(u).sum())),
       'perfectly_correlated_nodes':float(norm.sf(u.min()))}
mc=[]
for row in summary['paired_simulation']['scenarios']:
    if row['scenario'] in exact:
        prob=exact[row['scenario']];sample=row['k_above_unchanged_raw_peak']/row['n'];se=np.sqrt(prob*(1-prob)/row['n'])
        mc.append({'control':row['scenario'],'analytic_tail':prob,'MC_frequency':sample,'MC_z':float((sample-prob)/se),'within_5_MC_SE':bool(abs(sample-prob)<5*se)})
bridge=read(R/'bias_projection_bridge.csv');misses=[]
for row in read(G/'local_moments_all_scopes.csv'):
    if row['scope'] in ('2016',2016.0) and not row['sd_bonf95_normal_lo']<=row['model_scale_s']<=row['sd_bonf95_normal_hi']:
        misses.append({k:row[k] for k in ['mass_MeV','model_scale_s','direct_signed_sd','sd_bonf95_normal_lo','sd_bonf95_normal_hi']})
checks={
 'reviewer':'Independent skeptical statistics review; no changes to parent computations',
 'manual_final_report_review':{'files':['source/report.tex','source/counterfactual_table.tex','source/residual_table.tex','source/residual_findings.tex','source/cross_scope_table.tex'],'scientific_blockers':[],'verified':'Scientific definitions, numeric claims, tail constructions, conditional boundaries and selected-mass qualification agree with audited outputs. Root owns rendered PDF QA.'},
 'deterministic_checks':{
  '201_residual_masses_and_8_controls_each':len(a)==201 and len(c)==1608,
  'all_1608_arrays_have_256_finite_entries':len(keys)==1608 and all(arrays[k].shape==(256,) and np.isfinite(arrays[k]).all() for k in keys),
  'original_slide14_replay_exact':replay==0,
  'trace_plus_bias_decomposition':components<1e-12,
  'empirical_fixedV_identity':identityerr<1e-12,
  'control_CSV_means_match_saved_arrays':meanerr<1e-12,
  'exploratory_residual_maximum_exceedances_20_of_256':k==20 and n==256,
  'GLS_projection_fraction_within_Cauchy_bound':all(0<=r['GLS_signal_direction_fraction']<=1+1e-10 for r in bridge),
 },
 'max_errors':{'slide14_replay':replay,'trace_plus_bias':components,'empirical_fixedV':identityerr,'control_CSV_means':meanerr},
 'analytic_Gaussian_control_checks':mc,
 'residual_scan':{'observed_max_Q_per_bin':obs,'k':k,'N':n,'p_addone':(k+1)/(n+1),'CP95':[float(beta.ppf(.025,k,n-k+1)),float(beta.ppf(.975,k+1,n-k))],'interpretation':'Exploratory fixed-source residual scan; not resonance significance or unconditional model goodness of fit.'},
 'residual_quantitative_bounds':{
  'first_order_E_Q_per_bin_range':[min(r['expected_refit_Q_per_bin'] for r in a),max(r['expected_refit_Q_per_bin'] for r in a)],
  'noise_trace_per_bin_range':[min(r['expected_refit_noise_per_bin'] for r in a),max(r['expected_refit_noise_per_bin'] for r in a)],
  'max_abs_exact_minus_linear_mean':max(abs(r['mean']) for r in p if r['contrast']=='exact_minus_linear'),
  'max_abs_adaptive_minus_fixedV_mean':max(abs(r['mean']) for r in p if r['contrast']=='adaptiveV_minus_fixedV'),
  'max_conditioning_GLS_root_difference':max(abs(r['conditioned_GLS_minus_raw']) for r in bridge),
  'projection_at_78':next(r for r in bridge if r['mass_MeV']==78),
 },
 'unresolved_2016_width_region':misses,
 'limitations':['A non-rejection is not equality of moments or physical-source validation.','Normal-theory SD intervals are assumption-dependent; nearby 2016 failures are correlated.','No uncertainty of the observed-data-derived source is included.','Counterfactual Gaussian tails do not prescribe recentering the observed scan.','Empirical squared mean shifts require their positive finite-MC floor; corrected descriptive values may be negative.'],
 'files_reviewed_sha256':{str(path.relative_to(B)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [G/'summary.json',R/'analytic_scan.csv',R/'poisson_controls.csv',R/'paired_contrasts.csv',R/'empirical_decomposition.csv',R/'bias_projection_bridge.csv',R/'validation.json']},
}
checks['passed_numeric_and_definition_audit']=all(checks['deterministic_checks'].values()) and all(r['within_5_MC_SE'] for r in mc)
(H/'review_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'passed':checks['passed_numeric_and_definition_audit'],'Gaussian_controls':mc,'residual_scan':checks['residual_scan'],'max_errors':checks['max_errors']},indent=2))
