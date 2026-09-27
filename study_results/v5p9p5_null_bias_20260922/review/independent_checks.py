"""Small read-only audit of pinned inputs; writes only this review directory."""
from pathlib import Path
import os, json, hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
import numpy as np
from scipy.stats import norm, beta

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
# Prefer the study's bundled, identity-checked copies for a portable audit.
F=HERE.parent/'scan_maximum/inputs/2021.npz'
N=HERE.parent/'residual_diagnostic/inputs/null_2021.npz'
S=HERE.parent/'residual_diagnostic/inputs/spectrum_2021.npz'
f=dict(np.load(F));n=dict(np.load(N));s=dict(np.load(S))
cov=f['D'].T@f['D'];sd=np.sqrt(np.diag(cov));K=cov/np.outer(sd,sd)
j=int(np.argmax(f['observed_r']));r=float(f['observed_r'][j]);T=float(max(0,r));v=f['validation'];g=f['gaussian_raw_maximum']
kg=int(np.sum(g>=T));kd=int(np.sum(np.maximum(v.max(axis=1),0)>=T));kl=int(np.sum(v[:,j]>=T))
ci=lambda k,N:[float(beta.ppf(.025,k,N-k+1)) if k else 0.,float(beta.ppf(.975,k+1,N-k)) if k<N else 1.]
a=float(f['a'][j]);ss=float(f['s'][j]);mean=float(v[:,j].mean());width=float(v[:,j].std(ddof=1))
checks={
 'review_scope':'2021 fixed-source input identities, raw ordering, positive-part convention, and field covariance normalization; no new fits or toys',
 'input_hashes':{str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [F,N,S]},
 'shape':{k:list(f[k].shape) for k in ['D','K','validation','masses','gaussian_raw_maximum']},
 'checks':{
  'D_norm_is_s':bool(np.allclose(sd,f['s'],atol=1e-11,rtol=1e-11)),
  'saved_K_is_correlation_not_covariance':bool(np.allclose(K,f['K'],atol=1e-11,rtol=1e-11)),
  'K_unit_diagonal':bool(np.allclose(np.diag(f['K']),1,atol=1e-11)),
  'source_observed_counts_equal_v505':bool(np.array_equal(n['observed'],s['n'])),
  'source_edges_equal_v505':bool(np.array_equal(n['edges_GeV'],s['edges'])),
  'slide50_grid_half_MeV':bool(np.allclose(np.diff(f['masses']),.5)),
  'all_training_means_positive':bool(np.all(n['truth']>0)),
 },
 'covariance_identity_max_abs_error':float(np.max(abs(K-f['K']))),
 'field_domain_MeV':[float(f['masses'].min()),float(f['masses'].max())],
 'source_support_MeV':[float(s['edges'][0]*1000),float(s['edges'][-1]*1000)],
 'bin_width_MeV':[float(np.diff(s['edges']).min()*1000),float(np.diff(s['edges']).max()*1000)],
 'source_min_mean':float(n['truth'].min()),
 'mass_resolution_GeV_range':[float(s['sigma'].min()),float(s['sigma'].max())],
 'peak_illustration':{
  'selection':'mass chosen by observed scan maximum; local diagnostics here are illustrative, not preselected tests',
  'mass_MeV':float(f['masses'][j]),'raw_signed_root':r,'conventional_local_p':float(norm.sf(T)),
  'a_Asimov':a,'s_response':ss,'raw_toy_mean':mean,'raw_toy_sd':width,
  'mean_minus_a':mean-a,'mean_minus_a_SE':width/np.sqrt(len(v)),
  'response_conditional_marginal_tail':float(norm.sf((T-a)/ss)),
  'Gaussian_global_k':kg,'Gaussian_global_N':len(g),'Gaussian_global_addone':(kg+1)/(len(g)+1),
  'Poisson_global_k':kd,'Poisson_global_N':len(v),'Poisson_global_CP95':ci(kd,len(v)),
  'Poisson_fixed_mass_k':kl,'Poisson_fixed_mass_CP95':ci(kl,len(v)),
  'Gaussian_fixed_mass_expected_positive_part':float(ss*norm.pdf(a/ss)+a*norm.cdf(a/ss)),
  'direct_fixed_mass_mean_positive_part':float(np.maximum(v[:,j],0).mean()),
 },
 'zero_mean_unit_variance_reference':{
  'E_signed_root':0.,'E_positive_part':float(norm.pdf(0)),'E_q0':.5,
  'positive_part_atom_at_zero':.5,
  'interpretation':'Positive upward-excess mean and positive scan maximum occur under a perfectly centered null and are not evidence of fixed-mass estimator bias.',
 },
 'claims_not_established':['unconditional source calibration','confidence-limit coverage','validity of all kernel/support choices','absence of all implementation errors'],
}
checks['passed']=all(checks['checks'].values())
(HERE/'independent_input_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'passed':checks['passed'],'peak':checks['peak_illustration'],'source_support_MeV':checks['source_support_MeV'],'bin_width_MeV':checks['bin_width_MeV']},indent=2))
