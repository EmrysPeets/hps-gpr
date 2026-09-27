"""Read-only audit of historical full-100 matched-reference injection rows."""
import os, sys, hashlib, json
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
ROW=OUT/'inputs/minimal_accepted_rows.csv'
hashes=json.loads((OUT/'inputs/original_source_hashes.json').read_text())
input_hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'inputs').iterdir() if p.is_file()}
d=pd.read_csv(ROW)
zero=d[d.inj_nsigma==0].copy();positive=d[d.inj_nsigma>0].copy()
keys=['scenario','mass_MeV','background_toy_index']
assert not zero.duplicated(keys).any()
joined=positive.merge(zero[keys+['A_hat','sigma_A','sigmaA_reference','_source_line']],on=keys,suffixes=('','_0'),validate='many_to_one')
joined['paired_recovery']=(joined.A_hat-joined.A_hat_0)/joined.strength
joined['unpaired_recovery']=joined.A_hat/joined.strength
joined['sigma_ref_over_baseline_sigma']=joined.sigmaA_ref/joined.sigma_A_0
joined['sigma_post_over_ref']=joined.sigma_A/joined.sigmaA_ref
joined['injection_identity_error']=joined.strength-joined.inj_nsigma*joined.sigmaA_ref
joined['poisson_signal_residual']=(joined.Nsig_full-joined.strength)/np.sqrt(joined.strength)
summary=[]
for (scenario,mass),s in zero.groupby(['scenario','mass_MeV']):
    ref=s.sigmaA_ref
    summary.append(dict(scenario=scenario,mass_MeV=mass,N=len(s),
        sigma_ref_mean=ref.mean(),sigma_ref_sd=ref.std(ddof=1),sigma_ref_cv=ref.std(ddof=1)/ref.mean(),
        sigma_ref_min=ref.min(),sigma_ref_max=ref.max(),
        corr_sigma_ref_Ahat0=ref.corr(s.A_hat),corr_sigma_ref_Z0=ref.corr(s.Zhat),
        reference_sigma_matches_sigma0_maxerr=float(abs(ref-s.sigma_A).max()),
        ref_over_asimov_mean=float((ref/s.sigmaA_reference).mean()),
        zero_pull_mean=s.pull.mean(),zero_pull_sd=s.pull.std(ddof=1),
        common_mean_injection_over_toy_sigma_mean=float((ref.mean()/ref).mean())))
summary=pd.DataFrame(summary);summary.to_csv(OUT/'sigma_reference_by_scenario_mass.csv',index=False)
summary[summary.scenario=='2021_10pct'].to_csv(OUT/'native2021_10pct_summary.csv',index=False)
rec=joined.groupby(['scenario','mass_MeV','inj_nsigma']).agg(N=('paired_recovery','size'),paired_recovery_mean=('paired_recovery','mean'),paired_recovery_sd=('paired_recovery','std'),unpaired_recovery_mean=('unpaired_recovery','mean'),sigma_post_over_ref_mean=('sigma_post_over_ref','mean')).reset_index()
rec.to_csv(OUT/'paired_recovery_by_cell.csv',index=False)
zero[keys+['sigmaA_ref','sigma_A','sigmaA_reference','A_hat','Zhat','pull','_source_line']].to_csv(OUT/'reference_rows.csv',index=False)
joined[keys+['inj_nsigma','strength','Nsig_full','sigmaA_ref','sigma_A','A_hat','A_hat_0','paired_recovery','_source_line','_source_line_0']].to_csv(OUT/'paired_rows.csv',index=False)
report=dict(source_rows=len(d),zero_rows=len(zero),paired_positive_rows=len(joined),
    cells=len(summary),scenarios=sorted(zero.scenario.unique()),
    mass_MeV=sorted(zero.mass_MeV.unique()),
    sigma_ref_cv_range=[float(summary.sigma_ref_cv.min()),float(summary.sigma_ref_cv.max())],
    injection_identity_max_absolute_error=float(abs(joined.injection_identity_error).max()),
    sigma_ref_over_baseline_sigma_range=[float(joined.sigma_ref_over_baseline_sigma.min()),float(joined.sigma_ref_over_baseline_sigma.max())],
    sigma_ref_varies_within_all_cells=bool((summary.sigma_ref_sd>0).all()),
    signal_total_equals_expectation_fraction=float(np.mean(joined.Nsig_full==joined.strength)),
    signal_poisson_residual_mean=float(joined.poisson_signal_residual.mean()),
    signal_poisson_residual_sd=float(joined.poisson_signal_residual.std(ddof=1)),
    caution='Signal residual summary is descriptive across accepted heterogeneous cells; not an independent calibration. No new fits or random draws.',
    row_column_warning='sigmaA_ref is the actual selected zero-signal fit uncertainty. sigmaA_reference on role=reference_bonly is a separate Asimov diagnostic; on injected rows it is the selected actual uncertainty.')
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in hashes.items() if Path(p).exists())
assert all(hashlib.sha256((OUT/p).read_bytes()).hexdigest()==s for p,s in input_hashes.items())
(OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
(OUT/'input_hashes.json').write_text(json.dumps(input_hashes,indent=2)+'\n')
print(json.dumps(report,indent=2));print(summary.to_string(index=False))
