"""Numerical/provenance audit for the interpretation artifact, without refits."""
from pathlib import Path
import os,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np
import pandas as pd
from scipy.stats import norm,beta
B=Path(__file__).resolve().parents[1];I=B/'inputs/v5p8p2';checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)
for r in json.loads((B/'inputs/provenance.json').read_text()):
    check('pinned source '+r['copy'],hashlib.sha256((B/r['copy']).read_bytes()).hexdigest()==r['sha256'])
s=pd.read_csv(I/'results/summary.csv',dtype={'scope':str}).set_index('scope')
a=pd.read_csv(B/'results/resolution_audit.csv',dtype={'scope':str}).set_index('scope')
for scope,r in s.iterrows():
    f=np.load(I/f'fields/{scope}.npz');i=np.argmin(abs(f['masses']-r.mass_MeV));m=f['gaussian_maximum'];n=len(m);u=(f['observed_r'][i]-f['a'][i])/f['s'][i];k=int(np.sum(m>=u));p=(k+1)/(n+1)
    check(scope+' reference local Z',abs(u-r.peak_local_Z)<1e-12)
    check(scope+' maximum count and tail',k==int(r.global_k) and abs(p-r.global_p_addone)<1e-12)
    check(scope+' local tail',abs(norm.sf(u)-r.peak_local_p)<1e-14)
    check(scope+' equivalent independent tests',abs(np.log1p(-p)/np.log1p(-norm.sf(u))-a.loc[scope,'peak_tail_equivalent_independent_tests'])<1e-8)
    calc=f['D'].T@f['D']/np.outer(f['s'],f['s'])
    check(scope+' response covariance identity',np.max(abs(calc-f['K']))<1e-10)
    check(scope+' PSD',np.linalg.eigvalsh(f['K']).min()>-1e-10)
    check(scope+' gates below plotted thresholds',np.max(-f['a']/f['s'])<1.5)
    check(scope+' finite high-end tail',np.sum(m>=4.5)>0)
    check(scope+' nested grids',np.all(m>=f['gaussian_coarse_maximum']))
    check(scope+' overlap subset',np.all(m>=f['gaussian_overlap_maximum']))
    check(scope+' direct validation interval',r.direct_global_lo95<=p<=r.direct_global_hi95)
    check(scope+' all observed peak gates positive',f['observed_r'][i]>0)
    check(scope+' global >= local',p>=norm.sf(u))
x=pd.read_csv(B/'results/gaussian_cls_illustration.csv')
check('Gaussian CLs endpoint identity',np.max(abs(norm.sf(x.upper90_sigma-x.observed_estimate_sigma)/norm.cdf(x.observed_estimate_sigma)-.1))<1e-14)
d=pd.read_csv(B/'results/combined_domain_comparison.csv')
check('identical combined threshold across domains',np.ptp(d.threshold)==0)
check('ordered combined domain tails',d.p.iloc[0]>=d.p.iloc[1]>=s.loc['combined','peak_local_p'])
check('source choice ordering',pd.read_csv(B/'inputs/v5p8p1_asimov_summary.csv').set_index('truth').loc['gp_full_nominal','rms_root']<1)
for name in ['peak_summary','combined_domain_effect','cls_vs_discovery','grid_and_source_choices','method_map','correlation_resolution','response_resolution','trials_comparison','trials_effective_counts']:
    for ext in ['pdf','png']:check('figure '+name+'.'+ext,(B/f'figures/{name}.{ext}').stat().st_size>1000)
out={'passed':True,'checks':len(checks),'check_names':checks,'fits_or_toys_performed':0,'conditional_on_fixed_source':True}
(B/'qa/numerical_validation.json').write_text(json.dumps(out,indent=2)+'\n');print('Passed',len(checks),'checks')
