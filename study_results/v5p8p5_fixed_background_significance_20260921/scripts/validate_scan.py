"""Structural likelihood and whole-scan tail checks, separate from fitter code."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import fixed_core as F
import numpy as np,pandas as pd
from scipy.stats import norm,chi2,beta
from math import comb
B=Path(__file__).resolve().parents[1]
d=pd.read_csv(B/'results/observed_curves.csv',dtype={'scope':str})
checks={}
checks['observed_rows']=len(d)==1773
checks['finite_probabilities']=bool(np.isfinite(d.nominal_local_p).all() and d.nominal_local_p.between(0,1).all())
checks['optimizer_scores']=bool((d.numerical_score<2e-7).all())
checks['positive_poisson_means']=bool((d.minimum_lambda>0).all())
violations=[]
for mass,g in d.groupby('mass_MeV'):
    y=g[g.scope.isin(F.YEARS)];s=g[g.scope=='shared'].iloc[0];f=g[g.scope=='free'].iloc[0]
    if abs(f.q0-y.q0.sum())>1e-8 or s.q0>f.q0+1e-8:violations.append(float(mass))
    if len(y)==1 and abs(s.q0-y.q0.iloc[0])>1e-8:violations.append(float(mass))
    k=len(y);p=1. if f.q0==0 else sum(comb(k,j)*chi2.sf(f.q0,j)/2**k for j in range(1,k+1))
    assert np.isclose(p,f.nominal_local_p,rtol=1e-12,atol=1e-100)
checks['likelihood_nesting_and_factorization']=not violations
protocol=json.loads((B/'results/protocol.json').read_text())
checks['frozen_input_hashes']=all(hashlib.sha256((B/p).read_bytes()).hexdigest()==h for p,h in protocol['input_sha256'].items())
checks['blind_width_2p25']=protocol['blind_half_width_sigma']==2.25
if (B/'results/completion.json').exists():
    allcurves=pd.read_csv(B/'results/significance_curves.csv',dtype={'scope':str})
    checks['global_no_smaller_than_local']=bool((allcurves.conditional_global_k>=allcurves.conditional_local_k).all())
    tail_errors=[]
    for peak in json.loads((B/'results/peaks.json').read_text()):
        z=np.load(B/f"results/fields_{peak['scope']}.npz")
        sel=np.ones(len(z['masses']),bool) if peak['domain']=='full' else (z['masses']>=50)&(z['masses']<=100)
        i=int(np.flatnonzero(z['masses']==peak['mass_MeV'])[0])
        kl=int(np.sum(z['toy_q0'][:,i]>=peak['q0']))
        kg=int(np.sum(np.min(z['toy_nominal_p'][:,sel],axis=1)<=peak['nominal_local_p']))
        if kl!=peak['conditional_local_k'] or kg!=peak['conditional_global_k']:tail_errors.append(peak['scope']+peak['domain'])
    checks['independent_complete_scan_tail_counts']=not tail_errors
checks={k:bool(v) for k,v in checks.items()}
result={'passed':all(checks.values()),'checks':checks,'violating_masses':violations}
(B/'qa/scan_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert result['passed'],result
