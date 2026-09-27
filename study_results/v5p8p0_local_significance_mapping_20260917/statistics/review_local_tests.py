#!/usr/bin/env python3
"""Independent read-only check of released local-root arrays and binomial interval inversion."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import binom,norm
B=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
D=pd.read_csv(B/'local/local_tail_tests.csv');reference=pd.read_csv(OUT/'individual_information.csv');records=[];seeds=[]
for _,r in D.iterrows():
 f=B/f'local/{int(r.dataset)}_{int(r.mass_MeV)}.npz';z=np.load(f);roots=z[r.truth+'_roots'];robs=float(z['baseline_roots'][0]);n=len(roots)
 k=int(np.count_nonzero(roots>=robs)) if robs>0 else n
 seed=tuple(int(x) for x in z[r.truth+'_seed']);seeds.append(seed)
 probs_ok=(abs((k+1)/(n+1)-r.p_addone)<2e-14 and abs(k/n-r.p_mle)<2e-14 and abs(norm.sf(max(0,robs))-r.nominal_p0)<2e-14)
 # Clopper-Pearson endpoints invert the binomial survival/CDF, independently of beta quantiles.
 interval_error=0. if robs<=0 and r.p_lo95==1 and r.p_hi95==1 else max(abs(binom.sf(k-1,n,r.p_lo95)-.025) if k>0 else abs(r.p_lo95),abs(binom.cdf(k,n,r.p_hi95)-.025) if k<n else abs(r.p_hi95-1))
 q=reference[(reference.dataset==r.dataset)&(reference.mass_MeV==r.mass_MeV)&(reference.truth=='observed')]
 replay_error=abs(float(q.signed_r.iloc[0])-robs) if len(q) else None
 records.append(dict(dataset=int(r.dataset),mass_MeV=float(r.mass_MeV),truth=r.truth,n=n,exceedance_match=k==r.exceedances,probability_match=bool(probs_ok),binomial_interval_inversion_error=float(interval_error),independent_engine_root_error=replay_error,arrays_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
passed=all(r['exceedance_match'] and r['probability_match'] and r['binomial_interval_inversion_error']<1e-10 and (r['independent_engine_root_error'] is None or r['independent_engine_root_error']<2e-5) for r in records) and len(set(seeds))==len(seeds)
summary=dict(passed=passed,rows_checked=len(D),seed_streams_unique=len(set(seeds))==len(seeds),records=records,method_review=['Positive observed signed-root threshold equals inclusive q0 tail; negative/zero observed root has exact inclusive tail1 by definition.','Add-one estimate avoids zero point probability; confidence intervals concern conditional Monte Carlo tail, not background-model uncertainty.','Maximum across two named plug-in truths is finite-family diagnostic, not unconditional physical calibration.','Split-sample KS probability against fitted standard normal is nominal: estimated training moments induce shared uncertainty, so it is not an exact formal closure p-value.'])
(OUT/'independent_local_review.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2));assert passed
