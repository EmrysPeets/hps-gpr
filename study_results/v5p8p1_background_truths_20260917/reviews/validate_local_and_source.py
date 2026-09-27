#!/usr/bin/env python3
"""Recount local null arrays and source-construction injections; no new sampling."""
from pathlib import Path
import os,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd
from scipy.stats import binom,norm
B=Path(__file__).resolve().parents[1];OUT=B/'reviews';P=json.loads((B/'protocol.json').read_text());checks={};detail={}
def check(k,v):checks[k]=bool(v)
T=dict(np.load(B/'truths/backgrounds.npz'));T.update({k:v for k,v in dict(np.load(B/'truths/supplementary.npz')).items() if k not in ['edges_GeV','observed','x_MeV']});names=[k for k in T if k not in ['edges_GeV','observed','x_MeV']]
L=pd.read_csv(B/'results/local_tail_tests.csv');D=pd.read_csv(B/'results/asimov_scan.csv');O=pd.read_csv(B/'results/observed_anchors.csv');qa=json.loads((B/'qa/local_validation.json').read_text());failed=qa['failed_scenarios'];seeds=[];records=[]
for _,r in L.iterrows():
 a=np.load(B/f'results/local_{int(r.mass_MeV)}.npz');z=a[r.truth];robs=float(a['observed_r']);n=len(z);raw=int(np.count_nonzero(z>=robs));k=n if robs<=0 else raw
 seed=tuple(int(v) for v in a[r.truth+'_seed']);seeds.append(seed)
 error=0. if robs<=0 else max(abs(binom.sf(k-1,n,r.p_lo95)-.025) if k>0 else abs(r.p_lo95),abs(binom.cdf(k,n,r.p_hi95)-.025) if k<n else abs(r.p_hi95-1))
 upper_error=abs(r.one_sided_upper95-1) if k==n else abs(binom.cdf(k,n,r.one_sided_upper95)-.05)
 q=D[(D.mass_MeV==r.mass_MeV)&(D.truth==r.truth)].iloc[0];o=O[O.mass_MeV==r.mass_MeV].iloc[0]
 good=n==P['new_poisson_per_anchor_truth'] and k==r.exceedances and raw==r.raw_signed_exceedances and abs(r.p_addone-(k+1)/(n+1))<1e-14 and abs(r.p_mle-k/n)<1e-14 and error<1e-10 and upper_error<1e-10 and abs(robs-o.signed_r)<2e-5 and abs(r.asimov_r-q.signed_r)<2e-5 and int(np.sum(z>norm.isf(.05)))==r.raw_nominal05_rejections
 if robs<=0:good=good and r.p_lo95==1 and r.p_hi95==1
 check(f'{int(r.mass_MeV)}_{r.truth}_local_array_and_interval',good)
 records.append(dict(mass_MeV=int(r.mass_MeV),truth=r.truth,raw_signed_exceedances=raw,inclusive_q0_exceedances=k,binomial_interval_inversion_error=error,one_sided_upper_inversion_error=upper_error))
for r in failed:seeds.append(tuple(r['seed']))
check('unique_seed_streams_including_failed',len(set(seeds))==len(seeds))
expected={(m,t) for m in P['anchors_MeV'] for t in names};success={(int(r.mass_MeV),r.truth) for _,r in L.iterrows()};failure={(r['mass_MeV'],r['truth']) for r in failed}
check('all_declared_scenarios_accounted',success.isdisjoint(failure) and success|failure==expected)
check('failed_scenarios_have_no_reported_probability',all(not r['probability_reported'] for r in failed))
check('completed_experiments_count',qa['new_poisson_experiments']==len(L)*P['new_poisson_per_anchor_truth'])
A=np.load(B/'truths/source_absorption.npz');S=pd.read_csv(B/'truths/source_absorption.csv');abs_records=[]
for _,r in S.iterrows():
 mass=int(r.mass_MeV);inj=A[f'injection_{mass}'];mask=A[f'signal_window_{mass}'];delta=A[f'{r.truth}_{mass}']-T[r.truth]
 frac=float(delta[mask].sum()/inj[mask].sum());l2=float(delta[mask]@inj[mask]/(inj[mask]@inj[mask]));weighted=float((inj[mask]/T[r.truth][mask])@delta[mask]/np.sum(inj[mask]**2/T[r.truth][mask]))
 check(f'{mass}_{r.truth}_source_absorption_recounted',abs(frac-r.source_absorbed_window_sum_fraction)<1e-10 and abs(l2-r.template_L2_projection_fraction)<1e-10 and abs(weighted-r.template_poisson_projection_fraction)<1e-10 and np.all(A[f'{r.truth}_{mass}']>0))
 abs_records.append(dict(mass_MeV=mass,truth=r.truth,window_sum_fraction=frac,L2_template_projection_fraction=l2,Poisson_template_projection_fraction=weighted,all_support_sum_fraction=float(delta.sum()/inj.sum())))
G=np.load(B/'truths/block_geometry.npz');x=500*(G['edges_GeV'][1:]+G['edges_GeV'][:-1]);w=G['weights'];sig=1000*np.polynomial.polynomial.polyval(x/1000,[.00038,.041,-.27,3.49,-11.11]);search=(x>=39)&(x<=180)
check('primary_block_partition_valid',np.min(w)>=0 and np.max(abs(w.sum(axis=0)-1))<1e-12)
check('primary_block_exclusion_geometry',np.min(G['minimum_exclusion_clearance_MeV'][search]-2.25*sig[search])>=0)
edge=np.array(json.loads((B/'truths/supplementary_manifest.json').read_text())['edge_self_fit_weight']);detail['supplementary_edge_self_fit_mass_range']=dict(lowest_MeV=float(x[edge>0].min()),highest_MeV=float(x[edge>0].max()),search_masses_with_some_self_fit_count=int(np.sum(search&(edge>0))))
files=[B/'truths/source_absorption.npz',B/'truths/source_absorption.csv',B/'results/local_tail_tests.csv',B/'qa/local_validation.json']
result=dict(passed=all(checks.values()),execution_complete=qa['complete'],all_requested_scenarios_numerically_qualified=not failed,checks=checks,completed_scenarios=len(L),failed_scenarios=failed,completed_null_experiments=len(L)*P['new_poisson_per_anchor_truth'],local_records=records,source_absorption=abs_records,details=detail,input_sha256={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},interpretation='Pass means saved arithmetic, accounting and conventions are verified. Failed numerical scenarios stay unavailable; no physical null or discovery calibration is conferred.')
(OUT/'statistics_local_source_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['checks','local_records','input_sha256','source_absorption']},indent=2));assert result['passed']
