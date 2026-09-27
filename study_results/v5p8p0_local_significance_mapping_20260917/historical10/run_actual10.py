"""Actual historical 2016 subset: fixed parent-policy deterministic diagnostics.

No random draws; the scaled stress shape is not the actual 10% observation.
Each local-GP control is mass-specific and cannot be joined into a global null.
"""
from pathlib import Path
import os, sys, json, hashlib, shutil, time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent
REPO=B.parents[2]
PARENT=REPO/'study_results/v5p5p3_92mev_profile_combination_20260912'
SOURCE=REPO/'study_results/v4p9p7_2016_support_combined_100toy_20260902/inputs/source_2016_10pct.root'
EXPECTED='789e619fcbeb5e81f9193d3e224bc17919983477a037bf3d79692327555f9fd4'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={}
for sub,names in [('engine',['common.py','parent_core.py','limit_solver.py']),('inputs',['scopes.json','spectrum_2015.npz','spectrum_2016.npz','spectrum_2021.npz'])]:
    (B/sub).mkdir(exist_ok=True)
    for name in names:
        dest=B/sub/name
        if not dest.exists():shutil.copy2(PARENT/sub/name,dest)
        manifest[str(dest.relative_to(B))]={'source':str(PARENT/sub/name),'sha256':sha(dest)}
dest=B/'inputs/source_2016_10pct.root'
if not dest.exists():shutil.copy2(SOURCE,dest)
assert sha(dest)==EXPECTED
manifest[str(dest.relative_to(B))]={'source':str(SOURCE),'sha256':sha(dest)}
sys.path.insert(0,str(B/'engine'))
from common import DATA, moving_context, OneSignalProfile, predict
import numpy as np, pandas as pd, uproot
from scipy.stats import norm

with uproot.open(dest) as f:y,edges=f['h_Minv_General_Final_1'].to_numpy()
old=DATA['2016'];new=dict(old)
indices=np.array([np.argmin(abs(edges-x)) for x in old['edges']])
assert np.max(abs(edges[indices]-old['edges']))<1e-12
new['n']=np.diff(np.r_[0.,np.cumsum(y)][indices])
new['native_counts']=y.copy();new['native_edges']=edges.copy()
fraction=float(new['n'].sum()/old['n'].sum())
stress=fraction*old['stress']
rows=[];failures=[];started=time.monotonic()

def evaluate(mass,counts):
    p=moving_context('2016',float(mass),counts=counts)
    mod=OneSignalProfile(p['b'],p['L'],p['S'][:,0])
    free=mod.fit(p['n']);zero=mod.fit(p['n'],0.)
    q=2*(zero['nll']-free['nll'])
    if q < -1e-8:raise RuntimeError('Negative likelihood-ratio statistic')
    r=float(np.sign(free['A'])*np.sqrt(max(0.,q)))
    score=float(max(free['score'],zero['score']))
    minb=float(min(free['min_background'],zero['min_background']))
    minlam=float(min(np.min(free['lam']),np.min(zero['lam'])))
    if score>2e-7 or minb<=0 or minlam<=0:raise RuntimeError(f'Fit quality: {score}, {minb}, {minlam}')
    return p,dict(signed_root=r,nominal_local_p=float(norm.sf(max(0.,r))),signal_parameter_1e8=float(free['A']),
        signal_parameter_error_1e8=float(free['sigma']),max_score=score,min_background=minb,min_expectation=minlam,
        covariance_load=float(p['diagnostic']['load']),covariance_rank=int(p['diagnostic']['rank']))

for mass in range(39,181):
    DATA['2016']=new
    try:
        p,observed=evaluate(mass,new['n'])
        _,scaled=evaluate(mass,stress)
        local,_=predict(new['x'],new['n'],p['mask'],p['const'],p['ls'],query=new['x'])
        _,matched=evaluate(mass,local)
        for label,row in [('actual_historical10_observed',observed),('parent_stress_scaled_to_historical10_support_counts',scaled),('historical10_mass_specific_local_gp_mean',matched)]:
            rows.append(dict(mass_MeV=mass,lane=label,**row))
        DATA['2016']=old
        _,full=evaluate(mass,old['stress'])
        rows.append(dict(mass_MeV=mass,lane='parent_full_stress_reference',**full))
    except Exception as exc:
        failures.append(dict(mass_MeV=mass,error=repr(exc)))
    finally:DATA['2016']=old
    if mass%10==0 or mass==180:
        pd.DataFrame(rows).to_csv(B/'actual10_scan.csv',index=False,float_format='%.17g')
        print(f'mass={mass}, elapsed={time.monotonic()-started:.1f}s, failures={len(failures)}',flush=True)

frame=pd.DataFrame(rows)
pivot=frame.pivot(index='mass_MeV',columns='lane',values='signed_root')
pivot.to_csv(B/'signed_roots_wide.csv',float_format='%.17g')
delta=pivot['parent_stress_scaled_to_historical10_support_counts']-np.sqrt(fraction)*pivot['parent_full_stress_reference']
validation=dict(no_failures=not failures,complete_rows=len(frame)==568,masses=142,
    matched_prior92_absdiff=float(abs(pivot.loc[92,'actual_historical10_observed']-1.2591697270276649)),
    square_root_support_ratio=float(np.sqrt(fraction)),stress_scaled_minus_sqrtf_full_rms=float(np.sqrt(np.mean(delta**2))),
    stress_scaled_minus_sqrtf_full_absmax=float(np.max(abs(delta))),maximum_covariance_load=float(frame.covariance_load.max()),
    max_optimizer_score=float(frame.max_score.max()),min_background=float(frame.min_background.min()),min_expectation=float(frame.min_expectation.min()))
(B/'validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
summaries={}
for lane,g in frame.groupby('lane'):
    i=g.signed_root.idxmax();h=g.loc[i]
    summaries[lane]=dict(rows=len(g),root_min=float(g.signed_root.min()),root_max=float(g.signed_root.max()),
        root_rms=float(np.sqrt(np.mean(g.signed_root**2))),max_root_mass_MeV=int(h.mass_MeV),
        max_score=float(g.max_score.max()))
summary=dict(passed=not failures and len(frame)==568,elapsed_seconds=time.monotonic()-started,
    support_count_ratio=fraction,subset_support_rows=float(new['n'].sum()),parent_support_rows=float(old['n'].sum()),
    nominal_sample_fraction=.1,exact_luminosity_fraction_verified=False,selection_same_as_parent_verified=False,event_overlap_verified=False,
    policy='Parent reviewed kernel at each mass; count-dependent log-GP update; inherited resolution; 30–210 MeV support; integer39–180 grid',
    stress_policy='Parent full stress shape scaled by measured support-count ratio, not an independently fitted historical10 null',
    local_gp_policy='Mass-specific subset-sideband posterior mean over full support, reconditioned at the same mass; not a coherent global null',
    significance_status='Nominal signed roots and asymptotic p displays; no local-tail or global calibration',
    signal_parameter_status='Inherited radiative fraction and resolution proxy; single-channel root invariant to positive scalar template normalization',
    lanes=summaries,failures=failures,inputs=manifest)
(B/'manifest.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k in ('passed','elapsed_seconds','support_count_ratio','lanes','failures')},indent=2))
