"""Separate fixed-background recovery and rebuilt-source absorption, 2021 control."""
from pathlib import Path
import os,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1]
P=B/'inputs/v5p8p2_nominal_gp_significance_20260917'
if not P.exists():P=ROOT/'study_results/v5p8p2_nominal_gp_significance_20260917'
sys.path.insert(0,str(P/'scripts'))
import engine as E
import numpy as np,pandas as pd
R=B/'results';data=np.load(R/'physics_sources.npz');x=data['x'];n1=data['high1_observed'];b1=data['high1_gp'];c,l=E.C.kernel_state('2021',76.)
DEADLINE=time.time()+120
def check():
    if (B/'STOP').exists() or time.time()>=DEADLINE:raise SystemExit('Resource stop')
rows=[]
for mass in [76.,90.]:
    check();ctx=E.Context('2021',mass);truth=10*b1
    f0=E.fit([ctx.predict(truth)]);S=E.C.signal('2021',mass);A=5*f0['f']['sigma'];added=A*S
    fixed=E.fit([ctx.predict(truth+added)])
    rebuilt,_=E.C.predict(x,n1+added/10,np.zeros(len(x),bool),c,l,query=x)
    delta=10*(rebuilt-b1);mask=ctx.mask
    rows.append(dict(mass_MeV=mass,target_injection_sigma=5,target_injected_full_yield=float(added.sum()),target_injected_window_yield=float(added[mask].sum()),fixed_source_fit_recovery=float((fixed['f']['A']-f0['f']['A'])/A),rebuilt_source_window_absorption=float(delta[mask].sum()/added[mask].sum()),rebuilt_source_template_absorption=float(np.sum(delta*S/truth)/np.sum(added*S/truth)),all_support_change_fraction=float(delta.sum()/added.sum()),source_level_injection_scale=0.1,max_fit_score=max(f0['score'],fixed['score'])))
pd.DataFrame(rows).to_csv(R/'physics_signal_transfer.csv',index=False,float_format='%.17g');print(pd.DataFrame(rows).to_string(index=False))
