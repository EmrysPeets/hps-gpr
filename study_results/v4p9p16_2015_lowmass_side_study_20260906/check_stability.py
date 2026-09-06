#!/usr/bin/env python3
"""Retain all wider kernel ceilings; no choice based on observed p-values."""
import run_study as r
from scipy.stats import chi2
import json
cfg,data,_=r.load_data()
original=r.pd.read_csv(r.HERE/'derived/scan.csv')
rows=[]
for upper in (16.,32.,64.):
    for m in r.MASSES:
        row,arr,notes=r.gp_fit(data,m,cfg,upper_factor=upper,restarts=7)
        row['method']=f'gp_ceiling{int(upper)}'
        rows.append(row)
        r.np.savez_compressed(r.HERE/'derived/fits'/f"{row['method']}_m{m:05.2f}.npz",**arr)
    print('ceiling',upper,'complete',flush=True)
r.pd.DataFrame(rows).to_csv(r.HERE/'derived/kernel_stability.csv',index=False)
summary=json.loads((r.HERE/'derived/summary.json').read_text())
checks=[]
for m in sorted(set(summary['extraction_masses_MeV']+[21.])):
    old=original[(original.method=='expcheb5')&(original.mass_MeV==m)].iloc[0]
    new,_,_=r.poly_fit(data,m,quad_order=16)
    dr=new['r']-old.r
    checks.append(dict(mass_MeV=m,quadrature_delta_r=float(dr),
        poisson_deviance=float(old.poisson_deviance),nominal_dof=int(old.gof_nominal_dof),
        approximate_gof_p=float(chi2.sf(old.poisson_deviance,old.gof_nominal_dof)),passed=bool(abs(dr)<1e-5)))
assert all(x['passed'] for x in checks)
r.dump(r.HERE/'qa/polynomial_quadrature.json',dict(passed=True,checks=checks,
    gof_warning='Nominal chi-square reference for a fit with some low-count bins; not a calibrated model test.'))
