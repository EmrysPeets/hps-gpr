"""Checkpointed observed scans; one worker per dataset, one BLAS thread each."""
from tail_model import *
import argparse,time

def run(year,benchmark=False):
    start=time.monotonic();excluded=[];rows=[]
    dependencies=[B/'protocol.json',B/f'inputs/spectrum_{year}.npz']+[B/'scripts'/name for name in ('tail_model.py','run_scan.py','common.py','parent_core.py','limit_solver.py')]
    fingerprint=hashlib.sha256(''.join(sha(p) for p in dependencies).encode()).hexdigest()
    masses=[{'2015':51,'2016':90,'2021':78}[year]] if benchmark else range(int(LIMITS[year][0]),int(LIMITS[year][1])+1)
    for mass in masses:
        cp=B/f'derived/chunks/{year}_{mass:03d}.json'
        if cp.exists() and not benchmark:
            saved=json.loads(cp.read_text())
            if saved.get('fingerprint')==fingerprint:continue
        records=[];missing=[]
        for window,padding in WINDOWS.items():
            d=DATA[year];sig=sigma(year,mass);lo=mass/1000-padding*sig;hi=mass/1000+padding*sig
            if min(np.sum(d['x']<lo),np.sum(d['x']>hi))<3:
                missing.append(dict(scope=year,mass_MeV=mass,window=window,reason='fewer than three exterior training bins on a side'))
                continue
            part=context(year,[mass],padding=padding,anchor=mass)
            scale=signal_scale(year,mass)
            for family,k in SCENARIOS:
                w,meta=weights(year,mass,k,family);S=w[part['mask']]*scale
                model=OneSignalProfile(part['b'],part['L'],S)
                row=model.limit(part['n'])
                fraction=float(w[part['mask']].sum())
                row.update(scope=year,mass_MeV=mass,window=window,half_window_sigma=padding,
                           family=family,kappa=k,epsilon2_90=row['A90']*1e-8,
                           display_epsilon2_90=row['A90']*1e-8*branching_factor(mass),
                           branching_factor=branching_factor(mass),sigma_MeV=sig*1000,
                           full_yield_90=row['A90']*scale,window_yield_90=row['A90']*scale*fraction,
                           signal_fraction_in_fit=fraction,signal_fraction_in_training=1-fraction,
                           signal_counts_per_amplitude=scale,gp_const=part['const'],gp_ls=part['ls'],
                           gp_fractional_rms=float(np.sqrt(np.mean(np.diag(part['C'])/part['b']**2))),
                           covariance_load=part['diagnostic']['load'],**meta)
                records.append(row)
        if benchmark:print(json.dumps(records,indent=2),flush=True)
        else:write(cp,dict(rows=records,excluded=missing,elapsed_seconds=time.monotonic()-start,fingerprint=fingerprint))
        rows+=records;excluded+=missing
        if mass%10==0:print(year,mass,'new rows',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    if not benchmark:write(B/f'qa/execution_{year}.json',dict(year=year,new_rows=len(rows),seconds=time.monotonic()-start,complete=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',required=True,choices=list(DATA));p.add_argument('--benchmark',action='store_true');a=p.parse_args();run(a.year,a.benchmark)
