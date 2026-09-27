#!/usr/bin/env python3
"""Independent interpolation calibration versus omitted direct-MC evaluation.

This diagnostic is fixed before the primary evaluation is inspected. It does
not select or retune a window. Calibration injects the predicted common or
neighboring template; evaluation remains the independent full direct MC.
"""
from pathlib import Path
import json, hashlib, time
import run_study as R
import numpy as np
import pandas as pd

B=R.B
POLICIES=('common_starter','morph_starter')
def main():
    start=time.monotonic()
    spec=dict(version='6.3.7',purpose='Frozen interpolation-mismatch diagnostic, no window optimization',
        source='gp_mean',calibration_signal='Omitted-mass predicted common or neighboring full-selected MC CDF',
        evaluation_signal='Independent direct full selected TC MC at the omitted mass',
        masses=list(R.MASSES),policies=list(POLICIES),calibration_toys=R.N,grid=list(R.GRID),
        background_seed='[master,4,0,0,toy]',signal_seed='[master,4,policy_index+1,mass,toy]',
        protocol_sha256=R.sha(B/'provenance/toy_protocol.json'),script_sha256=R.sha(__file__),
        inference='p_A=(1+# calibration Ahat<=evaluation Ahat)/101; accept p_A>0.1',
        endpoint_convention='Median largest accepted grid value among nonempty sets; report empty and censor counts alongside; no interpolation')
    pp=B/'provenance/calibration_mismatch_protocol.json'
    if pp.exists(): assert json.loads(pp.read_text())==spec
    else:R.write(pp,spec)
    truth=R.TRUTHS['gp_mean'];pieces=[]
    backgrounds=np.array([np.random.default_rng(np.random.SeedSequence([R.SEED,4,0,0,i])).poisson(truth) for i in range(R.N)])
    for mass in R.MASSES:
        for pi,policy in enumerate(POLICIES):
            path=B/f'results/checkpoints/mismatch_{policy}_m{mass}.csv';marker=path.with_suffix('.json')
            if marker.exists():
                meta=json.loads(marker.read_text());assert meta['protocol_sha256']==R.sha(pp) and meta['rows_sha256']==R.sha(path)
                pieces.append(pd.read_csv(path,float_precision='round_trip'));continue
            ctx=R.context(mass,policy);shape=R.CANDIDATES[policy]['shape']
            cats=R.T.categories(mass,R.D['edges']*1000,method=shape,omit=mass)
            s0=float(R.REF[str(mass)]['s0']);rows=[];draws=[]
            for toy in range(R.N):
                rng=np.random.default_rng(np.random.SeedSequence([R.SEED,4,pi+1,mass,toy]))
                draw=np.zeros(len(cats),dtype=np.int64);previous=0.
                for z in R.GRID:
                    draw+=rng.poisson((z-previous)*s0*cats);previous=z
                    counts=backgrounds[toy]+draw[1:-1];draws.append(draw.copy())
                    row=dict(source='gp_mean',mass_MeV=mass,policy=policy,toy=toy,z=z,A_expected=z*s0,
                        background_hash=R.ahash(backgrounds[toy]),signal_hash=R.ahash(draw),actual_full=int(draw.sum()),
                        actual_training=int(draw[1:-1][~ctx.guard].sum()),actual_outside_support=int(draw[0]+draw[-1]))
                    row.update(ctx.fit_counts(counts));rows.append(row)
            frame=pd.DataFrame(rows);R.atomic_csv(path,frame)
            np.savez_compressed(path.with_suffix('.npz'),backgrounds=backgrounds,signal_categories=np.array(draws),probabilities=cats)
            R.write(marker,dict(protocol_sha256=R.sha(pp),rows_sha256=R.sha(path),draws_sha256=R.sha(path.with_suffix('.npz')),rows=len(frame),valid=int(frame.fit_valid.sum())))
            assert frame.fit_valid.all();pieces.append(frame);print('mismatch calibration',mass,policy,'complete',flush=True)
    cal=pd.concat(pieces,ignore_index=True);R.atomic_csv(B/'results/mismatch_calibration_rows.csv',cal)
    # Waiting here does not run or alter the independent primary evaluation.
    ep=B/'results/evaluation_rows.csv'
    if not ep.exists():
        print('Calibration complete; run again after primary evaluation exists.',flush=True);return
    evaluation=pd.read_csv(ep,float_precision='round_trip');evaluation=evaluation[(evaluation.source=='gp_mean')&evaluation.policy.isin(POLICIES)]
    rows=[]
    for (mass,policy),q in evaluation.groupby(['mass_MeV','policy']):
        c=cal[(cal.mass_MeV==mass)&(cal.policy==policy)]
        byz={float(z):v.Ahat.to_numpy() for z,v in c.groupby('z')}
        for r in q.itertuples(index=False):
            ps=np.array([(1+np.count_nonzero(byz[z]<=r.Ahat))/(R.N+1) for z in R.GRID])
            accepted=np.flatnonzero(ps>.1);empty=len(accepted)==0
            p0=(1+np.count_nonzero(byz[0]>=r.Ahat))/(R.N+1)
            rows.append(dict(source='gp_mean',mass_MeV=mass,policy=policy,toy=r.toy,z=r.z,A_expected=r.A_expected,
                accepted_z_json=json.dumps([R.GRID[i] for i in accepted]),p_A_json=json.dumps(ps.tolist()),
                empty=empty,holes=False if empty else len(accepted)!=(accepted[-1]-accepted[0]+1),
                upper_grid_censored=bool(not empty and accepted[-1]==len(R.GRID)-1),
                largest_accepted_A=None if empty else R.GRID[accepted[-1]]*r.s0,
                true_grid_rejected=bool(ps[list(R.GRID).index(float(r.z))]<=.1),
                p_background_only=p0,reject_background_only=bool(p0<=.1)))
    inf=pd.DataFrame(rows);R.atomic_csv(B/'results/mismatch_inference.csv',inf)
    summaries=[]
    for (mass,policy,z),q in inf.groupby(['mass_MeV','policy','z']):
        n=len(q);k=int(q.true_grid_rejected.sum());lo,hi=R.cp(k,n);power=int(q.reject_background_only.sum());plo,phi=R.cp(power,n)
        summaries.append(dict(source='gp_mean',mass_MeV=mass,policy=policy,z=z,toys=n,
            true_grid_rejected_count=k,true_grid_rejected_fraction=k/n,true_grid_rejected95_low=lo,true_grid_rejected95_high=hi,
            reject_background_only_count=power,reject_background_only_fraction=power/n,reject95_low=plo,reject95_high=phi,
            empty_count=int(q['empty'].sum()),holes_count=int(q.holes.sum()),upper_grid_censored_count=int(q.upper_grid_censored.sum()),
            median_largest_accepted_A_among_nonempty=float(q.largest_accepted_A.median()),nonempty_count=int((~q['empty']).sum())))
    R.atomic_csv(B/'results/mismatch_summary.csv',pd.DataFrame(summaries))
    R.write(B/'qa/calibration_mismatch.json',dict(passed=bool(cal.fit_valid.all()),calibration_rows=len(cal),evaluation_rows=len(inf),
        calibration_namespace=4,protocol_sha256=R.sha(pp),rows_sha256=R.sha(B/'results/mismatch_calibration_rows.csv'),
        primary_evaluation_rows_sha256=R.sha(ep),runtime_seconds=time.monotonic()-start,
        limitation='Direct detector-MC truth only at known anchors; no independent 90 MeV detector sample, no global-significance or physical-limit claim'))

if __name__=='__main__':main()
