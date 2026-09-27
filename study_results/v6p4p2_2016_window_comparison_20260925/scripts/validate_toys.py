"""Independent source, seed, category and inversion audit; no new toy cohort."""
from pathlib import Path
import hashlib
import json
import traceback
import extraction as E
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import beta

B=Path(__file__).resolve().parents[1]
REPORT=dict(passed=False,checks=0,scope='Fixed-source pointwise checks after observed mass selection; no coverage evaluation or global correction.')


def check(condition,message):
    REPORT['checks']+=1
    if not bool(condition):raise AssertionError(message)


def digest(data):
    return hashlib.sha256(b''.join(np.ascontiguousarray(data[y]).tobytes() for y in sorted(data))).hexdigest()


def read(name):
    return pd.read_csv(B/'results'/name,float_precision='round_trip',dtype={'scope':str})


def ref(frame,scope,mass,method):
    rows=frame[(frame.scope==scope)&(frame.mass_MeV==mass)&(frame.method==method)]
    check(len(rows)==1,'Expected unique summary row')
    return rows.iloc[0]


def ctxmap(scope,mass):
    ys=['2016'] if scope=='2016' else E.years(mass)
    methods=list(E.KINDS2016) if scope=='2016' else list(E.METHODS)
    keys={('2016',k) for k in methods} if scope=='2016' else {
        (y,'mc' if y=='2021' and method!='all_gaussian' or y=='2016' and method=='mc2016_2021' else 'gaussian')
        for method in methods for y in ys}
    return ys,methods,{key:E.Context(key[0],mass,key[1]) for key in keys}


def selected(scope,method,ys,contexts):
    keys=[('2016',method)] if scope=='2016' else [
        (y,'mc' if y=='2021' and method!='all_gaussian' or y=='2016' and method=='mc2016_2021' else 'gaussian') for y in ys]
    return [contexts[k] for k in keys]


def replay(scope,mass,method,ys,contexts,counts,row,toy,z=None):
    parts=[ctx.part(counts[ctx.year]) for ctx in selected(scope,method,ys,contexts)]
    model,n=E.model(parts);free=model.fit(n);null=model.fit(n,fixed=0,initial=free['theta'])
    q=2*(null['nll']-free['nll']);check(q>=-2e-6,'Replay likelihood nesting failure')
    root=float(np.sign(free['A'])*np.sqrt(max(0.,q)))
    yield_error=abs(float(free['A']-row.psi_hat));root_error=abs(root-float(row.signed_root))
    check(yield_error<2e-7*max(1.,abs(row.psi_hat)) and root_error<2e-6,'Toy fit replay mismatch')
    return dict(scope=scope,mass_MeV=mass,method=method,toy=int(toy),
                z=z,yield_absolute_difference=yield_error,
                signed_root_absolute_difference=root_error)


def main():
    local=read('local_toy_rows.csv');local_summary=read('local_checks.csv')
    ranks=read('rank_calibration_rows.csv');rank_summary=read('rank_limits.csv');rank_grid=read('rank_grid.csv')
    observed=read('observed_scan.csv')
    lp=json.loads((B/'provenance/local_check_protocol.json').read_text())
    rp=json.loads((B/'provenance/rank_limit_protocol.json').read_text())
    lhash=E.sha(B/'provenance/local_check_protocol.json');rhash=E.sha(B/'provenance/rank_limit_protocol.json')
    check(lp['script_sha256']==E.sha(B/'scripts/run_local_checks.py'),'Local protocol script hash mismatch')
    check(rp['script_sha256']==E.sha(B/'scripts/run_rank_limits.py'),'Rank protocol script hash mismatch')
    check(lp['observed_protocol_sha256']==rp['observed_protocol_sha256']==E.sha(B/'provenance/protocol.json'),'Observed protocol linkage changed')
    check(json.loads((B/'qa/rank_limits.json').read_text())['protocol_sha256']==rhash,'Rank run has not finished under current protocol')
    truths={y:np.load(B/f'inputs/null_{y}.npz')['truth'] for y in ('2015','2016','2021')}
    for y,truth in truths.items():
        check(lp['truths'][y]==E.sha(B/f'inputs/null_{y}.npz'),'Null source hash mismatch')
        check(truth.shape==E.C.DATA[y]['n'].shape and np.all(np.isfinite(truth)) and np.all(truth>=0),'Invalid fixed null source')
    for folder,signature in [('toy_checkpoints',lhash),('rank_checkpoints',rhash)]:
        paths=list((B/'results'/folder).glob('*.json'));check(len(paths)==30,'Unexpected checkpoint count')
        for path in paths:check(json.loads(path.read_text())['signature']==signature,'Stale checkpoint signature')
    check(len(local)==5000 and len(ranks)==21600,'Wrong toy-row count')
    check(not local.duplicated(['scope','mass_MeV','method','toy']).any(),'Duplicated null rows')
    check(not ranks.duplicated(['scope','mass_MeV','method','toy','z']).any(),'Duplicated calibration rows')
    for d in [local,ranks]:
        check(d.valid.all() and np.isfinite(d[['psi_hat','sigma_psi','signed_root','q0','max_score','min_lambda']].to_numpy()).all(),'Invalid toy fit')
        check((d.min_lambda>0).all() and (d.max_score<3e-5).all(),'Toy stationarity or positivity failure')
        check(np.allclose(d.q0,np.maximum(d.signed_root,0)**2,atol=1e-12,rtol=0),'Toy q0 differs from signed root')
        check(np.allclose(d.p0_asymptotic,ndtr(-np.maximum(d.signed_root,0)),atol=2e-15,rtol=0),'Toy local asymptotic p mismatch')
    check((ranks.realized_full_signal==ranks.realized_signal_in_fit+ranks.realized_signal_in_training+ranks.realized_signal_outside_support).all(),'Incomplete realized signal partition')
    for _,q in ranks.groupby(['scope','mass_MeV','toy','z']):
        check(len(q)==3 and all(q[k].nunique()==1 for k in ['background_hash','signal_hash','counts_hash','realized_full_signal','realized_signal_outside_support','psi_expected']),'Unpaired rank methods')

    tails=[]
    for (scope,mass,method),q in local.groupby(['scope','mass_MeV','method']):
        o=ref(observed,scope,mass,method);s=ref(local_summary,scope,mass,method)
        check(len(q)==lp['toys'] and set(q.toy)==set(range(lp['toys'])),'Incomplete null cohort')
        k=int(np.count_nonzero(q.q0>=o.q0));n=len(q)
        low=0. if k==0 else float(beta.ppf(.025,k,n-k+1))
        high=1. if k==n else float(beta.ppf(.975,k+1,n-k))
        check(k==s.tail_count and abs(s.p_rank-(k+1)/(n+1))<1e-15,'Null exceedance/rank mismatch')
        check(abs(s.cp95_low-low)<1e-15 and abs(s.cp95_high-high)<1e-15,'Clopper-Pearson interval mismatch')
        check(abs(s.background_mean_psi-q.psi_hat.mean())<1e-11 and abs(s.background_sd_psi-q.psi_hat.std(ddof=1))<1e-11,'Null yield summary mismatch')
        check(abs(s.mean_signed_root-q.signed_root.mean())<1e-14 and abs(s.sd_signed_root-q.signed_root.std(ddof=1))<1e-14,'Null root summary mismatch')
        tails.append(dict(scope=scope,mass_MeV=int(mass),method=method,tail_count=k,toys=n,p_rank=float(s.p_rank),cp95=[low,high]))

    grid=rp['strength_grid'];N=rp['calibration_toys'];inversions=[]
    for (scope,mass,method),q in ranks.groupby(['scope','mass_MeV','method']):
        o=ref(observed,scope,mass,method);s=ref(rank_summary,scope,mass,method)
        s0=rp['reference_scales_psi'][f'{scope}_{mass}'];pvalues=[]
        check(len(q)==N*len(grid) and set(q.z)==set(grid),'Incomplete rank grid')
        check(np.allclose(q.psi_expected,q.z*s0,atol=0,rtol=2e-15),'Expected shared signal strength mismatch')
        for z in grid:
            a=q[q.z==z];check(len(a)==N and set(a.toy)==set(range(N)),'Incomplete grid cell')
            k=int(np.count_nonzero(a.psi_hat<=o.psi_hat));p=(k+1)/(N+1);pvalues.append(p)
            stored=rank_grid[(rank_grid.scope==scope)&(rank_grid.mass_MeV==mass)&(rank_grid.method==method)&(rank_grid.z==z)]
            check(len(stored)==1,'Missing grid summary')
            rr=stored.iloc[0]
            check(rr.lower_tail_count==k and abs(rr.p_rank-p)<1e-15 and bool(rr.accepted)==(p>.1),'Wrong rank ordering or acceptance')
        indices=np.flatnonzero(np.array(pvalues)>.1);accepted=[grid[i] for i in indices]
        empty=len(indices)==0;holes=False if empty else len(indices)!=(indices[-1]-indices[0]+1)
        censored=bool(not empty and indices[-1]==len(grid)-1)
        check(json.loads(s.accepted_z_json)==accepted and np.max(abs(np.array(json.loads(s.p_rank_json))-pvalues))<1e-15,'Accepted set differs from stored inversion')
        check(bool(s.empty)==empty and bool(s.holes)==holes and bool(s.upper_grid_censored)==censored,'Incorrect inversion flags')
        endpoint=np.nan if empty else grid[indices[-1]]*s0
        nextpoint=np.nan if empty or censored else grid[indices[-1]+1]*s0
        check(np.isclose(s.largest_accepted_psi,endpoint,rtol=2e-15,atol=0,equal_nan=True),'Accepted endpoint mismatch')
        check(np.isclose(s.next_grid_psi,nextpoint,rtol=2e-15,atol=0,equal_nan=True),'Next-grid bracket mismatch')
        check(np.isclose(s.largest_accepted_epsilon2,endpoint*1e-8*E.branch(mass),rtol=2e-15,atol=0,equal_nan=True),'Endpoint display conversion mismatch')
        if scope=='2016':check(np.isclose(s.largest_accepted_full_yield_2016,endpoint*E.conversion('2016',mass),rtol=2e-15,atol=0,equal_nan=True),'Full selected yield conversion mismatch')
        inversions.append(dict(scope=scope,mass_MeV=int(mass),method=method,accepted_z=accepted,empty=empty,holes=bool(holes),upper_grid_censored=censored,largest_accepted_psi=None if empty else endpoint,next_grid_psi=None if np.isnan(nextpoint) else nextpoint))

    replays=[];local_draws=0;rank_draws=0;namespace_checks=0
    for scope,mass in lp['targets']:
        ys,methods,contexts=ctxmap(scope,mass);methods=['mc'] if scope=='2016' else methods
        q=local[(local.scope==scope)&(local.mass_MeV==mass)].set_index(['toy','method'])
        for toy in range(lp['toys']):
            counts={y:np.random.default_rng(np.random.SeedSequence([lp['seed'],int(y),toy])).poisson(truths[y]) for y in ys}
            h=digest(counts)
            for method in methods:check(q.loc[toy,method].counts_hash==h,'Null seed/draw mismatch')
            if toy==0:
                for method in methods:replays.append(replay(scope,mass,method,ys,contexts,counts,q.loc[toy,method],toy))
            local_draws+=1
    for scope,mass in rp['targets']:
        ys,methods,contexts=ctxmap(scope,mass)
        s0=rp['reference_scales_psi'][f'{scope}_{mass}']
        baseline='gaussian' if scope=='2016' else 'all_gaussian'
        parts=[ctx.part(truths[ctx.year]) for ctx in selected(scope,baseline,ys,contexts)]
        model,n=E.model(parts);f=model.fit(n)
        check(abs(f['sigma']/s0-1)<1e-10,'Reference scale not the fixed-source Gaussian Hessian error')
        cats={y:E.Context(y,mass,'gaussian' if y=='2015' else 'mc').categories for y in ys}
        factors={y:E.conversion(y,mass) for y in ys}
        for y in ys:check(np.min(cats[y])>=0 and abs(cats[y].sum()-1)<1e-12,'Invalid injected full categories')
        q=ranks[(ranks.scope==scope)&(ranks.mass_MeV==mass)].set_index(['toy','z','method'])
        critical_z=4. if scope=='2016' and mass==69 else 5. if scope=='2016' else 6.
        for toy in range(N):
            bg={y:np.random.default_rng(np.random.SeedSequence([rp['seed'],rp['background_namespace'],int(y),toy])).poisson(truths[y]) for y in ys}
            rng={y:np.random.default_rng(np.random.SeedSequence([rp['seed'],rp['signal_namespace'],mass,int(y),toy])) for y in ys}
            signal={y:np.zeros(len(cats[y]),dtype=np.int64) for y in ys};previous=0.;bh=digest(bg)
            for y in ys:
                eval_bg=np.random.default_rng(np.random.SeedSequence([lp['seed'],int(y),toy])).poisson(truths[y])
                check(not np.array_equal(bg[y],eval_bg),'Calibration/evaluation draw collision');namespace_checks+=1
            for z in grid:
                for y in ys:signal[y]+=rng[y].poisson((z-previous)*s0*factors[y]*cats[y])
                previous=z;counts={y:bg[y]+signal[y][1:-1] for y in ys};sh=digest(signal);ch=digest(counts)
                for method in methods:
                    row=q.loc[toy,z,method]
                    check(row.background_hash==bh and row.signal_hash==sh and row.counts_hash==ch,'Rank seed/category draw mismatch')
                    active=selected(scope,method,ys,contexts)
                    full=sum(int(signal[y].sum()) for y in ys)
                    fitted=sum(int(signal[c.year][1:-1][c.fit].sum()) for c in active)
                    training=sum(int(signal[c.year][1:-1][~c.guard].sum()) for c in active)
                    outside=sum(int(signal[y][0]+signal[y][-1]) for y in ys)
                    check(row.realized_full_signal==full and row.realized_signal_in_fit==fitted and row.realized_signal_in_training==training and row.realized_signal_outside_support==outside,'Reconstructed signal partition mismatch')
                    if toy==0 and z==critical_z:replays.append(replay(scope,mass,method,ys,contexts,counts,row,toy,z))
                rank_draws+=1
    check(E.sha(B/'provenance/rank_limit_protocol.json')==rhash,'Protocol changed during validation')
    REPORT.update(passed=True,null_rows=len(local),rank_rows=len(ranks),null_experiments_regenerated=local_draws,
        calibration_grid_experiments_regenerated=rank_draws,calibration_evaluation_namespace_checks=namespace_checks,
        fixed_source_hashes_checked=True,common_full_MC_injection_across_methods=True,all_yield_partitions_reconstructed=True,
        paired_background_and_signal_draws_verified=True,independent_calibration_and_local_null_namespaces=True,
        finite_sample_floor=dict(null_rank_probability=1/(lp['toys']+1),calibration_rank_probability=1/(N+1)),
        null_tail_summaries=tails,rank_inversions=inversions,targeted_replays=replays,
        interpretation='Null-source checks use the selected observed masses. The rank endpoint is the largest accepted tested strength, with the next tested rejected point saved; it is not a continuously solved or empirically coverage-validated limit.',
        hashes=dict(local_protocol_sha256=lhash,rank_protocol_sha256=rhash,
            local_rows_sha256=E.sha(B/'results/local_toy_rows.csv'),rank_rows_sha256=E.sha(B/'results/rank_calibration_rows.csv'),
            rank_summary_sha256=E.sha(B/'results/rank_limits.csv'),script_sha256=E.sha(__file__)))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        REPORT['failure']=f'{type(exc).__name__}: {exc}';REPORT['traceback']=traceback.format_exc();raise
    finally:
        E.write(B/'qa/toy_validation.json',REPORT)
        print(json.dumps({k:REPORT[k] for k in ['passed','checks']},indent=2),flush=True)
