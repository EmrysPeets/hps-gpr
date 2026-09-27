"""Prepare, smoke-test, freeze the independent pilot, then evaluate and resume."""
import argparse, fcntl, json, time, sys, platform
from concurrent.futures import ProcessPoolExecutor, as_completed
import core as I
import numpy as np
import pandas as pd

def protocol():
    return dict(version='6.3.1',dataset='2021 10%',masses_MeV=list(I.MASSES),shapes=list(I.SHAPES),
        primary_cases=['Gaussian generated/Gaussian fitted','native MC generated/native MC fitted'],
        pilot_backgrounds=100,evaluation_backgrounds=100,levels=list(I.LEVELS),master_seed=I.MASTER,
        seed_namespaces=dict(pilot=1,evaluation_background=2,evaluation_signal=3,bootstrap=4,smoke_background=91,smoke_signal=93),
        seed_key='[master,namespace,mass_MeV,toy,shape_id,level_id]; background mass/shape/level=0; shape gaussian=1 mc=2; positive level IDs=1,2,3',
        background_truth='Pinned v6.2 nominal2021 GP arithmetic mean; one fixed source, Poisson counting only',
        background_sha256=I.sha(I.B/'inputs/null_2021.npz'),
        cohorts='100 pilot plus 100 independent evaluation full-support spectra; each cohort reused across all masses; evaluation backgrounds reused across levels and shapes',
        pilot_rule='s0(m)=arithmetic mean of 100 valid Gaussian pilot returned observed profile-Hessian yield errors; freeze once before evaluation',
        expected_yields='A_expected=z*s0(m), same floating-point mean for Gaussian and MC, fixed across evaluation toy IDs',
        signal_generation='Independent Poisson counts in all full-selected categories, including below/above support; no realized-count rescaling',
        signal_units='Full-selected candidate yield; no support/window normalization and no physics conversion',
        gaussian_template='Integrated nominal Gaussian at pole mass; CDF differences plus outside tails',
        mc_template='Exact native mass histogram CDF; inherited uniform within native bins; retained selected normalization and overflow',
        fit_window='pole mass +/- 2.25 sigma_nominal; fixed bin-center mask, complement training, >=3 exterior bins each side',
        GP='Archived mass-dependent kernel fixed; recompute log targets and alpha from each spectrum: positive log(n),1/n; zero 0,1; conditional arithmetic mean and correlated count covariance',
        covariance='Inherited first Cholesky-success loading 1e-10..1e-5 times max(maxdiag,1); retain Poisson-whitened eigenmodes >1e-8',
        likelihood='Poisson means b_GP+L theta+A w; Gaussian nuisance penalty theta^2/2; signed Ahat; positive means; fixed-expected-truth nuisance profile uses same b_GP,L',
        uncertainty='Independently recompute observed profile-Hessian sigma; reject unavailable curvature or Fisher substitution',
        acceptance=dict(score_lt=3e-5,min_lambda_gt=0,sigma_finite_positive=True,q_true_min=-2e-6),
        retry_policy='At most 3 attempts on same counts: origin tolerance2e-7; origin tolerance2e-9 (profile theta0); deterministic free start A/scale=.5 tolerance2e-9. If a profile is available for a failed nesting check use its expected-A point as free start. Select valid minimum objective; preserve attempts. Never regenerate/select by physics outcome.',
        containment=dict(q68=1.0,q95=3.841459,interpretation='nominal signed profile-likelihood sets; valid-decision Clopper-Pearson95% intervals plus all-attempt accounting bounds'),
        resources=dict(local_workers=4,numerical_threads_per_worker=1,wall_timeout_seconds=1800,checkpoint_toys_per_mass=10,remote_compute=False),
        exclusions=['260 MeV','40 MeV','shifted windows','shape mismatch','new widths','kernel reoptimization','additional datasets','global scans'],
        limitations=['Conditional on one background source, fixed selected MC and archived fit prescription.',
            'MC selection equivalence and signal-daughter association unvalidated.',
            'Finite-MC, detector-response and source-estimation uncertainties not propagated.',
            '100 toys per cell: coarse finite-sample precision; cells paired, not independent.',
            'No physical-background adequacy, unconditional coverage, calibrated discovery significance or exclusion claim.'],
        handoff_sha256=I.sha(I.B/'provenance/HANDOFF.md'))

def prepare():
    target=I.B/'protocol.json';spec=protocol()
    if target.exists():
        if json.loads(target.read_text())!=spec:raise RuntimeError('Frozen protocol differs')
    else:I.write_json(target,spec)
    cohorts=I.B/'inputs/cohorts.npz'
    expected={label:np.array([I.rng(I.seed_key(namespace,toy=t)).poisson(I.TRUTH) for t in range(I.NTOYS)])
              for label,namespace in [('pilot',1),('evaluation',2)]}
    if cohorts.exists():
        old=np.load(cohorts)
        for label,a in expected.items():assert np.array_equal(old[label],a)
    else:np.savez_compressed(cohorts,**expected,truth=I.TRUTH,edges_GeV=I.D['edges'])
    hashes={label:[I.array_hash(a) for a in rows] for label,rows in expected.items()}
    assert len(set(hashes['pilot']+hashes['evaluation']))==200
    I.write_json(I.B/'provenance/cohort_hashes.json',hashes)
    contexts=[I.Context(m) for m in I.MASSES]
    categories=np.array([[ctx.categories[s] for s in I.SHAPES] for ctx in contexts])
    masks=np.array([ctx.mask for ctx in contexts])
    templates=I.B/'inputs/templates.npz'
    if templates.exists():
        old=np.load(templates);assert np.array_equal(old['categories'],categories) and np.array_equal(old['masks'],masks)
    else:np.savez_compressed(templates,masses_MeV=I.MASSES,shapes=I.SHAPES,categories=categories,masks=masks,edges_GeV=I.D['edges'])
    fractions=[]
    for ctx in contexts:
        for shape in I.SHAPES:
            p=ctx.categories[shape]
            fractions.append(dict(mass_MeV=ctx.m,shape=shape,support_fraction=float(p[1:-1].sum()),
                window_fraction=float(p[1:-1][ctx.mask].sum()),training_fraction=float(p[1:-1][~ctx.mask].sum()),
                below_support_fraction=float(p[0]),above_support_fraction=float(p[-1]),fit_bins=int(ctx.mask.sum()),
                template_hash=I.array_hash(p),mask_hash=I.array_hash(ctx.mask)))
    pd.DataFrame(fractions).to_csv(I.B/'inputs/template_fractions.csv',index=False,float_format='%.17g')
    I.write_json(I.B/'provenance/runtime.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        numpy=np.__version__,pandas=pd.__version__,created_utc=I.utc()))
    sig=I.signature()
    I.write_json(I.B/'provenance/computation_signature.json',dict(signature=sig,code_hashes={p.name:I.sha(p) for p in (I.B/'scripts/core.py',I.B/'scripts/run_study.py')},
        protocol_sha256=I.sha(target),cohorts_sha256=I.sha(cohorts),templates_sha256=I.sha(templates)))
    return sig

def smoke(sig):
    path=I.B/'qa/smoke.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['signature']==sig and old['passed']:return old
    begin=time.monotonic();rows=[];checks=[]
    for m in (60,140,240):
        ctx=I.Context(m)
        background=I.rng(I.seed_key(91,toy=0)).poisson(I.TRUTH)
        pred=ctx.predict(background)
        null=I.make_row(ctx,background,np.zeros(len(I.TRUTH)+2,dtype=np.int64),0.,0,'gaussian',0,'smoke',prediction=pred)
        assert null['fit_valid'] and null['profile_valid'],null
        scale=null['sigma_postfit']
        for shape in I.SHAPES:
            expected=5*scale;draw=I.signal_draw(ctx,shape,5,0,expected,namespace=93)
            assert np.array_equal(draw,I.signal_draw(ctx,shape,5,0,expected,namespace=93))
            assert np.issubdtype(draw.dtype,np.integer) and draw.min()>=0
            for control in ('primary','known_background'):
                row=I.make_row(ctx,background,draw,expected,0,shape,5,'smoke',s0=scale,control=control)
                row['background_seed_key']=json.dumps(I.seed_key(91,toy=0))
                row['signal_seed_key']=json.dumps(I.seed_key(93,m,0,I.SHAPES.index(shape)+1,3))
                assert row['fit_valid'] and row['profile_valid'],row
                rows.append(row)
            if draw[1:-1][~ctx.mask].sum()>0:
                contaminated=ctx.predict(background+draw[1:-1])
                assert not np.array_equal(pred[0],contaminated[0])
                assert not np.array_equal(pred[1],contaminated[1])
            p=ctx.categories[shape][1:-1][ctx.mask]
            n=I.TRUTH[ctx.mask]+expected*p
            free,fixed,valid,attempts=I.fit_pair(I.TRUTH[ctx.mask],np.zeros((ctx.mask.sum(),0)),p,n,expected,True)
            assert valid and abs(free['A']/expected-1)<1e-7 and abs(fixed['nll']-free['nll'])<1e-7
        changed=background.copy();changed[ctx.mask]+=17
        again=ctx.predict(changed)
        assert np.array_equal(pred[0],again[0]) and np.array_equal(pred[1],again[1])
    pd.DataFrame(rows).to_csv(I.B/'qa/smoke_rows.csv',index=False,float_format='%.17g')
    checks=['separate smoke RNG namespaces; deterministic replay','nonnegative integer category draws',
        'full-selected normalization and fixed same-shape extraction','known-background mean-data yield closure at 60,140,240 MeV for both shapes',
        'fit and truth-profile numerical gates for primary and small known-background controls',
        'training excludes window exactly','injected exterior tails change count-dependent GP mean and covariance']
    out=dict(passed=True,signature=sig,created_utc=I.utc(),checks=checks,rows=len(rows),seconds=time.monotonic()-begin)
    I.write_json(path,out);print(json.dumps(dict(stage='smoke',**out)),flush=True)
    return out

def run_cohort(cohort,workers,sig,reference_sha=None):
    I.write_json(I.B/'status.json',dict(status='running_'+cohort,updated_utc=I.utc(),signature=sig,reference_sha256=reference_sha))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(I.run_chunk,cohort,m,start,min(start+10,I.NTOYS),sig,reference_sha)
                 for m in I.MASSES for start in range(0,I.NTOYS,10)]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    frames=[pd.read_csv(I.chunk_path(cohort,m,start).with_suffix('.csv'),float_precision='round_trip')
            for m in I.MASSES for start in range(0,I.NTOYS,10)]
    df=pd.concat(frames,ignore_index=True).sort_values(['mass_MeV','shape','z','toy'])
    assert len(df)==(2000 if cohort=='pilot' else 8000)
    assert not df.duplicated(['mass_MeV','shape','z','toy']).any()
    df.to_csv(I.B/f'results/{cohort}_rows.csv',index=False,float_format='%.17g')
    return df

def freeze(pilot,sig):
    path=I.B/'pilot_reference.json'
    if path.exists():
        reference=json.loads(path.read_text())
        if reference['signature']!=sig or reference['pilot_rows_sha256']!=I.sha(I.B/'results/pilot_rows.csv'):
            raise RuntimeError('Frozen pilot reference does not match completed pilot')
        return reference
    masses={}
    for m in I.MASSES:
        cell=pilot[pilot.mass_MeV==m];stats={}
        for shape in I.SHAPES:
            q=cell[cell['shape']==shape]
            assert set(q.toy)==set(range(100)) and len(q)==100
            valid=q[q.fit_valid]
            if shape=='gaussian' and len(valid)!=100:
                raise RuntimeError(f'Gaussian pilot incomplete at {m}; do not freeze fewer than100 valid errors')
            errors=valid.sigma_postfit
            stats[shape]=dict(attempted=100,fit_valid=len(valid),mean_sigma=float(errors.mean()) if len(valid) else None,
                sd_sigma=float(errors.std(ddof=1)) if len(valid)>1 else None,
                se_mean_sigma=float(errors.std(ddof=1)/np.sqrt(len(valid))) if len(valid)>1 else None,
                cv_sigma=float(errors.std(ddof=1)/errors.mean()) if len(valid)>1 else None)
        s0=stats['gaussian']['mean_sigma']
        masses[str(m)]=dict(s0=s0,expected_yields={str(z):float(z*s0) for z in I.LEVELS},shape_errors=stats,
            mc_gaussian_mean_error_ratio=stats['mc']['mean_sigma']/s0 if stats['mc']['mean_sigma'] else None)
    reference=dict(frozen_utc=I.utc(),signature=sig,pilot_rows_sha256=I.sha(I.B/'results/pilot_rows.csv'),
        cohort_sha256=I.sha(I.B/'inputs/cohorts.npz'),template_sha256=I.sha(I.B/'inputs/templates.npz'),
        protocol_sha256=I.sha(I.B/'protocol.json'),masses=masses)
    I.write_json(path,reference)
    I.write_json(I.B/'provenance/pilot_reference.sha256.json',dict(sha256=I.sha(path),frozen_utc=reference['frozen_utc']))
    print(json.dumps(dict(stage='pilot_frozen',sha256=I.sha(path),s0={m:r['s0'] for m,r in masses.items()})),flush=True)
    return reference

def representatives():
    reference=json.loads((I.B/'pilot_reference.json').read_text())
    cohorts=np.load(I.B/'inputs/cohorts.npz')
    out=I.B/'results/representative';out.mkdir(exist_ok=True)
    for m in (60,140,240):
        ctx=I.Context(m);background=cohorts['evaluation'][0];clean=ctx.predict(background)
        for shape in I.SHAPES:
            expected=5*reference['masses'][str(m)]['s0'];draw=I.signal_draw(ctx,shape,5,0,expected)
            signal=draw[1:-1];counts=background+signal;b,L,diag=ctx.predict(counts)
            free,_,valid,_=I.fit_pair(b,L,ctx.categories[shape][1:-1][ctx.mask],counts[ctx.mask],expected,True)
            assert valid
            np.savez_compressed(out/f'm{m:03d}_{shape}.npz',mass_MeV=m,shape=shape,toy=0,z=5,A_expected=expected,Ahat=free['A'],
                sigma_postfit=free['sigma'],s0=reference['masses'][str(m)]['s0'],edges_GeV=I.D['edges'],truth=I.TRUTH,
                background=background,signal=signal,counts=counts,mask=ctx.mask,categories=ctx.categories[shape],
                gp_mean=b,clean_gp_mean=clean[0],gp_sigma=np.sqrt(np.sum(L*L,axis=1)),bfit=free['bfit'],lambda_fit=free['lam'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--stage',choices=['prepare','smoke','pilot','all'],default='all')
    args=ap.parse_args();assert 1<=args.workers<=4
    with (I.B/'run.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('An existing study run owns the lock; no duplicate launch')
        sig=prepare()
        if args.stage=='prepare':return
        smoke(sig)
        if args.stage=='smoke':return
        pilot=run_cohort('pilot',args.workers,sig)
        freeze(pilot,sig)
        if args.stage=='pilot':return
        evaluation=run_cohort('evaluation',args.workers,sig,I.sha(I.B/'pilot_reference.json'))
        representatives()
        I.write_json(I.B/'status.json',dict(status='numerical_complete',updated_utc=I.utc(),signature=sig,
            pilot_rows=len(pilot),evaluation_rows=len(evaluation),pilot_fit_valid=int(pilot.fit_valid.sum()),
            evaluation_fit_valid=int(evaluation.fit_valid.sum()),evaluation_profile_valid=int(evaluation.profile_valid.sum())))
        print(json.dumps(dict(stage='numerical_complete',rows=len(evaluation))),flush=True)

if __name__=='__main__':main()
