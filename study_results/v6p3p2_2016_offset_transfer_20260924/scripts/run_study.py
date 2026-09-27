"""Frozen three-cohort exposure transfer, deterministic replay and bounded workers."""
import argparse,fcntl,json,time,sys,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
import core as I
import numpy as np
import pandas as pd

def protocol():
    return dict(version='6.3.2',dataset='2016 controlled 0.1 and 1.0 exposure',master_seed=I.MASTER,
        masses_MeV=list(I.MASSES),exposures=list(I.EXPOSURES),sources=list(I.SOURCES),signal='nominal2016 integrated Gaussian',
        pilot_backgrounds_per_exposure=100,calibration_backgrounds_per_source_exposure=100,evaluation_backgrounds_per_source_exposure=100,
        calibration_levels=list(I.GRID),evaluation_levels=list(I.LEVELS),
        seed_namespaces=dict(pilot_background=1,calibration_background=2,evaluation_background=3,bootstrap=4,calibration_signal=20,evaluation_signal=30,smoke_signal=90,smoke_background=91),
        background_seed_key='[master,cohort_namespace,source_id,toy,increment_id]; source nominal0 stress1; increment low0 high1',
        signal_seed_key='[master,cohort_namespace,source_id,mass_MeV,toy,z,increment_id]',
        background_truth='nominal: pinned v5.8.2 full2016 all-data GP arithmetic mean; stress: archived source sensitivity control that failed broad-source-fit qualification',
        nominal_truth_sha256=I.sha(I.B/'inputs/null_2016.npz'),
        exposures_pairing='low ~ Poisson(0.1*b), independent increment ~ Poisson(0.9*b), full=low+increment; no multiplication of realized counts',
        cohorts='Independent pilot, calibration, evaluation RNG namespaces. Backgrounds shared across masses/levels within each cohort/source/exposure; nominal and stress sources independent.',
        pilot_rule='Nominal source only. Freeze arithmetic mean of100 valid Gaussian returned observed-profile-Hessian errors independently at each mass/exposure before calibration.',
        expected_yields='A=z*s0(m,exposure), same nominal-pilot mean for both truth sources. Exposure signal pairs use Poisson(A_low*p) plus independent Poisson((A_full-A_low)*p).',
        calibration_rule='Freeze all100 calibration fits per source/exposure/mass/level and hashes before any evaluation fits; grid fixed by independent pilot, not calibration/evaluation outcomes.',
        signal_units='Full-selected Gaussian yield. CDF bin integrals plus below/above30-210MeV support; no support/window renormalization.',
        fit_window='Pole +/-2.25sigma_nominal2016; bin-center mask; complement training with>=3 exterior bins on each side.',
        GP='Same archived full2016 mass-specific kernel at both exposures. Recompute log target/noise alpha, conditional mean and count covariance from every spectrum; no kernel reoptimization.',
        preprocessing='Positive bins log(n), alpha=1/n; zero bins target0, alpha1. Arithmetic lognormal mean exp(mu+diag(cov)/2).',
        covariance='Inherited first Cholesky-success loading1e-10..1e-5 times max(maxdiag,1), Poisson-whitened mode cutoff1e-8.',
        likelihood='Poisson means b_GP+L*theta+A*p with Gaussian theta penalty; signed Ahat, positive means; fixed-truth nuisance profile at the same GP state.',
        uncertainty='Returned observed-profile-Hessian yield error recomputed independently; unavailable curvature or Fisher substitution rejected.',
        acceptance=dict(score_lt=3e-5,min_lambda_gt=0,sigma_finite_positive=True,q_true_min=-2e-6),
        retries='At most3 deterministic attempts on identical counts: origin tolerance2e-7; origin2e-9; A/scale=.5 at2e-9. Use available truth-profile start when nesting needs repair. Select valid minimum objective. Never regenerate or select by outcome.',
        offset_response='delta=mean calibration null Ahat; R=mean paired(Ahat_z3-Ahat_0)/(3*s0); k0=sample SD((Ahat_0-delta)/sigma_0). Keep offset, response, and pull width separate.',
        scaling_tests=['delta_full-10*delta_low','mean_pull_full-sqrt(10)*mean_pull_low','mean_sigma_full/(sqrt(10)*mean_sigma_low)'],
        scaling_uncertainty='Resample whole calibration toy IDs, preserving exposure/mass/level pairing, independently for each source; bootstrap seed namespace4.',
        point_estimators='Held-out raw; subtract10*delta_low; subtract delta_full; affine(Ahat-delta_full)/R_full. Calibration variance/covariance propagated for affine pull diagnostic, not a production interval.',
        native_limits='Inherited profiled asymptotic CLs90 at full exposure only; numerical failures reported separately. No observed-data limit changed.',
        toy_limits='Fixed grid lower-tail plus-one p_A=(1+#calibration Ahat<=evaluation Ahat)/101. Reject if p_A<=.1. Report finite-grid Neyman accepted sets, maximum accepted node, holes, right censoring; no interpolation or continuous endpoint claim.',
        toy_cls='Estimator-ordering finite-MC CLs=min(1,p_A/p_0); separate from native profiled CLs. With100 calibration toys, granularity can prevent any rejection for small p_0.',
        calibration_models='Matched source to held-out same source, and nominal calibration transferred to stress evaluation. These are conditional source/model checks.',
        rank_boundary='Rank size<=10/101 is marginal over calibration and new experiment at fixed source/yield; frozen-table rejection rate is random, Beta(10,91) under continuous distributions, and assessed with independent100-toy evaluation plus exact binomial intervals.',
        resources=dict(local_workers=4,numerical_threads_per_worker=1,wall_timeout_seconds=1800,checkpoint_toys_per_mass=10,remote_compute=False),
        exclusions=['historical10% selection/exposure equivalence claim','2021MC as2016 template','full mass scan','global significance','kernel optimization','production corrections or observed upper limits'],
        limitations=['Controlled thinning of a fixed full-data source is not a validated historical10%-to-full transfer.',
            'Conditional on fixed sources and archived kernel prescription; source/selection/systematic uncertainty unpropagated.',
            'Archived stress source is a sensitivity control, not a qualified physical alternative.',
            'Historical2016 support/optimizer qualification exception remains; successful conditional toys do not repair it.',
            '100 calibration/evaluation toys per cell give coarse tails; masses and exposures are correlated.',
            'No unconditional coverage, physical discovery or exclusion claim.'],
        user_extension='2026-09-24: investigate2016 mean-offset exposure transfer and proper implications for signal extraction/upper limits; append to existing2021 report.')

def prepare():
    target=I.B/'protocol.json';spec=protocol()
    if target.exists():assert json.loads(target.read_text())==spec,'Frozen protocol differs'
    else:I.write_json(target,spec)
    expected={}
    for cohort in ('pilot','calibration','evaluation'):
        for source in (('nominal',) if cohort=='pilot' else I.SOURCES):
            low=np.array([I.rng(I.background_key(cohort,source,t,0)).poisson(.1*I.TRUTHS[source]) for t in range(100)])
            inc=np.array([I.rng(I.background_key(cohort,source,t,1)).poisson(.9*I.TRUTHS[source]) for t in range(100)])
            expected[cohort+'_'+source]=np.stack([low,low+inc])
    cohorts=I.B/'inputs/cohorts.npz'
    if cohorts.exists():
        old=np.load(cohorts)
        for k,a in expected.items():assert np.array_equal(old[k],a)
    else:np.savez_compressed(cohorts,**expected,truth_nominal=I.TRUTHS['nominal'],truth_stress=I.TRUTHS['stress'],edges_GeV=I.D['edges'])
    hashes={k:[[I.ahash(row) for row in a[e]] for e in range(2)] for k,a in expected.items()}
    flat=[h for es in hashes.values() for rows in es for h in rows];assert len(set(flat))==1000
    I.write_json(I.B/'provenance/cohort_hashes.json',hashes)
    contexts=[I.Context(m) for m in I.MASSES]
    cats=np.array([c.categories for c in contexts]);masks=np.array([c.mask for c in contexts])
    templates=I.B/'inputs/templates.npz'
    if templates.exists():
        old=np.load(templates);assert np.array_equal(old['categories'],cats) and np.array_equal(old['masks'],masks)
    else:np.savez_compressed(templates,masses_MeV=I.MASSES,categories=cats,masks=masks,edges_GeV=I.D['edges'])
    pd.DataFrame([dict(mass_MeV=c.m,window_fraction=c.p[c.mask].sum(),training_fraction=c.p[~c.mask].sum(),
        outside_support_fraction=c.categories[0]+c.categories[-1],kernel_const=c.const,kernel_ls=c.ls,
        sigma_MeV=c.sigma*1000,fit_bins=c.mask.sum(),template_hash=I.ahash(c.categories),mask_hash=I.ahash(c.mask)) for c in contexts]).to_csv(I.B/'inputs/template_fractions.csv',index=False,float_format='%.17g')
    runtime=I.B/'provenance/runtime.json'
    if not runtime.exists():I.write_json(runtime,dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,created_utc=I.utc()))
    sig=I.signature()
    I.write_json(I.B/'provenance/computation_signature.json',dict(signature=sig,code_hashes={p.name:I.sha(p) for p in (I.B/'scripts/core.py',I.B/'scripts/run_study.py')},protocol_sha256=I.sha(target),cohorts_sha256=I.sha(cohorts),templates_sha256=I.sha(templates)))
    return sig

def smoke(sig):
    path=I.B/'qa/smoke.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['signature']==sig and old['passed']:return old
    begin=time.monotonic();rows=[]
    for m in (42,76,178):
        ctx=I.Context(m)
        for source in I.SOURCES:
            for e,f in enumerate(I.EXPOSURES):
                background=I.rng([I.MASTER,91,I.SOURCES.index(source),e,0]).poisson(f*I.TRUTHS[source])
                pred=ctx.predict(background);empty=np.zeros(len(ctx.categories),dtype=np.int64)
                null=I.make_row(ctx,background,empty,0.,0,source,f,0,'smoke',prediction=pred,native_limit=f==1.)
                assert null['fit_valid'] and null['profile_valid'],null
                if f==1.:assert null['native_cls90_valid'],null
                rows.append(null);A=5*null['sigma_postfit']
                draw=I.rng(I.signal_key('smoke',source,m,0,5,e)).poisson(A*ctx.categories)
                assert np.array_equal(draw,I.rng(I.signal_key('smoke',source,m,0,5,e)).poisson(A*ctx.categories))
                row=I.make_row(ctx,background,draw,A,0,source,f,5,'smoke',native_limit=f==1.)
                assert row['fit_valid'] and row['profile_valid'],row
                if f==1.:assert row['native_cls90_valid'],row
                rows.append(row)
                changed=background.copy();changed[ctx.mask]+=17;again=ctx.predict(changed)
                assert np.array_equal(pred[0],again[0]) and np.array_equal(pred[1],again[1])
                contaminated=ctx.predict(background+draw[1:-1])
                assert not np.array_equal(pred[0],contaminated[0]) and not np.array_equal(pred[1],contaminated[1])
                b=f*I.TRUTHS[source][ctx.mask];p=ctx.p[ctx.mask];n=b+A*p
                free,fixed,valid,_=I.fit_pair(b,np.zeros((len(b),0)),p,n,A,True)
                assert valid and abs(free['A']/A-1)<1e-7 and abs(fixed['nll']-free['nll'])<1e-7
    pd.DataFrame(rows).to_csv(I.B/'qa/smoke_rows.csv',index=False,float_format='%.17g')
    out=dict(passed=True,signature=sig,created_utc=I.utc(),rows=len(rows),seconds=time.monotonic()-begin,
        checks=['independent deterministic smoke RNG','free/truth-profile observed-Hessian gates at both exposures/sources','native profileCLs roots at full exposure','GP training excludes extraction window','signal tails alter both GP mean and covariance','known-background mean-data full-yield normalization closure'])
    I.write_json(path,out);print(json.dumps(dict(stage='smoke',**out)),flush=True)

def run_cohort(cohort,workers,sig,reference_sha=None,calibration_sha=None):
    I.write_json(I.B/'status.json',dict(status='running_'+cohort,updated_utc=I.utc(),signature=sig,reference_sha256=reference_sha,calibration_sha256=calibration_sha))
    sources=('nominal',) if cohort=='pilot' else I.SOURCES
    jobs=[(cohort,s,m,start,min(start+10,100),sig,reference_sha,calibration_sha) for s in sources for m in I.MASSES for start in range(0,100,10)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(I.run_chunk,*job) for job in jobs]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    frames=[pd.read_csv(I.chunk_base(cohort,s,m,start).with_suffix('.csv'),float_precision='round_trip') for _,s,m,start,*_ in jobs]
    df=pd.concat(frames,ignore_index=True).sort_values(['source','exposure','mass_MeV','z','toy'])
    assert len(df)==dict(pilot=2000,calibration=36000,evaluation=16000)[cohort]
    assert not df.duplicated(['source','exposure','mass_MeV','z','toy']).any()
    df.to_csv(I.B/f'results/{cohort}_rows.csv',index=False,float_format='%.17g')
    return df

def freeze_pilot(pilot,sig):
    path=I.B/'pilot_reference.json'
    if path.exists():
        reference=json.loads(path.read_text());assert reference['signature']==sig and reference['pilot_rows_sha256']==I.sha(I.B/'results/pilot_rows.csv')
        return reference
    masses={}
    for m in I.MASSES:
        masses[str(m)]={}
        for f in I.EXPOSURES:
            cell=pilot[(pilot.mass_MeV==m)&(pilot.exposure==f)]
            assert len(cell)==100 and set(cell.toy)==set(range(100)) and cell.fit_valid.all(),'Incomplete pilot; cannot freeze'
            errors=cell.sigma_postfit
            masses[str(m)][str(f)]=dict(s0=float(errors.mean()),sd_sigma=float(errors.std(ddof=1)),se_sigma=float(errors.std(ddof=1)/10))
        assert masses[str(m)]['1.0']['s0']>masses[str(m)]['0.1']['s0']
    reference=dict(frozen_utc=I.utc(),signature=sig,pilot_rows_sha256=I.sha(I.B/'results/pilot_rows.csv'),masses=masses)
    I.write_json(path,reference);print(json.dumps(dict(stage='pilot_frozen',sha256=I.sha(path),masses=masses)),flush=True)
    return reference

def freeze_calibration(calibration,sig):
    path=I.B/'calibration_freeze.json'
    assert calibration.fit_valid.all(),'Calibration failure must be resolved explicitly before rank-table freeze'
    hashes={str(p.relative_to(I.B)):I.sha(p) for p in sorted((I.B/'results/calibration').glob('*'))}
    if path.exists():
        old=json.loads(path.read_text());assert old['signature']==sig and old['calibration_rows_sha256']==I.sha(I.B/'results/calibration_rows.csv') and old['checkpoint_hashes']==hashes
        return old
    obj=dict(frozen_utc=I.utc(),signature=sig,calibration_rows_sha256=I.sha(I.B/'results/calibration_rows.csv'),
        pilot_reference_sha256=I.sha(I.B/'pilot_reference.json'),checkpoint_hashes=hashes,rows=len(calibration),valid=int(calibration.fit_valid.sum()),
        rule='All100 observations per fixed source/exposure/mass/grid point frozen before evaluation. No fit/outcome selection.')
    I.write_json(path,obj);print(json.dumps(dict(stage='calibration_frozen',sha256=I.sha(path))),flush=True);return obj

def asimov():
    rows=[];reference=json.loads((I.B/'pilot_reference.json').read_text())
    for m in I.MASSES:
        ctx=I.Context(m)
        for source in I.SOURCES:
            for f in I.EXPOSURES:
                rows.append(I.make_row(ctx,f*I.TRUTHS[source],np.zeros(len(ctx.categories),dtype=np.int64),0.,-1,source,f,0,'asimov',s0=reference['masses'][str(m)][str(f)]['s0']))
    assert all(r['fit_valid'] for r in rows)
    pd.DataFrame(rows).to_csv(I.B/'results/asimov_rows.csv',index=False,float_format='%.17g')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--stage',choices=['prepare','smoke','pilot','calibration','all'],default='all')
    args=ap.parse_args();assert 1<=args.workers<=4
    with (I.B/'run.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Study run lock already held; no duplicate launch')
        sig=prepare()
        if args.stage=='prepare':return
        smoke(sig)
        if args.stage=='smoke':return
        pilot=run_cohort('pilot',args.workers,sig);freeze_pilot(pilot,sig);asimov()
        if args.stage=='pilot':return
        calibration=run_cohort('calibration',args.workers,sig,I.sha(I.B/'pilot_reference.json'));freeze_calibration(calibration,sig)
        if args.stage=='calibration':return
        evaluation=run_cohort('evaluation',args.workers,sig,I.sha(I.B/'pilot_reference.json'),I.sha(I.B/'calibration_freeze.json'))
        I.write_json(I.B/'status.json',dict(status='numerical_complete',updated_utc=I.utc(),signature=sig,pilot_rows=len(pilot),calibration_rows=len(calibration),evaluation_rows=len(evaluation),
            pilot_fit_valid=int(pilot.fit_valid.sum()),calibration_fit_valid=int(calibration.fit_valid.sum()),evaluation_fit_valid=int(evaluation.fit_valid.sum()),evaluation_profile_valid=int(evaluation.profile_valid.sum()),
            native_cls90_attempted=int((evaluation.exposure==1.).sum()),native_cls90_valid=int(evaluation.native_cls90_valid.sum())))
        print(json.dumps(dict(stage='numerical_complete',rows=len(evaluation))),flush=True)

if __name__=='__main__':main()
