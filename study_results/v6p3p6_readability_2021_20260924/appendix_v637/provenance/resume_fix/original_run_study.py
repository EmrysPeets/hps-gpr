#!/usr/bin/env python3
"""Paired, independent-cohort v16 TC extraction and finite-grid rank study.

No observed counts choose the window. GP training and likelihood bins are
disjoint. Signal probabilities and injections use all selected events, including
the part outside both the fitted interval and the recorded analysis support.
"""
from pathlib import Path
import os, sys, json, hashlib, time, datetime, argparse
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B/'inputs/v6p1/scripts'))
import common as C
import templates as T
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.special import ndtr
from scipy.stats import beta

MASSES = (100,160,220)
SEED = 637250925
N = 100
GRID = (0.,.5,1.,2.,3.,4.,5.,6.,8.,10.,12.,16.)
D = C.DATA['2021']
TRUTHS = {'gp_mean':np.load(B/'inputs/null_2021.npz')['truth'], 'functional':D['stress']}
REF = json.loads((B/'inputs/window_pilot_reference.json').read_text())['masses']

# Every candidate is specified before any tuning outcomes are read.
CANDIDATES = {
 'gaussian_baseline':dict(shape='gaussian',fit=(-2.25,2.25),guard=(-2.25,2.25),reference=True),
 'gaussian_starter':dict(shape='gaussian',fit=(-4,3),guard=(-4,3)),
 'common_starter':dict(shape='common',fit=(-4,3),guard=(-4,3)),
 'morph_short':dict(shape='morph',fit=(-2.5,2.25),guard=(-2.5,2.25)),
 'morph_middle':dict(shape='morph',fit=(-3,2.5),guard=(-3,2.5)),
 'morph_starter':dict(shape='morph',fit=(-4,3),guard=(-4,3)),
 'morph_wide':dict(shape='morph',fit=(-5,4),guard=(-5,4)),
 'morph_extended':dict(shape='morph',fit=(-6,5),guard=(-6,5)),
 'morph_buffer':dict(shape='morph',fit=(-3,2.5),guard=(-4,3)),
 'morph_widebuffer':dict(shape='morph',fit=(-4,3),guard=(-6,5)),
 'direct_starter':dict(shape='direct',fit=(-4,3),guard=(-4,3)),
}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ahash(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def write(path, obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def atomic_csv(path, frame):
    tmp=Path(path).with_suffix('.tmp'); frame.to_csv(tmp,index=False,float_format='%.17g'); tmp.replace(path)
def cp(k,n):
    return (0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k)))

class Context:
    def __init__(self,mass,policy):
        self.mass,self.policy = mass,policy
        spec=CANDIDATES[policy]
        c,s=T.center_width(mass,method='morph',omit=mass)
        if spec.get('reference'):
            c=mass-3.2243308692909953-2.213992811446465*np.log(mass/150.)
            s=C.sigma('2021',mass)*1000
        self.center,self.width=c,s
        self.fit=(D['x']*1000>=c+spec['fit'][0]*s)&(D['x']*1000<=c+spec['fit'][1]*s)
        self.guard=(D['x']*1000>=c+spec['guard'][0]*s)&(D['x']*1000<=c+spec['guard'][1]*s)
        self.guard |= self.fit
        assert np.all(self.guard[self.fit]) and self.fit.sum()>3
        assert (D['x'][~self.guard]<D['x'][self.guard].min()).sum()>=3
        assert (D['x'][~self.guard]>D['x'][self.guard].max()).sum()>=3
        if spec['shape']=='gaussian':
            gc=mass-3.2243308692909953-2.213992811446465*np.log(mass/150.)
            self.probability=np.diff(ndtr((D['edges']*1000-gc)/(C.sigma('2021',mass)*1000)))
        else:
            self.probability=T.probabilities(mass,D['edges']*1000,method=spec['shape'],omit=mass if spec['shape']!='direct' else None)
        self.categories=T.categories(mass,D['edges']*1000,method='direct')
        assert np.min(self.categories)>=0 and abs(self.categories.sum()-1)<1e-12
        const,ls=C.kernel_state('2021',mass)
        self.K=C.kernel(D['x'][~self.guard],D['x'][~self.guard],const,ls)
        self.Kqt=C.kernel(D['x'][self.fit],D['x'][~self.guard],const,ls)
        self.Kqq=C.kernel(D['x'][self.fit],D['x'][self.fit],const,ls)

    def predict(self,counts):
        n=np.asarray(counts,float)[~self.guard]; pos=n>0
        target=np.zeros_like(n);target[pos]=np.log(n[pos])
        alpha=np.ones_like(n);alpha[pos]=1/n[pos]
        K=self.K.copy();K.flat[::len(K)+1]+=alpha
        L=cholesky(K,lower=True,check_finite=False)
        mu=self.Kqt@cho_solve((L,True),target,check_finite=False)
        v=solve_triangular(L,self.Kqt.T,lower=True,check_finite=False)
        cov=self.Kqq-v.T@v;cov=.5*(cov+cov.T)
        b=np.exp(mu+.5*np.maximum(np.diag(cov),0))
        cov=np.outer(b,b)*np.expm1(np.clip(cov,-40,40))
        factor,diag=C.factor_cov(cov,b)
        return b,factor,diag

    def fit_counts(self,counts,limit=False):
        b,L,diag=self.predict(counts)
        errors=[]
        for tolerance in (2e-7,2e-9):
            try:
                model=C.OneSignalProfile(b,L,self.probability[self.fit],score_tolerance=tolerance)
                n=np.asarray(counts)[self.fit]; r=model.fit(n)
                assert r['score']<3e-5 and r['min_lambda']>0
                _,_,H,_=model._objective(r['z'],n,model.Jfree,model.b,model.penfree)
                unit=np.zeros(len(H));unit[0]=1
                variance=float(cho_solve((cholesky(H,lower=True),True),unit)[0])
                sigma=model.scale*np.sqrt(variance)
                assert variance>0 and abs(r['sigma']/sigma-1)<1e-9
                result=dict(Ahat=r['A'],sigma=sigma,fit_valid=True,score=r['score'],min_lambda=r['min_lambda'],covariance_load=diag['load'],nuisance_rank=diag['rank'],failure_reason='')
                if limit:
                    lim=model.limit(n,alpha=.1)
                    assert lim['ok'] and abs(lim['cls']-.1)<2e-6 and lim['max_score']<3e-5
                    result.update(profile_CLs90=lim['A90'],profile_cls=lim['cls'],profile_max_score=lim['max_score'])
                return result
            except Exception as exc: errors.append(type(exc).__name__+': '+str(exc))
        return dict(Ahat=None,sigma=None,fit_valid=False,failure_reason='; '.join(errors))

    def geometry(self):
        fitidx=np.flatnonzero(self.fit);guardidx=np.flatnonzero(self.guard)
        p=self.categories[1:-1]
        return dict(mass_MeV=self.mass,policy=self.policy,shape=CANDIDATES[self.policy]['shape'],center_MeV=self.center,coordinate_width_MeV=self.width,
            fit_low_MeV=D['edges'][fitidx[0]]*1000,fit_high_MeV=D['edges'][fitidx[-1]+1]*1000,
            guard_low_MeV=D['edges'][guardidx[0]]*1000,guard_high_MeV=D['edges'][guardidx[-1]+1]*1000,
            fit_bins=int(self.fit.sum()),guard_bins=int(self.guard.sum()),training_bins=int((~self.guard).sum()),
            MC_fit_fraction=float(p[self.fit].sum()),MC_training_fraction=float(p[~self.guard].sum()),
            MC_unused_guard_fraction=float(p[self.guard&~self.fit].sum()),MC_outside_support_fraction=float(self.categories[0]+self.categories[-1]),
            template_fit_fraction=float(self.probability[self.fit].sum()),fit_mask_sha256=ahash(self.fit),guard_mask_sha256=ahash(self.guard))

CONTEXTS={}
def context(m,p):
    key=(m,p)
    if key not in CONTEXTS: CONTEXTS[key]=Context(m,p)
    return CONTEXTS[key]

def protocol():
    obj=dict(version='6.3.7',seed=SEED,masses_MeV=list(MASSES),candidates=CANDIDATES,toys_per_cohort=N,
        strengths_grid=list(GRID),evaluation_strengths=[0,3,5],sources=list(TRUTHS),
        tuning_source='gp_mean',selection='One global deployable candidate minimizing equal-mass geometric mean SD(background-only Ahat)/R; R=mean(Ahat3-Ahat0)/(3*s0); direct MC is a benchmark, not a deployable interpolated candidate; require all fits valid and positive R.',
        independent_cohorts='tuning namespace1, calibration2, evaluation3; source0GPmean/source1functional; backgrounds paired across masses, policies, strengths; signal Poisson increments paired across policies and nested strengths',
        seed_background='SeedSequence([seed,cohort_namespace,source_index,0,toy])',
        seed_signal='SeedSequence([seed,cohort_namespace,source_index,mass,toy]); independent increments sorted by expected signal strength',
        normalization='full selected TC MC with below/above analysis-support categories; no probability renormalization after window restriction',
        training='GP train bins exclude union of blind region and likelihood support; full MC injection tails remain in GP training',
        omission='Tested mass omitted from core, width, shared-shape and neighboring-shape construction; direct MC uses the test sample as a best-case extraction benchmark',
        background_source='Poisson fluctuation of fixed GP arithmetic mean; functional source separate anchored form stress; no random GP-function draws',
        statistical_scope='Finite-grid source-matched rank-test inversion; conditional local sensitivity, not observed limits, global significance or physical exclusions',
        rank_limit='p_A=(1+# calibration Ahat<=evaluation Ahat)/101; retain all A with p_A>0.1, report complete accepted set and empty/holes/censor flags',
        local_test='p0=(1+# calibration background-only Ahat>=evaluation Ahat)/101; reject p0<=0.1',
        bootstrap='2000 whole-toy resamples; common resampling indices across policies and masses, separate source/cohort streams',
        reference_sha256=sha(B/'inputs/window_pilot_reference.json'),templates_sha256=sha(B/'scripts/templates.py'),
        null_sha256=sha(B/'inputs/null_2021.npz'),script_sha256=sha(__file__))
    path=B/'provenance/toy_protocol.json'
    if path.exists(): assert json.loads(path.read_text())==obj,'Frozen protocol changed'
    else:write(path,obj)
    return path

def run_cohort(name,sources,policies,strengths):
    namespace={'tuning':1,'calibration':2,'evaluation':3}[name]
    pp=B/'provenance/toy_protocol.json'; pieces=[]
    for source in sources:
        sourceid=list(TRUTHS).index(source);truth=TRUTHS[source]
        backgrounds=np.array([np.random.default_rng(np.random.SeedSequence([SEED,namespace,sourceid,0,i])).poisson(truth) for i in range(N)])
        for mass in MASSES:
            ctxs={p:context(mass,p) for p in policies};s0=float(REF[str(mass)]['s0'])
            cats=next(iter(ctxs.values())).categories
            for first in range(0,N,10):
                path=B/f'results/checkpoints/{name}_{source}_m{mass:03d}_t{first:03d}.csv'
                marker=path.with_suffix('.json')
                if marker.exists():
                    meta=json.loads(marker.read_text());assert meta['protocol_sha256']==sha(pp) and meta['rows_sha256']==sha(path)
                    frame=pd.read_csv(path,float_precision='round_trip');pieces.append(frame);continue
                rows=[];draws=[]
                for toy in range(first,first+10):
                    background=backgrounds[toy]
                    rng=np.random.default_rng(np.random.SeedSequence([SEED,namespace,sourceid,mass,toy]))
                    draw=np.zeros(len(cats),dtype=np.int64); previous=0.
                    for z in sorted(strengths):
                        draw=draw+rng.poisson((z-previous)*s0*cats);previous=z
                        counts=background+draw[1:-1]
                        draws.append(draw.copy())
                        for policy,ctx in ctxs.items():
                            row=dict(cohort=name,source=source,mass_MeV=mass,toy=toy,policy=policy,z=z,A_expected=z*s0,s0=s0,
                                background_hash=ahash(background),signal_hash=ahash(draw),counts_hash=ahash(counts),actual_full=int(draw.sum()),
                                actual_window=int(draw[1:-1][ctx.fit].sum()),actual_training=int(draw[1:-1][~ctx.guard].sum()),actual_outside_support=int(draw[0]+draw[-1]))
                            row.update(ctx.fit_counts(counts));rows.append(row)
                frame=pd.DataFrame(rows);atomic_csv(path,frame)
                np.savez_compressed(path.with_suffix('.npz'),signal_categories=np.array(draws),backgrounds=backgrounds[first:first+10],toys=np.arange(first,first+10),strengths=np.array(sorted(strengths)))
                write(marker,dict(protocol_sha256=sha(pp),rows_sha256=sha(path),draws_sha256=sha(path.with_suffix('.npz')),valid=int(frame.fit_valid.sum()),rows=len(frame)))
                assert frame.fit_valid.all(),f'Failed numerical fits retained at {path}'
                pieces.append(frame)
            print(name,source,mass,'complete',flush=True)
    frame=pd.concat(pieces,ignore_index=True);atomic_csv(B/f'results/{name}_rows.csv',frame)
    return frame

def response_summary(frame,name):
    rows=[]
    for (source,mass),group in frame.groupby(['source','mass_MeV']):
        policies=sorted(group.policy.unique()); bg=group[group.z==0].pivot(index='toy',columns='policy',values='Ahat')[policies].to_numpy()
        inj=group[group.z==3].pivot(index='toy',columns='policy',values='Ahat')[policies].to_numpy()
        sig=group[group.z==0].pivot(index='toy',columns='policy',values='sigma')[policies].to_numpy()
        A=float(group[group.z==3].A_expected.iloc[0]);n=len(bg)
        ns={'tuning':1,'evaluation':3}[name];sourceid=list(TRUTHS).index(source)
        indices=np.random.default_rng(np.random.SeedSequence([SEED,7,ns,sourceid])).integers(0,n,(2000,n))
        R=(inj-bg).mean(axis=0)/A; SD=bg.std(axis=0,ddof=1)
        bootR=(inj-bg)[indices].mean(axis=1)/A;bootSD=bg[indices].std(axis=1,ddof=1)
        precision=SD/R;bootPrecision=bootSD/bootR
        baseline=policies.index('gaussian_baseline')
        for i,policy in enumerate(policies):
            low,high=np.quantile(bootR[:,i],[.025,.975]);ratio=precision[i]/precision[baseline]
            rlow,rhigh=np.quantile(bootPrecision[:,i]/bootPrecision[:,baseline],[.025,.975])
            rows.append(dict(cohort=name,source=source,mass_MeV=mass,policy=policy,toys=n,A_expected=A,
                background_only_yield_bias=float(bg[:,i].mean()),background_only_mean_SE=float(SD[i]/np.sqrt(n)),
                background_only_SD=float(SD[i]),background_only_SD_bootstrap_SE=float(bootSD[:,i].std(ddof=1)),
                mean_returned_sigma=float(sig[:,i].mean()),background_only_mean_pull=float((bg[:,i]/sig[:,i]).mean()),
                response=float(R[i]),response95_low=float(low),response95_high=float(high),
                sensitivity_SD_over_response=float(precision[i]),sensitivity_ratio=float(ratio),sensitivity_ratio95_low=float(rlow),sensitivity_ratio95_high=float(rhigh)))
    result=pd.DataFrame(rows);atomic_csv(B/f'results/{name}_response_summary.csv',result);return result

def select(summary):
    rows=[]
    for policy,q in summary.groupby('policy'):
        eligible=CANDIDATES[policy]['shape']!='direct' and np.all(q.response>0)
        objective=float(np.exp(np.mean(np.log(q.sensitivity_SD_over_response))))
        rows.append(dict(policy=policy,eligible=bool(eligible),equal_mass_geometric_SD_over_R=objective))
    rows=sorted(rows,key=lambda r:r['equal_mass_geometric_SD_over_R']);winner=next(r['policy'] for r in rows if r['eligible'])
    # Retain same-geometry comparisons irrespective of tuning outcome.
    policies=list(dict.fromkeys(['gaussian_baseline',winner,'gaussian_starter','common_starter','morph_starter','direct_starter']))
    result=dict(selected_policy=winner,selection_rows=rows,evaluation_policies=policies,
        tuning_rows_sha256=sha(B/'results/tuning_rows.csv'),selected_before_calibration_and_evaluation=True)
    path=B/'results/frozen_selection.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:write(path,result)
    print('FROZEN WINNER',winner,flush=True);return result

def inference(cal,evaluation):
    rows=[]
    for (source,mass,policy),q in evaluation.groupby(['source','mass_MeV','policy']):
        c=cal[(cal.source==source)&(cal.mass_MeV==mass)&(cal.policy==policy)]
        byz={float(z):v.Ahat.to_numpy() for z,v in c.groupby('z')}
        assert all(len(v)==N for v in byz.values())
        for r in q.itertuples(index=False):
            pvalues=np.array([(1+np.count_nonzero(byz[z]<=r.Ahat))/(N+1) for z in GRID])
            accepted=np.flatnonzero(pvalues>.1)
            empty=len(accepted)==0;holes=False if empty else len(accepted)!=(accepted[-1]-accepted[0]+1)
            p0=(1+np.count_nonzero(byz[0]>=r.Ahat))/(N+1)
            endpoint=None if empty else GRID[accepted[-1]]*r.s0
            rows.append(dict(source=source,mass_MeV=mass,policy=policy,toy=r.toy,z=r.z,A_expected=r.A_expected,
                Ahat=r.Ahat,p_background_only=p0,reject_background_only=bool(p0<=.1),
                accepted_z_json=json.dumps([GRID[i] for i in accepted]),p_A_json=json.dumps(pvalues.tolist()),
                empty=empty,holes=holes,upper_grid_censored=bool(not empty and accepted[-1]==len(GRID)-1),
                largest_accepted_A=endpoint,true_grid_value_accepted=bool(pvalues[list(GRID).index(float(r.z))]>.1)))
    df=pd.DataFrame(rows);atomic_csv(B/'results/evaluation_inference.csv',df)
    summaries=[]
    for (source,mass,policy,z),q in df.groupby(['source','mass_MeV','policy','z']):
        n=len(q);k=int(q.reject_background_only.sum());low,high=cp(k,n)
        excluded=int((~q.true_grid_value_accepted).sum());elo,ehi=cp(excluded,n)
        valid=q[~q['empty']&~q.holes&~q.upper_grid_censored]
        summaries.append(dict(source=source,mass_MeV=mass,policy=policy,z=z,toys=n,reject_background_only_count=k,
            reject_background_only_fraction=k/n,reject95_low=low,reject95_high=high,
            true_grid_rejected_count=excluded,true_grid_rejected_fraction=excluded/n,true_grid_rejected95_low=elo,true_grid_rejected95_high=ehi,
            empty_count=int(q['empty'].sum()),holes_count=int(q.holes.sum()),upper_grid_censored_count=int(q.upper_grid_censored.sum()),
            finite_contiguous_count=len(valid),median_largest_accepted_A=float(q.largest_accepted_A.median()) if q.largest_accepted_A.notna().any() else None,
            largest_accepted_A_q16=float(q.largest_accepted_A.quantile(.16)) if q.largest_accepted_A.notna().any() else None,
            largest_accepted_A_q84=float(q.largest_accepted_A.quantile(.84)) if q.largest_accepted_A.notna().any() else None))
    atomic_csv(B/'results/inference_summary.csv',pd.DataFrame(summaries));return df

def main():
    start=time.monotonic()
    for folder in ('results/checkpoints','provenance','qa'): (B/folder).mkdir(parents=True,exist_ok=True)
    pp=protocol()
    geom=[context(m,p).geometry() for m in MASSES for p in CANDIDATES]
    atomic_csv(B/'results/window_geometry.csv',pd.DataFrame(geom))
    np.savez_compressed(B/'results/window_masks.npz',**{f'{p}_m{m}_{name}':getattr(context(m,p),name) for m in MASSES for p in CANDIDATES for name in ('fit','guard','probability','categories')})
    tuning=run_cohort('tuning',['gp_mean'],list(CANDIDATES),(0,3))
    summary=response_summary(tuning,'tuning');selection=select(summary)
    cal=run_cohort('calibration',list(TRUTHS),selection['evaluation_policies'],GRID)
    evaluation=run_cohort('evaluation',list(TRUTHS),selection['evaluation_policies'],(0,3,5))
    response_summary(evaluation,'evaluation');inferred=inference(cal,evaluation)
    asimov=[]
    for source,truth in TRUTHS.items():
        for m in MASSES:
            for p in selection['evaluation_policies']:
                r=context(m,p).fit_counts(truth,limit=True)
                asimov.append(dict(source=source,mass_MeV=m,policy=p,**r))
    atomic_csv(B/'results/asimov_profile_limits.csv',pd.DataFrame(asimov))
    allrows=pd.concat([tuning,cal,evaluation],ignore_index=True)
    valid=bool(allrows.fit_valid.all())
    paired_bg=bool(allrows.groupby(['cohort','source','toy']).background_hash.nunique().max()==1)
    paired_signal=bool(allrows.groupby(['cohort','source','mass_MeV','toy','z']).signal_hash.nunique().max()==1)
    disjoint=all(not np.any(context(m,p).fit&~context(m,p).guard) for m in MASSES for p in CANDIDATES)
    audit=dict(passed=valid and paired_bg and paired_signal and disjoint,all_fits_valid=valid,tuning_rows=len(tuning),calibration_rows=len(cal),evaluation_rows=len(evaluation),
        backgrounds_paired=paired_bg,signals_paired=paired_signal,fit_and_training_disjoint=disjoint,
        selected_policy=selection['selected_policy'],evaluation_policies=selection['evaluation_policies'],sources=list(TRUTHS),
        all_asimov_limits_valid=bool(all(r['fit_valid'] for r in asimov)),bootstrap_resamples=2000,
        seed=SEED,runtime_seconds=time.monotonic()-start,protocol_sha256=sha(pp),
        inference_empty=int(inferred['empty'].sum()),
        inference_holes=int(inferred.holes.sum()),inference_upper_censored=int(inferred.upper_grid_censored.sum()),
        caution='Conditional finite-grid local study. No observed-data fit, global discovery significance, physical exclusion, or continuous confidence-interval coverage claim.')
    write(B/'qa/toy_validation.json',audit);assert audit['passed'] and audit['all_asimov_limits_valid'];print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
