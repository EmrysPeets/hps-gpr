#!/usr/bin/env python3
"""Paired fixed/GP-profile extraction and local-calibration diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/'study_results/background_profile_comparison_20260905'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE))
import run_comparison as c
from scipy.stats import beta,norm
np,pd=c.np,c.pd
MASTER=491305
MASSES=(65,71,78,100,182,231)
LANES=('known_background','gp_uncertainty','retrained_sidebands')


def interval(k,n,level=.95):
    a=(1-level)/2
    return (0. if k==0 else float(beta.ppf(a,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-a,k+1,n-k)))


def covariance_scales(b,C,w):
    a=w/b;information=float(w@a)
    sf=1/np.sqrt(information)
    kap=np.sqrt(1+float(a@C@a)/information)
    sp=1/np.sqrt(float(w@np.linalg.solve(np.diag(b)+C,w)))
    return sf,kap,sp


def evaluate(model,counts,Atrue,shortcut_check=False):
    free=model.fit(counts);null=model.fit(counts,0.)
    q0=2*(null['nll']-free['nll'])
    if q0 < -1e-7:raise RuntimeError('Free/null nesting')
    r=float(np.sign(free['A'])*np.sqrt(max(q0,0.)))
    cls=float('nan');rejected=False;delta_ul=None
    if Atrue>0:
        fixed=model.fit(counts,Atrue);asimov=model.fit(model.b,Atrue)
        denom=free if free['A']>=0 else null
        q=0. if free['A']>Atrue else 2*(fixed['nll']-denom['nll'])
        if q < -1e-7:raise RuntimeError('Fixed/free nesting')
        cls=c.bounded_tildeq_asymptotic_tails(max(q,0.),2*asimov['nll']).cls
        rejected=bool(cls<.1)
        if shortcut_check:
            limit=model.limit(counts)
            if rejected != (limit['A90']<Atrue):raise RuntimeError('UL classification shortcut mismatch')
            delta_ul=float(limit['A90']-Atrue)
    return dict(Ahat=free['A'],sigma_A=free['sigma'],pull=(free['A']-Atrue)/free['sigma'],
        signed_r=r,p0_asymptotic=float(norm.sf(max(0.,r))),cls_at_true=cls,
        true_yield_excluded=rejected,min_lambda=min(free['min_lambda'],null['min_lambda']),
        max_score=model.max_score,shortcut_delta_ul=delta_ul)


def collect(out,refs,ntoys):
    files=sorted((out/'checkpoints').glob('*.csv.gz'))
    frame=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    frame.to_csv(out/'toy_results.csv.gz',index=False,compression='gzip')
    rows=[];cal=[]
    for (lane,mass,strength,method),d in frame.groupby(['ensemble','mass_MeV','strength_sigma','method']):
        n=len(d);exc=int(d.true_yield_excluded.sum());fp=int((d.signed_r>norm.isf(.05)).sum())
        el,eh=interval(exc,n);fl,fh=interval(fp,n)
        rows.append(dict(ensemble=lane,mass_MeV=int(mass),strength_sigma=int(strength),method=method,n=n,
            Atrue=float(d.Atrue.iloc[0]),mean_Ahat=float(d.Ahat.mean()),std_Ahat=float(d.Ahat.std(ddof=1)),
            mean_sigma_A=float(d.sigma_A.mean()),pull_mean=float(d.pull.mean()),pull_std=float(d.pull.std(ddof=1)),
            mean_r=float(d.signed_r.mean()),std_r=float(d.signed_r.std(ddof=1)),
            exclusion_count=exc,exclusion_fraction=exc/n,exclusion_low=el,exclusion_high=eh,
            false_positive_count=fp,false_positive_fraction=fp/n,false_positive_low=fl,false_positive_high=fh,
            kappa_reference=float(d.kappa_reference.iloc[0])))
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'extraction_summary.csv',index=False)
    for (lane,mass),d in frame[(frame.strength_sigma==0)&(frame.method=='fixed')].groupby(['ensemble','mass_MeV']):
        train=d[d.toy_id<100];test=d[d.toy_id>=100]
        if len(test)==0:continue
        mean=float(train.signed_r.mean());width=float(train.signed_r.std(ddof=1))
        modes={'raw':test.signed_r,'variance_scaled':test.signed_r/test.kappa_reference,
               'split_calibrated':(test.signed_r-mean)/width}
        for name,z in modes.items():
            k=int((z>norm.isf(.05)).sum());lo,hi=interval(k,len(z))
            cal.append(dict(ensemble=lane,mass_MeV=int(mass),correction=name,train_n=len(train),test_n=len(test),
                train_mean=mean,train_width=width,test_mean=float(z.mean()),test_width=float(z.std(ddof=1)),
                false_positive_count=k,false_positive_fraction=k/len(z),low=lo,high=hi))
    pd.DataFrame(cal).to_csv(out/'local_calibration_holdout.csv',index=False)
    # Paired response subtracts the B-only average on the same generating lane.
    for method in ('fixed','profiled'):
        for lane in LANES:
            for mass in summary.mass_MeV.unique():
                sel=(summary.method==method)&(summary.ensemble==lane)&(summary.mass_MeV==mass)
                block=summary[sel].set_index('strength_sigma');base=block.loc[0,'mean_Ahat']
                for strength in (2,5):
                    take=sel&(summary.strength_sigma==strength)
                    summary.loc[take,'mean_signal_response']=(block.loc[strength,'mean_Ahat']-base)/block.loc[strength,'Atrue']
    summary.to_csv(out/'extraction_summary.csv',index=False)
    assert len(frame)==ntoys*len(refs)*len(LANES)*3*2
    assert frame.min_lambda.min()>0 and frame.max_score.max()<2e-7
    return frame,summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    ntoys=3 if args.pilot else 500;masses=(65,231) if args.pilot else MASSES
    out=HERE/('pilot' if args.pilot else 'derived');out.mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    started=time.monotonic()
    cfg=c.production.load_config(c.production.DEFAULT_CARD);ds=c.production.make_datasets(cfg)['2021']
    states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
    source=ROOT/'study_results/v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/reverse_injection/derived/common_truth_and_signals.csv'
    truth_table=pd.read_csv(source)
    files=[Path(__file__),HERE/'PROTOCOL.md',BASE/'run_comparison.py',source,
        c.production.DEFAULT_CARD,c.production.DEFAULT_STATES,Path(cfg.path_2021)]
    hashes={str(p):c.sha(p) for p in files}
    if hashes[str(Path(cfg.path_2021))]!=c.production.EXPECTED_HISTOGRAM_SHA256['2021']:
        raise RuntimeError('Native input hash mismatch')
    refs={};fisher=[]
    for mass in range(50,251):
        st=states['2021',mass];m=mass/1000
        p=c.production.estimate_background_for_dataset(ds,m,cfg,restarts=0,
            kernel=c.make_fixed_kernel(st['const_opt'],st['ls_opt']),optimize=False,train_exclude_nsigma=2.25)
        cov,condition=c.production.condition_covariance_block(p.cov,p.mu);L=c._chol_with_jitter(cov);C=L@L.T
        w,full=c.build_window_template_from_full(p.edges_full,p.blind_mask,m,p.sigma_val,config=cfg)
        fraction=float(w.sum());w=w/fraction
        sf,kap,sp=covariance_scales(p.mu,C,w)
        fisher.append(dict(mass_MeV=mass,sigma_fixed=sf,kappa=kap,sigma_fixed_covariance_corrected=sf*kap,
                           sigma_profiled=sp,corrected_fixed_over_profiled=sf*kap/sp))
        if mass in masses:
            assert np.allclose(truth_table.mass_MeV.to_numpy()/1000,p.x_full,rtol=0,atol=1e-15)
            truth=truth_table.smooth_truth_counts.to_numpy(float)
            refs[mass]=dict(pred=p,mean=p.mu,cov=cov,factor=L,template=w,full=full,
                            fraction=fraction,sigma_profiled=sp,kappa=kap,truth=truth)
    pd.DataFrame(fisher).to_csv(out/'fisher_variance_scan.csv',index=False)
    definitions=[dict(mass_MeV=m,sigma_profiled=r['sigma_profiled'],kappa=r['kappa'],
                     window_fraction=r['fraction'],n_window=len(r['mean'])) for m,r in refs.items()]
    (out/'frozen_injection_strengths.json').write_text(json.dumps(definitions,indent=2)+'\n')
    rejections=0;shortcut_checks=0
    for laneid,lane in enumerate(LANES):
        for mass in masses:
            ref=refs[mass];p=ref['pred'];m=mass/1000;st=states['2021',mass]
            for strength in (0,2,5):
                path=out/'checkpoints'/f'{lane}_m{mass:03d}_s{strength}.csv.gz'
                Atrue=strength*ref['sigma_profiled']
                if path.exists():
                    existing=pd.read_csv(path)
                    if len(existing)!=2*ntoys:raise RuntimeError('Incomplete checkpoint')
                    continue
                rng=np.random.default_rng(np.random.SeedSequence([MASTER,laneid,mass,strength]))
                rows=[]
                base_models={'fixed':c.Profile(ref['mean'],np.zeros((len(ref['mean']),0)),ref['template'],'linear'),
                             'profiled':c.Profile(ref['mean'],ref['factor'],ref['template'],'linear')}
                for toy in range(ntoys):
                    if lane=='known_background':
                        counts=rng.poisson(ref['mean']+Atrue*ref['template']).astype(float);models=base_models
                    elif lane=='gp_uncertainty':
                        b=ref['mean']+ref['factor']@rng.standard_normal(len(ref['mean']))
                        while np.any(b<=0):
                            rejections+=1;b=ref['mean']+ref['factor']@rng.standard_normal(len(ref['mean']))
                        counts=rng.poisson(b+Atrue*ref['template']).astype(float);models=base_models
                    else:
                        whole=rng.poisson(ref['truth']+Atrue/ref['fraction']*ref['full']).astype(float)
                        keep=~p.blind_mask
                        gp=c.fit_gpr(p.x_full[keep],whole[keep],cfg,restarts=0,
                            kernel=c.make_fixed_kernel(st['const_opt'],st['ls_opt']),optimize=False)
                        b,C=c.predict_counts_from_log_gpr(gp,p.x_full[p.blind_mask],cfg)
                        C,_=c.production.condition_covariance_block(C,b)
                        models={'fixed':c.Profile(b,np.zeros((len(b),0)),ref['template'],'linear'),
                                'profiled':c.Profile(b,c._chol_with_jitter(C),ref['template'],'linear')}
                        counts=whole[p.blind_mask]
                    for method,model in models.items():
                        check=toy<3 and strength>0
                        result=evaluate(model,counts,Atrue,shortcut_check=check)
                        shortcut_checks+=int(check)
                        rows.append(dict(ensemble=lane,mass_MeV=mass,strength_sigma=strength,toy_id=toy,method=method,
                            Atrue=Atrue,kappa_reference=ref['kappa'],**result))
                pd.DataFrame(rows).to_csv(path,index=False,compression='gzip')
                print(f'Completed {lane}: {mass} MeV, {strength} sigma, {ntoys} paired extractions',flush=True)
    frame,table=collect(out,refs,ntoys)
    for p,h in hashes.items():
        if c.sha(p)!=h:raise RuntimeError('Source changed: '+p)
    summary=dict(status='pilot_passed' if args.pilot else 'completed',toys_per_coordinate=ntoys,
        mass_points=list(masses),generating_ensembles=list(LANES),strengths_sigma=[0,2,5],
        generated_spectra=len(frame)//2,paired_method_fits=len(frame),negative_vector_redraws=rejections,
        shortcut_full_UL_checks_this_run=shortcut_checks,max_scaled_score=float(frame.max_score.max()),
        sources=hashes,source_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started,
        claim_boundary='Pointwise conditional generating-model diagnostics; no unconditional coverage, full hyperparameter retraining, or global significance.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
