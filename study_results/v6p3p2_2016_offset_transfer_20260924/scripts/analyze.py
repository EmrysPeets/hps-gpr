#!/usr/bin/env python3
"""Conditional exposure-transfer diagnostics and finite-toy grid test inversion.

Reads frozen results only. No scientific toys, fits, or evaluation-driven tuning.
Calibration uncertainty and heldout uncertainty are distinct. Sources have
independent bootstrap streams; masses/exposures/levels share whole toy IDs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

BASE = Path(__file__).resolve().parents[1]
MASSES = [42,44,60,66,76,90,92,117,160,178]
EXPOSURES = [.1,1.]
SOURCES = ['nominal','stress']
LEVELS = [0,1,3,5]
N = 100


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(s):
    x=s.astype(str).str.lower()
    if not x.isin(['true','false','1','0','1.0','0.0']).all():
        raise ValueError('Malformed boolean '+s.name)
    return x.isin(['true','1','1.0'])


def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x,np.generic): x=x.item()
    if isinstance(x,float) and not np.isfinite(x): return None
    return x


def stats(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)];n=len(x)
    sd=np.std(x,ddof=1) if n>1 else np.nan
    return dict(n=n,mean=np.mean(x) if n else np.nan,sd=sd,se=sd/np.sqrt(n) if n else np.nan)


def boot(x,indices,kind='mean'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        samples=np.asarray(x,float)[indices]
        return np.nanstd(samples,axis=1,ddof=1) if kind=='sd' else np.nanmean(samples,axis=1)


def interval(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if len(x)<2:return dict(se=np.nan,ci95_lo=np.nan,ci95_hi=np.nan)
    return dict(se=np.std(x,ddof=1),ci95_lo=np.quantile(x,.025),ci95_hi=np.quantile(x,.975))


def cp(k,n):
    if n==0:return np.nan,np.nan
    return (0. if k==0 else beta.ppf(.025,k,n-k+1),1. if k==n else beta.ppf(.975,k+1,n-k))


def binary_summary(label,values,out):
    v=np.asarray(values,float);v=v[np.isfinite(v)];n=len(v);k=int(v.sum());lo,hi=cp(k,n)
    out.update({label+'_k':k,label+'_n':n,label+'_fraction':k/n if n else np.nan,
                label+'_cp95_lo':lo,label+'_cp95_hi':hi,
                label+'_all_attempt_lo':k/N,label+'_all_attempt_hi':(k+N-n)/N})


def add_stats(out,name,x,indices=None,width=False):
    out.update({name+'_'+k:v for k,v in stats(x).items()})
    if width:
        out.update({name+'_sd_bootstrap_'+k:v for k,v in interval(boot(x,indices,'sd')).items()})


def read_rows(path,cohort):
    d=pd.read_csv(path)
    for c in ['fit_valid','profile_valid']:
        if c in d:d[c]=boolean(d[c])
    for c in ['mass_MeV','z','toy']:d[c]=d[c].astype(int)
    d['exposure']=d.exposure.astype(float)
    if cohort=='pilot' and 'source' not in d:d['source']='nominal'
    keys=['source','exposure','mass_MeV','z','toy']
    if d.duplicated(keys).any():raise ValueError('Duplicate '+cohort+' keys')
    lev=[0] if cohort=='pilot' else (list(range(9)) if cohort=='calibration' else LEVELS)
    sources=['nominal'] if cohort=='pilot' else SOURCES
    expected={(s,e,m,z,t) for s in sources for e in EXPOSURES for m in MASSES for z in lev for t in range(N)}
    found=set(d[keys].itertuples(index=False,name=None))
    if found!=expected:raise ValueError(f'{cohort}: missing {len(expected-found)}, unexpected {len(found-expected)}')
    good=d.fit_valid
    if not np.isfinite(d.loc[good,['Ahat','sigma_postfit']]).all().all() or not d.loc[good,'sigma_postfit'].gt(0).all():
        raise ValueError('Invalid free fit marked valid')
    if cohort!='pilot':
        for key,q in d.groupby(['exposure','mass_MeV','z']):
            if q.A_expected.nunique()!=1 or q.s0.nunique()!=1:raise ValueError('Expected yield/reference not common across sources and toys')
            if not np.allclose(q.A_expected,q.z*q.s0,rtol=1e-11,atol=1e-11):raise ValueError('Yield not z*s0')
    return d.sort_values(keys)


def analyze(base=BASE,replicates=2000):
    results=base/'results'
    protocol=json.loads((base/'protocol.json').read_text())
    master=int(protocol['master_seed'])
    pilot=read_rows(results/'pilot_rows.csv','pilot')
    cal=read_rows(results/'calibration_rows.csv','calibration')
    evaluation=read_rows(results/'evaluation_rows.csv','evaluation')
    reference=json.loads((base/'pilot_reference.json').read_text())
    if not (base/'calibration_freeze.json').is_file():raise ValueError('Calibration must be frozen before analysis')
    for m in MASSES:
        for e in EXPOSURES:
            q=pilot[pilot.mass_MeV.eq(m)&pilot.exposure.eq(e)]
            if not q.fit_valid.all():raise ValueError('Incomplete Gaussian pilot')
            s0=reference['masses'][str(m)][str(e)]['s0']
            if not np.isclose(q.sigma_postfit.mean(),s0,rtol=1e-10):raise ValueError('Pilot reference mismatch')
            for frame in (cal,evaluation):
                if not np.allclose(frame.loc[frame.mass_MeV.eq(m)&frame.exposure.eq(e),'s0'],s0,rtol=1e-10):raise ValueError('Frozen yield reference mismatch')
    # Namespace 4; source_id makes calibration-source resampling independent.
    indices={s:np.stack([np.random.default_rng(np.random.SeedSequence([master,4,0,r,si+1,0])).integers(0,N,N) for r in range(replicates)]) for si,s in enumerate(SOURCES)}
    eval_indices={s:np.stack([np.random.default_rng(np.random.SeedSequence([master,4,1,r,si+1,0])).integers(0,N,N) for r in range(replicates)]) for si,s in enumerate(SOURCES)}
    cells={}
    for key,q in cal.groupby(['source','exposure','mass_MeV','z']):
        cells[key]=q.set_index('toy').reindex(range(N))
    pars={};pboot={};parameter_rows=[];quantile_rows=[]
    for source in SOURCES:
        ix=indices[source]
        for e in EXPOSURES:
            for m in MASSES:
                null=cells[source,e,m,0];pos=cells[source,e,m,3]
                x=null.Ahat.where(null.fit_valid).to_numpy(); sig=null.sigma_postfit.where(null.fit_valid).to_numpy()
                rawpull=x/sig
                A3=float(pos.A_expected.iloc[0]);response=(pos.Ahat.where(pos.fit_valid).to_numpy()-x)/A3
                delta=stats(x)['mean'];R=stats(response)['mean'];centered=(x-delta)/sig
                k0=stats(centered)['sd']
                db=boot(x,ix);Rb=boot(response,ix)
                # Re-estimate the null center inside each bootstrap replicate.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    kb=np.nanstd((x[ix]-db[:,None])/sig[ix],axis=1,ddof=1)
                finite=np.isfinite(db)&np.isfinite(Rb)
                cov=np.cov(db[finite],Rb[finite],ddof=1) if finite.sum()>1 else np.full((2,2),np.nan)
                valid=int(null.fit_valid.sum())==N and np.isfinite(response).sum()==N and R>0 and np.isfinite(k0) and k0>0
                p=dict(source=source,exposure=e,mass_MeV=m,s0=float(pos.s0.iloc[0]),n_null=int(null.fit_valid.sum()),
                       n_response=int(np.isfinite(response).sum()),delta=delta,delta_se=stats(x)['se'],R=R,R_se=stats(response)['se'],
                       k0=k0,k0_bootstrap_se=interval(kb)['se'],var_delta=cov[0,0],var_R=cov[1,1],cov_delta_R=cov[0,1],
                       mean_null_sigma=stats(sig)['mean'],raw_null_pull_mean=stats(rawpull)['mean'],raw_null_pull_sd=stats(rawpull)['sd'],
                       calibration_valid=bool(valid))
                for field,arr in [('delta',db),('R',Rb),('k0',kb)]:
                    p.update({field+'_bootstrap_'+k:v for k,v in interval(arr).items()})
                pars[source,e,m]=p;pboot[source,e,m]=dict(delta=db,R=Rb,k0=kb,pull=boot(rawpull,ix),sigma=boot(sig,ix))
                parameter_rows.append(p)
                for z in range(9):
                    c=cells[source,e,m,z];v=np.sort(c.loc[c.fit_valid,'Ahat'].to_numpy())
                    n=len(v);order=int(np.floor(.1*(N+1)));validq=n==N
                    tail_lo,tail_hi=beta.ppf([.025,.975],order,N+1-order)
                    quantile_rows.append(dict(source=source,exposure=e,mass_MeV=m,z=z,A_expected=float(c.A_expected.iloc[0]),
                         n_calibration=n,rank_order90=order,critical90=v[order-1] if validq else np.nan,
                         tail_beta95_lo=tail_lo,tail_beta95_hi=tail_hi,acceptance_beta95_lo=1-tail_hi,acceptance_beta95_hi=1-tail_lo,
                         rank_marginal_acceptance_lower=1-order/(N+1),distinct_statistics=len(np.unique(v)),
                         quantile_reference='continuous order-statistic Beta; inclusive ties give conservative rank tests',valid=validq))
    asimov_path=results/'asimov_rows.csv'
    asimov=pd.read_csv(asimov_path) if asimov_path.is_file() else None
    scaling=[]
    for source in SOURCES:
        for m in MASSES:
            a=pars[source,.1,m];b=pars[source,1.,m];ab=pboot[source,.1,m];bb=pboot[source,1.,m]
            q=dict(source=source,mass_MeV=m,delta10=a['delta'],delta100=b['delta'],scaled_delta10=10*a['delta'],
                   delta_difference=b['delta']-10*a['delta'],pull10=a['raw_null_pull_mean'],pull100=b['raw_null_pull_mean'],
                   scaled_pull10=np.sqrt(10)*a['raw_null_pull_mean'],pull_difference=b['raw_null_pull_mean']-np.sqrt(10)*a['raw_null_pull_mean'],
                   sigma10=a['mean_null_sigma'],sigma100=b['mean_null_sigma'],sigma_scaling_ratio=b['mean_null_sigma']/(np.sqrt(10)*a['mean_null_sigma']),
                   R10=a['R'],R100=b['R'],R_difference=b['R']-a['R'])
            for field,arr in [('delta_difference',bb['delta']-10*ab['delta']),('pull_difference',bb['pull']-np.sqrt(10)*ab['pull']),
                              ('sigma_scaling_ratio',bb['sigma']/(np.sqrt(10)*ab['sigma'])),('R_difference',bb['R']-ab['R'])]:
                q.update({field+'_'+k:v for k,v in interval(arr).items()})
            if asimov is not None:
                aq=asimov[asimov['source'].eq(source)&asimov.mass_MeV.eq(m)].set_index('exposure')
                if set(aq.index)==set(EXPOSURES):
                    q.update(asimov_delta10=float(aq.loc[.1,'Ahat']),asimov_delta100=float(aq.loc[1.,'Ahat']),
                             asimov_delta_difference=float(aq.loc[1.,'Ahat']-10*aq.loc[.1,'Ahat']),
                             asimov_pull10=float(aq.loc[.1,'Ahat']/aq.loc[.1,'sigma_postfit']),
                             asimov_pull100=float(aq.loc[1.,'Ahat']/aq.loc[1.,'sigma_postfit']))
            scaling.append(q)
    eval_cells={key:q.set_index('toy').reindex(range(N)) for key,q in evaluation.groupby(['source','exposure','mass_MeV','z'])}
    diagnostic_rows=[];diagnostic_arrays={};limit_rows=[];rank_grid_rows=[]
    for source in SOURCES:
        ix=eval_indices[source]
        for e in EXPOSURES:
            calibration_sources=[source]+(['nominal'] if source=='stress' and e==1. else [])
            for m in MASSES:
                null=eval_cells[source,e,m,0];null_y=null.Ahat.where(null.fit_valid).to_numpy()
                for cs in calibration_sources:
                    p=pars[cs,e,m];p10=pars[cs,.1,m]
                    grid=np.stack([cells[cs,e,m,z].Ahat.where(cells[cs,e,m,z].fit_valid).to_numpy() for z in range(9)])
                    grid_A=np.array([float(cells[cs,e,m,z].A_expected.iloc[0]) for z in range(9)])
                    rank_valid=np.isfinite(grid).all()
                    for z in LEVELS:
                        q=eval_cells[source,e,m,z];good=q.fit_valid.to_numpy();y=q.Ahat.where(good).to_numpy();sig=q.sigma_postfit.where(good).to_numpy()
                        A=float(q.A_expected.iloc[0]);s0=float(q.s0.iloc[0])
                        for method in ['raw','offset_transfer','direct_offset','affine']:
                            delta=0.;R=1.;valid_cal=True
                            if method=='offset_transfer':delta=(e/.1)*p10['delta'];valid_cal=p10['n_null']==N
                            if method=='direct_offset':delta=p['delta'];valid_cal=p['n_null']==N
                            if method=='affine':delta=p['delta'];R=p['R'];valid_cal=p['calibration_valid']
                            est=(y-delta)/R if valid_cal else np.full(N,np.nan)
                            err=sig/abs(R) if valid_cal else np.full(N,np.nan)
                            bias=est-A;pull=bias/err
                            response=(y-null_y)/(A*R) if A>0 and valid_cal else np.full(N,np.nan)
                            out=dict(source=source,calibration_source=cs,exposure=e,mass_MeV=m,z=z,method=method,
                                     attempted=N,fit_valid=int(np.isfinite(est).sum()),A_expected=A,s0=s0,
                                     rmse=np.sqrt(np.nanmean(bias**2)),k_scaled_pull_mean=np.nan,k_scaled_pull_sd=np.nan)
                            for name,v in [('estimate',est),('bias',bias),('pull',pull),('paired_response',response),('sigma',err)]:
                                add_stats(out,name,v,ix,width=name=='pull')
                            if method=='affine' and valid_cal:
                                add_stats(out,'k_scaled_pull',pull/p['k0'],ix,width=True)
                                variance=(p['k0']**2*sig**2+p['var_delta']+est**2*p['var_R']+2*est*p['cov_delta_R'])/R**2
                                approximate=np.sqrt(np.maximum(variance,0))
                                add_stats(out,'sigma_with_calibration',approximate)
                                add_stats(out,'pull_with_calibration',bias/approximate,ix,width=True)
                            diagnostic_rows.append(out)
                            diagnostic_arrays[source,cs,e,m,z,method]=dict(estimate=est,pull=pull,bias=bias,response=response)
                        for toy in range(N):
                            valid=bool(good[toy] and rank_valid)
                            pgrid=(1+np.sum(grid<=y[toy],axis=1))/(N+1) if valid else np.full(9,np.nan)
                            cls=np.minimum(1,pgrid/pgrid[0]) if valid else np.full(9,np.nan)
                            rank_grid_rows.append(dict(source=source,calibration_source=cs,exposure=e,mass_MeV=m,z=z,toy=toy,
                                valid=valid,A_expected=A,p_grid_json=json.dumps(clean(pgrid)),cls_grid_json=json.dumps(clean(cls))))
                            for method,pvalues in [('rank_neyman90',pgrid),('rank_cls90',cls)]:
                                accepted=(pvalues>.1) if valid else np.zeros(9,bool)
                                loc=np.flatnonzero(accepted)
                                empty=valid and len(loc)==0;right=valid and bool(accepted[-1])
                                hole=valid and len(loc)>0 and not accepted[loc[0]:loc[-1]+1].all()
                                upper=grid_A[loc[-1]] if valid and len(loc) else (0. if valid else np.nan)
                                limit_rows.append(dict(source=source,calibration_source=cs,exposure=e,mass_MeV=m,z=z,toy=toy,method=method,
                                    valid=valid,A_expected=A,s0=s0,p_at_truth=float(pvalues[z]) if valid else np.nan,
                                    accepted_at_truth=float(accepted[z]) if valid else np.nan,
                                    U_grid=upper,upper_contains_truth=float(upper>=A-1e-9) if valid else np.nan,
                                    empty_set=bool(empty),right_censored=bool(right),holey_set=bool(hole),
                                    accepted_z_json=json.dumps(loc.tolist()) if valid else 'null'))
                # Native profile-CLs belongs to the original fit, not to a calibration-source choice.
                if e==1.:
                    for z in LEVELS:
                        q=eval_cells[source,e,m,z]
                        if not {'native_cls90','native_cls90_valid'}<=set(q.columns):raise ValueError('Native full-exposure CLs columns missing')
                        nv=boolean(q.native_cls90_valid.fillna(False))
                        for toy in range(N):
                            v=bool(nv.iloc[toy]);u=float(q.native_cls90.iloc[toy]) if v else np.nan;A=float(q.A_expected.iloc[toy])
                            limit_rows.append(dict(source=source,calibration_source='none',exposure=e,mass_MeV=m,z=z,toy=toy,
                                method='native_profile_cls90',valid=v,A_expected=A,s0=float(q.s0.iloc[toy]),p_at_truth=np.nan,
                                accepted_at_truth=float(u>=A) if v else np.nan,U_grid=u,upper_contains_truth=float(u>=A) if v else np.nan,
                                empty_set=False,right_censored=False,holey_set=False,accepted_z_json='null'))
    diag=pd.DataFrame(diagnostic_rows);limits=pd.DataFrame(limit_rows)
    limit_summaries=[]
    for key,d in limits.groupby(['source','calibration_source','exposure','mass_MeV','z','method']):
        q=dict(zip(['source','calibration_source','exposure','mass_MeV','z','method'],key))
        q.update(attempted=len(d),valid=int(d.valid.sum()),empty_sets=int(d.empty_set.sum()),right_censored=int(d.right_censored.sum()),
                 holey_sets=int(d.holey_set.sum()),median_U_grid=d.loc[d.valid,'U_grid'].median(),
                 median_U_over_s0=(d.loc[d.valid,'U_grid']/d.loc[d.valid,'s0']).median(),
                 finite_uncensored=int((d.valid&~d.right_censored).sum()))
        binary_summary('acceptance',d.accepted_at_truth,q);binary_summary('upper_coverage',d.upper_contains_truth,q)
        limit_summaries.append(q)
    comparisons=[]
    for m in MASSES:
        for z in LEVELS:
            for method in ['offset_transfer','direct_offset','affine']:
                own=diagnostic_arrays['stress','stress',1.,m,z,method]
                transfer=diagnostic_arrays['stress','nominal',1.,m,z,method]
                for metric in ['estimate','bias','pull','response']:
                    delta=transfer[metric]-own[metric]
                    q=dict(source='stress',exposure=1.,mass_MeV=m,z=z,method=method,metric=metric,direction='nominal_calibration_minus_own_calibration')
                    q.update({'difference_'+k:v for k,v in stats(delta).items()})
                    q.update({'bootstrap_'+k:v for k,v in interval(boot(delta,eval_indices['stress'])).items()})
                    comparisons.append(q)
    failed=[]
    for name,frame in [('pilot',pilot),('calibration',cal),('evaluation',evaluation)]:
        mask=~frame.fit_valid
        if name=='evaluation' and 'profile_valid' in frame:mask=mask|~frame.profile_valid
        q=frame[mask].copy();q['failure_stage']=name;failed.append(q)
    native=evaluation[evaluation.exposure.eq(1.)].copy()
    native_valid=boolean(native.native_cls90_valid.fillna(False))
    nf=native.loc[~native_valid].copy();nf['failure_stage']='native_profile_cls90'
    failed.append(nf)
    fail=pd.concat(failed,ignore_index=True)
    products={'calibration_parameters.csv':pd.DataFrame(parameter_rows),'scaling_comparisons.csv':pd.DataFrame(scaling),
              'calibration_quantiles.csv':pd.DataFrame(quantile_rows),'evaluation_diagnostics.csv':diag,
              'rank_grid_rows.csv':pd.DataFrame(rank_grid_rows),'limit_rows.csv':limits,'limit_summary.csv':pd.DataFrame(limit_summaries),
              'nominal_to_stress_comparisons.csv':pd.DataFrame(comparisons),'failure_ledger.csv':fail}
    for name,frame in products.items():frame.to_csv(results/name,index=False,float_format='%.17g')
    source_paths=[results/(c+'_rows.csv') for c in ['pilot','calibration','evaluation']]+[base/'pilot_reference.json',base/'calibration_freeze.json',base/'protocol.json',Path(__file__).resolve()]
    if asimov_path.is_file():source_paths.append(asimov_path)
    summary=dict(schema_version=1,master_seed=master,bootstrap_replicates=replicates,pilot_rows=len(pilot),calibration_rows=len(cal),evaluation_rows=len(evaluation),
        pilot_valid=int(pilot.fit_valid.sum()),calibration_valid=int(cal.fit_valid.sum()),evaluation_valid=int(evaluation.fit_valid.sum()),
        all_toy_ids_accounted=True,calibration_parameter_cells=len(parameter_rows),valid_calibration_parameter_cells=sum(p['calibration_valid'] for p in parameter_rows),
        failure_rows=len(fail),evaluation_profile_valid=int(evaluation.profile_valid.sum()),native_cls90_attempted=len(native),native_cls90_valid=int(native_valid.sum()),source_hashes={str(p.relative_to(base)):sha(p) for p in source_paths},
        definitions={'offset':'delta=mean cal Ahat(0)','response':'R=mean paired cal[Ahat(3*s0)-Ahat(0)]/(3*s0)',
          'width':'k0=sample SD[(cal Ahat(0)-delta)/sigmahat]',
          'rank':'p_A=(1+#100 calibration Ahat<=evaluation Ahat)/101; reject if p_A<=0.10',
          'CLs_diagnostic':'min(1,p_A/p_0), estimator ordering; not native profile CLs',
          'bootstrap':str(replicates)+' whole toy-ID replicates within source, separate source streams, paired exposures/masses/levels; pilot reference fixed',
          'rank_bound':'test size <=10/101 marginal over random calibration and evaluation at fixed source and independent pilot reference',
          'frozen_table':'heldout acceptance and upper-envelope coverage reported separately, with 95% Clopper-Pearson intervals',
          'grid':'z=0..8, no interpolation; empty sets, holes and z8 right-censoring preserved',
          'calibration_beta':'Beta(10,91) continuous order-statistic reference for lower-tail probability of the10th calibration statistic; inclusive rank ties conservative'},
        limitations='Conditional fixed-source validation. Scaling is tested, not assumed. No production bias correction or calibrated continuous/global/physical-exclusion claim.')
    (results/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ['pilot_rows','calibration_rows','evaluation_rows','pilot_valid','calibration_valid','evaluation_valid','failure_rows']}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=BASE);p.add_argument('--bootstrap-replicates',type=int,default=2000)
    a=p.parse_args()
    if a.bootstrap_replicates<100:p.error('At least100 bootstrap replicates required')
    analyze(a.base.resolve(),a.bootstrap_replicates)
