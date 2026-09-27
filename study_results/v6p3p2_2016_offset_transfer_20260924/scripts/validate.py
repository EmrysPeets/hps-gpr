#!/usr/bin/env python3
"""Independent 2016 source, replay, freeze, likelihood and calibration audit.

No production fits are launched. Scientific bias or undercoverage is an outcome,
never an acceptance criterion. Core generation/statistics code is not imported.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import traceback
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import beta
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize

B=Path(__file__).resolve().parents[1]
MASSES=(42,44,60,66,76,90,92,117,160,178)
TRUTHS=('nominal','stress')
EXPOSURES=(.1,1.)
MASTER=63420160924
LEVELS=(0,1,3,5)
GRID=tuple(range(9))
NULL_SHA='995bb3e22eac968c280e6ed2adb268c08af188fa7c07c3dff78fd9c7f85a6378'
SPECTRUM_SHA='11bc1b24af6a0aaa2345fe87e83fce60ee8fb45c77d6b1d4e80adc37fa7af176'
STATES_SHA='fa87f1cc310dc40c04b9731e89e86bcc175444cb1510582f94f35e4cf21df5d7'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def readcsv(path):return pd.read_csv(path,keep_default_na=False,float_precision='round_trip')
def require(condition,message):
    if not bool(condition):raise AssertionError(message)
def close(actual,expected,name,rtol=2e-12,atol=2e-12):
    require(np.allclose(np.asarray(actual,float),np.asarray(expected,float),rtol=rtol,atol=atol,equal_nan=False),f'{name}: mismatch')
def boolcol(frame,key):
    values=frame[key].astype(str).str.lower()
    require(values.isin(['true','false']).all(),f'boolean {key}')
    return values.eq('true')

class Audit:
    def __init__(self):self.checks=[]
    def run(self,name,fn):
        try:self.checks.append(dict(name=name,passed=True,details=fn()))
        except Exception as exc:self.checks.append(dict(name=name,passed=False,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(limit=3)))
        print(name+(': PASS' if self.checks[-1]['passed'] else ': FAIL'),flush=True)

def check_inputs():
    hashes=readj(B/'provenance/input_hashes.json')
    require(len({r['path'] for r in hashes})==len(hashes),'unique input hash paths')
    for r in hashes:require(sha(B/r['path'])==r['sha256'],'immutable input '+r['path'])
    require(sha(B/'inputs/null_2016.npz')==NULL_SHA,'known nominal2016 source')
    require(sha(B/'inputs/v6p1/inputs/spectrum_2016.npz')==SPECTRUM_SHA,'known full2016 spectrum')
    require(sha(B/'provenance/reviewed_gp_states.csv')==STATES_SHA,'known full2016 reviewed states')
    null=dict(np.load(B/'inputs/null_2016.npz'))
    data=dict(np.load(B/'inputs/v6p1/inputs/spectrum_2016.npz'))
    require(np.array_equal(null['edges_GeV'],data['edges']),'null/spectrum edges identical')
    require(np.array_equal(null['observed'],data['n']),'observed release identity')
    require(len(data['n'])==720 and data['edges'][0]==.03 and data['edges'][-1]==.21,'720bins30–210MeV')
    require(np.array_equal(data['masses'],np.arange(39,181)),'integer archived mass grid')
    require(np.all(null['truth']>0) and np.all(data['stress']>0),'positive complete truth spectra')
    close(data['sigma_coeffs'],[.00038,.041,-.27,3.49,-11.11],'2016 resolution coefficients',atol=0.)
    ledger=readcsv(B/'provenance/reviewed_gp_states.csv')
    ledger=ledger[ledger.dataset==2016]
    close(ledger.mass_GeV.to_numpy()*1000,data['masses'],'kernel mass ledger')
    close(ledger.const_opt.to_numpy(),data['const'],'kernel amplitude ledger')
    close(ledger.ls_opt.to_numpy(),data['ls'],'kernel lengthscale ledger')
    history=readj(B/'provenance/historical10_manifest.json')
    for key in ('exact_luminosity_fraction_verified','selection_same_as_parent_verified','event_overlap_verified'):
        require(history[key] is False,'historical10 qualification retained '+key)
    previous=readj(B/'provenance/parent_release_hashes.json')
    for name,digest in previous.items():require(sha(B/name)==digest,'preserved previous report/release '+name)
    original=B.parent/'v5p8p2_nominal_gp_significance_20260917/inputs/null_2016.npz'
    source_checked=original.exists()
    if source_checked:require(sha(original)==NULL_SHA,'original2016 null preserved')
    return dict(pinned_files=len(hashes),source_original_checked=source_checked,
        bins=720,support_GeV=[.03,.21],nominal_full_total=float(null['truth'].sum()),
        stress_full_total=float(data['stress'].sum()),historical10_selection_fraction_overlap_unverified=True,
        reviewed_kernel_state=str(ledger.source_state.iloc[0]),
        scope='Fixed archived full2016 states; a conditional exposure scaling study, not verified historical10 transfer.')

def independent_templates():
    data=dict(np.load(B/'inputs/v6p1/inputs/spectrum_2016.npz'))
    result={}
    for m in MASSES:
        sigma=float(np.polynomial.polynomial.polyval(m/1000.,data['sigma_coeffs']))
        cdf=ndtr((data['edges']-m/1000.)/sigma)
        categories=np.r_[cdf[0],np.diff(cdf),1-cdf[-1]]
        mask=(data['x']>=m/1000.-2.25*sigma)&(data['x']<=m/1000.+2.25*sigma)
        require(np.all(categories>=0),'nonnegative Gaussian probabilities')
        close(categories.sum(),1.,'full Gaussian sum',atol=2e-15)
        require(np.sum(data['x']<m/1000.-2.25*sigma)>=3 and np.sum(data['x']>m/1000.+2.25*sigma)>=3,'two exterior sidebands')
        result[m]=dict(categories=categories,mask=mask,sigma=sigma)
    return data,result

def check_rank_algebra():
    # Fixed continuous calibration table: exactly10 of101 exchangeable ranks reject.
    cal=np.arange(100,dtype=float)
    positions=np.r_[-.5,np.arange(99)+.5,99.5]
    p=(1+(cal[None,:]<=positions[:,None]).sum(axis=1))/101.
    require(int((p<=.1).sum())==10,'90% lower-tail Monte Carlo rank size')
    p_tie=(1+(cal<=cal[9]).sum())/101.
    require(p_tie>.1,'inclusive equality makes ties conservative')
    lo,hi=beta.ppf([.025,.975],10,91)
    return dict(calibration_n=100,nominal_rejection_probability=10/101,
                frozen_table_rejection_probability_beta=[10,91],
                frozen_table_rejection_probability_ci95=[float(lo),float(hi)],
                frozen_table_coverage_ci95=[float(1-hi),float(1-lo)],
                distinction='Rank guarantee is marginal over calibration and evaluation, not a fixed-table guarantee.')

def independent_gp(counts,m):
    data,templates=independent_templates();mask=templates[m]['mask'];x=data['x']
    index=int(np.flatnonzero(data['masses']==m)[0]);constant,length=data['const'][index],data['ls'][index]
    kernel=lambda a,b:constant*np.exp(-.5*((np.log(a)[:,None]-np.log(b)[None,:])/length)**2)
    ntrain=np.asarray(counts[~mask],float)
    target=np.where(ntrain>0,np.log(np.maximum(ntrain,1.)),0.)
    alpha=np.where(ntrain>0,1./np.maximum(ntrain,1.),1.)
    train=kernel(x[~mask],x[~mask])+np.diag(alpha);cross=kernel(x[mask],x[~mask])
    fac=cholesky(train,lower=True);mu=cross@cho_solve((fac,True),target)
    projected=solve_triangular(fac,cross.T,lower=True)
    cov=kernel(x[mask],x[mask])-projected.T@projected;cov=.5*(cov+cov.T)
    b=np.exp(mu+.5*np.maximum(np.diag(cov),0.))
    covariance=np.outer(b,b)*np.expm1(np.clip(cov,-40,40));covariance=.5*(covariance+covariance.T)
    scale=max(float(np.diag(covariance).max()),1.)
    for load in (1e-10,1e-9,1e-8,1e-7,1e-6,1e-5):
        try:
            loaded=covariance+load*scale*np.eye(len(b));cholesky(loaded,lower=True);break
        except np.linalg.LinAlgError:pass
    else:raise AssertionError('independent GP covariance load')
    sd=np.sqrt(b);eig,U=np.linalg.eigh(loaded/sd[:,None]/sd[None,:]);keep=eig>1e-8
    L=sd[:,None]*U[:,keep]*np.sqrt(eig[keep])
    return b,L,mask

def independent_fit(n,b,L,p,expected):
    n=np.asarray(n,float);scale=1./np.sqrt(np.sum(p*p/b));J=np.column_stack([scale*p,L]);pen=np.r_[0.,np.ones(L.shape[1])]
    def objective(z,jac,base,penalty):
        lam=base+jac@z
        if np.any(lam<=0):return np.inf,np.zeros_like(z)
        delta=(lam-n)/n
        return float(np.sum(n*(delta-np.log1p(delta)))+.5*np.sum(penalty*z*z)),jac.T@(1-n/lam)+penalty*z
    fit=minimize(objective,np.zeros(J.shape[1]),args=(J,b,pen),jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':600})
    nll,grad=objective(fit.x,J,b,pen);lam=b+J@fit.x
    require(np.max(np.abs(grad))<3e-5,'independent free optimizer gradient')
    H=(J.T*(n/lam**2))@J+np.diag(pen)
    sigma=scale*np.sqrt(np.linalg.inv(H)[0,0])
    profiles={}
    for A in set([0.,float(expected)]):
        fixed=minimize(objective,np.zeros(L.shape[1]),args=(L,b+A*p,np.ones(L.shape[1])),jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':600})
        pnll,pgrad=objective(fixed.x,L,b+A*p,np.ones(L.shape[1]))
        require(np.max(np.abs(pgrad))<3e-5,'independent profile optimizer gradient')
        profiles[A]=pnll
    q0=2*(profiles[0.]-nll);qtrue=2*(profiles[float(expected)]-nll)
    require(q0>=-2e-6 and qtrue>=-2e-6,'independent likelihood ordering')
    return dict(Ahat=float(fit.x[0]*scale),sigma_postfit=float(sigma),free_nll=float(nll),
                q_true=max(0.,qtrue),signed_root=float(np.sign(fit.x[0])*np.sqrt(max(0.,q0))))

def signature():
    parts=[(r['path'],r['sha256']) for r in readj(B/'provenance/input_hashes.json')]
    for rel in ('scripts/core.py','scripts/run_study.py','protocol.json','inputs/cohorts.npz','inputs/templates.npz'):
        parts.append((rel,sha(B/rel)))
    return hashlib.sha256(json.dumps(sorted(parts)).encode()).hexdigest()

def cohort_keys(cohort):return ('nominal',) if cohort=='pilot' else TRUTHS
def levels_for(cohort):return (0,) if cohort=='pilot' else GRID if cohort=='calibration' else LEVELS
def load_rows(cohort):
    files=sorted((B/f'results/{cohort}').glob('*_m*_t*.csv'))
    require(bool(files),'checkpoint CSVs '+cohort)
    return pd.concat([readcsv(p) for p in files],ignore_index=True),files

def check_cohorts_and_templates():
    data,templates=independent_templates()
    truth={'nominal':np.load(B/'inputs/null_2016.npz')['truth'],'stress':data['stress']}
    saved=dict(np.load(B/'inputs/cohorts.npz'))
    independent_draw_hashes=[];spectra=0
    for ci,cohort in enumerate(('pilot','calibration','evaluation'),1):
        for source in cohort_keys(cohort):
            a=saved[cohort+'_'+source]
            require(a.shape==(2,100,720),'paired exposure cohort shape')
            require(np.issubdtype(a.dtype,np.integer) and np.all(a>=0),'integer cohort counts')
            for toy in range(100):
                draws=[]
                for inc,frac in [(0,.1),(1,.9)]:
                    key=[MASTER,ci,TRUTHS.index(source),toy,inc]
                    draw=np.random.default_rng(np.random.SeedSequence(key)).poisson(frac*truth[source])
                    draws.append(draw);independent_draw_hashes.append(ahash(draw))
                require(np.array_equal(a[0,toy],draws[0]),'exact10% background replay')
                require(np.array_equal(a[1,toy],draws[0]+draws[1]),'exact100% background/increment replay')
                require(np.all(a[1,toy]>=a[0,toy]),'nested exposure Poisson counts')
                spectra+=2
    require(len(set(independent_draw_hashes))==len(independent_draw_hashes),'distinct cohort/truth/increment RNG streams')
    ts=dict(np.load(B/'inputs/templates.npz'))
    require(np.array_equal(ts['masses_MeV'],MASSES),'saved template mass grid')
    require(ts['categories'].shape==(10,722) and ts['masks'].shape==(10,720),'Gaussian template/mask shapes')
    for mi,m in enumerate(MASSES):
        close(ts['categories'][mi],templates[m]['categories'],'independent Gaussian CDF',atol=3e-15)
        require(np.array_equal(ts['masks'][mi],templates[m]['mask']),'fixed pole mask')
    return dict(background_spectra=spectra,independent_Poisson_components=len(independent_draw_hashes),
        exact_seed_replay=True,exposure_pairing='B10~Pois(.1B); B100=B10+Pois(.9B)',gaussian_templates=10)

def check_rows():
    out={}
    for cohort in ('pilot','calibration','evaluation'):
        frame,files=load_rows(cohort);levels=levels_for(cohort)
        keys=['source','exposure','mass_MeV','z','toy']
        expected={(s,e,m,z,t) for s in cohort_keys(cohort) for e in EXPOSURES for m in MASSES for z in levels for t in range(100)}
        require(len(frame)==len(expected),'attempted row total '+cohort)
        require(not frame.duplicated(keys).any(),'unique row keys '+cohort)
        require(set(frame[keys].itertuples(index=False,name=None))==expected,'complete planned row IDs '+cohort)
        require(set(frame.cohort)=={cohort},'cohort column identity')
        fv,pv,nv=boolcol(frame,'fit_valid'),boolcol(frame,'profile_valid'),boolcol(frame,'native_cls90_valid')
        require(not np.any(pv & ~fv),'profile requires valid free fit')
        require(not np.any(nv & ~fv),'native limit requires valid free fit')
        for r in frame.to_dict('records'):
            valid=str(r['fit_valid']).lower()=='true';profile=str(r['profile_valid']).lower()=='true';native=str(r['native_cls90_valid']).lower()=='true'
            require(r['sigma_method']=='observed_profile_hessian','observed curvature sigma')
            if valid:
                for k in ('Ahat','sigma_postfit','free_nll','fit_score','min_lambda','pull'):
                    require(np.isfinite(float(r[k])),'finite valid-fit '+k)
                require(float(r['sigma_postfit'])>0 and float(r['min_lambda'])>0 and float(r['fit_score'])<3e-5,'free numerical acceptance')
                close(r['pull'],(r['Ahat']-r['A_expected'])/r['sigma_postfit'],'expected-yield pull')
                require(0<int(r['nuisance_rank'])<=int(r['fit_bins']),'GP covariance rank')
                require(0<float(r['covariance_load'])<=1e-5 and float(r['max_omitted_covariance_mode'])<=1.00001e-8,'covariance factor acceptance')
            else:require(bool(str(r['failure_reason']).strip()),'free failure reason')
            if profile:
                require(cohort=='evaluation','truth profiles only evaluation')
                qraw=2*(float(r['true_nll'])-float(r['free_nll']))
                require(qraw>=-2e-6 and float(r['profile_score'])<3e-5 and float(r['profile_min_lambda'])>0,'profile numerical acceptance')
                close(r['q_true_raw'],qraw,'raw likelihood ratio');close(r['q_true'],max(0.,qraw),'clipped likelihood ratio')
                for name,thr in [('profile_contains68',1.),('profile_contains95',3.841459)]:
                    require((str(r[name]).lower()=='true')==(max(0.,qraw)<=thr),'profile containment threshold')
            elif cohort=='evaluation':require(bool(str(r['failure_reason']).strip()),'profile failure reason')
            if native:
                require(cohort=='evaluation' and r['exposure']==1.,'nativeCLs planned scope')
                require(np.isfinite(float(r['native_cls90'])) and float(r['native_cls90'])>=0,'finite nonnegative nativeCLs endpoint')
                require(float(r['native_cls90_max_score'])<3e-5 and float(r['native_cls90_root_error'])<2e-6,'nativeCLs solver gates')
                require((str(r['native_cls90_contains']).lower()=='true')==(float(r['native_cls90'])>=r['A_expected']),'nativeCLs truth containment')
            elif cohort=='evaluation' and r['exposure']==1. and valid:
                require(bool(str(r['native_cls90_failure']).strip()),'nativeCLs failure reason')
            if str(r.get('attempts_json','')).strip():
                attempts=json.loads(r['attempts_json']);require(1<=len(attempts)<=3 and len(attempts)==int(r['attempt_count']),'bounded retained attempts')
                if valid:close(r['free_nll'],min(a['free_nll'] for a in attempts if a.get('free_valid')),'minimum valid objective')
                if profile:close(r['true_nll'],min(a['true_nll'] for a in attempts if a.get('profile_valid')),'minimum valid profile objective')
        aggregate=readcsv(B/f'results/{cohort}_rows.csv')
        pd.testing.assert_frame_equal(frame.sort_values(keys).reset_index(drop=True),aggregate[frame.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=True)
        out[cohort]=dict(attempted=len(frame),fit_valid=int(fv.sum()),profile_valid=int(pv.sum()),nativeCLs90_valid=int(nv.sum()))
    return out

def check_replay():
    cohorts=dict(np.load(B/'inputs/cohorts.npz'));saved=dict(np.load(B/'inputs/templates.npz'))
    data,templates=independent_templates();ref=readj(B/'pilot_reference.json')
    checked=0;positive=0
    for cohort in ('pilot','calibration','evaluation'):
        _,files=load_rows(cohort)
        for path in files:
            frame=readcsv(path);pairs={}
            stored=None
            if cohort!='pilot':
                pack=dict(np.load(path.with_suffix('.npz')))
                require(len(pack['draws'])==len(frame) and len(pack['keys'])==len(frame),'saved signal draw count')
                stored={tuple(map(int,k)):d for k,d in zip(pack['keys'],pack['draws'])}
                require(len(stored)==len(frame),'unique saved signal keys')
            for r in frame.itertuples(index=False):
                mi=MASSES.index(r.mass_MeV);ei=EXPOSURES.index(r.exposure)
                cats,mask=saved['categories'][mi],saved['masks'][mi]
                bg=cohorts[cohort+'_'+r.source][ei,r.toy]
                A=0. if cohort=='pilot' else r.z*ref['masses'][str(r.mass_MeV)][str(r.exposure)]['s0']
                close(r.A_expected,A,'frozen expected yield',atol=0.)
                if cohort!='pilot':close(r.s0,ref['masses'][str(r.mass_MeV)][str(r.exposure)]['s0'],'shared source scale')
                key=(r.toy,r.z)
                if key not in pairs:
                    if r.z==0:pair=np.zeros((2,722),dtype=np.int64)
                    else:
                        a10=r.z*ref['masses'][str(r.mass_MeV)]['0.1']['s0'];a100=r.z*ref['masses'][str(r.mass_MeV)]['1.0']['s0']
                        require(a100>a10,'nonnegative signal exposure increment')
                        namespace=20 if cohort=='calibration' else 30
                        low=np.random.default_rng(np.random.SeedSequence([MASTER,namespace,TRUTHS.index(r.source),r.mass_MeV,r.toy,r.z,0])).poisson(a10*cats)
                        inc=np.random.default_rng(np.random.SeedSequence([MASTER,namespace,TRUTHS.index(r.source),r.mass_MeV,r.toy,r.z,1])).poisson((a100-a10)*cats)
                        pair=np.array([low,low+inc]);positive+=2
                    pairs[key]=pair
                draw=pairs[key][ei];counts=bg+draw[1:-1]
                for name,arr in [('background_hash',bg),('signal_hash',draw),('counts_hash',counts),('template_hash',cats),('mask_hash',mask)]:
                    require(getattr(r,name)==ahash(arr),'exact replay '+name)
                for name,val in [('actual_full',draw.sum()),('actual_support',draw[1:-1].sum()),('actual_window',draw[1:-1][mask].sum()),('actual_training',draw[1:-1][~mask].sum()),('actual_outside_support',draw[0]+draw[-1])]:
                    require(getattr(r,name)==int(val),'realized region count '+name)
                if stored is not None:require(np.array_equal(stored[(r.toy,ei,r.z)],draw),'saved signal vector replay')
                close(r.window_fraction,cats[1:-1][mask].sum(),'window probability');close(r.training_fraction,cats[1:-1][~mask].sum(),'sideband probability')
                require(r.fit_bins==int(mask.sum()),'fit bins');close(r.nominal_sigma_MeV,1000*templates[r.mass_MeV]['sigma'],'2016 resolution')
                idx=int(np.flatnonzero(data['masses']==r.mass_MeV)[0]);close(r.kernel_const,data['const'][idx],'archived kernel amplitude');close(r.kernel_ls,data['ls'][idx],'archived kernel length')
                checked+=1
    return dict(replayed_rows=checked,positive_signal_vectors=positive,paired_exposure_signal_increments=True,
                equal_expected_yields_across_sources=True,background_reuse_across_mass_and_levels=True)

def check_calibration_statistics():
    cal=readcsv(B/'results/calibration_rows.csv')
    pars=readcsv(B/'results/calibration_parameters.csv');quantiles=readcsv(B/'results/calibration_quantiles.csv')
    require(len(pars)==40 and len(quantiles)==360,'calibration summary sizes')
    cells={k:q.sort_values('toy') for k,q in cal.groupby(['source','exposure','mass_MeV','z'])}
    expected_pars={};bootstrap={}
    replicates=int(readj(B/'results/summary.json')['bootstrap_replicates'])
    indices={s:np.stack([np.random.default_rng(np.random.SeedSequence([MASTER,4,0,r,si+1,0])).integers(0,100,100)
               for r in range(replicates)]) for si,s in enumerate(TRUTHS)}
    for r in pars.itertuples(index=False):
        null=cells[r.source,r.exposure,r.mass_MeV,0];pos=cells[r.source,r.exposure,r.mass_MeV,3]
        x=np.where(boolcol(null,'fit_valid'),null.Ahat.astype(float),np.nan)
        err=np.where(boolcol(null,'fit_valid'),null.sigma_postfit.astype(float),np.nan)
        y=np.where(boolcol(pos,'fit_valid'),pos.Ahat.astype(float),np.nan)
        delta=np.nanmean(x);response=(y-x)/float(pos.A_expected.iloc[0]);R=np.nanmean(response)
        k=np.nanstd((x-delta)/err,ddof=1)
        close(r.delta,delta,'calibration offset');close(r.R,R,'paired calibration response');close(r.k0,k,'centered calibration width')
        close(r.delta_se,np.nanstd(x,ddof=1)/np.sqrt(np.isfinite(x).sum()),'calibration offset SE')
        close(r.R_se,np.nanstd(response,ddof=1)/np.sqrt(np.isfinite(response).sum()),'paired response SE')
        close(r.raw_null_pull_mean,np.nanmean(x/err),'raw null pull mean')
        ix=indices[r.source];db=np.nanmean(x[ix],axis=1);rb=np.nanmean(response[ix],axis=1)
        kb=np.nanstd((x[ix]-db[:,None])/err[ix],axis=1,ddof=1)
        cov=np.cov(db,rb,ddof=1)
        close([r.var_delta,r.var_R,r.cov_delta_R],[cov[0,0],cov[1,1],cov[0,1]],'paired delta-response covariance')
        for name,v in [('delta',db),('R',rb),('k0',kb)]:
            close(getattr(r,name+'_bootstrap_se'),np.std(v,ddof=1),'calibration bootstrap SE '+name)
            close([getattr(r,name+'_bootstrap_ci95_lo'),getattr(r,name+'_bootstrap_ci95_hi')],np.quantile(v,[.025,.975]),'calibration bootstrap interval '+name)
        key=(r.source,r.exposure,r.mass_MeV)
        expected_pars[key]=dict(delta=delta,R=R,pull=np.nanmean(x/err),sigma=np.nanmean(err))
        bootstrap[key]=dict(delta=db,R=rb,pull=np.nanmean((x/err)[ix],axis=1),sigma=np.nanmean(err[ix],axis=1))
    lo,hi=beta.ppf([.025,.975],10,91)
    for r in quantiles.itertuples(index=False):
        cell=cells[r.source,r.exposure,r.mass_MeV,r.z]
        vals=np.sort(cell.loc[boolcol(cell,'fit_valid'),'Ahat'].astype(float).to_numpy())
        require(r.n_calibration==len(vals) and r.rank_order90==10,'calibration quantile accounting')
        require((str(r.valid).lower()=='true')==(len(vals)==100),'calibration quantile validity')
        if len(vals)==100:close(r.critical90,vals[9],'10th calibration order statistic')
        close([r.tail_beta95_lo,r.tail_beta95_hi,r.acceptance_beta95_lo,r.acceptance_beta95_hi],[lo,hi,1-hi,1-lo],'finite calibration Beta interval')
        close(r.rank_marginal_acceptance_lower,91/101,'finite rank marginal bound')
        require(r.distinct_statistics==len(np.unique(vals)),'calibration tie accounting')
    scaling=readcsv(B/'results/scaling_comparisons.csv');require(len(scaling)==20,'exposure comparison count')
    for r in scaling.itertuples(index=False):
        a=expected_pars[r.source,.1,r.mass_MeV];b=expected_pars[r.source,1.,r.mass_MeV]
        ab=bootstrap[r.source,.1,r.mass_MeV];bb=bootstrap[r.source,1.,r.mass_MeV]
        values={'delta_difference':b['delta']-10*a['delta'],'pull_difference':b['pull']-np.sqrt(10)*a['pull'],
                'sigma_scaling_ratio':b['sigma']/(np.sqrt(10)*a['sigma']),'R_difference':b['R']-a['R']}
        replic={'delta_difference':bb['delta']-10*ab['delta'],'pull_difference':bb['pull']-np.sqrt(10)*ab['pull'],
                'sigma_scaling_ratio':bb['sigma']/(np.sqrt(10)*ab['sigma']),'R_difference':bb['R']-ab['R']}
        for name,v in values.items():
            close(getattr(r,name),v,'exposure comparison '+name)
            close(getattr(r,name+'_se'),np.std(replic[name],ddof=1),'paired exposure bootstrap SE')
            close([getattr(r,name+'_ci95_lo'),getattr(r,name+'_ci95_hi')],np.quantile(replic[name],[.025,.975]),'paired exposure bootstrap interval')
    return dict(calibration_parameter_cells=40,quantile_cells=360,exposure_comparisons=20,bootstrap_replicates=replicates,
                calibration_uncertainty_and_frozen_table_uncertainty_distinguished=True)

def check_rank_results():
    cal=readcsv(B/'results/calibration_rows.csv');ev=readcsv(B/'results/evaluation_rows.csv')
    cells={k:q.sort_values('toy') for k,q in cal.groupby(['source','exposure','mass_MeV','z'])}
    evrows={(r.source,r.exposure,r.mass_MeV,r.z,r.toy):r for r in ev.itertuples(index=False)}
    rank=readcsv(B/'results/rank_grid_rows.csv');limits=readcsv(B/'results/limit_rows.csv')
    require(len(rank)==20000 and len(limits)==48000,'planned rank/native decision rows')
    keys=['source','calibration_source','exposure','mass_MeV','z','toy']
    require(not rank.duplicated(keys).any(),'unique rank decision keys')
    require(not limits.duplicated(keys+['method']).any(),'unique limit decision keys')
    expected={};tables={}
    for r in rank.itertuples(index=False):
        er=evrows[r.source,r.exposure,r.mass_MeV,r.z,r.toy]
        key=(r.calibration_source,r.exposure,r.mass_MeV)
        if key not in tables:
            raw=[cells[key+(z,)] for z in GRID]
            valid=all(len(c)==100 and boolcol(c,'fit_valid').all() for c in raw)
            tables[key]=(valid,[np.sort(c.Ahat.astype(float).to_numpy()) for c in raw],np.array([float(c.A_expected.iloc[0]) for c in raw]))
        table_valid,sorted_tables,gridA=tables[key]
        valid=table_valid and str(er.fit_valid).lower()=='true'
        require((str(r.valid).lower()=='true')==valid,'rank valid complete calibration table')
        rid=tuple(getattr(r,k) for k in keys)
        if valid:
            p=np.array([(1+np.searchsorted(v,er.Ahat,side='right'))/101 for v in sorted_tables])
            cls=np.minimum(1.,p/p[0])
            close(json.loads(r.p_grid_json),p,'independent inclusive rank grid',atol=0.)
            close(json.loads(r.cls_grid_json),cls,'independent estimator-ordered MC-CLs',atol=0.)
            expected[rid]=(p,cls,gridA)
        else:expected[rid]=None
    empty=holes=right=0
    for r in limits.itertuples(index=False):
        if r.method=='native_profile_cls90':
            er=evrows[r.source,r.exposure,r.mass_MeV,r.z,r.toy]
            valid=str(er.native_cls90_valid).lower()=='true'
            require((str(r.valid).lower()=='true')==valid,'native limit validity preserved')
            if valid:
                close(r.U_grid,er.native_cls90,'native profileCLs endpoint preserved')
                require(float(r.upper_contains_truth)==float(float(er.native_cls90)>=er.A_expected),'native profileCLs coverage preserved')
            continue
        rid=tuple(getattr(r,k) for k in keys);item=expected[rid]
        require((str(r.valid).lower()=='true')==(item is not None),'rank limit validity')
        if item is None:continue
        p,cls,gridA=item;pv=p if r.method=='rank_neyman90' else cls
        require(r.method in ('rank_neyman90','rank_cls90'),'declared rank methods')
        accepted=np.flatnonzero(pv>.1);is_empty=len(accepted)==0;is_right=bool(8 in accepted)
        is_holey=bool(len(accepted)>0 and len(accepted)!=accepted[-1]-accepted[0]+1)
        U=float(gridA[accepted[-1]]) if len(accepted) else 0.
        close(r.p_at_truth,pv[r.z],'true-grid pvalue')
        require(float(r.accepted_at_truth)==float(r.z in accepted),'true-grid acceptance separate from envelope')
        close(r.U_grid,U,'discrete upper envelope')
        require(float(r.upper_contains_truth)==float(U>=r.A_expected-1e-9),'upper-envelope coverage')
        require(json.loads(r.accepted_z_json)==list(accepted),'accepted grid set')
        for name,val in [('empty_set',is_empty),('right_censored',is_right),('holey_set',is_holey)]:
            require((str(getattr(r,name)).lower()=='true')==val,'set geometry '+name)
        empty+=is_empty;holes+=is_holey;right+=is_right
    summary=readcsv(B/'results/limit_summary.csv')
    groupkeys=['source','calibration_source','exposure','mass_MeV','z','method']
    grouped={k:q for k,q in limits.groupby(groupkeys)}
    require(len(summary)==len(grouped)==480,'complete limit summaries')
    for r in summary.itertuples(index=False):
        group=grouped[tuple(getattr(r,k) for k in groupkeys)];valid=boolcol(group,'valid');n=int(valid.sum())
        require(r.attempted==100 and r.valid==n,'summary valid decision denominator')
        for prefix,field in [('acceptance','accepted_at_truth'),('upper_coverage','upper_contains_truth')]:
            k=int(group.loc[valid,field].astype(float).sum())
            require(getattr(r,prefix+'_k')==k and getattr(r,prefix+'_n')==n,'contained decision accounting')
            if n:
                lo=0. if k==0 else beta.ppf(.025,k,n-k+1);hi=1. if k==n else beta.ppf(.975,k+1,n-k)
                close([getattr(r,prefix+'_cp95_lo'),getattr(r,prefix+'_cp95_hi')],[lo,hi],'heldout binomial interval')
                close(getattr(r,prefix+'_fraction'),k/n,'conditional heldout proportion')
            close([getattr(r,prefix+'_all_attempt_lo'),getattr(r,prefix+'_all_attempt_hi')],[k/100,(k+100-n)/100],'all-attempt bounds')
    return dict(rank_grid_rows=len(rank),limit_rows=len(limits),summary_cells=len(summary),
                empty_rank_sets=empty,holey_rank_sets=holes,right_censored_rank_sets=right,
                accepted_true_hypothesis_and_upper_envelope_checked_separately=True)

def check_freezes_and_checkpoints():
    sig=signature();ref=readj(B/'pilot_reference.json');calfreeze=readj(B/'calibration_freeze.json')
    require(ref['signature']==sig and calfreeze['signature']==sig,'frozen artifact computation signature')
    require(ref['pilot_rows_sha256']==sha(B/'results/pilot_rows.csv'),'pilot aggregate freeze hash')
    require(calfreeze['calibration_rows_sha256']==sha(B/'results/calibration_rows.csv'),'calibration aggregate freeze hash')
    require(calfreeze['pilot_reference_sha256']==sha(B/'pilot_reference.json'),'calibration references exact pilot')
    require(calfreeze['rows']==36000 and calfreeze['valid']==36000,'complete calibration beforefreeze')
    pilot=readcsv(B/'results/pilot_rows.csv')
    for m in MASSES:
        for exposure in EXPOSURES:
            q=pilot[(pilot.mass_MeV==m)&(pilot.exposure==exposure)]
            require(len(q)==100 and boolcol(q,'fit_valid').all(),'100 valid independent piloterrors')
            errors=q.sigma_postfit.astype(float).to_numpy();r=ref['masses'][str(m)][str(exposure)]
            close(r['s0'],np.mean(errors),'pilot frozen meanerror')
            close(r['sd_sigma'],np.std(errors,ddof=1),'pilot error SD');close(r['se_sigma'],np.std(errors,ddof=1)/10,'pilot error SE')
    ptime=dt.datetime.fromisoformat(ref['frozen_utc']);ctime=dt.datetime.fromisoformat(calfreeze['frozen_utc'])
    require(ptime<=ctime,'pilotbeforecalibration freeze')
    calibration_files={str(p.relative_to(B)):sha(p) for p in sorted((B/'results/calibration').glob('*'))}
    require(calfreeze['checkpoint_hashes']==calibration_files,'all calibration checkpoint bytes frozen')
    checkpoint_count=0;first_cal=[];first_eval=[];coverage={}
    for cohort in ('pilot','calibration','evaluation'):
        markers=sorted((B/f'results/{cohort}').glob('*_m*_t*.json'))
        require(len(markers)==(100 if cohort=='pilot' else 200),'checkpoint count '+cohort)
        for path in markers:
            r=readj(path);require(r['complete'] is True,'checkpoint complete flag')
            require(r['cohort']==cohort and r['source'] in cohort_keys(cohort) and r['mass_MeV'] in MASSES,'marker scientific coordinates')
            require(r['signature']==sig,'checkpoint computation signature')
            require(0<=r['start']<r['stop']<=100,'checkpoint IDbounds')
            nrows=(r['stop']-r['start'])*2*len(levels_for(cohort))
            require(r['rows']==nrows,'checkpoint rows')
            require(len(r['output_hashes'])==(1 if cohort=='pilot' else 2),'checkpoint outputcount')
            for name,digest in r['output_hashes'].items():require(sha(B/name)==digest,'checkpoint outputSHA '+name)
            require(len(readcsv(path.with_suffix('.csv')))==nrows,'checkpoint actualCSV rows')
            started=dt.datetime.fromisoformat(r['started_utc']);completed=dt.datetime.fromisoformat(r['completed_utc'])
            require(started<=completed,'checkpoint chronological order')
            if cohort=='pilot':
                require(completed<=ptime,'all pilot completedbeforefreeze')
                require(r['reference_sha256'] is None and r['calibration_sha256'] is None,'pilot references')
            else:
                require(r['reference_sha256']==sha(B/'pilot_reference.json'),'frozen pilotSHA eachchunk')
                require(started>=ptime,'calibration/eval afterpilotfreeze')
                if cohort=='calibration':
                    require(completed<=ctime,'all calibration completedbeforefreeze')
                    require(r['calibration_sha256'] is None,'calibration has no evaluationfreeze dependency')
                    first_cal.append(started)
                else:
                    require(r['calibration_sha256']==sha(B/'calibration_freeze.json'),'frozen calibrationSHA eachchunk')
                    require(started>=ctime,'all evaluation aftercalibrationfreeze');first_eval.append(started)
            key=(cohort,r['source'],r['mass_MeV']);coverage.setdefault(key,[]).extend(range(r['start'],r['stop']))
            checkpoint_count+=1
    for key,ids in coverage.items():require(sorted(ids)==list(range(100)),'unique full checkpoint IDs '+str(key))
    protocol=readj(B/'protocol.json')
    require(any('exception' in text.lower() for text in protocol['limitations']),'2016 numericalexception qualification retained')
    require(protocol['master_seed']==MASTER and protocol['masses_MeV']==list(MASSES),'frozen scientificsettings')
    require(protocol['resources']['local_workers']<=4 and protocol['resources']['numerical_threads_per_worker']==1,'bounded resource settings')
    return dict(checkpoints=checkpoint_count,signature=sig,pilot_frozen_utc=ref['frozen_utc'],
                first_calibration_started_utc=min(first_cal).isoformat(),calibration_frozen_utc=calfreeze['frozen_utc'],
                first_evaluation_started_utc=min(first_eval).isoformat(),all_checkpoint_cache_eligibility_recomputed=True)

def check_representative_fits():
    ev=readcsv(B/'results/evaluation_rows.csv');cohorts=dict(np.load(B/'inputs/cohorts.npz'))
    ref=readj(B/'pilot_reference.json');templates=dict(np.load(B/'inputs/templates.npz'));details=[]
    for source,e,m,z in [('stress',.1,42,5),('stress',1.,76,3),('nominal',1.,92,5),('nominal',.1,178,0)]:
        toy=0;r=ev[(ev.source==source)&(ev.exposure==e)&(ev.mass_MeV==m)&(ev.z==z)&(ev.toy==toy)].iloc[0]
        cats=templates['categories'][MASSES.index(m)];bg=cohorts['evaluation_'+source][EXPOSURES.index(e),toy]
        if z:
            a10=z*ref['masses'][str(m)]['0.1']['s0'];afull=z*ref['masses'][str(m)]['1.0']['s0']
            low=np.random.default_rng(np.random.SeedSequence([MASTER,30,TRUTHS.index(source),m,toy,z,0])).poisson(a10*cats)
            inc=np.random.default_rng(np.random.SeedSequence([MASTER,30,TRUTHS.index(source),m,toy,z,1])).poisson((afull-a10)*cats)
            draw=low if e==.1 else low+inc
        else:draw=np.zeros(722,dtype=np.int64)
        counts=bg+draw[1:-1];b,L,mask=independent_gp(counts,m)
        fitted=independent_fit(counts[mask],b,L,cats[1:-1][mask],r.A_expected)
        for field in ('Ahat','sigma_postfit','free_nll','q_true'):
            close(fitted[field],float(r[field]),'independent BFGS '+field,rtol=1e-8,atol=.003 if field=='Ahat' else 2e-6)
        record=dict(source=source,exposure=e,mass_MeV=m,z=z,toy=0,
                    Ahat_difference=float(fitted['Ahat']-r.Ahat),sigma_difference=float(fitted['sigma_postfit']-r.sigma_postfit),
                    q_difference=float(fitted['q_true']-r.q_true),optimizer='independent SciPy BFGS/likelihood/GP')
        if e==1. and str(r.native_cls90_valid).lower()=='true':
            sys.path.insert(0,str(B/'inputs/v6p1/scripts'))
            from limit_solver import OneSignalProfile
            limit=OneSignalProfile(b,L,cats[1:-1][mask]).limit(counts[mask],alpha=.1)
            close(limit['A90'],r.native_cls90,'nativeCLs count-unit wiring from independentGP',rtol=2e-8,atol=.003)
            record['native_cls90_difference']=float(limit['A90']-float(r.native_cls90))
        details.append(record)
    return details

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only',action='store_true')
    args=parser.parse_args()
    audit=Audit()
    audit.run('immutable_inputs_and2016_provenance',check_inputs)
    audit.run('rank_test_finite_calibration_algebra',check_rank_algebra)
    if not args.inputs_only:
        audit.run('independent_cohorts_templates_and_paired_exposures',check_cohorts_and_templates)
        audit.run('complete_rows_and_numerical_diagnostics',check_rows)
        audit.run('every_signal_count_and_row_hash_replay',check_replay)
        audit.run('pilot_calibration_freezes_and_checkpoint_signatures',check_freezes_and_checkpoints)
        audit.run('independent_GP_and_representative_optimizers',check_representative_fits)
        audit.run('calibration_scaling_bootstrap_and_Beta_uncertainty',check_calibration_statistics)
        audit.run('rank_Neyman_MC_CLs_and_grid_coverage',check_rank_results)
    passed=all(r['passed'] for r in audit.checks)
    out=dict(status='passed' if passed else 'failed',inputs_only=args.inputs_only,
             checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),validator_sha256=sha(__file__),checks=audit.checks,
             scope='Numerical/source/replay/statistical-computation audit. Bias and undercoverage are measured outcomes.')
    target=B/'qa/independent_validation.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
