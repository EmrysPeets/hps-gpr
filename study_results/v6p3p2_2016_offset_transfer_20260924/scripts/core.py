"""2016 conditional exposure transfer; full-count Gaussian templates."""
from pathlib import Path
import os,sys,json,hashlib,time,datetime
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v632-mpl')
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'inputs/v6p1/scripts'))
import common as C
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.linalg import cholesky,cho_solve,solve_triangular
MASTER=63420160924
MASSES=(42,44,60,66,76,90,92,117,160,178)
EXPOSURES=(.1,1.)
SOURCES=('nominal','stress')
LEVELS=(0,1,3,5)
GRID=tuple(range(9))
NTOYS=100
D=C.DATA['2016']
NULL=dict(np.load(B/'inputs/null_2016.npz'))
TRUTHS={'nominal':NULL['truth'],'stress':D['stress']}
assert np.array_equal(D['edges'],NULL['edges_GeV'])
assert np.array_equal(D['n'],NULL['observed'])
assert all(np.all(np.isfinite(v)) and np.all(v>0) for v in TRUTHS.values())

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write_json(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rng(key):return np.random.default_rng(np.random.SeedSequence(key))
def background_key(cohort,source,toy,increment):
    return [MASTER,{'pilot':1,'calibration':2,'evaluation':3}[cohort],SOURCES.index(source),int(toy),int(increment)]
def signal_key(cohort,source,m,toy,z,increment):
    return [MASTER,{'calibration':20,'evaluation':30,'smoke':90}[cohort],SOURCES.index(source),int(m),int(toy),int(z),int(increment)]
def signature():
    inputs=json.loads((B/'provenance/input_hashes.json').read_text())
    for r in inputs:
        if sha(B/r['path'])!=r['sha256']:raise RuntimeError('Pinned input changed:'+r['path'])
    parts=[(r['path'],r['sha256']) for r in inputs]
    for rel in ('scripts/core.py','scripts/run_study.py','protocol.json','inputs/cohorts.npz','inputs/templates.npz'):
        parts.append((rel,sha(B/rel)))
    return hashlib.sha256(json.dumps(sorted(parts)).encode()).hexdigest()

class Context:
    def __init__(self,m):
        self.m=m;self.sigma=C.sigma('2016',m)
        x=D['x'];lo,hi=m/1000-2.25*self.sigma,m/1000+2.25*self.sigma
        self.mask=(x>=lo)&(x<=hi)
        assert (x<lo).sum()>=3 and (x>hi).sum()>=3
        self.const,self.ls=C.kernel_state('2016',m)
        self.K=C.kernel(x[~self.mask],x[~self.mask],self.const,self.ls)
        self.Kqt=C.kernel(x[self.mask],x[~self.mask],self.const,self.ls)
        self.Kqq=C.kernel(x[self.mask],x[self.mask],self.const,self.ls)
        cdf=ndtr((D['edges']-m/1000)/self.sigma)
        self.categories=np.r_[cdf[0],np.diff(cdf),1-cdf[-1]]
        self.p=self.categories[1:-1]
        assert self.categories.min()>=0 and abs(self.categories.sum()-1)<1e-12
    def predict(self,counts):
        n=np.asarray(counts,float)[~self.mask];pos=n>0
        target=np.zeros_like(n);target[pos]=np.log(n[pos])
        alpha=np.ones_like(n);alpha[pos]=1/n[pos]
        K=self.K.copy();K.flat[::len(K)+1]+=alpha
        L=cholesky(K,lower=True,check_finite=False)
        mu=self.Kqt@cho_solve((L,True),target,check_finite=False)
        v=solve_triangular(L,self.Kqt.T,lower=True,check_finite=False)
        cov=self.Kqq-v.T@v;cov=.5*(cov+cov.T)
        b=np.exp(mu+.5*np.maximum(np.diag(cov),0))
        covariance=np.outer(b,b)*np.expm1(np.clip(cov,-40,40))
        factor,diag=C.factor_cov(covariance,b)
        return b,factor,diag

def checked_fit(model,n,fixed=None,initial=None):
    fit=model.fit(n,fixed=fixed,initial=initial)
    assert np.isfinite(fit['nll']) and np.isfinite(fit['score']) and fit['score']<3e-5
    assert fit['min_lambda']>0
    if fixed is None:
        _,_,H,_=model._objective(fit['z'],n,model.Jfree,model.b,model.penfree)
        unit=np.zeros(len(H));unit[0]=1
        variance=float(cho_solve((cholesky(H,lower=True),True),unit)[0])
        assert variance>0 and np.isfinite(variance)
        sigma=float(model.scale*np.sqrt(variance))
        assert np.isfinite(sigma) and sigma>0 and abs(fit['sigma']/sigma-1)<1e-10
        fit['sigma']=sigma
    return fit

def fit_pair(b,L,p,n,expected,need_profile):
    attempts=[];frees=[];profiles=[]
    for attempt in range(3):
        tolerance=2e-7 if attempt==0 else 2e-9
        model=C.OneSignalProfile(b,L,p,score_tolerance=tolerance)
        initial=None if attempt<2 else np.r_[.5,np.zeros(model.rank)]
        if attempt>0 and profiles:
            previous=min(profiles,key=lambda q:q['nll'])
            initial=np.r_[expected/model.scale,previous['theta']]
        rec=dict(attempt=attempt,tolerance=tolerance);free=None
        try:
            free=checked_fit(model,n,initial=initial);frees.append(free)
            rec.update(free_valid=True,free_nll=free['nll'],free_score=free['score'],sigma=free['sigma'])
        except Exception as e:rec.update(free_valid=False,free_error=type(e).__name__+':'+str(e))
        if free is not None and need_profile:
            try:
                fixed=checked_fit(model,n,fixed=float(expected),initial=free['theta'] if attempt!=1 else np.zeros(model.rank))
                profiles.append(fixed);rec.update(profile_valid=True,true_nll=fixed['nll'],profile_score=fixed['score'])
            except Exception as e:rec.update(profile_valid=False,profile_error=type(e).__name__+':'+str(e))
        attempts.append(rec)
        if frees and (not need_profile or (profiles and 2*(min(f['nll'] for f in profiles)-min(f['nll'] for f in frees))>=-2e-6)):
            break
    free=min(frees,key=lambda f:f['nll']) if frees else None
    fixed=min(profiles,key=lambda f:f['nll']) if profiles else None
    valid=bool(free is not None and fixed is not None and 2*(fixed['nll']-free['nll'])>=-2e-6)
    return free,fixed,valid,attempts

def make_row(ctx,background,draw,A,toy,source,exposure,z,cohort,s0=None,prediction=None,native_limit=False,known_background=False):
    signal=draw[1:-1];counts=background+signal
    row=dict(cohort=cohort,source=source,exposure=float(exposure),mass_MeV=ctx.m,toy=toy,z=z,A_expected=float(A),s0=s0,
        Ahat=None,sigma_postfit=None,pull=None,q_true=None,profile_contains68=None,profile_contains95=None,
        actual_full=int(draw.sum()),actual_support=int(signal.sum()),actual_window=int(signal[ctx.mask].sum()),
        actual_training=int(signal[~ctx.mask].sum()),actual_outside_support=int(draw[0]+draw[-1]),
        background_hash=ahash(background),counts_hash=ahash(counts),signal_hash=ahash(draw),
        template_hash=ahash(ctx.categories),mask_hash=ahash(ctx.mask),
        window_fraction=float(ctx.p[ctx.mask].sum()),training_fraction=float(ctx.p[~ctx.mask].sum()),
        kernel_const=ctx.const,kernel_ls=ctx.ls,fit_valid=False,profile_valid=False,failure_reason='',
        native_cls90=None,native_cls90_valid=False,native_cls90_failure='',sigma_method='observed_profile_hessian',
        fit_bins=int(ctx.mask.sum()),nominal_sigma_MeV=ctx.sigma*1000)
    try:
        if known_background:b,L,diag=exposure*TRUTHS[source][ctx.mask],np.zeros((ctx.mask.sum(),0)),dict(rank=0,load=0,max_omitted=0)
        else:b,L,diag=ctx.predict(counts) if prediction is None else prediction
        row.update(nuisance_rank=int(diag['rank']),covariance_load=float(diag['load']),max_omitted_covariance_mode=float(diag['max_omitted']),gp_mean_hash=ahash(b),gp_covariance_factor_hash=ahash(L))
        free,fixed,valid,attempts=fit_pair(b,L,ctx.p[ctx.mask],counts[ctx.mask],A,cohort in ('evaluation','smoke'))
        row.update(attempts_json=json.dumps(attempts),attempt_count=len(attempts),fit_valid=free is not None,profile_valid=valid)
        if free is not None:
            row.update(Ahat=free['A'],sigma_postfit=free['sigma'],pull=(free['A']-A)/free['sigma'],free_nll=free['nll'],
                fit_score=free['score'],min_lambda=free['min_lambda'],min_background=free['min_background'],fit_method=free['method'])
        else:row['failure_reason']='free_fit_failed'
        if fixed is not None:row.update(true_nll=fixed['nll'],profile_score=fixed['score'],profile_min_lambda=fixed['min_lambda'])
        if valid:
            qraw=2*(fixed['nll']-free['nll']);q=max(qraw,0.)
            row.update(q_true_raw=qraw,q_true=q,profile_contains68=q<=1.,profile_contains95=q<=3.841459)
        elif cohort in ('evaluation','smoke'):row['failure_reason']+=';profile_failed'
        if native_limit and free is not None:
            try:
                ul=C.OneSignalProfile(b,L,ctx.p[ctx.mask]).limit(counts[ctx.mask],alpha=.1)
                assert ul['ok'] and ul['max_score']<3e-5 and ul['min_lambda']>0 and abs(ul['cls']-.1)<2e-6
                row.update(native_cls90=ul['A90'],native_cls90_valid=True,native_cls90_contains=bool(ul['A90']>=A),
                    native_cls90_max_score=ul['max_score'],native_cls90_root_error=abs(ul['cls']-.1))
            except Exception as e:row['native_cls90_failure']=type(e).__name__+':'+str(e)
    except Exception as e:row['failure_reason']=type(e).__name__+':'+str(e)
    return row

def draw_pair(ctx,cohort,source,toy,z,reference):
    if z==0:return np.zeros((2,len(ctx.categories)),dtype=np.int64)
    a10=z*reference['masses'][str(ctx.m)]['0.1']['s0']
    a100=z*reference['masses'][str(ctx.m)]['1.0']['s0']
    assert a100>a10
    low=rng(signal_key(cohort,source,ctx.m,toy,z,0)).poisson(a10*ctx.categories)
    increment=rng(signal_key(cohort,source,ctx.m,toy,z,1)).poisson((a100-a10)*ctx.categories)
    return np.array([low,low+increment])

def chunk_base(cohort,source,m,start):return B/f'results/{cohort}/{source}_m{m:03d}_t{start:03d}'

def run_chunk(cohort,source,m,start,stop,sig,reference_sha,calibration_sha):
    if signature()!=sig:raise RuntimeError('Compute signature changed')
    if reference_sha is not None and sha(B/'pilot_reference.json')!=reference_sha:raise RuntimeError('Pilot reference changed')
    if calibration_sha is not None and sha(B/'calibration_freeze.json')!=calibration_sha:raise RuntimeError('Calibration freeze changed')
    base=chunk_base(cohort,source,m,start);marker=base.with_suffix('.json')
    if marker.exists():
        old=json.loads(marker.read_text())
        assert old['signature']==sig and old['reference_sha256']==reference_sha and old['calibration_sha256']==calibration_sha
        assert all(sha(B/p)==h for p,h in old['output_hashes'].items())
        return dict(cohort=cohort,source=source,mass=m,start=start,cached=True,rows=old['rows'])
    begin=time.monotonic();started=utc();ctx=Context(m)
    backgrounds=np.load(B/'inputs/cohorts.npz')[cohort+'_'+source]
    reference=json.loads((B/'pilot_reference.json').read_text()) if reference_sha is not None else None
    rows=[];signals=[];keys=[]
    for toy in range(start,stop):
        if (B/'STOP').exists():raise RuntimeError('STOP marker; completed chunks retained')
        predictions=[ctx.predict(backgrounds[e,toy]) for e in range(2)]
        for z in ((0,) if cohort=='pilot' else GRID if cohort=='calibration' else LEVELS):
            draw=np.zeros((2,len(ctx.categories)),dtype=np.int64) if z==0 else draw_pair(ctx,cohort,source,toy,z,reference)
            for e,f in enumerate(EXPOSURES):
                s0=reference['masses'][str(m)][str(f)]['s0'] if reference else None
                A=0. if z==0 else z*s0
                row=make_row(ctx,backgrounds[e,toy],draw[e],A,toy,source,f,z,cohort,s0,
                    prediction=predictions[e] if z==0 else None,native_limit=cohort=='evaluation' and f==1.)
                rows.append(row)
                if cohort!='pilot':signals.append(draw[e]);keys.append((toy,e,z))
    out=base.with_suffix('.csv');tmp=base.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(tmp,index=False,float_format='%.17g');tmp.replace(out);paths=[out]
    if cohort!='pilot':
        p=base.with_suffix('.npz');np.savez_compressed(p,draws=np.array(signals),keys=np.array(keys));paths.append(p)
    write_json(marker,dict(complete=True,cohort=cohort,source=source,mass_MeV=m,start=start,stop=stop,rows=len(rows),
        started_utc=started,completed_utc=utc(),seconds=time.monotonic()-begin,signature=sig,reference_sha256=reference_sha,
        calibration_sha256=calibration_sha,output_hashes={str(p.relative_to(B)):sha(p) for p in paths}))
    return dict(cohort=cohort,source=source,mass=m,start=start,cached=False,rows=len(rows),seconds=round(time.monotonic()-begin,3))
