#!/usr/bin/env python3
"""Raw conditional observed scans and forward-calibrated common-coupling anchors.

The common parameter is psi=epsilon^2/1e-8. The 2021 Gaussian is normalized
over the full selected yield; old campaign signal arrays are inherited exactly.
Independent pilot/calibration/evaluation cohorts use a native 2021 MC signal.
Finite rank inversions are discrete conditional diagnostics, never smooth limits.
"""
from pathlib import Path
import os, sys, json, hashlib, datetime, argparse
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.linalg import block_diag, cholesky, cho_solve
from scipy.stats import beta
from concurrent.futures import ProcessPoolExecutor
import core as K
C=K.C
B=Path(__file__).resolve().parents[1]
MASTER=63520210925
MASSES=tuple(range(60,241,20))
POLICIES=('pole','logshift')
GRID=tuple(K.GRID)
LEVELS=(0,1,3,5)
YEARS=('2015','2016','2021')
N=100

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def writej(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(tmp,index=False,float_format='%.17g');tmp.replace(p)
def readcsv(p):return pd.read_csv(p,keep_default_na=False,float_precision='round_trip')
def rng(key):return np.random.default_rng(np.random.SeedSequence(key))
def years(m):return tuple(y for y in YEARS if y=='2021' or (y=='2015' and m<=100) or (y=='2016' and m<=180))
def conversion(y,m):
    d=C.DATA[y];x=m/1000.;s=C.sigma(y,m);e=d['native_edges'];w=np.diff(e)
    overlap=np.maximum(0.,np.minimum(e[1:],x+1.64*s)-np.maximum(e[:-1],x-1.64*s))
    density=float(np.sum(d['native_counts']*overlap/w)/(3.28*s))
    return float(3*np.pi*x*float(d['frad_effective'])*density/(2/137.))
def context(y,m,policy):return K.Context(m,policy if y=='2021' else 'pole',year=y)
def signal(y,m,ctx):
    return ctx.p[ctx.mask]*conversion(y,m)*1e-8 if y=='2021' else C.signal(y,m,ctx.mask)
def model_from_parts(parts):
    n=np.concatenate([p[0] for p in parts]);b=np.concatenate([p[1] for p in parts])
    L=block_diag(*[p[2] for p in parts]);S=np.concatenate([p[3] for p in parts])
    return C.OneSignalProfile(b,L,S),n
def checked_free(model,n):
    f=model.fit(n);assert f['score']<3e-5 and f['min_lambda']>0
    _,_,H,_=model._objective(f['z'],n,model.Jfree,model.b,model.penfree)
    v=np.zeros(len(H));v[0]=1
    sigma=float(model.scale*np.sqrt(cho_solve((cholesky(H,lower=True),True),v)[0]))
    assert np.isfinite(sigma) and sigma>0 and abs(sigma/f['sigma']-1)<1e-10
    f['sigma']=sigma
    return f
def signed_root(model,n,f):
    null=model.fit(n,fixed=0.);assert null['score']<3e-5 and null['min_lambda']>0
    q=2*(null['nll']-f['nll']);assert q>=-2e-6
    return float(np.sign(f['A'])*np.sqrt(max(0.,q)))
def signature():
    files=[(r['path'],r['sha256']) for r in readj(B/'provenance/input_hashes.json')]
    for rel,digest in files:assert sha(B/rel)==digest, 'Changed pinned input: '+rel
    for rel in ('scripts/core.py','scripts/observed.py','protocol.json'):
        files.append((rel,sha(B/rel)))
    if (B/'results/combined_cohorts.npz').exists():files.append(('results/combined_cohorts.npz',sha(B/'results/combined_cohorts.npz')))
    return hashlib.sha256(json.dumps(sorted(files)).encode()).hexdigest()

def dense_one(m):
    if (B/'STOP').exists():raise RuntimeError('STOP marker; retain checkpoints')
    path=B/f'results/observed_chunks/m{m:03d}.csv';mark=path.with_suffix('.json');sig=signature()
    if mark.exists() and path.exists():
        old=readj(mark)
        if old['signature']==sig and old['sha256']==sha(path):return str(path)
    start=utc();oldparts={};rows=[]
    for y in years(m):
        if y=='2021':continue
        ctx=context(y,m,'pole');n=C.DATA[y]['n'];b,L,d=ctx.predict(n)
        oldparts[y]=(n[ctx.mask],b,L,signal(y,m,ctx),d)
    for policy in POLICIES:
        ctx=context('2021',m,policy);n=C.DATA['2021']['n'];b,L,d=ctx.predict(n)
        part=(n[ctx.mask],b,L,signal('2021',m,ctx),d)
        for scope,parts in [('2021',[part]),('combined',[oldparts[y] for y in years(m) if y!='2021']+[part])]:
            model,counts=model_from_parts(parts);free=checked_free(model,counts)
            result=model.limit(counts)
            assert result['ok'] and result['max_score']<3e-5 and result['min_lambda']>0
            assert abs(result['Ahat']-free['A'])<2e-6*free['sigma']
            assert abs(result['nll_free']-free['nll'])<2e-6
            row=dict(mass_MeV=m,policy=policy,scope=scope,campaigns='+'.join(years(m)) if scope=='combined' else '2021',
                psi_hat=result['Ahat'],sigma_psi=free['sigma'],psi90=result['A90'],epsilon2_90=result['A90']*1e-8,
                signed_root=result['signed_r'],p0_asymptotic=result['p0_fixed_mass'],
                cls=result['cls'],max_score=result['max_score'],min_lambda=result['min_lambda'],
                max_covariance_load=max(p[4]['load'] for p in parts),nuisance_rank=model.rank,
                calibration='raw conditional fixed-mass asymptotic',valid=True)
            rows.append(row)
    csv(path,rows);writej(mark,dict(signature=sig,started_utc=start,completed_utc=utc(),sha256=sha(path)))
    return str(path)

def prepare():
    path=B/'results/combined_cohorts.npz'
    if path.exists():return
    arrays={}
    for cohort,index in [('pilot',1),('calibration',2),('evaluation',3)]:
        for yi,y in enumerate(YEARS):
            truth=np.load(B/f'inputs/null_{y}.npz')['truth']
            arrays[f'{cohort}_{y}']=np.array([rng([MASTER,index,yi,t]).poisson(truth) for t in range(N)])
    np.savez_compressed(path,**arrays)
    writej(B/'qa/observed_protocol.json',dict(created_utc=utc(),signature=signature(),seed=MASTER,
        masses_MeV=MASSES,n_per_cohort=N,grid=GRID,evaluation_levels=LEVELS,policies=POLICIES,
        cohorts_sha256=sha(path),background_seed='[MASTER,cohort(1/2/3),yearindex,toy]',
        signal_seed='[MASTER,cohort(20/30),yearindex,mass,toy,z]',
        source='fixed campaign GP nominal means; source-estimation uncertainty unpropagated',
        signal='2021 full selected native MC; old years inherited support-normalized Gaussian',
        amplitude='common psi=epsilon2/1e-8; physical conversion evaluated at pole mass',
        calibration='100 independent calibration ranks, same generator and fixed shared coupling',
        transfer='Only 2021 signal policy changes. No affine division of combined limits.',
        limitations='Native anchors only; conditional marginal rank calibration, no dense or global calibration.'))

def categories(y,m):
    if y!='2021':
        p=C.signal(y,m)/(conversion(y,m)*1e-8)
        assert abs(p.sum()-1)<1e-12
        return np.r_[0.,p,0.]
    h=dict(np.load(B/f'inputs/v6p1/histograms/m{m:03d}.npz'))
    meta=json.loads(str(h['metadata']));cdf=np.r_[0.,np.cumsum(h['sumw'])]/float(meta['sumw'])
    c=np.interp(C.DATA[y]['edges'],h['edges_GeV'],cdf,left=0.,right=float(cdf[-1]))
    p=np.r_[c[0],np.diff(c),1-c[-1]];assert p.min()>=0 and abs(p.sum()-1)<1e-12
    return p

def joint_one(job):
    cohort,m=job;path=B/f'results/combined_{cohort}/m{m:03d}.csv';mark=path.with_suffix('.json')
    sig=signature();ref=None if cohort=='pilot' else readj(B/'results/combined_reference.json')
    frozen=None if cohort!='evaluation' else readj(B/'results/combined_calibration_freeze.json')
    refsha='' if ref is None else sha(B/'results/combined_reference.json')
    freezesha='' if frozen is None else sha(B/'results/combined_calibration_freeze.json')
    if ref is not None:assert ref['signature']==sig
    if frozen is not None:
        assert frozen['signature']==sig and frozen['reference_sha256']==refsha
        assert frozen['calibration_sha256']==sha(B/'results/combined_calibration.csv')
    if path.exists() and mark.exists():
        old=readj(mark)
        if old['signature']==sig and old['reference_sha256']==refsha and old['calibration_freeze_sha256']==freezesha and old['sha256']==sha(path) and old['draws_sha256']==sha(path.with_suffix('.draws.json')):return str(path)
    start=utc();cohorts=np.load(B/'results/combined_cohorts.npz');ys=years(m)
    contexts={(y,p):context(y,m,p) for y in ys for p in POLICIES if y=='2021' or p=='pole'}
    probs={y:categories(y,m) for y in ys};conversions={y:conversion(y,m)*1e-8 for y in ys}
    levels=(0,) if cohort=='pilot' else GRID if cohort=='calibration' else LEVELS
    s0=0. if ref is None else ref['masses'][str(m)]['s0_psi'];rows=[];drawhashes=[]
    for t in range(N):
        if (B/'STOP').exists():raise RuntimeError('STOP marker; retain checkpoints')
        for z in levels:
            psi=z*s0;counts={}
            for y in ys:
                base=cohorts[f'{cohort}_{y}'][t]
                draw=np.zeros(len(probs[y]),dtype=np.int64) if not z else rng([MASTER,20 if cohort=='calibration' else 30,YEARS.index(y),m,t,z]).poisson(psi*conversions[y]*probs[y])
                counts[y]=base+draw[1:-1]
                drawhashes.append(dict(toy=t,z=z,year=y,sha256=ahash(draw),drawn_total=int(draw.sum()),expected_total=float(psi*conversions[y])))
            oldparts={}
            for y in ys:
                if y=='2021':continue
                ctx=contexts[y,'pole'];b,L,d=ctx.predict(counts[y]);oldparts[y]=(counts[y][ctx.mask],b,L,signal(y,m,ctx),d)
            for policy in POLICIES:
                ctx=contexts['2021',policy];b,L,d=ctx.predict(counts['2021'])
                part=(counts['2021'][ctx.mask],b,L,signal('2021',m,ctx),d)
                parts=[oldparts[y] for y in ys if y!='2021']+[part];model,n=model_from_parts(parts);f=checked_free(model,n)
                root=signed_root(model,n,f) if cohort=='calibration' and z==0 else None
                rows.append(dict(cohort=cohort,mass_MeV=m,toy=t,z=z,policy=policy,campaigns='+'.join(ys),
                    psi_expected=psi,psi_hat=f['A'],sigma_psi=f['sigma'],pull=(f['A']-psi)/f['sigma'],
                    signed_root=root,free_nll=f['nll'],score=f['score'],min_lambda=f['min_lambda'],
                    max_covariance_load=max(p[4]['load'] for p in parts),nuisance_rank=model.rank,valid=True))
    csv(path,rows);drawpath=path.with_suffix('.draws.json');writej(drawpath,drawhashes)
    writej(mark,dict(signature=sig,reference_sha256=refsha,
        calibration_freeze_sha256=freezesha,
        started_utc=start,completed_utc=utc(),sha256=sha(path),draws_sha256=sha(drawpath),rows=len(rows)))
    return str(path)

def aggregate(cohort):
    files=sorted((B/f'results/combined_{cohort}').glob('m*.csv'))
    assert len(files)==len(MASSES)
    df=pd.concat([readcsv(p) for p in files],ignore_index=True)
    assert not df.duplicated(['mass_MeV','toy','z','policy']).any()
    csv(B/f'results/combined_{cohort}.csv',df)
    return df

def freeze_reference():
    path=B/'results/combined_reference.json'
    if path.exists():return
    d=aggregate('pilot');assert len(d)==2000
    masses={}
    for m in MASSES:
        g=d[(d.mass_MeV==m)&(d.policy=='pole')];assert len(g)==N
        masses[str(m)]=dict(s0_psi=float(g.sigma_psi.mean()),expected_psi=[float(z*g.sigma_psi.mean()) for z in GRID])
    writej(path,dict(frozen_utc=utc(),signature=signature(),masses=masses,pilot_sha256=sha(B/'results/combined_pilot.csv')))
def freeze_calibration():
    path=B/'results/combined_calibration_freeze.json'
    if path.exists():return
    d=aggregate('calibration');assert len(d)==len(MASSES)*N*len(POLICIES)*len(GRID)
    writej(path,dict(frozen_utc=utc(),signature=signature(),calibration_sha256=sha(B/'results/combined_calibration.csv'),
        reference_sha256=sha(B/'results/combined_reference.json'),statistic='signed psi_hat lower tail; inclusive ties',
        rank_rule='p_psi=(1+number of100 calibration psi_hat <= test psi_hat)/101; reject<=0.1',
        p0_rule='(1+number of100 null signed roots >= observed signed root)/101',
        alpha=.1,minimum_p=1/101,continuous_frozen_rejection_beta=[10,91]))

def ranks(cal,policy,m,test):
    g=cal[(cal.mass_MeV==m)&(cal.policy==policy)];ans=[]
    for z in GRID:
        a=g[g.z==z].psi_hat.to_numpy();assert len(a)==N
        ans.append((1+np.count_nonzero(a<=test))/101.)
    return np.array(ans)
def envelope(p,s0):
    keep=p>.1;return dict(upper_psi=float(max(np.array(GRID)[keep],default=0)*s0),empty=not bool(keep.any()),right_censored=bool(keep[-1]),
        holes=bool(keep.any() and np.any(~keep[:np.flatnonzero(keep)[-1]+1])))
def summarize():
    cal=aggregate('calibration');ev=aggregate('evaluation');obs=readcsv(B/'results/observed_dense.csv')
    ref=readj(B/'results/combined_reference.json');rows=[];tests=[];summaries=[]
    for m in MASSES:
        s0=ref['masses'][str(m)]['s0_psi']
        for p in POLICIES:
            observed=obs[(obs.mass_MeV==m)&(obs.policy==p)&(obs.scope=='combined')].iloc[0]
            rp=ranks(cal,p,m,float(observed.psi_hat));null=cal[(cal.mass_MeV==m)&(cal.policy==p)&(cal.z==0)]
            k=int(np.count_nonzero(null.signed_root.astype(float)>=float(observed.signed_root)))
            ci=(0. if k==0 else float(beta.ppf(.025,k,N-k+1)),1. if k==N else float(beta.ppf(.975,k+1,N-k)))
            env=envelope(rp,s0)
            rows.append(dict(mass_MeV=m,policy=p,p0_rank=(1+k)/101.,null_exceedances=k,p0_cp95_low=ci[0],p0_cp95_high=ci[1],
                psi_hat=float(observed.psi_hat),raw_psi90=float(observed.psi90),**env,epsilon2_90_grid=env['upper_psi']*1e-8,
                calibrated_scope='fixed-source native-anchor discrete rank upper envelope',grid_rank_p=json.dumps(rp.tolist())))
            for _,r in ev[(ev.mass_MeV==m)&(ev.policy==p)].iterrows():
                q=ranks(cal,p,m,float(r.psi_hat));e=envelope(q,s0)
                tests.append(dict(mass_MeV=m,policy=p,toy=int(r.toy),z=int(r.z),psi_expected=float(r.psi_expected),
                    p_at_truth=float(q[GRID.index(int(r.z))]),truth_accepted=bool(q[GRID.index(int(r.z))]>.1),upper_covers=bool(e['upper_psi']>=float(r.psi_expected)),**e))
    csv(B/'results/combined_observed_rank.csv',rows);csv(B/'results/combined_evaluation_rank.csv',tests)
    t=pd.DataFrame(tests)
    for keys,g in t.groupby(['mass_MeV','policy','z']):
        row=dict(zip(['mass_MeV','policy','z'],map(lambda v:v.item() if hasattr(v,'item') else v,keys)))
        for field in ('truth_accepted','upper_covers','empty','right_censored','holes'):
            k=int(g[field].sum());row[field+'_count']=k;row[field+'_fraction']=k/N
            row[field+'_cp95_low']=0. if not k else float(beta.ppf(.025,k,N-k+1))
            row[field+'_cp95_high']=1. if k==N else float(beta.ppf(.975,k+1,N-k))
        summaries.append(row)
    csv(B/'results/combined_evaluation_summary.csv',summaries)
    peak=[]
    for (scope,p),g in obs.groupby(['scope','policy']):
        r=g.loc[g.p0_asymptotic.idxmin()];peak.append(dict(scope=scope,policy=p,mass_MeV=int(r.mass_MeV),p0_asymptotic=float(r.p0_asymptotic),signed_root=float(r.signed_root)))
    writej(B/'results/observed_summary.json',dict(dense_rows=len(obs),pilot_rows=2000,calibration_rows=len(cal),evaluation_rows=len(ev),peaks=peak,
        native_anchor_rank_rows=len(rows),minimum_rank_p=1/101,zero_exceedance_one_sided95_upper=float(1-.05**(1/100)),
        fixed_table_rejection_beta95=beta.ppf([.025,.975],10,91).tolist(),
        no_global_or_dense_empirical_calibration=True,no_affine_limit_rescaling=True))
    writej(B/'qa/observed_completion.json',dict(passed=True,completed_utc=utc(),signature=signature(),dense_rows=len(obs),joint_rows=2000+len(cal)+len(ev),
        numerical_not_scientific_gate=True,outputs={p.name:sha(p) for p in [B/'results/observed_dense.csv',B/'results/combined_observed_rank.csv',B/'results/combined_evaluation_rank.csv',B/'results/combined_evaluation_summary.csv']}))

def run_jobs(fn,jobs,workers):
    if workers==1:
        for j in jobs:fn(j)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:list(pool.map(fn,jobs))
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','dense','pilot','freeze','calibration','freeze-calibration','evaluation','summary','all'],default='all');p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    stages=['prepare','dense','pilot','freeze','calibration','freeze-calibration','evaluation','summary'] if args.stage=='all' else [args.stage]
    for s in stages:
        print(s,utc(),flush=True)
        if s=='prepare':prepare()
        elif s=='dense':
            run_jobs(dense_one,range(60,241),args.workers)
            csv(B/'results/observed_dense.csv',pd.concat([readcsv(B/f'results/observed_chunks/m{m:03d}.csv') for m in range(60,241)],ignore_index=True))
        elif s in ('pilot','calibration','evaluation'):run_jobs(joint_one,[(s,m) for m in MASSES],args.workers);aggregate(s)
        elif s=='freeze':freeze_reference()
        elif s=='freeze-calibration':freeze_calibration()
        else:summarize()
if __name__=='__main__':main()
