"""Independent pointwise Poisson tests for the actual historical 2016 10% input.

The inherited six anchors are diagnostics, not a pre-data discovery search.
Whole-support counts are retrained at each fixed mass, using frozen kernels.
"""
from pathlib import Path
import os,sys,json,hashlib,time,datetime
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'engine'))
from common import DATA,moving_context,OneSignalProfile,predict
import numpy as np,pandas as pd,uproot
from scipy.stats import beta,norm

N=512;ANCHORS=[42,66,76,90,92,117];SEED=580201610
DEADLINE=datetime.datetime(2026,9,17,17,43,tzinfo=datetime.timezone.utc).timestamp()
TRUTHS=['scaled_parent_stress','mass_specific_local_gp']
(B/'local_checkpoints').mkdir(exist_ok=True)
source=B/'inputs/source_2016_10pct.root'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(source)=='789e619fcbeb5e81f9193d3e224bc17919983477a037bf3d79692327555f9fd4'
with uproot.open(source) as f:y,edges=f['h_Minv_General_Final_1'].to_numpy()
old=DATA['2016'];new=dict(old)
idx=np.array([np.argmin(abs(edges-x)) for x in old['edges']])
assert np.max(abs(edges[idx]-old['edges']))<1e-12
new.update(n=np.diff(np.r_[0.,np.cumsum(y)][idx]),native_counts=y.copy(),native_edges=edges.copy())
fraction=float(new['n'].sum()/old['n'].sum());DATA['2016']=new
rows=[];failures=[];started=time.monotonic()

def fit_root(mass,counts):
    p=moving_context('2016',float(mass),counts=counts)
    mod=OneSignalProfile(p['b'],p['L'],p['S'][:,0]);f=mod.fit(p['n']);z=mod.fit(p['n'],0.)
    q=2*(z['nll']-f['nll']);score=max(f['score'],z['score'])
    if q < -1e-8 or score>2e-7:raise RuntimeError(f'q={q}, score={score}')
    root=float(np.sign(f['A'])*np.sqrt(max(q,0.)))
    return p,root,float(score),float(p['diagnostic']['load'])

def interval(k,n):
    return (0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k)))

def summarize(path,mass,truthid,truth,observed,a):
    cache=np.load(path);roots=cache['roots'];finite=np.isfinite(roots);r=roots[finite];n=len(r)
    if not n:return
    qobs=max(0.,observed)**2;qtoy=np.maximum(0.,r)**2
    k=int(np.sum(qtoy>=qobs));lo,hi=interval(k,n)
    rawk=int(np.sum(r>norm.isf(.05)));rawlo,rawhi=interval(rawk,n)
    rows.append(dict(mass_MeV=mass,truth=truth,truth_id=truthid,seed_base=SEED,n_generated=len(roots),n_finite=n,
        observed_signed_root=observed,observed_q0=qobs,nominal_asymptotic_local_p=float(norm.sf(max(0.,observed))),
        q0_tail_count=k,empirical_q0_inclusive_p=k/n,empirical_q0_CI95_low=lo,empirical_q0_CI95_high=hi,
        q0_boundary_exact_p_one=bool(qobs==0.),deterministic_root=a,toy_root_mean=float(np.mean(r)),
        toy_root_sd=float(np.std(r,ddof=1)),toy_mean_minus_deterministic=float(np.mean(r)-a),
        mean_se=float(np.std(r,ddof=1)/np.sqrt(n)),raw_nominal05_rejection_count=rawk,raw_nominal05_rejection_rate=rawk/n,
        raw_nominal05_CI95_low=rawlo,raw_nominal05_CI95_high=rawhi,
        max_optimizer_score=float(np.nanmax(cache['scores'])),max_covariance_load=float(np.nanmax(cache['loads'])),
        checkpoint=str(path.relative_to(B)),conditional_only=True,full_scan_calibrated=False))

for mass in ANCHORS:
    p,observed,_,_=fit_root(mass,new['n'])
    local,_=predict(new['x'],new['n'],p['mask'],p['const'],p['ls'],query=new['x'])
    for truthid,truth in enumerate(TRUTHS):
        mean=fraction*old['stress'] if truthid==0 else local
        _,a,_,_=fit_root(mass,mean)
        path=B/'local_checkpoints'/f'm{mass:03d}_{truth}.npz'
        rng=np.random.default_rng(np.random.SeedSequence([SEED,mass,truthid]))
        counts=rng.poisson(mean,size=(N,len(mean))).astype(np.int32)
        roots=np.full(N,np.nan);scores=np.full(N,np.nan);loads=np.full(N,np.nan);done=0
        if path.exists():
            saved=np.load(path)
            assert np.array_equal(saved['counts'],counts) and np.allclose(saved['truth'],mean,rtol=0,atol=0)
            done=int(saved['completed']);roots[:done]=saved['roots'][:done];scores[:done]=saved['scores'][:done];loads[:done]=saved['loads'][:done]
        for i in range(done,N):
            if time.time()>DEADLINE:break
            try:_,roots[i],scores[i],loads[i]=fit_root(mass,counts[i])
            except Exception as exc:failures.append(dict(mass_MeV=mass,truth=truth,toy_id=i,error=repr(exc)))
            done=i+1
            if done%128==0 or done==N:
                np.savez_compressed(path,counts=counts,truth=mean,roots=roots,scores=scores,loads=loads,
                    completed=done,seed_sequence=np.array([SEED,mass,truthid]),observed_root=observed,deterministic_root=a)
        np.savez_compressed(path,counts=counts,truth=mean,roots=roots,scores=scores,loads=loads,
            completed=done,seed_sequence=np.array([SEED,mass,truthid]),observed_root=observed,deterministic_root=a)
        summarize(path,mass,truthid,truth,observed,a)
        pd.DataFrame(rows).to_csv(B/'actual10_local_tests.csv',index=False,float_format='%.17g')
        print(f'mass={mass}, truth={truth}, completed={done}/{N}, elapsed={time.monotonic()-started:.1f}s, failures={len(failures)}',flush=True)

DATA['2016']=old
frame=pd.DataFrame(rows)
manifest=dict(complete=len(rows)==12 and int(frame.n_finite.sum())==6144 and not failures,elapsed_seconds=time.monotonic()-started,
    N=N,anchors_MeV=ANCHORS,seed_base=SEED,seed_recipe='numpy SeedSequence [580201610, mass_MeV, truth_id]; default_rng; one whole-support Poisson spectrum per trial',
    truth_ids=dict(enumerate(TRUTHS)),support_count_ratio=fraction,input_sha256=sha(source),script_sha256=sha(__file__),
    local_tail_ordering='inclusive q0=max(0,r)^2 >= observed q0; exact p=1 if observed q0=0',
    binomial_intervals='two-sided 95% Clopper-Pearson; at qobs=0 true inclusive tail is exactly 1 by statistic support',
    fitted_policy='fixed reviewed parent mass-specific kernel and inherited signal resolution; sideband log-counts and count-dependent errors updated per whole-spectrum toy',
    provenance_caveats='historical10 selection/exposure/parent overlap unverified; stress shape scaled from parent; each localGP truth is mass-specific and source-conditioned',
    failures=failures,no_global_calibration=True)
(B/'actual10_local_tests_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
print(frame[['mass_MeV','truth','observed_signed_root','deterministic_root','toy_root_mean','toy_root_sd','q0_tail_count','empirical_q0_inclusive_p','raw_nominal05_rejection_rate']].to_string(index=False))
