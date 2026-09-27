"""v6.2 exact-N native-MC injection and paired extraction, in candidate counts."""
from pathlib import Path
import os, sys, json, hashlib, time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hps-v62-mpl')
sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B/'inputs/v6p1/scripts'))
import common as C
import core_centering as MC
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.stats import chi2

D = C.DATA['2021']
NULL = dict(np.load(B/'inputs/null_2021.npz'))
TRUTH = NULL['truth']
SEED = 62220260923
LEVELS = (1000, 5000, 10000, 30000)
MASSES = tuple(range(60,261,20))
METHODS = ('pole_centered', 'core_shifted')
TOYS = 40
assert np.array_equal(D['edges'], NULL['edges_GeV'])
assert np.array_equal(D['n'], NULL['observed'])
assert np.all(TRUTH > 0)

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def dependency_signature():
    paths=[B/'scripts/injection_core.py',B/'scripts/run_study.py']
    paths += [B/r['bundled'] for r in json.loads((B/'provenance/input_hashes.json').read_text())]
    return hashlib.sha256(''.join(str(p.relative_to(B))+hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)).encode()).hexdigest()

def source(m):
    """Probabilities remain normalized to every selected MC candidate."""
    a=MC.MC[m]
    meta=json.loads(str(a['metadata']))
    assert np.array_equal(a['sumw'],a['sumw2']), 'Unit-weight MC required'
    cdf=MC.cumulative(m,D['edges'])
    p=np.diff(cdf)
    # Retain both outside-support categories, including recorded histogram overflow.
    categories=np.r_[cdf[0],p,1-cdf[-1]]
    assert np.min(categories)>=0 and abs(categories.sum()-1)<1e-12
    center, _=MC.locate(m)
    assert center['core_location_valid']
    if m<=240:
        old=pd.read_csv(B/'inputs/v6p1/derived/core_centers.csv')
        old_c=float(old.loc[old.mass_MeV==m,'core_center_MeV'].iloc[0])
        assert abs(center['core_center_MeV']-old_c)<1e-6
    return p, categories, center, meta

class Context:
    """Cached kernel geometry; noise, targets, mean and covariance update per toy."""
    def __init__(self,m,method,center,p):
        self.m=m;self.method=method;self.center=center
        self.window_center=m if method=='pole_centered' else center['core_center_MeV']
        self.sigma=C.sigma('2021',m)
        self.lo=self.window_center/1000-2.25*self.sigma
        self.hi=self.window_center/1000+2.25*self.sigma
        x=D['x'];self.mask=(x>=self.lo)&(x<=self.hi)
        assert np.sum(x<self.lo)>=3 and np.sum(x>self.hi)>=3
        self.anchor=250 if m==260 else m
        self.const,self.ls=C.kernel_state('2021',self.anchor)
        self.K=C.kernel(x[~self.mask],x[~self.mask],self.const,self.ls)
        self.Kqt=C.kernel(x[self.mask],x[~self.mask],self.const,self.ls)
        self.Kqq=C.kernel(x[self.mask],x[self.mask],self.const,self.ls)
        self.p=p;self.S=p[self.mask]

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
        factor,diagnostic=C.factor_cov(covariance,b)
        return b,factor,diagnostic

    def row(self,counts,background,signal,N,toy,control,prediction=None):
        if control=='known_background':
            b=TRUTH[self.mask];L=np.zeros((self.mask.sum(),0))
            diag={'load':0.,'rank':0,'max_omitted':0.}
        else:
            b,L,diag=self.predict(counts) if prediction is None else prediction
        model=C.OneSignalProfile(b,L,self.S)
        n=counts[self.mask]
        fit=model.fit(n)
        at_truth=model.fit(n,fixed=float(N),initial=fit['theta'])
        q=2*(at_truth['nll']-fit['nll'])
        assert q>=-2e-6 and fit['sigma']>0
        assert max(fit['score'],at_truth['score'])<3e-5
        assert min(fit['min_lambda'],at_truth['min_lambda'])>0
        q=max(0.,q)
        row=dict(mass_MeV=self.m,injected_N=N,toy=toy,method=self.method,control=control,
            Ahat=fit['A'],sigma_A=fit['sigma'],pull=(fit['A']-N)/fit['sigma'],q_true=q,
            profile_contains68=bool(q<=1.),profile_contains95=bool(q<=chi2.ppf(.95,1)),
            wald_contains68=bool(abs(fit['A']-N)<=fit['sigma']),
            wald_contains95=bool(abs(fit['A']-N)<=1.959963984540054*fit['sigma']),
            wald68_low=fit['A']-fit['sigma'],wald68_high=fit['A']+fit['sigma'],
            wald95_low=fit['A']-1.959963984540054*fit['sigma'],
            wald95_high=fit['A']+1.959963984540054*fit['sigma'],
            actual_support=float(signal.sum()),actual_window=float(signal[self.mask].sum()),
            actual_outside_support=float(N-signal.sum()),actual_training=float(signal[~self.mask].sum()),
            expected_support_fraction=float(self.p.sum()),expected_window_fraction=float(self.S.sum()),
            fitted_support_yield=fit['A']*self.p.sum(),fitted_window_yield=fit['A']*self.S.sum(),
            fit_score=max(fit['score'],at_truth['score']),min_lambda=min(fit['min_lambda'],at_truth['min_lambda']),
            min_fitted_background=fit['min_background'],nuisance_rank=diag['rank'],covariance_load=diag['load'],
            max_omitted_covariance_mode=diag['max_omitted'],free_nll=fit['nll'],true_nll=at_truth['nll'],
            fit_method=fit['method'],fit_iterations=fit['iterations'],
            core_center_MeV=self.center['core_center_MeV'],window_center_MeV=self.window_center,
            nominal_sigma_MeV=self.sigma*1000,window_low_MeV=self.lo*1000,window_high_MeV=self.hi*1000,
            fit_bins=int(self.mask.sum()),kernel_anchor_MeV=self.anchor,extension_260=bool(self.m==260),
            background_draw_hash=array_hash(background),injected_counts_hash=array_hash(counts))
        return row

def run_mass(m,ntoys=TOYS,pilot=False):
    start=time.monotonic();target=B/('qa/pilot' if pilot else 'results/checkpoints')
    target.mkdir(parents=True,exist_ok=True)
    marker=target/f'm{m:03d}.json'
    signature=dependency_signature()
    if marker.exists():
        old=json.loads(marker.read_text())
        valid_outputs=all((target/name).exists() and hashlib.sha256((target/name).read_bytes()).hexdigest()==digest
            for name,digest in old.get('output_hashes',{}).items())
        if (old.get('toys')==ntoys and old.get('complete') and old.get('dependency_signature')==signature
                and len(old.get('output_hashes',{}))==3 and valid_outputs):
            return dict(mass=m,cached=True,seconds=old['seconds'])
    p,categories,center,meta=source(m)
    contexts={method:Context(m,method,center,p) for method in METHODS}
    rows=[];backgrounds=[];injections=[];asimov=[]
    for toy in range(ntoys):
        if (B/'STOP').exists():raise RuntimeError('Study STOP marker found')
        rng=np.random.default_rng(np.random.SeedSequence([SEED,m,toy,0]))
        background=rng.poisson(TRUTH);backgrounds.append(background)
        clean={method:ctx.predict(background) for method,ctx in contexts.items()}
        for method,ctx in contexts.items():
            for control in ('contaminated_gp','known_background'):
                rows.append(ctx.row(background,background,np.zeros_like(background),0,toy,control,
                    clean[method] if control=='contaminated_gp' else None))
        signals=[]
        for N in LEVELS:
            rng=np.random.default_rng(np.random.SeedSequence([SEED,m,toy,N]))
            draw=rng.multinomial(N,categories)
            assert int(draw.sum())==N
            signal=draw[1:-1];signals.append(draw)
            counts=background+signal
            for method,ctx in contexts.items():
                for control in ('contaminated_gp','clean_sidebands','known_background'):
                    rows.append(ctx.row(counts,background,signal,N,toy,control,
                        clean[method] if control=='clean_sidebands' else None))
        injections.append(signals)
    for method,ctx in contexts.items():
        clean=ctx.predict(TRUTH)
        for N in (0,*LEVELS):
            for control in ('contaminated_gp','clean_sidebands','known_background'):
                if N==0 and control=='clean_sidebands':continue
                asimov.append(ctx.row(TRUTH+N*p,TRUTH,N*p,N,-1,control,
                    clean if control=='clean_sidebands' else None))
    frame=pd.DataFrame(rows);expected=ntoys*(4+4*6)
    assert len(frame)==expected
    for name,data in [('toys',frame),('asimov',pd.DataFrame(asimov))]:
        tmp=target/f'm{m:03d}_{name}.csv.tmp'
        data.to_csv(tmp,index=False,float_format='%.17g');tmp.replace(target/f'm{m:03d}_{name}.csv')
    np.savez_compressed(target/f'm{m:03d}_draws.npz',backgrounds=np.array(backgrounds),
        injections=np.array(injections),categories=categories,probability=p,truth=TRUTH,
        levels=np.array(LEVELS),edges_GeV=D['edges'],
        pole_mask=contexts['pole_centered'].mask,core_mask=contexts['core_shifted'].mask)
    summary=dict(complete=True,mass_MeV=m,toys=ntoys,rows=len(frame),asimov_rows=len(asimov),
        seconds=time.monotonic()-start,center=center,source_selected_entries=meta['stats']['selected'],
        support_fraction=float(p.sum()),max_fit_score=float(frame.fit_score.max()),
        minimum_lambda=float(frame.min_lambda.min()))
    summary['dependency_signature']=signature
    summary['output_hashes']={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (target/f'm{m:03d}_toys.csv',target/f'm{m:03d}_asimov.csv',target/f'm{m:03d}_draws.npz')}
    write_json(marker,summary)
    return dict(mass=m,cached=False,seconds=summary['seconds'],rows=len(frame))
