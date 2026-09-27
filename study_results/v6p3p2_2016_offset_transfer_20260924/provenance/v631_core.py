"""Independent-pilot, fixed-expected-yield conditional injection experiments."""
from pathlib import Path
import os, sys, json, hashlib, time, datetime
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/hps-v631-mpl')
sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B/'inputs/v6p1/scripts'))
import common as C
import core_centering as MC
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.linalg import cholesky, cho_solve, solve_triangular

MASTER = 63220260924
MASSES = tuple(range(60, 241, 20))
SHAPES = ('gaussian', 'mc')
LEVELS = (0, 1, 3, 5)
NTOYS = 100
D = C.DATA['2021']
NULL = dict(np.load(B/'inputs/null_2021.npz'))
TRUTH = NULL['truth']
assert np.array_equal(D['edges'], NULL['edges_GeV'])
assert np.array_equal(D['n'], NULL['observed'])
assert np.all(TRUTH > 0)

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def seed_key(namespace, mass=0, toy=0, shape_id=0, level_id=0):
    return [MASTER, namespace, int(mass), int(toy), int(shape_id), int(level_id)]

def rng(key):
    return np.random.default_rng(np.random.SeedSequence(key))

def signature():
    rows = json.loads((B/'provenance/input_hashes.json').read_text())
    for row in rows:
        if sha(B/row['path']) != row['sha256']:
            raise RuntimeError('Pinned dependency changed: '+row['path'])
    parts = [(row['path'], row['sha256']) for row in rows]
    for path in (B/'scripts/core.py', B/'scripts/run_study.py', B/'protocol.json', B/'inputs/cohorts.npz', B/'inputs/templates.npz'):
        parts.append((str(path.relative_to(B)), sha(path)))
    return hashlib.sha256(json.dumps(sorted(parts)).encode()).hexdigest()

def template(m, shape):
    if shape == 'gaussian':
        cdf = ndtr((D['edges']-m/1000.)/C.sigma('2021', m))
    else:
        a = MC.MC[m]
        assert np.array_equal(a['sumw'], a['sumw2'])
        assert json.loads(str(a['metadata']))['stats'].get('underflow', 0) == 0
        cdf = MC.cumulative(m, D['edges'])
    categories = np.r_[cdf[0], np.diff(cdf), 1-cdf[-1]]
    assert np.all(np.isfinite(categories)) and categories.min() >= 0
    assert abs(categories.sum()-1) < 1e-12
    return categories

class Context:
    def __init__(self, m):
        self.m = m
        self.sigma = C.sigma('2021', m)
        self.lo, self.hi = m/1000.-2.25*self.sigma, m/1000.+2.25*self.sigma
        x = D['x']
        self.mask = (x >= self.lo) & (x <= self.hi)
        assert np.sum(x < self.lo) >= 3 and np.sum(x > self.hi) >= 3
        self.const, self.ls = C.kernel_state('2021', m)
        self.K = C.kernel(x[~self.mask], x[~self.mask], self.const, self.ls)
        self.Kqt = C.kernel(x[self.mask], x[~self.mask], self.const, self.ls)
        self.Kqq = C.kernel(x[self.mask], x[self.mask], self.const, self.ls)
        self.categories = {shape:template(m, shape) for shape in SHAPES}

    def predict(self, counts):
        n = np.asarray(counts, float)[~self.mask]
        pos = n > 0
        target = np.zeros_like(n); target[pos] = np.log(n[pos])
        alpha = np.ones_like(n); alpha[pos] = 1/n[pos]
        K = self.K.copy(); K.flat[::len(K)+1] += alpha
        L = cholesky(K, lower=True, check_finite=False)
        mu = self.Kqt @ cho_solve((L, True), target, check_finite=False)
        v = solve_triangular(L, self.Kqt.T, lower=True, check_finite=False)
        cov = self.Kqq-v.T@v; cov = .5*(cov+cov.T)
        b = np.exp(mu+.5*np.maximum(np.diag(cov), 0))
        covariance = np.outer(b,b)*np.expm1(np.clip(cov,-40,40))
        factor, diagnostic = C.factor_cov(covariance,b)
        assert np.all(np.isfinite(b)) and np.all(np.isfinite(factor))
        return b, factor, diagnostic

def checked_fit(model, n, fixed=None, initial=None):
    fit = model.fit(n, fixed=fixed, initial=initial)
    assert np.isfinite(fit['nll']) and np.isfinite(fit['score'])
    assert fit['score'] < 3e-5 and fit['min_lambda'] > 0
    if fixed is None:
        # Reject the inherited solver's unmarked Fisher-error fallback.
        _, _, H, _ = model._objective(fit['z'], n, model.Jfree, model.b, model.penfree)
        unit = np.zeros(len(H)); unit[0] = 1
        variance = float(cho_solve((cholesky(H,lower=True),True), unit)[0])
        assert variance > 0 and np.isfinite(variance)
        sigma = float(model.scale*np.sqrt(variance))
        assert sigma > 0 and np.isfinite(sigma)
        assert abs(fit['sigma']/sigma-1) < 1e-10
        fit['sigma'] = sigma
    return fit

def fit_pair(b, L, p, n, expected, need_profile):
    """At most three deterministic attempts; no outcome-based selection."""
    attempts, frees, profiles = [], [], []
    for attempt in range(3):
        tolerance = 2e-7 if attempt == 0 else 2e-9
        model = C.OneSignalProfile(b, L, p, score_tolerance=tolerance)
        initial = None if attempt < 2 else np.r_[.5, np.zeros(model.rank)]
        # A profile-based start only restores numerical likelihood nesting.
        if attempt > 0 and profiles:
            f = min(profiles,key=lambda q:q['nll'])
            initial = np.r_[expected/model.scale, f['theta']]
        record = dict(attempt=attempt, score_tolerance=tolerance,
                      free_start='origin' if initial is None else 'deterministic_alternative')
        free = None
        try:
            free = checked_fit(model,n,initial=initial)
            frees.append(free)
            record.update(free_valid=True, free_nll=free['nll'], free_score=free['score'],
                          sigma_observed=free['sigma'], free_method=free['method'])
        except Exception as e:
            record.update(free_valid=False, free_error=type(e).__name__+': '+str(e))
        if need_profile and free is not None:
            try:
                initial_theta = free['theta'] if attempt != 1 else np.zeros(model.rank)
                fixed = checked_fit(model,n,fixed=float(expected),initial=initial_theta)
                profiles.append(fixed)
                record.update(profile_valid=True, true_nll=fixed['nll'], profile_score=fixed['score'],
                              profile_method=fixed['method'])
            except Exception as e:
                record.update(profile_valid=False, profile_error=type(e).__name__+': '+str(e))
        attempts.append(record)
        if frees and (not need_profile or (profiles and 2*(min(p['nll'] for p in profiles)-min(f['nll'] for f in frees)) >= -2e-6)):
            break
    free = min(frees,key=lambda q:q['nll']) if frees else None
    fixed = min(profiles,key=lambda q:q['nll']) if profiles else None
    valid_profile = bool(free is not None and fixed is not None and 2*(fixed['nll']-free['nll']) >= -2e-6)
    return free, fixed, valid_profile, attempts

def make_row(ctx, background, draw, expected, toy, shape, z, cohort, s0=None, prediction=None, control='primary'):
    p = ctx.categories[shape][1:-1]
    signal = draw[1:-1]
    counts = background+signal
    need_profile = cohort != 'pilot'
    row = dict(mass_MeV=ctx.m, shape=shape, z=z, toy=toy, cohort=cohort, control=control,
               row_id=f'{cohort}:m{ctx.m}:{shape}:z{z}:t{toy}',
               A_expected=float(expected),s0=s0,
               actual_full=int(draw.sum()),actual_support=int(signal.sum()),actual_window=int(signal[ctx.mask].sum()),
               actual_training=int(signal[~ctx.mask].sum()),actual_outside_support=int(draw[0]+draw[-1]),
               actual_below_support=int(draw[0]),actual_above_support=int(draw[-1]),
               support_fraction=float(p.sum()),window_fraction=float(p[ctx.mask].sum()),training_fraction=float(p[~ctx.mask].sum()),
               background_hash=array_hash(background),counts_hash=array_hash(counts),template_hash=array_hash(ctx.categories[shape]),
               signal_draw_hash=array_hash(draw),mask_hash=array_hash(ctx.mask),
               background_seed_key=json.dumps(seed_key(1 if cohort=='pilot' else 2,toy=toy)),
               signal_seed_key=json.dumps(seed_key(3,ctx.m,toy,SHAPES.index(shape)+1,LEVELS.index(z))) if z else '',
               fit_valid=False,profile_valid=False,failure_reason='',sigma_method='observed_profile_hessian',
               fit_bins=int(ctx.mask.sum()),kernel_const=ctx.const,kernel_ls=ctx.ls,
               nominal_sigma_MeV=ctx.sigma*1000,window_low_MeV=ctx.lo*1000,window_high_MeV=ctx.hi*1000)
    try:
        if control == 'known_background':
            b,L,diag=TRUTH[ctx.mask],np.zeros((ctx.mask.sum(),0)),dict(rank=0,load=0.,max_omitted=0.)
        else:
            b,L,diag=ctx.predict(counts) if prediction is None else prediction
        row.update(nuisance_rank=int(diag['rank']),covariance_load=float(diag['load']),max_omitted_covariance_mode=float(diag['max_omitted']),
                   gp_mean_hash=array_hash(b),gp_covariance_factor_hash=array_hash(L))
        free,fixed,pvalid,attempts=fit_pair(b,L,p[ctx.mask],counts[ctx.mask],expected,need_profile)
        row['attempts_json']=json.dumps(attempts,allow_nan=False)
        row['attempt_count']=len(attempts)
        row['fit_valid']=free is not None
        row['profile_valid']=pvalid
        if free is not None:
            row.update(Ahat=free['A'],sigma_postfit=free['sigma'],pull=(free['A']-expected)/free['sigma'],
                       free_nll=free['nll'],fit_score=free['score'],min_lambda=free['min_lambda'],
                       min_fitted_background=free['min_background'],fit_method=free['method'],fit_iterations=free['iterations'])
        else:
            row['failure_reason']='free_fit_failed; see attempts_json'
        if fixed is not None:
            row.update(true_nll=fixed['nll'],profile_score=fixed['score'],profile_min_lambda=fixed['min_lambda'],
                       profile_method=fixed['method'])
        if pvalid:
            qraw=2*(fixed['nll']-free['nll']);q=max(0.,qraw)
            row.update(q_true_raw=qraw,q_true=q,profile_contains68=bool(q<=1.),profile_contains95=bool(q<=3.841459))
        elif need_profile:
            row['failure_reason']+='; truth_profile_or_likelihood_ordering_failed'
    except Exception as e:
        row['failure_reason']=type(e).__name__+': '+str(e)
    return row

def signal_draw(ctx,shape,z,toy,expected,namespace=3):
    if z==0:
        return np.zeros(len(TRUTH)+2,dtype=np.int64)
    key=seed_key(namespace,ctx.m,toy,SHAPES.index(shape)+1,LEVELS.index(z))
    # Independent category Poissons exactly implement the fluctuating-rate experiment.
    return rng(key).poisson(expected*ctx.categories[shape])

def chunk_path(cohort,m,start):
    return B/f'results/{cohort}/m{m:03d}_t{start:03d}'

def run_chunk(cohort,m,start,stop,expected_signature,reference_sha):
    if signature()!=expected_signature:
        raise RuntimeError('Dependency signature changed')
    if cohort=='evaluation' and sha(B/'pilot_reference.json')!=reference_sha:
        raise RuntimeError('Frozen pilot reference changed')
    base=chunk_path(cohort,m,start);marker=base.with_suffix('.json')
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['signature']!=expected_signature or old['reference_sha256']!=reference_sha:
            raise RuntimeError('Checkpoint signature mismatch: '+str(marker))
        if old['complete'] and all(sha(B/name)==digest for name,digest in old['output_hashes'].items()):
            return dict(cohort=cohort,mass=m,start=start,cached=True,rows=old['rows'])
        raise RuntimeError('Checkpoint checksum failure: '+str(marker))
    ctx=Context(m)
    cohorts=np.load(B/'inputs/cohorts.npz')
    backgrounds=cohorts[cohort]
    reference=json.loads((B/'pilot_reference.json').read_text()) if cohort=='evaluation' else None
    s0=reference['masses'][str(m)]['s0'] if reference else None
    rows=[];draws=[];draw_keys=[]
    begin=time.monotonic()
    started=utc()
    for toy in range(start,stop):
        if (B/'STOP').exists():
            raise RuntimeError('STOP requested; completed chunks preserved')
        background=backgrounds[toy]
        prediction=ctx.predict(background)
        for shape in SHAPES:
            for z in ((0,) if cohort=='pilot' else LEVELS):
                expected=0. if z==0 else float(z*s0)
                draw=signal_draw(ctx,shape,z,toy,expected)
                rows.append(make_row(ctx,background,draw,expected,toy,shape,z,cohort,s0,
                                     prediction=prediction if z==0 else None))
                if cohort=='evaluation':
                    draws.append(draw);draw_keys.append((toy,SHAPES.index(shape)+1,z))
    csv=base.with_suffix('.csv');tmp=base.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(tmp,index=False,float_format='%.17g');tmp.replace(csv)
    outputs=[csv]
    if cohort=='evaluation':
        path=base.with_suffix('.npz')
        np.savez_compressed(path,draws=np.array(draws),keys=np.array(draw_keys),mask=ctx.mask)
        outputs.append(path)
    write_json(marker,dict(complete=True,cohort=cohort,mass_MeV=m,start=start,stop=stop,rows=len(rows),
                          started_utc=started,completed_utc=utc(),seconds=time.monotonic()-begin,
                          signature=expected_signature,reference_sha256=reference_sha,
                          output_hashes={str(p.relative_to(B)):sha(p) for p in outputs}))
    return dict(cohort=cohort,mass=m,start=start,cached=False,rows=len(rows),seconds=round(time.monotonic()-begin,3))
