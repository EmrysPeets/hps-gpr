"""MC-only exclusion guards, kept separate from the Gaussian likelihood window.

No observed counts enter window selection. Full selected MC normalization is
retained, including reported overflow. The deterministic Asimov comparisons
are diagnostics, not toy coverage or a window-selection criterion.
"""
from pathlib import Path
import os, sys, json, hashlib, datetime
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
REPO = B.parents[1]
PIN = B/'inputs/v6p1'
if not (PIN/'scripts/common.py').exists():
    PIN = REPO/'study_results/v6p3p1_fixed_yield_2021_100toy_20260924/inputs/v6p1'
sys.path.insert(0, str(PIN/'scripts'))
import common as C
import core_centering as MC
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.linalg import cholesky, cho_solve, solve_triangular

MASSES = tuple(range(60, 261, 20))
POLICIES = ('baseline', 'equal95', 'shortest95')
D = C.DATA['2021']
LOG_COEFFICIENTS = (-3.2243308692909953, -2.213992811446465)
NULL_PATH = B/'inputs/null_2021.npz'
if not NULL_PATH.exists():
    NULL_PATH = REPO/'study_results/v6p3p1_fixed_yield_2021_100toy_20260924/inputs/null_2021.npz'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ahash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def center(m):
    if 60 <= m <= 240:
        return m + LOG_COEFFICIENTS[0] + LOG_COEFFICIENTS[1]*np.log(m/150.)
    return float(MC.locate(m)[0]['core_center_MeV'])

def native(m):
    h = MC.MC[m]
    meta = json.loads(str(h['metadata']))
    assert np.array_equal(h['sumw'], h['sumw2'])
    assert meta['stats']['underflow'] == 0
    total = float(meta['sumw'])
    assert h['sumw'].sum()+meta['stats']['overflow'] == total
    edges = h['edges_GeV']*1000
    cdf = np.r_[0., np.cumsum(h['sumw'])]/total
    return h, meta, edges, cdf

def inverse(q, edges, cdf):
    # Piecewise-uniform native histogram; retain the full selected denominator.
    return np.interp(q, cdf, edges)

def interval_mask(lo, hi, fit=None):
    # Include every analysis bin intersecting the continuous MC interval.
    e = D['edges']*1000
    mask = (e[:-1] < hi) & (e[1:] > lo)
    if fit is not None:
        mask |= fit
    i = np.flatnonzero(mask)
    assert len(i)
    return mask, min(lo, float(e[i[0]])), max(hi, float(e[i[-1]+1]))

def geometry(m):
    h, meta, edges, cdf = native(m)
    c = center(m)
    sigma = C.sigma('2021', m)*1000
    fitted_core_sigma = float(MC.locate(m)[0]['fitted_core_sigma_MeV'])
    lo, hi = c-2.25*sigma, c+2.25*sigma
    fit = (D['x']*1000 >= lo) & (D['x']*1000 <= hi)
    i = np.flatnonzero(fit)
    fitlo, fithi = D['edges'][[i[0], i[-1]+1]]*1000
    qlo, qhi = inverse(np.array([.025,.975]), edges, cdf)
    # The minimum of a piecewise-linear quantile-width function occurs at one
    # of its knots. This covers 95% of all selected MC within recorded bins.
    candidates = np.unique(np.r_[0., cdf, cdf-.95, cdf[-1]-.95])
    candidates = candidates[(candidates >= 0) & (candidates <= cdf[-1]-.95)]
    lower = inverse(candidates, edges, cdf)
    upper = inverse(candidates+.95, edges, cdf)
    j = int(np.argmin(upper-lower))
    intervals = {'baseline':(fitlo,fithi), 'equal95':(qlo,qhi),
                 'shortest95':(float(lower[j]),float(upper[j]))}
    p = np.diff(np.interp(D['edges']*1000, edges, cdf, left=0., right=cdf[-1]))
    rows, masks = [], {}
    for policy,(a,b) in intervals.items():
        if policy == 'baseline':
            guard,glo,ghi = fit.copy(),float(fitlo),float(fithi)
        else:
            guard,glo,ghi = interval_mask(a,b,fit)
        masks[policy] = guard
        left,right = int(np.sum(D['x']*1000 < glo)),int(np.sum(D['x']*1000 > ghi))
        fraction = float(np.interp(ghi,edges,cdf)-np.interp(glo,edges,cdf))
        const,ls = C.kernel_state('2021',m)
        rows.append(dict(mass_MeV=m,policy=policy,scope='primary' if m<=240 else 'outside_log_law_diagnostic',
            center_MeV=c,center_definition='logarithmic_MC_law' if m<=240 else 'direct_MC_core_diagnostic',
            nominal_sigma_MeV=sigma,fitted_core_sigma_MeV=fitted_core_sigma,
            fit_low_MeV=float(fitlo),fit_high_MeV=float(fithi),fit_bins=int(fit.sum()),
            raw_MC_interval_low_MeV=float(a),raw_MC_interval_high_MeV=float(b),
            guard_low_MeV=glo,guard_high_MeV=ghi,guard_bins=int(guard.sum()),
            guard_width_nominal_sigma=(ghi-glo)/sigma,guard_width_core_sigma=(ghi-glo)/fitted_core_sigma,
            guard_left_wig_nominal=(c-glo)/(2*sigma),guard_right_wig_nominal=(ghi-c)/(2*sigma),
            guard_left_wig_core=(c-glo)/(2*fitted_core_sigma),guard_right_wig_core=(ghi-c)/(2*fitted_core_sigma),
            full_selected_MC=int(meta['sumw']),histogram_overflow=int(meta['stats']['overflow']),
            histogram_coverage=float(cdf[-1]),support_fraction=float(p.sum()),
            MC_fraction_in_fit=float(p[fit].sum()),MC_fraction_in_guard=fraction,
            MC_fraction_training=float(p[~guard].sum()),MC_fraction_outside_support=float(1-p.sum()),
            left_training_bins=left,right_training_bins=right,feasible=bool(left>=3 and right>=3),
            kernel_const=const,kernel_ls=ls,kernel_anchor_MeV=min(m,250),
            fit_mask_sha256=ahash(fit),guard_mask_sha256=ahash(guard)))
    return rows,masks,fit,p

def asimov_fit(m, truth, guard, fit, pgauss):
    const,ls = C.kernel_state('2021',m)
    b,cov = C.predict(D['x'],truth,guard,const,ls,query=D['x'][fit])
    L,diag = C.factor_cov(cov,b)
    model = C.OneSignalProfile(b,L,pgauss[fit])
    r = model.fit(truth[fit])
    assert r['score'] < 3e-5 and r['min_lambda'] > 0 and r['sigma'] > 0
    return r,diag

class ToyContext:
    def __init__(self,m,policy):
        self.m,self.policy = m,policy
        c,s = center(m)/1000,C.sigma('2021',m)
        widths = {'baseline':(2.25,2.25),'tight':(2,2),'wideleft':(2.5,2),'wideright':(2,2.5)}
        if policy in widths:
            left,right = widths[policy]
            self.fit = (D['x'] >= c-left*s) & (D['x'] <= c+right*s)
            self.guard = self.fit.copy()
        else:
            _,g,self.fit,_ = geometry(m)
            self.guard = g[policy]
        self.pgauss = np.diff(ndtr((D['edges']-c)/s))
        _,_,e,cdf = native(m)
        supportcdf = np.interp(D['edges']*1000,e,cdf,left=0.,right=cdf[-1])
        self.categories = np.r_[supportcdf[0],np.diff(supportcdf),1-supportcdf[-1]]
        self.const,self.ls = C.kernel_state('2021',m)
        self.K = C.kernel(D['x'][~self.guard],D['x'][~self.guard],self.const,self.ls)
        self.Kqt = C.kernel(D['x'][self.fit],D['x'][~self.guard],self.const,self.ls)
        self.Kqq = C.kernel(D['x'][self.fit],D['x'][self.fit],self.const,self.ls)

    def prediction(self,counts):
        n = np.asarray(counts,float)[~self.guard]
        pos = n>0
        target = np.zeros_like(n);target[pos] = np.log(n[pos])
        alpha = np.ones_like(n);alpha[pos] = 1/n[pos]
        K = self.K.copy();K.flat[::len(K)+1] += alpha
        L = cholesky(K,lower=True,check_finite=False)
        mu = self.Kqt@cho_solve((L,True),target,check_finite=False)
        v = solve_triangular(L,self.Kqt.T,lower=True,check_finite=False)
        cov = self.Kqq-v.T@v;cov = .5*(cov+cov.T)
        b = np.exp(mu+.5*np.maximum(np.diag(cov),0))
        cov = np.outer(b,b)*np.expm1(np.clip(cov,-40,40))
        factor,diag = C.factor_cov(cov,b)
        return b,factor,diag

    def fit_counts(self,counts,known_truth=None):
        if known_truth is None:
            b,L,diag = self.prediction(counts)
        else:
            b = known_truth[self.fit]
            L,diag = np.zeros((len(b),0)),dict(load=0.,rank=0)
        errors = []
        for tolerance in (2e-7,2e-9):
            try:
                model = C.OneSignalProfile(b,L,self.pgauss[self.fit],score_tolerance=tolerance)
                r = model.fit(np.asarray(counts)[self.fit])
                assert r['score']<3e-5 and r['min_lambda']>0
                _,_,H,_ = model._objective(r['z'],np.asarray(counts)[self.fit],model.Jfree,model.b,model.penfree)
                unit = np.zeros(len(H));unit[0] = 1
                var = float(cho_solve((cholesky(H,lower=True),True),unit)[0])
                sigma = model.scale*np.sqrt(var)
                assert var>0 and abs(r['sigma']/sigma-1)<1e-9
                return dict(Ahat=r['A'],sigma_postfit=float(sigma),fit_valid=True,
                            failure_reason='',score=r['score'],min_lambda=r['min_lambda'],
                            covariance_load=diag['load'],nuisance_rank=diag['rank'])
            except Exception as exc:
                errors.append(type(exc).__name__+': '+str(exc))
        return dict(Ahat=None,sigma_postfit=None,fit_valid=False,failure_reason='; '.join(errors))

TOY_POLICIES = ('baseline','tight','wideleft','wideright','equal95')
TOY_MASTER = 63555260924

def candidate_asimov():
    truth = np.load(NULL_PATH)['truth']
    rows,masks = [],{}
    for m in MASSES[:-1]:
        contexts = {p:ToyContext(m,p) for p in TOY_POLICIES}
        for policy,ctx in contexts.items():
            masks[f'{policy}_fit_m{m:03d}'] = ctx.fit
            masks[f'{policy}_guard_m{m:03d}'] = ctx.guard
        for source,t in (('nominal',truth),('stress',D['stress'])):
            baseline = contexts['baseline'].fit_counts(t)
            assert baseline['fit_valid']
            A = 3*baseline['sigma_postfit']
            for policy,ctx in contexts.items():
                null = ctx.fit_counts(t)
                inj = ctx.fit_counts(t+A*ctx.categories[1:-1])
                known0 = ctx.fit_counts(t,known_truth=t)
                known1 = ctx.fit_counts(t+A*ctx.categories[1:-1],known_truth=t)
                assert all(r['fit_valid'] for r in (null,inj,known0,known1))
                response = (inj['Ahat']-null['Ahat'])/A
                known_response = (known1['Ahat']-known0['Ahat'])/A
                b,L,_ = ctx.prediction(t)
                limit = C.OneSignalProfile(b,L,ctx.pgauss[ctx.fit]).limit(t[ctx.fit],alpha=.1)
                assert limit['ok'] and limit['max_score']<3e-5 and abs(limit['cls']-.1)<2e-6
                rows.append(dict(mass_MeV=m,source=source,policy=policy,A_expected=A,
                    fit_bins=int(ctx.fit.sum()),guard_bins=int(ctx.guard.sum()),
                    left_training_bins=int(np.sum(D['x']<D['x'][ctx.guard].min())),
                    right_training_bins=int(np.sum(D['x']>D['x'][ctx.guard].max())),
                    MC_fit_fraction=float(ctx.categories[1:-1][ctx.fit].sum()),
                    MC_training_fraction=float(ctx.categories[1:-1][~ctx.guard].sum()),
                    Ahat_null=null['Ahat'],sigma_null=null['sigma_postfit'],paired_response=response,
                    known_background_response=known_response,GP_response_over_known_background=response/known_response,
                    response_adjusted_error=null['sigma_postfit']/response,
                    local_SNR_per_expected_yield=response/null['sigma_postfit'],
                    raw_native_CLs90=limit['A90'],response_scaled_CLs90_diagnostic=limit['A90']/response))
    df = pd.DataFrame(rows)
    for key in ('response_adjusted_error','response_scaled_CLs90_diagnostic'):
        base = df[df.policy=='baseline'].set_index(['mass_MeV','source'])[key]
        df[key+'_ratio_to_baseline'] = [v/base.loc[(m,s)] for v,m,s in zip(df[key],df.mass_MeV,df.source)]
    df.to_csv(B/'results/window_candidate_asimov.csv',index=False,float_format='%.17g')
    np.savez_compressed(B/'results/window_candidate_masks.npz',**masks)

def run_toys():
    import time
    start = time.monotonic()
    truth = np.load(NULL_PATH)['truth']
    protocol = dict(master_seed=TOY_MASTER,masses=list(MASSES[:-1]),policies=list(TOY_POLICIES),
        pilot=100,evaluation=100,source='nominal_2021_10pct_GP_mean',
        expected_signal='3 times mean baseline independent-pilot returned observed-Hessian error',
        shape='full selected native MC categories including outside support; no renormalization',
        pairing='Same evaluation background and signal draws across all policies; background shared across masses',
        background_seeds='[master,1(pilot) or 2(evaluation),toy]',signal_seeds='[master,3,mass,toy]',
        frozen_without_observed_data=True,numerical_threads=1,workers=1,watchdog_seconds=1800,
        script_sha256=sha(__file__),null_sha256=sha(NULL_PATH),
        template_sha256={str(m):sha(PIN/f'histograms/m{m:03d}.npz') for m in MASSES[:-1]})
    pp = B/'provenance/window_toy_protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol, 'Frozen window-toy protocol changed'
    else:
        write_json(pp,protocol)
    out = B/'results/window_checkpoints';out.mkdir(exist_ok=True)
    cohorts = {name:np.array([np.random.default_rng(np.random.SeedSequence([TOY_MASTER,ns,i])).poisson(truth)
                             for i in range(100)]) for name,ns in (('pilot',1),('evaluation',2))}
    np.savez_compressed(B/'results/window_cohorts.npz',**cohorts)
    def chunk(cohort,m,first,last,reference=None):
        path = out/f'{cohort}_m{m:03d}_t{first:03d}.csv'
        marker = path.with_suffix('.json')
        if marker.exists():
            old = json.loads(marker.read_text())
            assert old['protocol_sha256']==sha(pp) and old['rows_sha256']==sha(path)
            if reference is not None:
                assert old['pilot_reference_sha256']==sha(B/'results/window_pilot_reference.json')
            return
        contexts = {p:ToyContext(m,p) for p in (('baseline',) if cohort=='pilot' else TOY_POLICIES)}
        A = 0. if cohort=='pilot' else 3*reference[str(m)]['s0']
        rows,draws = [],[]
        for toy in range(first,last):
            assert time.monotonic()-start < 1800, 'Window study wall-time limit reached; resume checkpoints'
            background = cohorts[cohort][toy]
            draw = np.zeros(len(truth)+2,dtype=np.int64)
            if cohort=='evaluation':
                draw = np.random.default_rng(np.random.SeedSequence([TOY_MASTER,3,m,toy])).poisson(A*contexts['baseline'].categories)
            draws.append(draw)
            for policy,ctx in contexts.items():
                for z in ((0,) if cohort=='pilot' else (0,3)):
                    counts = background+(draw[1:-1] if z else 0)
                    row = dict(cohort=cohort,mass_MeV=m,toy=toy,policy=policy,z=z,A_expected=A if z else 0.,
                        background_hash=ahash(background),counts_hash=ahash(counts),signal_hash=ahash(draw) if z else '',
                        actual_full=int(draw.sum()) if z else 0,actual_window=int(draw[1:-1][ctx.fit].sum()) if z else 0,
                        actual_training=int(draw[1:-1][~ctx.guard].sum()) if z else 0,
                        actual_outside_support=int(draw[0]+draw[-1]) if z else 0)
                    row.update(ctx.fit_counts(counts))
                    rows.append(row)
        df = pd.DataFrame(rows)
        temp = path.with_suffix('.tmp');df.to_csv(temp,index=False,float_format='%.17g');temp.replace(path)
        draws_path = path.with_suffix('.npz')
        np.savez_compressed(draws_path,draws=np.array(draws),toys=np.arange(first,last))
        write_json(marker,dict(protocol_sha256=sha(pp),rows_sha256=sha(path),draws_sha256=sha(draws_path),
            pilot_reference_sha256=sha(B/'results/window_pilot_reference.json') if reference is not None else None,
            rows=len(rows),valid=int(df.fit_valid.sum())))
        assert df.fit_valid.all(), f'Window fit failure retained in {path}'
    for m in MASSES[:-1]:
        for first in range(0,100,10):
            chunk('pilot',m,first,first+10)
        print('window pilot',m,round(time.monotonic()-start,1),flush=True)
    pilot = pd.concat([pd.read_csv(p,float_precision='round_trip') for p in sorted(out.glob('pilot_*.csv'))],ignore_index=True)
    assert len(pilot)==1000 and pilot.fit_valid.all()
    pilot.to_csv(B/'results/window_pilot_rows.csv',index=False,float_format='%.17g')
    reference = {str(int(m)):dict(s0=float(q.sigma_postfit.mean()),sigma_sd=float(q.sigma_postfit.std(ddof=1)),
                    n=len(q)) for m,q in pilot.groupby('mass_MeV')}
    refp = B/'results/window_pilot_reference.json'
    if refp.exists():
        assert json.loads(refp.read_text())['masses']==reference
    else:
        write_json(refp,dict(masses=reference,pilot_rows_sha256=sha(B/'results/window_pilot_rows.csv'),
                    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    for m in MASSES[:-1]:
        for first in range(0,100,10):
            chunk('evaluation',m,first,first+10,reference)
        print('window evaluation',m,round(time.monotonic()-start,1),flush=True)
    evaluation = pd.concat([pd.read_csv(p,float_precision='round_trip') for p in sorted(out.glob('evaluation_*.csv'))],ignore_index=True)
    assert len(evaluation)==10000 and evaluation.fit_valid.all()
    evaluation.to_csv(B/'results/window_evaluation_rows.csv',index=False,float_format='%.17g')
    summarize_toys(evaluation)

def summarize_toys(df):
    rows = []
    for m,q in df.groupby('mass_MeV'):
        zero = q[q.z==0].pivot(index='toy',columns='policy',values='Ahat')[list(TOY_POLICIES)].to_numpy()
        injected = q[q.z==3].pivot(index='toy',columns='policy',values='Ahat')[list(TOY_POLICIES)].to_numpy()
        errors = q[q.z==0].pivot(index='toy',columns='policy',values='sigma_postfit')[list(TOY_POLICIES)].to_numpy()
        A = float(q[q.z==3].A_expected.iloc[0])
        rng = np.random.default_rng(np.random.SeedSequence([TOY_MASTER,4,int(m)]))
        indices = rng.integers(0,100,size=(2000,100))
        response = np.mean(injected-zero,axis=0)/A
        bootR = np.mean((injected-zero)[indices],axis=1)/A
        sd = np.std(zero,axis=0,ddof=1)
        bootSD = np.std(zero[indices],axis=1,ddof=1)
        meanerr = np.mean(errors,axis=0)
        booterr = np.mean(errors[indices],axis=1)
        empirical = sd/response
        returned = meanerr/response
        bootEmp,bootRet = bootSD/bootR,booterr/bootR
        for i,policy in enumerate(TOY_POLICIES):
            ratio = empirical[i]/empirical[0]
            bootratio = bootEmp[:,i]/bootEmp[:,0]
            retRatio = returned[i]/returned[0]
            bootRetRatio = bootRet[:,i]/bootRet[:,0]
            lo,hi = np.quantile(bootratio,[.025,.975])
            rlo,rhi = np.quantile(bootR[:,i],[.025,.975])
            erlo,erhi = np.quantile(bootRetRatio,[.025,.975])
            rows.append(dict(mass_MeV=int(m),policy=policy,evaluation_toys=100,A_expected=A,
                null_mean=float(zero[:,i].mean()),null_empirical_sigma=float(sd[i]),mean_returned_sigma=float(meanerr[i]),
                response=float(response[i]),response_bootstrap95_low=float(rlo),response_bootstrap95_high=float(rhi),
                empirical_sigma_over_response=float(empirical[i]),returned_sigma_over_response=float(returned[i]),
                empirical_precision_ratio=float(ratio),empirical_precision_ratio95_low=float(lo),empirical_precision_ratio95_high=float(hi),
                returned_precision_ratio=float(retRatio),returned_precision_ratio95_low=float(erlo),returned_precision_ratio95_high=float(erhi),
                null_mean_pull=float(np.mean(zero[:,i]/errors[:,i])),null_pull_sd=float(np.std(zero[:,i]/errors[:,i],ddof=1)),
                caution='Approximate paired bootstrap sensitivity diagnostic; no policy adoption or limit coverage claim'))
    pd.DataFrame(rows).to_csv(B/'results/window_toy_summary.csv',index=False,float_format='%.17g')
    checks = dict(passed=True,pilot_rows=1000,evaluation_rows=len(df),all_free_fits_valid=bool(df.fit_valid.all()),
        backgrounds_paired=bool(df.groupby(['mass_MeV','toy']).background_hash.nunique().max()==1),
        injected_counts_paired=bool(df[df.z==3].groupby(['mass_MeV','toy']).counts_hash.nunique().max()==1),
        signal_normalization='All full selected MC, outside-support categories retained',
        bootstrap_replicates=2000,bootstrap_unit='whole evaluation toy ID jointly across policies',
        profile_or_CLs_coverage_tested=False,adopted_policy=None,
        pilot_reference_sha256=sha(B/'results/window_pilot_reference.json'),
        evaluation_rows_sha256=sha(B/'results/window_evaluation_rows.csv'))
    assert checks['backgrounds_paired'] and checks['injected_counts_paired']
    write_json(B/'qa/window_toys.json',checks)

def main():
    for d in ('results','provenance','qa'):
        (B/d).mkdir(exist_ok=True)
    rows,masks,asimov = [],{},[]
    truth = np.load(NULL_PATH)['truth']
    assert np.array_equal(np.load(NULL_PATH)['edges_GeV'],D['edges'])
    for m in MASSES:
        r,g,fit,pmc = geometry(m)
        rows.extend(r)
        masks[f'fit_m{m:03d}'] = fit
        masks[f'mc_probability_m{m:03d}'] = pmc
        for policy in POLICIES:
            masks[f'{policy}_m{m:03d}'] = g[policy]
        if m > 240:
            continue
        pg = np.diff(ndtr((D['edges']-center(m)/1000)/C.sigma('2021',m)))
        for source,t in (('nominal',truth),('stress',D['stress'])):
            base,_ = asimov_fit(m,t,g['baseline'],fit,pg)
            A = 3*base['sigma']
            for row in r:
                if not row['feasible']:
                    continue
                policy = row['policy']
                null,diag = asimov_fit(m,t,g[policy],fit,pg)
                signal,_ = asimov_fit(m,t+A*pmc,g[policy],fit,pg)
                asimov.append(dict(mass_MeV=m,source=source,policy=policy,A_expected=A,
                    Ahat_null=null['A'],sigma_null=null['sigma'],pull_null=null['A']/null['sigma'],
                    Ahat_signal=signal['A'],sigma_signal=signal['sigma'],
                    paired_response=(signal['A']-null['A'])/A,
                    sigma_ratio_to_baseline=null['sigma']/base['sigma'],
                    covariance_load=diag['load'],nuisance_rank=diag['rank']))
    df = pd.DataFrame(rows)
    df.to_csv(B/'results/window_metrics.csv',index=False,float_format='%.17g')
    pd.DataFrame(asimov).to_csv(B/'results/window_asimov.csv',index=False,float_format='%.17g')
    np.savez_compressed(B/'results/window_masks.npz',**masks)
    files = [PIN/f'histograms/m{m:03d}.npz' for m in MASSES]
    files += [PIN/'inputs/spectrum_2021.npz',NULL_PATH]
    write_json(B/'provenance/window_inputs.json',dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inputs=[dict(path=str(p.relative_to(REPO)),sha256=sha(p)) for p in files],
        script_sha256=sha(__file__),log_shift_coefficients_MeV=LOG_COEFFICIENTS,
        logarithmic_law_domain_MeV=[60,240],
        quantiles='Piecewise-uniform native histogram CDF divided by all selected MC, including overflow',
        shortest_interval='Shortest interval in recorded histogram covering 95% of full selected MC',
        exclusion='Union with fixed core-centered fit window; include intersecting analysis bins',
        coordinate='u=(m_rec-c_MC)/sigma_core is dimensionless; nominal and core widths are kept separate',
        wiggle_parameters='For physical width s, guard=[c-2*s*L_wig,c+2*s*R_wig]; coefficients tabulated for both s definitions',
        no_observed_count_window_selection=True,no_MC_tail_removal=True,no_injection_renormalization=True,
        scope='MC-only shape/exclusion proposal; Asimov diagnostics are not held-out toy validation or coverage'))
    q = df[(df.scope=='primary') & (df.policy!='baseline')]
    check = dict(passed=bool((q.MC_fraction_in_guard>=.95-1e-12).all()),
        masses=len(MASSES),policies=len(POLICIES),quantile_guards_retain_at_least95pct=bool((q.MC_fraction_in_guard>=.95-1e-12).all()),
        infeasible=df.loc[~df.feasible,['mass_MeV','policy']].to_dict('records'),
        fit_in_every_guard=all(np.all(masks[f'{p}_m{m:03d}'][masks[f'fit_m{m:03d}']]) for m in MASSES for p in POLICIES),
        output_hashes={p.name:sha(p) for p in (B/'results/window_metrics.csv',B/'results/window_asimov.csv',B/'results/window_masks.npz')})
    assert check['passed'] and check['fit_in_every_guard']
    write_json(B/'qa/window_geometry.json',check)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,
                         'savefig.bbox':'tight','pdf.fonttype':42})
    fig,axs = plt.subplots(2,2,figsize=(8.1,6.0))
    primary = df[df.scope=='primary']
    colors = dict(baseline='#222222',equal95='#0072B2',shortest95='#D55E00')
    labels = dict(baseline='Core fit guard',equal95='Equal-tail 95% guard',shortest95='Shortest 95% guard')
    for p in POLICIES:
        q = primary[primary.policy==p]
        a = pd.DataFrame(asimov)
        a = a[(a.source=='nominal') & (a.policy==p)]
        axs[0,0].plot(q.mass_MeV,100*q.MC_fraction_training,'o-',ms=3,color=colors[p],label=labels[p])
        axs[0,1].plot(q.mass_MeV,q.guard_width_nominal_sigma,'o-',ms=3,color=colors[p])
        axs[1,0].plot(a.mass_MeV,a.sigma_ratio_to_baseline,'o-',ms=3,color=colors[p])
        axs[1,1].plot(a.mass_MeV,a.paired_response,'o-',ms=3,color=colors[p])
    axs[0,0].set_ylabel('Full selected MC in GP training [%]')
    axs[0,0].legend(loc='upper right')
    axs[0,1].set_ylabel('Exclusion width / nominal resolution')
    axs[1,0].set_ylabel('Asimov null yield error / baseline')
    axs[1,0].set_yscale('log')
    axs[1,1].set_ylabel('Asimov paired Gaussian response')
    axs[1,1].axhline(1,color='.5',ls='--',lw=.8)
    for ax in axs.flat:
        ax.set_xlabel('Generated mass [MeV]')
        ax.grid(alpha=.2)
    fig.suptitle('MC-only exclusion geometry and deterministic tradeoffs',fontsize=11)
    fig.tight_layout()
    for suffix in ('pdf','png'):
        fig.savefig(B/f'results/window_tradeoffs.{suffix}',dpi=170)
    plt.close(fig)
    print(df[['mass_MeV','policy','MC_fraction_in_guard','MC_fraction_training','feasible']].to_string(index=False))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--toys',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize_toys(pd.read_csv(B/'results/window_evaluation_rows.csv',float_precision='round_trip'))
    else:
        main()
        candidate_asimov()
        if args.toys:
            run_toys()
