"""Post-hoc signed-score stability and paired-null mass-coherence diagnostics.

The statistic/region choices are fixed in this script before observed peak
positions are evaluated. These conditional lower tails are diagnostic, not
discovery p values: neither the 80--105 region nor the tested widths were blind.
"""
from engine import B, C, P, YEARS, Context, fit
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import beta, norm

WIDTHS = tuple(P['blind_halfwidth_sigma'])
COMMON_M = np.arange(80., 101.)
GAUSSIAN_N = 100000
GAUSSIAN_SEED = 58520260919
SIGMA = np.array([1000*C.sigma(y, 92.) for y in YEARS])
DEFINITION = {
    'primary_statistic': 'T = mean_d,w [ (mhat_d,w - mbar) / sigma_d(92) ]^2',
    'weighted_center': 'mbar = sum_d,w mhat_d,w / sigma_d(92)^2 divided by sum_d,w 1 / sigma_d(92)^2',
    'peak_ordering': 'Maximum signed standardized Z=(r-a)/s among raw r>0 grid sites; exact ties choose lowest mass',
    'requested_peak_ordering': 'Individual 80--105/support summary maximizes signed Z across all available sites; the calibrated common-region coherence additionally requires positive raw r. All reported observed peaks satisfy raw r>0.',
    'ineligible_rule': 'If any dataset-width has no positive raw r in region, set T and W to infinity (never counted as coherent)',
    'primary_region_MeV': [80, 100],
    'requested_diagnostic_region_MeV': [80, 105],
    'region_reason': '2015 support ends at 100 MeV; all-dataset coherence uses the common intersection 80--100. Individual requested-region summaries retain 80--105 where supported.',
    'secondary_statistic': 'W = mean_d ( [max_w mhat_d,w - min_w mhat_d,w] / sigma_d(92) )^2',
    'tail': 'Inclusive lower tail; small values mean more coherent/stable. T is the single primary scalar; T and W p values are not multiplied.',
    'significance_interpretation': 'Exploratory fixed-region conditional null diagnostic; not a discovery p value, not corrected for region, method, or width selection',
    'direct_null': '256 paired full-spectrum Poisson rows with exact profile scans, reused at every mass and width',
    'gaussian_null': '100000 supplemental linear-response Gaussian draws per independent dataset; full covariance across all diagnostic masses and widths from stacked D columns',
    'amplitude_units': 'Amplitude A is epsilon2/1e-8 in the original dataset template parameterization; amplitudes are independent here. Full-spectrum signal events=A*sum(full Gaussian template); window signal events=A*sum(window template). Nonnegative MLE=max(A,0).',
}


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def lower_tail(samples, threshold):
    n = len(samples)
    k = int(np.count_nonzero(samples <= threshold))
    return dict(k=k, N=n, p=(k+1)/(n+1), raw_p=k/n,
                lo95=0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
                hi95=1. if k == n else float(beta.ppf(.975, k+1, n-k)),
                upper95=1. if k == n else float(beta.ppf(.95, k+1, n-k)),
                status='upper_bound_only' if k == 0 else 'finite_MC',
                eligible_null=int(np.count_nonzero(np.isfinite(samples))))


def peak_masses(raw, offset, scale):
    """raw (..., dataset, width, mass); each year supplies matching a/s."""
    z = (raw-offset)/scale
    positive = raw > 0
    eligible = positive.any(axis=-1)
    indices = np.argmax(np.where(positive, z, -np.inf), axis=-1)
    return COMMON_M[indices], eligible


def metrics(peaks, eligible):
    weight = np.broadcast_to(1/SIGMA[:, None]**2, peaks.shape)
    center = (peaks*weight).sum(axis=(-2, -1))/weight.sum(axis=(-2, -1))
    T = np.mean(((peaks-center[..., None, None])/SIGMA[:, None])**2,
                axis=(-2, -1))
    delta = peaks.max(axis=-1)-peaks.min(axis=-1)
    W = np.mean((delta/SIGMA)**2, axis=-1)
    ok = eligible.all(axis=(-2, -1))
    return np.where(ok, T, np.inf), np.where(ok, W, np.inf), center


def main():
    for folder in ('results', 'fields', 'qa'):
        (B/folder).mkdir(exist_ok=True)
    save_json(B/'results/stability_definition.json', DEFINITION)
    parent = pd.read_csv(B/'inputs/parent_results/significance_and_reach.csv')
    curves, peaks, at92, checks, null_provenance = [], [], [], [], []
    observed, offsets, scales, direct, factors = [], [], [], [], []
    for year in YEARS:
        bank = dict(np.load(B/f'inputs/null_{year}.npz'))
        regenerated = np.random.default_rng(np.random.SeedSequence(bank['seed'])).poisson(
            bank['truth'], size=bank['counts'].shape).astype(float)
        assert np.array_equal(regenerated, bank['counts'])
        null_provenance.append(dict(dataset=year, seed=bank['seed'].tolist(),
                                    toy_ids=list(range(len(bank['counts']))),
                                    counts_sha256=hashlib.sha256(bank['counts'].tobytes()).hexdigest(),
                                    replay_equal=True))
        oo, aa, ss, vv, derivatives = [], [], [], [], []
        for width in WIDTHS:
            f = dict(np.load(B/f'inputs/parent_fields/w{width:.2f}_{year}.npz'))
            mass = f['masses']; r = f['observed_r']; z = (r-f['a'])/f['s']
            local_p = np.where(r > 0, norm.sf(z), 1.)
            for j, m in enumerate(mass):
                curves.append(dict(dataset=year, width_sigma=width, mass_MeV=m,
                                   signed_Z=z[j], local_p=local_p[j],
                                   local_Z=max(0., float(norm.isf(local_p[j]))),
                                   raw_r=r[j], q_nonnegative=max(r[j], 0.)**2))
            diagnostic = np.flatnonzero((mass >= 80)&(mass <= 105))
            j = diagnostic[np.argmax(z[diagnostic])]
            peaks.append(dict(dataset=year, width_sigma=width,
                              region_lo_MeV=80., region_hi_MeV=float(mass[diagnostic].max()),
                              peak_mass_MeV=mass[j], peak_Z=z[j], peak_raw_r=r[j]))
            common = np.flatnonzero(np.isin(mass, COMMON_M))
            assert np.array_equal(mass[common], COMMON_M)
            assert f['validation'].shape == (len(bank['counts']), len(mass))
            oo.append(r[common]); aa.append(f['a'][common]); ss.append(f['s'][common])
            vv.append(f['validation'][:, common]); derivatives.append(f['D'][:, common])
            i92 = int(np.flatnonzero(mass == 92.)[0])
            row = parent[(parent.scope.astype(str) == year)&(parent.domain == 'full')&
                         (parent.width_sigma == width)&(parent.mass_MeV == 92.)]
            assert len(row) == 1
            A = float(row.iloc[0].observed_amplitude)
            ctx = Context(year, 92., width)
            replay = fit([ctx.predict(bank['observed'])])
            toy_replay = fit([ctx.predict(bank['counts'][0])])
            error = abs(replay['r']-r[i92]); toy_error = abs(toy_replay['r']-f['validation'][0,i92])
            assert error < 2e-5 and toy_error < 2e-5
            assert abs(A-replay['f']['A']) < 1e-5*max(1., abs(A))
            full_yield = float(C.signal(year, 92.).sum())
            window_yield = float(ctx.S.sum())
            at92.append(dict(dataset=year, width_sigma=width, mass_MeV=92.,
                             sigma92_MeV=1000*C.sigma(year,92.),
                             signed_Z=z[i92], local_p=local_p[i92],
                             local_Z=max(0.,float(norm.isf(local_p[i92]))),
                             raw_r=r[i92], q_nonnegative=max(r[i92],0.)**2,
                             amplitude_raw_1e8=A, amplitude_nonnegative_1e8=max(A,0.),
                             signal_events_raw=A*full_yield,
                             signal_events_nonnegative=max(A,0.)*full_yield,
                             window_signal_events_nonnegative=max(A,0.)*window_yield,
                             window_template_fraction=window_yield/full_yield))
            checks.append(dict(dataset=year,width_sigma=width,observed92_r_error=error,
                               paired_toy0_92_r_error=toy_error))
        observed.append(oo); offsets.append(aa); scales.append(ss)
        direct.append(np.stack(vv,axis=1))
        D = np.concatenate(derivatives, axis=1)
        cov = D.T@D; eigenvalues, U = np.linalg.eigh((cov+cov.T)/2)
        assert eigenvalues.min() > -1e-8
        factor = U*np.sqrt(np.maximum(eigenvalues, 0.))
        assert np.max(np.abs(factor@factor.T-cov)) < 1e-8
        factors.append(factor)
    observed=np.array(observed); offsets=np.array(offsets); scales=np.array(scales)
    direct=np.stack(direct,axis=1)
    obs_peaks, obs_eligible=peak_masses(observed,offsets,scales)
    direct_peaks,direct_eligible=peak_masses(direct,offsets,scales)
    Tobs,Wobs,center=metrics(obs_peaks,obs_eligible)
    assert np.isfinite(Tobs), 'Observed diagnostic includes a nonpositive-only dataset-width'
    Td,Wd,_=metrics(direct_peaks,direct_eligible)
    # One independent stream per dataset; shared variates correlate every width
    # and mass within that dataset through the jointly factored derivative map.
    rngs=[np.random.default_rng(np.random.SeedSequence([GAUSSIAN_SEED,int(y),991])) for y in YEARS]
    gaussian_peaks=[]; gaussian_eligible=[]; gaussian_T=[]; gaussian_W=[]
    for first in range(0,GAUSSIAN_N,2048):
        n=min(2048,GAUSSIAN_N-first)
        raw=np.stack([offsets[i]+(rng.standard_normal((n,fac.shape[1]))@fac.T).reshape(n,len(WIDTHS),len(COMMON_M))
                      for i,(rng,fac) in enumerate(zip(rngs,factors))],axis=1)
        pp,ee=peak_masses(raw,offsets,scales); tt,ww,_=metrics(pp,ee)
        gaussian_peaks.append(pp);gaussian_eligible.append(ee);gaussian_T.append(tt);gaussian_W.append(ww)
    gp=np.concatenate(gaussian_peaks); ge=np.concatenate(gaussian_eligible)
    Tg=np.concatenate(gaussian_T); Wg=np.concatenate(gaussian_W)
    requested_peaks=pd.DataFrame(peaks)
    ranges=[]; common_rows=[]
    for i,year in enumerate(YEARS):
        ps=requested_peaks[requested_peaks.dataset == year]
        delta=float(ps.peak_mass_MeV.max()-ps.peak_mass_MeV.min())
        common_delta=float(np.ptp(obs_peaks[i]))
        ranges.append(dict(dataset=year,sigma92_MeV=SIGMA[i],
                           region_lo_MeV=80.,region_hi_MeV=float(ps.region_hi_MeV.iloc[0]),
                           delta_m_MeV=delta,R=delta/SIGMA[i],
                           common_delta_m_MeV=common_delta,common_R=common_delta/SIGMA[i]))
        for k,width in enumerate(WIDTHS):
            common_rows.append(dict(dataset=year,width_sigma=width,
                                    region_lo_MeV=80.,region_hi_MeV=100.,
                                    peak_mass_MeV=float(obs_peaks[i,k]),eligible=bool(obs_eligible[i,k])))
    summary=dict(definition=DEFINITION, sigma92_MeV=dict(zip(YEARS,SIGMA)),
                 observed_T=float(Tobs),observed_W=float(Wobs),weighted_center_MeV=float(center),
                 direct_T=lower_tail(Td,float(Tobs)),direct_W=lower_tail(Wd,float(Wobs)),
                 gaussian_T=lower_tail(Tg,float(Tobs)),gaussian_W=lower_tail(Wg,float(Wobs)),
                 gaussian_seed=GAUSSIAN_SEED,gaussian_stream=[int(y) for y in YEARS],
                 posthoc_selection_corrected=False,
                 discovery_p_value=False)
    curve_frame=pd.DataFrame(curves)
    competing=curve_frame[(curve_frame.dataset == '2021')&curve_frame.mass_MeV.isin([80.,92.,93.])].copy()
    competing['interpretation']='Fixed comparison of the 80 MeV boundary excursion and the 92--93 MeV local structure; rank changes do not imply a moving resonance'
    summary['competing_peak_note']='For 2021, the region maximum changes from 93 to the 80 MeV diagnostic boundary at widths 2.5 and 2.6. This is a competing-excursion rank switch, not evidence of a resonance moving 13 MeV. No narrower optimized diagnostic region is introduced.'
    for name,df in [('curves',curve_frame),('peaks',requested_peaks),
                    ('at92',pd.DataFrame(at92)),('ranges',pd.DataFrame(ranges)),
                    ('coherence_peaks',pd.DataFrame(common_rows)),('competing_peaks',competing)]:
        df.to_csv(B/f'results/stability_{name}.csv',index=False,float_format='%.17g')
    np.savez_compressed(B/'fields/stability_null.npz',
                        datasets=np.array(YEARS),widths=np.array(WIDTHS),masses=COMMON_M,
                        observed_peaks=obs_peaks,observed_eligible=obs_eligible,
                        direct_T=Td,direct_W=Wd,direct_peaks=direct_peaks,
                        direct_eligible=direct_eligible,direct_toy_ids=np.arange(len(Td)),
                        gaussian_T=Tg,gaussian_W=Wg,gaussian_peaks=gp.astype(np.int16),
                        gaussian_eligible=ge,gaussian_seed=GAUSSIAN_SEED)
    # Independent explicit scalar replay checks the vectorized metric, including
    # resolution weighting, the 12-cell denominator, and ineligible handling.
    metric_error=0.
    for k in [0,1,17,255]:
        if direct_eligible[k].all():
            vals=[(direct_peaks[k,i,j],SIGMA[i]) for i in range(3) for j in range(4)]
            mean=sum(m/s**2 for m,s in vals)/sum(1/s**2 for _,s in vals)
            explicit=sum(((m-mean)/s)**2 for m,s in vals)/12
            metric_error=max(metric_error,abs(explicit-Td[k]))
        else:assert np.isinf(Td[k]) and np.isinf(Wd[k])
    assert metric_error < 1e-12
    assert metrics(np.zeros((3,4)),np.zeros((3,4),dtype=bool))[0] == np.inf
    save_json(B/'results/stability_coherence_summary.json',summary)
    save_json(B/'qa/stability_validation.json',dict(complete=True,source_counts=null_provenance,
              replay_checks=checks,statistic_explicit_scalar_max_error=metric_error,
              signed_curve_rows=len(curves),at92_rows=len(at92),poisson_rows=len(Td),
              gaussian_rows=len(Tg),cross_width_draws_paired=True,
              min_sigma92_MeV=float(SIGMA.min())))
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(ranges).to_string(index=False))
    print(pd.DataFrame(at92)[['dataset','width_sigma','signed_Z','signal_events_nonnegative']].to_string(index=False))


if __name__ == '__main__':
    main()
