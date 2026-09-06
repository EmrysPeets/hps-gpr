#!/usr/bin/env python3
"""Check derivative tables and saved echo likelihoods without fitting or drawing."""
from pathlib import Path
import hashlib
import json
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
from scipy.stats import beta, norm

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
DERIVED = HERE/'derived'
OLD = ROOT/'study_results/v4p9p12_2021_peak_dip_diagnostic_20toys_20260905'
GLOBAL = ROOT/'study_results/v4p9p16_combined_global_20260906/global'
checked = 0
failures = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition):
    global checked
    checked += 1
    if not bool(condition):
        failures.append(name)


def close(a, b, atol=2e-13, rtol=2e-12):
    return np.allclose(a, b, atol=atol, rtol=rtol)


def nll_score_information(n, lam, theta, L, w, scale, free):
    positive = n > 0
    t = (lam[positive]-n[positive])/n[positive]
    nll = float(np.sum(n[positive]*(t-np.log1p(t)))
                +lam[~positive].sum()+.5*theta@theta)
    residual = (lam-n)/lam
    grad = L.T@residual+theta
    if free:
        grad = np.r_[scale*w@residual, grad]
    weight = n/lam**2
    nuisance = (L.T*weight)@L+np.eye(L.shape[1])
    cross = L.T@(weight*w)
    information = float(w@(weight*w)-cross@np.linalg.solve(nuisance, cross))
    return nll, float(abs(grad).max()), information


def main():
    provenance = json.loads((HERE/'provenance/numerical_inputs.json').read_text())
    for kind in ('input_sha256', 'output_sha256'):
        for relative, digest in provenance[kind].items():
            check(kind+':'+relative, sha(ROOT/relative) == digest)
    # Additional direct-local probabilities in the revised display.
    grid = pd.read_csv(DERIVED/'probability_grid.csv', float_precision='round_trip').set_index('mass_MeV')
    vectors = np.load(GLOBAL/'scan_vectors.npz')
    maxima = np.load(GLOBAL/'analysis/maxima.npz')
    masses = vectors['masses_MeV']
    toy = vectors['validation1000_profiled']; a = vectors['asimov_profiled'][0]
    D = vectors['asimov_profiled'][1:]-a; C = D.T@D; s = np.sqrt(C.diagonal())
    K = C/np.outer(s, s)
    rz = (toy-a)/s; scores = np.where(toy > 0, rz, -np.inf)
    observed = pd.read_csv(GLOBAL/'observed.csv', float_precision='round_trip').set_index('mass_MeV').profiled_r
    check('derivative_full_grid', np.array_equal(grid.index, masses))
    direct_count_examples = {}
    for j, mass in enumerate(masses):
        row = grid.loc[mass]; r = observed.loc[mass]; z = (r-a[j])/s[j]
        threshold = z if r > 0 else -np.inf
        check(f'grid_scalars:{mass}', close([row.observed_r, row.asimov_r, row.response_sd, row.z], [r, a[j], s[j], z]))
        check(f'grid_local:{mass}', close([row.nominal_local_p, row.conditional_local_gaussian,
              row.ungated_signed_gaussian], [norm.sf(max(0., r)), norm.sf(z) if r > 0 else 1., norm.sf(z)], atol=0))
        groups = dict(direct_local=(scores[:, j], threshold), direct_global=(scores.max(axis=1), threshold),
              gp_global=(maxima['profiled_gp'], threshold),
              gp_raw_global=(maxima['profiled_gp_raw'], max(0., r)),
              direct_raw_global=(np.maximum(toy, 0).max(axis=1), max(0., r)))
        counts = {}
        for label, (samples, cutoff) in groups.items():
            n = len(samples); k = int(np.count_nonzero(samples >= cutoff)); counts[label] = k
            low = 0. if k == 0 else beta.ppf(.025, k, n-k+1)
            high = 1. if k == n else beta.ppf(.975, k+1, n-k)
            upper = 1. if k == n else beta.ppf(.95, k+1, n-k)
            check(f'grid_tails:{mass}:{label}', close([row[label+'_'+key] for key in
                  ('k', 'n', 'p', 'low95', 'high95', 'upper95')], [k, n, k/n, low, high, upper], atol=0))
        check(f'local_global_sample_inclusion:{mass}', counts['direct_local'] <= counts['direct_global'])
        if mass in (21, 41, 66, 75, 76, 77, 78, 90, 91, 92, 93):
            direct_count_examples[str(mass)] = counts
    corr = pd.read_csv(DERIVED/'combined_correlations.csv', float_precision='round_trip')
    for row in corr.itertuples():
        i = int(row.left_MeV-19); j = int(row.right_MeV-19)
        check(f'correlation:{row.left_MeV}:{row.right_MeV}', close(
              [row.gp_combined_rho, row.direct_combined_rho], [K[i,j], np.corrcoef(toy[:,i], toy[:,j])[0,1]]))
    # Reconstruct every deterministic likelihood from arrays.
    table = pd.read_csv(DERIVED/'echo_dense_scans.csv', float_precision='round_trip').set_index(['mass_MeV', 'lane'])
    components = np.load(DERIVED/'echo_likelihood_components.npz')
    summary = json.loads((DERIVED/'echo_summary.json').read_text())
    rev = pd.read_csv(OLD/'reverse_injection/derived/common_truth_and_signals.csv')
    pair = pd.read_csv(OLD/'double_peak_injection/derived/generating_spectrum_and_gp_response.csv')
    resolutions = pd.read_csv(ROOT/'study_results/v4p9p12_expanded_snapshot_20260905/derived/nominal_mass_resolutions.csv',
              dtype={'dataset':str}).set_index(['dataset','mass_MeV']).sigma_MeV
    dense = pd.read_csv(ROOT/'study_results/v4p9p13_calibration_20260905/summary/observed_calibrated_limits.csv',
              float_precision='round_trip').set_index(['scope_key','mass_MeV'])
    signal = dict(background=np.zeros(len(rev)), inject_66=rev.signal_m66_counts.to_numpy(),
          inject_78=rev.signal_m78_counts.to_numpy(), double_65_78=pair.signal_pair_counts.to_numpy())
    check('positive_only_injections', all(np.all(v >= 0) for v in signal.values()))
    check('same_echo_background', close(rev.smooth_truth_counts, pair.smooth_truth_counts, atol=1e-7, rtol=1e-14))
    check('true_double_65_78_identity', close(pair.signal_pair_counts, pair.signal_65_counts+pair.signal_78_counts))
    check('echo_dimensions', len(table) == 145 and len(components.files) == 116*8)
    check('echo_no_random_toys_or_new_data', summary['new_random_toys'] == summary['new_unblinded_events'] == 0)
    max_nll_error = max_root_error = max_lambda_error = max_sigma_error = max_gradient = max_observed_error = 0.
    min_nesting_q = np.inf
    for mass in range(60, 89):
        resolution = resolutions.loc['2021',mass]
        mask = abs(rev.mass_MeV.to_numpy()-mass) <= 2.25*resolution
        centers = rev.mass_MeV.to_numpy(); step = np.median(np.diff(centers))
        full = norm.cdf((centers+step/2-mass)/resolution)-norm.cdf((centers-step/2-mass)/resolution)
        w_ref = full[mask]/full[mask].sum()
        error = abs(table.loc[(mass,'observed'),'r']-dense.loc[('individual_2021_10pct',mass),'signed_r_profiled_asymptotic'])
        max_observed_error = max(max_observed_error, error)
        check(f'observed_dense_root:{mass}', error < 1e-13)
        for lane, injection in signal.items():
            label = f'm{mass:03d}__{lane}__'; row = table.loc[mass,lane]
            get = lambda key: components[label+key]
            n, b, L, w = [get(key) for key in ('counts','gp_mean','L','w')]
            check(label+'generating_counts', np.array_equal(n, (rev.smooth_truth_counts.to_numpy()+injection)[mask]))
            check(label+'template', close(w, w_ref, atol=2e-14, rtol=2e-11))
            check(label+'factor', L.shape == (len(n),len(n)) and not np.any(np.triu(L,1)) and np.all(np.diag(L)>0))
            values = {}
            for fit, A in [('free',row.Ahat),('null',0.)]:
                theta, lam = get(fit+'_theta'), get(fit+'_lambda')
                lam_delta = float(abs(b+L@theta+A*w-lam).max()); max_lambda_error = max(max_lambda_error,lam_delta)
                check(label+fit+'_expectation', lam_delta < 1e-8 and np.all(lam>0))
                nll, score, information = nll_score_information(n,lam,theta,L,w,np.sqrt(b.sum()),fit=='free')
                values[fit] = nll; max_gradient = max(max_gradient,score)
                nll_delta = abs(nll-row['nll' if fit=='free' else 'null_nll']); max_nll_error = max(max_nll_error,nll_delta)
                check(label+fit+'_nll', nll_delta < 1e-11)
                check(label+fit+'_score', score < 2.01e-7)
                if fit == 'free':
                    sigma_delta = abs((1/np.sqrt(information))/row.sigma_A-1); max_sigma_error = max(max_sigma_error,sigma_delta)
                    check(label+'curvature', sigma_delta < 1e-10)
            q = 2*(values['null']-values['free']); min_nesting_q = min(min_nesting_q,q)
            root = float(np.sign(row.Ahat)*np.sqrt(max(0.,q))); root_delta = abs(root-row.r)
            max_root_error = max(max_root_error,root_delta)
            check(label+'root', root_delta < 1e-10 and q >= -1e-7)
    changes = pd.read_csv(DERIVED/'echo_injection_changes.csv', float_precision='round_trip').set_index('mass_MeV')
    pivot = table.r.unstack('lane')
    for lane in ('inject_66','inject_78','double_65_78'):
        check('echo_delta:'+lane, close(pivot[lane]-pivot.background, changes[lane]))
    for mass, values in summary['absolute_roots'].items():
        for lane, root in values.items():
            check('echo_summary:'+mass+':'+lane, close(pivot.loc[int(mass),lane],root))
    for mass, values in summary['injection_changes'].items():
        for lane, delta in values.items():
            check('echo_summary_change:'+mass+':'+lane, close(changes.loc[int(mass),lane],delta))
    for lane, injection in signal.items():
        check('echo_injection_yield:'+lane, close(injection.sum(),summary['injection_yields'][lane]))
    response = pd.read_csv(DERIVED/'echo_background_response.csv', float_precision='round_trip')
    for (mass,lane), block in response.groupby(['test_mass_MeV','lane']):
        block = block.sort_values('bin_mass_MeV'); label = f'm{mass:03d}__{lane}__'
        mask = abs(rev.mass_MeV.to_numpy()-mass) <= 2.25*resolutions.loc['2021',mass]
        check(f'echo_background_display:{mass}:{lane}', close(block.bin_mass_MeV, rev.mass_MeV[mask])
              and np.array_equal(block.gp_mean.to_numpy(),components[label+'gp_mean'])
              and np.array_equal(block.generating_counts.to_numpy(),components[label+'counts'])
              and np.array_equal(block.injected_counts.to_numpy(),signal[lane][mask]))
    output = dict(passed=not failures, checked_conditions=checked, failures=failures,
          probability_masses_checked=len(masses),direct_local_global_counts=direct_count_examples,
          deterministic_fits_checked=116,observed_dense_roots_checked=29,
          max_NLL_absolute_error=max_nll_error,max_signed_root_error=max_root_error,
          max_lambda_absolute_error=max_lambda_error,max_sigma_relative_error=max_sigma_error,
          max_likelihood_gradient=max_gradient,max_observed_reference_root_error=max_observed_error,
          minimum_free_null_q=min_nesting_q,
          echo_at_71_MeV={str(k):float(v) for k,v in pivot.loc[71].items()},
          injection_delta_at_71_MeV={str(k):float(v) for k,v in changes.loc[71].items()},
          positive_cross_responses=dict(inject_66_at_78=float(changes.loc[78,'inject_66']),
                inject_78_at_66=float(changes.loc[66,'inject_78'])),
          scope='Saved expectation, score and curvature evaluation only. No GP retraining, likelihood optimization, toys, fields, or additional data read.',
          input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'analyze.py',HERE/'PROTOCOL.md',
                HERE/'provenance/numerical_inputs.json',DERIVED/'probability_grid.csv',DERIVED/'echo_dense_scans.csv',
                DERIVED/'echo_likelihood_components.npz',DERIVED/'echo_injection_changes.csv',DERIVED/'echo_summary.json']})
    print(json.dumps(output,indent=2,allow_nan=False))
    raise SystemExit(0 if not failures else 1)


if __name__ == '__main__':
    main()
