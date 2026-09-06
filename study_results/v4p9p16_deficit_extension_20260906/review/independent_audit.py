#!/usr/bin/env python3
"""Read-only audit of saved deficit products; no fits or random draws."""
from pathlib import Path
import csv
import hashlib
import json
import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
from scipy.stats import beta, ks_2samp, norm

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PARENT = HERE.parent / 'v4p9p16_combined_global_20260906'
checked = 0
failures = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition):
    global checked
    checked += 1
    if not bool(condition):
        failures.append(name)


def close(left, right, probability=False):
    return np.allclose(left, right, rtol=1e-11,
                       atol=0 if probability else 1e-13)


def tail(values, thresholds):
    n = len(values)
    k = np.array([np.count_nonzero(values >= x) for x in thresholds])
    low = np.zeros(len(k))
    high = np.ones(len(k))
    upper = np.ones(len(k))
    positive = k > 0
    below = k < n
    low[positive] = beta.ppf(.025, k[positive], n-k[positive]+1)
    high[below] = beta.ppf(.975, k[below]+1, n-k[below])
    upper[below] = beta.ppf(.95, k[below]+1, n-k[below])
    return dict(k=k, n=np.full(len(k), n), p=k/n,
                low95=low, high95=high, upper95=upper)


def main():
    provenance = json.loads((HERE/'provenance/parent.json').read_text())
    summary = json.loads((HERE/'analysis/summary.json').read_text())
    check('parent_manifest_binding', sha(PARENT/'MANIFEST.csv') == provenance['manifest_sha256'])
    manifest = list(csv.DictReader((PARENT/'MANIFEST.csv').open()))
    for row in manifest:
        path = ROOT/row['path']
        check('parent_file:'+row['path'], path.stat().st_size == int(row['bytes'])
              and sha(path) == row['sha256'])
    for path, digest in summary['input_sha256'].items():
        check('input_hash:'+path, sha(ROOT/path) == digest)
    check('declared_no_new_fits_or_draws', summary['new_likelihood_fits'] == 0
          and summary['new_independent_toys'] == 0)
    check('declared_ensemble_sizes', summary['gp_realizations_reused'] == 200000
          and summary['direct_joint_scans_reused'] == 1000)
    source = np.load(PARENT/'global/scan_vectors.npz')
    covariance = np.load(PARENT/'global/analysis/covariance.npz')
    maxima = np.load(HERE/'analysis/deficit_maxima.npz')
    observed = pd.read_csv(PARENT/'global/observed.csv').set_index('mass_MeV')
    frame = pd.read_csv(HERE/'analysis/deficit_curves.csv', float_precision='round_trip')
    masses = source['masses_MeV']
    check('232_point_grid', np.array_equal(masses, np.arange(19, 251)))
    check('observed_order_matches_vector_order', np.array_equal(observed.index, masses))
    check('curve_row_count', len(frame) == 464)
    diagnostics = {}
    for method in ('profiled', 'fixed'):
        rows = frame[frame.method == method].set_index('mass_MeV')
        check(method+':curve_order', np.array_equal(rows.index, masses))
        check(method+':membership', np.array_equal(rows.dataset_set, observed.dataset_set))
        asimov = source['asimov_'+method]
        mean = asimov[0]
        response = asimov[1:]-mean
        C = response.T@response
        sd = np.sqrt(np.diag(C))
        K = C/np.outer(sd, sd)
        check(method+':unchanged_C', np.array_equal(C, covariance[method+'_C']))
        check(method+':unchanged_K', np.array_equal(K, covariance[method+'_K']))
        r = observed[method+'_r'].to_numpy()
        z = (r-mean)/sd
        score = np.where(r < 0, -z, -np.inf)
        local = np.where(r < 0, norm.cdf(z), 1.)
        reference = np.where(r < 0, norm.cdf(r), 1.)
        for column, values in [('observed_r', r), ('asimov_r', mean),
                               ('response_sd', sd), ('z_standardized', z)]:
            check(method+':'+column, close(rows[column], values))
        check(method+':local_probability', close(rows.p_local_deficit, local, True))
        check(method+':raw_gaussian_reference', close(rows.p_raw_gaussian, reference, True))
        valid = source['validation1000_'+method]
        valid_z = (valid-mean)/sd
        direct = np.where(valid < 0, -valid_z, -np.inf).max(axis=1)
        direct_raw = np.maximum(-valid, 0).max(axis=1)
        check(method+':direct_maxima', np.array_equal(direct, maxima[method+'_direct']))
        check(method+':direct_raw_maxima', np.array_equal(direct_raw, maxima[method+'_direct_raw']))
        gp = maxima[method+'_gp']
        gp_raw = maxima[method+'_gp_raw']
        check(method+':GP_dimensions', gp.shape == gp_raw.shape == (200000,))
        check(method+':direct_dimensions', direct.shape == direct_raw.shape == (1000,))
        check(method+':saved_positive_replay_gate', all(summary['checks'][method+x]
              for x in ('_positive_principal_bitwise_replay', '_positive_raw_bitwise_replay')))
        check(method+':finite_maxima', all(np.isfinite(x).all()
              for x in (gp, gp_raw, direct, direct_raw)))
        check(method+':nonnegative_raw_depth', np.all(gp_raw >= 0) and np.all(direct_raw >= 0))
        tails = {}
        for label, values, thresholds in [('gp', gp, score), ('direct', direct, score),
                ('raw_gp', gp_raw, np.maximum(-r, 0)),
                ('raw_direct', direct_raw, np.maximum(-r, 0))]:
            tails[label] = tail(values, thresholds)
            for field, expected in tails[label].items():
                check(method+':'+label+'_'+field,
                      close(rows[label+'_'+field], expected, field not in ('k', 'n')))
        check(method+':nonnegative_atom', np.all(rows.loc[r >= 0,
              ['p_local_deficit', 'p_raw_gaussian', 'gp_p', 'direct_p',
               'raw_gp_p', 'raw_direct_p']].to_numpy() == 1))
        count = tails['gp']['k']
        inclusion_bound = np.ones(len(count))
        unresolved = count < 200000
        inclusion_bound[unresolved] = beta.ppf(1-1e-7/len(masses),
              count[unresolved]+1, 200000-count[unresolved])
        check(method+':global_local_inclusion', np.all(inclusion_bound >= local))
        peak = int(np.argmax(score))
        raw_peak = int(np.argmax(np.maximum(-r, 0)))
        info = summary['methods'][method]
        check(method+':principal_peak', info['peak_mass_MeV'] == int(masses[peak]))
        check(method+':raw_peak', info['raw_ordering']['peak_mass_MeV'] == int(masses[raw_peak]))
        for label, destination, j in [('gp', info['gp_global'], peak),
                ('direct', info['direct_global'], peak),
                ('raw_gp', info['raw_ordering']['gp_global'], raw_peak),
                ('raw_direct', info['raw_ordering']['direct_global'], raw_peak)]:
            check(method+':summary_'+label, all(close(destination[k], v[j], True)
                  for k, v in tails[label].items()))
        for label, a, b in [('principal', gp, direct), ('raw_depth', gp_raw, direct_raw)]:
            ks = ks_2samp(a, b, method='asymp')
            saved = info[label+'_maximum_KS']
            check(method+':'+label+'_KS', close([ks.statistic, ks.pvalue],
                  [saved['statistic'], saved['pvalue']], True))
        diagnostics[method] = dict(principal_peak_MeV=int(masses[peak]),
              observed_r=float(r[peak]), asimov_r=float(mean[peak]),
              response_sd=float(sd[peak]), observed_z=float(z[peak]),
              local_deficit_p=float(local[peak]), raw_reference_p=float(reference[peak]),
              principal_gp_k=int(tails['gp']['k'][peak]),
              principal_direct_k=int(tails['direct']['k'][peak]),
              raw_peak_MeV=int(masses[raw_peak]), raw_depth=float(-r[raw_peak]),
              raw_gp_k=int(tails['raw_gp']['k'][raw_peak]),
              raw_direct_k=int(tails['raw_direct']['k'][raw_peak]),
              gp_principal_range=[float(gp.min()), float(gp.max())],
              direct_principal_range=[float(direct.min()), float(direct.max())],
              gp_raw_range=[float(gp_raw.min()), float(gp_raw.max())],
              direct_raw_range=[float(direct_raw.min()), float(direct_raw.max())],
              raw_negative_observed=int(np.count_nonzero(r < 0)))
    p = diagnostics['profiled']
    selected = sorted({30, 66, 76, 120, 220, p['principal_peak_MeV'], p['raw_peak_MeV']})
    check('representative_selection', selected == summary['representative_masses_MeV'])
    representatives = pd.read_csv(HERE/'analysis/representative_deficits.csv', float_precision='round_trip')
    check('representative_rows', representatives.equals(frame[(frame.method == 'profiled')
          & frame.mass_MeV.isin(selected)].reset_index(drop=True)))
    output = dict(passed=not failures, checked_conditions=checked, failures=failures,
          parent_manifest_files_checked=len(manifest), methods=diagnostics,
          scope='Saved products only; no likelihood fits or random generation. Direct maxima and all GP/direct counts independently reconstructed. GP generation reviewed in source; saved positive replay gates inspected, not rerun.',
          input_sha256={str(path.relative_to(ROOT)): sha(path) for path in
              [Path(__file__), HERE/'analyze_deficits.py', HERE/'analysis/summary.json',
               HERE/'analysis/deficit_curves.csv', HERE/'analysis/deficit_maxima.npz']})
    print(json.dumps(output, indent=2, allow_nan=False))
    raise SystemExit(0 if not failures else 1)


if __name__ == '__main__':
    main()
