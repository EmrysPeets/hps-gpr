#!/usr/bin/env python3
"""Read-only probability/extraction audit; no likelihood fits or random draws."""
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
from scipy.stats import beta, ks_2samp, kstest, norm

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
GLOBAL = ROOT/'study_results/v4p9p16_combined_global_20260906'
EXTRACT = ROOT/'study_results/v4p9p16_presentation_extractions_20260906'
ECHO = ROOT/'study_results/v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/double_peak_injection'
checked = 0
failures = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition):
    global checked
    checked += 1
    if not bool(condition):
        failures.append(name)


def close(a, b, probability=False):
    return np.allclose(a, b, atol=0 if probability else 2e-13, rtol=2e-12)


def tails(values, thresholds):
    n = len(values)
    k = np.array([np.count_nonzero(values >= x) for x in thresholds])
    low = np.zeros(len(k)); high = np.ones(len(k)); upper = np.ones(len(k))
    positive = k > 0; below = k < n
    low[positive] = beta.ppf(.025, k[positive], n-k[positive]+1)
    high[below] = beta.ppf(.975, k[below]+1, n-k[below])
    upper[below] = beta.ppf(.95, k[below]+1, n-k[below])
    return dict(k=k, n=np.full(len(k), n), p=k/n, low=low, high=high, upper95=upper)


def main():
    manifest_counts = {}
    for parent in (GLOBAL, EXTRACT):
        rows = list(csv.DictReader((parent/'MANIFEST.csv').open()))
        manifest_counts[parent.name] = len(rows)
        for row in rows:
            path = ROOT/row['path']
            check('sealed:'+row['path'], path.stat().st_size == int(row['bytes']) and sha(path) == row['sha256'])
    folder = GLOBAL/'global'; analysis = folder/'analysis'
    vectors = np.load(folder/'scan_vectors.npz')
    covariance = np.load(analysis/'covariance.npz'); maxima = np.load(analysis/'maxima.npz')
    observed = pd.read_csv(folder/'observed.csv').set_index('mass_MeV')
    curves = pd.read_csv(analysis/'pvalue_curves.csv', float_precision='round_trip')
    summary = json.loads((analysis/'summary.json').read_text())
    for relative, digest in summary['input_sha256'].items():
        check('analysis_binding:'+relative, sha(ROOT/relative) == digest)
    masses = vectors['masses_MeV']
    check('full_grid', np.array_equal(masses, np.arange(19, 251)))
    check('observed_order', np.array_equal(observed.index, masses))
    check('all_probability_rows', len(curves) == 464 and not curves[['method', 'mass_MeV']].duplicated().any())
    max_root_delta = 0.
    for mass in masses:
        point = json.loads((folder/'points'/f'm{mass:03d}_qa.json').read_text())
        check('point_accepted:'+str(mass), point['passed'])
        check('point_membership:'+str(mass), point['active_datasets'] == observed.loc[mass, 'dataset_set'].split('+'))
        for method in ('profiled', 'fixed'):
            delta = abs(point['observed'][method+'_r']-observed.loc[mass, method+'_r'])
            max_root_delta = max(max_root_delta, delta)
            check('original_root:'+method+':'+str(mass), delta < 1e-13)
    results = {}
    for method in ('profiled', 'fixed'):
        rows = curves[curves.method == method].set_index('mass_MeV')
        check(method+':full_curve', np.array_equal(rows.index, masses))
        asimov = vectors['asimov_'+method]; mean = asimov[0]; response = asimov[1:]-mean
        C = response.T@response; sd = np.sqrt(np.diag(C)); K = C/np.outer(sd, sd)
        check(method+':C', np.array_equal(C, covariance[method+'_C']))
        check(method+':K', np.array_equal(K, covariance[method+'_K']))
        check(method+':response', np.array_equal(response, covariance[method+'_response']))
        check(method+':positive_width', np.all(sd > 0))
        r = observed[method+'_r'].to_numpy(); z = (r-mean)/sd
        local = np.where(r > 0, norm.sf(z), 1.)
        asymptotic = norm.sf(np.maximum(r, 0.))
        score = np.where(r > 0, z, -np.inf)
        for column, expected in [('observed_r', r), ('asimov_r', mean), ('response_sd', sd), ('z_standardized', z)]:
            check(method+':'+column, close(rows[column], expected))
        check(method+':Gaussian_local', close(rows.p_local_common_truth, local, True))
        check(method+':asymptotic_reference', close(rows.p_asymptotic, asymptotic, True))
        check(method+':nonpositive_atom', np.all(rows.loc[r <= 0, ['p_local_common_truth', 'p_global_gp', 'p_global_direct']].to_numpy() == 1))
        check(method+':asymptotic_zero_convention', np.all(rows.loc[r <= 0, 'p_asymptotic'] == .5))
        valid = vectors['validation1000_'+method]; zv = (valid-mean)/sd
        direct = np.where(valid > 0, zv, -np.inf).max(axis=1)
        direct_raw = np.maximum(valid, 0).max(axis=1)
        gp = maxima[method+'_gp']; gp_raw = maxima[method+'_gp_raw']
        check(method+':direct_maxima', np.array_equal(direct, maxima[method+'_direct']))
        check(method+':raw_direct_maxima', np.array_equal(direct_raw, maxima[method+'_direct_raw']))
        check(method+':ensemble_sizes', gp.shape == gp_raw.shape == (200000,) and direct.shape == (1000,))
        all_tails = {label:tails(values, thresholds) for label, values, thresholds in
              [('gp', gp, score), ('direct', direct, score), ('raw', gp_raw, np.maximum(r, 0.))]}
        for label, item in all_tails.items():
            if label == 'raw':
                columns = dict(k='raw_gp_k', p='p_global_raw_ordering', low='raw_gp_low', high='raw_gp_high', upper95='raw_gp_upper95')
            else:
                columns = dict(k=label+'_k', n=label+'_n', p='p_global_'+label,
                      low='p_global_'+label+'_low', high='p_global_'+label+'_high', upper95='p_global_'+label+'_upper95')
            for key, column in columns.items():
                check(method+':'+column, close(rows[column], item[key], key not in ('k', 'n')))
        bound = np.ones(len(masses)); k = all_tails['gp']['k']; below = k < 200000
        bound[below] = beta.ppf(1-1e-7/len(masses), k[below]+1, 200000-k[below])
        check(method+':global_local_inclusion', np.all(bound >= local))
        ks = {}
        for label, first, second in [('principal', gp, direct), ('raw', gp_raw, direct_raw)]:
            test = ks_2samp(first, second, method='asymp'); saved = summary['methods'][method][label+'_maximum_KS']
            check(method+':'+label+'_KS', close([test.statistic, test.pvalue], [saved['statistic'], saved['pvalue']], True))
            ks[label] = dict(distance=float(test.statistic), nominal_p=float(test.pvalue))
        normal_p = np.array([kstest(zv[:, j], 'norm').pvalue for j in range(len(masses))])
        order = np.argsort(normal_p); adjusted = np.minimum(1., np.maximum.accumulate((len(order)-np.arange(len(order)))*normal_p[order]))
        flags = int(np.count_nonzero(adjusted < .05))
        check(method+':Holm', flags == summary['methods'][method]['marginal_normality_holm_flags'])
        selected_mass = int(masses[np.argmax(score)]); raw_mass = int(masses[np.argmax(r)])
        check(method+':principal_peak', selected_mass == summary['methods'][method]['peak_mass_MeV'])
        check(method+':raw_peak', raw_mass == summary['methods'][method]['raw_ordering']['peak_mass_MeV'])
        examples = []
        for mass in (21, 41, 65, 66, 72, 74, 75, 76, 77, 78, 79, 83, 90, 91, 92, 93):
            j = int(mass-19)
            examples.append(dict(mass_MeV=mass, raw_r=float(r[j]), asimov_r=float(mean[j]), sd=float(sd[j]),
                  standardized_r=float(z[j]), raw_asymptotic_p=float(asymptotic[j]), Gaussian_stress_local_p=float(local[j]),
                  GP_global_k=int(all_tails['gp']['k'][j]), direct_global_k=int(all_tails['direct']['k'][j]),
                  raw_GP_global_k=int(all_tails['raw']['k'][j])))
        results[method] = dict(n_masses=len(masses), raw_positive_points=int(np.count_nonzero(r > 0)),
              raw_nonpositive_points=int(np.count_nonzero(r <= 0)),
              gate_crossings=int(np.count_nonzero((r[1:] > 0) != (r[:-1] > 0))),
              GP_zero_masses=masses[all_tails['gp']['k'] == 0].tolist(),
              direct_zero_masses=masses[all_tails['direct']['k'] == 0].tolist(),
              local_values_below_original_display_floor=masses[local < 1e-8].tolist(),
              GP_maximum_range=[float(gp.min()), float(gp.max())], direct_maximum_range=[float(direct.min()), float(direct.max())],
              raw_GP_maximum_range=[float(gp_raw.min()), float(gp_raw.max())], raw_direct_maximum_range=[float(direct_raw.min()), float(direct_raw.max())],
              raw_global_curve_all_one=bool(np.all(rows.p_global_raw_ordering == 1)),
              centered_principal_peak_MeV=selected_mass, raw_peak_MeV=raw_mass, marginal_Holm_flags=flags,
              maximum_KS=ks, examples=examples)
    extracts = json.loads((EXTRACT/'derived/fit_closure.json').read_text())['checks']
    dense = pd.read_csv(ROOT/'study_results/v4p9p13_calibration_20260905/summary/observed_calibrated_limits.csv').set_index(['scope_key', 'mass_MeV'])
    extraction_max_error = 0.
    for fit in extracts:
        if fit['group'] == 'combined':
            reference = observed.loc[fit['mass_MeV'], 'profiled_r']
        else:
            scope = 'individual_'+fit['group']+('_10pct' if fit['group'] == '2021' else '_full')
            reference = dense.loc[(scope, fit['mass_MeV']), 'signed_r_profiled_asymptotic']
        error = abs(fit['root']-reference); extraction_max_error = max(extraction_max_error, error)
        check('extraction_root:'+fit['fit_id'], error < 1e-13)
    # Recheck saved deterministic echo arithmetic; these are not new fits.
    echo = pd.read_csv(ECHO/'derived/deterministic_scans.csv', float_precision='round_trip').pivot(index='mass_MeV', columns='lane', values='signed_r')
    changes = pd.read_csv(ECHO/'derived/injection_induced_changes.csv', float_precision='round_trip').set_index('mass_MeV')
    for lane in ('inject_65', 'inject_78', 'double_full', 'double_half'):
        check('echo_change:'+lane, close(echo[lane]-echo.background, changes[lane]))
    summed = changes.inject_65+changes.inject_78
    nonadditivity = changes.double_full-summed
    check('echo_additivity_identity', close(summed, changes.individual_changes_sum) and close(nonadditivity, changes.nonadditivity))
    echo_summary = json.loads((ECHO/'derived/summary.json').read_text())
    check('echo_no_new_toys', echo_summary['new_toys'] == 0 and not echo_summary['newly_unblinded_data'])
    check('echo_max_nonadditivity', close(abs(nonadditivity).max(), echo_summary['max_abs_response_nonadditivity']))
    legacy = pd.read_csv(ECHO/'derived/observed_reconstruction.csv').set_index('mass_MeV')
    latest = dense.loc['individual_2021_10pct', 'signed_r_profiled_asymptotic']
    legacy_delta = legacy.saved_signed_r-latest.loc[legacy.index]
    output = dict(passed=not failures, checked_conditions=checked, failures=failures,
          manifest_files_checked=manifest_counts, original_checkpoint_to_CSV_root_max_error=max_root_delta,
          extraction_root_max_error=extraction_max_error, selected_extraction_fits_checked=len(extracts), methods=results,
          MC_zero_upper95=dict(GP_200000=float(beta.ppf(.95, 1, 200000)), direct_1000=float(beta.ppf(.95, 1, 1000))),
          echo=dict(at_71_MeV={key:float(value) for key, value in echo.loc[71].items()},
              delta_65_at_71=float(changes.loc[71, 'inject_65']), delta_78_at_71=float(changes.loc[71, 'inject_78']),
              double_delta_at_71=float(changes.loc[71, 'double_full']),
              maximum_abs_nonadditivity=float(abs(nonadditivity).max()),
              legacy_vs_dense_observed_root_max_delta=float(abs(legacy_delta).max()),
              largest_legacy_difference_mass_MeV=int(abs(legacy_delta).idxmax()),
              interpretation='Archived deterministic mechanism study on one different smooth truth, not signal evidence or new calibrated probabilities.'),
          scope='Saved artifacts only. GP/direct exceedance counts independently recomputed from stored maxima; direct maxima and covariance recomputed from saved scan vectors. No GP fields or toys generated.',
          input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), GLOBAL/'analyze_combined.py',
              GLOBAL/'make_figures.py', GLOBAL/'MANIFEST.csv', EXTRACT/'MANIFEST.csv', folder/'observed.csv',
              folder/'scan_vectors.npz', analysis/'pvalue_curves.csv', analysis/'covariance.npz', analysis/'maxima.npz',
              EXTRACT/'derived/fit_closure.json', ECHO/'derived/deterministic_scans.csv', ECHO/'derived/injection_induced_changes.csv']})
    print(json.dumps(output, indent=2, allow_nan=False))
    raise SystemExit(0 if not failures else 1)


if __name__ == '__main__':
    main()
