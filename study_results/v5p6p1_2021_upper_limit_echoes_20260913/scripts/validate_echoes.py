"""Serial QA of the completed v5.6.1 scans. Do not run during scan_limits.py.

Checks the existing spectra and saved results; never generates another toy.
Only the ten existing closed one-bin solver checks perform additional fits.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import scan_limits as scan

B = Path(__file__).resolve().parents[1]
QA = B/'qa'
CHECKS, DETAILS = [], {}
LABELS = ['background_asimov', 'matched_asimov', 'yield_asimov'] + [f'toy_{i:02}' for i in range(20)]


def check(name, condition, **details):
    CHECKS.append(dict(check=name, passed=bool(condition), **details))


def close(a, b, rtol=1e-10, atol=1e-12):
    return bool(np.allclose(a, b, rtol=rtol, atol=atol))


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def finite_frame(frame):
    return bool(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all())


def validate_copies():
    records = load_json(B/'inputs/copy_manifest.json')
    for destination, record in records.items():
        path = B/destination
        check('copied_input:'+destination, path.exists() and scan.sha(path) == record['sha256'])
    parent_manifest = {}
    for line in (B/'inputs/parent/MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        parent_manifest[name.strip()] = digest
    references = load_json(B/'inputs/parent/fit_reference_manifest.json')['files']
    for record in references:
        path = B/record['snapshot']
        check('parent_reference:'+record['snapshot'], path.exists()
              and scan.sha(path) == record['sha256']
              and parent_manifest.get(record['parent_relative_path']) == record['sha256'])
    DETAILS['copied_input_count'] = len(records)
    DETAILS['compact_parent_reference_count'] = len(references)


def density_weights(masses, edges):
    widths = np.diff(edges)
    sigmas = np.array([scan.sigma('2021', float(m)) for m in masses])
    low, high = masses/1000 - 1.64*sigmas, masses/1000 + 1.64*sigmas
    overlap = np.maximum(0., np.minimum(edges[None, 1:], high[:, None])
                         - np.maximum(edges[None, :-1], low[:, None]))
    check('density_coverage:'+str(float(masses[0])), np.all(low >= edges[0])
          and np.all(high <= edges[-1])
          and close(overlap.sum(axis=1), high-low, rtol=1e-12, atol=1e-14))
    weights = overlap/widths[None, :]/(high-low)[:, None]
    # Independent unit convention: values per MeV become per GeV by x1000.
    mev_edges, mev_low, mev_high = edges*1000, low*1000, high*1000
    mev_overlap = np.maximum(0., np.minimum(mev_edges[None, 1:], mev_high[:, None])
                             - np.maximum(mev_edges[None, :-1], mev_low[:, None]))
    mev_weights = mev_overlap/np.diff(mev_edges)[None, :]/(mev_high-mev_low)[:, None]
    check('density_units:'+str(float(masses[0])), close(weights, 1000*mev_weights,
          rtol=2e-11, atol=1e-8))
    return weights


def closed_one_bin_checks():
    """Same ten independent analytic comparisons provided by the pinned solver."""
    def closed_nll(n, b, variance, signal):
        c = b + signal - variance
        lam = .5*(c + math.sqrt(c*c + 4*n*variance)) if variance else b+signal
        if variance and c < 0 and n > 0:
            lam = 2*n*variance/(math.sqrt(c*c + 4*n*variance)-c)
        poisson = lam-n+n*math.log(n/lam) if n > 0 else lam
        return poisson + ((lam-b-signal)**2/(2*variance) if variance else 0.)
    def independent_cls(q, qa):
        square = math.sqrt(qa)
        if q <= qa:
            zsb, zb = math.sqrt(q), math.sqrt(q)-square
        else:
            zsb, zb = (q+qa)/(2*square), (q-qa)/(2*square)
        return math.exp(norm.logsf(zsb)-norm.logsf(zb))
    rows = []
    for n in (0., 1., 60., 100., 200.):
        for sd in (0., 20.):
            b = 100.
            model = scan.OneSignalProfile([b], [[sd]] if sd else np.empty((1, 0)), [1.])
            result = model.limit([n], alpha=.1)
            denominator = 0. if n >= b else closed_nll(n, b, sd*sd, 0.)
            def root(a):
                q = max(0., 2*(closed_nll(n, b, sd*sd, a)-denominator))
                qa = 2*closed_nll(b, b, sd*sd, a)
                return independent_cls(q, qa)-.1
            expected = brentq(root, max(n-b, 0.)+1e-5, max(n-b, 0.)+500., xtol=1e-10)
            relative = abs(result['A90']/expected-1)
            check(f'closed_one_bin:n{n:g}:sd{sd:g}', relative <= 3e-7
                  and abs(result['Ahat']-(n-b)) <= 2e-5,
                  relative_A90_error=relative)
            rows.append(dict(n=n, sd=sd, A90=result['A90'], independent_A90=expected,
                             relative_error=relative, cls_branch=result['cls_branch']))
    save_json(QA/'closed_one_bin_validation.json', dict(rows=rows, new_toys=0))
    DETAILS['closed_one_bin_max_relative_error'] = max(row['relative_error'] for row in rows)


def main():
    catalogue = csv(B/'inputs/catalogue.csv')
    selected = csv(B/'inputs/selected_peaks.csv')
    scenarios = set(catalogue.scenario)
    directories = {path.name for path in (B/'derived/scans').iterdir() if path.is_dir()}
    check('scenario_inventory', len(catalogue) == 15 and catalogue.scenario.is_unique
          and scenarios == set(selected.scenario) and directories == scenarios
          and dict(catalogue.lane.value_counts()) == {'one': 9, 'ten': 6})
    check('345_spectrum_inventory', all(
        {path.stem for path in (B/'derived/scans'/sid).glob('*.csv')} == set(LABELS)
        and {path.stem for path in (B/'derived/scans'/sid).glob('*.json')} == set(LABELS)
        for sid in scenarios))
    if not all(row['passed'] for row in CHECKS):
        raise ValueError('Scans are incomplete; do not run numerical checks yet')
    validate_copies()
    protocol = load_json(B/'protocol.json')
    check('frozen_protocol', protocol['new_toys'] == 0 and protocol['CL'] == .9
          and protocol['workers'] == 1 and protocol['existing_toys_per_scenario'] == 20)
    with np.load(B/'inputs/lanes.npz', allow_pickle=False) as packed:
        lanes = {key: packed[key] for key in packed.files}
    edges = scan.DATA['2021']['edges']
    check('source_axes', np.array_equal(lanes['edges'], edges)
          and np.array_equal(lanes['x'], scan.DATA['2021']['x']))
    weights = {lane: density_weights(scan.masses(lane), edges) for lane in ('one', 'ten')}
    branching = csv(B/'inputs/parent/released_2021_asymptotic.csv').set_index('mass_MeV').dimuon_factor
    parent_comparisons, minima = [], []
    rows_total = 0
    maxima = dict(cls_root_error=0., score=0., monotonicity_error=0., prior_relative_error=0.,
                  covariance_load_over_poisson=0., fixed_ratio_relative_error=0.,
                  moving_ratio_relative_error=0., parent_Z_difference=0., parent_Ahat_sigma_difference=0.)
    minimum_background = minimum_lambda = float('inf')
    for row in catalogue.itertuples(index=False):
        sid, lane, mass0 = row.scenario, row.lane, float(row.mass_MeV)
        scenario_data, spectra = scan.spectra(row)
        expected_grid = scan.masses(lane)
        signature = scan.signature(row)
        check(sid+':parent_toy_identity', scan.sha(B/'inputs/toys'/f'{sid}.npz') == row.counts_sha256
              and scenario_data['counts'].shape == (20, len(edges)-1)
              and np.all(scenario_data['counts'] >= 0)
              and np.array_equal(scenario_data['counts'], np.floor(scenario_data['counts'])))
        parent_toys = csv(B/'inputs/parent/toy_fit_reference'/f'{sid}_toys.csv').set_index('toy')
        check(sid+':parent_toy_ids', len(parent_toys) == 20
              and set(parent_toys.index) == set(range(20)) and parent_toys.index.is_unique)
        reference_density = weights[lane] @ scenario_data['background']
        frad = float(scan.DATA['2021']['frad_effective'])
        reference_K = 3*np.pi*(expected_grid/1000)*frad*reference_density/(2/137.)
        branch = branching.loc[expected_grid].to_numpy()
        frames = {}
        for label, counts in spectra:
            path = B/'derived/scans'/sid/f'{label}.csv'
            frame, metadata = csv(path), load_json(path.with_suffix('.json'))
            frames[label] = frame
            check(f'{sid}:{label}:provenance', metadata['scenario'] == sid
                  and metadata['spectrum'] == label and metadata['dependencies'] == signature
                  and metadata['csv_sha256'] == scan.sha(path))
            check(f'{sid}:{label}:grid', len(frame) == len(expected_grid)
                  and metadata['rows'] == len(expected_grid)
                  and np.array_equal(frame.mass_MeV.to_numpy(), expected_grid)
                  and frame.mass_MeV.is_unique)
            check(f'{sid}:{label}:finite_and_positive', finite_frame(frame)
                  and np.all(frame.A90 > 0) and np.all(frame.sigma_A > 0)
                  and np.all(frame.min_background > 0) and np.all(frame.min_lambda > 0)
                  and np.all(frame.epsilon2_90 > 0) and np.all(frame.epsilon2_90_fixed_density > 0)
                  and np.all(frame.density_observed > 0) and np.all(frame.K_counts_per_epsilon2 > 0)
                  and np.all(frame.q_obs >= 0) and np.all(frame.q_asimov > 0)
                  and np.all(frame.Z0 >= 0) and np.all(frame.Ahat_bounded >= 0))
            check(f'{sid}:{label}:solver_acceptance', np.all(frame.ok == True)
                  and set(frame.status) == {'converged'} and np.all(np.abs(frame.cls-.1) <= 2e-6)
                  and np.all(frame.monotonicity_error <= 5e-5) and np.all(frame.max_score <= 3e-5)
                  and np.all(frame.nll_free <= frame.nll_null+2e-6)
                  and np.all(frame.prior_relative_error <= 1e-12))
            observed_density = weights[lane] @ counts
            conversion_K = 3*np.pi*(expected_grid/1000)*frad*observed_density/(2/137.)
            check(f'{sid}:{label}:density_and_branching', close(frame.density_observed, observed_density)
                  and close(frame.K_counts_per_epsilon2, conversion_K)
                  and close(frame.branching_factor, branch, rtol=1e-11)
                  and close(frame.epsilon2_90, frame.A90/conversion_K*branch)
                  and close(frame.epsilon2_90_fixed_density, frame.A90/reference_K*branch)
                  and close(frame.epsilon2_90*conversion_K/branch, frame.A90))
            fixed = frame[frame.mass_MeV == mass0]
            check(f'{sid}:{label}:injected_mass_present', len(fixed) == 1)
            if len(fixed) == 1 and (label == 'matched_asimov' or label.startswith('toy_')):
                point = fixed.iloc[0]
                if label == 'matched_asimov':
                    expected_z, expected_a = row.matched_asimov_Z, row.matched_asimov_Ahat
                else:
                    toy = parent_toys.loc[int(label[4:])]
                    expected_z, expected_a = toy.Z_fixed, toy.A_hat
                delta_z = float(point.Z0-expected_z)
                delta_a = float((point.Ahat-expected_a)/point.sigma_A)
                parent_comparisons.append(dict(scenario=sid, spectrum=label, mass_MeV=mass0,
                    previous_Z=float(expected_z), new_Z=float(point.Z0), delta_Z=delta_z,
                    delta_Ahat_over_sigma=delta_a))
                check(f'{sid}:{label}:parent_fixed_mass', abs(delta_z) <= 1e-5 and abs(delta_a) <= 2e-5,
                      delta_Z=delta_z, delta_Ahat_over_sigma=delta_a)
                maxima['parent_Z_difference'] = max(maxima['parent_Z_difference'], abs(delta_z))
                maxima['parent_Ahat_sigma_difference'] = max(maxima['parent_Ahat_sigma_difference'], abs(delta_a))
                if label == 'matched_asimov':
                    check(sid+':injection_target', abs(point.Z0-row.target_Z) <= 1e-5)
            rows_total += len(frame)
            maxima['cls_root_error'] = max(maxima['cls_root_error'], float(np.max(np.abs(frame.cls-.1))))
            for key, column in [('score', 'max_score'), ('monotonicity_error', 'monotonicity_error'),
                                ('prior_relative_error', 'prior_relative_error'),
                                ('covariance_load_over_poisson', 'cov_load_over_poisson')]:
                maxima[key] = max(maxima[key], float(frame[column].max()))
            minimum_background = min(minimum_background, float(frame.min_background.min()))
            minimum_lambda = min(minimum_lambda, float(frame.min_lambda.min()))
        background = frames['background_asimov']
        for label, frame in frames.items():
            ratio = frame.A90.to_numpy()/background.A90.to_numpy()
            fixed_ratio = frame.epsilon2_90_fixed_density.to_numpy()/background.epsilon2_90_fixed_density.to_numpy()
            moving_ratio = frame.epsilon2_90.to_numpy()/background.epsilon2_90.to_numpy()
            expected_moving = ratio*background.density_observed.to_numpy()/frame.density_observed.to_numpy()
            fixed_error = float(np.max(np.abs(fixed_ratio/ratio-1)))
            moving_error = float(np.max(np.abs(moving_ratio/expected_moving-1)))
            check(f'{sid}:{label}:echo_conversion_identity', fixed_error <= 2e-10 and moving_error <= 2e-10,
                  fixed_relative_error=fixed_error, per_spectrum_relative_error=moving_error)
            maxima['fixed_ratio_relative_error'] = max(maxima['fixed_ratio_relative_error'], fixed_error)
            maxima['moving_ratio_relative_error'] = max(maxima['moving_ratio_relative_error'], moving_error)
    check('total_limit_rows', rows_total == 68724, rows=rows_total)
    check('315_parent_fixed_mass_comparisons', len(parent_comparisons) == 315)
    pd.DataFrame(parent_comparisons).to_csv(QA/'parent_fixed_mass_comparison.csv', index=False,
                                           float_format='%.17g')
    DETAILS.update(scenarios=15, spectrum_files=345, limit_rows=rows_total, saved_toys=300,
        new_toys=0, max_errors_and_diagnostics=maxima, minimum_fitted_background=minimum_background,
        minimum_fitted_expectation=minimum_lambda, fixed_mass_comparisons=len(parent_comparisons),
        inference_scope='Conditional pseudo-data limit shapes; no coverage, rare-tail or global calibration.',
        density_scope='Per-spectrum pseudo-data density plus fixed-continuum alternative; only fixed-density ratios equal yield ratios.')
    if not all(row['passed'] for row in CHECKS):
        raise ValueError('Saved-result checks failed; closed one-bin checks were not run')
    closed_one_bin_checks()


if __name__ == '__main__':
    QA.mkdir(parents=True, exist_ok=True)
    error = None
    try:
        main()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        check('validation_completed', False, error=error)
    passed = bool(CHECKS) and all(row['passed'] for row in CHECKS)
    result = dict(passed=passed, checks_total=len(CHECKS),
                  checks_passed=sum(row['passed'] for row in CHECKS), checks=CHECKS, **DETAILS)
    save_json(QA/'numerical_validation.json', result)
    print(json.dumps(dict(passed=passed, checks_total=len(CHECKS),
                          checks_passed=result['checks_passed'], error=error)))
    raise SystemExit(0 if passed else 1)
