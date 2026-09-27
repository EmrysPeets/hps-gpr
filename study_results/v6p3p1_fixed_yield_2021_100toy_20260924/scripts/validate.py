#!/usr/bin/env python3
"""Independent replay/identity checks; never runs production fits or changes inputs.

Scientific bias, pull width and containment are measured outcomes, not validation
gates. This checker reconstructs RNG draws and templates without importing core.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize

B = Path(__file__).resolve().parents[1]
MASSES = tuple(range(60, 241, 20))
SHAPES = ('gaussian', 'mc')
LEVELS = (0, 1, 3, 5)
MASTER = 63220260924
HANDOFF_SHA = '68358b6d4a24b83ddb25617f82064c4a3fc1f5bce4b1f7772a9620a58ff94ad8'
NULL_SHA = '306a915b9ba6230aafbe058c94438c6af11f85b60d68f5aae4ebbfec4d4a9424'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ahash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def close(actual, expected, name, rtol=2e-12, atol=2e-12):
    require(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False),
            f'{name}: numerical mismatch')


def expected_rng(namespace, mass, toy, shape, level):
    return np.random.default_rng(np.random.SeedSequence(
        [MASTER, namespace, mass, toy, shape, level]))


def independent_templates():
    """CDF differences directly from pinned coefficients/histograms."""
    data = dict(np.load(B/'inputs/v6p1/inputs/spectrum_2021.npz'))
    edges, centers = data['edges'], data['x']
    result = {}
    for m in MASSES:
        sigma = np.polynomial.polynomial.polyval(m/1000., data['sigma_coeffs'])
        mask = (centers >= m/1000.-2.25*sigma) & (centers <= m/1000.+2.25*sigma)
        require(np.sum(centers < m/1000.-2.25*sigma) >= 3 and
                np.sum(centers > m/1000.+2.25*sigma) >= 3, f'sideband extent {m}')
        gauss_cdf = ndtr((edges-m/1000.)/sigma)
        hist = dict(np.load(B/f'inputs/v6p1/histograms/m{m:03d}.npz'))
        meta = json.loads(str(hist['metadata']))
        require(np.array_equal(hist['sumw'], hist['sumw2']), f'unit MC weights {m}')
        require(meta['stats']['underflow'] == 0, f'unsupported native underflow {m}')
        hist_cdf = np.r_[0., np.cumsum(hist['sumw'])]/float(meta['sumw'])
        mc_cdf = np.interp(edges, hist['edges_GeV'], hist_cdf,
                           left=0., right=hist_cdf[-1])
        for shape, cdf in [('gaussian', gauss_cdf), ('mc', mc_cdf)]:
            categories = np.r_[cdf[0], np.diff(cdf), 1.-cdf[-1]]
            require(np.min(categories) >= 0., f'nonnegative categories {m} {shape}')
            close(categories.sum(), 1., f'category sum {m} {shape}', atol=2e-15)
            result[(m, shape)] = dict(categories=categories, probability=categories[1:-1],
                                      mask=mask, sigma=sigma)
    return data, result


class Audit:
    def __init__(self):
        self.checks = []

    def run(self, name, fn):
        try:
            details = fn()
            self.checks.append(dict(name=name, passed=True, details=details))
        except Exception as exc:
            self.checks.append(dict(name=name, passed=False,
                                   error=f'{type(exc).__name__}: {exc}',
                                   traceback=traceback.format_exc(limit=3)))
        print(f'{name}: {"PASS" if self.checks[-1]["passed"] else "FAIL"}', flush=True)


def check_inputs():
    hashes = readj(B/'provenance/input_hashes.json')
    require(len({r['path'] for r in hashes}) == len(hashes), 'duplicate input manifest paths')
    for row in hashes:
        require(sha(B/row['path']) == row['sha256'], f'input hash {row["path"]}')
    require(sha(B/'inputs/null_2021.npz') == NULL_SHA, 'known null input SHA')
    require(sha(B/'provenance/HANDOFF.md') == HANDOFF_SHA, 'final handoff SHA')
    null = dict(np.load(B/'inputs/null_2021.npz'))
    data, templates = independent_templates()
    require(np.array_equal(null['edges_GeV'], data['edges']), 'source edges identity')
    require(np.array_equal(null['observed'], data['n']), 'observed release identity')
    require(np.all(null['truth'] > 0) and np.all(np.isfinite(null['truth'])), 'positive finite truth')
    original = B.parents[1]/'docs/2021_10pct_fixed_yield_100toy_handoff.md'
    original_checked = original.exists()
    if original_checked:
        require(sha(original) == HANDOFF_SHA, 'original handoff preserved')
    original_root = B.parent/'v6p2_mc_injection_20260923'
    source_checks = 0
    if original_root.exists():
        for row in hashes:
            if row['path'].startswith('inputs/'):
                source = original_root/row['path']
            elif row['path'] == 'provenance/v62_injection_core.py':
                source = original_root/'scripts/injection_core.py'
            elif row['path'] == 'provenance/v62_protocol.json':
                source = original_root/'protocol.json'
            elif row['path'] == 'provenance/v63_design_report.tex':
                source = B.parent/'v6p3_injection_design_20260923/source/report.tex'
            else:
                continue
            require(sha(source) == row['sha256'], f'prior study source preserved {source}')
            source_checks += 1
    return dict(pinned_files=len(hashes), original_handoff_checked=original_checked,
                prior_study_source_files_checked=source_checks,
                input_bins=len(data['x']), support_GeV=[float(data['edges'][0]), float(data['edges'][-1])],
                template_fractions=[dict(mass_MeV=m, shape=s,
                    support=float(t['probability'].sum()), window=float(t['probability'][t['mask']].sum()))
                    for (m,s),t in templates.items()])


def check_solver_analytic():
    """A one-bin exact result checks signed units and nuisance-free profiling."""
    sys.path.insert(0, str(B/'inputs/v6p1/scripts'))
    from limit_solver import OneSignalProfile
    b, weight = 113., .37
    details = []
    for variance in (0., 37.):
        factor = np.zeros((1, 0)) if variance == 0 else np.array([[np.sqrt(variance)]])
        for observed in (90., 140.):
            model = OneSignalProfile([b], factor, [weight])
            fit = model.fit([observed])
            close(fit['A'], (observed-b)/weight, 'analytic signed amplitude', atol=3e-5)
            close(fit['sigma'], np.sqrt(observed+variance)/weight,
                  'analytic observed-Hessian sigma', atol=3e-5)
            profile = model.fit([observed], fixed=17., initial=fit['theta'])
            base = b+17.*weight
            lam = base if variance == 0 else .5*(base-variance+
                np.sqrt((base-variance)**2+4*variance*observed))
            penalty = 0. if variance == 0 else (lam-base)**2/(2*variance)
            independent_q = 2.*(lam-observed+observed*np.log(observed/lam)+penalty)
            close(2*(profile['nll']-fit['nll']), independent_q,
                  'analytic fixed-truth q', atol=3e-10)
            require(fit['score'] < 3e-5 and fit['min_lambda'] > 0, 'analytic solver convergence')
            details.append(dict(observed=observed, background_variance=variance,
                                signed_Ahat=float(fit['A']), q_true=float(independent_q)))
    return details


def expected_signature():
    rows = readj(B/'provenance/input_hashes.json')
    parts = [(r['path'], r['sha256']) for r in rows]
    for name in ('scripts/core.py', 'scripts/run_study.py', 'protocol.json',
                 'inputs/cohorts.npz', 'inputs/templates.npz'):
        parts.append((name, sha(B/name)))
    return hashlib.sha256(json.dumps(sorted(parts)).encode()).hexdigest()


def boolcol(frame, name):
    require(name in frame, f'missing column {name}')
    values = frame[name].astype(str).str.lower()
    require(values.isin(['true', 'false']).all(), f'invalid booleans {name}')
    return values == 'true'


def readcsv(path):
    return pd.read_csv(path, keep_default_na=False, float_precision='round_trip')


def load_rows(cohort):
    paths = sorted((B/f'results/{cohort}').glob('m*_t*.csv'))
    require(bool(paths), f'{cohort} checkpoints absent')
    return pd.concat([readcsv(p) for p in paths], ignore_index=True), paths


def check_cohorts():
    saved = dict(np.load(B/'inputs/cohorts.npz'))
    truth = np.load(B/'inputs/null_2021.npz')['truth']
    hashes = []
    for cohort, namespace in [('pilot', 1), ('evaluation', 2)]:
        counts = saved[cohort]
        require(counts.shape == (100, len(truth)), f'{cohort} shape')
        require(np.issubdtype(counts.dtype, np.integer) and np.all(counts >= 0), f'{cohort} integer counts')
        for toy in range(100):
            replay = expected_rng(namespace, 0, toy, 0, 0).poisson(truth)
            require(np.array_equal(counts[toy], replay), f'{cohort} background RNG replay {toy}')
            hashes.append(ahash(counts[toy]))
    require(len(set(hashes)) == 200, 'all 200 spectra unique across pilot/evaluation')
    return dict(background_spectra=200, exact_rng_replay=True,
                independence_evidence='Distinct SeedSequence namespaces; exact deterministic replay; 200 distinct spectra.')


def check_templates():
    data, independently_derived = independent_templates()
    saved = dict(np.load(B/'inputs/templates.npz'))
    require(np.array_equal(saved['masses_MeV'], MASSES), 'saved mass grid')
    require(list(saved['shapes']) == list(SHAPES), 'saved shape IDs')
    require(np.array_equal(saved['edges_GeV'], data['edges']), 'template saved edges')
    require(saved['categories'].shape == (10, 2, len(data['x'])+2), 'template category shape')
    require(saved['masks'].shape == (10, len(data['x'])), 'template mask shape')
    fractions = readcsv(B/'inputs/template_fractions.csv')
    require(len(fractions) == 20 and not fractions.duplicated(['mass_MeV', 'shape']).any(), 'fraction table IDs')
    max_difference = 0.
    for mi, m in enumerate(MASSES):
        for si, shape in enumerate(SHAPES):
            expected = independently_derived[(m, shape)]
            categories = saved['categories'][mi, si]
            mask = saved['masks'][mi]
            max_difference = max(max_difference, float(np.max(np.abs(categories-expected['categories']))))
            close(categories, expected['categories'], f'independent template CDF {m} {shape}', atol=3e-15)
            require(np.array_equal(mask, expected['mask']), f'pole window mask {m} {shape}')
            require(np.all(categories >= 0) and abs(categories.sum()-1) < 1e-12, 'saved probabilities closure')
            row = fractions[(fractions.mass_MeV == m) & (fractions['shape'] == shape)].iloc[0]
            for key, value in [('support_fraction', categories[1:-1].sum()),
                               ('window_fraction', categories[1:-1][mask].sum()),
                               ('training_fraction', categories[1:-1][~mask].sum()),
                               ('below_support_fraction', categories[0]), ('above_support_fraction', categories[-1])]:
                close(row[key], value, f'fraction {key} {m} {shape}')
            require(row.template_hash == ahash(categories) and row.mask_hash == ahash(mask), 'template fraction hashes')
    return dict(templates=20, masks=10, maximum_independent_CDF_difference=max_difference,
                convention='Full-selected count probabilities; no normalization on support or fitted window.')


def check_reference():
    ref = readj(B/'pilot_reference.json')
    require(ref['signature'] == expected_signature(), 'frozen reference signature')
    for field, name in [('pilot_rows_sha256', 'results/pilot_rows.csv'),
                        ('cohort_sha256', 'inputs/cohorts.npz'),
                        ('template_sha256', 'inputs/templates.npz'),
                        ('protocol_sha256', 'protocol.json')]:
        require(ref[field] == sha(B/name), f'reference {field}')
    pin = readj(B/'provenance/pilot_reference.sha256.json')
    require(pin['sha256'] == sha(B/'pilot_reference.json') and pin['frozen_utc'] == ref['frozen_utc'], 'freeze artifact hash/time')
    pilot, _ = load_rows('pilot')
    require(set(ref['masses']) == {str(m) for m in MASSES}, 'all pilot reference masses')
    scales = {}
    for m in MASSES:
        cell = ref['masses'][str(m)]
        stats = {}
        for shape in SHAPES:
            rows = pilot[(pilot.mass_MeV == m) & (pilot['shape'] == shape)]
            require(len(rows) == 100 and set(rows.toy) == set(range(100)), f'pilot IDs {m} {shape}')
            good = rows[boolcol(rows, 'fit_valid')]
            if shape == 'gaussian':
                require(len(good) == 100, f'100 valid Gaussian pilot errors {m}')
            supplied = cell['shape_errors'][shape]
            require(supplied['attempted'] == 100 and supplied['fit_valid'] == len(good), 'pilot error accounting')
            if len(good):
                values = np.asarray(good.sigma_postfit, float)
                require(np.all(np.isfinite(values)) and np.all(values > 0), 'valid pilot errors')
                stats[shape] = float(values.mean())
                close(supplied['mean_sigma'], values.mean(), f'pilot mean returned error {m} {shape}')
                if len(values) > 1:
                    sd = np.std(values, ddof=1)
                    close(supplied['sd_sigma'], sd, 'pilot error sample SD')
                    close(supplied['cv_sigma'], sd/values.mean(), 'pilot error CV')
                    close(supplied['se_mean_sigma'], sd/np.sqrt(len(values)), 'pilot mean error SE')
            else:
                require(supplied['mean_sigma'] is None, 'missing MC pilot uncertainty explicit')
        close(cell['s0'], stats['gaussian'], f'common Gaussian reference {m}')
        if 'mc' in stats:
            close(cell['mc_gaussian_mean_error_ratio'], stats['mc']/stats['gaussian'], 'MC/Gaussian pilot error ratio')
        for z in LEVELS:
            close(cell['expected_yields'][str(z)], z*stats['gaussian'], 'frozen common expected yield')
        scales[str(m)] = cell['s0']
    return dict(frozen_scales=scales, pilot_reference_sha256=sha(B/'pilot_reference.json'),
                uncertainty='Mean of returned Gaussian pilot Hessian errors; sample SD/sqrt(100) recorded separately.')


def check_row_replay():
    cohorts = dict(np.load(B/'inputs/cohorts.npz'))
    saved = dict(np.load(B/'inputs/templates.npz'))
    ref = readj(B/'pilot_reference.json')
    data, independently_derived = independent_templates()
    drawn_total = 0
    shared_null_gp_hashes = 0
    totals = {cohort: 0 for cohort in ('pilot', 'evaluation')}
    for cohort in ('pilot', 'evaluation'):
        frame, paths = load_rows(cohort)
        null_gp = {}
        for row in frame.itertuples(index=False):
            r = row._asdict()
            m, shape, z, toy = int(r['mass_MeV']), r['shape'], int(r['z']), int(r['toy'])
            mi, si = MASSES.index(m), SHAPES.index(shape)
            categories, mask = saved['categories'][mi, si], saved['masks'][mi]
            p = categories[1:-1]
            background = cohorts[cohort][toy]
            expected = 0. if cohort == 'pilot' else z*float(ref['masses'][str(m)]['s0'])
            close(r['A_expected'], expected, f'fixed common expected yield {r["row_id"]}', atol=0.)
            if cohort == 'evaluation':
                close(r['s0'], ref['masses'][str(m)]['s0'], 'frozen row scale')
            require(json.loads(r['background_seed_key']) == [MASTER, 1 if cohort == 'pilot' else 2, 0, toy, 0, 0],
                    f'background seed key {r["row_id"]}')
            if z:
                key = [MASTER, 3, m, toy, si+1, LEVELS.index(z)]
                require(json.loads(r['signal_seed_key']) == key, f'signal seed key {r["row_id"]}')
                draw = expected_rng(3, m, toy, si+1, LEVELS.index(z)).poisson(expected*categories)
                drawn_total += 1
            else:
                require(r['signal_seed_key'] == '', 'null has no signal RNG key')
                draw = np.zeros(len(background)+2, dtype=np.int64)
            counts = background+draw[1:-1]
            for key, array in [('background_hash', background), ('counts_hash', counts),
                               ('signal_draw_hash', draw), ('template_hash', categories), ('mask_hash', mask)]:
                require(r[key] == ahash(array), f'{key} {r["row_id"]}')
            regions = dict(actual_full=draw.sum(), actual_support=draw[1:-1].sum(),
                actual_window=draw[1:-1][mask].sum(), actual_training=draw[1:-1][~mask].sum(),
                actual_outside_support=draw[0]+draw[-1], actual_below_support=draw[0], actual_above_support=draw[-1])
            for key, value in regions.items():
                require(int(r[key]) == int(value), f'{key} {r["row_id"]}')
            for key, value in [('support_fraction', p.sum()), ('window_fraction', p[mask].sum()),
                               ('training_fraction', p[~mask].sum())]:
                close(r[key], value, f'row {key}')
            sigma = independently_derived[(m, shape)]['sigma']
            close(r['nominal_sigma_MeV'], 1000*sigma, 'row nominal resolution')
            close(r['window_low_MeV'], m-2250*sigma, 'window low')
            close(r['window_high_MeV'], m+2250*sigma, 'window high')
            require(int(r['fit_bins']) == int(mask.sum()), 'fit-bin count')
            index = int(np.flatnonzero(data['masses'] == m)[0])
            close(r['kernel_const'], data['const'][index], 'archived kernel constant')
            close(r['kernel_ls'], data['ls'][index], 'archived kernel lengthscale')
            if z == 0 and str(r['fit_valid']).lower() == 'true':
                key = (m, toy)
                gp_pair = (r['gp_mean_hash'], r['gp_covariance_factor_hash'])
                if key in null_gp:
                    require(null_gp[key] == gp_pair, 'same-spectrum/mask GP shared across templates')
                    shared_null_gp_hashes += 1
                else:
                    null_gp[key] = gp_pair
            totals[cohort] += 1
        if cohort == 'evaluation':
            for csv in paths:
                counts = dict(np.load(csv.with_suffix('.npz')))
                rows = readcsv(csv)
                require(counts['draws'].shape == (len(rows), len(data['x'])+2), 'saved signal draws shape')
                require(np.issubdtype(counts['draws'].dtype, np.integer) and np.all(counts['draws'] >= 0), 'saved signal integers')
                require(len(counts['keys']) == len(rows), 'signal draw key count')
                lookup = {(int(t), int(s), int(z)): d for (t,s,z),d in zip(counts['keys'], counts['draws'])}
                require(len(lookup) == len(rows), 'unique saved signal draw keys')
                for r in rows.itertuples(index=False):
                    key = (int(r.toy), SHAPES.index(r.shape)+1, int(r.z))
                    require(ahash(lookup[key]) == r.signal_draw_hash, 'saved draw hash matches replayed row')
                mi = MASSES.index(int(rows.mass_MeV.iloc[0]))
                require(np.array_equal(counts['mask'], saved['masks'][mi]), 'saved chunk mask')
    return dict(replayed_rows=totals, positive_signal_rng_draws=drawn_total,
                paired_null_GP_predictions=shared_null_gp_hashes,
                expected_truth_separate_from_realized_counts=True,
                signal_replay_note='Replay uses saved category vectors after independent native-CDF agreement; byte-level RNG streams reproduced.')


def check_checkpoints():
    signature = expected_signature()
    reference = readj(B/'pilot_reference.json')
    ref_sha = sha(B/'pilot_reference.json')
    # The exact reference timestamp field is resolved once the launcher schema is frozen.
    frozen = reference.get('frozen_utc', reference.get('created_utc'))
    require(frozen is not None, 'pilot reference has freeze timestamp')
    freeze_time = dt.datetime.fromisoformat(frozen)
    checkpoint_count = 0
    eval_starts = []
    coverage = {cohort: {m: [] for m in MASSES} for cohort in ('pilot', 'evaluation')}
    for cohort in ('pilot', 'evaluation'):
        markers = sorted((B/f'results/{cohort}').glob('m*_t*.json'))
        require(bool(markers), f'{cohort} checkpoint markers absent')
        for path in markers:
            marker = readj(path)
            require(marker['complete'] is True, f'checkpoint complete {path.name}')
            require(marker['cohort'] == cohort, f'checkpoint cohort {path.name}')
            require(marker['signature'] == signature, f'checkpoint signature {path.name}')
            require(marker['mass_MeV'] in MASSES, f'checkpoint mass {path.name}')
            require(0 <= marker['start'] < marker['stop'] <= 100, f'checkpoint toy bounds {path.name}')
            expected_rows = (marker['stop']-marker['start'])*(2 if cohort == 'pilot' else 8)
            require(marker['rows'] == expected_rows, f'checkpoint row count {path.name}')
            require(len(marker['output_hashes']) == (1 if cohort == 'pilot' else 2), f'checkpoint output count {path.name}')
            for name, digest in marker['output_hashes'].items():
                require(sha(B/name) == digest, f'checkpoint output hash {name}')
            frame = readcsv(path.with_suffix('.csv'))
            require(len(frame) == expected_rows, f'actual checkpoint rows {path.name}')
            require(set(frame.mass_MeV) == {marker['mass_MeV']} and set(frame.cohort) == {cohort},
                    f'checkpoint row identity {path.name}')
            require(set(frame.toy) == set(range(marker['start'], marker['stop'])), f'checkpoint toy IDs {path.name}')
            completed = dt.datetime.fromisoformat(marker['completed_utc'])
            started = dt.datetime.fromisoformat(marker['started_utc'])
            require(started <= completed, f'checkpoint timestamps {path.name}')
            if cohort == 'evaluation':
                require(marker['reference_sha256'] == ref_sha, f'frozen reference checksum {path.name}')
                require(freeze_time <= started, f'reference frozen before evaluation {path.name}')
                eval_starts.append(started)
            else:
                require(completed <= freeze_time, f'pilot completed before freeze {path.name}')
            coverage[cohort][marker['mass_MeV']].extend(range(marker['start'], marker['stop']))
            checkpoint_count += 1
    for cohort in coverage:
        for mass, toys in coverage[cohort].items():
            require(sorted(toys) == list(range(100)), f'unique full checkpoint toy coverage {cohort} {mass}')
    return dict(checkpoints=checkpoint_count, dependency_signature=signature,
                pilot_reference_sha256=ref_sha, pilot_frozen_utc=frozen,
                first_evaluation_started_utc=min(eval_starts).isoformat(), exact_cache_eligibility_recomputed=True)


def check_aggregate_rows():
    details = {}
    keys = ['mass_MeV', 'shape', 'z', 'toy']
    for cohort in ('pilot', 'evaluation'):
        chunks, _ = load_rows(cohort)
        combined = readcsv(B/f'results/{cohort}_rows.csv')
        require(set(chunks.columns) == set(combined.columns), f'aggregate columns {cohort}')
        expected = chunks.sort_values(keys).reset_index(drop=True)
        actual = combined.sort_values(keys).reset_index(drop=True)[expected.columns]
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=True)
        details[cohort] = dict(rows=len(actual), exact_aggregate_matches_chunks=True)
    return details


def check_independent_representative_fits():
    """Rebuild GP and use SciPy BFGS, independent of core/OneSignalProfile."""
    data, _ = independent_templates()
    x = data['x']
    allrows = readcsv(B/'results/evaluation_rows.csv')
    details = []
    for m, shape in [(60, 'mc'), (140, 'gaussian'), (240, 'mc')]:
        saved = dict(np.load(B/f'results/representative/m{m:03d}_{shape}.npz'))
        row = allrows[(allrows.mass_MeV == m) & (allrows['shape'] == shape) &
                      (allrows.z == 5) & (allrows.toy == 0)].iloc[0]
        mask, counts = saved['mask'], saved['counts']
        index = int(np.flatnonzero(data['masses'] == m)[0])
        const, length = data['const'][index], data['ls'][index]
        kernel = lambda a,b: const*np.exp(-.5*((np.log(a)[:,None]-np.log(b)[None,:])/length)**2)
        ntrain = counts[~mask].astype(float)
        target = np.where(ntrain > 0, np.log(np.maximum(ntrain, 1.)), 0.)
        alpha = np.where(ntrain > 0, 1./np.maximum(ntrain, 1.), 1.)
        train_kernel = kernel(x[~mask], x[~mask])+np.diag(alpha)
        cross = kernel(x[mask], x[~mask])
        chol = cholesky(train_kernel, lower=True)
        latent_mean = cross@cho_solve((chol, True), target)
        projected = solve_triangular(chol, cross.T, lower=True)
        latent_cov = kernel(x[mask], x[mask])-projected.T@projected
        latent_cov = (latent_cov+latent_cov.T)/2
        mean = np.exp(latent_mean+.5*np.maximum(np.diag(latent_cov), 0.))
        covariance = np.outer(mean, mean)*np.expm1(np.clip(latent_cov, -40, 40))
        covariance = (covariance+covariance.T)/2
        scale_cov = max(float(np.diag(covariance).max()), 1.)
        for load in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
            try:
                loaded = covariance+load*scale_cov*np.eye(len(mean))
                cholesky(loaded, lower=True)
                break
            except np.linalg.LinAlgError:
                pass
        else:
            raise AssertionError('independent GP covariance factorization')
        root = np.sqrt(mean)
        eig, vectors = np.linalg.eigh(loaded/root[:,None]/root[None,:])
        keep = eig > 1e-8
        factor = root[:,None]*vectors[:,keep]*np.sqrt(eig[keep])
        close(mean, saved['gp_mean'], f'independent GP mean {m} {shape}', rtol=3e-11, atol=1e-7)
        close(np.sqrt(np.sum(factor**2, axis=1)), saved['gp_sigma'],
              f'independent GP uncertainty {m} {shape}', rtol=3e-9, atol=1e-6)
        n = counts[mask].astype(float)
        p = saved['categories'][1:-1][mask]
        amp_scale = 1./np.sqrt(np.sum(p*p/mean))
        jac = np.column_stack([amp_scale*p, factor])
        penalty = np.r_[0., np.ones(factor.shape[1])]

        def function(z, matrix, base, penalize):
            lam = base+matrix@z
            if np.any(lam <= 0):
                return np.inf, np.zeros_like(z)
            delta = (lam-n)/n
            f = float(np.sum(n*(delta-np.log1p(delta)))+.5*np.sum(penalize*z*z))
            grad = matrix.T@(1.-n/lam)+penalize*z
            return f, grad

        fitted = minimize(function, np.zeros(jac.shape[1]), args=(jac, mean, penalty),
                          jac=True, method='BFGS', options={'gtol': 1e-8, 'maxiter': 500})
        fnll, gradient = function(fitted.x, jac, mean, penalty)
        require(np.max(np.abs(gradient)) < 3e-5, f'independent optimizer free gradient {m} {shape}')
        lam = mean+jac@fitted.x
        hessian = (jac.T*(n/lam**2))@jac+np.diag(penalty)
        covariance_parameters = np.linalg.inv(hessian)
        fit_amp = fitted.x[0]*amp_scale
        fit_sigma = amp_scale*np.sqrt(covariance_parameters[0,0])
        expected = float(saved['A_expected'])
        fixed = minimize(function, np.zeros(factor.shape[1]),
                         args=(factor, mean+expected*p, np.ones(factor.shape[1])),
                         jac=True, method='BFGS', options={'gtol': 1e-8, 'maxiter': 500})
        pnll, pgradient = function(fixed.x, factor, mean+expected*p, np.ones(factor.shape[1]))
        require(np.max(np.abs(pgradient)) < 3e-5, f'independent optimizer profile gradient {m} {shape}')
        close(fit_amp, row.Ahat, f'independent signed amplitude {m} {shape}', rtol=1e-8, atol=3e-3)
        close(fit_sigma, row.sigma_postfit, f'independent Hessian uncertainty {m} {shape}', rtol=1e-8, atol=1e-4)
        close(fnll, row.free_nll, 'independent minimum nll', atol=2e-6)
        close(2*(pnll-fnll), row.q_true, 'independent profile containment statistic', atol=2e-6)
        details.append(dict(mass_MeV=m, shape=shape, toy=0, z=5,
                            Ahat_difference=float(fit_amp-row.Ahat),
                            sigma_difference=float(fit_sigma-row.sigma_postfit),
                            q_difference=float(2*(pnll-fnll)-row.q_true),
                            free_gradient=float(np.max(np.abs(gradient))),
                            profile_gradient=float(np.max(np.abs(pgradient))),
                            optimizer='SciPy BFGS with independent likelihood/gradient'))
    return details


def check_row_diagnostics():
    summary = {}
    for cohort in ('pilot', 'evaluation'):
        frame, paths = load_rows(cohort)
        levels = (0,) if cohort == 'pilot' else LEVELS
        expected_ids = {f'{cohort}:m{m}:{shape}:z{z}:t{toy}' for m in MASSES
                        for shape in SHAPES for z in levels for toy in range(100)}
        require(len(frame) == len(expected_ids), f'{cohort} attempted row count')
        require(not frame.row_id.duplicated().any(), f'{cohort} unique row IDs')
        require(set(frame.row_id) == expected_ids, f'{cohort} complete IDs')
        require(set(frame.control) == {'primary'}, f'{cohort} primary-only rows')
        fits, profiles = boolcol(frame, 'fit_valid'), boolcol(frame, 'profile_valid')
        require(not np.any(profiles & ~fits), 'valid profile needs free fit')
        require((frame.sigma_method == 'observed_profile_hessian').all(), 'observed Hessian uncertainty method')
        for row in frame.itertuples(index=False):
            r = row._asdict()
            require(r['row_id'] == f'{cohort}:m{r["mass_MeV"]}:{r["shape"]}:z{r["z"]}:t{r["toy"]}', 'ID/columns agree')
            valid = str(r['fit_valid']).lower() == 'true'
            pvalid = str(r['profile_valid']).lower() == 'true'
            if valid:
                for key in ('Ahat', 'sigma_postfit', 'pull', 'free_nll', 'fit_score', 'min_lambda'):
                    require(np.isfinite(float(r[key])), f'finite {key} {r["row_id"]}')
                require(float(r['sigma_postfit']) > 0 and float(r['min_lambda']) > 0,
                        f'positive error and means {r["row_id"]}')
                require(float(r['fit_score']) < 3e-5, f'free score {r["row_id"]}')
                close(float(r['pull']), (float(r['Ahat'])-float(r['A_expected']))/float(r['sigma_postfit']),
                      f'expected-yield pull {r["row_id"]}')
                require(0 <= int(r['nuisance_rank']) <= int(r['fit_bins']), f'covariance rank {r["row_id"]}')
                require(0 < float(r['covariance_load']) <= 1e-5, f'bounded covariance load {r["row_id"]}')
                require(float(r['max_omitted_covariance_mode']) <= 1.00001e-8,
                        f'covariance mode threshold {r["row_id"]}')
            if cohort == 'pilot':
                require(not pvalid, 'pilot does not run unnecessary truth profiles')
            elif pvalid:
                qraw = 2*(float(r['true_nll'])-float(r['free_nll']))
                require(qraw >= -2e-6, f'likelihood ordering {r["row_id"]}')
                require(float(r['profile_score']) < 3e-5 and float(r['profile_min_lambda']) > 0,
                        f'valid profile score and means {r["row_id"]}')
                close(float(r['q_true_raw']), qraw, 'raw likelihood ratio')
                close(float(r['q_true']), max(0., qraw), 'clipped likelihood ratio')
                for field, threshold in [('profile_contains68', 1.), ('profile_contains95', 3.841459)]:
                    require((str(r[field]).lower() == 'true') == (max(0., qraw) <= threshold),
                            f'nominal containment threshold {r["row_id"]}')
            if not valid or (cohort == 'evaluation' and not pvalid):
                require(bool(str(r['failure_reason']).strip()), f'failure reason recorded {r["row_id"]}')
            if str(r.get('attempts_json', '')).strip():
                attempts = json.loads(r['attempts_json'])
                require(1 <= len(attempts) <= 3 and len(attempts) == int(r['attempt_count']),
                        f'bounded retained attempts {r["row_id"]}')
                good_f = [a['free_nll'] for a in attempts if a.get('free_valid')]
                good_p = [a['true_nll'] for a in attempts if a.get('profile_valid')]
                if valid:
                    close(float(r['free_nll']), min(good_f), 'minimum valid free objective')
                if pvalid:
                    close(float(r['true_nll']), min(good_p), 'minimum valid profile objective')
        summary[cohort] = dict(attempted=len(frame), fit_valid=int(fits.sum()), profile_valid=int(profiles.sum()),
                               free_failures=int((~fits).sum()),
                               unresolved_profiles=int((~profiles).sum()) if cohort == 'evaluation' else 0)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only', action='store_true')
    args = parser.parse_args()
    audit = Audit()
    audit.run('immutable_inputs_handoff_and_independent_templates', check_inputs)
    audit.run('independent_one_bin_solver_control', check_solver_analytic)
    if not args.inputs_only:
        audit.run('all_background_counts_independent_RNG_replay', check_cohorts)
        audit.run('saved_templates_against_independent_CDFs', check_templates)
        audit.run('complete_IDs_and_numerical_diagnostics', check_row_diagnostics)
        audit.run('aggregate_rows_exactly_match_checkpoints', check_aggregate_rows)
        audit.run('independent_pilot_scale_and_freeze_hash', check_reference)
        audit.run('all_signal_counts_expected_truth_and_pairing_replay', check_row_replay)
        audit.run('checkpoint_signatures_checksums_and_freeze_order', check_checkpoints)
        audit.run('independent_GP_and_representative_BFGS_fits', check_independent_representative_fits)
    passed = all(c['passed'] for c in audit.checks)
    out = dict(status='passed' if passed else 'failed', inputs_only=args.inputs_only,
               checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               validator_sha256=sha(__file__), checks=audit.checks,
               scope='Numerical/provenance/replay validity only; scientific bias, pull width and containment are outcomes.')
    target = B/'qa/independent_validation.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
