#!/usr/bin/env python3
"""Bounded, single-thread timing/closure pilot for exact frozen-kernel GP refits.

Only this script and gp_refit_pilot.json are pilot-owned outputs. No production
files are changed. CachedCholeskyPredictor is an exact algebraic replacement
for fit_gpr(..., optimize=False) plus the requested predictive moments; it
updates the archived count-dependent alpha for every toy. It omits sklearn's
redundant fixed-kernel LML factorization and caches only invariant kernels.
The preconditioned candidate is a diagnostic, not an approved production path.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/'study_results/background_profile_comparison_20260905'))
import run_comparison as c
from scipy.linalg import cholesky, cho_solve, solve_triangular
from hps_gpr.gpr import preprocess_xy_for_gpr
from threadpoolctl import threadpool_limits, threadpool_info
import sklearn

np, pd = c.np, c.pd


def count_moments(mu, cov, config):
    """Bit-for-bit expression ordering from the archived count transform."""
    if not config.pre_log:
        return mu, cov
    diagonal = np.clip(np.diag(cov), 0., None)
    expected = np.exp(mu + .5*diagonal)
    covariance = np.outer(expected, expected) * (np.exp(np.clip(cov, -40., 40.))-1.)
    return expected, covariance


class CachedCholeskyPredictor:
    """Cache invariant kernels; factor the exact updated K+diag(alpha) each toy."""
    def __init__(self, x_train, x_query, kernel, config):
        self.x_train = np.asarray(x_train, float)
        self.x_query = np.asarray(x_query, float)
        self.config = config
        xt = np.log(np.clip(self.x_train, 1e-12, None)) if config.pre_log else self.x_train
        xq = np.log(np.clip(self.x_query, 1e-12, None)) if config.pre_log else self.x_query
        self.K = kernel(xt[:, None])
        self.Kqt = kernel(xq[:, None], xt[:, None])
        self.Kqq = kernel(xq[:, None])
        self.diagonal = np.diag_indices_from(self.K)

    def latent(self, y):
        _, target, alpha = preprocess_xy_for_gpr(self.x_train, y, self.config)
        matrix = self.K.copy()
        matrix[self.diagonal] += alpha
        factor = cholesky(matrix, lower=True, check_finite=False)
        coefficient = cho_solve((factor, True), target, check_finite=False)
        v = solve_triangular(factor, self.Kqt.T, lower=True, check_finite=False)
        return self.Kqt@coefficient, self.Kqq-v.T@v

    def predict(self, y):
        return count_moments(*self.latent(y), self.config)


class PreconditionedCandidate:
    """Converged diagonal-update iteration with an explicit contraction guard."""
    def __init__(self, cached, reference_y):
        self.cached = cached
        _, _, self.alpha0 = preprocess_xy_for_gpr(cached.x_train, reference_y, cached.config)
        matrix = cached.K.copy()
        matrix[cached.diagonal] += self.alpha0
        self.factor = cholesky(matrix, lower=True, check_finite=False)
        self.query0 = cho_solve((self.factor, True), cached.Kqt.T, check_finite=False)
        v = solve_triangular(self.factor, cached.Kqt.T, lower=True, check_finite=False)
        self.cov0 = cached.Kqq-v.T@v

    def predict(self, y):
        cached = self.cached
        _, target, alpha = preprocess_xy_for_gpr(cached.x_train, y, cached.config)
        delta = alpha-self.alpha0
        contraction_bound = float(np.max(np.abs(delta)/self.alpha0))
        if contraction_bound >= .5:
            return None, {'status': 'guard_fallback', 'contraction_bound': contraction_bound}
        mean0 = cho_solve((self.factor, True), target, check_finite=False)
        rhs0 = np.column_stack((mean0, self.query0))
        value = rhs0.copy()
        for iteration in range(1, 33):
            new = rhs0-cho_solve((self.factor, True), delta[:, None]*value, check_finite=False)
            relative_step = float(np.max(np.abs(new-value)/np.maximum(1., np.abs(new))))
            value = new
            if relative_step < 2e-12:
                break
        else:
            return None, {'status': 'convergence_fallback', 'contraction_bound': contraction_bound,
                          'iterations': iteration, 'relative_step': relative_step}
        mu = cached.Kqt@value[:, 0]
        covariance = self.cov0+cached.Kqt@(self.query0-value[:, 1:])
        covariance = .5*(covariance+covariance.T)
        moments = count_moments(mu, covariance, cached.config)
        return moments, {'status': 'converged', 'contraction_bound': contraction_bound,
                         'iterations': iteration, 'relative_step': relative_step}


def differences(reference, candidate):
    b, covariance = reference
    bm, cm = candidate
    scale = max(float(np.max(np.diag(covariance))), 1e-300)
    return {'mean_max_relative': float(np.max(np.abs(bm-b)/b)),
            'mean_max_over_poisson_sd': float(np.max(np.abs(bm-b)/np.sqrt(b))),
            'cov_max_abs_over_max_diag': float(np.max(np.abs(cm-covariance))/scale),
            'cov_diag_max_relative': float(np.max(np.abs(np.diag(cm)-np.diag(covariance))/np.maximum(np.diag(covariance), 1e-300))),
            'mean_bit_exact': bool(np.array_equal(b, bm)),
            'cov_bit_exact': bool(np.array_equal(covariance, cm))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--toys', type=int, default=32)
    parser.add_argument('--max-seconds', type=float, default=25.)
    args = parser.parse_args()
    if not (1 <= args.toys <= 100 and 0 < args.max_seconds <= 30):
        raise ValueError('Pilot limits: at most 100 toys/coordinate and 30 seconds')
    started = time.perf_counter()
    cfg = c.production.load_config(c.production.DEFAULT_CARD)
    c.production.validate_card(cfg)
    datasets = c.production.make_datasets(cfg)
    states = c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
    ledger = pd.read_csv(c.PARENT/'derived/prediction_state_ledger.csv')
    ledger['dataset'] = ledger.dataset.astype(str)
    ledger = ledger.set_index(['dataset', 'mass_MeV'])
    sources = [Path(__file__), c.production.DEFAULT_CARD, c.production.DEFAULT_STATES,
               Path(preprocess_xy_for_gpr.__code__.co_filename), Path(c.__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    points, details = [], []
    with threadpool_limits(limits=1):
        for dataset, mass in [('2015', 51), ('2016', 88), ('2021', 71)]:
            state = states[dataset, mass]
            kernel = c.make_fixed_kernel(state['const_opt'], state['ls_opt'])
            prediction = c.production.estimate_background_for_dataset(
                datasets[dataset], mass/1000., cfg, restarts=0, kernel=kernel,
                optimize=False, train_exclude_nsigma=2.25)
            digest = c.production.prediction_state_sha256(prediction)
            if digest != ledger.loc[(dataset, mass), 'prediction_state_sha256']:
                raise RuntimeError('Frozen native prediction mismatch')
            keep = ~prediction.blind_mask
            xt = prediction.x_full[keep]
            xq = prediction.x_full[prediction.blind_mask]
            nominal = c.fit_gpr(xt, prediction.y_full[keep], cfg, restarts=0,
                                kernel=kernel, optimize=False)
            # A positive full-support expectation solely for this numerical pilot.
            full_mean, _ = c.predict_counts_from_log_gpr(nominal, prediction.x_full, cfg)
            rng = np.random.default_rng(np.random.SeedSequence([491313, int(dataset), mass]))
            toys = rng.poisson(full_mean, size=(args.toys, len(full_mean)))
            before = time.perf_counter()
            cached = CachedCholeskyPredictor(xt, xq, kernel, cfg)
            cached_setup = time.perf_counter()-before
            before = time.perf_counter()
            candidate = PreconditionedCandidate(cached, full_mean[keep])
            candidate_setup = time.perf_counter()-before
            def direct(y):
                gp = c.fit_gpr(xt, y, cfg, restarts=0, kernel=kernel, optimize=False)
                return c.predict_counts_from_log_gpr(gp, xq, cfg)
            direct(toys[0, keep]); cached.predict(toys[0, keep])
            point_rows = []
            for toy_id, whole in enumerate(toys):
                if time.perf_counter()-started > args.max_seconds:
                    break
                y = whole[keep]
                t0 = time.perf_counter(); reference = direct(y); direct_time = time.perf_counter()-t0
                t0 = time.perf_counter(); output = cached.predict(y); cached_time = time.perf_counter()-t0
                closure = differences(reference, output)
                t0 = time.perf_counter(); pre, diagnostic = candidate.predict(y); pre_time = time.perf_counter()-t0
                if pre is None:
                    diagnostic['effective_seconds_with_cached_fallback'] = pre_time+cached_time
                else:
                    diagnostic['effective_seconds_with_cached_fallback'] = pre_time
                    diagnostic.update(differences(reference, pre))
                row = {'dataset': dataset, 'mass_MeV': mass, 'toy_id': toy_id,
                       'direct_seconds': direct_time, 'cached_cholesky_seconds': cached_time,
                       'preconditioned_seconds': pre_time, 'cached_closure': closure,
                       'preconditioned': diagnostic}
                point_rows.append(row); details.append(row)
            if not point_rows:
                raise RuntimeError('Time cap prevented requested coordinate coverage')
            maximum = lambda key: max(row['cached_closure'][key] for row in point_rows)
            direct_sum = sum(row['direct_seconds'] for row in point_rows)
            cached_sum = sum(row['cached_cholesky_seconds'] for row in point_rows)
            pre_sum = sum(row['preconditioned']['effective_seconds_with_cached_fallback'] for row in point_rows)
            converged = [row['preconditioned'] for row in point_rows if row['preconditioned']['status'] == 'converged']
            points.append({'dataset': dataset, 'mass_MeV': mass, 'training_bins': int(keep.sum()),
                'query_bins': len(xq), 'completed_toys': len(point_rows),
                'native_prediction_hash_exact': True,
                'direct_median_seconds': float(np.median([row['direct_seconds'] for row in point_rows])),
                'cached_cholesky_median_seconds': float(np.median([row['cached_cholesky_seconds'] for row in point_rows])),
                'cached_cholesky_speedup': direct_sum/cached_sum,
                'cached_cholesky_speedup_including_setup': direct_sum/(cached_sum+cached_setup),
                'cached_kernel_setup_seconds': cached_setup,
                'preconditioned_setup_seconds': candidate_setup,
                'cached_closure_maxima': {key: maximum(key) for key in ('mean_max_relative', 'mean_max_over_poisson_sd', 'cov_max_abs_over_max_diag', 'cov_diag_max_relative')},
                'all_cached_means_bit_exact': all(row['cached_closure']['mean_bit_exact'] for row in point_rows),
                'all_cached_covariances_bit_exact': all(row['cached_closure']['cov_bit_exact'] for row in point_rows),
                'preconditioned_converged_toys': len(converged),
                'preconditioned_guarded_speedup': direct_sum/pre_sum,
                'preconditioned_guarded_speedup_including_setup': direct_sum/(pre_sum+cached_setup+candidate_setup),
                'preconditioned_maxima': {key: max((row[key] for row in converged), default=None)
                    for key in ('mean_max_relative', 'mean_max_over_poisson_sd', 'cov_max_abs_over_max_diag', 'cov_diag_max_relative', 'iterations')}})
    elapsed = time.perf_counter()-started
    source_drift = [p for p, digest in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
    if source_drift:
        raise RuntimeError(f'Source drift: {source_drift}')
    exact = all(point['all_cached_means_bit_exact'] and point['all_cached_covariances_bit_exact'] for point in points)
    payload = {'status': 'completed', 'wall_seconds': elapsed, 'time_cap_seconds': args.max_seconds,
        'requested_toys_per_coordinate': args.toys, 'total_toys': len(details),
        'recommendation': 'Use cached invariant kernels plus one fresh Cholesky per toy; retain direct fit_gpr for audit.' if exact else 'Inspect cached Cholesky numerical discrepancies before adoption.',
        'preconditioned_recommendation': 'Diagnostic only; use the exact cached-Cholesky path unless an independent numerical gate approves this candidate.',
        'archived_semantics': {'pre_log': bool(cfg.pre_log), 'alpha_model': cfg.alpha_model,
            'pre_zero_alpha': cfg.pre_zero_alpha, 'pre_alpha_first_n': cfg.pre_alpha_first_n,
            'pre_alpha_first_factor': cfg.pre_alpha_first_factor, 'normalize_y': False,
            'alpha_updated_per_toy': True, 'kernel_optimized': False,
            'counts_transform': 'exp(mu+0.5*max(diag(C),0)); outer(E,E)*(exp(clip(C,-40,40))-1)'},
        'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
        'threads_requested': 1, 'threadpool_info': threadpool_info(),
        'sources_sha256': hashes, 'sources_unchanged': True,
        'points': points, 'toy_details': details,
        'scope': 'Numerical refit/prediction timing only; no calibration or coverage study.'}
    (HERE/'gp_refit_pilot.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({key: value for key, value in payload.items() if key not in ('toy_details','threadpool_info','sources_sha256')}, indent=2))


if __name__ == '__main__':
    main()
