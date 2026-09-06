#!/usr/bin/env python3
"""Bounded numerical pilot of an APPROXIMATE eigenfeature frozen-kernel GP.

No statistical calibration runs here. Tiny eigenvalues are truncated from a
joint train/query kernel. Accuracy is judged against the dense archived GP,
including downstream fixed and Gaussian-profiled likelihood-ratio statistics.
The approximation is not bit-exact and is not approved beyond tested inputs.
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
import gp_refit_pilot as dense
from scipy.linalg import cholesky, cho_solve, solve_triangular, eigh
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
c, np, pd = dense.c, dense.np, dense.pd
RTOLS = (1e-13, 1e-14, 1e-15)


class LowRankPredictor:
    """Approximate eigenfeature GP; call predict(y_train) for count moments.

    Parameters match CachedCholeskyPredictor, with an eigenvalue-relative
    truncation rtol. Every call updates alpha and solves the Bayesian feature
    posterior; no training data, alpha, or predictive covariance are frozen.
    """
    def __init__(self, x_train, x_query, kernel, config, rtol=1e-15, eigensystem=None):
        self.x_train = np.asarray(x_train, float)
        self.config = config
        self.rtol = float(rtol)
        if eigensystem is None:
            eigenvalues, eigenvectors, kernel_matrix = self.decompose(x_train, x_query, kernel, config)
        else:
            eigenvalues, eigenvectors, kernel_matrix = eigensystem
        threshold = float(eigenvalues[-1]*rtol)
        use = eigenvalues > threshold
        features = eigenvectors[:, use]*np.sqrt(eigenvalues[use])
        self.train = np.ascontiguousarray(features[:len(self.x_train)])
        self.query = np.ascontiguousarray(features[len(self.x_train):])
        self.rank = int(use.sum())
        self.identity = np.eye(self.rank)
        self.kernel_max_abs_error = float(np.max(np.abs(kernel_matrix-features@features.T)))
        self.eigenvalue_threshold = threshold
        self.smallest_joint_eigenvalue = float(eigenvalues[0])
        self.dropped_positive_eigenvalue_sum = float(np.sum(eigenvalues[(~use)&(eigenvalues > 0)]))

    @staticmethod
    def decompose(x_train, x_query, kernel, config):
        x = np.r_[x_train, x_query]
        if config.pre_log:
            x = np.log(np.clip(x, 1e-12, None))
        matrix = kernel(x[:, None])
        values, vectors = eigh(matrix, check_finite=False, driver='evr')
        return values, vectors, matrix

    def predict(self, y):
        _, target, alpha = dense.preprocess_xy_for_gpr(self.x_train, y, self.config)
        weight = 1./alpha
        precision = self.identity+self.train.T@(self.train*weight[:, None])
        # Congruence scaling improves rank-space conditioning without changing
        # the finite-rank Gaussian model.
        scale = 1./np.sqrt(np.diag(precision))
        scaled = precision*np.outer(scale, scale)
        factor = cholesky(scaled, lower=True, check_finite=False)
        rhs = scale*(self.train.T@(weight*target))
        coefficient = cho_solve((factor, True), rhs, check_finite=False)
        query_scaled = self.query*scale[None, :]
        mu = query_scaled@coefficient
        v = solve_triangular(factor, query_scaled.T, lower=True, check_finite=False)
        covariance = v.T@v
        return dense.count_moments(mu, covariance, self.config)


def compare_moments(reference, approximate):
    b, C = reference; q, D = approximate
    gp_sd = np.sqrt(np.maximum(np.diag(C), 1e-300))
    return dict(mean_error_over_gp_sd=float(np.max(np.abs(q-b)/gp_sd)),
                mean_error_over_total_sd=float(np.max(np.abs(q-b)/np.sqrt(b+np.diag(C)))),
                covariance_max_error_over_max_diag=float(np.max(np.abs(D-C))/max(float(np.diag(C).max()), 1e-300)),
                mean_max_relative=float(np.max(np.abs(q-b)/b)))


def likelihood_values(counts, b, C, w, amplitudes):
    covariance, _ = c.production.condition_covariance_block(C, b)
    models = {'profiled': c.Profile(b, c._chol_with_jitter(covariance), w, 'linear'),
              'fixed': c.Profile(b, np.zeros((len(b), 0)), w, 'linear')}
    result = {}
    for method, model in models.items():
        free = model.fit(counts); null = model.fit(counts, 0.)
        r = float(np.sign(free['A'])*np.sqrt(max(0., 2*(null['nll']-free['nll']))))
        denominator = free if free['A'] >= 0 else null
        row = {'signed_r': r}
        for strength, amplitude in amplitudes.items():
            fitted = model.fit(counts, amplitude)
            q = 0. if free['A'] > amplitude else max(0., 2*(fitted['nll']-denominator['nll']))
            row[f'q_{strength}sigma'] = float(q)
        result[method] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--toys', type=int, default=12)
    parser.add_argument('--max-seconds', type=float, default=25.)
    args = parser.parse_args()
    if not (3 <= args.toys <= 30 and 0 < args.max_seconds <= 30):
        raise ValueError('Pilot limit is <=30 toys/coordinate and <=30 seconds')
    start = time.perf_counter()
    cfg = c.production.load_config(c.production.DEFAULT_CARD)
    c.production.validate_card(cfg)
    datasets = c.production.make_datasets(cfg)
    states = c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
    source_paths = [Path(__file__), Path(dense.__file__), c.production.DEFAULT_CARD,
                    c.production.DEFAULT_STATES, Path(dense.preprocess_xy_for_gpr.__code__.co_filename)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    points, details = [], []
    with threadpool_limits(limits=1):
        for dataset, mass in [('2015', 51), ('2016', 88), ('2021', 71)]:
            st = states[dataset, mass]
            kernel = c.make_fixed_kernel(st['const_opt'], st['ls_opt'])
            p = c.production.estimate_background_for_dataset(datasets[dataset], mass/1000., cfg,
                restarts=0, kernel=kernel, optimize=False, train_exclude_nsigma=2.25)
            keep = ~p.blind_mask
            xt, xq = p.x_full[keep], p.x_full[p.blind_mask]
            nominal_gp = c.fit_gpr(xt, p.y_full[keep], cfg, restarts=0, kernel=kernel, optimize=False)
            full_mean, _ = c.predict_counts_from_log_gpr(nominal_gp, p.x_full, cfg)
            template, full = c.build_window_template_from_full(p.edges_full, p.blind_mask,
                mass/1000., p.sigma_val, config=cfg)
            fraction = float(template.sum()); w = template/fraction
            Cnom, _ = c.production.condition_covariance_block(p.cov, p.mu)
            nominal_factor = c._chol_with_jitter(Cnom)
            sigma = float(1./np.sqrt(w@np.linalg.solve(np.diag(p.mu)+nominal_factor@nominal_factor.T, w)))
            amplitudes = {2: 2*sigma, 5: 5*sigma}
            rng = np.random.default_rng(np.random.SeedSequence([491314, int(dataset), mass]))
            toys = [rng.poisson(full_mean+(0, 2, 5)[toy%3]*sigma/fraction*full)
                    for toy in range(args.toys)]
            exact_cached = dense.CachedCholeskyPredictor(xt, xq, kernel, cfg)
            t0 = time.perf_counter()
            eigensystem = LowRankPredictor.decompose(xt, xq, kernel, cfg)
            decomposition_seconds = time.perf_counter()-t0
            candidates = {rtol: LowRankPredictor(xt, xq, kernel, cfg, rtol, eigensystem)
                          for rtol in RTOLS}
            candidate_rows = {rtol: [] for rtol in RTOLS}
            direct_times, cached_times = [], []
            for toy_id, counts in enumerate(toys):
                if time.perf_counter()-start > args.max_seconds:
                    raise RuntimeError('Pilot time cap reached before completion')
                y = counts[keep]
                t0 = time.perf_counter()
                gp = c.fit_gpr(xt, y, cfg, restarts=0, kernel=kernel, optimize=False)
                reference = c.predict_counts_from_log_gpr(gp, xq, cfg)
                direct_times.append(time.perf_counter()-t0)
                t0 = time.perf_counter(); exact = exact_cached.predict(y); cached_times.append(time.perf_counter()-t0)
                if not (np.array_equal(reference[0], exact[0]) and np.array_equal(reference[1], exact[1])):
                    raise RuntimeError('Dense cached reference lost exact closure')
                ref_stats = likelihood_values(counts[p.blind_mask], *reference, w, amplitudes)
                for rtol, candidate in candidates.items():
                    t0 = time.perf_counter(); approximate = candidate.predict(y); duration = time.perf_counter()-t0
                    errors = compare_moments(reference, approximate)
                    stats = likelihood_values(counts[p.blind_mask], *approximate, w, amplitudes)
                    stat_errors = {f'{method}_{name}_abs_error': abs(stats[method][name]-ref_stats[method][name])
                                   for method in stats for name in stats[method]}
                    row = dict(dataset=dataset, mass_MeV=mass, toy_id=toy_id,
                               injected_sigma=(0, 2, 5)[toy_id%3], rtol=rtol,
                               lowrank_seconds=duration, **errors, **stat_errors)
                    candidate_rows[rtol].append(row); details.append(row)
            summaries = []
            for rtol, candidate in candidates.items():
                rows = candidate_rows[rtol]
                columns = [key for key in rows[0] if 'error' in key or key == 'mean_max_relative']
                maxima = {key: max(row[key] for row in rows) for key in columns}
                moment_gate = maxima['mean_error_over_gp_sd'] < 1e-3 and maxima['covariance_max_error_over_max_diag'] < 1e-3
                statistic_gate = all(value < 1e-3 for key, value in maxima.items() if key.endswith('_abs_error'))
                total_time = sum(row['lowrank_seconds'] for row in rows)
                summaries.append(dict(rtol=rtol, rank=candidate.rank,
                    eigenvalue_threshold=candidate.eigenvalue_threshold,
                    smallest_joint_eigenvalue=candidate.smallest_joint_eigenvalue,
                    dropped_positive_eigenvalue_sum=candidate.dropped_positive_eigenvalue_sum,
                    kernel_max_abs_error=candidate.kernel_max_abs_error,
                    lowrank_median_seconds=float(np.median([row['lowrank_seconds'] for row in rows])),
                    speedup_vs_direct=sum(direct_times)/total_time,
                    speedup_vs_exact_cached=sum(cached_times)/total_time,
                    approximation_gate_passed=moment_gate and statistic_gate,
                    speed_goal_20x_vs_direct_met=sum(direct_times)/total_time >= 20,
                    error_maxima=maxima))
            points.append(dict(dataset=dataset, mass_MeV=mass, training_bins=len(xt), query_bins=len(xq),
                completed_toys=args.toys, direct_median_seconds=float(np.median(direct_times)),
                exact_cached_median_seconds=float(np.median(cached_times)),
                joint_decomposition_seconds=decomposition_seconds, candidates=summaries))
    passing = [rtol for rtol in RTOLS if all(next(row for row in point['candidates'] if row['rtol'] == rtol)['approximation_gate_passed'] for point in points)]
    selected = min(passing) if passing else None
    source_drift = [p for p, h in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    if source_drift:
        raise RuntimeError('Source drift: '+str(source_drift))
    payload = dict(status='completed', wall_seconds=time.perf_counter()-start,
        time_cap_seconds=args.max_seconds, threads_requested=1, toys_per_coordinate=args.toys,
        statistical_checks='Both fixed and Gaussian-profiled signed r and bounded q_mu at 2/5 nominal profiled sigma; spectra contain 0/2/5 sigma injections.',
        predeclared_gates={'mean_over_gp_sd': 1e-3, 'covariance_over_max_diag': 1e-3, 'r_abs': 1e-3, 'q_abs': 1e-3},
        passing_rtol_values=passing, most_conservative_passing_rtol=selected,
        recommendation='Approximation passed the bounded pilot; extend numerical audit to the entire planned mass grid before production.' if selected is not None else 'Reject low-rank acceleration: numerical pilot gate failed.',
        exact=False, points=points, toy_details=details, sources_sha256=hashes,
        sources_unchanged=True, scope='Numerical approximation/timing pilot only; no calibrated inference or coverage claim.')
    (HERE/'gp_lowrank_pilot.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({key: value for key, value in payload.items() if key not in ('toy_details', 'sources_sha256')}, indent=2))


if __name__ == '__main__':
    main()
