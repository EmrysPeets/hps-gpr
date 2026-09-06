#!/usr/bin/env python3
"""Observed fixed-GP diagnostic, preserving every v4.9.12 production input."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PARENT = REPO / 'study_results/v4p9p12_final_dataset_combinations_20260902'
PREVIOUS = REPO / 'study_results/background_profile_comparison_20260905'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PREVIOUS))
import run_comparison as comparison
production = comparison.production
np, pd = comparison.np, comparison.pd
from scipy.optimize import brentq
from scipy.stats import norm
from hps_gpr.template import build_window_template_from_full

SCOPES = [item for item in production.SCOPES
          if item[0].startswith('individual_') or item[0].startswith('all_')]
OUT = HERE / 'derived'
MUON_MASS_MEV = 105.6583745


def correction(mass):
    if mass <= 2 * MUON_MASS_MEV:
        return 1.
    r = (MUON_MASS_MEV / mass)**2
    return float(1 + np.sqrt(1 - 4*r) * (1 + 2*r))


def independent_fixed(n, b, w, fit):
    """Scalar score and ratio calculation, independent of Profile derivatives."""
    lo = -float(np.min(b[w > 0] / w[w > 0])) * (1 - 1e-10)
    score = lambda a: float(np.sum(w * (1 - n / (b + a*w))))
    hi = max(10., abs(float(fit['Ahat'])) + 10 * float(fit['sigma_A']))
    while score(hi) < 0:
        hi *= 2
    ahat = brentq(score, lo, hi, xtol=1e-8, rtol=1e-13)
    bounded = max(ahat, 0.)
    lam_den = b + bounded*w
    lam_best = b + ahat*w
    def ratio(a):
        lam = b + a*w
        qo = 0. if a < ahat else 2 * float(np.sum((a-bounded)*w - n*np.log1p((a-bounded)*w/lam_den)))
        qa = 2 * float(np.sum(a*w - b*np.log1p(a*w/b)))
        qo = max(qo, 0.)
        if qo <= qa:
            zsb, zden = np.sqrt(qo), np.sqrt(qo)-np.sqrt(qa)
        else:
            zsb, zden = (qo+qa)/(2*np.sqrt(qa)), (qo-qa)/(2*np.sqrt(qa))
        return float(np.exp(norm.logsf(zsb)-norm.logsf(zden)))
    upper = max(bounded + 5 * fit['sigma_A'], 1.)
    while ratio(upper) > .1:
        upper *= 2
    ul = brentq(lambda a: ratio(a)-.1, bounded+1e-4, upper, xtol=1e-7, rtol=1e-12)
    signed = float(np.sign(ahat) * np.sqrt(max(0., 2*np.sum(-ahat*w + n*np.log1p(ahat*w/b)))))
    result = dict(Ahat_abs_difference=abs(ahat-fit['Ahat']),
                  A90_relative_difference=abs(ul/fit['A90']-1),
                  r_abs_difference=abs(signed-fit['signed_r']),
                  independent_cls_at_profile_limit=ratio(fit['A90']))
    if result['Ahat_abs_difference'] > 2e-3 or result['A90_relative_difference'] > 2e-7 or result['r_abs_difference'] > 2e-6:
        raise RuntimeError(f'Independent fixed fit disagreement: {result}')
    return result


def compact_result(result):
    return {key: value for key, value in result.items()
            if key not in ('free', 'null', 'trace')}


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = production.load_config(production.DEFAULT_CARD)
    production.validate_card(cfg)
    state_frame = production.load_states(production.DEFAULT_STATES, cfg)
    input_provenance = production.validate_input_provenance(
        production.DEFAULT_INPUT_PROVENANCE, production.DEFAULT_CARD,
        production.DEFAULT_STATES, cfg)
    histograms = production.validate_histogram_inputs(cfg)
    datasets = production.make_datasets(cfg)
    states = production.state_map(state_frame)
    released_path = PARENT/'derived/final_dataset_result_curves.csv'
    ledger_path = PARENT/'derived/prediction_state_ledger.csv'
    prior_path = PREVIOUS/'derived/observed_limits.csv'
    released = pd.read_csv(released_path).set_index(['scope_key', 'mass_MeV'])
    ledger = pd.read_csv(ledger_path)
    ledger['dataset'] = ledger.dataset.astype(str)
    ledger = ledger.set_index(['dataset', 'mass_MeV'])
    previous = pd.read_csv(prior_path).set_index('mass_MeV')
    sources = [production.DEFAULT_CARD, production.DEFAULT_STATES,
               production.DEFAULT_INPUT_PROVENANCE, released_path, ledger_path,
               prior_path, PREVIOUS/'run_comparison.py', HERE/'PROTOCOL.md',
               Path(__file__), Path(production.__file__),
               PARENT/'piecewise_cached_solver.py',
               PARENT/'runtime/bounded_tildeq_cls.py',
               REPO/'study_results/v4p9p12_expanded_snapshot_20260905/make_figures.py']
    sources += [Path(item['path']) for item in histograms.values()]
    sources += list((production.RUNTIME_CAMPAIGN/'runtime_combined/hps_gpr').glob('*.py'))
    sources += list((REPO/'gp').glob('*.py'))
    sources += [Path(input_provenance['numerical_exception_path'])]
    protected = {str(path): production.sha256(path) for path in sources}
    records, predictions_ledger, checks, diagnostics = [], [], [], []
    for mass in range(19, 251):
        m = mass / 1000.
        predictions, covariances, conditioning, predrows = production.reconstruct_predictions(m, datasets, cfg, states)
        for item in predrows:
            key = (item['dataset'], mass)
            if item['prediction_state_sha256'] != ledger.loc[key, 'prediction_state_sha256']:
                raise RuntimeError(f'Prediction hash mismatch: {key}')
            item['prediction_hash_exact'] = True
            predictions_ledger.append(item)
        for scope, label, keys, low, high in SCOPES:
            if not low <= mass <= high:
                continue
            obs, b, _, sunit = production.build_combined_components(
                m, [datasets[key] for key in keys], [predictions[key] for key in keys], config=cfg)
            conversion = float(sunit.sum())
            w = sunit / conversion
            saved = released.loc[(scope, mass)]
            if not np.isclose(conversion, saved.signal_yield_per_eps2_fitted_window, rtol=3e-14):
                raise RuntimeError('Released signal normalization does not close')
            # Explicit independent component normalization, especially for the combination.
            rebuilt = []
            for key in keys:
                pred = predictions[key]
                template, _ = build_window_template_from_full(pred.edges_full, pred.blind_mask, m, pred.sigma_val, config=cfg)
                kfactor = production.A_from_epsilon2(datasets[key], m, 1., pred.integral_density)
                rebuilt.append(kfactor*template)
            if not np.array_equal(sunit, np.concatenate(rebuilt)):
                raise RuntimeError('Shared signal coordinate reconstruction mismatch')
            covariance = production.block_diagonal([covariances[key] for key in keys])
            fixed = comparison.Profile(b, np.zeros((len(b), 0)), w, 'linear')
            profiled = comparison.Profile(b, comparison._chol_with_jitter(covariance), w, 'linear')
            fit = fixed.limit(obs)
            fixed_asimov = fixed.limit(b)
            profiled_asimov = profiled.limit(b)
            check = independent_fixed(obs, b, w, fit)
            checks.append(dict(scope_key=scope, mass_MeV=mass, **check))
            z = max(fit['signed_r'], 0.)
            dimuon = correction(mass)
            current = float(saved.eps2_90)
            p0_current = float(saved.p0_local_asymptotic)
            current_z = float(saved.Z_local_asymptotic)
            if not np.isclose(norm.sf(current_z), p0_current, rtol=2e-8, atol=1e-300):
                raise RuntimeError('Released p0/Z identity fails')
            row = dict(scope_key=scope, scope_label=label, dataset_set='+'.join(keys),
                       mass_MeV=mass, n_fit_bins=len(b), dimuon_factor=dimuon,
                       signal_yield_per_eps2_fitted_window=conversion,
                       eps2_current_ee_raw=current, eps2_fixed_ee_raw=fit['A90']/conversion,
                       eps2_current_display=current*dimuon,
                       eps2_fixed_display=fit['A90']/conversion*dimuon,
                       fixed_over_current=fit['A90']/conversion/current,
                       p0_current=p0_current, log_p0_current=float(norm.logsf(current_z)),
                       Z_current=current_z, p0_fixed=float(norm.sf(z)),
                       log_p0_fixed=float(norm.logsf(z)), Z_fixed=z,
                       signed_r_fixed=fit['signed_r'],
                       Ahat_fixed_window=fit['Ahat'], sigma_A_fixed_window=fit['sigma_A'],
                       eps2_hat_fixed_unbounded_ee_raw=fit['Ahat']/conversion,
                       eps2_fixed_asimov_ee_raw=fixed_asimov['A90']/conversion,
                       eps2_profiled_asimov_ee_raw=profiled_asimov['A90']/conversion,
                       eps2_fixed_asimov_display=fixed_asimov['A90']/conversion*dimuon,
                       eps2_profiled_asimov_display=profiled_asimov['A90']/conversion*dimuon,
                       fixed_over_profiled_asimov=fixed_asimov['A90']/profiled_asimov['A90'],
                       fixed_cls_at_limit=fit['cls'],
                       inherited_2016_numerical_exception='2016' in keys,
                       fixed_background_uncertainty_omitted=True,
                       conditional_on_frozen_gp=True, gp_reoptimized=False)
            if keys == ('2021',):
                prior = previous.loc[mass]
                row['prior_2021_fixed_limit_relative_delta'] = float(row['eps2_fixed_ee_raw']/prior.eps2_fixed-1)
                row['prior_2021_fixed_r_delta'] = float(fit['signed_r']-prior.r_fixed)
                if abs(row['prior_2021_fixed_limit_relative_delta']) > 3e-10 or abs(row['prior_2021_fixed_r_delta']) > 2e-8:
                    raise RuntimeError('Prior 2021 fixed result does not close')
            records.append(row)
            for method, result in [('fixed_observed', fit), ('fixed_asimov', fixed_asimov), ('profiled_asimov', profiled_asimov)]:
                diagnostics.append(dict(scope_key=scope, mass_MeV=mass, method=method, **compact_result(result)))
        if mass % 25 == 0 or mass == 250:
            print(f'{mass} MeV: {len(records)} scope rows; {len(predictions_ledger)} exact prediction hashes', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'observed_fixed_comparison.csv', index=False)
    pd.DataFrame(predictions_ledger).to_csv(OUT/'prediction_verification.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT/'independent_fixed_checks.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'fit_diagnostics.csv', index=False)
    minima, summary_rows = [], []
    for scope, label, keys, low, high in SCOPES:
        group = frame[frame.scope_key == scope].sort_values('mass_MeV')
        if group.mass_MeV.tolist() != list(range(low, high+1)):
            raise RuntimeError('Requested grid did not close')
        for method in ('current', 'fixed'):
            minimum = group.loc[group[f'log_p0_{method}'].idxmin()]
            minima.append(dict(scope_key=scope, scope_label=label, method=method,
                               mass_MeV=int(minimum.mass_MeV),
                               p0=float(minimum[f'p0_{method}']),
                               log_p0=float(minimum[f'log_p0_{method}']),
                               Z_local=float(minimum[f'Z_{method}']),
                               inherited_2016_numerical_exception='2016' in keys))
        ratio = group.fixed_over_current
        aratio = group.fixed_over_profiled_asimov
        summary_rows.append(dict(scope_key=scope, scope_label=label, mass_points=len(group),
                           min_fixed_over_current=float(ratio.min()),
                           median_fixed_over_current=float(ratio.median()),
                           max_fixed_over_current=float(ratio.max()),
                           fixed_limit_smaller_count=int((ratio < 1).sum()),
                           min_fixed_over_profiled_asimov=float(aratio.min()),
                           median_fixed_over_profiled_asimov=float(aratio.median()),
                           max_fixed_over_profiled_asimov=float(aratio.max())))
    pd.DataFrame(minima).to_csv(OUT/'local_p0_minima.csv', index=False)
    pd.DataFrame(summary_rows).to_csv(OUT/'scope_summary.csv', index=False)
    changed = [path for path, digest in protected.items() if production.sha256(Path(path)) != digest]
    if changed:
        raise RuntimeError(f'Protected source changed: {changed}')
    if len(frame) != 456 or len(predictions_ledger) != 415:
        raise RuntimeError('Output totals differ from protocol')
    payload = dict(status='complete', scope_rows=len(frame), native_predictions=415,
                   new_observed_fixed_limits=456, conditional_asimov_limits=912,
                   new_toys=0, source_hashes_unchanged=True,
                   all_native_prediction_hashes_exact=True,
                   independent_fixed_mle_and_limit_checks=456,
                   max_independent_fixed_A90_relative_difference=float(max(x['A90_relative_difference'] for x in checks)),
                   max_independent_fixed_Ahat_abs_difference=float(max(x['Ahat_abs_difference'] for x in checks)),
                   max_independent_fixed_r_abs_difference=float(max(x['r_abs_difference'] for x in checks)),
                   prior_2021_rows_checked=201,
                   prior_2021_fixed_limit_max_relative_difference=float(frame.prior_2021_fixed_limit_relative_delta.abs().max()),
                   prior_2021_fixed_r_max_abs_difference=float(frame.prior_2021_fixed_r_delta.abs().max()),
                   p0_fixed_underflow_rows=int((frame.p0_fixed == 0).sum()),
                   all_log_p0_finite=bool(np.isfinite(frame.log_p0_fixed).all()),
                   dimuon_factor_at_250MeV=correction(250),
                   sources=protected, native_histograms=histograms,
                   runtime_provenance=production.RUNTIME_PROVENANCE,
                   runtime_import_origins=production.RUNTIME_IMPORT_ORIGINS,
                   inherited_2016_status='conditional_user_accepted_numerical_exception',
                   interpretation='Fixed GP mean ignores GP estimation uncertainty; no coverage or global discovery claim.',
                   scopes=summary_rows, minima=minima,
                   elapsed_seconds=time.monotonic()-started)
    (OUT/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ('sources', 'runtime_provenance', 'runtime_import_origins', 'native_histograms')}, indent=2))


if __name__ == '__main__':
    main()
