#!/usr/bin/env python3
"""Audit saved extraction likelihoods and displays without fitting or drawing."""
from pathlib import Path
import hashlib
import json
import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
DERIVED = HERE/'derived'
checked = 0
failures = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, value):
    global checked
    checked += 1
    if not bool(value):
        failures.append(name)


def near(a, b, atol=1e-7, rtol=2e-12):
    return np.allclose(a, b, atol=atol, rtol=rtol)


def objective(n, b, L, fitted_background, total, unit, theta):
    positive = n > 0
    x = (total[positive]-n[positive])/n[positive]
    nll = float(np.sum(n[positive]*(x-np.log1p(x)))
                +total[~positive].sum()+.5*theta@theta)
    weight = n/total**2
    cross = L.T@(weight*unit)
    nuisance_hessian = (L.T*weight)@L+np.eye(len(n))
    information = float(unit@(weight*unit)
                        -cross@np.linalg.solve(nuisance_hessian, cross))
    gradient = L.T@((total-n)/total)+theta
    return nll, information, float(np.max(np.abs(gradient)))


def selected(frame, resolutions, count=2):
    frame = frame.sort_values('mass_MeV').reset_index(drop=True)
    values = frame.r.to_numpy()
    candidates = [i for i, value in enumerate(values) if value > 0
        and (i == 0 or value >= values[i-1])
        and (i == len(values)-1 or value >= values[i+1])]
    output = []
    for i in sorted(candidates, key=lambda j: (-values[j], int(frame.iloc[j].mass_MeV))):
        mass = int(frame.iloc[i].mass_MeV)
        if all(abs(mass-other) > 2.25*max(resolutions(mass), resolutions(other))
               for other in output):
            output.append(mass)
        if len(output) == count:
            break
    return output


def audit_map(name, W, edges, mask, displayed_edges):
    check(name+':binary', np.isin(W, (0., 1.)).all())
    check(name+':disjoint', np.all(W.sum(axis=0) <= 1))
    check(name+':inside_fit', not np.any(W[:, ~mask]))
    check(name+':dimensions', W.shape == (len(displayed_edges), len(mask)))
    for i, (low, high) in enumerate(displayed_edges):
        indices = np.flatnonzero(W[i])
        check(name+':contiguous:'+str(i), len(indices) > 0 and
              np.array_equal(indices, np.arange(indices[0], indices[-1]+1)))
        check(name+':edges:'+str(i), near([edges[indices[0]], edges[indices[-1]+1]],
              [low, high], atol=1e-8, rtol=0))


def main():
    provenance = json.loads((HERE/'provenance/extraction.json').read_text())
    for kind in ('input_sha256', 'output_sha256'):
        for relative, digest in provenance[kind].items():
            check(kind+':'+relative, sha(ROOT/relative) == digest)
    arrays = np.load(DERIVED/'fit_arrays.npz')
    summary = pd.read_csv(DERIVED/'fit_summary.csv', dtype={'dataset': str}, float_precision='round_trip')
    consistency = pd.read_csv(DERIVED/'dataset_consistency.csv', dtype={'dataset': str}, float_precision='round_trip')
    bins = pd.read_csv(DERIVED/'display_bins.csv', dtype={'panel': str}, float_precision='round_trip')
    infos = pd.read_csv(DERIVED/'information.csv', dtype={'dataset': str}, float_precision='round_trip')
    closure = json.loads((DERIVED/'fit_closure.json').read_text())
    selection = json.loads((DERIVED/'selection.json').read_text())
    V12 = ROOT/'study_results/v4p9p12_expanded_snapshot_20260905/derived'
    old = pd.read_csv(V12/'selected_fit_plot_data.csv', dtype={'dataset': str}, float_precision='round_trip')
    resolutions = pd.read_csv(V12/'nominal_mass_resolutions.csv', dtype={'dataset': str}).set_index(['dataset', 'mass_MeV']).sigma_MeV
    dense = pd.read_csv(ROOT/'study_results/v4p9p13_calibration_20260905/summary/observed_calibrated_limits.csv')
    union = pd.read_csv(ROOT/'study_results/v4p9p16_combined_global_20260906/global/observed.csv')
    ledger = pd.read_csv(ROOT/'study_results/v4p9p12_final_dataset_combinations_20260902/derived/prediction_state_ledger.csv', dtype={'dataset': str}).set_index(['dataset', 'mass_MeV'])
    check('fit_inventory', len(closure['checks']) == len(selection['fits']) == 15 and len(summary) == len(consistency) == 24)
    check('fit_ids', set(summary.fit_id) == {x['fit_id'] for x in selection['fits']})
    for record in selection['rankings']:
        group = record['group']
        if group == 'combined':
            frame = union.rename(columns={'profiled_r': 'r'})
            sigma = lambda m: max(resolutions.loc[k, m]
                  for k in union.set_index('mass_MeV').loc[m, 'dataset_set'].split('+'))
            check('selection:multi', selected(frame[frame.n_active >= 2], sigma)
                  == record['multidataset_positive_peaks_MeV'])
        else:
            scope = 'individual_'+group+('_10pct' if group == '2021' else '_full')
            frame = dense[dense.scope_key == scope].rename(columns={'signed_r_profiled_asymptotic': 'r'})
            sigma = lambda m, key=group: resolutions.loc[key, m]
        check('selection:'+group, selected(frame, sigma) == record['positive_peaks_MeV'])
        check('deficit:'+group, int(frame.loc[frame.r.idxmin(), 'mass_MeV']) == record['deepest_deficit_MeV'])
    diagnostics = []
    retained = []
    max_nll_error = max_root_error = max_independent_root_error = max_sigma_error = 0.
    max_information_error = max_nuisance_gradient = max_display_mean_difference = 0.
    for record in closure['checks']:
        fid = record['fit_id']
        rows = summary[summary.fit_id == fid]
        common_nll = null_nll = independent_nll = observed_information = prior_information = signal_sum = conversion_sum = 0.
        amplitude_gradient = sum_background = 0.
        common_values = {name: None for name in ('observed', 'gp_mean', 'profiled_background', 'signal', 'total', 'null_background')}
        individual_results = []
        for row in rows.itertuples():
            prefix = fid+'__'+row.dataset+'__'
            get = lambda name: arrays[prefix+name]
            edges, mask = get('edges'), get('mask')
            n, b, L, C, unit = get('fit_counts'), get('fit_gp_mean'), get('fit_factor'), get('fit_covariance'), get('signal_unit')
            check(prefix+'exact_count_representation', np.array_equal(n, get('observed')[mask]))
            max_display_mean_difference = max(max_display_mean_difference, float(np.max(abs(b-get('gp_mean')[mask]))))
            independent = consistency[(consistency.fit_id == fid) & (consistency.dataset == row.dataset)].iloc[0]
            old_data = old[old.dataset == row.dataset]
            old_data = old_data[old_data.fit_id == old_data.iloc[0].fit_id].sort_values('bin_center_MeV')
            check(prefix+'released_observed_counts', np.array_equal(get('observed'), old_data.observed))
            check(prefix+'released_edges', near((edges[:-1]+edges[1:])/2, old_data.bin_center_MeV, atol=1e-8))
            check(prefix+'prediction_hash', row.prediction_state_sha256 == ledger.loc[(row.dataset, row.mass_MeV), 'prediction_state_sha256'])
            check(prefix+'integer_counts', np.all(get('observed') >= 0) and np.array_equal(get('observed'), np.rint(get('observed'))))
            check(prefix+'mask_size', int(mask.sum()) == row.n_fit_bins)
            check(prefix+'fit_span', near([edges[np.flatnonzero(mask)[0]], edges[np.flatnonzero(mask)[-1]+1]], [row.fit_low_MeV, row.fit_high_MeV]))
            check(prefix+'factor_covariance', np.array_equal(C, L@L.T) and np.all(np.diag(L) > 0))
            check(prefix+'factor_triangular', not np.any(np.triu(L, 1)))
            check(prefix+'positive_expectations', np.all(get('total')[mask] > 0) and np.all(get('null_background')[mask] > 0) and np.all(get('independent_total') > 0))
            check(prefix+'no_profile_outside_window', all(np.isnan(get(name)[~mask]).all()
                  for name in ('profiled_background', 'total', 'null_background')))
            check(prefix+'signal_scale', near(get('signal')[mask], record['eps2_hat']*unit))
            check(prefix+'total_arithmetic', near(get('profiled_background')[mask]+get('signal')[mask], get('total')[mask]))
            check(prefix+'independent_total_arithmetic', near(get('independent_background')+independent.individual_eps2_hat*unit, get('independent_total')))
            gaussian = np.maximum(0., np.diff(norm.cdf((edges-row.mass_MeV)/row.sigma_MeV)))
            gaussian /= gaussian.sum()
            check(prefix+'Gaussian_integrated_shape', near(get('signal')/get('signal').sum(), gaussian, atol=3e-14, rtol=1e-10))
            check(prefix+'yield_metadata', near([get('signal')[mask].sum(), get('signal').sum(), unit.sum()],
                  [row.signal_window, row.signal_full, row.signal_yield_per_eps2_window]))
            check(prefix+'full_yield_conversion', near(row.signal_full, row.eps2_hat*row.signal_yield_per_eps2_full))
            signal_sum += float(get('signal')[mask].sum())
            conversion_sum += float(unit.sum())
            for bname, tname in [('profiled_background', 'common_free_theta'),
                                ('null_background', 'common_null_theta')]:
                check(prefix+bname+'_nuisance_identity', near(b+L@get(tname), get(bname)[mask], atol=1e-8))
            for bname, tname in [('independent_background', 'independent_free_theta'),
                                ('independent_null_background', 'independent_null_theta')]:
                check(prefix+bname+'_nuisance_identity', near(b+L@get(tname), get(bname), atol=1e-8))
            f = objective(n, b, L, get('profiled_background')[mask], get('total')[mask], unit, get('common_free_theta'))
            z = objective(n, b, L, get('null_background')[mask], get('null_background')[mask], unit, get('common_null_theta'))
            independent_fit = objective(n, b, L, get('independent_background'), get('independent_total'), unit, get('independent_free_theta'))
            independent_null = objective(n, b, L, get('independent_null_background'), get('independent_null_background'), unit, get('independent_null_theta'))
            max_nuisance_gradient = max(max_nuisance_gradient, *(x[2] for x in (f, z, independent_fit, independent_null)))
            check(prefix+'nuisance_scores', all(x[2] < 2.01e-7 for x in (f, z, independent_fit, independent_null)))
            amplitude_gradient += float(unit@((get('total')[mask]-n)/get('total')[mask]))
            sum_background += float(b.sum())
            independent_amplitude_gradient = float(unit@((get('independent_total')-n)/get('independent_total')))*np.sqrt(b.sum())/unit.sum()
            check(prefix+'independent_amplitude_score', abs(independent_amplitude_gradient) < 2.01e-7)
            common_nll += f[0]; null_nll += z[0]; independent_nll += independent_fit[0]; observed_information += f[1]
            ir = float(np.sign(independent.individual_eps2_hat)*np.sqrt(max(0., 2*(independent_null[0]-independent_fit[0]))))
            max_independent_root_error = max(max_independent_root_error, abs(ir-independent.individual_r))
            check(prefix+'independent_NLL', near(independent_fit[0], independent.individual_nll, atol=1e-5, rtol=0))
            check(prefix+'independent_root', abs(ir-independent.individual_r) < 2e-5)
            check(prefix+'independent_curvature', near(1/np.sqrt(independent_fit[1]), independent.individual_sigma_eps2, atol=0, rtol=1e-10))
            info = float(unit@np.linalg.solve(np.diag(b)+C, unit))
            prior_information += info
            saved_info = infos[(infos.fit_id == fid) & (infos.dataset == row.dataset)].iloc[0].information
            max_information_error = max(max_information_error, abs(info/saved_info-1))
            check(prefix+'precision_information', near(info, saved_info, atol=0, rtol=1e-12))
            W, display_edges = get('display_map'), get('display_edges')
            audit_map(prefix+'display', W, edges, mask, display_edges)
            step = np.diff(edges)[0]
            k = max(1, int(np.floor(.5*row.sigma_MeV/step+.5)))
            check(prefix+'resolution_grouping', k == row.display_native_bins_per_bin and np.all(W.sum(axis=1) == k)
                  and all(np.flatnonzero(line)[0] % k == 0 for line in W))
            check(prefix+'display_fraction', near((W@get('observed')).sum()/n.sum(), row.display_observed_fraction, atol=0))
            exported = bins[(bins.fit_id == fid) & (bins.panel == row.dataset)].sort_values('bin')
            check(prefix+'display_bin_edges', near(display_edges, exported[['low_MeV', 'high_MeV']]))
            for name in common_values:
                values = W@np.nan_to_num(get(name), nan=0.)
                check(prefix+'display_'+name, near(values, exported[name]))
            if len(rows) > 1:
                W = get('common_map'); display_edges = arrays[fid+'__sum__display_edges']
                audit_map(prefix+'common', W, edges, mask, display_edges)
                check(prefix+'common_lattice', near((display_edges-36)/1.25, np.rint((display_edges-36)/1.25), atol=1e-8))
                for name in common_values:
                    values = W@np.nan_to_num(get(name), nan=0.)
                    common_values[name] = values if common_values[name] is None else common_values[name]+values
                retained.append(dict(fit_id=fid, dataset=row.dataset,
                      fit_bins=int(mask.sum()), retained_native_bins=int(W.sum()),
                      bin_fraction=float(W.sum()/mask.sum()),
                      count_fraction=float((W@get('observed')).sum()/n.sum()),
                      signal_fraction=float((W@get('signal')).sum()/get('signal')[mask].sum())))
            individual_results.append(dict(dataset=row.dataset, eps2=float(independent.individual_eps2_hat),
                  sigma_eps2=float(independent.individual_sigma_eps2), root=ir))
        if len(rows) > 1:
            exported = bins[(bins.fit_id == fid) & (bins.panel == 'sum')].sort_values('bin')
            for name, values in common_values.items():
                check(fid+':sum_'+name, near(values, exported[name]))
            check(fid+':sum_integer_counts', np.array_equal(common_values['observed'], np.rint(common_values['observed'])))
        check(fid+':common_signal_yield', near(signal_sum, record['Ahat']))
        check(fid+':common_amplitude_score', abs(amplitude_gradient*np.sqrt(sum_background)/record['conversion']) < 2.01e-7)
        check(fid+':common_conversion', near(conversion_sum, record['conversion']))
        check(fid+':common_coupling', near(signal_sum/conversion_sum, record['eps2_hat'], atol=0))
        root = float(np.sign(record['eps2_hat'])*np.sqrt(max(0., 2*(null_nll-common_nll))))
        compatibility = max(0., 2*(common_nll-independent_nll))
        nll_error = abs(common_nll-record['common_nll'])
        root_error = abs(root-record['root'])
        sigma_error = abs((1/np.sqrt(observed_information))/record['sigma_eps2']-1)
        max_nll_error = max(max_nll_error, nll_error); max_root_error = max(max_root_error, root_error)
        max_sigma_error = max(max_sigma_error, sigma_error)
        check(fid+':common_NLL', nll_error < 1e-5)
        check(fid+':signed_root', root_error < 2e-5 and abs(root-record['reference_root']) < 2e-5)
        check(fid+':common_curvature', sigma_error < 1e-10)
        check(fid+':common_independent_NLL', abs(compatibility-record['individual_common_deviance']) < 2e-5)
        check(fid+':information_sum', near(prior_information, record['sum_information'], atol=0, rtol=1e-12))
        diagnostics.append(dict(fit_id=fid, root=root, root_error=root_error,
              NLL_error=nll_error, eps2=record['eps2_hat'], sigma_eps2=record['sigma_eps2'],
              common_independent_deviance=compatibility, df=len(rows)-1,
              nominal_fixed_mass_chi2_p=float(chi2.sf(compatibility, len(rows)-1)) if len(rows)>1 else None,
              individual=individual_results))
    retained_table = pd.read_csv(DERIVED/'common_display_retention.csv', dtype={'dataset': str}, float_precision='round_trip')
    check('common_retention_row_count', len(retained_table) == len(retained))
    for item in retained:
        row = retained_table[(retained_table.fit_id == item['fit_id']) & (retained_table.dataset == item['dataset'])].iloc[0]
        for field, column in [('fit_bins', 'native_fit_bins'), ('retained_native_bins', 'retained_native_bins'),
                              ('bin_fraction', 'native_bin_fraction'), ('count_fraction', 'observed_fraction'),
                              ('signal_fraction', 'fitted_signal_fraction')]:
            check(item['fit_id']+':retention:'+item['dataset']+':'+field,
                  near(item[field], row[column], atol=0, rtol=1e-12))
    exposure = pd.read_csv(DERIVED/'exposure_display_bins.csv', float_precision='round_trip')
    yields = pd.read_csv(DERIVED/'exposure_signal_yields.csv', float_precision='round_trip')
    precision = pd.read_csv(DERIVED/'exposure_precision.csv', float_precision='round_trip')
    exposure_contract = json.loads((DERIVED/'exposure_contract.json').read_text())
    check('exposure_no_new_data', exposure_contract['new_toys'] == exposure_contract['new_unblinded_events'] == 0)
    exposure_diagnostics = []
    for mass in (66, 92):
        fid = f'combined_m{mass:03d}'
        reference = bins[(bins.fit_id == fid) & (bins.panel == '2021')].set_index('bin')
        for (view, factor), view_rows in exposure[exposure.mass_MeV == mass].groupby(['view', 'exposure_factor']):
            check(fid+':exposure_bin_count:'+view+':'+str(factor), len(view_rows) == len(reference))
            for row in view_rows.itertuples():
                original = reference.loc[row.bin]
                n, b, signal = original.observed, original.profiled_background, original.signal
                added = factor if view == 'increment' else factor-1
                null = 0. if view == 'increment' else n-b
                persistent = null+added*signal
                label = fid+':'+view+':'+str(factor)+':'+str(row.bin)
                check(label+':source', near([row.original_observed, row.reference_background, row.assumed_signal_per_10pct], [n, b, signal]))
                check(label+':means', near([row.null_residual, row.persistent_residual, row.added_factor], [null, persistent, added]))
                check(label+':conditional_counting_variance', near(row.added_background_counting_sd**2, added*b))
                check(label+':future_status', row.future_view == (view != 'observed'))
                check(label+':positive_future_mean', b > 0 and b+signal > 0)
        reference = summary[(summary.fit_id == fid) & (summary.dataset == '2021')].iloc[0]
        for row in yields[yields.mass_MeV == mass].itertuples():
            check(fid+':yield:'+str(row.exposure_percent), near(row.template_yield_window,
                  row.exposure_percent/10*reference.signal_window) and near(row.assumed_eps2, reference.eps2_hat, atol=0))
        component = infos[infos.fit_id == fid].set_index('dataset').information
        current = component.sum(); i21 = component['2021']; iold = current-i21
        ratios = {}
        for row in precision[precision.mass_MeV == mass].itertuples():
            factor = row.exposure_percent/10
            gain = np.sqrt((iold+factor*i21)/current)
            check(fid+':precision:'+str(row.exposure_percent), near(
                  [row.combined_precision_gain, row.combined_uncertainty_ratio,
                   row.original_2021_information_fraction, row.individual_2021_precision_gain],
                  [gain, 1/gain, i21/current, np.sqrt(factor)], atol=0))
            ratios[str(row.exposure_percent)] = float(1/gain)
        exposure_diagnostics.append(dict(mass_MeV=mass, initial_2021_information_fraction=float(i21/current),
              combined_uncertainty_ratio=ratios,
              interpretation='Conditional future counting envelopes are distinct from the separate fC precision illustration.'))
    output = dict(passed=not failures, checked_conditions=checked, failures=failures,
          scope='Read-only saved-component reconstruction; no fits, random draws or unreleased-data access.',
          common_root_max_abs_error=max_root_error, independent_root_max_abs_error=max_independent_root_error,
          common_NLL_max_abs_error=max_nll_error, common_sigma_max_relative_error=max_sigma_error,
          information_max_relative_error=max_information_error, maximum_nuisance_gradient=max_nuisance_gradient,
          display_mean_vs_exact_fit_mean_max_counts=max_display_mean_difference,
          nuisance_reconstruction_note='Uses saved exact likelihood mean, counts and nuisance vectors; checks Gaussian penalties and nuisance first-order gradients without fitting.',
          fits=diagnostics, common_display_retention=retained, exposure=exposure_diagnostics,
          input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), HERE/'extract.py',
              HERE/'PROTOCOL.md', DERIVED/'fit_arrays.npz', DERIVED/'fit_summary.csv',
              DERIVED/'fit_closure.json', DERIVED/'display_bins.csv', DERIVED/'dataset_consistency.csv',
              DERIVED/'information.csv', DERIVED/'selection.json', DERIVED/'common_display_retention.csv',
              HERE/'make_figures.py', DERIVED/'exposure_display_bins.csv', DERIVED/'exposure_precision.csv',
              DERIVED/'exposure_signal_yields.csv', DERIVED/'exposure_contract.json']})
    print(json.dumps(output, indent=2, allow_nan=False))
    raise SystemExit(0 if not failures else 1)


if __name__ == '__main__':
    main()
