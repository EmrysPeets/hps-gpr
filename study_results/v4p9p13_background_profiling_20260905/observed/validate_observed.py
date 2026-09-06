#!/usr/bin/env python3
"""Check persisted numerical contracts without editing any input artifact."""
from pathlib import Path
import hashlib
import json
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DATA = HERE/'derived'
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = json.loads((DATA/'summary.json').read_text())
    frame = pd.read_csv(DATA/'observed_fixed_comparison.csv')
    fits = pd.read_csv(DATA/'fit_diagnostics.csv')
    checks = pd.read_csv(DATA/'independent_fixed_checks.csv')
    predictions = pd.read_csv(DATA/'prediction_verification.csv')
    released = pd.read_csv(REPO/'study_results/v4p9p12_final_dataset_combinations_20260902/derived/final_dataset_result_curves.csv')
    released = released.set_index(['scope_key', 'mass_MeV'])
    tests = []
    def check(name, truth):
        tests.append({'check': name, 'passed': bool(truth)})
        if not truth:
            raise RuntimeError(name)
    expected = {'individual_2015_full': (19, 90), 'individual_2016_full': (39, 180),
                'individual_2021_10pct': (50, 250), 'all_2015_2016_2021': (50, 90)}
    check('456 unique requested scope/mass coordinates', len(frame) == 456 and not frame.duplicated(['scope_key', 'mass_MeV']).any())
    for scope, (low, high) in expected.items():
        check(f'exact grid {scope}', frame[frame.scope_key == scope].mass_MeV.tolist() == list(range(low, high+1)))
    check('415 exact native prediction hashes', len(predictions) == 415 and predictions.prediction_hash_exact.all())
    check('1368 successful fits/limits across observed and two Asimov models', len(fits) == 1368 and fits.min_lambda.min() > 0 and fits.max_score.max() <= 2e-7)
    check('all CLs roots at 0.1', np.max(np.abs(fits.cls-.1)) < 2e-6)
    check('all CLs traces monotone', fits.monotonicity_error.max() <= 5e-5)
    check('456 independent scalar checks', len(checks) == 456 and checks.A90_relative_difference.max() < 2e-7 and checks.r_abs_difference.max() < 2e-6)
    for method in ('current', 'fixed'):
        check(f'{method} local p0 has correct one-sided normal mapping', np.allclose(frame[f'p0_{method}'], norm.sf(frame[f'Z_{method}']), rtol=2e-8, atol=1e-300))
        check(f'{method} log-p0 mapping', np.allclose(frame[f'log_p0_{method}'], norm.logsf(frame[f'Z_{method}']), rtol=1e-13, atol=1e-13))
    check('fixed signed deficits map to p0=0.5', np.all(frame.loc[frame.signed_r_fixed <= 0, 'p0_fixed'] == .5))
    check('all fixed log p0 values finite', np.isfinite(frame.log_p0_fixed).all())
    for row in frame.itertuples():
        baseline = released.loc[(row.scope_key, row.mass_MeV)]
        if row.eps2_current_ee_raw != baseline.eps2_90 or row.p0_current != baseline.p0_local_asymptotic:
            raise RuntimeError('Released reference changed')
    check('released observed limits and p0 retained exactly', True)
    check('correct 2016 numerical exception scope', np.array_equal(frame.inherited_2016_numerical_exception, frame.dataset_set.str.contains('2016')))
    check('fixed epistemic status explicit', frame.fixed_background_uncertainty_omitted.all() and frame.conditional_on_frozen_gp.all() and not frame.gp_reoptimized.any())
    for stem in ('current', 'fixed', 'fixed_asimov', 'profiled_asimov'):
        check(f'once-only dimuon conversion for {stem}', np.allclose(frame[f'eps2_{stem}_display'], frame[f'eps2_{stem}_ee_raw']*frame.dimuon_factor, rtol=4e-15))
    check('dimuon factor at 250 MeV matches expanded note', np.isclose(frame[frame.mass_MeV == 250].dimuon_factor.iloc[0], 1.7252323083862526, rtol=1e-14))
    check('Asimov fixed limit is positive and no larger than profiled', (frame.fixed_over_profiled_asimov > 0).all() and (frame.fixed_over_profiled_asimov <= 1).all())
    check('201 existing fixed-2021 results close', frame.prior_2021_fixed_limit_relative_delta.notna().sum() == 201 and frame.prior_2021_fixed_limit_relative_delta.abs().max() < 3e-10)
    check('all protected sources unchanged', all(sha(path) == digest for path, digest in summary['sources'].items()))
    manifest = json.loads((DATA/'figure_manifest.json').read_text())
    check('six static figure artifacts', len(manifest['files']) == 6)
    check('figure hashes unchanged', all(sha(path) == digest for path, digest in manifest['files'].items()))
    check('figures use final CSV', manifest['source_csv_sha256'] == sha(DATA/'observed_fixed_comparison.csv'))
    payload = {'status': 'passed', 'passed_checks': len(tests), 'checks': tests,
               'visual_qa': 'PNG counterparts of all three PDF figures visually inspected; see README.',
               'script_sha256': sha(__file__)}
    (DATA/'validation.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'checks': len(tests)}))


if __name__ == '__main__':
    main()
