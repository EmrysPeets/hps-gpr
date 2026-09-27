"""Fixed-source pointwise sideband diagnostic; all writes stay in this directory.

Reuse 256 saved full Poisson spectra. At each excluded-window center, replay the
archived fixed-kernel GP and evaluate Poisson deviance on fitted search sidebands.
Checkpoint each center. No hyperparameter optimization, new draws or signal fits.
"""
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
for folder in ('assets', 'data', 'provenance'):
    (OUT / folder).mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT / '.mpl-cache')

import csv
import hashlib
import json
import time
import numpy as np
from scipy.special import xlogy
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = ROOT / 'study_results/v5p0p5_analysis_note_20260916'
NULL_FILE = ROOT / 'study_results/v5p9p5_null_bias_20260922/residual_diagnostic/inputs/null_2021.npz'
OLD = ROOT / 'output/slides/unblind_meeting_RCmeet_20260923/science'
sys.path.insert(0, str(SRC / 'scripts'))
from common import DATA, predict, kernel_state, sigma

paths = [SRC / 'inputs' / f'spectrum_{year}.npz' for year in ('2015', '2016', '2021')]
paths += [SRC / 'scripts' / f for f in ('common.py', 'parent_core.py', 'limit_solver.py')]
paths += [SRC / 'inputs/scopes.json', NULL_FILE,
          OLD / 'data/slide14_conditional_sideband_toys78.csv',
          OLD / 'data/slide14_observed_sideband_metrics.csv']
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths}

data = DATA['2021']
null = np.load(NULL_FILE)
assert np.array_equal(null['observed'], data['n'])
assert np.array_equal(null['edges_GeV'], data['edges'])
search = (data['x'] * 1000 >= 50) & (data['x'] * 1000 <= 250)
centers = np.unique(np.r_[np.arange(50, 251, 5), 78]).astype(int)
n_toys = len(null['counts'])
assert n_toys == 256

def interval(k, n):
    return [float(beta.ppf(.025, k, n-k+1)) if k else 0.,
            float(beta.ppf(.975, k+1, n-k)) if k < n else 1.]

def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def deviance(counts, blind, side, constant, length_scale):
    mean, _ = predict(data['x'], counts, blind, constant, length_scale,
                      query=data['x'][side])
    n = np.asarray(counts)[side]
    assert np.all(mean > 0)
    value = float(2 * np.sum(xlogy(n, n / mean) - n + mean))
    assert np.isfinite(value) and value >= 0
    return value / int(side.sum())

checkpoint = OUT / 'data/checkpoint.npz'
observed = np.full(len(centers), np.nan)
toy_values = np.full((len(centers), n_toys), np.nan)
bin_counts = np.zeros(len(centers), dtype=int)
replay_seconds = np.zeros(len(centers))
completed = np.zeros(len(centers), dtype=bool)
if checkpoint.exists():
    saved = np.load(checkpoint)
    assert np.array_equal(saved['centers'], centers)
    assert str(saved['input_hashes']) == json.dumps(hashes, sort_keys=True)
    observed = saved['observed'].copy()
    toy_values = saved['toy_values'].copy()
    bin_counts = saved['bin_counts'].copy()
    replay_seconds = saved['replay_seconds'].copy()
    completed = saved['completed'].copy()

started = time.monotonic()
for j, center in enumerate(centers):
    if completed[j]:
        continue
    tick = time.monotonic()
    blind = np.abs(data['x'] - center / 1000.) <= 2.25 * sigma('2021', float(center))
    side = search & ~blind
    assert not np.any(side & blind)
    constant, length_scale = kernel_state('2021', float(center))
    bin_counts[j] = side.sum()
    observed[j] = deviance(data['n'], blind, side, constant, length_scale)
    if center == 78:
        with (OLD / 'data/slide14_conditional_sideband_toys78.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row['toy_id']) for row in rows] == list(range(n_toys))
        toy_values[j] = [float(row['D_per_side_bin']) for row in rows]
        # Two direct replays verify compatibility with the reused saved column.
        for index in (0, n_toys-1):
            current = deviance(null['counts'][index], blind, side, constant, length_scale)
            assert np.isclose(current, toy_values[j, index], rtol=0, atol=1e-12)
    else:
        for index, counts in enumerate(null['counts']):
            toy_values[j, index] = deviance(counts, blind, side, constant, length_scale)
    replay_seconds[j] = time.monotonic() - tick
    completed[j] = True
    staged_checkpoint = OUT / 'data/checkpoint.tmp.npz'
    np.savez_compressed(staged_checkpoint, centers=centers, observed=observed,
                        toy_values=toy_values, bin_counts=bin_counts,
                        completed=completed, replay_seconds=replay_seconds,
                        input_hashes=json.dumps(hashes, sort_keys=True))
    staged_checkpoint.replace(checkpoint)
    print(f'{j+1}/{len(centers)} center={center} MeV, {replay_seconds[j]:.2f} s; '
          f'elapsed {time.monotonic()-started:.1f} s', flush=True)

q05, median, q95 = np.quantile(toy_values, [.05, .5, .95], axis=1)
rows = []
for j, center in enumerate(centers):
    resolution = sigma('2021', float(center)) * 1000
    constant, length_scale = kernel_state('2021', float(center))
    blind = np.abs(data['x'] - center / 1000.) <= 2.25 * resolution / 1000
    upper = int(np.sum(toy_values[j] >= observed[j]))
    lower = int(np.sum(toy_values[j] <= observed[j]))
    hi_interval = interval(upper, n_toys)
    lo_interval = interval(lower, n_toys)
    rows.append(dict(center_MeV=int(center), N_side_bins=int(bin_counts[j]),
                     sigma_MeV=float(resolution),
                     blind_lower_MeV=float(center-2.25*resolution),
                     blind_upper_MeV=float(center+2.25*resolution),
                     N_training_bins=int((~blind).sum()),
                     kernel_constant=float(constant), kernel_length_scale=float(length_scale),
                     observed_D_side=float(observed[j] * bin_counts[j]),
                     observed_D_per_bin=float(observed[j]),
                     toy_mean=float(toy_values[j].mean()),
                     toy_q05=float(q05[j]), toy_median=float(median[j]),
                     toy_q95=float(q95[j]), upper_exceedances=upper,
                     upper_tail_k_over_N=upper/n_toys,
                     upper_tail_addone=(upper+1)/(n_toys+1),
                     upper_cp95_low=hi_interval[0], upper_cp95_high=hi_interval[1],
                     lower_exceedances=lower, lower_tail_k_over_N=lower/n_toys,
                     lower_tail_addone=(lower+1)/(n_toys+1),
                     lower_cp95_low=lo_interval[0], lower_cp95_high=lo_interval[1],
                     outside_pointwise_90=bool(observed[j] < q05[j] or observed[j] > q95[j])))
write_csv(OUT / 'data/sideband_center_summary.csv', rows)
write_csv(OUT / 'data/paired_toy_deviance.csv',
          [dict(toy_id=i, **{f'm{center}_MeV': float(toy_values[j, i])
                            for j, center in enumerate(centers)}) for i in range(n_toys)])
assert np.all(completed) and np.all(np.isfinite(toy_values))
assert all(sha(ROOT / name) == value for name, value in hashes.items())
with (OLD / 'data/slide14_observed_sideband_metrics.csv').open() as handle:
    prior_rows = list(csv.DictReader(handle))
for row in prior_rows:
    j = int(np.flatnonzero(centers == int(row['anchor_MeV']))[0])
    assert np.isclose(observed[j], float(row['D_per_side_bin']), rtol=0, atol=1e-12)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                     'axes.titlesize': 15, 'axes.labelsize': 14,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .15, 'pdf.fonttype': 42})
BLUE, RED = '#1b77a4', '#bb3636'
def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / 'assets' / (name + '.' + ext), dpi=240,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)

fig, ax = plt.subplots(figsize=(11.5, 4.0))
ax.fill_between(centers, q05, q95, color=BLUE, alpha=.16,
                label='256 toys: pointwise central 90%')
ax.plot(centers, median, color=BLUE, lw=2, ls='--', label='Conditional toy median')
ax.plot(centers, observed, color=RED, lw=2.1, marker='o', ms=3.3,
        label='Observed sideband deviance')
ax.set(xlim=(50, 250), xlabel='Excluded-window center [MeV]',
       ylabel=r'$D_{\rm side}/N_{\rm side}$',
       title='2021 10% · sideband fit across the search region')
ax.legend(loc='upper center', ncol=3, frameon=False, fontsize=10.5)
low, high = min(q05.min(), observed.min()), max(q95.max(), observed.max())
ax.set_ylim(low-.055, high+.12)
fig.tight_layout()
save(fig, 'slide14_sideband_scan')

fig, ax = plt.subplots(figsize=(11.5, 3.5))
lo = np.array([row['lower_tail_k_over_N'] for row in rows])
lo_l = np.array([row['lower_cp95_low'] for row in rows])
lo_h = np.array([row['lower_cp95_high'] for row in rows])
ax.fill_between(centers, lo_l, lo_h, color=BLUE, alpha=.16,
                label='95% binomial intervals at each center')
ax.plot(centers, lo, color=BLUE, lw=2, marker='o', ms=3,
        label='Fraction of toys with smaller deviance')
ax.axhline(.05, color='.45', ls=':', lw=1)
ax.set(xlim=(50, 250), ylim=(0, max(.3, lo_h.max()+.17)),
       xlabel='Excluded-window center [MeV]', ylabel='Lower-tail fraction',
       title='2021 10% · position within each conditional reference')
ax.legend(loc='upper right', frameon=False, fontsize=11)
fig.tight_layout()
save(fig, 'sideband_lower_tail_reference')

summary = dict(dataset='2021 10%', centers_MeV=centers.tolist(), n_centers=len(centers),
               regular_spacing_MeV=5, supplemental_reference_center_MeV=78,
               N_toys=n_toys, N_side_range=[int(bin_counts.min()), int(bin_counts.max())],
               observed_range=[float(observed.min()), float(observed.max())],
               median_range=[float(median.min()), float(median.max())],
               lower_tail_range=[float(lo.min()), float(lo.max())],
               outside_pointwise90_centers=[row['center_MeV'] for row in rows if row['outside_pointwise_90']],
               minimum_observed_center_MeV=int(centers[np.argmin(observed)]),
               maximum_observed_center_MeV=int(centers[np.argmax(observed)]),
               center78=next(row for row in rows if row['center_MeV'] == 78),
               total_replay_seconds=float(replay_seconds.sum()),
               tail_intervals='Clopper-Pearson 95%, finite toy-count uncertainty only',
               inference_scope='Pointwise conditional comparisons only; not simultaneous/global calibration.')
(OUT / 'data/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
protocol = dict(input_hashes=hashes, all_parent_inputs_and_code_unchanged=True,
                script_sha256=sha(Path(__file__)),
                quantile_method='NumPy linear interpolation, evaluated separately at each center',
                method='GP trains all available support outside each ±2.25 sigma window. '
                       'Poisson deviance evaluates search bins 50–250 MeV outside the same window. '
                       'Each center uses its archived kernel state, fixed across toys; count-dependent '
                       'training noise and log targets are recomputed for each saved spectrum.',
                support_bin_edges_MeV=(data['edges'][[0,-1]]*1000).tolist(),
                source='256 paired full Poisson spectra from the frozen observed-data-derived '
                       'nominal GP source used by v5.9.5, reused across all centers.',
                definition='D_side=2 sum_side[n_i log(n_i/bhat_i)-n_i+bhat_i]; '
                           'N_side is bin count, not effective degrees of freedom.',
                training_and_evaluation_overlap=True,
                hyperparameter_optimizations=0, new_poisson_spectra=0,
                new_toy_GP_replays=int(41*256+2), reused_toy_metrics_at78=256,
                observed_GP_replays=len(centers), thread_limit=1,
                checkpoint='data/checkpoint.npz',
                checks=['observed and bin edges match saved toy inputs exactly',
                        '65, 78, 120 MeV observed values reproduce prior display to1e-12',
                        'first and last78 MeV toy replays match reused column to1e-12',
                        'sideband bins never intersect the excluded window',
                        'all completed values finite and positive', 'parent hashes unchanged'],
                caveats=['In-sample sideband fit, not held-out-window prediction.',
                         'Fixed observed-data-derived source and kernel states condition the reference.',
                         'Adjacent centers and tail estimates are strongly dependent.',
                         'Pointwise90% bands and95% tail intervals are not simultaneous.',
                         'No fit degrees of freedom or automatic chi-square target of1 is assumed.',
                         'Source/model uncertainty and anchor selection are not calibrated.'],
                summary=summary)
(OUT / 'provenance/protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
