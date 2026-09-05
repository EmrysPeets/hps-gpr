#!/usr/bin/env python3
"""Typeset four-scope conditional observed and Asimov comparisons."""
from pathlib import Path
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator, LogFormatterMathtext
import numpy as np
import pandas as pd

DATA = HERE/'derived'
FIGURES = HERE/'figures'
BLUE, RED = '#2166ac', '#b2182b'
SCOPES = [('individual_2015_full', '2015, full sample'),
          ('individual_2016_full', '2016, full sample'),
          ('individual_2021_10pct', '2021, 10% sample'),
          ('all_2015_2016_2021', 'All three, shared coupling')]
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix', 'font.size': 11, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'grid.linewidth': .5,
    'pdf.fonttype': 42, 'ps.fonttype': 42})


def decorate(ax):
    ax.grid(True, which='major', color='#d9d9d9', alpha=.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))


def save(fig, name):
    fig.savefig(FIGURES/f'{name}.pdf', facecolor='white')
    fig.savefig(FIGURES/f'{name}.png', dpi=200, facecolor='white')
    plt.close(fig)


def limit_grid(frame, asimov=False):
    fig = plt.figure(figsize=(10.6, 8.1))
    outer = fig.add_gridspec(2, 2, left=.09, right=.975, bottom=.145,
                            top=.815, wspace=.24, hspace=.37)
    current = 'eps2_profiled_asimov_display' if asimov else 'eps2_current_display'
    fixed = 'eps2_fixed_asimov_display' if asimov else 'eps2_fixed_display'
    ratio = 'fixed_over_profiled_asimov' if asimov else 'fixed_over_current'
    for index, (scope, label) in enumerate(SCOPES):
        d = frame[frame.scope_key == scope].sort_values('mass_MeV')
        sub = outer[index//2, index%2].subgridspec(2, 1, height_ratios=[3.3, 1], hspace=.04)
        ax = fig.add_subplot(sub[0]); lower = fig.add_subplot(sub[1], sharex=ax)
        ax.plot(d.mass_MeV, d[current], color=BLUE, lw=1.55)
        ax.plot(d.mass_MeV, d[fixed], color=RED, lw=1.4, ls='--')
        ax.set(yscale='log', ylabel=r'90% CL$_s$ limit on $\epsilon^2$', title=label)
        ax.tick_params(labelbottom=False)
        ax.set_ylim(min(d[current].min(), d[fixed].min())*.78,
                    max(d[current].max(), d[fixed].max())*1.22)
        lower.plot(d.mass_MeV, d[ratio], color=RED, lw=1.25)
        lower.axhline(1., color='.4', lw=.7, ls=':')
        lower.set(xlabel=r"Mass hypothesis $m_{A'}$ (MeV)", ylabel='Fixed /\nprofiled')
        if asimov:
            lower.set_ylim(.25, 1.1)
            lower.set_yticks([.5, 1.])
        else:
            lower.set_ylim(0., max(1.15, d[ratio].max()*1.13))
            lower.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        lower.set_xlim(d.mass_MeV.min()-.5, d.mass_MeV.max()+.5)
        for axis in (ax, lower):
            decorate(axis)
    title = ('Conditional background-only Asimov limits' if asimov else
             'Observed limits with and without GP background uncertainty')
    fig.suptitle(title, x=.5, y=.97, fontsize=17)
    fig.text(.5, .927, 'Frozen v4.9.12 inputs; identical masks, signal templates and yield conversion',
             ha='center', fontsize=11)
    baseline = 'Gaussian profile (same solver)' if asimov else 'Released Gaussian profile'
    handles = [Line2D([], [], color=BLUE, lw=1.7, label=baseline),
               Line2D([], [], color=RED, lw=1.6, ls='--', label='Fixed GP mean (uncertainty omitted)')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .898),
               ncol=2, frameon=False, fontsize=11, handlelength=3)
    first = ('Deterministic reference at the GP mean; not expected-toy bands or validated sensitivity.'
             if asimov else 'Fixed-mean curves are conditional diagnostics; observed tightening does not establish improved sensitivity.')
    fig.text(.5, .052, first, ha='center', fontsize=10)
    fig.text(.5, .029, '2016 and the combination inherit the disclosed numerical exception. The dimuon correction is applied once.',
             ha='center', fontsize=10, color='.25')
    save(fig, 'conditional_asimov_limits_four_scopes' if asimov else 'observed_limits_four_scopes')


def pvalue_grid(frame):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8))
    fig.subplots_adjust(left=.10, right=.97, bottom=.15, top=.81, hspace=.4, wspace=.24)
    for ax, (scope, label) in zip(axes.flat, SCOPES):
        d = frame[frame.scope_key == scope].sort_values('mass_MeV')
        ax.plot(d.mass_MeV, d.p0_current, color=BLUE, lw=1.55)
        ax.plot(d.mass_MeV, d.p0_fixed, color=RED, lw=1.4, ls='--')
        exponent = int(np.ceil(-min(d.log_p0_current.min(), d.log_p0_fixed.min())/np.log(10))) + 1
        step = max(1, int(np.ceil(exponent/6)))
        ticks = np.arange(0, exponent+1, step)
        ax.set(yscale='log', ylim=(10.**(-exponent), 1.),
               xlim=(d.mass_MeV.min()-.5, d.mass_MeV.max()+.5),
               xlabel=r"Mass hypothesis $m_{A'}$ (MeV)",
               ylabel=r'Local asymptotic $p_0$', title=label)
        ax.set_yticks(10.**(-ticks))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_locator(NullLocator())
        decorate(ax)
    fig.suptitle('Local asymptotic p-values under the two background treatments', y=.97, fontsize=17)
    fig.text(.5, .925, r'One-sided discovery statistic: $Z=\max(r,0)$, $p_0=1-\Phi(Z)$; no trials correction',
             ha='center', fontsize=11)
    handles = [Line2D([], [], color=BLUE, lw=1.7, label='Released Gaussian profile'),
               Line2D([], [], color=RED, lw=1.6, ls='--', label='Fixed GP mean (uncertainty omitted)')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .893),
               ncol=2, frameon=False, fontsize=11, handlelength=3)
    fig.text(.5, .055, 'Fixed-mean p-values assume that the estimated GP mean is exactly known; they are not calibrated significances.',
             ha='center', fontsize=10)
    fig.text(.5, .031, '2016 and the combination inherit the disclosed numerical exception. Neighboring mass hypotheses are correlated.',
             ha='center', fontsize=10, color='.25')
    save(fig, 'local_asymptotic_pvalues_four_scopes')


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    source = DATA/'observed_fixed_comparison.csv'
    frame = pd.read_csv(source)
    limit_grid(frame)
    pvalue_grid(frame)
    limit_grid(frame, asimov=True)
    paths = sorted(FIGURES.glob('*.pdf')) + sorted(FIGURES.glob('*.png'))
    payload = {'status': 'generated', 'source_csv_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'figure_count': len(paths), 'rows_per_figure': len(frame),
               'files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (DATA/'figure_manifest.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({key: value for key, value in payload.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
