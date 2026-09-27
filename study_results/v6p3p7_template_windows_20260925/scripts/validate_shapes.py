#!/usr/bin/env python3
"""Deterministic v16 TC interpolation checks and standalone scientific figures."""
from pathlib import Path
import json
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from templates import TemplateBank

B = Path(__file__).resolve().parents[1]
R = B / 'results'; F = B / 'figures'
BLUE = '#245c91'; RED = '#a54b35'; GRAY = '#444444'; GREEN = '#43856b'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
                     'axes.titlesize': 9, 'legend.fontsize': 8, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save(fig, name):
    fig.savefig(F / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(F / (name + '.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)


def run():
    R.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
    bank = TemplateBank()
    edges = np.linspace(0, 400, 4001)
    rows = []
    for mass in range(100, 221, 20):
        center, width = bank.parameters(mass, 'direct')
        truth = bank.categories(mass, edges, 'direct')
        true_cdf = bank.cdf(mass, edges, 'direct')
        cuts = center + width * np.array([-6., -4., -2., 2., 3., 5.])
        truth_cut = bank.cdf(mass, cuts, 'direct')
        core_edges = center + width*np.linspace(-2,2,401)
        truth_core = bank.cdf(mass,core_edges,'direct')
        truth_core = (truth_core-truth_core[0])/(truth_core[-1]-truth_core[0])
        for method in ('common', 'morph'):
            prediction = bank.categories(mass, edges, method, omit=mass)
            prediction_cdf = bank.cdf(mass, edges, method, omit=mass)
            predicted_center, predicted_width = bank.parameters(mass, method, omit=mass)
            predicted_cut = bank.cdf(mass, cuts, method, omit=mass)
            predicted_core = bank.cdf(mass,core_edges,method,omit=mass)
            predicted_core = (predicted_core-predicted_core[0])/(predicted_core[-1]-predicted_core[0])
            row = dict(mass_MeV=mass, method=method,
                       lower_anchor_MeV=mass-20, upper_anchor_MeV=mass+20,
                       true_center_MeV=center, predicted_center_MeV=predicted_center,
                       center_error_MeV=predicted_center-center,
                       true_width_MeV=width, predicted_width_MeV=predicted_width,
                       width_fractional_error=predicted_width/width-1,
                       full_CDF_max_abs_error=float(abs(prediction_cdf-true_cdf).max()),
                       conditional_core_CDF_max_abs_error=float(abs(predicted_core-truth_core).max()),
                       full_histogram_total_variation=float(abs(prediction-truth).sum()/2),
                       truth_core_minus2_plus2=float(truth_cut[3]-truth_cut[2]),
                       predicted_core_minus2_plus2=float(predicted_cut[3]-predicted_cut[2]),
                       truth_blind_minus4_plus3=float(truth_cut[4]-truth_cut[1]),
                       predicted_blind_minus4_plus3=float(predicted_cut[4]-predicted_cut[1]),
                       truth_fit_minus6_plus5=float(truth_cut[5]-truth_cut[0]),
                       predicted_fit_minus6_plus5=float(predicted_cut[5]-predicted_cut[0]),
                       truth_lower_tail_below_minus4=float(truth_cut[1]),
                       predicted_lower_tail_below_minus4=float(predicted_cut[1]),
                       truth_upper_tail_above_plus3=float(1-truth_cut[4]),
                       predicted_upper_tail_above_plus3=float(1-predicted_cut[4]))
            rows.append(row)
    closure = pd.DataFrame(rows)
    closure.to_csv(R / 'template_shape_closure.csv', index=False)
    interpolation = []
    for mass in range(90, 240, 20):
        center, width = bank.parameters(mass)
        cuts = center + width*np.array([-6., -4., -2., 2., 3., 5.])
        for method in ('common', 'morph'):
            cut = bank.cdf(mass, cuts, method)
            interpolation.append(dict(mass_MeV=mass, method=method,
                                      center_MeV=center, width_MeV=width,
                                      lower_anchor_MeV=mass-10, upper_anchor_MeV=mass+10,
                                      core_minus2_plus2=float(cut[3]-cut[2]),
                                      blind_minus4_plus3=float(cut[4]-cut[1]),
                                      fit_minus6_plus5=float(cut[5]-cut[0]),
                                      lower_tail_below_minus4=float(cut[1]),
                                      upper_tail_above_plus3=float(1-cut[4]),
                                      scope='conditional prediction; no direct MC sample at this mass'))
    pd.DataFrame(interpolation).to_csv(R / 'template_intermediate_predictions.csv', index=False)
    checks = []
    for mass in bank.anchors:
        direct = bank.categories(mass, edges, 'direct')
        same = bank.categories(mass, edges, 'morph')
        assert np.max(abs(direct-same)) < 1e-12
        checks.append(dict(check='anchor interpolation identity', mass_MeV=int(mass),
                           max_abs_difference=float(np.max(abs(direct-same)))))
    for mass in np.arange(80, 240.01, .5):
        for method in ('common', 'morph'):
            cats = bank.categories(mass, edges, method)
            assert cats.min() >= 0 and abs(cats.sum()-1) < 1e-12
    for mass, sample in bank.samples.items():
        probs = bank.probabilities(mass, sample['edges'], 'direct')
        assert abs(probs.sum()-sample['counts'].sum()/sample['total']) < 1e-12
        assert abs(bank.categories(mass, sample['edges'], 'direct')[-1]-sample['over']/sample['total']) < 1e-12
    assert np.allclose(bank.parameters(90),
                       np.mean([bank.parameters(80,'direct'),bank.parameters(100,'direct')],axis=0),
                       rtol=0,atol=1e-12)
    summary = dict(
        description='v16 TC aligned empirical-CDF interpolation; full selected normalization',
        formula='F_m(x)=(1-t) F_low(c_low+s_low*(x-c_m)/s_m)+t F_high(c_high+s_high*(x-c_m)/s_m); t=(m-low)/(high-low); c_m and s_m use the same linear interpolation',
        common_formula='Equal-weight mean of all retained anchor CDFs after individual core alignment; target center and width still interpolate adjacent anchors',
        anchors_MeV=bank.anchors.tolist(), omitted_validation_masses_MeV=list(range(100,221,20)),
        full_normalization='All selected rows, including stored histogram overflow. Missing upper-tail locations remain in the above-support category; no fit-window renormalization.',
        interpolation_scope='No extrapolation below 80 or above 240 MeV. Direct samples at 60 and 260 remain available as separate checks.',
        uncertainty_scope='Deterministic empirical-shape diagnostics. No uncertainty bars; finite MC statistics and fitted-core uncertainties are not propagated. Omitted-anchor validation predicts center, width, core and tails without that sample.',
        common_CDF_error_range=closure[closure.method=='common'].full_CDF_max_abs_error.agg(['min','max']).to_dict(),
        morph_CDF_error_range=closure[closure.method=='morph'].full_CDF_max_abs_error.agg(['min','max']).to_dict(),
        common_conditional_core_CDF_error_range=closure[closure.method=='common'].conditional_core_CDF_max_abs_error.agg(['min','max']).to_dict(),
        morph_conditional_core_CDF_error_range=closure[closure.method=='morph'].conditional_core_CDF_max_abs_error.agg(['min','max']).to_dict(),
        core_diagnostic_normalization='Conditional core CDF is normalized only within |u|<2 for a diagnostic of core shape; neither fitted signal probabilities nor full-distribution metrics use this renormalization.',
        passed=True, checks=checks, dense_category_checks=642)
    write(R / 'template_shape_summary.json', summary)

    # The 90 MeV example separates core alignment from tail interpolation.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    uedges = np.linspace(-12, 22, 341); u = (uedges[1:]+uedges[:-1])/2
    for mass, color, ls in [(80, BLUE, '-'), (100, RED, '--')]:
        sample = bank.samples[mass]
        density = bank.probabilities(mass, sample['center']+sample['width']*uedges, 'direct')/np.diff(uedges)
        for ax in axes: ax.plot(u, density, color=color, ls=ls, label=f'{mass} MeV signal MC')
    center, width = bank.parameters(90)
    density = bank.probabilities(90, center+width*uedges, 'morph')/np.diff(uedges)
    for ax in axes:
        ax.plot(u, density, color=GRAY, lw=1.7, label='90 MeV: neighboring-anchor prediction')
        ax.axvspan(-4, 3, color=BLUE, alpha=.07)
        ax.set(xlabel=r'Core-aligned coordinate $u=(m_{rec}-c)/s$', ylabel='Full selected probability per unit u')
        ax.grid(alpha=.15)
    axes[0].set(xlim=(-4, 4), title='Core: align each anchor before interpolation')
    axes[1].set(xlim=(-12, 22), ylim=(1e-5, .3), yscale='log', title='Tails: retain their measured probability')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(top=.78, bottom=.26, left=.08, right=.99, wspace=.27)
    fig.text(.08,.86, f'2021 v16 TC signal MC; 90 MeV prediction: c={center:.3f} MeV, s={width:.3f} MeV', fontsize=10)
    fig.text(.03,.015, 'How to read the figure. At 90 MeV, each aligned CDF receives weight 1/2; center and width also interpolate halfway.\n'
             'Shading marks the proposed [-4, 3] blind region. Areas use all selected signal MC, including overflow in the denominator.\n'
             'Lines have no uncertainty bands. This is a conditional prediction, because no independent 90 MeV signal-MC sample is available.', fontsize=8)
    save(fig, 'v637_neighbor_interpolation')

    # Independent shape prediction at existing sample masses, with target omitted.
    fig, axes = plt.subplots(2,3,figsize=(9.2,5.7))
    uedges = np.linspace(-12,18,301); u=(uedges[1:]+uedges[:-1])/2
    for col, mass in enumerate((100,160,220)):
        center,width=bank.parameters(mass,'direct'); physical_edges=center+width*uedges
        for method, color, ls, label in [('direct',GRAY,'-','Signal MC at omitted mass'),
                                        ('morph',BLUE,'--','Two neighboring anchors'),
                                        ('common',RED,':','Common shape from other anchors')]:
            density=bank.probabilities(mass,physical_edges,method,omit=None if method=='direct' else mass)/np.diff(uedges)
            for row in range(2):axes[row,col].plot(u,density,color=color,ls=ls,lw=1.1,label=label)
        axes[0,col].set(xlim=(-4,4),title=f'{mass} MeV predicted from {mass-20}/{mass+20} MeV')
        axes[1,col].set(xlim=(-12,18),ylim=(1e-5,.3),yscale='log',xlabel=r'$u=(m_{rec}-c_{MC})/s_{MC}$')
        for row in range(2):
            axes[row,col].grid(alpha=.15)
            if col==0:axes[row,col].set_ylabel('Full selected probability / u')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=8)
    fig.suptitle('2021 v16 TC: predict an existing mass after excluding its entire signal-MC sample',y=.935,fontsize=11)
    fig.subplots_adjust(top=.85,bottom=.19,left=.075,right=.99,hspace=.22,wspace=.2)
    fig.text(.03,.012,'How to read the figure. Black is the excluded sample, shown only for checking the prediction; it is never used to build either model.\n'
             'Blue interpolates the two neighboring aligned CDFs. Red averages every other anchor. Both predict center and width from neighbors.\n'
             'Top: core on a linear scale. Bottom: tails on a logarithmic scale. Full selected normalization is retained; no uncertainty bands are shown.\n'
             'The horizontal coordinate uses the measured target core only to display the comparison, not to construct a prediction.',fontsize=8)
    save(fig,'v637_omitted_mass_shapes')

    fig,axes=plt.subplots(2,2,figsize=(8.6,5.9))
    morph=closure[closure.method=='morph']
    axes[0,0].plot(morph.mass_MeV,morph.center_error_MeV,'o-',color=BLUE)
    axes[0,0].set(ylabel='Predicted center − MC center\n[MeV]',title='Core location')
    axes[0,1].plot(morph.mass_MeV,100*morph.width_fractional_error,'o-',color=BLUE)
    axes[0,1].set(ylabel='Predicted width / MC width − 1\n[%]',title='Core width')
    for method,color,label in [('common',RED,'Common aligned shape'),('morph',BLUE,'Two neighboring anchors')]:
        q=closure[closure.method==method]
        axes[1,0].plot(q.mass_MeV,100*q.full_CDF_max_abs_error,'o-',color=color,label=label)
        axes[1,1].plot(q.mass_MeV,100*(q.predicted_blind_minus4_plus3-q.truth_blind_minus4_plus3),'o-',color=color,label=label)
    axes[1,0].set(ylabel='Largest CDF difference\n[percentage points]',title='Full shape and broad tails')
    axes[1,1].set(ylabel='Predicted − MC containment\n[percentage points]',title='Probability in the proposed [-4, 3] region')
    for i,ax in enumerate(axes.flat):
        ax.grid(alpha=.15);ax.axhline(0,color='.65',lw=.6)
        if i>=2:ax.set_xlabel('Omitted generated signal mass [MeV]')
    axes[1,0].legend(frameon=False,fontsize=8)
    fig.suptitle('2021 v16 TC: omitted-mass checks of location, width, core and tails',y=.99,fontsize=11)
    fig.subplots_adjust(top=.91,bottom=.2,left=.1,right=.985,hspace=.35,wspace=.34)
    fig.text(.025,.015,'How to read the figure. Each point excludes the target mass from every construction step. Smaller absolute differences are better.\n'
             'The lower-left value is the largest difference in cumulative full signal probability across 0–400 MeV; it is not a significance.\n'
             'The lower-right window uses the measured target core to compare containment on the same interval. Top panels apply to both models.\n'
             'Lines connect deterministic estimates. No bars are drawn; finite signal-MC and core-fit uncertainties have not been propagated.',fontsize=8)
    save(fig,'v637_shape_closure_metrics')
    captions = {
        'v637_neighbor_interpolation': 'How to read the figure. The 90 MeV model interpolates the 80 and 100 MeV signal-MC samples with equal weights after aligning each fitted core. Both the fitted core location and width interpolate linearly; the complete aligned CDF carries the tails. The left panel resolves the core and the right panel resolves small tail probabilities on a logarithmic scale. The shaded interval is the proposed training exclusion [-4,3]. The 90 MeV curve is a conditional prediction, not closure against an independent sample. No uncertainty bands are shown.',
        'v637_omitted_mass_shapes': 'How to read the figure. At each displayed mass, the black signal-MC distribution is excluded from model construction and then used to check the prediction. Blue interpolates the two adjacent samples, including core and tails. Red uses the equal-weight common shape from all other anchors. Both models predict the target center and width from its neighbors. Upper panels show the core, and lower panels show the tails. All densities retain the full selected-event normalization; no fit-window renormalization or uncertainty bands are applied.',
        'v637_shape_closure_metrics': 'How to read the figure. Each target mass is removed before predicting its center, width and distribution. The upper panels measure core-parameter errors. The lower-left panel gives the maximum absolute CDF difference, which compares integrated probability throughout the stored mass range. The lower-right panel compares probability in the same [-4,3] window defined using the excluded MC core only for evaluation. Smaller absolute differences indicate better agreement. These are deterministic shape checks; finite signal-MC and fitted-core uncertainties are not propagated, so no bars are shown.'}
    write(R/'template_figure_captions.json',captions)
    print(json.dumps({k:summary[k] for k in ['passed','common_CDF_error_range','morph_CDF_error_range']}))


if __name__=='__main__':
    run()
