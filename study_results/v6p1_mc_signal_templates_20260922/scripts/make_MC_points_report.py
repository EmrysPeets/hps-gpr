"""Audit Fig. 12 against native MC counts and convert nominal-sigma masks to u.

This is histogram bookkeeping, not an independent goodness-of-fit test.
No observed likelihood is refitted and no production window is changed.
"""
from make_report import *
import hashlib


def main():
    core = pd.read_csv(D/'core_native_diagnostics.csv').set_index('mass_MeV')
    native = {}
    rows = []
    source_hashes = {}
    for mass, r in core.iterrows():
        path = B/f'histograms/m{mass:03d}.npz'
        source_hashes[str(path.relative_to(B))] = hashlib.sha256(path.read_bytes()).hexdigest()
        a = np.load(path, allow_pickle=False)
        edges, h, variance = a['edges_GeV']*1000, a['sumw'], a['sumw2']
        total = h.sum()
        cdf = np.r_[0, h.cumsum()]/total
        native[mass] = (edges, h, variance, cdf)
        Fm = lambda x: np.interp(x, edges, cdf, left=0, right=1)
        c, sc, sn = r.core_center_MeV, r.fitted_core_sigma_MeV, r.nominal_sigma_MeV
        meta = json.loads(path.with_suffix('.json').read_text())
        readout = selected_cutflow = 0.
        for file in meta['files']:
            cf = file['cutflows']['event_cutflow_h']
            counts = dict(zip(cf['labels'], cf['counts']))
            readout += counts['readout']
            selected_cutflow += counts['no_extra_true_ap']
        assert selected_cutflow == meta['stats']['selected']
        assert np.array_equal(h, variance), 'Poisson display assumes unit weights'
        rows.append(dict(mass_MeV=int(mass), center_MeV=c, sigma_core_MeV=sc,
            sigma_nominal_MeV=sn, old_halfwidth_u=2.25*sn/sc,
            u2p25_halfwidth_nominal_sigma=2.25*sc/sn,
            old_halfwidth_MeV=2.25*sn, u2p25_halfwidth_MeV=2.25*sc,
            fractional_width_reduction=1-sc/sn,
            core_fraction_abs_u_lt_2=float(Fm(c+2*sc)-Fm(c-2*sc)),
            old_window_fraction=float(Fm(c+2.25*sn)-Fm(c-2.25*sn)),
            u2p25_window_fraction=float(Fm(c+2.25*sc)-Fm(c-2.25*sc)),
            selected_entries=meta['stats']['selected'], histogram_entries=int(total),
            readout_cutflow=readout, selected_over_readout=selected_cutflow/readout))
    frame = pd.DataFrame(rows)
    frame.to_csv(D/'MC_points_window_units.csv', index=False, float_format='%.17g')

    def shared_cdf(u):
        values = []
        for mass, r in core.iterrows():
            edges, _, _, cdf = native[mass]
            x = r.core_center_MeV + r.fitted_core_sigma_MeV*np.asarray(u)
            values.append(np.interp(x, edges, cdf, left=0, right=1))
        return np.mean(values, axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(9, 6.6), layout='constrained')
    bins = []
    closure = []
    for column, mass in enumerate([60, 160]):
        r = core.loc[mass]
        c, sc = r.core_center_MeV, r.fitted_core_sigma_MeV
        edges, h, variance, cdf = native[mass]
        total = h.sum()
        Fm = lambda x: np.interp(x, edges, cdf, left=0, right=1)
        core_fraction = float(Fm(c+2*sc)-Fm(c-2*sc))
        for row, group in enumerate([20, 2]):
            # Direct sums of native 0.1 MeV bins supply the MC points.
            # A separate CDF integration supplies the line's bin predictions.
            coarse_edges = edges[::group]
            counts = h.reshape(-1, group).sum(axis=1)
            sumw2 = variance.reshape(-1, group).sum(axis=1)
            prediction = total*np.diff(Fm(coarse_edges))
            assert np.allclose(counts, prediction, rtol=1e-10, atol=1e-9)
            closure.append(dict(mass_MeV=mass, grouping=group,
                max_absolute_count_error=float(abs(counts-prediction).max())))
            x = (coarse_edges[1:]+coarse_edges[:-1])/2
            widths = np.diff(coarse_edges)
            density = counts/total/widths
            errors = np.sqrt(sumw2)/total/widths
            ax = axs[row, column]
            if row == 0:
                ax.plot((edges[1:]+edges[:-1])/2, h/total/np.diff(edges),
                    color=BLUE, lw=.85, label='Native empirical MC curve')
                ax.errorbar(x, density, yerr=errors, xerr=widths/2, fmt='o',
                    ms=2.1, lw=.6, color='black', label='MC bin counts (2 MeV)')
                ax.set(xlim=(0, 300), ylim=(2e-6, .3), yscale='log',
                    xlabel=r'Reconstructed $m_{ee}$ (MeV)', ylabel=r'Density (MeV$^{-1}$)',
                    title=f'{mass} MeV sample: full normalization')
                ax.axvline(c, color=GRAY, lw=.6, ls=':')
            else:
                # Exactly the 0.02-u CDF rebinning used for Fig. 12.
                ug = np.linspace(-60, 100, 8001)
                uc = (ug[1:]+ug[:-1])/2
                empirical = np.diff(Fm(c+sc*ug))/np.diff(ug)/core_fraction
                pool_norm = np.diff(shared_cdf(np.array([-2., 2.])))[0]
                pooled = np.diff(shared_cdf(ug))/np.diff(ug)/pool_norm
                ax.plot(uc, empirical, color=BLUE, lw=1.1, label='Figure 12 native MC curve')
                ax.plot(uc, pooled, 'k--', lw=1., label='Equal-mass common shape')
                ax.errorbar((x-c)/sc, density*sc/core_fraction,
                    yerr=errors*sc/core_fraction, xerr=widths/(2*sc),
                    fmt='o', ms=2.3, lw=.6, color=RED, label='MC bin counts (0.2 MeV)')
                ax.set(xlim=(-2, 2), ylim=(0, .5), xlabel=r'$u=(m_{ee}-c)/\sigma_{core}$',
                    ylabel='Conditional core density', title=r'Unit area on $|u|<2$')
            ax.legend(frameon=False, fontsize=7, loc='upper right')
            for lo, hi, n, v, expected in zip(coarse_edges[:-1], coarse_edges[1:], counts, sumw2, prediction):
                bins.append(dict(mass_MeV=mass, panel='full' if row == 0 else 'core',
                    lower_MeV=lo, upper_MeV=hi, MC_count=n, MC_sumw2=v,
                    empirical_template_count=expected))
    save(fig, 'MC_points_vs_figure12')
    pd.DataFrame(bins).to_csv(D/'MC_points_overlay_bins.csv', index=False, float_format='%.17g')
    table('MC_window_units', [r'$m_0$', r'$\sigma_{core}$', r'$\sigma_{nom}$',
        r'Old half-width ($u$)', r'Old MC (\%)', r'$\pm2.25u$ MC (\%)',
        r'Selected/readout (\%)'],
        [[int(r.mass_MeV), f'{r.sigma_core_MeV:.3f}', f'{r.sigma_nominal_MeV:.3f}',
          f'{r.old_halfwidth_u:.3f}', f'{100*r.old_window_fraction:.1f}',
          f'{100*r.u2p25_window_fraction:.1f}', f'{100*r.selected_over_readout:.1f}']
         for r in frame.itertuples()], 'rrrrrrr')
    result = dict(passed=True, masses=[60,160], native_masses_checked=len(frame),
        MC_point_grouping_MeV=[2., .2], figure12_CDF_rebin_du=.02,
        source_hashes=source_hashes, closure=closure,
        maximum_absolute_count_difference=max(z['max_absolute_count_error'] for z in closure),
        equivalence_halfwidth_u_range=[float(frame.old_halfwidth_u.min()),float(frame.old_halfwidth_u.max())],
        fractional_width_reduction_range=[float(frame.fractional_width_reduction.min()),float(frame.fractional_width_reduction.max())],
        uncertainty='sqrt(sumw2), with plotted normalization fixed; not a shape-fit covariance',
        closure_scope='Same-sample CDF/bin bookkeeping identity, not independent validation',
        readout_scope='Conditional stored-readout selection retention, not generated-signal acceptance',
        new_observed_fits=0, primary_mask_changed=False)
    (B/'qa/MC_points_window_units.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))


if __name__ == '__main__':
    main()
