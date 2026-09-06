#!/usr/bin/env python3
"""Build the versioned LaTeX extension directly from saved study results."""
from pathlib import Path
import csv, hashlib, json, subprocess
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent / 'v4p9p14_interpretation_global_20260906'
NOTE = HERE / 'note'
OUTPUT = ROOT / 'output/pdf/v4p9p15_global_2016_2021_20260906'
LABELS = {'2015': '2015 full', '2016': '2016 full', '2021': r'2021 10\%'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prob(x):
    if x == 0:
        return '0'
    if x >= .001:
        return f'{x:.4f}'.rstrip('0').rstrip('.')
    a, b = f'{x:.2e}'.split('e')
    return '$' + a + r'\times10^{' + str(int(b)) + '}$'

def estimate(t):
    if t['k'] == 0:
        return '$<$' + prob(t.get('upper95_one_sided', t['interval95'][1]))
    return prob(t['p'])

def bounds(t):
    return prob(t['interval95'][0]) + '--' + prob(t['interval95'][1])

def write(name, text):
    (NOTE / name).write_text(text + '\n')

def table(header, rows, caption, spec):
    return ('\\begin{table}[H]\\centering\\small\n'
            + '\\begin{tabular}{' + spec + '}\\toprule\n'
            + header + r'\\\midrule' + '\n'
            + '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
            + '\n' + r'\bottomrule\end{tabular}' + '\n'
            + r'\caption{' + caption + '}\\end{table}\n')

def main():
    inputs = [HERE.parent/'v4p9p7_2016_support_combined_100toy_20260902/SCIENTIFIC_SCOPE_CLARIFICATION.md',
              HERE.parent/'v4p9p5_2021_gp_support_edge_optimization_20260820/README.md',
              PREVIOUS/'review/HEP_STATISTICAL_REVIEW.md']
    summaries, curves, phases = {}, {}, {}
    for year in ('2015', '2016', '2021'):
        base = PREVIOUS if year == '2015' else HERE
        source = base / ('global' if year == '2015' else 'global_fast') / year / 'analysis/summary.json'
        summaries[year] = json.loads(source.read_text())
        assert summaries[year]['ensemble_contracts_and_all_point_checks_verified']
        curvefile = source.with_name('pvalue_curves.csv')
        curves[year] = pd.read_csv(curvefile)
        inputs.extend([source, curvefile])
        if year != '2015':
            phases[year] = {}
            for phase in ('pilot10', 'validation1000', 'asimov'):
                path = HERE / 'global_fast' / year / phase / 'summary.json'
                phases[year][phase] = json.loads(path.read_text())
                assert phases[year][phase]['passed'] and phases[year][phase]['complete']
                inputs.append(path)

    rows = []
    for year, s in summaries.items():
        x = s['methods']['profiled']
        d = x['global_direct']
        rows.append([LABELS[year], x['peak_mass_MeV'], prob(x['local_common_truth_p']),
                     estimate(x['global_gp']), f"{d['k']}/{d['n']}"])
    write('summary_table.tex', table(
        r'Dataset & Mass [MeV] & Local $p$ & GP global $p$ & Direct exceedances', rows,
        r'Profiled statistic, principal minimum-local-p ordering. The 2015 row is reused from the frozen study. Each row is a separate search. A less-than sign gives a one-sided 95\% Monte Carlo upper bound for a zero-count tail, conditional on the sampled model.',
        'lrrrr'))

    headline = []
    for year in ('2016', '2021'):
        x = summaries[year]['methods']['profiled']
        if x['global_direct']['k'] == 0:
            point=curves[year][(curves[year].method=='profiled') & (curves[year].mass_MeV==x['peak_mass_MeV'])].iloc[0]
            headline.append(LABELS[year]+f": at {x['peak_mass_MeV']} MeV the observed root is {x['observed_raw_r']:.3f}, while the stress background predicts {point.asimov_r:.3f}. The earlier two-background local envelope gives p="+prob(point.parent_envelope_p)+'. The new extreme score measures tension with one background construction; it is not a particle discovery significance. The direct sample does not resolve this rare tail.')
        else:
            headline.append(LABELS[year] + ': the GP estimate ' + ('lies inside' if x['gp_global_inside_direct_interval95'] else 'lies outside')
                            + r' the direct-scan 95\% interval at the observed threshold. This tests the approximation under that background, not the background itself.')
    write('headline.tex', '\n\n'.join(headline))

    rows = []
    for year in ('2016', '2021'):
        p = phases[year]
        mass = summaries[year]['masses_MeV']
        rows.append([LABELS[year], f'{mass[0]}--{mass[-1]}', len(mass), p['pilot10']['full_bins'],
                     p['asimov']['n_spectra']])
    write('scope_table.tex', table(
        'Dataset & Mass range [MeV] & Masses & Data bins & Asimov scans', rows,
        'Both mass grids have 1 MeV spacing. Each dataset also has ten pilot and 1,000 independent validation scans. One Asimov scan is the unperturbed mean; the remaining scans perturb one bin each.', 'lrrrr'))

    correlations, tails, interpretations, rawrows, ledger = [], [], [], [], []
    for year in ('2016', '2021'):
        s = summaries[year]
        text, peakrows, diagrows = [], [], []
        for method in ('profiled', 'fixed'):
            x = s['methods'][method]
            g, d, raw = x['global_gp'], x['global_direct'], x['raw_ordering']
            point = curves[year][(curves[year].method == method) & (curves[year].mass_MeV == x['peak_mass_MeV'])].iloc[0]
            text.append(r'\textbf{' + method.capitalize() + ' statistic.} The most extreme point for the declared rule is '
                        + str(x['peak_mass_MeV']) + ' MeV. The local common-background probability is '
                        + prob(x['local_common_truth_p']) + ', and the GP scan-wide estimate is ' + estimate(g)
                        + f". Direct scans exceed this threshold in {d['k']}/{d['n']} cases")
            if d['k'] == 0:
                text[-1] += ', giving a one-sided 95\\% upper bound of ' + prob(d['upper95_one_sided']) + '.'
            else:
                text[-1] += ' (95\\% interval ' + bounds(d) + ').'
            if 0<d['k']<20:
                text[-1]+=' This small direct count gives only a weak rare-tail check; agreement with its broad interval is not precise validation of the GP estimate.'
            peakrows.append([method.capitalize(), x['peak_mass_MeV'], f"{x['observed_raw_r']:+.3f}",
                             f"{point.asimov_r:+.3f}", f'{point.response_sd:.3f}', f"{x['observed_standardized_r']:+.3f}"])
            diagrows.append([method.capitalize(), f"{x['valid_z_mean_range'][0]:+.3f} to {x['valid_z_mean_range'][1]:+.3f}",
                             f"{x['valid_z_sd_range'][0]:.3f} to {x['valid_z_sd_range'][1]:.3f}", x['marginal_normality_holm_flags']])
            rawrows.append([LABELS[year], method.capitalize(), raw['peak_mass_MeV'], estimate(raw['global_gp']),
                            f"{raw['global_direct']['k']}/{raw['global_direct']['n']}"])
            ledger.append(dict(dataset=year, method=method, mass_MeV=x['peak_mass_MeV'], observed_r=x['observed_raw_r'],
                               asimov_r=float(point.asimov_r), response_sd=float(point.response_sd), standardized_r=x['observed_standardized_r'],
                               local_p=x['local_common_truth_p'], gp_global_p=g['p'], gp_exceedances=g['k'],
                               gp_upper95=g['upper95_one_sided'], direct_global_p=d['p'], direct_exceedances=d['k'],
                               direct_upper95=d['upper95_one_sided'], raw_peak_MeV=raw['peak_mass_MeV'], raw_gp_global_p=raw['global_gp']['p']))
        write('results' + year + '.tex', '\n\n'.join(text))
        write('peak' + year + '.tex', table(
            r'Method & Mass & Observed $r$ & Null offset $a$ & Width $s$ & $(r-a)/s$', peakrows,
            r'The peak of the principal rule, with its raw value and calibration ingredients. The standardized score is conditional on the common stress background; it is not an independently established Gaussian discovery significance.', 'lrrrrr'))
        diagnostic = table(r'Method & Standardized mean range & Spread range & Flagged masses', diagrows,
            f"Independent validation after deterministic centering and scaling. Normality flags use a Holm adjustment over {len(s['masses_MeV'])} masses within each method. Non-rejection has finite statistical power and does not certify extreme tails.", 'lrrr')
        diagnostic += 'The response uses the unfluctuated spectrum and its bin perturbations; the independent validation sample is not used to retune the offset, width or covariance.'
        write('diagnostics' + year + '.tex', diagnostic)
        p, f = s['methods']['profiled'], s['methods']['fixed']
        correlations.append(LABELS[year] + ': the root-mean-square difference between the response correlation and the direct-scan correlation is '
                            + f"{p['correlation_rms_difference']:.3f} (profiled) and {f['correlation_rms_difference']:.3f} (fixed).")
        tails.append(LABELS[year] + ': the distribution-shape (KS) diagnostic gives nominal, unadjusted p-values '
                     + prob(p['minimum_local_p_maximum_distribution_KS']['pvalue']) + ' (profiled) and '
                     + prob(f['minimum_local_p_maximum_distribution_KS']['pvalue'])
                     + '. These compare the two simulation methods, not the particle significance of the data.')
        if min(p['minimum_local_p_maximum_distribution_KS']['pvalue'],f['minimum_local_p_maximum_distribution_KS']['pvalue'])<.05:
            tails[-1]+=' A nominal value below 0.05 is a warning about approximation error within this set of four comparisons for the ordering; it is not a definitive failure or a far-tail validation.'
        coarse = p['coarse_2MeV_global_at_fine_peak']
        interpretations.append(LABELS[year] + ': the two 2 MeV subgrids give profiled global estimates '
                               + estimate(coarse[0]) + ' and ' + estimate(coarse[1]) + ', compared with '
                               + estimate(p['global_gp']) + ' on the 1 MeV grid, all at the same fine-grid observed threshold. '
                               + 'Zero-count comparisons are unresolved. A coarser-grid comparison cannot establish convergence to a continuous mass search.')
    write('correlation_text.tex', '\n\n'.join(correlations))
    write('tail_text.tex', '\n\n'.join(tails) + '\n\nThe direct-scan counts calibrate the declared score under the chosen background without requiring a jointly Gaussian field. The GP approximation has that additional assumption.')
    interpretation = (r'A small conditional probability can reject the behavior predicted by one background construction without identifying the reason. In 2016, large null offsets make this distinction essential. An unusually high centered score and an ordinary raw maximum can coexist because the two rules ask different questions.'
                      + '\n\n' + '\n\n'.join(interpretations))
    write('interpretation.tex', interpretation)
    write('raw_table.tex', table('Dataset & Method & Raw peak [MeV] & GP global $p$ & Direct exceedances', rawrows,
        r'Separate maximum-raw-root ordering, shown for comparison. A probability near one means that the selected background often produces a larger positive peak somewhere. It is not a goodness-of-fit test or evidence that the generating model is correct.', 'llrrr'))

    times = []
    numeric = []
    for year in ('2016', '2021'):
        p = phases[year]
        auditfile=HERE/'global_fast'/year/'acceleration_validation.json'
        executionfile=HERE/'global_fast'/year/'execution_summary.json'
        exactpilot=HERE/'global'/year/'pilot10/summary.json'
        audit=json.loads(auditfile.read_text())
        execution=json.loads(executionfile.read_text())
        inputs.extend([auditfile,executionfile,exactpilot])
        error=max(x['max_root_error'] for x in audit['overlaps'].values() if x['comparison_available'])
        response=max(x['max_relative_l2_response_error'] for x in audit['full_response'].values())
        corr=max(x['max_absolute_correlation_error'] for x in audit['full_response'].values())
        numeric.append(LABELS[year]+f": {len(audit['exact_fallback_masses_MeV'])} masses use the exact fallback; the largest paired-root difference is {error:.2g}; complete exact response columns at six masses differ by at most {response:.2g} in relative vector norm, and their correlations by at most {corr:.2g}.")
        times.append(LABELS[year]+f": the accepted accelerated calculation, including accuracy checks, took {execution['seconds_this_invocation']/60:.1f} minutes.")
        times.append(LABELS[year] + ': accelerated replay of the ten pilot scans took '
                     + f"{p['pilot10']['seconds_this_invocation']:.1f} s; 1,000 validation scans took {p['validation1000']['seconds_this_invocation']/60:.1f} minutes; "
                     + f"{p['asimov']['n_spectra']} Asimov scans took {p['asimov']['seconds_this_invocation']/60:.1f} minutes.")
    write('numerical_accuracy.tex', 'Exact comparisons cover ten pilot scans at every mass, plus 1,000 scans at 81 of the 2016 masses. The 2021 validation ensemble uses the gated accelerator, with exact checks supplied by the pilot and response audits. Every mass has an exact baseline and bin-response stencil; six masses per dataset have complete exact response columns. '+ ' '.join(numeric))
    write('execution_text.tex', 'Numerical fits ran sequentially with one worker and one BLAS thread. The times below sum ensemble evaluations; shared exact accuracy gates and setup add to the total runtime. ' + ' '.join(times)
          + ' Timing is descriptive and depends on the host. The Gaussian field sampling and plots are a separate, inexpensive step.')
    pd.DataFrame(ledger).to_csv(HERE / 'provenance/peak_summary.csv', index=False)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(['tectonic', '--keep-logs', '--outdir', str(OUTPUT), str(NOTE / 'reader_report.tex')],
                         cwd=NOTE, capture_output=True, text=True)
    (NOTE / 'build.log').write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(run.stderr)
    generated = OUTPUT / 'reader_report.pdf'
    pdf = OUTPUT / 'HPS_GPR_v4p9p15_Global_Study_2016_Full_2021_10pct.pdf'
    pdf.write_bytes(generated.read_bytes())
    generated.unlink()
    inputs += [Path(__file__), HERE / 'PROTOCOL.md', HERE / 'ACCELERATION_PROTOCOL.md', HERE / 'ACCELERATION_RESPONSE_GATES.md', *NOTE.glob('*.tex'), *HERE.joinpath('figures').glob('*.pdf')]
    record = dict(pdf=str(pdf), pdf_sha256=sha(pdf), inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs})
    (HERE / 'provenance/report_build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(run.stdout + run.stderr)
    print(pdf)

if __name__ == '__main__':
    main()
