#!/usr/bin/env python3
"""Generate tables from the saved results and build the v4.9.13 LaTeX note."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOTE = HERE / 'note'
PDF = REPO / 'output/pdf/v4p9p13_background_profiling_20260905'
BASE = REPO / 'study_results/background_profile_comparison_20260905/derived'


def sci(value):
    mantissa, exponent = f'{value:.2e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'


def main():
    source_paths = [BASE/'observed_limits.csv', BASE/'summary.json',
        HERE/'observed/derived/scope_summary.csv',
        HERE/'observed/derived/local_p0_minima.csv',
        HERE/'injections/derived/extraction_summary.csv',
        HERE/'injections/derived/fisher_variance_scan.csv']
    d = pd.read_csv(source_paths[0]); s = json.loads(source_paths[1].read_text())
    scopes = pd.read_csv(source_paths[2]); pzero = pd.read_csv(source_paths[3])
    toy = pd.read_csv(source_paths[4]); fisher = pd.read_csv(source_paths[5])
    values = {
        'LogReleaseMax':f"{100*s['max_abs_log_current_relative']:.3f}",
        'LogControlMax':f'{100*abs(d.eps2_log_gp/d.eps2_gaussian_control-1).max():.3f}',
        'GaussianReleaseMax':f"{100*s['max_abs_gaussian_release_relative']:.3f}",
        'MeanMedianMax':f"{100*s['maximum_log_mean_median_relative']:.6f}",
        'GPFractionalSD':f"{100*s['maximum_gp_fractional_sd']:.3f}",
        'KappaRange':f'{fisher.kappa.min():.2f}--{fisher.kappa.max():.2f}',
        'KappaMedian':f'{fisher.kappa.median():.2f}',
        'CorrectedMedian':f'{fisher.corrected_fixed_over_profiled.median():.3f}',
        'CorrectedRange':f'{fisher.corrected_fixed_over_profiled.min():.3f}--{fisher.corrected_fixed_over_profiled.max():.3f}',
    }
    for key, lane, method in [('KnownWidthRange','known_background','fixed'),
        ('UncertainWidthRange','gp_uncertainty','fixed'),
        ('ProfileWidthRange','gp_uncertainty','profiled')]:
        v=toy[(toy.ensemble==lane)&(toy.method==method)&(toy.strength_sigma==0)].pull_std
        values[key]=f'{v.min():.2f}--{v.max():.2f}'
    (NOTE/'generated_values.tex').write_text(''.join(
        rf'\newcommand{{\{k}}}{{{v}}}'+'\n' for k,v in values.items()))

    labels=['2015, 100\\%','2016, 100\\%','2021, 10\\%','All three']
    lines=[r'\begin{table}[H]\centering\small',
        r'\begin{tabular}{lrrrr}\toprule',
        r'Scope & Minimum ratio & Median ratio & Maximum ratio & Fixed smaller\\\midrule']
    for label,row in zip(labels,scopes.itertuples()):
        lines.append(f'{label} & {row.min_fixed_over_current:.3f} & {row.median_fixed_over_current:.3f} & '
            f'{row.max_fixed_over_current:.3f} & {row.fixed_limit_smaller_count}/{row.mass_points}'+r'\\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{Fixed/released observed-limit ratios. Ratios describe the displayed conditional limits; a smaller value alone does not establish increased sensitivity.}',
        r'\label{tab:observed}\end{table}']
    (NOTE/'observed_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[H]\centering\small',r'\setlength{\tabcolsep}{5pt}',
        r'\begin{tabular}{lrrrrrr}\toprule',
        r' & \multicolumn{3}{c}{Released Gaussian profile} & \multicolumn{3}{c}{Fixed GP mean}\\',
        r'Scope & $m$ (MeV) & Minimum $p_0$ & $Z$ & $m$ (MeV) & Minimum $p_0$ & $Z$\\\midrule']
    for label,key in zip(labels,scopes.scope_key):
        rows=pzero[pzero.scope_key==key].set_index('method')
        a=rows.loc['current'];b=rows.loc['fixed']
        lines.append(f'{label} & {a.mass_MeV:.0f} & {sci(a.p0)} & {a.Z_local:.3f} & '
            f'{b.mass_MeV:.0f} & {sci(b.p0)} & {b.Z_local:.3f}'+r'\\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{Minimum local asymptotic $p_0$ in each scope. Fixed columns assume an exactly known GP mean; their quoted $Z$ values are model-conditional transformations, not calibrated significances. No global correction or local rescaling is applied.}',
        r'\label{tab:pzero}\end{table}']
    (NOTE/'pzero_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[H]\centering\small',r'\setlength{\tabcolsep}{5pt}',
        r'\begin{tabular}{rrrrrrr}\toprule',
        r' & \multicolumn{2}{c}{Released profile} & \multicolumn{2}{c}{Direct log-GP} & \multicolumn{2}{c}{Fixed mean}\\',
        r'$m$ (MeV) & $\hat A$ & $r$ & $\hat A$ & $r$ & $\hat A$ & $r$\\\midrule']
    for row in d[d.mass_MeV.isin([65,71,78,182])].itertuples():
        lines.append(f'{row.mass_MeV} & {row.current_Ahat_window:,.0f} & {row.r_current:.3f} & '
            f'{row.Ahat_log_gp:,.0f} & {row.r_log_gp:.3f} & {row.Ahat_fixed:,.0f} & {row.r_fixed:.3f}'+r'\\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{Signed signal yields in the actual fit window (events) and signed likelihood roots. Negative values are extraction diagnostics; discovery uses $Z=\max(r,0)$. The three columns of $r$ are conditional on their respective background treatments.}',
        r'\label{tab:fits}\end{table}']
    (NOTE/'selected_fits_table.tex').write_text('\n'.join(lines)+'\n')
    ledger={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (NOTE/'table_sources.json').write_text(json.dumps(ledger,indent=2)+'\n')
    tectonic=shutil.which('tectonic') or '/opt/homebrew/bin/tectonic'
    subprocess.run([tectonic,'--keep-logs','--keep-intermediates','analysis_note.tex'],cwd=NOTE,check=True)
    PDF.mkdir(parents=True,exist_ok=True)
    target=PDF/'HPS_GPR_Analysis_Note_v4p9p13_background_profiling.pdf'
    shutil.copy2(NOTE/'analysis_note.pdf',target)
    print(target)


if __name__=='__main__':
    main()
