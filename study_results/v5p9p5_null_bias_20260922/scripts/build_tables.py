"""Generate report tables from saved numerical results."""
from pathlib import Path
import csv,json
B=Path(__file__).resolve().parents[1]
S=B/'source'
s=json.loads((B/'scan_maximum/results/summary.json').read_text())
labels=['Nominal $a,s,R$','Remove $a$ only','Set $s=1$ only','Set $a=0$, $s=1$','Independent mass nodes','Perfectly correlated nodes']
out=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrr}\toprule',r'Null control & Median $T^*$ & Tail at $T_{\rm obs}$ & 95\% MC interval\\\midrule']
for label,r in zip(labels,s['paired_simulation']['scenarios']):
    out.append(f"{label} & {r['median']:.3f} & {r['p_addone']:.5f} & [{r['p_cp95_lo']:.4f}, {r['p_cp95_hi']:.4f}]"+r'\\')
out += [r'\bottomrule\end{tabular}',r'\caption{Paired Gaussian controls, 200,000 draws per row. Intervals quantify Monte Carlo error within each specified null.}\end{table}']
(S/'counterfactual_table.tex').write_text('\n'.join(out)+'\n')
analytic=list(csv.DictReader((B/'residual_diagnostic/results/analytic_scan.csv').open()))
controls=list(csv.DictReader((B/'residual_diagnostic/results/poisson_controls.csv').open()))
out=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{rrrrrr}\toprule',r'$m$ [MeV] & Variance & Bias & Sum & Exact mean (95\% CI) & Observed\\\midrule']
for m in [50,71,78,120,250]:
    a=next(r for r in analytic if float(r['mass_MeV'])==m)
    r=next(r for r in controls if float(r['mass_MeV'])==m and r['control']=='exact_refit_adaptiveV')
    out.append(f"{m} & {float(a['expected_refit_noise_per_bin']):.3f} & {float(a['deterministic_bias_Q_per_bin']):.3f} & {float(a['expected_refit_Q_per_bin']):.3f} & {float(r['mean']):.3f} [{float(r['mean_CI95_low']):.3f}, {float(r['mean_CI95_high']):.3f}] & {float(a['observed_Q_per_bin']):.3f}"+r'\\')
out += [r'\bottomrule\end{tabular}',r'\caption{All entries use $Q/N_{\rm bin}$. The first three columns decompose the fixed-$V$ linear prediction. Exact replay updates $V$ on each toy. Mean intervals are pointwise.}\end{table}']
(S/'residual_table.tex').write_text('\n'.join(out)+'\n')
out=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrr}\toprule',r'Scope & $\min a$ & $\max a$ & RMS $a$ & Range of $s$\\\midrule']
for key,label in [('2015','Full 2015'),('2016','Full 2016'),('2021','2021 native 10\%'),('combined','Shared-coupling union')]:
    r=s['scopes'][key]
    out.append(f"{label} & {r['source_a_range'][0]:.3f} & {r['source_a_range'][1]:.3f} & {r['source_a_rms']:.3f} & {r['source_s_range'][0]:.3f}--{r['source_s_range'][1]:.3f}"+r'\\')
out += [r'\bottomrule\end{tabular}',r'\caption{Saved source-response summaries. Mass ranges are 19--100, 39--180, 50--250 and 19--250 MeV, respectively. RMS and ranges are descriptive across correlated mass nodes.}\end{table}']
(S/'cross_scope_table.tex').write_text('\n'.join(out)+'\n')
(S/'residual_findings.tex').write_text(r'''
All 201 masses were replayed on the same 256 spectra (51,456 exact GP
predictions). The largest absolute paired mean change from the linear
prediction is $2.14\times10^{-4}$; updating rather than freezing $V$ changes
the mean by at most $9.52\times10^{-4}$ in $Q/N_{\rm bin}$.

The original slide-14 curve is reproduced exactly. Its maximum is
$Q/N_{\rm bin}=2.329997$ at 78~MeV, with 16 window bins. The exact
conditional mean there is 1.0217 (95\% mean interval $[0.9762,1.0672]$).
Removing the deterministic residual in the fixed-$V$ replay gives mean
0.9903. The total deterministic bias power is 0.487 before division by
16, while the signed-root response squared is $a^2=0.333$.

As an exploratory check, define $T_Q=\max_m Q(m)/N_{\rm bin}(m)$ over
the same 201 masses. Twenty of 256 exact paired scans exceed the observed
$T_Q$. The add-one tail is 0.0817 with a 95\% binomial interval
$[0.0484,0.1181]$. This is a conditional residual-scan diagnostic, not a
resonance significance or a calibrated physical-background goodness-of-fit
probability. It omits source-estimation uncertainty.
'''+'\n')
print('Generated report tables from saved results.')
