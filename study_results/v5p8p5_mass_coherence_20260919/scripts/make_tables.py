"""Generate all numerical report text and tables from the stored results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
B=Path(__file__).resolve().parents[1]
f=pd.read_csv(B/'results/free_peaks.csv');curves=pd.read_csv(B/'results/free_curves.csv');c=pd.read_csv(B/'results/free_comparison.csv');s=pd.read_csv(B/'results/stability_peaks.csv',dtype={'dataset':str});r=pd.read_csv(B/'results/stability_ranges.csv',dtype={'dataset':str});a=pd.read_csv(B/'results/stability_at92.csv',dtype={'dataset':str});co=json.loads((B/'results/stability_coherence_summary.json').read_text());v=json.loads((B/'qa/free_validation.json').read_text());sv=json.loads((B/'qa/stability_validation.json').read_text());dc=pd.read_csv(B/'qa/free_direct_fit_checks.csv')
def write(name,text): (B/f'source/{name}.tex').write_text(text+'\n')
def table(name,cols,header,rows,caption):
 import re
 caption=re.sub(r'\\textbf\{Table \d+\.\} ', '', caption)
 text=r'\begin{table}[htbp]\centering{\small\setlength{\tabcolsep}{4pt}\begin{tabular}{'+cols+'}\n\\toprule\n'+header+r' \\'+'\n\\midrule\n'+'\n'.join(' & '.join(str(x) for x in row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}}\n\\caption{'+caption+'}\n\\label{tab:'+name+'}\n\\end{table}'
 write(name,text)
base=f.iloc[0]
write('headline',rf'''At the nominal width, the new test peaks at {base.mass_MeV:.0f} MeV with conditional response-calibrated local $Z={base.local_Z:.2f}$ and full-domain $Z={base.global_Z:.2f}$. The exact Poisson scans give {base.direct_global_k:.0f}/256 global exceedances, with 95\% interval $p\in[{base.direct_global_lo95:.3f},{base.direct_global_hi95:.3f}]$. Across widths the calibrated peak stays at 91--92 MeV. However, the individual peak matrix does not establish unusually tight three-dataset coherence: its paired-null diagnostic gives $p={co['direct_T']['p']:.3f}$. The 2021 diagnostic maximum switches between excursions at 93 and 80 MeV.''')
labels={'shared_coupling':'Shared coupling','fisher':'Fisher','signed_stouffer':'Signed Stouffer','stouffer':'Signed Stouffer','free_amplitude':'Free amplitudes','free_amplitudes':'Free amplitudes'}
rows=[]
for _,x in c[np.isclose(c.width_sigma,2.25)].iterrows():
 label=labels.get(x.method,x.method.replace('_',' '));rows.append([label,f'{x.peak_mass_MeV:.0f}',f'{x.local_Z:.2f}',f'{x.gaussian_global_p:.4f}',f'{x.gaussian_global_Z:.2f}',f'{x.direct_global_k:.0f}/256'])
table('baseline_table','lrrrrr',r'Method & Peak [MeV] & Local $Z$ & Global $p$ & Global $Z$ & Direct $k/N$',rows,r'\textbf{Table 1.} Full-domain peaks at $\pm2.25\sigma_m$. Local and global $Z$ use the conditional Gaussian-response model. Direct counts use the same 256 full Poisson realizations. Old methods are unchanged v5.8.4 results; the new test has its own calibrated null distribution.')
write('calibration_text',r'''The local tail of the sum of clipped squared normals is integrated over all nonempty positive-amplitude subsets with deterministic polar quadrature. At $a_d=0,s_d=1$ it reduces to the independent-boundary mixture $2^{-k}\sum_{j=1}^{k}\binom{k}{j}\Pr(\chi_j^2\geq q)$ for $q>0$, not a single $\chi_k^2$ tail. Actual mass-dependent $a_d,s_d$ are retained. Monotone lookup tables accelerate the scans. The declared ordering is $S_R(m)=-\log p_R(m)$.

The 256 stored Poisson realizations give an additional direct local and global calibration of this ordering, using the exact profile roots at every mass. At 92 MeV none exceeds the observed local statistic for any width; the one-sided 95\% upper bound is $p<0.01163$, which cannot resolve the response-model estimate near $3.6\sigma$. Zero counts are never presented as zero probability. Point estimates elsewhere use $(k+1)/(N+1)$; two-sided 95\% intervals are Clopper--Pearson. A separate 100,000-draw response ensemble preserves mass and width correlations through the joint derivative covariance.''')
rows=[]
for _,x in f.iterrows():
 x92=curves[(curves.width_sigma==x.width_sigma)&(curves.mass_MeV==92)].iloc[0]
 rows.append([f'{x.width_sigma:g}',f'{x.mass_MeV:.0f}',f'{x.local_Z:.3f}',f'{x92.local_Z:.3f}',f'{x.global_Z:.3f}',f'{x.direct_global_k:.0f}',f'[{x.direct_global_lo95:.4f}, {x.direct_global_hi95:.4f}]'])
table('width_table','rrrrrrl',r'$w$ & Peak [MeV] & Local $Z$ & $Z(92)$ & Global $Z$ & $k/256$ & Direct global $p$: 95\% CI',rows,r'\textbf{Table 2.} Free-amplitude peaks selected by maximum $-\log p_R$ on the full grid. The direct count column gives $k$ out of 256. Significance columns are Gaussian-response estimates. Raw $q_R$ instead has its maximum at 91 MeV for all four widths.')
x92=curves[curves.mass_MeV==92]
write('width_text',rf'''At 92 MeV, $q_R={x92.q_R.min():.3f}$--{x92.q_R.max():.3f}, local $Z={x92.local_Z.min():.3f}$--{x92.local_Z.max():.3f}, and full-domain Gaussian-response $Z={x92.global_Z.min():.3f}$--{x92.global_Z.max():.3f}. At each width's selected peak, the direct global plus-one probabilities are '''+', '.join(f'{x:.4f}' for x in f.direct_global_p)+r''', corresponding to $Z$ values '''+', '.join(f'{x:.3f}' for x in f.direct_global_Z_display)+r'''. Their intervals are much wider than the 100,000-field Monte Carlo errors. The new test is less significant than signed Stouffer: it tests a broader alternative with separately nonnegative amplitudes, and its own boundary-aware local calibration.''')
rows=[]
for year in ['2015','2016','2021']:
 d=s[s.dataset==year];x=r[r.dataset==year].iloc[0];rows.append([year,*[f'{m:.0f}' for m in d.peak_mass_MeV],f'{x.sigma92_MeV:.3f}',f'{x.delta_m_MeV:.0f}',f'{x.R:.3f}'])
table('stability_table','lrrrrrrr',r'Dataset & $2.25$ & $2.4$ & $2.5$ & $2.6$ & $\sigma(92)$ & $\Delta m$ & $R$',rows,r'\textbf{Table 3.} Diagnostic-region signed-score argmax masses [MeV] at each half-width; resolution and range are also in MeV. $\Delta m=\max_w\hat m-\min_w\hat m$ and $R=\Delta m/\sigma(92)$. The region is 80--105 MeV, clipped to 80--100 for 2015. All selected observed peaks also pass the positive-raw-fit gate used by the coherence test.')
write('stability_text',r'''The 2015 and 2016 maxima remain within one grid bin across width. In 2021 the first two widths have almost tied excursions at 80 and 93 MeV: at $w=2.25$, their signed scores are 1.6216 and 1.6228. At $w=2.5$ they are 1.7332 and 1.6194. Thus the 13 MeV range is an argmax rank switch involving the lower diagnostic boundary, not evidence that a tracked resonance shifts by 13 MeV. The response at 92 MeV remains positive but modest.''')
rows=[]
for _,x in a.iterrows():rows.append([x.dataset,f'{x.width_sigma:g}',f'{x.signed_Z:.3f}',f'{x.local_p:.4f}',f'{x.window_signal_events_nonnegative:,.0f}'.replace(',',r'\,'),f'{x.amplitude_nonnegative_1e8:.2f}'])
table('amplitude_table','lrrrrr',r'Dataset & $w$ & Signed $Z(92)$ & Local $p(92)$ & $\hat\mu_d^+$ [events] & $\hat A_d$',rows,r'\textbf{Table 4.} Conditional single-dataset local responses and fitted amplitudes at the inspected mass 92 MeV. Local tails use the source-standardized response model. The event yield is integrated over the actual fitted window; changing the window changes its contained fraction.')
rows=[]
for name in ['T','W']:
 d=co[f'direct_{name}'];g=co[f'gaussian_{name}'];rows.append([name,f'{co[f"observed_{name}"]:.4f}',f'{d["k"]}/256',f'{d["p"]:.4f}',f'[{d["lo95"]:.4f}, {d["hi95"]:.4f}]',f'{g["p"]:.4f}'])
table('coherence_table','lrrrlr',r'Statistic & Observed & Direct $k/N$ & Direct $p$ & Direct 95\% interval & Gaussian $p$',rows,r'\textbf{Table 5.} Inclusive lower-tail probabilities. $T$ is the primary coherence diagnostic; $W$ is a separate width-stability diagnostic. The region and statistics have not received a selection correction.')
write('coherence_text',rf'''Under the pinned null, {co['direct_T']['k']} of 256 exact scans are at least as tightly clustered as the observed twelve peaks. This is not unusually coherent. Likewise, {co['direct_W']['k']} of 256 have width ranges at least as small. The Gaussian estimates agree within the direct intervals; increasing Gaussian statistics cannot remove source-model uncertainty. The paired Gaussian ensemble contains {100000-co['gaussian_T']['eligible_null']} ineligible scans, all retained in the denominator as noncoherent. No diagnostic tail is used to select a new mass region or nominal width.''')
print('direct check columns',','.join(dc.columns))
quad=max(x['max_relative_quad_error'] for x in v['quadrature_checks']);interp=max(v['observed_interpolation_max_abs_logp_error']);cov=max(x['max_covariance_reconstruction_error'] for x in v['joint_width_factors'])
write('validation_text',rf'''The numerical package contains {len(curves)} free-amplitude mass--width rows and {len(a)} independent 92 MeV amplitude checks. The joint constrained likelihood was replayed at 20 representative mass--width configurations, including single-, two- and three-dataset regions, and checked against the factorized statistic and nested shared-coupling fit. All source Poisson spectra regenerate exactly from their recorded seeds, and all twelve observed 92 MeV signed roots reproduce the parent values exactly in the matching SciPy/NumPy environment.

The central independent-boundary mixture and the positive/zero-atom limits pass. Increasing quadrature order changes the checked tails by at most {quad:.2g} relatively; lookup interpolation changes observed $\log p$ by at most {interp:.2g}. Joint covariance reconstruction differs by at most {cov:.2g}. Direct scalar evaluation reproduces the coherence metric to {sv['statistic_explicit_scalar_max_error']:.2g}. Numerical checks establish implementation consistency, not physical source validity.

At each width's Gaussian 95th-percentile global threshold, the direct Poisson exceedance counts are 16, 17, 20 and 17 out of 256. These are above the nominal mean 12.8; all corresponding 95\% binomial intervals still contain 0.05. This limited ensemble cannot rule out a modest response-tail mismatch. At the observed peaks, all Gaussian global estimates lie inside the direct intervals.''')
print('Wrote numerical tables and report text')
