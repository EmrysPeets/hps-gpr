from pathlib import Path
import csv
B=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((B/'results/raw_peak_summary.csv').open()))
labels={'2015':'2015 full','2016':'2016 full','2021':r'2021 native 10\%','combined':'Shared coupling'}
lines=[r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{lrrrrr}\toprule',r'Scope & Raw peak [MeV] & Raw local $Z$ & Raw local $p$ & Ref. peak [MeV] & Ref. $Z$\\\midrule']
for r in rows:
 lines.append(f"{labels[r['scope']]} & {float(r['mass_MeV']):g} & {float(r['local_Z_unshifted']):.4f} & {float(r['local_p_unshifted']):.6f} & {float(r['reference_peak_mass_MeV']):g} & {float(r['reference_peak_Z']):.4f}"+r' \\')
lines += [r'\bottomrule\end{tabular}',r'\caption{Raw local peaks and the earlier reference-local peaks of the same observed scans. The final two columns identify what was changed by reference centering/scaling; they are not used to draw the requested local curves. No column contains a look-elsewhere correction.}',r'\label{tab:raw_local_peaks}',r'\end{table}']
(B/'source/raw_peak_table.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{table}[htbp]\centering\small',r'\begin{tabular}{lrrrrl}\toprule',r'Scope & Peak [MeV] & Gaussian $p$ & Global $Z$ & Direct $k/256$ & Direct 95\% interval\\\midrule']
for r in rows:
 lines.append(f"{labels[r['scope']]} & {float(r['mass_MeV']):g} & {float(r['raw_order_global_p_addone']):.5f} & {float(r['raw_order_global_Z']):.3f} & {r['direct_raw_global_k']}/256 & [{float(r['direct_raw_global_p_lo95']):.4f}, {float(r['direct_raw_global_p_hi95']):.4f}]"+r' \\')
lines += [r'\bottomrule\end{tabular}',r'\caption{Fixed-source global probabilities at the unshifted raw maxima. The local results are in Table~\ref{tab:raw_local_peaks}. These values reuse the saved raw-maximum ensembles and are not the old reference-standardized maximum tails.}',r'\end{table}']
(B/'source/raw_global_table.tex').write_text('\n'.join(lines)+'\n')
