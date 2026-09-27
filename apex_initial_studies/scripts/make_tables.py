from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

B=Path(__file__).resolve().parents[1]
def read(f):return pd.read_csv(B/'derived'/f)
def table(name,heads,rows,align=None):
    align=align or 'r'*len(heads)
    text='\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(heads)+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+' \\\\' for row in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n'
    (B/'source'/f'{name}.tex').write_text(text)

s=json.loads((B/'derived/summary.json').read_text());scan=read('common_mass_scan.csv')
a=read('apex_local_minima.csv').head(8)
rows=[]
for _,r in a.iterrows():
    rows.append([f'{r.mass_MeV:.1f}',f'{r.apex_p:.3f}',f'{r.pixel_low:.3f}--{r.pixel_high:.3f}',
                 f'{r.Z_one_sided_equivalent:.2f}',f'{r.hps_p_interp:.3f}' if np.isfinite(r.hps_p_interp) else 'Outside HPS'])
table('table_apex_minima',['$m$ [MeV]',r'$p_{\rm A}$','Pixel envelope',r'$Z_{\rm equiv}$',r'$p_{\rm H}(m)$'],rows,'rrrrl')
p=read('hps_peak_comparisons.csv');p=p[p.lane=='combined']
table('table_hps_peaks',['$m$ [MeV]',r'$p_{\rm H}$',r'$p_{\rm A}(m)$',r'$\sigma_{\rm H}$ [MeV]','Nearby APEX minimum'],
 [[f'{r.mass_MeV:.0f}',f'{r.hps_p:.3f}',f'{r.apex_p_same_mass:.3f}',f'{r.sigma_hps_MeV:.2f}',f'{r.apex_min_within_one_hps_sigma:.3f} at {r.apex_window_min_mass_MeV:.1f} MeV'] for _,r in p.iterrows()], 'rrrrl')
low=scan[scan.hps_2021_full_observed_equivalent_over_apex<1]
table('table_competitive',['$m$ [MeV]','APEX $U$','2021 full equiv. $U$','Ratio',r'$p_{\rm A}$',r'$p_{\rm H}$'],
 [[f'{r.mass_MeV:.0f}',f'{r.apex_limit*1e6:.2f}',f'{r.hps_2021_full_observed_equivalent*1e6:.2f}',f'{r.hps_2021_full_observed_equivalent_over_apex:.2f}',f'{r.apex_p:.3f}',f'{r.hps_p:.3f}'] for _,r in low.iterrows()])
c=read('projected_signal_compatibility.csv')
table('table_injections',['Source','$m$ [MeV]',r'$Z_{\rm source}$',r'$Z_{\rm target}$',r'$\epsilon^2_{\rm inj}$','Injection/APEX'],
 [[('10\%' if r.lane=='ten' else 'Historical 1\%'),f'{r.mass_MeV:.0f}',f'{r.source_Z:.2f}',f'{r.matched_Z:.2f}',f'{r.injected_epsilon2*1e6:.2f}',f'{r.injection_over_apex_limit:.1f}'] for _,r in c.iterrows()], 'lrrrrr')
r=read('apex_total_resolution.csv')
table('table_resolution',['$m$ [MeV]','Total axis value','$m$ [MeV]','Total axis value'],
 [[f'{r.iloc[i].mass_MeV:.0f}',f'{r.iloc[i].total_axis_value:.3f}',f'{r.iloc[i+6].mass_MeV:.0f}',f'{r.iloc[i+6].total_axis_value:.3f}'] for i in range(6)])
macros=dict(ApexMinMass=f"{s['apex_min']['mass_MeV']:.1f}",ApexMinP=f"{s['apex_min']['apex_p']:.3f}",
 ApexMinZ=f"{s['apex_min']['Z_one_sided_equivalent']:.2f}",
 BestConcordance=f"{s['best_same_mass']['p_concordance']:.3f}",
 DenseConcordance=f"{s['dense_min_concordance']:.3f}",
 MedianCurrentRatio=f"{s['competitive']['hps_limit']['ratio_median']:.1f}",
 MedianProjectionRatio=f"{s['competitive']['hps_2021_full_observed_equivalent']['ratio_median']:.2f}")
(B/'source/numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
print('Generated six tables and numeric macros')
