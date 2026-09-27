from pathlib import Path
import os,json,hashlib,shutil
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ['MPLCONFIGDIR']='/tmp/hps-external-stat-mpl'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];repo=B.parents[1]
paths=['study_results/v5p8p4_independent_combinations_windows_20260918/results/peak_composition.csv','study_results/v5p8p4_independent_combinations_windows_20260918/results/significance_and_reach.csv','study_results/v5p8p5_mass_coherence_20260919/results/free_peaks.csv','study_results/v5p8p5_mass_coherence_20260919/results/stability_coherence_summary.json','study_results/v5p0p5_analysis_note_20260916/source/sections/v5_calibration_appendix.tex','study_results/v5p8p4_independent_combinations_windows_20260918/results/peaks.csv','study_results/v5p8p2_nominal_gp_significance_20260917/results/summary.csv','study_results/v5p0p5_analysis_note_20260916/source/sections/v505_projections.tex','study_results/v5p0p5_analysis_note_20260916/source/sections/v5_global_results.tex']
ledger=[]
for name in paths:
 src=repo/name;dest=B/'inputs'/src.name
 if src.exists():shutil.copy2(src,dest)
 ledger.append({'path':name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
(B/'results/provenance.json').write_text(json.dumps(ledger,indent=2)+'\n')
p=pd.read_csv(B/'inputs/peak_composition.csv',dtype={'scope':str});p=p[(p.width_sigma==2.25)&(p.mass_MeV==92)].copy()
s=pd.read_csv(B/'inputs/significance_and_reach.csv');s=s[(s.width_sigma==2.25)&(s.domain=='full')]
p.to_csv(B/'results/92MeV_composition.csv',index=False,float_format='%.17g')
s[s.mass_MeV.isin([51,66,76,78,92])].to_csv(B/'results/region_rows.csv',index=False,float_format='%.17g')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.grid':True,'grid.alpha':.14})
fig,ax=plt.subplots(1,2,figsize=(10.0,3.75),gridspec_kw={'width_ratios':[1.6,1]})
colors=['#326a91','#ad5147','#348276','#333333']
for scope,c,label in zip(['2015','2016','2021','combined'],colors,['2015','2016','2021 (10%)','Shared coupling']):
 q=s[(s.scope==scope)&(s.mass_MeV>=50)&(s.mass_MeV<=100)].sort_values('mass_MeV')
 ax[0].plot(q.mass_MeV,q.observed_r,c=c,label=label,lw=1.4 if scope!='combined' else 1.8)
for lo,hi in [(65,67),(90,93)]:ax[0].axvspan(lo,hi,color='#aa8b50',alpha=.12)
ax[0].axhline(0,c='.5',lw=.7)
ax[0].set(xlabel='Tested mass [MeV]',ylabel='Raw signed likelihood root r',title='Observed fitted responses; no reference subtraction',xlim=(50,100))
ax[0].legend(fontsize=8,ncol=2,loc='lower right')
for j,(scope,c) in enumerate(zip(['2015','2016','2021'],colors)):
 q=p[p.scope==scope].iloc[0]
 ax[1].errorbar(q.epsilon2_hat*1e6,j,xerr=q.epsilon2_fit_sigma*1e6,fmt='o',c=c,capsize=4)
shared=p.iloc[0].shared_epsilon2_hat*1e6
ax[1].axvline(shared,c='.3',ls='--',label='Shared-coupling fit')
ax[1].set(xscale='log',xlim=(.2,160),ylim=(-.5,2.5),yticks=[0,1,2],yticklabels=['2015','2016','2021 (10%)'],xlabel=r'Fitted $\epsilon^2$ [$10^{-6}$]',title='At 92 MeV: common mass is not common rate')
ax[1].invert_yaxis();ax[1].legend(fontsize=8,loc='lower right')
fig.tight_layout(w_pad=2.2)
for ext in ['pdf','png']:fig.savefig(B/'figures'/('observed_evidence.'+ext),dpi=190,bbox_inches='tight')
plt.close(fig)
print(p[['scope','raw_r','epsilon2_hat','epsilon2_fit_sigma','null_information_fraction','shared_epsilon2_hat']].to_string(index=False))
