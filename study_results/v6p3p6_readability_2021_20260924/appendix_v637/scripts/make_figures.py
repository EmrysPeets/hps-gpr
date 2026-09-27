#!/usr/bin/env python3
"""Standalone figures from saved v6.3.7 numerical results, with explicit errors."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v637-figures')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];R=B/'results';F=B/'figures'
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
NAMES={'gaussian_baseline':'Shifted Gaussian: original window','gaussian_starter':'Shifted Gaussian: [-4, 3]','common_starter':'Common shape: [-4, 3]','morph_starter':'Neighbor interpolation: [-4, 3]','direct_starter':'Direct signal MC: [-4, 3]'}
COLORS=dict(zip(NAMES,['#a54b35','#9c71ac','#245c91','#278259','#444444']))
SRC={'gp_mean':'GP mean background source','functional':'Functional form background source'}

def finish(fig,name,footer,top=.88,bottom=.18):
 handles,labels=fig.axes[0].get_legend_handles_labels()
 if labels:fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False)
 fig.tight_layout(rect=(0,bottom,1,top))
 footer+='\nu = (reconstructed mass - predicted core center) / core width; s0 = archived Gaussian yield-error scale.'
 fig.text(.04,.018,footer,fontsize=7.2,linespacing=1.35,va='bottom')
 for ext in ('pdf','png'):fig.savefig(F/f'{name}.{ext}',dpi=180)
 plt.close(fig)

def main():
 F.mkdir(exist_ok=True)
 g=pd.read_csv(R/'window_geometry.csv');t=pd.read_csv(R/'tuning_response_summary.csv')
 policies=['morph_short','morph_middle','morph_starter','morph_wide','morph_extended','morph_buffer','morph_widebuffer']
 labels=['[-2.5, 2.25]','[-3, 2.5]','[-4, 3]','[-5, 4]','[-6, 5]','[-3, 2.5]\nG = [-4, 3]','[-4, 3]\nG = [-6, 5]']
 fig,axes=plt.subplots(2,1,figsize=(7.2,6.3),sharex=True)
 for mass,color in zip([100,160,220],['#245c91','#a54b35','#278259']):
  q=t[t.mass_MeV==mass].set_index('policy').loc[policies];d=g[g.mass_MeV==mass].set_index('policy').loc[policies];x=np.arange(len(policies))
  axes[0].plot(x,100*d.MC_training_fraction,'o-',label=f'{mass} MeV signal MC',color=color)
  axes[1].errorbar(x,q.sensitivity_ratio,yerr=[q.sensitivity_ratio-q.sensitivity_ratio95_low,q.sensitivity_ratio95_high-q.sensitivity_ratio],fmt='o-',capsize=2,color=color)
 axes[0].set(ylabel='Signal in GP training [%]',title='Tails left in the sidebands: full MC is injected everywhere')
 axes[1].set(ylabel='Scaled yield noise /\nGaussian baseline',xlabel='Signal-fit interval F in core-width units u; training exclusion G = F unless stated',xticks=np.arange(len(policies)),xticklabels=labels)
 axes[1].axhline(1,color='.4',ls='--',lw=.7)
 for ax in axes:ax.grid(axis='y',alpha=.2)
 finish(fig,'v637_window_scan','2021 10% | v16 TC signal-MC injections | Neighbor-interpolated extraction templates | 100 tuning toys\nHow to read: smaller lower-panel values indicate better precision; the upper panel is deterministic signal leakage.\nLower bars: 95% percentile intervals from 2,000 whole-toy bootstrap resamples, preserving masses and policies.\nResponse R = mean(fitted yield with signal - background-only fitted yield) / injected full yield A; A = 3 s0.\nNoise = sample SD(background-only fitted yield) / R. Shortening a template never renormalizes its probability.',top=.92,bottom=.23)
 if not (R/'evaluation_response_summary.csv').exists():return
 e=pd.read_csv(R/'evaluation_response_summary.csv');rows=pd.read_csv(R/'evaluation_rows.csv')
 fig,axes=plt.subplots(2,2,figsize=(7.2,5.7))
 for ri,source in enumerate(SRC):
  for policy in NAMES:
   q=e[(e.source==source)&(e.policy==policy)].sort_values('mass_MeV')
   sem=[]
   for m in q.mass_MeV:
    z=rows[(rows.source==source)&(rows.policy==policy)&(rows.mass_MeV==m)]
    pair=z.pivot(index='toy',columns='z',values='Ahat');a=float(z[z.z==3].A_expected.iloc[0])
    sem.append(((pair[3]-pair[0])/a).std(ddof=1)/np.sqrt(len(pair)))
   axes[ri,0].errorbar(q.mass_MeV,q.response,yerr=sem,fmt='o-',color=COLORS[policy],label=NAMES[policy],capsize=2,markersize=3)
   axes[ri,1].errorbar(q.mass_MeV,q.sensitivity_ratio,yerr=[q.sensitivity_ratio-q.sensitivity_ratio95_low,q.sensitivity_ratio95_high-q.sensitivity_ratio],fmt='o-',color=COLORS[policy],capsize=2,markersize=3)
  for ci in range(2):
   a=axes[ri,ci];a.axhline(1,color='.4',ls='--',lw=.7);a.set(xticks=[100,160,220],xlabel='Generated signal mass [MeV]',title=SRC[source]);a.grid(axis='y',alpha=.2)
  axes[ri,0].set_ylabel('Added fitted yield / A')
  axes[ri,1].set_ylabel('Scaled yield noise /\nGaussian baseline')
 finish(fig,'v637_independent_response','2021 10% | 100 independent evaluation toys per source | Full v16 TC signal-MC injection at A = 3 s0\nHow to read: response 1 returns the added full yield; a right-panel ratio below 1 means smaller yield noise.\nMean bars: sample standard errors. Ratio bars: 95% intervals from 2,000 whole-toy bootstrap resamples;\nresampling preserves mass, strength and policy correlations. Sources use separate streams. Pilot scale is fixed.\nThe candidate was selected using separate tuning toys; these evaluation toys did not choose the window.',top=.84,bottom=.22)
 s=pd.read_csv(R/'inference_summary.csv');pilot=json.loads((B/'inputs/window_pilot_reference.json').read_text())['masses']
 fig,axes=plt.subplots(2,2,figsize=(7.2,5.7))
 for ri,source in enumerate(SRC):
  for policy in NAMES:
   q=s[(s.source==source)&(s.policy==policy)&(s.z==0)].sort_values('mass_MeV');scale=np.array([pilot[str(int(m))]['s0'] for m in q.mass_MeV])
   axes[ri,0].plot(q.mass_MeV,q.median_largest_accepted_A/scale,'o-',color=COLORS[policy],label=NAMES[policy],markersize=3)
   q=s[(s.source==source)&(s.policy==policy)&(s.z==3)].sort_values('mass_MeV')
   axes[ri,1].errorbar(q.mass_MeV,q.reject_background_only_fraction,yerr=[q.reject_background_only_fraction-q.reject95_low,q.reject95_high-q.reject_background_only_fraction],fmt='o-',color=COLORS[policy],capsize=2,markersize=3)
  for ci in range(2):
   axes[ri,ci].set(xticks=[100,160,220],xlabel='Generated signal mass [MeV]',title=SRC[source]);axes[ri,ci].grid(axis='y',alpha=.2)
  axes[ri,0].set_ylabel('Median grid endpoint / $s_0$\n(nonempty sets only)')
  axes[ri,1].set(ylabel='Detection fraction\nat $A = 3s_0$',ylim=(0,1.05))
 finish(fig,'v637_limits_and_detection','2021 10% | 100 calibration and 100 independent evaluation toys per source | Full v16 TC MC injection\nHow to read: a lower left value is a smaller accepted-yield endpoint; a higher right value is more signal detection.\nLeft: coarse-grid medians, no error bars; empty sets are omitted, never set to zero. Endpoint flags are tabulated.\nRight: exact two-sided 95% Clopper-Pearson intervals, denominator 100, conditional on the saved calibration.\nCalibration is source matched and uses the full direct MC truth. These are local conditional tests, not discovery claims.',top=.84,bottom=.22)
 m=pd.read_csv(R/'mismatch_summary.csv')
 fig,axes=plt.subplots(1,2,figsize=(7.2,3.9))
 for ax,z in zip(axes,[3,5]):
  for policy in ['common_starter','morph_starter']:
   q=m[(m.policy==policy)&(m.z==z)].sort_values('mass_MeV')
   ax.errorbar(q.mass_MeV,q.true_grid_rejected_fraction,yerr=[q.true_grid_rejected_fraction-q.true_grid_rejected95_low,q.true_grid_rejected95_high-q.true_grid_rejected_fraction],fmt='o-',capsize=2,color=COLORS[policy],label=NAMES[policy])
  ax.axhline(10/101,color='.4',ls='--',lw=.7);ax.set(xticks=[100,160,220],xlabel='Generated signal mass [MeV]',ylabel='True-yield rejection\nfraction',title=f'Full direct signal-MC injection: A = {z} s0',ylim=(0,.25));ax.grid(axis='y',alpha=.2)
 finish(fig,'v637_calibration_mismatch','2021 10% | GP mean background source | 100 new calibration toys; 100 independent evaluation toys\nHow to read: calibrate with a predicted shape, then test against the omitted direct signal-MC sample.\nBars: exact two-sided 95% Clopper-Pearson intervals, denominator 100, conditional on the saved calibration.\nDashed line: 10/101 rejection probability for exchangeable continuous calibration and evaluation statistics.\nHere the signal distributions differ; agreement within these intervals does not prove exact coverage.',top=.86,bottom=.31)
 print('Generated numerical figures')
if __name__=='__main__':main()
