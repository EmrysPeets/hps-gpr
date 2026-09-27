"""Render the saved leakage diagnostics at the note's column width; no fitting."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-leakage-note-mpl')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;F=B/'note_figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
BLUE='#245c91';ORANGE='#ad5536';GREEN='#2d865f';GRAY='#777777'
def done(fig,name,rect=(0,0,1,.98)):
 fig.tight_layout(rect=rect)
 for ext in ('png','pdf'):fig.savefig(F/f'{name}.{ext}',dpi=190)
 plt.close(fig)
def axes_style(ax):
 ax.set(xlabel='Generated signal mass [MeV]',xticks=[80,120,160,200,240]);ax.grid(alpha=.18)
def main():
 g=pd.read_csv(B/'window_leakage_by_mass.csv');x=g.mass_MeV
 fig,axes=plt.subplots(1,2,figsize=(7.1,3.1))
 a=axes[0];a.plot(x,g.lower_edge_sigma_m,'o-',color=BLUE,label='Starter lower edge');a.plot(x,g.upper_edge_sigma_m,'s-',color=GREEN,label='Starter upper edge')
 for value in [-2.25,2.25]:a.axhline(value,color=GRAY,ls='--',label=r'Historical $\pm2.25$' if value>0 else None)
 a.set(ylabel=r'Offset from window center [$\sigma_m$]',title='Window edges in historical width units');a.legend(loc='center right',fontsize=6.8)
 a=axes[1];a.plot(x,g.continuous_width_increase_percent,'o-',color=BLUE,label='Continuous interval');a.plot(x,g.binned_width_increase_percent,'s--',color=GRAY,label='Actual excluded bins');a.set(ylabel='Width increase [%]',title='How much wider is the starter window?');a.legend(loc='lower right')
 for a in axes:axes_style(a)
 done(fig,'leakage_window_width')
 fig,axes=plt.subplots(1,2,figsize=(7.1,3.4))
 for prefix,color,label in [('historical',ORANGE,'Historical window'),('starter',BLUE,'Starter window')]:
  axes[0].plot(x,100*g[f'MC_{prefix}_training_fraction'],'o-',color=color,label='MC: '+label)
  axes[0].plot(x,100*g[f'Gaussian_{prefix}_training_fraction'],'s--',color=color,label='Shifted Gaussian: '+label)
  axes[1].plot(x,g[f'MC_over_Gaussian_{prefix}_leakage_ratio'],'o-',color=color,label=label)
 axes[0].set(ylabel='Full signal in GP training [%]',title='Signal MC has more tail probability',ylim=(0,36));axes[0].legend(loc='upper right',fontsize=6.4)
 axes[1].set(ylabel='MC / shifted Gaussian leakage ratio',title='Compare shapes in the same bins');axes[1].legend()
 for a in axes:axes_style(a)
 done(fig,'leakage_probability')
 d=pd.read_csv(B/'mean_count_training_removal.csv');q=pd.read_csv(B/'paired_training_removal_summary.csv')
 for source in ['gp_mean','functional']:
  fig=plt.figure(figsize=(7.1,5.0));grid=fig.add_gridspec(2,2,height_ratios=[1,1]);axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
  for policy,color,label in [('direct_starter',BLUE,'Direct MC, starter window'),('gaussian_baseline',ORANGE,'Shifted Gaussian, historical window')]:
   a=d[(d.source==source)&(d.policy==policy)].sort_values('mass_MeV');b=q[(q.source==source)&(q.policy==policy)].sort_values('mass_MeV')
   axes[0].plot(a.mass_MeV,a.full_response,color=color,label=label+'; contaminated')
   axes[0].plot(a.mass_MeV,a.clean_response,color=color,ls='--',label=label+'; clean training')
   axes[0].errorbar(b.mass_MeV,b.full_response,yerr=b.full_response_SE,fmt='o',color=color,ms=3,capsize=2)
   axes[0].errorbar(b.mass_MeV,b.clean_response,yerr=b.clean_response_SE,fmt='s',mfc='white',color=color,ms=3,capsize=2)
   axes[1].plot(a.mass_MeV,100*a.leakage_loss_fraction,color=color)
   axes[1].errorbar(b.mass_MeV,100*b.leakage_loss_fraction,yerr=100*b.leakage_loss_SE,fmt='o',color=color,ms=3,capsize=2)
   axes[2].plot(a.mass_MeV,a.response_scaling_penalty_percent,color=color)
   axes[2].errorbar(b.mass_MeV,b.response_scaling_penalty_percent,yerr=[b.response_scaling_penalty_percent-b.penalty95_low,b.penalty95_high-b.response_scaling_penalty_percent],fmt='o',color=color,ms=3,capsize=2)
  axes[0].axhline(1,color=GRAY,lw=.7);axes[0].set(ylabel='Added fitted yield / A',title='Signal response with identical fitted counts')
  axes[1].set(ylabel='Yield suppression [% of A]',title='Effect of signal in GP training')
  axes[2].set(ylabel='Extra scaled yield noise [%]',title='Precision penalty from contamination')
  for a in axes:axes_style(a)
  hs,ls=axes[0].get_legend_handles_labels();fig.legend(hs,ls,ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.5,.995),fontsize=7)
  done(fig,'leakage_fit_'+source,(0,0,1,.91))
if __name__=='__main__':
 main()
 from compare_gaussian_centers import plot
 plot(pd.read_csv(B/'gaussian_center_leakage_comparison.csv'), F/'gaussian_center_leakage_comparison', compact=True)
