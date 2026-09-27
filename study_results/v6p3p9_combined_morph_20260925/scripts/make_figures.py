"""Observed combined scans with legends outside the data panels."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-v639-figures')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];F=B/'figures'
plt.rcParams.update({'font.family':'serif','font.size':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def frame(title):
 fig,ax=plt.subplots(2,1,figsize=(7.2,5.0),sharex=True)
 for a in ax:
  a.set_xlim(60,240);a.set_yscale('log');a.grid(axis='y',alpha=.17)
  for m in [100,180]:a.axvline(m,color='.65',ls=':',lw=.8,zorder=-3)
 ax[0].set_ylabel('Conditional 90% upper limit\n$\\epsilon^2$ (inherited normalization)')
 ax[1].set(ylabel='Local excess $p_0$\n(asymptotic reference)',xlabel='Signal mass hypothesis [MeV]',ylim=(1e-5,.85))
 fig.suptitle(title,fontsize=11,y=.985)
 fig.subplots_adjust(left=.15,right=.98,top=.80,bottom=.11,hspace=.12)
 return fig,ax
def line(ax,q,label,color,ls='-',lw=1.3):
 q=q.sort_values('mass_MeV')
 ax[0].plot(q.mass_MeV,q.epsilon2_90_visible_legacy,color=color,ls=ls,lw=lw,label=label)
 ax[1].plot(q.mass_MeV,q.p0_asymptotic,color=color,ls=ls,lw=lw)
def save(fig,ax,name):
 handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.937),ncol=2,frameon=False)
 fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.png'),dpi=180);plt.close(fig)
def main():
 d=pd.read_csv(B/'results/combined_scan.csv');F.mkdir(exist_ok=True)
 fig,ax=frame('Combined observed scan: change only the 2021 signal model')
 for p,name,color,ls in [('gaussian_baseline','2021 shifted Gaussian: original window','#777777','--'),('gaussian_starter','2021 shifted Gaussian: new window','#a98224',':'),('morph_starter','2021 neighboring signal-MC template','#245c91','-')]:
  line(ax,d[(d.scope=='combined')&(d.policy==p)],name,color,ls)
 save(fig,ax,'v639_combined_comparison')
 fig,ax=frame('Observed campaign contributions with the new 2021 signal template')
 for scope,p,name,color,ls in [('2015','unchanged','2015 full','#9e4d3c','--'),('2016','unchanged','2016 full','#568570','-.'),('2021','morph_starter','2021 10%: neighboring signal MC','#245c91',':'),('combined','morph_starter','Combined available campaigns','#222222','-')]:
  line(ax,d[(d.scope==scope)&(d.policy==p)],name,color,ls,1.65 if scope=='combined' else 1.2)
 save(fig,ax,'v639_campaign_contributions')
 (B/'results/figure_manifest.json').write_text(json.dumps({'figures':['v639_combined_comparison','v639_campaign_contributions'],'uncertainty_bands':False,'campaign_boundaries_MeV':[100,180]},indent=2)+'\n')
if __name__=='__main__':main()
