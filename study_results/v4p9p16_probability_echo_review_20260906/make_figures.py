#!/usr/bin/env python3
"""Separate observed fit probabilities from conditional background diagnostics."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
V16=HERE.parent/'v4p9p16_combined_global_20260906'; OUT=HERE/'figures'
BLUE,RED,GREEN,PURPLE,ORANGE='#176A9B','#AE3E30','#1A7045','#714DA0','#B37816'
plt.rcParams.update({'font.family':'serif','font.size':10.5,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})
INVENTORY=[]
def save(fig,name):
 for ext in ('pdf','png'):fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig);INVENTORY.append(name)
def guides(ax,lo=18.5,hi=250.5):
 for b in (38.5,49.5,90.5,180.5):
  if lo<b<hi:ax.axvline(b,color='.72',ls=':',lw=.7,zorder=0)
 ax.set_xlim(lo,hi);ax.grid(axis='y',alpha=.17)
def main():
 d=pd.read_csv(HERE/'derived/probability_grid.csv');o=pd.read_csv(V16/'global/observed.csv')
 fig=plt.figure(figsize=(8.6,8.0));g=fig.add_gridspec(4,1,height_ratios=(.34,2.6,1.8,1.8),left=.13,right=.98,bottom=.075,top=.885,hspace=.30)
 strip=fig.add_subplot(g[0]);axes=[fig.add_subplot(g[i],sharex=strip) for i in range(1,4)]
 segments=[(19,38,'2015'),(39,49,'2015\n+2016'),(50,90,'All three'),(91,180,'2016 + 2021'),(181,250,'2021')]
 for (lo,hi,label),col in zip(segments,['#DDEAF3','#D8E7DD','#E8E0F0','#F2E5D5','#E5E5E5']):
  strip.add_patch(Rectangle((lo-.5,0),hi-lo+1,1,facecolor=col,edgecolor='white'))
  strip.text((lo+hi)/2,.5,label,ha='center',va='center',fontsize=7.5)
 strip.set(ylim=(0,1),xlim=(18.5,250.5));strip.set_ylabel('Active\ndata',rotation=0,ha='right',va='center',fontsize=8)
 strip.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
 for s in strip.spines.values():s.set_visible(False)
 ul,root,p=axes
 ul.plot(o.mass_MeV,o.profiled_eps2_display,color='black',lw=1.4,marker='.',ms=2,label='Observed combined limit')
 ul.axvline(2*105.6583745,color=PURPLE,lw=1,ls='-.',label=r'Dimuon threshold $2m_\mu$')
 ul.set(yscale='log',ylabel=r'90% CL$_s$ limit on $\epsilon^2$');ul.set_title('Observed upper limit',loc='left',fontsize=11,fontweight='semibold')
 ul.legend(loc='upper center',frameon=False,fontsize=8.7)
 root.axhline(0,color='.35',lw=.8)
 root.plot(d.mass_MeV,d.observed_r,color=BLUE,lw=1.1,marker='.',ms=2.5)
 root.set(ylabel=r'Signed fit root $r$',ylim=(-4.4,3.9));root.set_title('The same profiled statistic as the extraction displays',loc='left',fontsize=10.8,fontweight='semibold')
 for m,txt,xy in [(21,'21',(29,3.05)),(66,'66',(54,3.12)),(92,'92',(111,3.1)),(72,'72 MeV deficit',(113,-3.65))]:
  v=d.set_index('mass_MeV').loc[m,'observed_r'];root.annotate(txt,xy=(m,v),xytext=xy,fontsize=8.7,ha='center',arrowprops={'arrowstyle':'-','color':'.45','lw':.6})
 p.plot(d.mass_MeV,d.nominal_local_p,color=BLUE,lw=1.1,marker='.',ms=2.5)
 p.set(yscale='log',ylim=(.0015,.78),ylabel=r'Nominal local $p_0$',xlabel=r'Mass hypothesis $m_{A^\prime}$ [MeV]')
 p.set_title('One mass at a time: asymptotic reference, no look-elsewhere correction',loc='left',fontsize=10.1,fontweight='semibold')
 p.axhline(.5,color='.5',ls=':',lw=.7)
 p.annotate('66 MeV: $p_0=0.00289$',xy=(66,.0028886649746267),xytext=(108,.005),fontsize=9,
  arrowprops={'arrowstyle':'-','color':'.45','lw':.6})
 for ax in axes:guides(ax)
 for ax in (ul,root):ax.tick_params(labelbottom=False)
 fig.suptitle('Combined HPS search: full mass range',y=.972,fontsize=14,fontweight='semibold')
 fig.text(.5,.936,'2015 full + 2016 full + 2021 10%  |  232 masses at 1 MeV spacing',ha='center',fontsize=10)
 save(fig,'combined_observed_limit_and_pvalues')
 for zoom in (False,True):diagnostic(d,zoom)
 echoes()
 (HERE/'provenance/figure_inventory.json').write_text(json.dumps({'figures':INVENTORY,'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.pdf','.png')}},indent=2)+'\n')
def empirical(ax,d,prefix,color,marker,connect=False):
 k=d[prefix+'_k'].to_numpy();x=d.mass_MeV.to_numpy();p=d[prefix+'_p'].to_numpy();lo=d[prefix+'_low95'].to_numpy();hi=d[prefix+'_high95'].to_numpy()
 good=k>=25;sparse=(k>0)&(k<25);zero=k==0
 if connect:ax.plot(x,np.where(good,p,np.nan),color=color,lw=1.05,alpha=.8)
 ax.errorbar(x[good],p[good],yerr=[p[good]-lo[good],hi[good]-p[good]],fmt=marker,ms=2.3,color=color,elinewidth=.55,capsize=0,alpha=.8,zorder=4)
 if np.any(sparse):ax.errorbar(x[sparse],p[sparse],yerr=[p[sparse]-lo[sparse],hi[sparse]-p[sparse]],fmt=marker,ms=4,mfc='white',mec=color,color=color,elinewidth=.9,capsize=1.4,zorder=5)
 if np.any(zero):ax.scatter(x[zero],d.loc[zero,prefix+'_upper95'],color=color,marker='v',s=27,zorder=6)
def diagnostic(d,zoom):
 lo,hi=(59.5,100.5) if zoom else (18.5,250.5)
 x=d[(d.mass_MeV>=lo)&(d.mass_MeV<=hi)]
 fig,axes=plt.subplots(3,1,figsize=(8.6,8.1),sharex=True)
 fig.subplots_adjust(left=.13,right=.98,top=.865,bottom=.08,hspace=.29)
 a,l,g=axes
 a.fill_between(x.mass_MeV,x.asimov_r-x.response_sd,x.asimov_r+x.response_sd,color=ORANGE,alpha=.15)
 a.plot(x.mass_MeV,x.asimov_r,color=ORANGE,lw=1.3,marker='.',ms=2.4,label=r'Reference $a_m$ (shade: $\pm s_m$)')
 a.plot(x.mass_MeV,x.observed_r,color=BLUE,lw=1.15,marker='.',ms=2.5,label=r'Observed $r_m$')
 a.axhline(0,color='.5',lw=.65);a.set(ylabel='Signed fit root',ylim=(-13.3,11.8))
 a.legend(loc='upper right',frameon=False,fontsize=8.7,ncol=2)
 a.set_title('Why centering can change the apparent significance',loc='left',fontsize=10.5,fontweight='semibold')
 floor=1e-4;clip=x.conditional_local_gaussian<floor
 l.plot(x.mass_MeV,np.maximum(x.conditional_local_gaussian,floor),color=BLUE,lw=1,marker='.',ms=2)
 l.scatter(x.loc[clip,'mass_MeV'],np.full(clip.sum(),floor),color=BLUE,marker='v',s=28,zorder=7)
 empirical(l,x,'direct_local',GREEN,'s')
 gated=~x.eligible.astype(bool)
 l.scatter(x.loc[gated,'mass_MeV'],np.ones(gated.sum()),marker='x',s=11,color='.45',linewidths=.8,zorder=8)
 l.set(yscale='log',ylim=(2e-5,1.6),ylabel='Conditional local tail')
 l.set_title('Relative to the reference: Gaussian approximation and direct counts',loc='left',fontsize=10.2,fontweight='semibold')
 empirical(g,x,'gp_global',RED,'o',connect=True)
 empirical(g,x,'direct_global',GREEN,'s')
 g.set(yscale='log',ylim=(2e-7,1.65),ylabel='Conditional global tail',xlabel='Mass hypothesis [MeV]')
 g.set_title('Same reference and ordering, including the scan over all 232 masses',loc='left',fontsize=10.1,fontweight='semibold')
 handles=[Line2D([],[],color=BLUE,lw=1,label='Gaussian local'),Line2D([],[],color=RED,marker='o',ms=3,lw=1,label='GP global (200,000)'),Line2D([],[],color=GREEN,marker='s',ms=3,lw=0,label='Direct scans (1,000)'),Line2D([],[],color='.3',marker='x',ms=4,lw=0,label=r'$r\leq0$: local/global = 1')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.948),ncol=2,frameon=False,fontsize=9)
 fig.suptitle('Background-reference probability audit'+(' | 60–100 MeV detail' if zoom else ' | full scan'),fontsize=13,y=.992,fontweight='semibold')
 fig.text(.13,.014,'Bars: central 95% MC intervals. Hollow: fewer than 25 tails. ▼: upper bound; blue ▼: below display floor.',fontsize=8.1)
 if zoom:
  a.annotate('76 MeV: observed +0.17, reference −8.70',xy=(76,-8.700),xytext=(60.2,-12),fontsize=8.6,arrowprops={'arrowstyle':'-','color':'.4','lw':.6})
  g.annotate('2 GP tails at 92 MeV',xy=(92,1e-5),xytext=(82,5e-7),fontsize=8.6,arrowprops={'arrowstyle':'-','color':'.4','lw':.6})
 for ax in axes:guides(ax,lo,hi)
 save(fig,'probability_reference_'+('zoom' if zoom else 'full'))
def echoes():
 e=pd.read_csv(HERE/'derived/echo_dense_scans.csv');w=e.pivot(index='mass_MeV',columns='lane',values='r')
 fig,axes=plt.subplots(3,1,figsize=(8.6,8.5),sharex=False)
 fig.subplots_adjust(left=.13,right=.98,top=.89,bottom=.08,hspace=.37)
 a,b,c=axes
 for lane,label,col,ls in [('observed','Observed 2021 10%', 'black','-'),('background','Smooth background', '.55',':'),('inject_66','Background + positive 66 MeV',BLUE,'-'),('inject_78','Background + positive 78 MeV',RED,'-')]:
  a.plot(w.index,w[lane],color=col,ls=ls,lw=1.25,marker='.' if lane=='observed' else None,ms=3,label=label)
 a.set(ylabel='Signed fit root $r$',ylim=(-4.7,5.6));a.legend(loc='upper right',fontsize=8.5,frameon=False,ncol=2)
 a.set_title('A single positive injected peak can yield several fitted features',loc='left',fontsize=10.5,fontweight='semibold')
 for lane,label,col in [('inject_66','66 MeV injection',BLUE),('inject_78','78 MeV injection',RED)]:
  b.plot(w.index,w[lane]-w.background,color=col,lw=1.5,label=label)
 b.set(ylabel=r'Injection change $\Delta r$',ylim=(-2.5,4.1));b.legend(loc='upper center',fontsize=8.7,frameon=False,ncol=2)
 b.set_title('Subtracting the same background response isolates the echoes',loc='left',fontsize=10.5,fontweight='semibold')
 b.annotate('Negative echoes',xy=(71,-1.825),xytext=(65,-2.15),ha='center',fontsize=9,arrowprops={'arrowstyle':'-','lw':.6,'color':'.5'})
 b.annotate('Positive echo',xy=(66,.916),xytext=(61.5,2.1),fontsize=9,arrowprops={'arrowstyle':'-','lw':.6,'color':'.5'})
 b.annotate('Positive echo',xy=(80,.829),xytext=(82.0,2.1),fontsize=9,arrowprops={'arrowstyle':'-','lw':.6,'color':'.5'})
 c.plot(w.index,w.observed,color='black',lw=1.25,marker='.',ms=3,label='Observed 2021 10%')
 c.plot(w.index,w.double_65_78,color=PURPLE,lw=1.5,label='Background + positive 65 and 78 MeV')
 c.plot(w.index,w.background,color='.55',lw=1,ls=':')
 c.set(ylabel='Signed fit root $r$',xlabel='Tested mass [MeV]',ylim=(-4.9,5.9))
 c.set_title('Two injected peaks can reproduce a dip while overshooting the peaks',loc='left',fontsize=10.5,fontweight='semibold')
 c.legend(loc='upper right',fontsize=8.5,frameon=False,ncol=2)
 for ax in axes:
  ax.set_xlim(59.5,88.5);ax.axhline(0,color='.6',lw=.7);ax.grid(axis='y',alpha=.18)
  for m in (65,66,71,72,78,85):ax.axvline(m,lw=.5,color='.8',ls=':')
  ax.set_xticks([60,65,66,71,72,78,80,85,88]);ax.tick_params(axis='x',labelsize=8.3)
 for ax in (a,b):ax.set_xlabel('Tested mass [MeV]')
 fig.suptitle('Signal echoes in the moving-background fit',fontsize=14,y=.985,fontweight='semibold')
 fig.text(.5,.949,'Current dense solver  |  same smooth 2021 background  |  deterministic injections',ha='center',fontsize=10)
 save(fig,'signal_echo_dense_replay')
if __name__=='__main__':main()
