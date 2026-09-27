"""Replot saved values only; no new fitting or scientific changes."""
from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/rcmeet_23b_mpl'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
B=Path(__file__).resolve().parent
A=B/'assets';A.mkdir(exist_ok=True)
src=B.parent/'unblind_meeting_RCmeet_20260923/science/data/slide13_heldout_bins.csv'
d=np.genfromtxt(src,delimiter=',',names=True,dtype=None,encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'pdf.fonttype':42})
fig,axs=plt.subplots(2,3,figsize=(13.2,5.05),sharex='col',gridspec_kw={'height_ratios':[1.65,1]})
for k,(year,lo,hi) in enumerate([(2015,19,100),(2016,39,180),(2021,50,250)]):
 r=d[d['dataset']==year];x=r['bin_center_MeV'];bw=r['bin_width_MeV'];n=r['observed_counts'];b=r['heldout_GP_mean'];res=r['standardized_residual']
 a=axs[0,k];a.errorbar(x,n/bw,yerr=np.sqrt(n)/bw,fmt='.',c='#1b1b1b',ms=2.3,lw=.35,alpha=.6);a.plot(x,b/bw,c='#c99522',lw=2.2,zorder=4);a.set(title=str(year)+(' 10%' if year==2021 else ' full'),yscale='log',xlim=(lo,hi))
 a=axs[1,k];a.axhspan(-2,2,color='#2b9a50',alpha=.18);a.axhline(0,c='.5',lw=.7);a.plot(x,res,c='#1b1b1b',marker='.',ls='-',lw=.55,ms=2.4);a.set(xlabel=r'$m_{ee}$ [MeV]',xlim=(lo,hi),ylim=(-4,4),yticks=[-4,-2,0,2,4])
axs[0,0].set_ylabel('Events / MeV');axs[1,0].set_ylabel('Standardized\nresidual '+r'$r_i$')
fig.legend(handles=[Line2D([],[],c='#1b1b1b',marker='.',ls='',label='Observed counts'),Line2D([],[],c='#c99522',lw=2,label='Held-out GP prediction'),Patch(color='#2b9a50',alpha=.18,label='±2 reference band')],loc='upper center',ncol=3,frameon=False,fontsize=12)
fig.tight_layout(rect=(0,0,1,.92),h_pad=.35,w_pad=1.8)
for ext in ('png','pdf'):fig.savefig(A/f'slide13_explained.{ext}',dpi=280,bbox_inches='tight',facecolor='white')
plt.close(fig)
fig=plt.figure(figsize=(6.4,1.0));fig.text(.01,.5,r'$r_i=\frac{n_i-\widehat b_i}{\sqrt{\widehat b_i+C_{\mathrm{GP},ii}}}$',fontsize=29,va='center',color='black')
for ext in ('png','pdf','svg'):fig.savefig(A/f'standardized_residual.{ext}',dpi=320,bbox_inches='tight',pad_inches=.025,facecolor='white')
plt.close(fig)
print('Reused all',len(d),'saved bin values. No numerical changes.')
