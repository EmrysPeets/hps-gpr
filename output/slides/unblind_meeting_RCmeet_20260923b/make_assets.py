"""Presentation-sized views of verified scientific results and BEST Eq.(19)."""
from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/tmp/rcmeet_23b_mpl'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;A=B/'assets';A.mkdir(exist_ok=True)
c=json.loads((B/'content/slide29_copy.json').read_text())
plt.rcParams.update({'mathtext.fontset':'cm','font.family':'serif','pdf.fonttype':42})
fig=plt.figure(figsize=(8,1.3));fig.text(.01,.5,'$'+c['main_equation_latex'].replace('m_{A\\prime}',"m_{A'}")+'$',fontsize=38,va='center',color='black')
for ext in ['png','pdf','svg']:fig.savefig(A/f'BEST_yield_to_coupling.{ext}',dpi=340,bbox_inches='tight',pad_inches=.05,facecolor='white')
plt.close(fig)
plt.rcParams.update({'mathtext.fontset':'dejavusans','font.family':'DejaVu Sans','font.size':14,'axes.labelsize':15,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
d=np.genfromtxt(B/'science/data/sideband_center_summary.csv',delimiter=',',names=True)
x=d['center_MeV'];blue='#1b77a4';red='#bb3636'
fig,ax=plt.subplots(figsize=(11.5,3.3))
ax.fill_between(x,d['toy_q05'],d['toy_q95'],color=blue,alpha=.16,label='Pointwise 90% (256 toys)')
ax.plot(x,d['toy_median'],color=blue,lw=2,ls='--',label='Conditional toy median')
ax.plot(x,d['observed_D_per_bin'],color=red,lw=2.2,marker='o',ms=3.5,label='Observed')
ax.set(xlim=(50,250),ylim=(.77,1.19),xlabel='Excluded-window center [MeV]',ylabel=r'$D_{\rm side}/N_{\rm side}$',yticks=[.8,.9,1,1.1])
ax.legend(loc='upper center',ncol=3,frameon=False,fontsize=12)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(A/f'slide14_scan_for_deck.{ext}',dpi=280,bbox_inches='tight',facecolor='white')
plt.close(fig)
print('BEST equation and compact sideband scan saved.')
