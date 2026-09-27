"""Observed nominal-p plots; every value comes from exact C=0 fits or saved raw profiled roots."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/hps-fixed-background-mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]
data=pd.read_csv(B/'results/observed_curves.csv',dtype={'scope':str})
old=pd.read_csv(B/'results/profiled_raw_scan.csv',dtype={'scope':str})
labels={'2015':'2015 full','2016':'2016 full','2021':'2021 (10%)','shared':'Shared coupling','free':'Free nonnegative amplitudes'}
colors={'2015':'#326a91','2016':'#b35145','2021':'#278176','shared':'#202d42','free':'#9566a8'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42})
fig,axs=plt.subplots(2,1,figsize=(9.2,5.55),sharex=True)
for ax,scopes in zip(axs,[['2015','2016','2021'],['shared','free']]):
    for scope in scopes:
        d=data[data.scope==scope].sort_values('mass_MeV')
        ax.semilogy(d.mass_MeV,d.nominal_local_p,label=labels[scope],c=colors[scope],lw=1.3)
    ax.set(ylim=(1e-35,1.4),ylabel='Nominal local p-value')
    ax.set_yticks([1,1e-10,1e-20,1e-30])
    ax.axvspan(50,100,color='#bfab73',alpha=.10)
    ax.legend(loc='lower right',fontsize=8,ncol=2)
axs[0].set_title('Background held fixed in the signal likelihood (C = 0)',loc='left',fontsize=11)
axs[1].set(xlabel='Tested mass [MeV]',xlim=(19,250))
axs[1].text(.99,.48,'Shading: all three datasets contribute (50–100 MeV)',ha='right',va='top',transform=axs[1].transAxes,fontsize=8)
for x in [39,50,100,180]:axs[1].axvline(x,c='.65',ls=':',lw=.65)
axs[0].annotate('2015: 51 MeV',xy=(51,5.477e-33),xytext=(69,1e-28),fontsize=8,color=colors['2015'],arrowprops={'arrowstyle':'-','color':colors['2015']})
axs[1].annotate('22 MeV: only 2015 is active',xy=(22,3.3075e-30),xytext=(46,1e-23),fontsize=8,color=colors['shared'],arrowprops={'arrowstyle':'-','color':colors['shared']})
fig.tight_layout(h_pad=1.05)
for ext in ['pdf','png']:fig.savefig(B/f'figures/fixed_background_curves.{ext}',dpi=175,bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(10,5.05))
for ax,scope in zip(axs.flat,list(labels)):
    for table,label,color,style in [(data,'Fixed background','#ab4238','-'),(old,'Profiled GP uncertainty','#326a91','--')]:
        d=table[(table.scope==scope)&table.mass_MeV.between(50,100)].sort_values('mass_MeV')
        ax.plot(d.mass_MeV,-np.log10(d.nominal_local_p),label=label,c=color,ls=style,lw=1.1)
    ax.set_title(labels[scope],fontsize=9,loc='left')
    for lo,hi in [(65,67),(90,93)]:ax.axvspan(lo,hi,color='#bfab73',alpha=.10)
    ax.set(xlim=(50,100),ylim=(0,35),xticks=[50,60,70,80,90,100],yticks=[0,10,20,30])
for ax in axs[:,0]:ax.set_ylabel(r'$-\log_{10}(p_{\rm nominal})$')
for ax in axs[1,:2]:ax.set_xlabel('Tested mass [MeV]')
handles,labs=axs[0,0].get_legend_handles_labels()
fig.legend(handles,labs,loc='lower center',bbox_to_anchor=(.5,1.0),frameon=False,ncol=2,fontsize=9)
ax=axs[1,2]
from scipy.stats import norm
bank=np.load(B/'results/checkpoints/toys_m0132.npz')['shared_r']
obs=data[(data.scope=='shared')&(data.mass_MeV==66)].r.iloc[0]
xx=np.linspace(-10,10,300)
ax.hist(bank,bins=np.arange(-10,11,1),density=True,color='#327e80',alpha=.48,label='256 retrained-GP null toys')
ax.plot(xx,norm.pdf(xx),c='.25',ls='--',lw=1.2,label='Standard-normal reference')
ax.axvline(obs,c='#ab4238',lw=1.1,label=f'Observed r = {obs:.2f}')
ax.set(title='Shared fit at 66 MeV: null width',xlabel='Raw signed root r',ylabel='Density',xlim=(-10,10),ylim=(0,.65))
ax.title.set_fontsize(9);ax.title.set_position((0,1.));ax.title.set_ha('left')
ax.legend(loc='upper left',fontsize=6.8,frameon=False)
ax.text(.05,.64,f'Toy SD = {bank.std(ddof=1):.2f}',transform=ax.transAxes,ha='left',fontsize=8)
fig.tight_layout(w_pad=1.7,h_pad=1.3)
for ext in ['pdf','png']:fig.savefig(B/f'figures/fixed_background_comparison.{ext}',dpi=175,bbox_inches='tight')
plt.close(fig)
print('Wrote two vector and raster p-value figures.')
