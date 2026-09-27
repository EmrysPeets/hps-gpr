from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/v645_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
B=Path(__file__).resolve().parents[1]
SCOPES=('2016','2021','combined');LABEL={'2016':'2016','2021':'2021','combined':'Combined'}
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def save(fig,name):
    fig.savefig(B/f'figures/{name}.pdf',bbox_inches='tight')
    fig.savefig(B/f'figures/{name}.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def correlations():
    fig=plt.figure(figsize=(7.3,5.0));gs=fig.add_gridspec(2,6,height_ratios=[1,1.08],hspace=.66,wspace=.6)
    axes=[]
    for i,s in enumerate(SCOPES):
        ax=fig.add_subplot(gs[0,2*i:2*i+2]);axes.append(ax)
        z=np.load(B/f'results/correlations_{s}.npz');m=z['masses_MeV']
        im=ax.imshow(z['A'],origin='lower',extent=(m[0]-.5,m[-1]+.5,m[0]-.5,m[-1]+.5),vmin=-1,vmax=1,cmap='RdBu_r',interpolation='nearest',aspect='equal')
        ax.set_title(LABEL[s]);ax.set_xlabel('Mass [MeV]');ax.set_ylabel('Mass [MeV]' if i==0 else '')
        ax.tick_params(labelsize=8)
        if s=='combined':
            for bound in (100.5,175.5):ax.axvline(bound,color='k',lw=.5,ls=':');ax.axhline(bound,color='k',lw=.5,ls=':')
    fig.colorbar(im,ax=axes,location='right',fraction=.023,pad=.025,label='Signed-root correlation')
    d=pd.read_csv(B/'results/correlation_by_resolution_distance.csv',dtype={'scope':str})
    for i,s in enumerate(('2016','2021')):
        ax=fig.add_subplot(gs[1,3*i:3*i+3]);q=d[d.scope==s];x=q.u_center.to_numpy()
        ax.fill_between(x,q.A_p16.to_numpy(),q.A_p84.to_numpy(),color='.6',alpha=.22,lw=0)
        ax.plot(x,q.A_median,color='#b12a31',lw=1.5,label='A median; pair spread')
        ax.plot(x,q.B_median,color='#176899',lw=1.2,ls='--',label='B median')
        ax.plot(x,q.gaussian_overlap_median,color='.2',lw=1.1,ls=':',label='Gaussian overlap reference')
        ax.axhline(0,color='.4',lw=.5);ax.set(xlim=(0,6),ylim=(-1,1.03),title=LABEL[s],xlabel='Center separation / RMS core width',ylabel='Signed-root correlation' if i==0 else '')
        ax.grid(alpha=.15)
    handles=[Line2D([],[],color='#b12a31',lw=1.5,label='A median; gray = pair spread'),Line2D([],[],color='#176899',ls='--',lw=1.3,label='B median'),Line2D([],[],color='.2',ls=':',lw=1.2,label='Gaussian overlap reference')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.055),ncol=3,frameon=False,fontsize=7.5)
    save(fig,'mass_correlation')


def tails():
    d=pd.read_csv(B/'results/correlation_global_curves.csv',dtype={'scope':str})
    summary=pd.read_csv(B/'results/correlation_global_summary.csv',dtype={'scope':str}).set_index('scope')
    fig,axes=plt.subplots(1,3,figsize=(7.3,2.8),sharex=True,sharey=True)
    for s,ax in zip(SCOPES,axes):
        q=d[d.scope==s];x=q.alpha.to_numpy()
        ax.step(x,q.p,where='post',color='#b12a31',lw=1.4)
        ax.fill_between(x,q.low.to_numpy(),q.high.to_numpy(),step='post',color='#b12a31',alpha=.14,lw=0)
        ax.step(x,q.independent_empirical_p,where='post',color='#176899',lw=1.25,ls='--')
        if s!='combined':
            ax.plot(x,q.MC_resolution_sidak_p,color='#308453',ls='-.',lw=1.4)
            ax.plot(x,q.MC_resolution_linear_p,color='#bd7420',ls=':',lw=1.4)
        ax.axvline(summary.loc[s,'local_p'],color='.25',lw=.7)
        ax.set(xscale='log',yscale='log',xlim=(.85/1025,.101),ylim=(.025,1.12),title=LABEL[s],xlabel='Local-rank threshold')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Global tail probability')
    handles=[Line2D([],[],color='#b12a31',lw=1.5,label='Complete B scans; 95% interval'),Line2D([],[],color='#176899',lw=1.4,ls='--',label='Independent-mass control'),Line2D([],[],color='#308453',lw=1.4,ls='-.',label='Resolution count: Sidak'),Line2D([],[],color='#bd7420',lw=1.4,ls=':',label='Resolution count: linear')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.09),ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.09,right=.99,bottom=.19,top=.76,wspace=.16)
    save(fig,'correlation_global_tails')


if __name__=='__main__':correlations();tails()
