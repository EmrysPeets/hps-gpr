"""Presentation-only window extension; consumes saved arrays, performs no fits."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from make_report import B,D,F,BLUE,RED,GRAY,sigma,save,table,sci


def main():
    scans=pd.read_csv(D/'window_scans.csv')
    comp=pd.read_csv(D/'window_comparison.csv')
    loo=pd.read_csv(D/'window_loo.csv')
    precision=pd.read_csv(D/'window_mc_precision.csv')
    summaries=[]
    for width in [2.25,2.4,2.5,2.6]:
        c=comp[np.isclose(comp.blind_sigma,width)].sort_values('mass_MeV')
        s=scans[np.isclose(scans.blind_sigma,width)]
        if len(c)!=10 or len(s)!=402:raise ValueError(f'Incomplete width {width}: {len(c)}/{len(s)}')
        r=c.limit_ratio.to_numpy(); dr=c.delta_r.to_numpy()
        l=loo[np.isclose(loo.blind_sigma,width)]
        p=precision[np.isclose(precision.blind_sigma,width)]
        passed=l.shape_check_pass.astype(str).str.lower().isin(['true','1'])
        summaries.append([f'{width:g}',f'{np.median(r):.4f}',f'{r.min():.4f}--{r.max():.4f}',
            f'{100*np.median(abs(r-1)):.2f}',f'{abs(dr).max():.3f}',f'{int(passed.sum())}/{len(l)}',
            f'{100*p.relative_limit_std.max():.2f}'])
        if np.isclose(width,2.25):continue
        tag=str(width).replace('.','p')
        g=s[s.model=='gaussian'].sort_values('mass_MeV')
        mc=s[s.model!='gaussian'].sort_values('mass_MeV')
        native=mc[mc.model=='mc_direct']
        fig=plt.figure(figsize=(9,5.4),layout='constrained')
        grid=fig.add_gridspec(3,2,height_ratios=[1.05,1,1])
        axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1]),
              fig.add_subplot(grid[2,0]),fig.add_subplot(grid[2,1])]
        for data,color,ls,label in [(g,GRAY,'-','Gaussian'),(mc,RED,'--','MC morph: exploratory')]:
            axes[0].semilogy(data.mass_MeV,data.display_epsilon2_90,color=color,ls=ls,lw=1,label=label)
            axes[1].semilogy(data.mass_MeV,data.p0_fixed_mass,color=color,ls=ls,lw=.9)
            axes[2].plot(data.mass_MeV,data.signal_fraction_in_fit,color=color,ls=ls,lw=.9)
        axes[0].scatter(native.mass_MeV,native.display_epsilon2_90,color=BLUE,s=15,zorder=5,label='Direct MC')
        axes[1].scatter(native.mass_MeV,native.p0_fixed_mass,color=BLUE,s=15,zorder=5)
        axes[2].scatter(native.mass_MeV,native.signal_fraction_in_fit,color=BLUE,s=15,zorder=5)
        axes[3].plot(c.mass_MeV,c.limit_ratio,'o-',color=BLUE,ms=3,lw=.8)
        axes[4].plot(c.mass_MeV,c.delta_r,'o-',color=BLUE,ms=3,lw=.8)
        axes[3].axhline(1,color=GRAY,lw=.7);axes[4].axhline(0,color=GRAY,lw=.7)
        labels=[r'Observed $\varepsilon^2_{90}$',r'Local $p_0$','Fraction in fitted bins','Direct MC / Gaussian limit',r'Direct $\Delta r$']
        for ax,label in zip(axes,labels):
            ax.set_ylabel(label,fontsize=8);ax.set_xlim(50,250);ax.tick_params(labelsize=7)
        for ax in axes[3:]:ax.set_xlabel('Test mass (MeV)',fontsize=8)
        axes[0].legend(frameon=False,ncol=3,fontsize=7)
        axes[1].set_ylim(max(1e-9,s.p0_fixed_mass.min()*.65),.75)
        save(fig,f'window_{tag}')
        rows=[]
        for q in c.itertuples():
            rows.append([f'{q.mass_MeV:g}',sci(q.gaussian_limit,2),sci(q.mc_limit,2),f'{q.limit_ratio:.3f}',
                         f'{q.gaussian_p0:.4f}',f'{q.mc_p0:.4f}',f'{q.delta_r:+.3f}'])
        table(f'window_{tag}',['$m_0$','$\\eps^2_{90,G}$','$\\eps^2_{90,MC}$','Ratio','$p_{0,G}$','$p_{0,MC}$','$\\Delta r$'],rows,'rrrrrrr')
    table('window_summary',[r'$h/\sigma_m$','Median ratio','Ratio range',r'Med. $|R-1|$ (\%)',r'Max. $|\Delta r|$','LOO pass','MC spread (\%)'],summaries,'rrrrrrr')
    # Gallery shows the saved transported inputs and mixture itself, independently of fit windows.
    samples=[]
    for mass in range(50,251,20):
        path=B/'histograms'/'morphed'/f'm{mass:03d}.npz'
        z=np.load(path,allow_pickle=False)
        samples.append((mass,z['edges_GeV'],z['probability'],z['lower_probability'],z['upper_probability']))
    for zoom in [False,True]:
        fig,axs=plt.subplots(4,3,figsize=(9,8.5),layout='constrained')
        for ax,(mass,edges,p,lo,hi) in zip(axs.flat,samples):
            x=(edges[:-1]+edges[1:])*500;dx=np.diff(edges)*1000
            ax.step(x,lo/dx,where='mid',lw=.65,color='#889b46',alpha=.8,label='Lower mass transported')
            ax.step(x,hi/dx,where='mid',lw=.65,color='#9173aa',alpha=.8,label='Upper mass transported')
            ax.step(x,p/dx,where='mid',lw=1,color=BLUE,label='Midpoint CDF mixture')
            sig=float(sigma(mass))*1000
            for sign in [-1,1]:ax.axvline(mass+sign*2.25*sig,color=RED,lw=.7,ls=':')
            if zoom:
                ax.set_xlim(mass-7*sig,mass+7*sig)
                sel=abs(x-mass)<7*sig
                ax.set_ylim(0,max(p[sel].max(),lo[sel].max(),hi[sel].max())/dx[0]*1.05)
            else:ax.set_xlim(0,400);ax.set_yscale('log');ax.set_ylim(1e-6,.4)
            ax.set_title(f'{mass} MeV: {mass-10}/{mass+10} MeV inputs'+(' *' if mass<=90 else ''),fontsize=8)
            ax.tick_params(labelsize=7)
        axs.flat[-1].axis('off')
        axs.flat[-1].text(.03,.86,'MC-only shape interpolation\n\n* 50/70/90 MeV touch the\nlow-mass region with failed\nheld-out shape checks.\n\nNo new simulated sample.\nNo truth-response certification.',transform=axs.flat[-1].transAxes,fontsize=9,va='top')
        for ax in axs[-1,:2]:ax.set_xlabel(r'Reconstructed $m_{ee}$ (MeV)',fontsize=8)
        for ax in axs[:,0]:ax.set_ylabel(r'Density (MeV$^{-1}$)',fontsize=8)
        axs[0,0].legend(frameon=False,fontsize=5.8,loc='upper right')
        save(fig,'morph_gallery_core' if zoom else 'morph_gallery_full')
    print('Generated three separate window figures/tables, one summary, and two morph galleries; no fits.')

if __name__=='__main__':main()
