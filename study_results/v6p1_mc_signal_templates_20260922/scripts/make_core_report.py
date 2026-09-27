"""Saved-result plots and report tables for the MC-centered window study."""
from make_report import *
import core_centering as cc

def main():
    s=pd.read_csv(D/'core_scans.csv');c=pd.read_csv(D/'core_comparison.csv');n=c[c.input_kind=='direct'];cent=pd.read_csv(D/'core_native_diagnostics.csv');sens=pd.read_csv(D/'core_center_sensitivity.csv');a=json.loads((D/'core_agreement.json').read_text())
    fig,ax=plt.subplots(4,1,figsize=(8.5,8.2),sharex=True,layout='constrained')
    styles=[('pole_centered','gaussian',GRAY,':','Original Gaussian'),('pole_centered','mc',RED,':','Original MC window'),('core_centered','gaussian','#8b8e36','--','Gaussian at MC center'),('core_centered','mc',BLUE,'-','MC-centered window')]
    for framework,kind,color,style,label in styles:
        q=s[(s.framework==framework)&((s.model=='gaussian') if kind=='gaussian' else (s.model!='gaussian'))].sort_values('mass_MeV')
        ax[0].semilogy(q.mass_MeV,q.display_epsilon2_90,color=color,ls=style,lw=1,label=label)
        ax[1].semilogy(q.mass_MeV,q.p0_fixed_mass,color=color,ls=style,lw=1)
    fresh=s[(s.framework=='core_centered')&(s.model=='mc_direct')]
    ax[0].scatter(fresh.mass_MeV,fresh.display_epsilon2_90,color=BLUE,s=18,zorder=6);ax[1].scatter(fresh.mass_MeV,fresh.p0_fixed_mass,color=BLUE,s=18,zorder=6)
    ax[2].plot(n.mass_MeV,n.mc_shift_ratio,'o-',color=BLUE,label='MC: new / original window')
    ax[2].plot(n.mass_MeV,n.new_mc_to_gaussian_ratio,'s--',color='#8b8e36',label='MC / Gaussian at MC center')
    ax[2].axhline(1,color=GRAY,lw=.6);ax[2].legend(frameon=False,fontsize=8)
    ax[3].plot(n.mass_MeV,n.mc_delta_r,'o-',color=BLUE);ax[3].axhline(0,color=GRAY,lw=.6)
    labels=[r'Observed $\varepsilon^2_{90}$',r'Local $p_0$','Native limit ratios',r'MC change in signed $r$']
    for axis,label in zip(ax,labels):axis.set_ylabel(label);axis.set_xlim(50,250)
    ax[0].legend(frameon=False,ncol=2,fontsize=8);ax[-1].set_xlabel('Generated mass hypothesis (MeV)')
    save(fig,'core_centered_comparison')
    rows=[]
    for r in cent.itertuples():
        q=sens[sens.mass_MeV==r.mass_MeV]
        rows.append([f'{r.mass_MeV:g}',f'{r.core_center_MeV:.3f}',f'{r.center_shift_MeV:+.3f}',f'{r.fitted_core_sigma_MeV:.3f}',f'{r.center_MC_bin_resample_std_MeV:.3f}',f'{abs(q.center_delta_MeV).max():.3f}',f'{100*abs(q.limit_ratio_to_default-1).max():.2f}'])
    table('core_centers',[r'$m_0$',r'$c(m_0)$',r'$c-m_0$',r'$\sigma_{\rm core}$',r'MC spread',r'Max. $|\Delta c|$',r'Max. UL change (\%)'],rows,'rrrrrrr')
    rows=[]
    for r in n.itertuples():
        rows.append([f'{r.mass_MeV:g}',sci(r.old_mc_limit,2),sci(r.new_mc_limit,2),f'{r.mc_shift_ratio:.3f}',f'{r.old_mc_p0:.5f}',f'{r.new_mc_p0:.5f}',f'{r.new_gaussian_p0:.5f}'])
    table('core_limits',[r'$m_0$',r'Old MC limit',r'New MC limit',r'New/old',r'Old MC $p_0$',r'New MC $p_0$',r'New G $p_0$'],rows,'rrrrrrr')
    rows=[]
    for r in n.itertuples():rows.append([f'{r.mass_MeV:g}',sci(r.new_gaussian_limit,2),f'{r.new_mc_to_gaussian_ratio:.3f}',f'{r.new_mc_vs_gaussian_delta_r:+.3f}',f'{100*r.old_mc_fit_fraction:.2f}',f'{100*r.new_mc_fit_fraction:.2f}'])
    table('core_matched',[r'$m_0$',r'Gaussian at $c$ limit',r'MC/G at $c$',r'$r_{MC}-r_G$',r'Old MC fit (\%)',r'New MC fit (\%)'],rows,'rrrrrr')
    fig,axs=plt.subplots(4,3,figsize=(9,8.5),layout='constrained')
    for ax,r in zip(axs.flat,cent.itertuples()):
        m=r.mass_MeV;edges,h=cc.template(int(m));x=(edges[:-1]+edges[1:])/2;dx=np.diff(edges)[0];sig=r.nominal_sigma_MeV
        _,fit=cc.locate(int(m));ax.step(x,h/h.sum()/dx,where='mid',color=BLUE,lw=.85,label='Reconstructed MC')
        ax.plot(fit['x'],fit['fit']/h.sum()/dx,color=RED,lw=1,label='Local core + pedestal fit')
        for sign in [-1,1]:
            ax.axvline(m+sign*2.25*sig,color=GRAY,ls='--',lw=.6)
            ax.axvline(r.core_center_MeV+sign*2.25*sig,color=BLUE,ls=':',lw=.9)
        ax.axvline(r.core_center_MeV,color=RED,lw=.65)
        ax.set_xlim(m-5*sig,m+4*sig);sel=(x>m-5*sig)&(x<m+4*sig);ax.set_ylim(0,float((h/h.sum()/dx)[sel].max())*1.08)
        ax.set_title(f'{m:g} MeV; center {r.core_center_MeV:.2f}',fontsize=9);ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[10:]:ax.axis('off')
    axs.flat[10].text(0,.9,'Dashed gray: original window\nDotted blue: shifted window\nSolid red: MC core center\n\nSame nominal sigma and\nunchanged reconstructed MC.',transform=axs.flat[10].transAxes,va='top',fontsize=9)
    for ax in axs[:,0]:ax.set_ylabel(r'Density (MeV$^{-1}$)',fontsize=8)
    axs[3,0].set_xlabel(r'Reconstructed $m_{ee}$ (MeV)',fontsize=8);axs[0,0].legend(frameon=False,fontsize=6)
    save(fig,'core_centered_windows')
    peak=a['peaks']['core_centered_mc_direct'];r=n[n.mass_MeV==80].iloc[0]
    text=(f'The direct-template centers shift by {abs(a["native_shift_range_MeV"][1]):.2f}--{abs(a["native_shift_range_MeV"][0]):.2f}~MeV below their generated poles. '
      f'The median absolute MC/Gaussian limit difference in the matched centered windows is {100*a["median_abs_new_MC_Gaussian_difference"]:.2f}\\%, compared with 22.05\\% for the original paired prescription. '
      f'The native new/old MC limit ratios span {a["new_to_old_MC_ratio_range"][0]:.3f}--{a["new_to_old_MC_ratio_range"][1]:.3f}. '
      f'The smallest native MC local probability is at $m_0=80$~MeV, $c={r.core_center_MeV:.3f}$~MeV: '
      f'$p_0={peak["p0"]:.5f}$ ($Z_0={peak["Z"]:.3f}$), compared with $p_0={r.old_mc_p0:.5f}$ before shifting the window.\n')
    (D/'report_core_summary.tex').write_text(text)
    print('Saved centered comparison, ten native window panels, three tables and numerical summary.')
if __name__=='__main__':main()
