"""Presentation of the saved MC-only analytic and asymmetric diagnostics."""
from make_report import *
import analytic_shapes as a

def main():
    models=json.loads((D/'analytic_shift_models.json').read_text());pred=pd.read_csv(D/'analytic_shift_predictions.csv');metrics=pd.read_csv(D/'shared_shape_metrics.csv');held=pd.read_csv(D/'shared_shape_heldout.csv');scans=pd.read_csv(D/'asymmetric_scans.csv');core=a.CORE
    colors=plt.cm.viridis(np.linspace(.05,.95,10));dense=np.linspace(60,240,361)
    fig,axs=plt.subplots(2,1,figsize=(8,5.8),sharex=True,layout='constrained')
    axs[0].errorbar(a.NATIVE,core.center_shift_MeV,yerr=core.center_MC_bin_resample_std_MeV,fmt='ko',ms=4,label='MC cores; bin-resample spread')
    for model,col in zip(models['models'],['#888888','#b35b3a','#488c6b','#285fab']):
        axs[0].plot(dense,a.design(dense,model['model'])@model['coefficients'],color=col,label=model['model'])
        q=pred[pred.model==model['model']];axs[1].plot(q.mass_MeV,q.heldout_shift_MeV-q.measured_shift_MeV,'o-',ms=3,color=col,label=model['model'])
    axs[0].set_ylabel(r'Core shift $c-m_0$ (MeV)');axs[1].set_ylabel('Held-out prediction error (MeV)');axs[1].set_xlabel('Generated mass (MeV)');axs[1].axhline(0,color='k',lw=.5);axs[0].legend(frameon=False,ncol=2,fontsize=8);save(fig,'analytic_center_models')
    table('analytic_models',['Model','Parameters','Fit RMS (MeV)','LOO RMS (MeV)','Max. LOO (MeV)'],[[r['model'],r['parameters'],f'{r["train_RMS_MeV"]:.4f}',f'{r["LOO_RMS_MeV"]:.4f}',f'{r["LOO_max_abs_MeV"]:.4f}'] for r in models['models']],'lrrrr')
    fig,axs=plt.subplots(2,2,figsize=(9,6.5),layout='constrained')
    ug=np.linspace(-60,100,8001);uc=(ug[:-1]+ug[1:])/2;du=np.diff(ug)
    for m,color in zip(a.NATIVE,colors):
        r=core.loc[m];ed=a.MC[m]['edges_GeV']*1000;h=a.MC[m]['sumw'];cdf=np.r_[0,np.cumsum(h)]/h.sum();x=(ed[:-1]+ed[1:])/2
        F=lambda v:np.interp(v,ed,cdf,left=0,right=1)
        axs[0,0].plot(x-r.core_center_MeV,h/h.sum()/np.diff(ed),color=color,lw=.85,label=f'{m} MeV')
        p=np.diff(F(r.core_center_MeV+r.fitted_core_sigma_MeV*ug))/du
        axs[0,1].plot(uc,p,color=color,lw=.85)
        norm=float(F(r.core_center_MeV+2*r.fitted_core_sigma_MeV)-F(r.core_center_MeV-2*r.fitted_core_sigma_MeV))
        axs[1,0].plot(uc,p/norm,color=color,lw=.85)
        axs[1,1].semilogy(uc,np.maximum(p,1e-9),color=color,lw=.85)
    pool=np.diff(a.shared_cdf(ug))/du
    axs[0,1].plot(uc,pool,'k--',lw=1.4,label='Equal-mass common shape');axs[0,1].legend(frameon=False,fontsize=7)
    cp=np.diff(a.shared_cdf(ug))/(a.shared_cdf([2])[0]-a.shared_cdf([-2])[0])/du;axs[1,0].plot(uc,cp,'k--',lw=1.4)
    axs[0,0].set(xlim=(-20,20),xlabel=r'$m_{ee}-c$ (MeV)',ylabel=r'Full-normalized density (MeV$^{-1}$)',title='Centered, unit area; physical widths retained')
    axs[0,0].legend(frameon=False,ncol=2,fontsize=6)
    axs[0,1].set(xlim=(-8,8),xlabel=r'$u=(m_{ee}-c)/\sigma_{core}$',ylabel='Full-normalized density',title='Centered and width-scaled')
    axs[1,0].set(xlim=(-2,2),xlabel='u',ylabel='Conditional core density',title=r'Core only: unit area on $|u|<2$')
    axs[1,1].set(xlim=(-30,60),ylim=(1e-6,.6),xlabel='u',ylabel='Full-normalized density',title='Tails retain their original weights')
    save(fig,'shared_shape_overlays')
    rows=[]
    for r in held.itertuples():
        f=float(metrics[metrics.mass_MeV==r.mass_MeV].core_fraction_2core_sigma.iloc[0])
        rows.append([f'{r.mass_MeV:g}',f'{100*f:.1f}',f'{r.shared_core_CDF_distance:.3f}',f'{r.shared_full_CDF_distance:.3f}',f'{r.shared_UL_ratio:.3f}',f'{r.morph_UL_ratio:.3f}',f'{r.end_to_end_UL_ratio:.3f}'])
    table('shared_validation',[r'$m_0$',r'Core (\%)',r'$D_{core}$',r'$D_{full}$','Shared UL ratio','Morph UL ratio','Predictive UL ratio'],rows,'rrrrrrr')
    fig,axs=plt.subplots(2,1,figsize=(8,4.8),sharex=True,layout='constrained')
    axs[0].plot(held.mass_MeV,held.shared_full_CDF_distance,'o-',label='Common standardized shape',color=BLUE);axs[0].plot(held.mass_MeV,held.morph_CDF_distance,'s--',label='Neighboring-mass morph',color=RED);axs[0].set_ylabel('Held-out full-support CDF distance');axs[0].legend(frameon=False,fontsize=8)
    for col,label,color in [('shared_UL_ratio','Common shape; known center/width',BLUE),('end_to_end_UL_ratio','Common shape; predicted center/width','#8a9b39'),('morph_UL_ratio','Neighboring morph',RED)]:axs[1].plot(held.mass_MeV,held[col],'o-',label=label,color=color,ms=3)
    axs[1].axhline(1,color='k',lw=.5);axs[1].set_ylabel('Held-out / direct-MC upper limit');axs[1].set_xlabel('Generated mass (MeV)');axs[1].legend(frameon=False,fontsize=7);save(fig,'shared_shape_validation')
    fig,axs=plt.subplots(2,1,figsize=(8,5),sharex=True,layout='constrained')
    axs[0].plot(metrics.mass_MeV,metrics.left_tail_2p25nominal,'o-',color=BLUE,label='Left of centered -2.25 nominal sigma');axs[0].plot(metrics.mass_MeV,metrics.right_tail_2p25nominal,'s-',color=RED,label='Right of centered +2.25 nominal sigma');axs[0].set_ylabel('Selected MC probability');axs[0].legend(frameon=False,fontsize=8)
    axs[1].plot(metrics.mass_MeV,metrics.inside_left,'o-',color=BLUE,label='Left half inside window');axs[1].plot(metrics.mass_MeV,metrics.inside_right,'s-',color=RED,label='Right half inside window');axs[1].set_ylabel('Selected MC probability');axs[1].set_xlabel('Generated mass (MeV)');axs[1].legend(frameon=False,fontsize=8);save(fig,'left_right_MC_probability')
    fig,axs=plt.subplots(4,1,figsize=(8,8),sharex=True,layout='constrained');rows=[]
    for (name,(l,r)),color in zip(a.WINDOWS.items(),['#444444','#256daf','#b94d39','#3d9152','#9c61a7']):
        q=scans[(scans.window==name)&(scans.model=='mc')];g=scans[(scans.window==name)&(scans.model=='gaussian')];label=f'[-{l:g}, +{r:g}]'
        axs[0].plot(q.mass_MeV,q.mc_UL_ratio_to_symmetric,'o-',ms=3,color=color,label=label);axs[1].plot(q.mass_MeV,q.mc_delta_r,'o-',ms=3,color=color)
        axs[2].semilogy(q.mass_MeV,q.p0_fixed_mass,'o-',ms=3,color=color);axs[3].plot(q.mass_MeV,q.signal_fraction_in_fit,'o-',ms=3,color=color)
        peak=q.loc[q.p0_fixed_mass.idxmin()];ratio=q.epsilon2_90.to_numpy()/g.epsilon2_90.to_numpy()
        rows.append([f'$[-{l:g},+{r:g}]$',f'{q.mc_UL_ratio_to_symmetric.min():.3f}--{q.mc_UL_ratio_to_symmetric.max():.3f}',f'{100*np.median(abs(ratio-1)):.1f}',f'{peak.mass_MeV:g}',f'{peak.p0_fixed_mass:.5f}'])
    for ax,label in zip(axs,['MC UL / symmetric UL',r'MC change in signed $r$',r'Local MC $p_0$','MC probability in fitted bins']):ax.set_ylabel(label)
    axs[0].axhline(1,color='k',lw=.5);axs[1].axhline(0,color='k',lw=.5);axs[0].legend(frameon=False,ncol=3,fontsize=8);axs[-1].set_xlabel('Generated mass (MeV)');save(fig,'asymmetric_comparisons')
    table('asymmetric_summary',['Window in nominal sigma','MC UL / symmetric',r'Med. $|MC/G-1|$ (\%)','Min. mass','Min. local p0'],rows,'lrrrr')
    print('Saved analytic-shift, normalized-shape, held-out and asymmetric-window figures/tables.')
if __name__=='__main__':main()
