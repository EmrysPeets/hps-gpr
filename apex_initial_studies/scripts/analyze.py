"""Saved-result comparison. All mass-window rankings are descriptive, post selected."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B=Path(__file__).resolve().parents[1]
read=lambda f:pd.read_csv(B/f)
A=read('derived/apex_limit_digitized.csv'); P=read('derived/apex_pvalue_digitized.csv')
RES=read('derived/apex_total_resolution.csv')
H=read('inputs/hps_combined_v504.csv'); T=read('inputs/hps_2021_released.csv')
F=read('inputs/hps_figure2_contours.csv'); S=read('inputs/hps_source_scan.csv')
CAT=read('inputs/hps_projection_catalogue.csv')
COEFF=json.loads((B/'inputs/hps_resolution.json').read_text())

def interp(d,x,col,log=False):
    x=np.asarray(x); y=d[col].to_numpy()
    out=np.interp(x,d.mass_MeV,np.log(y) if log else y,left=np.nan,right=np.nan)
    return np.exp(out) if log else out

def sigma(m,year='2021'):
    return 1000*np.polynomial.polynomial.polyval(np.asarray(m)/1000,COEFF[year]['sigma_polynomial_GeV'])

def ar(m,col='value'):return interp(A,m,col,True)
def pr(m,col='value'):return interp(P,m,col,True)

def intervals(x,mask):
    x=np.asarray(x);idx=np.flatnonzero(mask)
    if not len(idx):return []
    return [[float(x[g[0]]),float(x[g[-1]])] for g in np.split(idx,np.flatnonzero(np.diff(idx)>1)+1)]

def savefig(fig,name):
    for ext in ['pdf','png']:fig.savefig(B/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    lo=int(np.ceil(max(A.mass_MeV.min(),P.mass_MeV.min(),H.mass_MeV.min())))
    hi=int(np.floor(min(A.mass_MeV.max(),P.mass_MeV.max(),H.mass_MeV.max())))
    m=np.arange(lo,hi+1); n=len(m)
    h=H.set_index('mass_MeV').loc[m]; t=T.set_index('mass_MeV').loc[m]; f=F.set_index('mass_MeV').loc[m]
    scan=pd.DataFrame(dict(mass_MeV=m,datasets=h.dataset_set.to_numpy(),
        apex_limit=ar(m),apex_limit_pixel_low=ar(m,'pixel_low'),apex_limit_pixel_high=ar(m,'pixel_high'),
        apex_p=pr(m),apex_p_pixel_low=pr(m,'pixel_low'),apex_p_pixel_high=pr(m,'pixel_high'),
        hps_limit=h.eps2_observed.to_numpy(),hps_p=h.p0_local_asymptotic.to_numpy(),hps_Z=h.Z_local_asymptotic.to_numpy(),
        hps_2021_limit=t.eps2_observed.to_numpy(),hps_2021_p=t.p0_local_asymptotic.to_numpy(),
        hps_2021_full_observed_equivalent=t.eps2_observed.to_numpy()/np.sqrt(10),
        hps_combined_full_observed_equivalent=f.projected_three_minimal.to_numpy(),
        hps_four_current=f.observed_four_minimal.to_numpy(),hps_four_full_equivalent=f.projected_four_minimal.to_numpy(),
        sigma_hps_2021_MeV=sigma(m),sigma_apex_if_MeV=interp(RES,m,'total_axis_value')))
    scan['sigma_apex_if_percent_MeV']=scan.sigma_apex_if_MeV*m/100
    scan['p_concordance']=np.maximum(scan.apex_p,scan.hps_p)
    scan['p_concordance_pixel_low']=np.maximum(scan.apex_p_pixel_low,scan.hps_p)
    # Conditional upper bound on a scan of the stated finite grid, provided the input p's are valid.
    scan['p_concordance_bonferroni']=np.minimum(1,n*scan.p_concordance)
    for col in ['hps_limit','hps_2021_full_observed_equivalent','hps_combined_full_observed_equivalent','hps_four_current','hps_four_full_equivalent']:
        scan[col+'_over_apex']=scan[col]/scan.apex_limit
    ten=S[S.lane=='ten'];scan['native_source_Z']=interp(ten,m,'Z')
    scan['conditional_sqrt10_Z']=np.sqrt(10)*scan.native_source_Z
    scan.to_csv(B/'derived/common_mass_scan.csv',index=False)
    dense=P[P.mass_MeV.between(lo,hi)].copy()
    dense['hps_p_log_interpolated']=interp(H,dense.mass_MeV,'p0_local_asymptotic',True)
    dense['p_concordance']=np.maximum(dense.value,dense.hps_p_log_interpolated)
    dense.to_csv(B/'derived/dense_display_scan.csv',index=False)

    # Catalogue all resolved APEX local minima, with an explicitly fixed image-scale separation.
    ii=find_peaks(-np.log(P.value),distance=8,prominence=.25)[0]
    minima=P.iloc[ii].copy().rename(columns={'value':'apex_p'})
    minima['Z_one_sided_equivalent']=norm.isf(minima.apex_p)
    minima['apex_limit']=ar(minima.mass_MeV)
    minima['hps_p_interp']=interp(H,minima.mass_MeV,'p0_local_asymptotic',True)
    minima['hps_limit_interp']=interp(H,minima.mass_MeV,'eps2_observed',True)
    minima['p_concordance']=np.maximum(minima.apex_p,minima.hps_p_interp)
    minima['sigma_apex_if_MeV']=interp(RES,minima.mass_MeV,'total_axis_value')
    minima['sigma_hps_2021_MeV']=sigma(minima.mass_MeV)
    minima['in_hps_support']=minima.mass_MeV.between(H.mass_MeV.min(),H.mass_MeV.max())
    minima.sort_values('apex_p').to_csv(B/'derived/apex_local_minima.csv',index=False)

    # HPS catalogue: fixed-mass APEX comparisons, then a separately labelled broad-window minimum.
    peaks=[]
    for label,d,pc,zc in [('combined',H,'p0_local_asymptotic','Z_local_asymptotic'),
                           ('native_10pct',ten,'p0','Z'),
                           ('historical_1pct',S[S.lane=='one'],'p0','Z')]:
        ids=find_peaks(d[zc].to_numpy(),distance=2)[0]
        for _,r in d.iloc[ids].iterrows():
            mass=float(r.mass_MeV)
            if not lo<=mass<=hi or r[pc]>.25:continue
            w=float(sigma(mass)); pwin=P[P.mass_MeV.between(mass-w,mass+w)]
            near=pwin.loc[pwin.value.idxmin()]
            # Resolution overlap is a display distance, not an uncertainty on a fitted centroid.
            sig_a=float(interp(RES,mass,'total_axis_value'))
            peaks.append(dict(lane=label,mass_MeV=mass,hps_p=r[pc],hps_Z=r[zc],apex_p_same_mass=float(pr(mass)),
                apex_limit=float(ar(mass)),sigma_hps_MeV=w,sigma_apex_if_MeV=sig_a,
                apex_min_within_one_hps_sigma=float(near.value),apex_window_min_mass_MeV=float(near.mass_MeV),
                offset_MeV=float(near.mass_MeV-mass),
                offset_over_quadrature_resolution=abs(float(near.mass_MeV-mass))/np.hypot(w,sig_a),
                conditional_full_Z=float(r[zc]*np.sqrt(10 if label=='native_10pct' else 100)) if label!='combined' else np.nan))
    peaks=pd.DataFrame(peaks);peaks.to_csv(B/'derived/hps_peak_comparisons.csv',index=False)

    # Infer the already chosen injected coupling using the saved background-density conversion.
    scenarios=[]
    for name in ['ten_160','ten_210','one_210','one_extra244']:
        r=CAT[CAT.scenario==name].iloc[0]; mass=float(r.mass_MeV)
        bg=read(f'inputs/{name}_background_asimov.csv').set_index('mass_MeV').loc[mass]
        ma=read(f'inputs/{name}_matched_asimov.csv').set_index('mass_MeV').loc[mass]
        K=float(bg.K_counts_per_epsilon2);br=float(bg.branching_factor)
        inj=float(r.injected_yield)/K*br; ordinary=float(r.naive_scaled_yield)/K*br
        scenarios.append(dict(scenario=name,lane=r.lane,mass_MeV=mass,source_p=r.p0,source_Z=r.source_Z,
            matched_Z=r.matched_asimov_Z,yield_scaled_Z=r.naive_yield_asimov_Z,
            injected_epsilon2=inj,yield_scaled_epsilon2=ordinary,
            projected_limit=float(ma.epsilon2_90),background_asimov_limit=float(bg.epsilon2_90),
            apex_limit=float(ar(mass)),apex_p=float(pr(mass)),
            injection_over_apex_limit=inj/float(ar(mass)),yield_scaled_over_apex_limit=ordinary/float(ar(mass))))
    scenarios=pd.DataFrame(scenarios);scenarios.to_csv(B/'derived/projected_signal_compatibility.csv',index=False)

    # Raw pixel envelopes, +/-2 horizontal pixels and threshold variations: extraction sensitivity.
    checks=[]
    for name,d in [('limit',A),('pvalue',P)]:
        x=d.mass_MeV.to_numpy(); vals=[]
        for threshold in [120,150,180]:
            dd=d if threshold==150 else read(f'derived/apex_{name}_threshold_{threshold}.csv')
            vals.append(interp(dd,x,'value',True))
        vv=np.asarray(vals)
        checks.append(dict(trace=name,n_columns=len(d),mass_lo=float(x.min()),mass_hi=float(x.max()),
            x_two_pixel_MeV=float(d.mass_pixel_error_MeV.iloc[0]),
            median_relative_stroke_halfspan=float(np.median((d.pixel_high-d.pixel_low)/(2*d.value))),
            max_threshold_log10_difference=float(np.nanmax(abs(np.log10(vv/vv[1]))))))
    pd.DataFrame(checks).to_csv(B/'derived/digitization_sensitivity.csv',index=False)
    # Competitive grid nodes using the full pixel envelope in a +/-2 pixel x neighbourhood.
    envelope_lo=[];envelope_hi=[]
    for mass in m:
        sub=A[abs(A.mass_MeV-mass)<=float(A.mass_pixel_error_MeV.iloc[0])+.09]
        envelope_lo.append(sub.pixel_low.min());envelope_hi.append(sub.pixel_high.max())
    scan['apex_local_pixel_envelope_low']=envelope_lo;scan['apex_local_pixel_envelope_high']=envelope_hi
    scan.to_csv(B/'derived/common_mass_scan.csv',index=False)
    competitive={}
    for col in ['hps_limit','hps_2021_full_observed_equivalent','hps_combined_full_observed_equivalent','hps_four_current','hps_four_full_equivalent']:
        competitive[col]=dict(central_nodes=intervals(m,scan[col]<scan.apex_limit),
           robust_nodes=intervals(m,scan[col]<scan.apex_local_pixel_envelope_low),
           possible_nodes=intervals(m,scan[col]<scan.apex_local_pixel_envelope_high),
           ratio_min=float((scan[col]/scan.apex_limit).min()),ratio_median=float((scan[col]/scan.apex_limit).median()),
           ratio_max=float((scan[col]/scan.apex_limit).max()))
    best=scan.loc[scan.p_concordance.idxmin()]
    summary=dict(common_grid=[lo,hi,n],apex_min=minima.sort_values('apex_p').iloc[0].to_dict(),
       best_same_mass=best.to_dict(),n_both_below_005=int(((scan.apex_p<.05)&(scan.hps_p<.05)).sum()),
       n_both_below_01=int(((scan.apex_p<.1)&(scan.hps_p<.1)).sum()),
       n_both_below_005_pixel_low=int(((scan.apex_p_pixel_low<.05)&(scan.hps_p<.05)).sum()),
       min_bonferroni=float(scan.p_concordance_bonferroni.min()),competitive=competitive,
       dense_min_concordance=float(dense.p_concordance.min()),
       dense_n_both_below_01=int(((dense.value<.1)&(dense.hps_p_log_interpolated<.1)).sum()),
       sigma_ratio_range_MeV=[float((scan.sigma_hps_2021_MeV/scan.sigma_apex_if_MeV).min()),float((scan.sigma_hps_2021_MeV/scan.sigma_apex_if_MeV).max())],
       sigma_ratio_range_percent=[float((scan.sigma_hps_2021_MeV/scan.sigma_apex_if_percent_MeV).min()),float((scan.sigma_hps_2021_MeV/scan.sigma_apex_if_percent_MeV).max())])
    (B/'derived/summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
                         'axes.grid':True,'grid.alpha':.17,'legend.frameon':False})
    blue='#235a9f';teal='#008b85';orange='#c16a18';purple='#8a4d99'
    fig,axs=plt.subplots(2,1,figsize=(8.5,6.0),sharex=True,gridspec_kw={'height_ratios':[1.7,1]})
    axs[0].semilogy(A.mass_MeV,A.value,color='black',lw=1,label='APEX supplied 70% contour (CL unverified)')
    axs[0].fill_between(A.mass_MeV,A.pixel_low,A.pixel_high,color='black',alpha=.1,lw=0)
    axs[0].semilogy(scan.mass_MeV,scan.hps_limit,color=blue,lw=1.8,label='HPS released combined contour (90% CLs)')
    axs[0].semilogy(scan.mass_MeV,scan.hps_2021_full_observed_equivalent,color=teal,ls='--',lw=1.7,label='2021 100%: observed-equivalent scaling')
    axs[0].semilogy(scan.mass_MeV,scan.hps_combined_full_observed_equivalent,color=orange,ls=':',lw=1.7,label='Full-equivalent combination (2015/16/21)')
    axs[0].set_ylabel(r'Upper contour in $\epsilon^2$');axs[0].legend(loc='upper left',fontsize=9,ncol=1)
    axs[0].set_ylim(8e-8,7e-5)
    axs[1].semilogy(scan.mass_MeV,scan.hps_limit_over_apex,color=blue,label='Current combination')
    axs[1].semilogy(scan.mass_MeV,scan.hps_2021_full_observed_equivalent_over_apex,color=teal,ls='--',label='2021 100% equivalent')
    axs[1].axhline(1,color='black',lw=.8);axs[1].set_ylabel('HPS / APEX contour');axs[1].set_xlabel(r'Mass [MeV]')
    axs[1].legend(loc='upper left',fontsize=9);axs[1].set_xlim(154,271)
    for ax in axs:
        ax.axvline(180,color='grey',lw=.8,ls=':');ax.axvline(250,color='grey',lw=.8,ls=':')
        ax.axvspan(250,271,color='grey',alpha=.07)
    axs[0].text(216,4.7e-5,'HPS: 2021 only above 180 MeV',fontsize=8,color=blue)
    fig.tight_layout();savefig(fig,'limit_comparison')

    fig,axs=plt.subplots(2,1,figsize=(8.5,5.8),sharex=True)
    axs[0].semilogy(P.mass_MeV,P.value,color='black',lw=1,label='APEX digitized local p')
    axs[0].fill_between(P.mass_MeV,P.pixel_low,P.pixel_high,color='black',alpha=.10,lw=0)
    axs[0].semilogy(scan.mass_MeV,scan.hps_p,color=blue,lw=1.8,label='HPS released combined local p')
    axs[0].semilogy(scan.mass_MeV,scan.hps_2021_p,color=teal,ls='--',lw=1.3,label='HPS 2021 10% local p')
    for val,label in [(.05,'p = 0.05'),(norm.sf(2),'2 sigma (one sided)')]:
        axs[0].axhline(val,color='grey',lw=.8,ls=':');axs[0].text(271,val,label,ha='right',va='bottom',fontsize=8)
    axs[0].set_ylim(.015,1.2);axs[0].set_ylabel('Local p-value')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',fontsize=8,ncol=3,bbox_to_anchor=(.5,1.01))
    axs[1].plot(scan.mass_MeV,scan.p_concordance,color=purple,lw=1.8,label='Same-mass max(APEX p, HPS p)')
    axs[1].axhline(.05,color='grey',ls=':');axs[1].set_ylabel('Concordance score');axs[1].set_ylim(0,1)
    axs[1].set_xlabel('Mass [MeV]');axs[1].legend(fontsize=9,loc='lower left',bbox_to_anchor=(0,.07),frameon=True,facecolor='white',framealpha=.9);axs[1].set_xlim(154,271)
    for ax in axs:ax.axvspan(250,271,color='grey',alpha=.07)
    fig.tight_layout(rect=[0,0,1,.955]);savefig(fig,'pvalue_comparison')

    fig,axs=plt.subplots(1,2,figsize=(8.5,3.3))
    axs[0].plot(RES.mass_MeV,RES.total_axis_value,'s-',color='black',label='APEX total, native vertical values')
    axs[0].set(xlabel='Mass [MeV]',ylabel='Total-resolution axis value',ylim=(.65,1.1));axs[0].legend(fontsize=8)
    axs[1].plot(m,sigma(m),color=blue,label='HPS 2021 scaled width')
    axs[1].plot(m,sigma(m,'2016'),color=orange,ls=':',label='HPS 2016 width (ends at 180)',alpha=.9)
    # Do not visually extend 2016 beyond its released search domain.
    axs[1].lines[-1].set_data(m[m<=180],sigma(m[m<=180],'2016'))
    axs[1].plot(m,scan.sigma_apex_if_MeV,color='black',label='APEX if axis is sigma [MeV]')
    axs[1].plot(m,scan.sigma_apex_if_percent_MeV,color='grey',ls='--',label='APEX if axis is sigma/m [%]')
    axs[1].set(xlabel='Mass [MeV]',ylabel=r'Gaussian width $\sigma_m$ [MeV]');axs[1].legend(fontsize=8)
    fig.tight_layout();savefig(fig,'resolution_comparison')

    fig,axs=plt.subplots(2,2,figsize=(8.5,6.8))
    for ax,name,window in zip(axs.flat,['ten_160','ten_210','one_210','one_extra244'],[(154,174),(193,224),(188,223),(226,250)]):
        ss=scenarios[scenarios.scenario==name].iloc[0]
        ax.semilogy(A.mass_MeV,A.value,color='black',lw=1,label='APEX contour')
        for spec,col,ls,label in [('background_asimov',teal,':','Background Asimov'),('yield_asimov',orange,'--','Yield-scaled Asimov'),('matched_asimov',blue,'-','Target-matched Asimov')]:
            d=read(f'inputs/{name}_{spec}.csv');ax.semilogy(d.mass_MeV,d.epsilon2_90,color=col,ls=ls,lw=1.4,label=label)
        ax.plot(ss.mass_MeV,ss.injected_epsilon2,'*',ms=10,color=purple,label='Injected coupling')
        ax.set_xlim(*window);ax.set_ylim(8e-8,2e-5 if ss.lane=='ten' else 2e-4);ax.set_title(f"{'Native 10%' if ss.lane=='ten' else 'Historical 1%'} to 100%: {ss.mass_MeV:.0f} MeV",fontsize=10)
        ax.set_xlabel('Mass [MeV]');ax.set_ylabel(r'$\epsilon^2$')
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=9,bbox_to_anchor=(.5,1.015))
    fig.tight_layout(rect=[0,0,1,.93]);savefig(fig,'conditional_projection_comparison')

    fig,axs=plt.subplots(1,2,figsize=(8.5,3.3))
    for ax,proj in zip(axs,[False,True]):
        ax.semilogy(A.mass_MeV,A.value,color='black',lw=1,label='APEX contour')
        ax.semilogy(scan.mass_MeV,scan.hps_combined_full_observed_equivalent if proj else scan.hps_limit,color=blue,label='2015/16/21')
        ax.semilogy(scan.mass_MeV,scan.hps_four_full_equivalent if proj else scan.hps_four_current,color=orange,ls='--',label='Added 2019 sample')
        ax.set(xlim=(154,250),ylim=(8e-8,6e-5),xlabel='Mass [MeV]',ylabel=r'Upper contour in $\epsilon^2$',title='Full-equivalent display' if proj else 'Current samples')
        ax.legend(fontsize=8)
    fig.tight_layout();savefig(fig,'added_2019_comparison')
    print(json.dumps({k:summary[k] for k in ['common_grid','n_both_below_005','n_both_below_01','min_bonferroni','competitive']},indent=2))
    print('APEX minima:\n',minima.sort_values('apex_p').head(9)[['mass_MeV','apex_p','hps_p_interp','apex_limit']].to_string(index=False))
    print('HPS peaks:\n',peaks.to_string(index=False))
    print('Signal comparisons:\n',scenarios.to_string(index=False))

if __name__=='__main__':main()
