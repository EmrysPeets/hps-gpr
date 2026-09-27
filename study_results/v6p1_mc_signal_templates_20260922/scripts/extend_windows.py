"""User-requested window diagnostics and MC-learned interpolated templates.

No fit to observed data enters shape morphing. Archived kernel coordinates
remain fixed; predictions and covariance are recalculated for each new mask.
"""
from run_mc_study import *
import uproot
WIDTHS=[2.25,2.4,2.5,2.6]

def main():
    rows=[];comparisons=[];agreements=[];loos=[];precisions=[];checks=[]
    base=pd.read_csv(B/'derived/scans.csv');base['blind_sigma']=2.25;rows.extend(base.to_dict('records'))
    for file,target in [('comparison.csv',comparisons),('leave_one_out.csv',loos),('mc_precision.csv',precisions)]:
        q=pd.read_csv(B/'derived'/file);q['blind_sigma']=2.25;target.extend(q.to_dict('records'))
    for width in WIDTHS[1:]:
        direct={};gauss={};contexts={}
        for mass in range(50,251):
            part=context(YEAR,[mass],padding=width,anchor=mass);contexts[mass]=part
            wg=np.diff(ndtr((EDGES-mass/1000)/sigma(YEAR,mass)));wg/=wg.sum()
            g=evaluate(mass,part,wg,'gaussian');g['blind_sigma']=width;rows.append(g);gauss[mass]=g
            w,f=distribution(mass);r=evaluate(mass,part,w,'mc_direct' if mass in MC else 'mc_morph',f)
            r['blind_sigma']=width;rows.append(r)
            if mass in MC:direct[mass]=r
        for mass,r in direct.items():
            g=gauss[mass];part=contexts[mass]
            comparisons.append(dict(blind_sigma=width,year=YEAR,mass_MeV=mass,gaussian_limit=g['display_epsilon2_90'],mc_limit=r['display_epsilon2_90'],limit_ratio=r['epsilon2_90']/g['epsilon2_90'],
             gaussian_p0=g['p0_fixed_mass'],mc_p0=r['p0_fixed_mass'],gaussian_r=g['signed_r'],mc_r=r['signed_r'],delta_r=r['signed_r']-g['signed_r'],gaussian_Z=g['Z0'],mc_Z=r['Z0'],delta_p0=r['p0_fixed_mass']-g['p0_fixed_mass']))
            w,f=distribution(mass,omit=mass);exact,_=distribution(mass,source=mass);rr=evaluate(mass,part,w,'mc_leave_one_out',f)
            ks=float(np.max(abs(np.cumsum(w)-np.cumsum(exact))));ce=float((w-exact)[part['mask']].sum())
            loos.append(dict(blind_sigma=width,mass_MeV=mass,conditional_support_cdf_distance=ks,core_fraction_difference=ce,limit_ratio_to_direct=rr['epsilon2_90']/r['epsilon2_90'],delta_r_to_direct=rr['signed_r']-r['signed_r'],shape_check_pass=ks<=.03 and abs(ce)<=.02))
            ratios=[]
            for toy in range(BOOTSTRAPS):
                rng=np.random.default_rng(np.random.SeedSequence([SEED,mass,toy]));counts=rng.poisson(MC[mass]['sumw'])
                wt,ft=distribution(mass,source=mass,counts=counts);rr=evaluate(mass,part,wt,'mc_bootstrap',ft);ratios.append(rr['epsilon2_90']/g['epsilon2_90'])
            v=np.quantile(ratios,[.16,.5,.84]);precisions.append(dict(blind_sigma=width,mass_MeV=mass,ratio_q16=v[0],ratio_median=v[1],ratio_q84=v[2],relative_limit_std=float(np.std(np.array(ratios)/(r['epsilon2_90']/g['epsilon2_90']),ddof=1)),toys=BOOTSTRAPS))
        print('completed width',width,flush=True)
    f=pd.DataFrame(rows);c=pd.DataFrame(comparisons)
    for width,q in c.groupby('blind_sigma'):
        r=q.limit_ratio;sc=f[(f.blind_sigma==width)&(f.model=='mc_direct')];peak=sc.loc[sc.p0_fixed_mass.idxmin()]
        agreements.append(dict(blind_sigma=width,native_points=len(q),median_limit_ratio=r.median(),min_ratio=r.min(),max_ratio=r.max(),median_absolute_fractional_difference=abs(r-1).median(),rms_log_ratio=np.sqrt(np.mean(np.log(r)**2)),within_1percent=np.mean(abs(r-1)<=.01),within_5percent=np.mean(abs(r-1)<=.05),within_10percent=np.mean(abs(r-1)<=.1),max_abs_delta_r=abs(q.delta_r).max(),min_direct_p0=peak.p0_fixed_mass,min_direct_p0_mass_MeV=peak.mass_MeV))
    for file,data in [('window_scans.csv',f),('window_comparison.csv',c),('window_agreement.csv',pd.DataFrame(agreements)),('window_loo.csv',pd.DataFrame(loos)),('window_mc_precision.csv',pd.DataFrame(precisions))]:
        data.to_csv(B/'derived'/file,index=False,float_format='%.17g')
    # Save every intervening 1-MeV hypothesis, not just the displayed midpoints.
    dest=B/'histograms/morphed';dest.mkdir(exist_ok=True);edges=MC[40]['edges_GeV'];morphrows=[]
    with uproot.recreate(dest/'morphed_signal_templates.root') as root:
        for mass in range(50,251):
            if mass in MC:continue
            low=int(MASSES[MASSES<mass][-1]);high=int(MASSES[MASSES>mass][0]);t=(mass-low)/(high-low)
            z=(edges-mass/1000)/sigma(YEAR,mass)
            lo=np.diff(cumulative(low,low/1000+z*sigma(YEAR,low)));hi=np.diff(cumulative(high,high/1000+z*sigma(YEAR,high)))
            raw=(1-t)*lo+t*hi;p=raw/raw.sum();ap,support=distribution(mass)
            meta=dict(mass_MeV=mass,lower_mass_MeV=low,upper_mass_MeV=high,upper_mixture_weight=t,method='Linear CDF mixture of MC shapes transported in nominal-resolution residual coordinates',learned_from='Selected reconstructed MC histograms only; no observed-data shape fitting',display_support_GeV=[0,.4],display_range_fraction=float(raw.sum()),analysis_support_fraction=support,exploratory=True,low_mass_interpolation_warning=mass<100,synthetic_MC_event_count=None)
            np.savez_compressed(dest/f'm{mass:03d}.npz',edges_GeV=edges,probability=p,lower_probability=lo/lo.sum(),upper_probability=hi/hi.sum(),analysis_edges_GeV=EDGES,analysis_probability=ap,metadata=json.dumps(meta))
            root[f'm{mass:03d}_unit_probability']=(p,edges)
            assert np.min(p)>=0 and abs(p.sum()-1)<1e-12 and np.min(ap)>=0 and abs(ap.sum()-1)<1e-12
            # Independent explicit source-bin transport and overlap rebinning.
            independent=np.zeros(len(EDGES)-1)
            for source,weight in [(low,1-t),(high,t)]:
                mapped=mass/1000+(MC[source]['edges_GeV']-source/1000)*sigma(YEAR,mass)/sigma(YEAR,source)
                overlap=np.maximum(0,np.minimum(EDGES[1:,None],mapped[None,1:])-np.maximum(EDGES[:-1,None],mapped[None,:-1]))
                total=json.loads(str(MC[source]['metadata']))['sumw']
                independent+=weight*((overlap/np.diff(mapped))@MC[source]['sumw'])/total
            independent/=independent.sum();err=float(np.max(abs(independent-ap)));assert err<2e-12
            morphrows.append(dict(mass_MeV=mass,lower_mass_MeV=low,upper_mass_MeV=high,upper_mixture_weight=t,display_range_fraction=raw.sum(),analysis_support_fraction=support,independent_max_bin_error=err,shown_in_gallery=mass%20==10))
    pd.DataFrame(morphrows).to_csv(B/'derived/morphed_template_inventory.csv',index=False,float_format='%.17g')
    assert len(f)==1608 and len(c)==40 and f.ok.all() and (f.groupby(['blind_sigma','mass_MeV']).size()==2).all()
    assert len(morphrows)==191 and sum(r['shown_in_gallery'] for r in morphrows)==11
    assert abs(f.cls-.1).max()<2e-6 and f.max_score.max()<3e-5 and f.min_lambda.min()>0
    binrows=[];x=DATA[YEAR]['x']
    for mass in range(50,251):
        previous=0
        for width in WIDTHS:
            lo=mass/1000-width*sigma(YEAR,mass);hi=mass/1000+width*sigma(YEAR,mass)
            count=int(((x>=lo)&(x<=hi)).sum());left=int((x<lo).sum());right=int((x>hi).sum())
            assert count>=previous and min(left,right)>=3
            binrows.append(dict(mass_MeV=mass,blind_sigma=width,fit_bins=count,left_training_bins=left,right_training_bins=right));previous=count
    pd.DataFrame(binrows).to_csv(B/'derived/window_bin_counts.csv',index=False)
    write(B/'qa/window_extension.json',dict(passed=True,blind_halfwidths_sigma=WIDTHS,observed_fits=1608,new_observed_fits=1206,new_bootstrap_fits=960,new_leave_one_out_fits=30,all_fit_classes_endpoint_checked=True,morphed_templates=191,midpoint_gallery_templates=11,independent_morph_rebin_max_error=max(r['independent_max_bin_error'] for r in morphrows),scope='Fixed archived kernel coordinates, recomputed masked GP predictions; conditional shape/window diagnostics.'))
    print(pd.DataFrame(agreements).to_string(index=False))
if __name__=='__main__':main()
