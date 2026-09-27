"""MC-only analytic location, shared residual shape, and asymmetric-mask tests."""
from core_centering import *
NATIVE=np.arange(60,241,20)
CORE=pd.read_csv(B/'derived/core_native_diagnostics.csv').set_index('mass_MeV')
WINDOWS={'symmetric':(2.25,2.25),'left_redistributed':(2.5,2.),'right_redistributed':(2.,2.5),'left_expanded':(2.75,2.25),'right_expanded':(2.25,2.75)}

def design(m,kind):
    m=np.atleast_1d(m).astype(float);x=(m-150)/100
    if kind=='proportional':return (m/100)[:,None]
    if kind=='affine':return np.column_stack([x*0+1,x])
    if kind=='quadratic':return np.column_stack([x*0+1,x,x*x])
    if kind=='logarithmic':return np.column_stack([x*0+1,np.log(m/150)])
    raise ValueError(kind)

def fitlaw(m,y,kind):return np.linalg.lstsq(design(m,kind),y,rcond=None)[0]

def shared_cdf(u,excluded=None):
    values=[]
    for m in NATIVE:
        if m==excluded:continue
        r=CORE.loc[m];edges=MC[m]['edges_GeV']*1000;h=MC[m]['sumw']
        values.append(np.interp(r.core_center_MeV+r.fitted_core_sigma_MeV*np.asarray(u),edges,np.r_[0,np.cumsum(h)]/h.sum(),left=0,right=1))
    return np.mean(values,axis=0)

def shared_probability(c,width,exclude=None):
    raw=np.maximum(np.diff(shared_cdf((EDGES*1000-c)/width,exclude)),0)
    return raw/raw.sum(),float(raw.sum())

def main():
    shift=CORE.loc[NATIVE].center_shift_MeV.to_numpy();models=[];predictions=[]
    for kind in ['proportional','affine','quadratic','logarithmic']:
        beta=fitlaw(NATIVE,shift,kind);pred=design(NATIVE,kind)@beta;loo=[]
        for i,m in enumerate(NATIVE):
            keep=NATIVE!=m;loo.append(float((design([m],kind)@fitlaw(NATIVE[keep],shift[keep],kind))[0]))
        err=np.array(loo)-shift
        models.append(dict(model=kind,parameters=len(beta),coefficients=beta.tolist(),train_RMS_MeV=float(np.sqrt(np.mean((pred-shift)**2))),LOO_RMS_MeV=float(np.sqrt(np.mean(err**2))),LOO_max_abs_MeV=float(abs(err).max()),LOO_RMS_nominal_sigma=float(np.sqrt(np.mean((err/CORE.loc[NATIVE].nominal_sigma_MeV)**2)))))
        for m,y,p,l in zip(NATIVE,shift,pred,loo):predictions.append(dict(model=kind,mass_MeV=int(m),measured_shift_MeV=y,fitted_shift_MeV=p,heldout_shift_MeV=l))
    best=min(models,key=lambda r:r['LOO_RMS_MeV']);kind=best['model']
    widthcoef=fitlaw(NATIVE,np.log(CORE.loc[NATIVE].fitted_core_sigma_MeV),'quadratic')
    write(B/'derived/analytic_shift_models.json',dict(models=models,selected_by_MC_only_LOO=kind,coordinate='m in MeV; x=(m-150)/100; log uses ln(m/150)',width_model=dict(form='ln(sigma_core/MeV)=a+b*x+c*x^2',coefficients=widthcoef.tolist()),valid_domain_MeV=[60,240],scope='Descriptive MC-only model comparison; LOO model selection is not independent validation; no extrapolation.'))
    pd.DataFrame(predictions).to_csv(B/'derived/analytic_shift_predictions.csv',index=False,float_format='%.17g')
    # Overlays keep all source probability. The core-only view is conditioned
    # separately and is never substituted for the full signal in a likelihood.
    u=np.linspace(-180,240,21001);overlays={};shape=[];heldout=[];asym=[]
    old=pd.read_csv(B/'derived/core_scans.csv');old=old[(old.framework=='core_centered')&(old.model!='gaussian')].set_index('mass_MeV')
    for m in NATIVE:
        r=CORE.loc[m];c=r.core_center_MeV;wcore=r.fitted_core_sigma_MeV;sig=r.nominal_sigma_MeV
        edges=MC[m]['edges_GeV']*1000;h=MC[m]['sumw'];cdf=np.r_[0,np.cumsum(h)]/h.sum()
        F=lambda v:np.interp(v,edges,cdf,left=0,right=1)
        fu=F(c+wcore*u);overlays[str(m)]=np.diff(fu)
        uc=np.linspace(-2,2,401);direct_core=F(c+wcore*uc);direct_core=(direct_core-direct_core[0])/(direct_core[-1]-direct_core[0])
        pool=shared_cdf(uc,m);pool=(pool-pool[0])/(pool[-1]-pool[0]);kscore=float(abs(pool-direct_core).max())
        w,f=distribution(m);part=shifted_part(m,c);base=evaluate(m,part,w,'mc_direct',f)
        wp,fp=shared_probability(c,wcore,m);rr=evaluate(m,part,wp,'shared_shape_LOO',fp)
        wm,fm=distribution(m,omit=m);rm=evaluate(m,part,wm,'existing_morph_LOO',fm)
        keep=NATIVE!=m;predc=m+float((design([m],kind)@fitlaw(NATIVE[keep],shift[keep],kind))[0]);predwidth=float(np.exp((design([m],'quadratic')@fitlaw(NATIVE[keep],np.log(CORE.loc[NATIVE[keep]].fitted_core_sigma_MeV),'quadratic'))[0]))
        we,fe=shared_probability(predc,predwidth,m);re=evaluate(m,part,we,'shared_shape_full_LOO',fe)
        left=float(F(c-2.25*sig));right=float(1-F(c+2.25*sig));inside_left=float(F(c)-F(c-2.25*sig));inside_right=float(F(c+2.25*sig)-F(c))
        shape.append(dict(mass_MeV=int(m),core_center_MeV=c,core_width_MeV=wcore,core_fraction_2core_sigma=float(F(c+2*wcore)-F(c-2*wcore)),left_tail_2p25nominal=left,right_tail_2p25nominal=right,inside_left=inside_left,inside_right=inside_right,core_conditioned_LOO_CDF_distance=kscore))
        heldout.append(dict(mass_MeV=int(m),direct_p0=base['p0_fixed_mass'],shared_p0=rr['p0_fixed_mass'],morph_p0=rm['p0_fixed_mass'],predictive_p0=re['p0_fixed_mass'],direct_epsilon2_90=base['display_epsilon2_90'],shared_epsilon2_90=rr['display_epsilon2_90'],morph_epsilon2_90=rm['display_epsilon2_90'],predictive_epsilon2_90=re['display_epsilon2_90'],shared_full_CDF_distance=float(abs(np.cumsum(wp)-np.cumsum(w)).max()),shared_core_CDF_distance=kscore,shared_UL_ratio=rr['epsilon2_90']/base['epsilon2_90'],shared_delta_r=rr['signed_r']-base['signed_r'],morph_CDF_distance=float(abs(np.cumsum(wm)-np.cumsum(w)).max()),morph_UL_ratio=rm['epsilon2_90']/base['epsilon2_90'],end_to_end_UL_ratio=re['epsilon2_90']/base['epsilon2_90'],end_to_end_CDF_distance=float(abs(np.cumsum(we)-np.cumsum(w)).max()),heldout_center_error_MeV=predc-c,heldout_width_ratio=predwidth/wcore))
        for name,(leftw,rightw) in WINDOWS.items():
            region=((c-leftw*sig)/1000,(c+rightw*sig)/1000);p=context(YEAR,[m],anchor=m,region=region)
            wg=np.diff(ndtr((EDGES-c/1000)/sigma(YEAR,m)));wg/=wg.sum()
            for model,prob,frac in [('mc',w,f),('gaussian',wg,1.)]:
                z=evaluate(m,p,prob,model,frac);z.update(window=name,left_sigma=leftw,right_sigma=rightw,core_center_MeV=c,fit_bins=int(p['mask'].sum()),left_MC_tail=float(F(region[0]*1000)),right_MC_tail=float(1-F(region[1]*1000)),mc_UL_ratio_to_symmetric=z['epsilon2_90']/base['epsilon2_90'] if model=='mc' else np.nan,mc_delta_r=z['signed_r']-base['signed_r'] if model=='mc' else np.nan);asym.append(z)
        assert abs(base['epsilon2_90']/old.loc[m].epsilon2_90-1)<2e-7
    pd.DataFrame(shape).to_csv(B/'derived/shared_shape_metrics.csv',index=False,float_format='%.17g');pd.DataFrame(heldout).to_csv(B/'derived/shared_shape_heldout.csv',index=False,float_format='%.17g');pd.DataFrame(asym).to_csv(B/'derived/asymmetric_scans.csv',index=False,float_format='%.17g')
    pooled=np.mean(list(overlays.values()),axis=0);assert np.all(pooled>=0) and abs(pooled.sum()-1)<1e-12
    dest=B/'histograms/shared_shape';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'standardized_shape.npz',u_edges=u,probability=pooled,**{f'm{k}':v for k,v in overlays.items()})
    # Analytic location and width plus one common residual distribution can be
    # instantiated independently of neighboring-mass morphing, within training range.
    for m in range(60,241):
        c=m+float((design([m],kind)@np.array(best['coefficients']))[0]);width=float(np.exp((design([m],'quadratic')@widthcoef)[0]));w,f=shared_probability(c,width)
        np.savez_compressed(dest/f'm{m:03d}.npz',edges_GeV=EDGES,probability=w,core_center_MeV=c,core_width_MeV=width,support_fraction=f,exploratory=True)
        assert w.min()>=0 and abs(w.sum()-1)<1e-12
    h=pd.DataFrame(heldout);ar=pd.DataFrame(asym);sm=pd.DataFrame(shape)
    summary=dict(center_best_model=kind,center_LOO_RMS_MeV=best['LOO_RMS_MeV'],core_LOO_CDF_range=[float(h.shared_core_CDF_distance.min()),float(h.shared_core_CDF_distance.max())],full_LOO_CDF_range=[float(h.shared_full_CDF_distance.min()),float(h.shared_full_CDF_distance.max())],shared_UL_ratio_range=[float(h.shared_UL_ratio.min()),float(h.shared_UL_ratio.max())],end_to_end_UL_ratio_range=[float(h.end_to_end_UL_ratio.min()),float(h.end_to_end_UL_ratio.max())],left_tail_dominant_masses=sm.loc[sm.left_tail_2p25nominal>sm.right_tail_2p25nominal,'mass_MeV'].tolist(),asymmetric_windows=WINDOWS,conditional_common_shape=True)
    write(B/'derived/analytic_shape_summary.json',summary)
    assert len(ar)==100 and ar.ok.all() and abs(ar.cls-.1).max()<2e-6
    write(B/'qa/analytic_shapes.json',dict(passed=True,native_mass_points=10,analytic_models=4,center_LOO_fits=40,shared_shape_heldout_tests=10,asymmetric_observed_fits=100,generated_shared_templates=181,source_shapes_in_each_LOO_pool=9,maximum_cls_error=float(abs(ar.cls-.1).max()),maximum_score=float(ar.max_score.max()),minimum_expectation=float(ar.min_lambda.min()),observed_data_used_for_design=False,scope='Conditional empirical MC diagnostics; shape-only LOO conditions on known center/width; end-to-end location/width predicted with heldout mass removed, using same evaluation mask.'))
    print(json.dumps(summary,indent=2));print(json.dumps(models,indent=2))
if __name__=='__main__':main()
