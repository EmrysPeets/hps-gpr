"""MC-only core location and shifted fit/training windows; no shape translation."""
from run_mc_study import *
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

def template(m):
    if m in MC:return MC[m]['edges_GeV']*1000,MC[m]['sumw'].astype(float)
    a=np.load(B/f'histograms/morphed/m{m:03d}.npz')
    return a['edges_GeV']*1000,a['probability']*1e6

def locate(m,span=1.5,counts=None):
    edges,h=template(m);h=h if counts is None else counts;x=(edges[1:]+edges[:-1])/2
    sig=sigma(YEAR,m)*1000;dx=float(np.diff(edges)[0]);sm=gaussian_filter1d(h,.2*sig/dx)
    search=np.flatnonzero(abs(x-m)<=3*sig);peak=int(search[np.argmax(sm[search])]);mode=x[peak]
    keep=abs(x-mode)<=span*sig;y=h[keep];lo=edges[:-1][keep];hi=edges[1:][keep]
    # Coordinates in nominal sigma; fit bin-integrated Gaussian plus a
    # nonnegative affine pedestal. The fitted width does not change the mask.
    u=(x[keep]-mode)/sig;ul=(lo-mode)/sig;uh=(hi-mode)/sig;mix=(u-u.min())/(u.max()-u.min())
    scale=max(y.max(),1.);yn=y/scale
    def expected(p):
        area,mu,s,b0,b1=p
        return area*(ndtr((uh-mu)/s)-ndtr((ul-mu)/s))+b0*(1-mix)+b1*mix
    def residual(p):
        lam=np.maximum(expected(p),1e-15);term=lam-yn
        pos=yn>0;term[pos]+=yn[pos]*np.log(yn[pos]/lam[pos])
        return np.sign(lam-yn)*np.sqrt(np.maximum(2*term,0))
    start=[max(float(yn.sum())*.8,1.),0.,.65,max(float(yn[0])*.5,.0001),max(float(yn[-1])*.5,.0001)]
    fit=least_squares(residual,start,bounds=([0,-1,.25,0,0],[np.inf,1,2.5,np.inf,np.inf]),xtol=1e-11,ftol=1e-11,gtol=1e-10,max_nfev=400)
    p=fit.x;center=float(mode+p[1]*sig);model=expected(p)*scale
    active=bool(abs(p[1])>.999 or p[2]<.251 or p[2]>2.499)
    edge=bool(peak in (search[0],search[-1]));valid=bool(fit.success and not active and not edge)
    out=dict(mass_MeV=m,core_center_MeV=center,center_shift_MeV=center-m,nominal_sigma_MeV=sig,
      fitted_core_sigma_MeV=float(p[2]*sig),mode_MeV=float(mode),fit_halfwidth_nominal_sigma=span,
      core_fit_success=bool(fit.success),core_fit_bound_hit=active,mode_search_edge=edge,core_location_valid=valid,
      fit_deviance=float(np.sum(residual(p)**2)*scale),fit_bins=int(keep.sum()),core_definition='Local Gaussian plus nonnegative affine pedestal, integrated bins; nominal-width mask unchanged',
      input_kind='direct' if m in MC else 'morph',low_mass_interpolation_warning=m<100 and m not in MC)
    return out,dict(x=x[keep],observed=y,fit=model,gaussian=p[0]*(ndtr((uh-p[1])/p[2])-ndtr((ul-p[1])/p[2]))*scale)

def shifted_part(m,c):
    half=2.25*sigma(YEAR,m)
    return context(YEAR,[m],anchor=m,region=(c/1000-half,c/1000+half))

def main():
    centers=[];rows=[];comparisons=[];checks=[];native=[];sensitivity=[]
    old=pd.read_csv(B/'derived/scans.csv');old['framework']='pole_centered';rows.extend(old.to_dict('records'))
    for m in range(50,251):
        center,_=locate(m);centers.append(center)
        if not center['core_location_valid']:
            print('invalid core',m,center,flush=True);continue
        c=center['core_center_MeV'];part=shifted_part(m,c);w,f=distribution(m)
        mc=evaluate(m,part,w,'mc_direct' if m in MC else 'mc_morph',f)
        wg=np.diff(ndtr((EDGES-c/1000)/sigma(YEAR,m)));wg/=wg.sum()
        g=evaluate(m,part,wg,'gaussian')
        for r in (mc,g):
            r.update(framework='core_centered',core_center_MeV=c,window_low_MeV=part['lo']*1000,window_high_MeV=part['hi']*1000,fit_bins=int(part['mask'].sum()))
            rows.append(r)
        om=old[(old.mass_MeV==m)&(old.model!='gaussian')].iloc[0];og=old[(old.mass_MeV==m)&(old.model=='gaussian')].iloc[0]
        comparisons.append(dict(mass_MeV=m,core_center_MeV=c,center_shift_MeV=c-m,input_kind=center['input_kind'],
          old_mc_limit=om.display_epsilon2_90,new_mc_limit=mc['display_epsilon2_90'],old_gaussian_limit=og.display_epsilon2_90,new_gaussian_limit=g['display_epsilon2_90'],
          mc_shift_ratio=mc['display_epsilon2_90']/om.display_epsilon2_90,new_mc_to_gaussian_ratio=mc['display_epsilon2_90']/g['display_epsilon2_90'],old_mc_to_gaussian_ratio=om.display_epsilon2_90/og.display_epsilon2_90,
          old_mc_p0=om.p0_fixed_mass,new_mc_p0=mc['p0_fixed_mass'],old_gaussian_p0=og.p0_fixed_mass,new_gaussian_p0=g['p0_fixed_mass'],
          mc_delta_r=mc['signed_r']-om.signed_r,new_mc_vs_gaussian_delta_r=mc['signed_r']-g['signed_r'],
          old_mc_fit_fraction=om.signal_fraction_in_fit,new_mc_fit_fraction=mc['signal_fraction_in_fit']))
        assert np.array_equal(part['mask'],(DATA[YEAR]['x']>=part['lo'])&(DATA[YEAR]['x']<=part['hi']))
        if m in MC:
            n=dict(center)
            for span in [1.25,2.]:
                alt,_=locate(m,span);cc=alt['core_center_MeV'];rr=evaluate(m,shifted_part(m,cc),w,'center_definition_diagnostic',f)
                sensitivity.append(dict(mass_MeV=m,fit_span=span,core_center_MeV=cc,center_delta_MeV=cc-c,valid=alt['core_location_valid'],limit_ratio_to_default=rr['display_epsilon2_90']/mc['display_epsilon2_90'],p0=rr['p0_fixed_mass']))
            replicas=[]
            for toy in range(32):
                rng=np.random.default_rng(np.random.SeedSequence([92361,m,toy]));ct,_=locate(m,counts=rng.poisson(MC[m]['sumw']));assert ct['core_location_valid'];replicas.append(ct['core_center_MeV'])
            n.update(center_MC_bin_resample_std_MeV=float(np.std(replicas,ddof=1)),replicas=32);native.append(n)
    allc=pd.DataFrame(centers);scan=pd.DataFrame(rows);comp=pd.DataFrame(comparisons);ns=comp[comp.input_kind=='direct'];rat=ns.new_mc_to_gaussian_ratio
    for name,d in [('core_centers',allc),('core_scans',scan),('core_comparison',comp),('core_native_diagnostics',pd.DataFrame(native)),('core_center_sensitivity',pd.DataFrame(sensitivity))]:
        d.to_csv(B/f'derived/{name}.csv',index=False,float_format='%.17g')
    outdir=B/'histograms/core_centered_windows';outdir.mkdir(exist_ok=True)
    for m in range(60,241,20):
        c=float(allc[allc.mass_MeV==m].core_center_MeV.iloc[0]);w,f=distribution(m)
        np.savez_compressed(outdir/f'm{m:03d}.npz',analysis_edges_GeV=EDGES,probability=w,fit_mask=shifted_part(m,c)['mask'],core_center_MeV=c,nominal_sigma_MeV=sigma(YEAR,m)*1000)
    peaks={}
    for framework in ['pole_centered','core_centered']:
        for model in ['gaussian','mc_direct','mc_all']:
            q=scan[(scan.framework==framework)&((scan.model!='gaussian') if model=='mc_all' else (scan.model==model))]
            r=q.loc[q.p0_fixed_mass.idxmin()];peaks[framework+'_'+model]=dict(mass_MeV=float(r.mass_MeV),p0=float(r.p0_fixed_mass),Z=float(r.Z0),limit=float(r.display_epsilon2_90))
    summary=dict(native_points=len(ns),valid_centers=int(allc.core_location_valid.sum()),failed_centers=allc.loc[~allc.core_location_valid,'mass_MeV'].tolist(),
      native_shift_range_MeV=[float(ns.center_shift_MeV.min()),float(ns.center_shift_MeV.max())],median_new_MC_Gaussian_ratio=float(rat.median()),median_abs_new_MC_Gaussian_difference=float(abs(rat-1).median()),
      new_MC_Gaussian_ratio_range=[float(rat.min()),float(rat.max())],median_new_to_old_MC_ratio=float(ns.mc_shift_ratio.median()),new_to_old_MC_ratio_range=[float(ns.mc_shift_ratio.min()),float(ns.mc_shift_ratio.max())],peaks=peaks)
    write(B/'derived/core_agreement.json',summary)
    fresh=scan[scan.framework=='core_centered'];assert len(ns)==10 and fresh.ok.all()
    check=dict(passed=True,core_centers=len(allc),valid_centers=int(allc.core_location_valid.sum()),new_observed_fits=len(fresh),native_points=len(ns),MC_bin_resample_core_fits=320,center_definition_refits=20,
      maximum_cls_error=float(abs(fresh.cls-.1).max()),maximum_score=float(fresh.max_score.max()),minimum_poisson_mean=float(fresh.min_lambda.min()),
      observed_data_used_for_center=False,template_translated=False,width_rescaled=False,kernel_anchor='generated mass; archived states unchanged',scope='Conditional selected-MC response and shifted-window comparison; local asymptotic probabilities only')
    assert check['maximum_cls_error']<2e-6 and check['maximum_score']<3e-5 and check['minimum_poisson_mean']>0
    write(B/'qa/core_centering.json',check);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
