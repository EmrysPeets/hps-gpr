"""Original-window MC/Gaussian comparisons with fixed, paired backgrounds.

Native MC masses are the primary results. Interpolated standardized-residual
CDF mixtures are exploratory and subjected to leave-one-mass-out checks.
"""
from common import *
from scipy.special import ndtr
import time
YEAR='2021';SEED=61120260922;BOOTSTRAPS=32
MC={int(p.stem[1:]):dict(np.load(p)) for p in sorted((B/'histograms').glob('m[0-9][0-9][0-9].npz'))}
MASSES=np.array(sorted(MC));EDGES=DATA[YEAR]['edges'];START=time.monotonic()

def cumulative(m,points,counts=None):
    d=MC[m];h=d['sumw'] if counts is None else counts
    # The stored range omits a few high-mass outliers. Retain their measured
    # probability in support bookkeeping; fitted shapes condition on support.
    coverage=float(d['sumw'].sum()/json.loads(str(d['metadata']))['sumw'])
    return np.interp(points,d['edges_GeV'],np.r_[0.,np.cumsum(h)]/h.sum()*coverage,left=0.,right=coverage)

def distribution(mass,source=None,counts=None,omit=None):
    if source is not None:
        c=cumulative(source,EDGES,counts)
    else:
        grid=MASSES if omit is None else MASSES[MASSES!=omit]
        if mass<grid[0] or mass>grid[-1]:raise ValueError('No extrapolation')
        if mass in grid:c=cumulative(int(mass),EDGES)
        else:
            low=int(grid[grid<mass][-1]);high=int(grid[grid>mass][0]);t=(mass-low)/(high-low)
            z=(EDGES-mass/1000)/sigma(YEAR,mass)
            c=(1-t)*cumulative(low,low/1000+z*sigma(YEAR,low))+t*cumulative(high,high/1000+z*sigma(YEAR,high))
    raw=np.maximum(np.diff(c),0.);fraction=float(raw.sum())
    if fraction<=0:raise ValueError('No signal in analysis support')
    return raw/fraction,fraction

def branch(m):
    r=(105.6583745/m)**2
    return 1. if m<=2*105.6583745 else 1+np.sqrt(1-4*r)*(1+2*r)

def evaluate(mass,part,w,label,support_fraction=1.):
    scale=float(continuous_signal(YEAR,mass).sum());model=OneSignalProfile(part['b'],part['L'],w[part['mask']]*scale)
    row=model.limit(part['n'])
    # Enforce the same checks for the observed, held-out, and resampled fits.
    if not (row['ok'] and abs(row['cls']-.1)<2e-6 and row['max_score']<3e-5 and row['min_lambda']>0):
        raise RuntimeError(f'Likelihood endpoint failed for {mass}, {label}: {row}')
    row.update(year=YEAR,mass_MeV=mass,model=label,epsilon2_90=row['A90']*1e-8,display_epsilon2_90=row['A90']*1e-8*branch(mass),
               full_yield_90=row['A90']*scale,signal_fraction_in_fit=float(w[part['mask']].sum()),
               support_fraction=support_fraction,full_selected_yield_90=row['A90']*scale/support_fraction)
    return row

def main():
    assert list(MASSES)==list(range(40,261,20))
    shapes=[]
    for mass,d in MC.items():
        meta=json.loads(str(d['metadata']));h=d['sumw'];edges=d['edges_GeV'];cdf=np.r_[0.,np.cumsum(h)]/h.sum()
        quantiles=np.interp([.00135,.025,.158655,.5,.841345,.975,.99865],cdf,edges)*1000
        center=(edges[:-1]+edges[1:])/2;mean=float(h@center/h.sum())*1000
        sig=sigma(YEAR,mass)*1000;a=mass-2.25*sig;b=mass+2.25*sig
        core=float(cumulative(mass,np.array([b,a])/1000)@[1.,-1.])
        support=float(cumulative(mass,EDGES[[0,-1]])@[-1.,1.])
        shapes.append(dict(mass_MeV=mass,entries=meta['stats']['entries'],selected=meta['stats']['selected'],neffective=meta['neffective'],
          mean_MeV=mean,median_MeV=quantiles[3],median_bias_MeV=quantiles[3]-mass,
          width68_MeV=(quantiles[4]-quantiles[2])/2,nominal_sigma_MeV=sig,width68_over_nominal=(quantiles[4]-quantiles[2])/(2*sig),
          low68_MeV=quantiles[2],high68_MeV=quantiles[4],low95_MeV=quantiles[1],high95_MeV=quantiles[5],
          tail_fraction_2p25sigma=1-core,support_fraction=support,histogram_range_fraction=float(h.sum()/meta['sumw']),
          histogram_overflow=meta['stats']['overflow'],psum_below_2p8=meta['stats']['psum_below_2p8'],
          smear_nonunit=meta['stats']['smear_nonunit'],wrong_type=meta['stats']['wrong_type']))
    pd.DataFrame(shapes).to_csv(B/'derived/shape_summary.csv',index=False,float_format='%.17g')
    rows=[];loo=[];boot=[];contexts={};gaussian={};direct={}
    for mass in range(50,251):
        part=moving_context(YEAR,mass);contexts[mass]=part
        w=np.diff(ndtr((EDGES-mass/1000)/sigma(YEAR,mass)));w/=w.sum()
        base=evaluate(mass,part,w,'gaussian');rows.append(base);gaussian[mass]=base
        mc,fraction=distribution(mass);label='mc_direct' if mass in MC else 'mc_morph'
        result=evaluate(mass,part,mc,label,fraction);rows.append(result)
        if mass in MC:direct[mass]=result
        if mass%20==0:print('observed',mass,'elapsed',round(time.monotonic()-START,1),flush=True)
    comparison=[]
    for mass,r in direct.items():
        base=gaussian[mass]
        comparison.append(dict(year=YEAR,mass_MeV=mass,gaussian_limit=base['display_epsilon2_90'],mc_limit=r['display_epsilon2_90'],
            limit_ratio=r['epsilon2_90']/base['epsilon2_90'],gaussian_p0=base['p0_fixed_mass'],mc_p0=r['p0_fixed_mass'],
            gaussian_r=base['signed_r'],mc_r=r['signed_r'],delta_r=r['signed_r']-base['signed_r'],
            gaussian_Z=base['Z0'],mc_Z=r['Z0'],delta_p0=r['p0_fixed_mass']-base['p0_fixed_mass']))
        part=contexts[mass];w,fraction=distribution(mass,omit=mass);exact,_=distribution(mass,source=mass)
        fit=evaluate(mass,part,w,'mc_leave_one_out',fraction)
        ks=float(np.max(abs(np.cumsum(w)-np.cumsum(exact))));core_error=float((w-exact)[part['mask']].sum())
        loo.append(dict(mass_MeV=mass,conditional_support_cdf_distance=ks,core_fraction_difference=core_error,
                        limit_ratio_to_direct=fit['epsilon2_90']/r['epsilon2_90'],delta_r_to_direct=fit['signed_r']-r['signed_r'],
                        shape_check_pass=bool(ks<=.03 and abs(core_error)<=.02)))
        h=MC[mass]['sumw'];h2=MC[mass]['sumw2']
        if not np.allclose(h,h2,atol=1e-8):raise ValueError('Non-unit weights: exact Poisson-bin bootstrap not supported')
        for toy in range(BOOTSTRAPS):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,mass,toy]));counts=rng.poisson(h)
            wt,ft=distribution(mass,source=mass,counts=counts);rr=evaluate(mass,part,wt,'mc_bootstrap',ft)
            boot.append(dict(mass_MeV=mass,toy=toy,limit_ratio_to_gaussian=rr['epsilon2_90']/base['epsilon2_90'],
                             limit_ratio_to_central=rr['epsilon2_90']/r['epsilon2_90'],signed_r=rr['signed_r']))
    pd.DataFrame(rows).to_csv(B/'derived/scans.csv',index=False,float_format='%.17g')
    c=pd.DataFrame(comparison);c.to_csv(B/'derived/comparison.csv',index=False,float_format='%.17g')
    pd.DataFrame(loo).to_csv(B/'derived/leave_one_out.csv',index=False,float_format='%.17g')
    bo=pd.DataFrame(boot);bo.to_csv(B/'derived/mc_bootstrap.csv',index=False,float_format='%.17g')
    bs=[]
    for mass,q in bo.groupby('mass_MeV'):
        v=q.limit_ratio_to_gaussian.quantile([.16,.5,.84]);bs.append(dict(mass_MeV=mass,ratio_q16=v.iloc[0],ratio_median=v.iloc[1],ratio_q84=v.iloc[2],
          relative_limit_std=float(q.limit_ratio_to_central.std()),toys=len(q)))
    pd.DataFrame(bs).to_csv(B/'derived/mc_precision.csv',index=False,float_format='%.17g')
    ratio=c.limit_ratio
    agreement=dict(native_points=len(c),median_limit_ratio=float(ratio.median()),descriptive_q16=float(ratio.quantile(.16)),descriptive_q84=float(ratio.quantile(.84)),
      min_ratio=float(ratio.min()),min_ratio_mass_MeV=int(c.loc[ratio.idxmin(),'mass_MeV']),max_ratio=float(ratio.max()),max_ratio_mass_MeV=int(c.loc[ratio.idxmax(),'mass_MeV']),
      median_absolute_fractional_difference=float(abs(ratio-1).median()),rms_log_ratio=float(np.sqrt(np.mean(np.log(ratio)**2))),
      fractions_within={str(t):float(np.mean(abs(ratio-1)<=t)) for t in [.01,.05,.1]},max_abs_delta_r=float(abs(c.delta_r).max()),
      loo_all_pass=bool(all(r['shape_check_pass'] for r in loo)),loo_max_cdf_distance=max(r['conditional_support_cdf_distance'] for r in loo),
      bootstrap_toys_per_mass=BOOTSTRAPS,scope='Native MC coordinates only; descriptive correlated-curve agreement, not a goodness-of-fit probability.',
      dense_scope='Exploratory standardized-residual CDF interpolation between supplied MC points; not an independently validated dense response model.')
    write(B/'derived/agreement.json',agreement)
    f=pd.DataFrame(rows);write(B/'qa/fit_checks.json',dict(passed=bool(f.ok.all() and abs(f.cls-.1).max()<2e-6 and f.max_score.max()<3e-5),
        observed_fits=len(rows),bootstrap_fits=len(boot),leave_one_out_fits=len(loo),min_lambda=float(f.min_lambda.min()),
        max_score=float(f.max_score.max()),max_cls_error=float(abs(f.cls-.1).max()),
        all_fit_classes_checked_at_evaluation=True,seconds=time.monotonic()-START))
    print(json.dumps(agreement,indent=2))

if __name__=='__main__':main()
