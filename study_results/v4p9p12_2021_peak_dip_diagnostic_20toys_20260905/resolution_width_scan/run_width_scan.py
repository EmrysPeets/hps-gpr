#!/usr/bin/env python3
"""Scan five signal widths at fixed per-mass backgrounds; no toy generation."""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
from contextlib import contextmanager
import hashlib
import json
import math
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
sys.path.insert(0, str(PILOT))
import run_diagnostic as pilot
import hps_gpr.statistics as likelihood
from piecewise_cached_solver import CachedPiecewiseBoundedLimit, _reconcile_feasible_profile_candidates
from scipy.signal import find_peaks

np, pd, plt = pilot.np, pilot.pd, pilot.plt
production = pilot.production
MASSES = np.arange(50,251)
SCALES = (0.8,0.9,1.0,1.1,1.2)
OUTPUT, FIGURES = HERE / "derived", HERE / "figures"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def tighter_profile_optimization():
    """Retry only numerical failures, without altering frozen runtime files."""
    original=likelihood.minimize
    def strict_minimize(*args,**kwargs):
        options=dict(kwargs.get('options',{}))
        options.update(ftol=1e-15,gtol=1e-8,maxiter=2000,maxls=50)
        kwargs['options']=options
        return original(*args,**kwargs)
    likelihood.minimize=strict_minimize
    try:
        yield
    finally:
        likelihood.minimize=original


def main():
    started = time.monotonic()
    OUTPUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    protected_paths = list((PILOT/"derived").glob("*"))+list((PILOT/"figures").glob("*"))
    protected_paths += list((PILOT/"reverse_injection").rglob("*"))
    protected_paths += [PILOT/"run_diagnostic.py",PILOT/"PROTOCOL.md"]
    protected = {str(p):sha(p) for p in protected_paths if p.is_file()}
    sources = {
        "card": production.DEFAULT_CARD, "states": production.DEFAULT_STATES,
        "spectrum": pilot.PARENT/"derived/all_three_peak_extraction_plot_data.csv",
        "curves": pilot.PARENT/"derived/final_dataset_result_curves.csv",
        "script": Path(__file__), "protocol": HERE/"PROTOCOL.md",
        "pilot_script": PILOT/"run_diagnostic.py",
    }
    provenance = {k:{"path":str(p),"sha256":sha(p)} for k,p in sources.items()}
    cfg = production.load_config(sources["card"])
    production.validate_card(cfg)
    frame = pd.read_csv(sources["spectrum"])
    frame = frame[frame.dataset.astype(str)=="2021"].sort_values("bin_center_GeV")
    x = frame.bin_center_GeV.to_numpy(float)
    y = frame.observed_events.to_numpy(float)
    widths = frame.bin_width_MeV.to_numpy(float)/1000
    edges = np.r_[x-widths/2,x[-1]+widths[-1]/2]
    assert np.all(y>0) and np.array_equal(y,np.rint(y))
    assert 0.035 < edges[0] < 0.037 and 0.299 < edges[-1] < 0.301
    states = production.state_map(pd.read_csv(sources["states"]))
    saved = pd.read_csv(sources["curves"])
    saved = saved[(saved.scope_key=="individual_2021_10pct") & saved.mass_MeV.isin(MASSES)].set_index("mass_MeV")
    assert len(saved)==len(MASSES)
    sigma = lambda m:float(np.polynomial.polynomial.polyval(m,cfg.sigma_coeffs_2021))
    records=[]
    limit_diagnostics=[]
    failed_limits=[]
    retried_limits=[]
    print("Scanning 201 masses x 5 template widths, including observed 90% CLs limits; no new toys...",flush=True)
    for index,mass in enumerate(MASSES):
        m=mass/1000; sig=sigma(m)
        mask=(x>=m-2.25*sig)&(x<=m+2.25*sig)
        state=states['2021',int(mass)]
        model=pilot.fit_gpr(x[~mask],y[~mask],cfg,restarts=0,
                kernel=pilot.make_fixed_kernel(state['const_opt'],state['ls_opt']),optimize=False)
        mean,rawcov=pilot.predict_counts_from_log_gpr(model,x[mask],cfg)
        cov,condition=production.condition_covariance_block(rawcov,mean)
        nominal,_=pilot.build_window_template_from_full(edges,mask,m,sig,config=cfg)
        nominal=np.asarray(nominal,float); nominal/=nominal.sum()
        null=pilot.profiled_gaussian_fixed_poi_nll(y[mask],mean,cov,nominal,A_fixed=0)
        if not null['success']:
            raise RuntimeError(f"Null fit failed at {mass}")
        reference=json.loads(saved.loc[mass,'limit_profile_status'])['observed']['base']
        rf=reference['fit_unbounded']
        reference_r=float(np.sign(rf['A_hat'])*np.sqrt(max(2*(reference['null']['nll']-rf['nll']),0)))
        k_factor=float(saved.loc[mass,'signal_yield_per_eps2_total'])
        reference_limit=float(saved.loc[mass,'eps2_90'])
        assert np.isfinite(k_factor) and k_factor>0 and reference_limit>0
        for scale in SCALES:
            w,_=pilot.build_window_template_from_full(edges,mask,m,sig*scale,config=cfg)
            w=np.asarray(w,float); fraction=float(w.sum());w/=fraction
            signed=pilot.fit_A_profiled_gaussian_details(y[mask],mean,cov,w,allow_negative=True)
            bounded=pilot.fit_A_profiled_gaussian_details(y[mask],mean,cov,w,allow_negative=False)
            base,nfb,nsf=_reconcile_feasible_profile_candidates({'fit_unbounded':signed,'fit_bounded':bounded,'null':null})
            fit=base['fit_unbounded']; q=2*(float(null['nll'])-float(fit['nll']))
            if q < -1e-6 or not np.isfinite(q):
                raise RuntimeError(f"Invalid likelihood difference at {mass}, scale {scale}: {q}")
            r=float(np.sign(fit['A_hat'])*np.sqrt(max(q,0)))
            if scale==1 and abs(r-reference_r)>.02:
                raise RuntimeError(f"Nominal closure failed at {mass}: {r} vs {reference_r}")
            s_unit=k_factor*fraction*w
            numerical_method='production'
            initial_error=''
            def solve_limit():
                solver=CachedPiecewiseBoundedLimit(mean,cov,s_unit,alpha=float(cfg.cls_alpha),
                                                   combined_mode=str(cfg.combined_mode))
                result=solver.limit(y[mask])
                if not result.optimizer_ok or not np.isfinite(result.eps2_90) or result.eps2_90<=0:
                    raise RuntimeError("Invalid upper limit")
                return result
            try:
                limit=solve_limit()
            except RuntimeError as exc:
                initial_error=str(exc)
                numerical_method='tighter_optimizer_retry'
                retry={'mass_MeV':int(mass),'width_scale':scale,'initial_error':initial_error}
                print(f"Numerical retry: {mass} MeV, width {scale}: {exc}",flush=True)
                try:
                    with tighter_profile_optimization():
                        limit=solve_limit()
                    retry['retry_accepted']=True
                except RuntimeError as retry_exc:
                    retry.update(retry_accepted=False,retry_error=str(retry_exc))
                    failed_limits.append(retry)
                    limit=SimpleNamespace(eps2_90=float('nan'),optimizer_ok=False,cls_at_limit=float('nan'),
                                          convergence_reason='rejected_numerical_gate')
                retried_limits.append(retry)
                print(f"Retry accepted: {retry['retry_accepted']}",flush=True)
            diagnostic={'mass_MeV':int(mass),'width_scale':scale,'numerical_method':numerical_method,
                        'initial_error':initial_error}
            diagnostic.update(asdict(limit) if limit.optimizer_ok else vars(limit))
            limit_diagnostics.append(diagnostic)
            if scale==1 and limit.optimizer_ok:
                if abs(limit.eps2_90/reference_limit-1)>5e-4:
                    raise RuntimeError(f"Nominal upper-limit closure failed at {mass}")
                if not np.isclose(s_unit.sum(),saved.loc[mass,'signal_yield_per_eps2_fitted_window'],rtol=1e-10):
                    raise RuntimeError(f"Nominal signal normalization closure failed at {mass}")
            records.append({'mass_MeV':int(mass),'width_scale':scale,'sigma_template_MeV':sig*scale*1000,
                'n_fixed_fit_bins':int(mask.sum()),'template_fraction_in_fixed_window':fraction,
                'signed_r':r,'excess_Z_reference':max(r,0),'deficit_magnitude_reference':max(-r,0),
                'p0_asymptotic_fixed_width_reference':.5*math.erfc(max(r,0)/np.sqrt(2)),
                'A_hat_window':float(fit['A_hat']),'sigma_A_window':float(fit['sigma_A']),
                'raw_signed_twice_delta_nll':2*(float(null['nll'])-float(signed['nll'])),
                'signed_fallback_source':fit['fallback_source'],'signed_fallback_nll_improvement':fit['fallback_nll_improvement'],
                'bounded_feasible_fallbacks':nfb,'signed_feasible_fallbacks':nsf,
                'covariance_diagonal_load_relative':condition['selected_diagonal_load_relative'],
                'nominal_saved_r':reference_r,'post_selection_width_scan':True,
                'eps2_90':float(limit.eps2_90),'epsilon_90':float(np.sqrt(limit.eps2_90)),
                'A90_full_template_events':float(k_factor*limit.eps2_90),
                'A90_fitted_window_events':float(k_factor*fraction*limit.eps2_90),
                'signal_yield_per_eps2_total':k_factor,
                'signal_yield_per_eps2_fitted_window':float(s_unit.sum()),
                'nominal_saved_eps2_90':reference_limit,
                'cls_at_limit':float(limit.cls_at_limit),'limit_optimizer_ok':bool(limit.optimizer_ok),
                'limit_convergence_reason':limit.convergence_reason,
                'limit_numerical_method':numerical_method,'limit_initial_error':initial_error})
        if (index+1)%40==0 or index+1==len(MASSES):
            print(f"Finished {index+1}/201 masses ({5*(index+1)}/1005 width fits), elapsed {time.monotonic()-started:.1f}s",flush=True)
    data=pd.DataFrame(records)
    assert len(data)==1005 and np.isfinite(data.signed_r).all()
    nominal_limits=data[data.width_scale==1].set_index('mass_MeV').eps2_90
    data['limit_ratio_to_nominal']=data.eps2_90/data.mass_MeV.map(nominal_limits)
    data.to_csv(OUTPUT/'width_scan_all_points.csv',index=False)
    limit_columns=['mass_MeV','width_scale','sigma_template_MeV','eps2_90','epsilon_90',
        'A90_full_template_events','A90_fitted_window_events','limit_ratio_to_nominal',
        'template_fraction_in_fixed_window','signal_yield_per_eps2_total',
        'signal_yield_per_eps2_fitted_window','nominal_saved_eps2_90','cls_at_limit',
        'limit_optimizer_ok','limit_convergence_reason','limit_numerical_method','limit_initial_error']
    data[limit_columns].to_csv(OUTPUT/'width_scan_upper_limits.csv',index=False)
    (OUTPUT/'limit_solver_diagnostics.json').write_text(json.dumps(limit_diagnostics,indent=2)+'\n')
    grid=data.pivot(index='mass_MeV',columns='width_scale',values='signed_r').reindex(index=MASSES,columns=SCALES)
    values=grid.to_numpy(); nominal=grid[1.0].to_numpy()
    plus=np.maximum(values,0).max(axis=1); minus=np.maximum(-values,0).max(axis=1)
    gains_plus=plus-np.maximum(nominal,0); gains_minus=minus-np.maximum(-nominal,0)
    envelope=pd.DataFrame({'mass_MeV':MASSES,'nominal_r':nominal,'r_min_across_widths':values.min(axis=1),
        'r_max_across_widths':values.max(axis=1),'best_excess_gain_same_mass':gains_plus,
        'best_deficit_gain_same_mass':gains_minus})
    envelope.to_csv(OUTPUT/'width_envelope.csv',index=False)

    regions=[]
    for polarity,sign in [('excess',1),('deficit',-1)]:
        score=sign*values; best=score.max(axis=1)
        candidates=list(find_peaks(best)[0])
        if best[0]>=best[1]:candidates.append(0)
        if best[-1]>=best[-2]:candidates.append(len(best)-1)
        accepted=[]
        for i in sorted(candidates,key=lambda i:best[i],reverse=True):
            if best[i]<1:continue
            mass=int(MASSES[i]);sm=sigma(mass/1000)*1000
            if any(abs(mass-int(MASSES[j]))<=2.25*max(sm,sigma(MASSES[j]/1000)*1000) for j in accepted):continue
            accepted.append(i)
            k=int(score[i].argmax()); scale=SCALES[k]
            nearby=np.where(np.abs(MASSES-mass)<=2.25*sm)[0]
            j=nearby[int(np.argmax(sign*nominal[nearby]))]
            regions.append({'polarity':polarity,'mass_MeV':mass,'best_width_scale':scale,
                'nominal_r_same_mass':float(nominal[i]),'best_width_r':float(values[i,k]),
                'gain_same_mass':float(best[i]-sign*nominal[i]),
                'nominal_regional_peak_mass_MeV':int(MASSES[j]),'nominal_regional_peak_r':float(nominal[j]),
                'gain_over_nominal_regional_peak':float(best[i]-sign*nominal[j]),
                'outside_prior_60_88_region':not (60<=mass<=88),'search_grid_endpoint':mass in (50,250)})
    regions=pd.DataFrame(regions)
    regions.to_csv(OUTPUT/'resolved_excursion_regions.csv',index=False)
    other=regions[regions.outside_prior_60_88_region].sort_values('gain_over_nominal_regional_peak',ascending=False)
    other.to_csv(OUTPUT/'other_regions_ranked_by_gain.csv',index=False)
    limit_ratios=data.pivot(index='mass_MeV',columns='width_scale',values='limit_ratio_to_nominal')
    limit_envelope=pd.DataFrame({'mass_MeV':MASSES,'minimum_limit_ratio':limit_ratios.min(axis=1).to_numpy(),
                                'maximum_limit_ratio':limit_ratios.max(axis=1).to_numpy()})
    limit_envelope.to_csv(OUTPUT/'upper_limit_width_envelope.csv',index=False)
    limit_summary={
        'confidence_level':0.9,'construction':'bounded piecewise asymptotic CLs; fixed width',
        'yield_coordinate':'full-template events; nominal K(m) held fixed',
        'coupling_plot_coordinate':'epsilon squared',
        'max_nominal_relative_difference':float(np.max(np.abs(nominal_limits.to_numpy()/saved.eps2_90.to_numpy()-1))),
        'max_absolute_cls_residual':float(np.max(np.abs(data.cls_at_limit-float(cfg.cls_alpha)))),
        'all_optimizer_statuses_ok':bool(data.limit_optimizer_ok.all()),
        'number_accepted':int(data.limit_optimizer_ok.sum()),'number_rejected':len(failed_limits),
        'rejected_points':failed_limits,
        'number_retried':len(retried_limits),'retried_points':retried_limits,
        'smallest_ratio_point':data.loc[data.limit_ratio_to_nominal.idxmin(),limit_columns].to_dict(),
        'largest_ratio_point':data.loc[data.limit_ratio_to_nominal.idxmax(),limit_columns].to_dict(),
        'median_minimum_ratio':float(limit_envelope.minimum_limit_ratio.median()),
        'median_maximum_ratio':float(limit_envelope.maximum_limit_ratio.median()),
        'per_width_median_ratio':{str(s):float(limit_ratios[s].median()) for s in SCALES},
        'focal_region_points':data[data.mass_MeV.isin([66,71,72,78,80])][limit_columns+['signed_r']].to_dict('records')}
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError(f"Existing study artifact changed: {path}")
    summary={'status':'completed' if not failed_limits else 'completed_with_rejected_upper_limit_points',
        'new_toy_spectra':0,'mass_grid_MeV':[50,250,1],'width_scales':list(SCALES),
        'n_width_fits':len(data),'n_gp_background_reconstructions':len(MASSES),
        'max_nominal_closure_abs_delta_r':float(np.max(np.abs(nominal-data[data.width_scale==1].nominal_saved_r.to_numpy()))),
        'signed_feasible_fallback_count':int(data.signed_feasible_fallbacks.sum()),
        'max_signed_feasible_nll_improvement':float(data.signed_fallback_nll_improvement.max()),
        'largest_excess':data.loc[data.signed_r.idxmax()].to_dict(),
        'largest_deficit':data.loc[data.signed_r.idxmin()].to_dict(),
        'maximum_excess_gain_same_mass':float(gains_plus.max()),'maximum_deficit_gain_same_mass':float(gains_minus.max()),
        'other_regions':other.to_dict('records'),'all_regions':regions.to_dict('records'),
        'upper_limits':limit_summary,
        'source_hashes':provenance,'protected_sha256':protected,'original_artifacts_unchanged':True,
        'runtime_manifest_sha256':production.RUNTIME_PROVENANCE['runtime_manifest_sha256'],
        'elapsed_seconds':time.monotonic()-started,
        'claim_boundary':'Template-only observed width sensitivity with backgrounds and fit bins fixed. Post-selection local scores; no calibrated global significance or detector-resolution measurement.'}
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    colors={.8:'#0072B2',.9:'#56B4E9',1.0:'black',1.1:'#E69F00',1.2:'#D55E00'}
    styles={.8:'--',.9:':',1.0:'-',1.1:':',1.2:'--'}
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.grid':True,'grid.alpha':.17,
                         'legend.frameon':False,'xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(2,1,figsize=(13.2,7.4),sharex=True,layout='constrained',height_ratios=[2,1])
    for scale in SCALES:
        axes[0].plot(MASSES,grid[scale],color=colors[scale],ls=styles[scale],lw=1.7 if scale==1 else .9,label=f'{scale:.1f} × nominal width')
    axes[0].axhline(0,color='0.6',lw=.7)
    axes[0].set(title='2021 template-width sensitivity: background and fitted bins held fixed',ylabel='Signed local diagnostic r')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.20),ncol=5,fontsize=8)
    axes[1].plot(MASSES,gains_plus,color='#0072B2',label='Largest excess-score gain')
    axes[1].plot(MASSES,gains_minus,color='#D55E00',label='Largest deficit-magnitude gain')
    axes[1].set(xlabel='Mass hypothesis (MeV)',ylabel='Gain at the same mass',xlim=(50,250))
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.27),ncol=2,fontsize=9)
    fig.savefig(FIGURES/'width_scan_overview.png',dpi=180)
    plt.close(fig)

    chosen=[]
    for polarity in ['excess','deficit']:
        subset=other[other.polarity==polarity].sort_values('gain_over_nominal_regional_peak',ascending=False).head(3)
        chosen.extend(subset.to_dict('records'))
    fig,axes=plt.subplots(2,3,figsize=(13.2,7.5),layout='constrained')
    for ax,row in zip(axes.flat,chosen):
        mass=row['mass_MeV'];radius=max(5,2.8*sigma(mass/1000)*1000)
        use=(MASSES>=mass-radius)&(MASSES<=mass+radius)
        for scale in SCALES:
            ax.plot(MASSES[use],grid.loc[MASSES[use],scale],color=colors[scale],ls=styles[scale],lw=1.7 if scale==1 else 1.1)
        ax.axvline(mass,color='0.7',lw=.6)
        ax.axhline(0,color='0.6',lw=.7)
        ax.set(title=f"{mass} MeV {row['polarity']} | best width {row['best_width_scale']:.1f}×",
               xlabel='Mass hypothesis (MeV)',ylabel='Signed local r')
    for ax in list(axes.flat)[len(chosen):]:ax.set_visible(False)
    handles=[plt.Line2D([],[],color=colors[s],ls=styles[s],label=f'{s:.1f} × nominal') for s in SCALES]
    fig.legend(handles=handles,loc='outside lower center',ncol=5,fontsize=9)
    fig.savefig(FIGURES/'other_regions_width_comparison.png',dpi=180)
    plt.close(fig)

    for column,filename,title,ylabel in [
        ('A90_full_template_events','upper_limits_signal_yield.png',
         '2021 (10%) observed 90% CLs signal-yield limits',r'Full-template signal yield $N_{90}$ [events]'),
        ('eps2_90','upper_limits_coupling.png',
         '2021 (10%) observed 90% CLs coupling limits',r'Coupling squared $\epsilon^2_{90}$')]:
        limit_grid=data.pivot(index='mass_MeV',columns='width_scale',values=column)
        fig,axes=plt.subplots(2,1,figsize=(12.8,7.4),sharex=True,layout='constrained',height_ratios=[2.3,1])
        for scale in SCALES:
            axes[0].plot(MASSES,limit_grid[scale],color=colors[scale],ls=styles[scale],
                         lw=1.8 if scale==1 else 1.05,label=f'{scale:.1f} × nominal width')
            axes[1].plot(MASSES,limit_grid[scale]/limit_grid[1.0],color=colors[scale],ls=styles[scale],
                         lw=1.6 if scale==1 else 1.05)
        axes[0].set(title=title,ylabel=ylabel,yscale='log')
        axes[1].set(xlabel='Mass hypothesis (MeV)',ylabel='Limit / nominal',xlim=(50,250))
        axes[1].axhline(1,color='0.6',lw=.6,zorder=0)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=5,fontsize=9)
        fig.savefig(FIGURES/filename,dpi=180)
        plt.close(fig)
    concise_limits={k:v for k,v in limit_summary.items() if k!='focal_region_points'}
    print(json.dumps({'n_width_fits':len(data),'upper_limits':concise_limits},indent=2),flush=True)


if __name__=='__main__':
    with pilot.threadpool_limits(limits=1):main()
