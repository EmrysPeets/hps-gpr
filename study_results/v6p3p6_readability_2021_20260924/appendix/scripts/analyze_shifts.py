#!/usr/bin/env python3
"""MC-only v16 TC/UC core shifts; exact v6.1 core locator, no observed-data fits.
Inputs: ../inputs/{tc,uc}/mNNN.npz (edges_GeV,sumw,sumw2) and mNNN.json.
Only unit-weight counts receive independent Poisson-bin resampling.
"""
from pathlib import Path
import argparse, hashlib, json, os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]; R=B/'results'; F=B/'figures'
BLUE='#245c91'; RED='#a54b35'; BLACK='#333333'
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'lines.markersize':3,'lines.linewidth':1,'pdf.fonttype':42})
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sigma(m):return 1000*np.polynomial.polynomial.polyval(m/1000.,[.00184825,-.001375,.085875])
def locate(m,edges,h,span=1.5):
    """Numerically identical algorithm to v6.1 core_centering.locate."""
    x=(edges[1:]+edges[:-1])/2;sig=sigma(m);dx=float(np.diff(edges)[0])
    sm=gaussian_filter1d(h,.2*sig/dx)
    search=np.flatnonzero(abs(x-m)<=3*sig);peak=int(search[np.argmax(sm[search])]);mode=x[peak]
    keep=abs(x-mode)<=span*sig;y=h[keep];lo=edges[:-1][keep];hi=edges[1:][keep]
    u=(x[keep]-mode)/sig;ul=(lo-mode)/sig;uh=(hi-mode)/sig;mix=(u-u.min())/(u.max()-u.min())
    scale=max(y.max(),1.);yn=y/scale
    def expected(p):
        area,mu,s,b0,b1=p
        return area*(ndtr((uh-mu)/s)-ndtr((ul-mu)/s))+b0*(1-mix)+b1*mix
    def residual(p):
        lam=np.maximum(expected(p),1e-15);term=lam-yn;pos=yn>0
        term[pos]+=yn[pos]*np.log(yn[pos]/lam[pos])
        return np.sign(lam-yn)*np.sqrt(np.maximum(2*term,0))
    start=[max(float(yn.sum())*.8,1.),0.,.65,max(float(yn[0])*.5,.0001),max(float(yn[-1])*.5,.0001)]
    fit=least_squares(residual,start,bounds=([0,-1,.25,0,0],[np.inf,1,2.5,np.inf,np.inf]),xtol=1e-11,ftol=1e-11,gtol=1e-10,max_nfev=400)
    p=fit.x;center=float(mode+p[1]*sig);active=bool(abs(p[1])>.999 or p[2]<.251 or p[2]>2.499)
    edge=bool(peak in (search[0],search[-1]));valid=bool(fit.success and not active and not edge)
    row=dict(mass_MeV=m,core_center_MeV=center,center_shift_MeV=center-m,analysis_sigma_MeV=sig,fitted_core_sigma_MeV=float(p[2]*sig),mode_MeV=float(mode),fit_halfwidth_analysis_sigma=span,core_fit_success=bool(fit.success),core_fit_bound_hit=active,mode_search_edge=edge,core_location_valid=valid,fit_deviance=float(np.sum(residual(p)**2)*scale),fit_bins=int(keep.sum()),shape_only=bool(m>240 or m<60))
    details=dict(x=x[keep],observed=y,fit=expected(p)*scale,gaussian=p[0]*(ndtr((uh-p[1])/p[2])-ndtr((ul-p[1])/p[2]))*scale)
    return row,details

def load_family(family):
    samples={}
    for path in sorted((B/'inputs'/family).glob('m[0-9][0-9][0-9].npz')):
        m=int(path.stem[1:]);d=np.load(path,allow_pickle=False)
        meta=json.loads(path.with_suffix('.json').read_text()) if path.with_suffix('.json').exists() else json.loads(str(d['metadata']))
        e=np.asarray(d['edges_GeV'],float)*1000;h=np.asarray(d['sumw'],float);v=np.asarray(d['sumw2'],float)
        assert len(e)==len(h)+1 and np.all(h>=0) and h.sum()>0 and np.allclose(np.diff(e),np.diff(e)[0])
        if 'sumw' not in meta and 'total_sumw' not in meta:raise ValueError(f'Full-selected sumw missing for {path}')
        total=float(meta['sumw'] if 'sumw' in meta else meta['total_sumw'])
        unit=bool(np.array_equal(h,v) and np.allclose(h,np.rint(h)))
        stats=meta.get('stats',{});under=float(stats.get('underflow',0))
        if under and not unit:raise ValueError('Weighted underflow lacks an explicit sumw convention')
        assert total>=h.sum()+under-1e-6
        samples[m]=dict(edges=e,h=h,v=v,total=total,unit=unit,under=under,meta=meta,file=str(path.relative_to(B)))
    return samples

def design(m,kind):
    m=np.atleast_1d(m).astype(float);x=(m-150)/100
    if kind=='constant':return np.ones((len(m),1))
    if kind=='linear':return np.column_stack([x*0+1,x])
    if kind=='logarithmic':return np.column_stack([x*0+1,np.log(m/150)])
    if kind=='quadratic':return np.column_stack([x*0+1,x,x*x])
    raise ValueError(kind)
def relations(core):
    q=core[(core.mass_MeV>=60)&(core.mass_MeV<=240)&core.core_location_valid]
    m=q.mass_MeV.to_numpy();y=q.center_shift_MeV.to_numpy();models=[];pred=[]
    for kind in ['constant','linear','logarithmic','quadratic']:
        X=design(m,kind)
        if len(m)<=X.shape[1]:continue
        beta=np.linalg.lstsq(X,y,rcond=None)[0];yf=X@beta;yp=[]
        for i,mi in enumerate(m):
            keep=m!=mi;coef=np.linalg.lstsq(X[keep],y[keep],rcond=None)[0];yp.append(float((design([mi],kind)@coef)[0]))
        error=np.asarray(yp)-y
        models.append(dict(model=kind,parameters=len(beta),coefficients=beta.tolist(),fit_RMS_MeV=float(np.sqrt(np.mean((yf-y)**2))),omit_one_RMS_MeV=float(np.sqrt(np.mean(error**2))),omit_one_max_abs_MeV=float(abs(error).max())))
        for mi,yi,fi,pi in zip(m,y,yf,yp):pred.append(dict(model=kind,mass_MeV=int(mi),measured_shift_MeV=yi,fitted_shift_MeV=fi,omitted_mass_predicted_shift_MeV=pi))
    return dict(models=models,selected_by_omit_one_RMS=min(models,key=lambda z:z['omit_one_RMS_MeV'])['model'] if models else None,coordinate='m in MeV; x=(m-150)/100; logarithmic ln(m/150)',fit_masses_MeV=m.tolist(),scope='MC-only descriptive model selection; omitted-mass tests are not independent validation; no extrapolation'),pd.DataFrame(pred)

def empirical_cdf(s,center,width,u):
    cdf=(np.r_[0,np.cumsum(s['h'])]+s['under'])/s['total']
    return np.interp(center+width*np.asarray(u),s['edges'],cdf,left=cdf[0],right=cdf[-1])
def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
def make_plots(family,samples,core,models,pred):
    name=family.upper();q=core[(core.mass_MeV>=60)&(core.mass_MeV<=240)&core.core_location_valid].set_index('mass_MeV')
    oldpath=B.parent/'provenance/intro_signal_mc/core_native_diagnostics.csv'
    old=pd.read_csv(oldpath) if oldpath.exists() else None
    log=next((x for x in models['models'] if x['model']=='logarithmic'),None)
    oldcoef=np.array([-3.2243308692909953,-2.213992811446465])
    fig,axs=plt.subplots(2,1,figsize=(7.1,5.2),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
    if old is not None:axs[0].errorbar(old.mass_MeV,old.center_shift_MeV,yerr=old.center_MC_bin_resample_std_MeV,fmt='s',color='.55',label='v13 TC signal MC')
    if family=='uc' and (R/'tc_core_fits.csv').exists():
        tc=pd.read_csv(R/'tc_core_fits.csv');tc=tc[(tc.mass_MeV>=60)&(tc.mass_MeV<=240)]
        axs[0].errorbar(tc.mass_MeV,tc.center_shift_MeV,yerr=tc.center_bin_resample_SD_MeV,fmt='^',color='#8853a1',label='v16 TC (different selection)')
    axs[0].errorbar(q.index,q.center_shift_MeV,yerr=q.center_bin_resample_SD_MeV,fmt='o',color=BLACK,label=f'v16 {name} signal MC')
    dense=np.linspace(60,240,361);axs[0].plot(dense,design(dense,'logarithmic')@oldcoef,color=RED,ls='--',label='v13 TC logarithmic relation')
    for mod,color,ls in zip(models['models'],['#777777','#b28b2e',BLUE,'#4b8660'],[':', '-.', '-', '--']):
        axs[0].plot(dense,design(dense,mod['model'])@mod['coefficients'],color=color,ls=ls,label=f"v16 {name}: {mod['model']}")
        d=pred[pred.model==mod['model']];axs[1].plot(d.mass_MeV,d.omitted_mass_predicted_shift_MeV-d.measured_shift_MeV,'o'+ls,color=color,label=mod['model'])
    axs[0].set(ylabel='Core shift c − generated mass [MeV]',title=f'2021 v16 {name} signal MC: core location and shift descriptions');fig.legend(*axs[0].get_legend_handles_labels(),frameon=False,ncol=3,fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.015));fig.get_layout_engine().set(rect=(0,0,1,.85))
    axs[1].set(xlabel='Generated signal mass [MeV]',ylabel='Omitted-mass prediction\nminus fitted core [MeV]');axs[1].axhline(0,color='.6',lw=.6)
    for ax in axs:ax.grid(alpha=.15)
    fig.text(.01,-.065,'Bars: sample SD of fitted centers across independent Poisson-bin replicas (32 by default); they describe MC statistics.\nPrediction test: omit one mass, fit the other masses, then predict the omitted core shift. No observed data or upper limits.\nTC and UC have different selection cuts; a TC–UC difference cannot be attributed solely to the vertex constraint.',fontsize=8)
    save(fig,f'{family}_core_shift')
    n=len(samples);nr=(n+2)//3;fig,axs=plt.subplots(nr,3,figsize=(9.1,2.35*nr),squeeze=False,layout='constrained')
    for ax,m in zip(axs.flat,sorted(samples)):
        s=samples[m];e=s['edges'];h=s['h'];x=(e[:-1]+e[1:])/2;dx=np.diff(e)
        ax.step(x,h/s['total']/dx,where='mid',color=BLACK,lw=.7,label=f'{name} signal MC')
        row=core[core.mass_MeV==m].iloc[0];fit,detail=locate(m,e,h)
        ax.plot(detail['x'],detail['fit']/s['total']/dx[0],color=BLUE,label='Local Gaussian + pedestal')
        ax.plot(detail['x'],detail['gaussian']/s['total']/dx[0],color=RED,ls='--',label='Fitted core Gaussian')
        ax.set(xlim=(m-6*sigma(m),m+10*sigma(m)),ylim=(1e-5,.4),yscale='log',title=f'{m} MeV'+(' — shape only' if m>240 else ''),xlabel='Reconstructed mass [MeV]',ylabel='Full probability / MeV');ax.grid(alpha=.12);ax.tick_params(labelsize=7)
    spare=list(axs.flat)[n:]
    for ax in spare:ax.axis('off')
    handles,labels=axs.flat[0].get_legend_handles_labels()
    if spare:
        spare[0].legend(handles,labels,loc='upper left',frameon=False,fontsize=9)
        spare[0].text(0,.58,'Fit: bin-integrated Gaussian\nplus nonnegative linear pedestal.\nCore fit half-span = 1.5 σref.\nσref is the TC analysis search scale;\nfitted σcore is free, also for UC.\n\nMC normalized to all selected\nevents including flow bins.\nNo observed fits or upper limits.',va='top',transform=spare[0].transAxes,fontsize=8)
    else:fig.legend(handles,labels,loc='lower center',ncol=3)
    fig.suptitle(f'2021 v16 {name}: selected signal MC and local Gaussian core fits',fontsize=12)
    save(fig,f'{family}_catalogue')
    u=np.linspace(-30,60,1801);uc=(u[:-1]+u[1:])/2;du=np.diff(u);curves={};metrics=[]
    for m,row in q.iterrows():curves[m]=empirical_cdf(samples[m],row.core_center_MeV,row.fitted_core_sigma_MeV,u)
    fig,axs=plt.subplots(1,2,figsize=(9.2,4.4),layout='constrained');colors=plt.cm.viridis(np.linspace(.08,.9,max(1,len(q)-1)))
    for (m,cdf),color in zip(curves.items(),[RED,*colors]):
        prob=np.diff(cdf)/du;lo=np.interp(-2,u,cdf);hi=np.interp(2,u,cdf);f=hi-lo
        axs[0].plot(uc,prob/f,color=color,lw=1.25 if m==60 else .8,label=f'{m} MeV');axs[1].semilogy(uc,np.maximum(prob,1e-9),color=color,lw=1.25 if m==60 else .8)
        other=np.mean([v for k,v in curves.items() if k!=m],axis=0);olo=np.interp(-2,u,other);ohi=np.interp(2,u,other)
        fine=np.linspace(-2,2,401);row=q.loc[m];direct=empirical_cdf(samples[m],row.core_center_MeV,row.fitted_core_sigma_MeV,fine)
        shared=np.mean([empirical_cdf(samples[k],q.loc[k].core_center_MeV,q.loc[k].fitted_core_sigma_MeV,fine) for k in curves if k!=m],axis=0)
        distance=float(np.max(np.abs((direct-direct[0])/(direct[-1]-direct[0])-(shared-shared[0])/(shared[-1]-shared[0]))))
        metrics.append(dict(mass_MeV=int(m),core_fraction_2core_sigma=float(f),omit_one_core_CDF_distance=distance,full_selected_normalization=True))
    mean=np.mean(list(curves.values()),axis=0);pool=np.diff(mean)/du;fraction=np.interp(2,u,mean)-np.interp(-2,u,mean)
    axs[0].plot(uc,pool/fraction,'k--',label='Equal-mass empirical shape',lw=1.4);axs[1].semilogy(uc,np.maximum(pool,1e-9),'k--',lw=1.4)
    axs[0].plot(uc,np.exp(-uc*uc/2)/np.sqrt(2*np.pi)/(ndtr(2)-ndtr(-2)),':',color='.5',label='Standard Gaussian (core)',lw=1.4)
    axs[0].set(xlim=(-2,2),ylim=(0,.5),ylabel='Density conditioned on |u| < 2',title=f'v16 {name}: aligned signal cores')
    axs[1].set(xlim=(-12,25),ylim=(1e-5,.5),ylabel='Full selected probability density',title='Retaining the original tail probabilities')
    for ax in axs:ax.set_xlabel(r'$u=(m_{\rm rec}-c_{\rm core})/\sigma_{\rm core}$');ax.grid(alpha=.15)
    fig.legend(*axs[0].get_legend_handles_labels(),ncol=4,loc='lower center',bbox_to_anchor=(.5,-.15),frameon=False,fontsize=7)
    fig.text(.01,-.22,'Centers and widths come from each sample’s own local fit. Comparison of core shape conditions on those fitted values.\nEmpirical lines, no error bars; 260 MeV is excluded from common-shape and shift-law fits. No efficiencies or upper limits.',fontsize=8)
    save(fig,f'{family}_common_core')
    return pd.DataFrame(metrics)

def analyze(family,args):
    samples=load_family(family)
    if not samples:return dict(family=family,status='missing input; skipped')
    rows=[];spans=[];replicas=[];warnings=[]
    for m,s in samples.items():
        row,_=locate(m,s['edges'],s['h']);row.update(family=family,unit_weights=s['unit'],selected_sumw=s['total'],stored_sumw=float(s['h'].sum()),center_bin_resample_SD_MeV=None,replicas_requested=args.replicas,replicas_valid=0)
        for span in [1.25,1.5,2.]:
            r,_=locate(m,s['edges'],s['h'],span);spans.append(dict(family=family,mass_MeV=m,span=span,core_center_MeV=r['core_center_MeV'],center_delta_MeV=r['core_center_MeV']-row['core_center_MeV'],core_sigma_MeV=r['fitted_core_sigma_MeV'],valid=r['core_location_valid']))
        vals=[]
        if s['unit']:
            for i in range(args.replicas):
                seed=np.random.SeedSequence([args.seed,1 if family=='tc' else 2,m,i]);rng=np.random.default_rng(seed)
                rr,_=locate(m,s['edges'],rng.poisson(s['h']))
                replicas.append(dict(family=family,mass_MeV=m,replica=i,core_center_MeV=rr['core_center_MeV'],core_sigma_MeV=rr['fitted_core_sigma_MeV'],valid=rr['core_location_valid']))
                if rr['core_location_valid']:vals.append(rr['core_center_MeV'])
            row['replicas_valid']=len(vals)
            if len(vals)==args.replicas and len(vals)>1:row['center_bin_resample_SD_MeV']=float(np.std(vals,ddof=1))
            else:warnings.append(f'{family} {m}: invalid replica(s); SD not reported')
        else:warnings.append(f'{family} {m}: nonunit weights; Poisson-count resampling disabled')
        rows.append(row);print(f'{family} {m}: shift {row["center_shift_MeV"]:.5f} MeV; valid replicas {len(vals)}/{args.replicas}',flush=True)
    core=pd.DataFrame(rows);models,pred=relations(core)
    core.to_csv(R/f'{family}_core_fits.csv',index=False,float_format='%.17g');pd.DataFrame(spans).to_csv(R/f'{family}_span_checks.csv',index=False,float_format='%.17g');pd.DataFrame(replicas).to_csv(R/f'{family}_bin_replicas.csv',index=False,float_format='%.17g');pred.to_csv(R/f'{family}_shift_predictions.csv',index=False,float_format='%.17g');write(R/f'{family}_shift_models.json',models)
    oldpath=B.parent/'provenance/intro_signal_mc/core_native_diagnostics.csv'
    if oldpath.exists():
        old=pd.read_csv(oldpath)[['mass_MeV','core_center_MeV','fitted_core_sigma_MeV']].rename(columns={'core_center_MeV':'v13_TC_core_center_MeV','fitted_core_sigma_MeV':'v13_TC_core_sigma_MeV'})
        compare=core[['mass_MeV','core_center_MeV','fitted_core_sigma_MeV']].merge(old,on='mass_MeV')
        compare['v16_minus_v13_TC_center_MeV']=compare.core_center_MeV-compare.v13_TC_core_center_MeV
        compare['v16_to_v13_TC_core_width_ratio']=compare.fitted_core_sigma_MeV/compare.v13_TC_core_sigma_MeV
        compare.to_csv(R/f'{family}_versus_v13_TC.csv',index=False,float_format='%.17g')
    shape=make_plots(family,samples,core,models,pred);shape.to_csv(R/f'{family}_shape_metrics.csv',index=False,float_format='%.17g')
    s=pd.DataFrame(spans);d=core[(core.mass_MeV>=60)&(core.mass_MeV<=240)];m80=shape[shape.mass_MeV>=80]
    return dict(family=family,status='completed',masses_MeV=sorted(samples),fit_masses_MeV=models['fit_masses_MeV'],selected_shift_model=models['selected_by_omit_one_RMS'],models=models['models'],shift_range_MeV=[float(d.center_shift_MeV.min()),float(d.center_shift_MeV.max())],all_central_fits_valid=bool(core.core_location_valid.all()),all_span_fits_valid=bool(s.valid.all()),max_span_shift_delta_MeV=float(abs(s.center_delta_MeV).max()),max_span_shift_delta_60to240_MeV=float(abs(s[(s.mass_MeV>=60)&(s.mass_MeV<=240)].center_delta_MeV).max()),valid_replicas=int(core.replicas_valid.sum()),all_unit_weights=bool(core.unit_weights.all()),core_CDF_distance_80to240_range=[float(m80.omit_one_core_CDF_distance.min()),float(m80.omit_one_core_CDF_distance.max())],warnings=warnings,scope='Selected signal-MC shape only. No observed fits, injection recovery, efficiencies, limits or global probabilities.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--families',nargs='+',default=['tc','uc']);ap.add_argument('--replicas',type=int,default=32);ap.add_argument('--seed',type=int,default=924636);args=ap.parse_args()
    R.mkdir(parents=True,exist_ok=True);F.mkdir(parents=True,exist_ok=True)
    summaries=[analyze(family,args) for family in args.families]
    for summary in summaries:write(R/f'{summary["family"]}_summary.json',summary)
    summaries=[json.loads(path.read_text()) for path in sorted(R.glob('*_summary.json'))]
    write(R/'summary.json',dict(seed=args.seed,replicas_per_mass=args.replicas,families=summaries,algorithm_source='v6.1 scripts/core_centering.py locate; same fit bin selection, initialization, optimizer tolerances, model and parameter bounds',uncertainty='Sample SD over independent Poisson-bin replicas; only for unit-weight histograms; not error on 100-toy mean',relation_selection='Unweighted least squares; omit each mass once, fit the other masses, predict omitted shift; model selection not independent validation'))
    ledger=[]
    for p in sorted((B/'inputs').glob('*/*')):
        if p.is_file():ledger.append(dict(file=str(p.relative_to(B)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write(R/'input_sha256.json',ledger)
if __name__=='__main__':main()
