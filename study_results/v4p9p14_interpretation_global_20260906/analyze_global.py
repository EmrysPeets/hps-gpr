#!/usr/bin/env python3
"""Asimov GP field, independent full-scan validation, and separate p-value figures."""
from pathlib import Path
import argparse, hashlib, json, os, time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.stats import norm,beta,kstest,skew,kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'v4p9p13_calibration_20260905'
METHODS=('profiled','fixed')
COLORS={'raw':'#777777','local':'#18699C','global':'#AE3E30','direct':'#1C7044'}
plt.rcParams.update({'font.family':'serif','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.dpi':170,'pdf.fonttype':42})

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def interval(k,n):return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
def summary_tail(samples,threshold):
    k=int(np.count_nonzero(samples>=threshold));n=len(samples)
    return dict(k=k,n=n,p=k/n,interval95=interval(k,n))
def holm(p):
    p=np.array(p);order=np.argsort(p);adjusted=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]));out=np.empty_like(p);out[order]=adjusted;return out
def savefig(fig,out,name):
    for suffix in ('pdf','png'):fig.savefig(out/f'{name}.{suffix}',bbox_inches='tight')
    plt.close(fig)
def write_json(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='2015',choices=['2015','2016','2021']);ap.add_argument('--gp-samples',type=int,default=200000)
    args=ap.parse_args();start=time.monotonic();folder=HERE/'global'/args.dataset;out=folder/'analysis';out.mkdir(exist_ok=True)
    figdir=HERE/'figures';figdir.mkdir(exist_ok=True)
    contracts=[]
    for ensemble in ('pilot10','validation1000','asimov'):
        base=folder/ensemble
        summary=json.loads((base/'summary.json').read_text());contract=json.loads((base/'contract.json').read_text())
        if not summary['passed'] or not summary['complete']:raise RuntimeError('Incomplete ensemble: '+ensemble)
        for name,key in [('contract.json','contract_sha256'),('scan_vectors.npz','vectors_sha256'),('spectra.npz','spectra_sha256')]:
            if sha(base/name)!=summary[key]:raise RuntimeError('Ensemble checksum mismatch: '+ensemble+'/'+name)
        for mass in contract['masses_MeV']:
            audit=json.loads((base/f'm{mass:03d}_qa.json').read_text())
            if not audit['passed'] or audit['checkpoint_sha256']!=sha(base/f'm{mass:03d}.npz'):raise RuntimeError('Failed point QA')
        contracts.append(contract)
    for key in ('dataset','masses_MeV','truth_array_sha256','source_sha256','parent_contract_sha256'):
        if any(c[key]!=contracts[0][key] for c in contracts[1:]):raise RuntimeError('Ensemble contracts differ: '+key)
    for name,digest in contracts[0]['source_sha256'].items():
        if sha(HERE.parents[1]/name)!=digest:raise RuntimeError('Source changed: '+name)
    a=np.load(folder/'asimov/scan_vectors.npz');v=np.load(folder/'validation1000/scan_vectors.npz');masses=a['masses_MeV'];assert np.array_equal(masses,v['masses_MeV'])
    observed=pd.read_csv(PARENT/'summary/observed_calibrated_limits.csv');observed=observed[observed.scope_key=='individual_'+args.dataset+('_10pct' if args.dataset=='2021' else '_full')].set_index('mass_MeV').loc[masses]
    result=dict(dataset=args.dataset,scope='finite 1 MeV grid; common archived stress truth; frozen kernels',masses_MeV=masses.tolist(),gp_samples=args.gp_samples,methods={})
    curves=[];diagnostics=[];matrices={};maxima={};nulls={};tailcurve=[]
    for method in METHODS:
        mean=a[method][0];response=a[method][1:]-mean
        C=response.T@response;sd=np.sqrt(np.diag(C));K=C/np.outer(sd,sd)
        eigen,U=np.linalg.eigh(K)
        if eigen.min()<-1e-9 or np.max(abs(np.diag(K)-1))>1e-12:raise RuntimeError('Invalid GP covariance')
        factor=U*np.sqrt(np.maximum(eigen,0))
        robs=observed['signed_r_'+method+'_asymptotic'].to_numpy();zobs=(robs-mean)/sd
        score_obs=np.where(robs>0,zobs,-np.inf)
        local=np.where(robs>0,norm.sf(zobs),1.)
        zvalid=(v[method]-mean)/sd
        valid_score=np.where(v[method]>0,zvalid,-np.inf)
        direct_max=valid_score.max(axis=1);direct_raw=np.maximum(v[method],0).max(axis=1)
        rng=np.random.default_rng(np.random.SeedSequence([491406,int(args.dataset),METHODS.index(method)]))
        gp_max=[];gp_raw=[];coarse=[[],[]]
        for first in range(0,args.gp_samples,5000):
            z=rng.standard_normal((min(5000,args.gp_samples-first),len(masses)))@factor.T
            r=mean+sd*z;scores=np.where(r>0,z,-np.inf)
            gp_max.extend(scores.max(axis=1));gp_raw.extend(np.maximum(r,0).max(axis=1))
            for offset in (0,1):coarse[offset].extend(scores[:,offset::2].max(axis=1))
        gp_max=np.array(gp_max);gp_raw=np.array(gp_raw)
        count=np.array([np.count_nonzero(gp_max>=s) for s in score_obs]);global_p=count/len(gp_max);global_p[robs<=0]=1.
        direct_count=np.array([np.count_nonzero(direct_max>=s) for s in score_obs]);direct_p=direct_count/len(direct_max);direct_p[robs<=0]=1.
        global_bounds=np.array([interval(k,len(gp_max)) for k in count])
        direct_bounds=np.array([interval(k,len(direct_max)) for k in direct_count])
        raw_global=np.array([np.mean(gp_raw>=max(0,r)) for r in robs])
        raw_local=norm.sf(np.maximum(robs,0))
        # Exact one-sided MC bounds remain valid when the exceedance count is zero.
        inclusion_upper=np.array([1. if k==len(gp_max) else beta.ppf(1-1e-7/len(masses),k+1,len(gp_max)-k) for k in count])
        if np.any(inclusion_upper<local):raise RuntimeError('Global/local inclusion failed')
        corr=np.corrcoef(zvalid,rowvar=False)
        peak=int(np.argmax(score_obs));rawpeak=int(np.argmax(robs))
        tests=[kstest(zvalid[:,j],'norm') for j in range(len(masses))]
        adj=holm([t.pvalue for t in tests])
        info=dict(peak_mass_MeV=int(masses[peak]),observed_raw_r=float(robs[peak]),observed_standardized_r=float(zobs[peak]),
            local_asymptotic_p=float(raw_local[peak]),local_common_truth_p=float(local[peak]),
            global_gp=summary_tail(gp_max,score_obs[peak]),global_direct=summary_tail(direct_max,score_obs[peak]),
            raw_ordering=dict(peak_mass_MeV=int(masses[rawpeak]),raw_r=float(robs[rawpeak]),global_gp=summary_tail(gp_raw,max(0,robs[rawpeak])),global_direct=summary_tail(direct_raw,max(0,robs[rawpeak]))),
            asimov_offset_range=[float(mean.min()),float(mean.max())],response_sd_range=[float(sd.min()),float(sd.max())],
            valid_z_mean_range=[float(zvalid.mean(axis=0).min()),float(zvalid.mean(axis=0).max())],valid_z_sd_range=[float(zvalid.std(axis=0,ddof=1).min()),float(zvalid.std(axis=0,ddof=1).max())],
            marginal_normality_holm_flags=int(np.count_nonzero(adj<.05)),correlation_rms_difference=float(np.sqrt(np.mean((corr-K)**2))),
            min_cov_eigenvalue=float(eigen.min()),adjacent_corr_range=[float(np.diag(K,1).min()),float(np.diag(K,1).max())],
            coarse_2MeV_global_at_fine_peak=[summary_tail(np.array(coarse[i]),score_obs[peak]) for i in (0,1)])
        info['gp_global_inside_direct_interval95']=bool(info['global_direct']['interval95'][0]<=info['global_gp']['p']<=info['global_direct']['interval95'][1])
        result['methods'][method]=info
        for j,mass in enumerate(masses):
            curves.append(dict(method=method,mass_MeV=int(mass),observed_r=robs[j],asimov_r=mean[j],response_sd=sd[j],z_standardized=zobs[j],p_asymptotic=raw_local[j],p_local_common_truth=local[j],p_global_gp=global_p[j],p_global_gp_low=global_bounds[j,0],p_global_gp_high=global_bounds[j,1],p_global_direct=direct_p[j],p_global_direct_low=direct_bounds[j,0],p_global_direct_high=direct_bounds[j,1],p_global_raw_ordering=raw_global[j],parent_envelope_p=observed.iloc[j]['p0_'+method+'_calibrated']))
            diagnostics.append(dict(method=method,mass_MeV=int(mass),asimov_r=mean[j],response_sd=sd[j],toy_r_mean=v[method][:,j].mean(),toy_r_sd=v[method][:,j].std(ddof=1),z_mean=zvalid[:,j].mean(),z_sd=zvalid[:,j].std(ddof=1),z_skew=skew(zvalid[:,j]),z_excess_kurtosis=kurtosis(zvalid[:,j]),normality_KS=tests[j].statistic,normality_p=tests[j].pvalue,normality_holm_p=adj[j]))
        for t in np.linspace(0,6,61):
            g=summary_tail(gp_max,t);d=summary_tail(direct_max,t)
            tailcurve.append(dict(method=method,threshold=t,gp_p=g['p'],gp_low=g['interval95'][0],gp_high=g['interval95'][1],direct_p=d['p'],direct_low=d['interval95'][0],direct_high=d['interval95'][1]))
        matrices.update({method+'_K':K,method+'_C':C,method+'_response':response,method+'_validation_K':corr})
        maxima.update({method+'_gp':gp_max,method+'_direct':direct_max,method+'_gp_raw':gp_raw})
        nulls[method]=dict(mean=mean,sd=sd,zvalid=zvalid)
    df=pd.DataFrame(curves);diag=pd.DataFrame(diagnostics);tails=pd.DataFrame(tailcurve)
    df.to_csv(out/'pvalue_curves.csv',index=False);diag.to_csv(out/'marginal_diagnostics.csv',index=False);tails.to_csv(out/'maximum_tail_curve.csv',index=False)
    np.savez_compressed(out/'covariance.npz',masses_MeV=masses,**matrices);np.savez_compressed(out/'maxima.npz',**maxima)
    result['seconds']=time.monotonic()-start
    result['input_sha256']={str(p.relative_to(HERE)):sha(p) for p in [folder/'asimov/scan_vectors.npz',folder/'validation1000/scan_vectors.npz',folder/'pilot10/summary.json',Path(__file__),HERE/'PROTOCOL.md']}
    result['parent_observed_sha256']=sha(PARENT/'summary/observed_calibrated_limits.csv')
    result['ensemble_contracts_and_all_point_checks_verified']=True
    write_json(out/'summary.json',result)

    fig,axes=plt.subplots(2,1,figsize=(9.4,7.3),sharex=True)
    for ax,method in zip(axes,METHODS):
        d=df[df.method==method]
        ax.plot(masses,np.maximum(d.p_asymptotic,1e-8),color=COLORS['raw'],ls='--',label='Local: asymptotic')
        ax.plot(masses,np.maximum(d.p_local_common_truth,1e-8),color=COLORS['local'],label='Local: common-truth Gaussian pilot')
        ax.plot(masses,np.where(d.p_global_gp>0,d.p_global_gp,np.nan),color=COLORS['global'],label='Global: GP field, 72 masses')
        ax.plot(masses,d.p_global_raw_ordering,color='#B87715',ls=':',label='Global: raw-root ordering')
        zero=d.p_global_gp==0
        if zero.any():ax.scatter(masses[zero],d.p_global_gp_high[zero],marker='v',color=COLORS['global'],label='Zero GP tail: 95% upper bound',s=25)
        for column,color in [('p_asymptotic',COLORS['raw']),('p_local_common_truth',COLORS['local'])]:
            clipped=d[column]<1e-8
            if clipped.any():ax.scatter(masses[clipped],np.full(clipped.sum(),1e-8),marker='v',s=18,color=color)
        ax.set_yscale('log');ax.set_ylim(8e-9,1.5);ax.set_ylabel('Tail probability');ax.grid(alpha=.18)
        ax.set_title('Gaussian-profiled background' if method=='profiled' else 'Fixed background statistic',loc='left',fontsize=12)
    axes[-1].set_xlabel('Tested mass [MeV]')
    fig.suptitle(f'{args.dataset}: local and global probabilities under one common stress background',fontsize=12,y=.995)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.96),ncol=2,fontsize=9,frameon=False)
    fig.tight_layout(rect=(0,0,1,.89));savefig(fig,figdir,'global_pvalues_'+args.dataset)

    fig,axes=plt.subplots(1,2,figsize=(9.4,4))
    for ax,method in zip(axes,METHODS):
        im=ax.imshow(matrices[method+'_K'],origin='lower',extent=[masses[0]-.5,masses[-1]+.5]*2,vmin=-1,vmax=1,cmap='RdBu_r',aspect='equal')
        ax.set_title(method.capitalize());ax.set_xlabel('Mass [MeV]');ax.set_ylabel('Mass [MeV]')
    fig.colorbar(im,ax=axes,shrink=.82,label='Signed-root correlation',pad=.04)
    savefig(fig,figdir,'global_correlation_'+args.dataset)

    fig,axes=plt.subplots(2,2,figsize=(9.4,6.5),sharex=True)
    for j,method in enumerate(METHODS):
        d=diag[diag.method==method]
        axes[0,j].plot(masses,d.asimov_r,label='Unfluctuated background',color=COLORS['local'])
        axes[0,j].errorbar(masses,d.toy_r_mean,yerr=d.toy_r_sd/np.sqrt(1000),fmt='.',ms=2,color=COLORS['direct'],label='1,000 direct toys: mean')
        axes[0,j].axhline(0,color='grey',lw=.7);axes[0,j].set_title(method.capitalize());axes[0,j].set_ylabel('Mean signed root')
        axes[1,j].plot(masses,d.response_sd,label='Asimov response',color=COLORS['local'])
        axes[1,j].plot(masses,d.toy_r_sd,label='Direct toy spread',color=COLORS['direct'],ls='--')
        axes[1,j].axhline(1,color='grey',lw=.7);axes[1,j].set_xlabel('Mass [MeV]');axes[1,j].set_ylabel('Standard deviation')
        for ax in axes[:,j]:ax.grid(alpha=.18)
    h1,l1=axes[0,0].get_legend_handles_labels();h2,l2=axes[1,0].get_legend_handles_labels()
    fig.legend(h1+h2,l1+l2,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,fontsize=9,frameon=False)
    fig.tight_layout(rect=(0,0,1,.91));savefig(fig,figdir,'global_marginal_checks_'+args.dataset)

    fig,axes=plt.subplots(1,2,figsize=(9.4,4.2),sharey=True)
    for ax,method in zip(axes,METHODS):
        d=tails[tails.method==method]
        ax.plot(d.threshold,np.where(d.gp_p>0,d.gp_p,np.nan),label='200,000 GP fields',color=COLORS['global'])
        ax.fill_between(d.threshold,np.maximum(d.direct_low,1e-6),d.direct_high,color=COLORS['direct'],alpha=.18,label='Direct toys: 95% interval')
        ax.plot(d.threshold,np.where(d.direct_p>0,d.direct_p,np.nan),'--',color=COLORS['direct'],label='1,000 direct scans')
        for column,upper,color in [('gp_p','gp_high',COLORS['global']),('direct_p','direct_high',COLORS['direct'])]:
            zero=d[d[column]==0]
            if len(zero):ax.scatter([zero.iloc[0].threshold],[zero.iloc[0][upper]],marker='v',s=30,color=color)
        ax.plot(d.threshold,norm.sf(d.threshold),':',color='grey',label='One standard-normal point')
        ax.set_title(method.capitalize());ax.set_yscale('log');ax.set_ylim(1e-5,1.1);ax.set_xlim(0,5);ax.set_xlabel('Scan threshold');ax.grid(alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=9,frameon=False)
    axes[0].set_ylabel('Probability scan exceeds threshold');fig.tight_layout(rect=(0,0,1,.89));savefig(fig,figdir,'global_tail_validation_'+args.dataset)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
