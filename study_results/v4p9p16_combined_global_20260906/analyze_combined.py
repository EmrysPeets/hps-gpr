#!/usr/bin/env python3
"""Apply the declared GP and direct global tests to the joint likelihood scan."""
from pathlib import Path
import argparse,hashlib,json,os,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.stats import norm,beta,kstest,ks_2samp,skew,kurtosis
from verify_combined import verify,sha
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
METHODS=('profiled','fixed')

def interval(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]
def tail(samples,threshold):
    k=int(np.count_nonzero(samples>=threshold));n=len(samples)
    return dict(k=k,n=n,p=k/n,interval95=interval(k,n),
        upper95_one_sided=1. if k==n else float(beta.ppf(.95,k+1,n-k)))
def holm(p):
    p=np.array(p);order=np.argsort(p)
    adjusted=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    out=np.empty_like(p);out[order]=adjusted;return out
def write(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gp-samples',type=int,default=200000)
    args=ap.parse_args();start=time.monotonic()
    accepted=verify()
    f=HERE/'global';out=f/'analysis';out.mkdir(exist_ok=True)
    vectors=np.load(f/'scan_vectors.npz')
    masses=vectors['masses_MeV']
    obs=pd.read_csv(f/'observed.csv').set_index('mass_MeV').loc[masses]
    result=dict(version='4.9.16',scope='One common stress scenario, independent datasets, finite union grid',
        masses_MeV=masses.tolist(),gp_samples=args.gp_samples,validation_scans=1000,
        numerical_audit_passed=accepted['passed'],methods={})
    curves=[];marginals=[];tailcurves=[];matrices={};maxima={};peakrows=[]
    for im,method in enumerate(METHODS):
        asimov=vectors['asimov_'+method];valid=vectors['validation1000_'+method]
        mean=asimov[0];response=asimov[1:]-mean
        C=response.T@response;sd=np.sqrt(np.diag(C))
        if not np.all(sd>0):raise RuntimeError('Zero response width')
        K=C/np.outer(sd,sd);eigen,U=np.linalg.eigh(K)
        if eigen.min()<-1e-9:raise RuntimeError('Indefinite correlation')
        factor=U*np.sqrt(np.maximum(eigen,0))
        if not np.allclose(factor@factor.T,K,atol=1e-10,rtol=1e-10):
            raise RuntimeError('GP factor does not reproduce correlation')
        robs=obs[method+'_r'].to_numpy()
        zobs=(robs-mean)/sd;score=np.where(robs>0,zobs,-np.inf)
        local=np.where(robs>0,norm.sf(zobs),1.)
        zvalid=(valid-mean)/sd
        direct=np.where(valid>0,zvalid,-np.inf).max(axis=1)
        direct_raw=np.maximum(valid,0).max(axis=1)
        rng=np.random.default_rng(np.random.SeedSequence([49160906,im]))
        gp=[];raw=[];coarse=[[],[]]
        for first in range(0,args.gp_samples,5000):
            z=rng.standard_normal((min(5000,args.gp_samples-first),len(masses)))@factor.T
            r=mean+sd*z;sc=np.where(r>0,z,-np.inf)
            gp.extend(sc.max(axis=1));raw.extend(np.maximum(r,0).max(axis=1))
            for offset in (0,1):coarse[offset].extend(sc[:,offset::2].max(axis=1))
        gp=np.array(gp);raw=np.array(raw)
        peak=int(np.argmax(score));rawpeak=int(np.argmax(robs))
        tests=[kstest(zvalid[:,j],'norm') for j in range(len(masses))]
        adjusted=holm([t.pvalue for t in tests])
        directK=np.corrcoef(zvalid,rowvar=False)
        info=dict(peak_mass_MeV=int(masses[peak]),observed_raw_r=float(robs[peak]),
            asimov_r=float(mean[peak]),response_sd=float(sd[peak]),
            observed_standardized_r=float(zobs[peak]),local_common_truth_p=float(local[peak]),
            local_asymptotic_p=float(norm.sf(max(robs[peak],0))),
            global_gp=tail(gp,score[peak]),global_direct=tail(direct,score[peak]),
            raw_ordering=dict(peak_mass_MeV=int(masses[rawpeak]),raw_r=float(robs[rawpeak]),
                global_gp=tail(raw,max(0,robs[rawpeak])),
                global_direct=tail(direct_raw,max(0,robs[rawpeak]))),
            asimov_offset_range=[float(mean.min()),float(mean.max())],
            response_sd_range=[float(sd.min()),float(sd.max())],
            valid_z_mean_range=[float(zvalid.mean(axis=0).min()),float(zvalid.mean(axis=0).max())],
            valid_z_sd_range=[float(zvalid.std(axis=0,ddof=1).min()),float(zvalid.std(axis=0,ddof=1).max())],
            marginal_normality_holm_flags=int(np.count_nonzero(adjusted<.05)),
            correlation_rms_difference=float(np.sqrt(np.mean((directK-K)**2))),
            correlation_min_eigenvalue=float(eigen.min()),
            coarse_2MeV_global_at_fine_peak=[tail(np.array(x),score[peak]) for x in coarse],
            boundary_correlations={str(m):float(K[m-19,m-18]) for m in (38,49,90,180)})
        for label,g,d in [('principal',gp,direct),('raw',raw,direct_raw)]:
            ks=ks_2samp(g,d,method='asymp')
            info[label+'_maximum_KS']=dict(statistic=float(ks.statistic),pvalue=float(ks.pvalue),
                interpretation='nominal diagnostic; two methods per ordering; not a data significance')
        info['gp_inside_direct_interval95']=info['global_direct']['interval95'][0]<=info['global_gp']['p']<=info['global_direct']['interval95'][1]
        result['methods'][method]=info
        for j,mass in enumerate(masses):
            g=tail(gp,score[j]);d=tail(direct,score[j]);rr=tail(raw,max(0,robs[j]))
            # An event at this mass is a subset of the union event. Use a
            # simultaneous, stringent MC bound rather than comparing zeros.
            inclusion_upper=1. if g['k']==g['n'] else beta.ppf(1-1e-7/len(masses),g['k']+1,g['n']-g['k'])
            if inclusion_upper<local[j]:raise RuntimeError('Global/local inclusion failed')
            curves.append(dict(method=method,mass_MeV=int(mass),
                dataset_set=obs.iloc[j].dataset_set,observed_r=robs[j],asimov_r=mean[j],
                response_sd=sd[j],z_standardized=zobs[j],
                p_asymptotic=float(norm.sf(max(0,robs[j]))),
                p_local_common_truth=local[j],
                p_global_gp=g['p'],gp_k=g['k'],gp_n=g['n'],
                p_global_gp_low=g['interval95'][0],p_global_gp_high=g['interval95'][1],
                p_global_gp_upper95=g['upper95_one_sided'],
                p_global_direct=d['p'],direct_k=d['k'],direct_n=d['n'],
                p_global_direct_low=d['interval95'][0],p_global_direct_high=d['interval95'][1],
                p_global_direct_upper95=d['upper95_one_sided'],
                p_global_raw_ordering=rr['p'],raw_gp_k=rr['k'],
                raw_gp_low=rr['interval95'][0],raw_gp_high=rr['interval95'][1],
                raw_gp_upper95=rr['upper95_one_sided']))
            marginals.append(dict(method=method,mass_MeV=int(mass),asimov_r=mean[j],
                response_sd=sd[j],toy_r_mean=float(valid[:,j].mean()),
                toy_r_sd=float(valid[:,j].std(ddof=1)),
                z_mean=float(zvalid[:,j].mean()),z_sd=float(zvalid[:,j].std(ddof=1)),
                z_skew=float(skew(zvalid[:,j])),z_excess_kurtosis=float(kurtosis(zvalid[:,j])),
                normality_KS=float(tests[j].statistic),normality_p=float(tests[j].pvalue),
                normality_holm_p=float(adjusted[j])))
        for threshold in np.linspace(0,6,61):
            g=tail(gp,threshold);d=tail(direct,threshold)
            tailcurves.append(dict(method=method,threshold=threshold,
                gp_p=g['p'],gp_low=g['interval95'][0],gp_high=g['interval95'][1],
                direct_p=d['p'],direct_low=d['interval95'][0],direct_high=d['interval95'][1],
                gp_k=g['k'],direct_k=d['k']))
        matrices.update({method+'_C':C,method+'_K':K,method+'_response':response,
                         method+'_validation_K':directK})
        maxima.update({method+'_gp':gp,method+'_direct':direct,
                       method+'_gp_raw':raw,method+'_direct_raw':direct_raw})
    frame=pd.DataFrame(curves)
    prof=result['methods']['profiled']
    representative=sorted({30,65,120,220,prof['peak_mass_MeV'],prof['raw_ordering']['peak_mass_MeV']})
    result['representative_masses_MeV']=representative
    result['v12_comparison']=dict(max_absolute_relative_limit_difference=float(abs(obs.v12_limit_relative_difference).max()),
        max_absolute_bounded_root_difference=float(abs(obs.v12_bounded_root_difference).max()),
        observed_limit_min=float(obs.profiled_eps2_display.min()),
        observed_limit_max=float(obs.profiled_eps2_display.max()),
        limit_best_mass_MeV=int(obs.profiled_eps2_display.idxmin()))
    frame.to_csv(out/'pvalue_curves.csv',index=False)
    pd.DataFrame(marginals).to_csv(out/'marginal_diagnostics.csv',index=False)
    pd.DataFrame(tailcurves).to_csv(out/'maximum_tail_curve.csv',index=False)
    frame[(frame.method=='profiled') & frame.mass_MeV.isin(representative)].to_csv(out/'representative_pvalues.csv',index=False)
    np.savez_compressed(out/'covariance.npz',masses_MeV=masses,**matrices)
    np.savez_compressed(out/'maxima.npz',**maxima)
    inputs=[Path(__file__),HERE/'verify_combined.py',HERE/'qa/numerical_validation.json',
            HERE/'provenance/observed_reference.json',f/'contract.json',f/'summary.json',
            f/'scan_vectors.npz',f/'observed.csv']
    result['input_sha256']={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    result['analysis_seconds']=time.monotonic()-start
    write(out/'summary.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
