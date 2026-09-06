#!/usr/bin/env python3
"""Mirror the frozen combined scan without new fitting or independent toys."""
from pathlib import Path
import csv,hashlib,json,os,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.stats import norm,beta,ks_2samp
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'v4p9p16_combined_global_20260906'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def tail(values,threshold):
    n=len(values);k=int(np.count_nonzero(values>=threshold))
    return dict(k=k,n=n,p=k/n,
        low95=0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
        high95=1. if k==n else float(beta.ppf(.975,k+1,n-k)),
        upper95=1. if k==n else float(beta.ppf(.95,k+1,n-k)))
def main():
    start=time.monotonic()
    prior=json.loads((HERE/'provenance/parent.json').read_text())
    assert sha(PARENT/'MANIFEST.csv')==prior['manifest_sha256']
    rows=list(csv.DictReader((PARENT/'MANIFEST.csv').open()))
    assert all((ROOT/r['path']).stat().st_size==int(r['bytes']) and sha(ROOT/r['path'])==r['sha256'] for r in rows)
    f=PARENT/'global';a=f/'analysis'
    source=np.load(f/'scan_vectors.npz')
    oldmax=np.load(a/'maxima.npz');cov=np.load(a/'covariance.npz')
    oldsummary=json.loads((a/'summary.json').read_text())
    observed=pd.read_csv(f/'observed.csv').set_index('mass_MeV')
    masses=source['masses_MeV']
    assert np.array_equal(masses,np.arange(19,251))
    checks={'parent_manifest_intact':True,'complete_grid':True}
    result=dict(version='4.9.16',extension='illustrative deficit scan',
        gp_realizations_reused=200000,direct_joint_scans_reused=1000,
        new_likelihood_fits=0,new_independent_toys=0,methods={})
    allrows=[];maxima={}
    for im,method in enumerate(('profiled','fixed')):
        asimov=source['asimov_'+method];mean=asimov[0]
        response=asimov[1:]-mean;C=response.T@response
        sd=np.sqrt(np.diag(C));K=C/np.outer(sd,sd)
        checks[method+'_unchanged_covariance']=np.array_equal(C,cov[method+'_C']) and np.array_equal(K,cov[method+'_K'])
        eigen,U=np.linalg.eigh(K);factor=U*np.sqrt(np.maximum(eigen,0))
        robs=observed[method+'_r'].to_numpy();zobs=(robs-mean)/sd
        score=np.where(robs<0,-zobs,-np.inf)
        local=np.where(robs<0,norm.cdf(zobs),1.)
        reference=np.where(robs<0,norm.cdf(robs),1.)
        valid=source['validation1000_'+method];zvalid=(valid-mean)/sd
        direct=np.where(valid<0,-zvalid,-np.inf).max(axis=1)
        directraw=np.maximum(-valid,0).max(axis=1)
        rng=np.random.default_rng(np.random.SeedSequence([49160906,im]))
        gp=[];gpraw=[];replayed=[];replayedraw=[]
        for first in range(0,200000,5000):
            z=rng.standard_normal((5000,len(masses)))@factor.T
            r=mean+sd*z
            gp.extend(np.where(r<0,-z,-np.inf).max(axis=1))
            gpraw.extend(np.maximum(-r,0).max(axis=1))
            replayed.extend(np.where(r>0,z,-np.inf).max(axis=1))
            replayedraw.extend(np.maximum(r,0).max(axis=1))
        gp=np.array(gp);gpraw=np.array(gpraw)
        checks[method+'_positive_principal_bitwise_replay']=np.array_equal(replayed,oldmax[method+'_gp'])
        checks[method+'_positive_raw_bitwise_replay']=np.array_equal(replayedraw,oldmax[method+'_gp_raw'])
        if not all(checks.values()):raise RuntimeError('Frozen covariance or GP replay mismatch')
        peak=int(np.argmax(score));rawpeak=int(np.argmin(robs))
        info=dict(peak_mass_MeV=int(masses[peak]),observed_r=float(robs[peak]),
            asimov_r=float(mean[peak]),response_sd=float(sd[peak]),
            observed_standardized_r=float(zobs[peak]),local_deficit_p=float(local[peak]),
            raw_gaussian_reference_p=float(reference[peak]),gp_global=tail(gp,score[peak]),
            direct_global=tail(direct,score[peak]),
            raw_ordering=dict(peak_mass_MeV=int(masses[rawpeak]),depth=float(-robs[rawpeak]),
                raw_gaussian_reference_p=float(reference[rawpeak]),
                local_deficit_p=float(local[rawpeak]),
                gp_global=tail(gpraw,max(0,-robs[rawpeak])),
                direct_global=tail(directraw,max(0,-robs[rawpeak]))))
        for label,g,d in [('principal',gp,direct),('raw_depth',gpraw,directraw)]:
            k=ks_2samp(g,d,method='asymp')
            info[label+'_maximum_KS']=dict(statistic=float(k.statistic),pvalue=float(k.pvalue))
        result['methods'][method]=info
        for j,m in enumerate(masses):
            tails={'gp':tail(gp,score[j]),'direct':tail(direct,score[j]),
                'raw_gp':tail(gpraw,max(0,-robs[j])),
                'raw_direct':tail(directraw,max(0,-robs[j]))}
            row=dict(method=method,mass_MeV=int(m),dataset_set=observed.iloc[j].dataset_set,
                observed_r=float(robs[j]),asimov_r=float(mean[j]),response_sd=float(sd[j]),
                z_standardized=float(zobs[j]),p_raw_gaussian=float(reference[j]),
                p_local_deficit=float(local[j]))
            for label,t in tails.items():
                row.update({label+'_'+k:v for k,v in t.items()})
            allrows.append(row)
            g=tails['gp'];upper=1. if g['k']==g['n'] else beta.ppf(1-1e-7/len(masses),g['k']+1,g['n']-g['k'])
            if upper<local[j]:raise RuntimeError('Global/local inclusion failed')
        maxima.update({method+'_gp':gp,method+'_direct':direct,
            method+'_gp_raw':gpraw,method+'_direct_raw':directraw})
    frame=pd.DataFrame(allrows)
    q=result['methods']['profiled']
    selected=sorted({30,66,76,120,220,q['peak_mass_MeV'],q['raw_ordering']['peak_mass_MeV']})
    result['representative_masses_MeV']=selected
    checks['raw_nonnegative_atom']=all((x[['p_local_deficit','gp_p','direct_p']]==1).all().all()
        for _,d in frame.groupby('method') for x in [d[d.observed_r>=0]])
    result['checks']={k:bool(v) for k,v in checks.items()}
    result['passed']=all(checks.values())
    assert result['passed']
    frame.to_csv(HERE/'analysis/deficit_curves.csv',index=False)
    frame[(frame.method=='profiled') & frame.mass_MeV.isin(selected)].to_csv(HERE/'analysis/representative_deficits.csv',index=False)
    np.savez_compressed(HERE/'analysis/deficit_maxima.npz',**maxima)
    inputs=[Path(__file__),HERE/'PROTOCOL.md',HERE/'provenance/parent.json',
        PARENT/'MANIFEST.csv',f/'observed.csv',f/'scan_vectors.npz',
        a/'summary.json',a/'maxima.npz',a/'covariance.npz']
    result['input_sha256']={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    result['seconds']=time.monotonic()-start
    dump(HERE/'analysis/summary.json',result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
