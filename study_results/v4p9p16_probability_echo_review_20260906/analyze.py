#!/usr/bin/env python3
"""Recompute probabilities from stored ensembles and replay deterministic echoes."""
from pathlib import Path
import csv,hashlib,json,os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
V16=HERE.parent/'v4p9p16_combined_global_20260906'
EXT=HERE.parent/'v4p9p16_presentation_extractions_20260906'
OLD=HERE.parent/'v4p9p12_2021_peak_dip_diagnostic_20toys_20260905'
sys.path.insert(0,str(V16))
import run_combined as production
core,c,np,pd=production.core,production.c,production.np,production.pd
from scipy.stats import norm,beta
SOURCES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
    p=Path(p);SOURCES[str(p.relative_to(ROOT))]=sha(p);return p
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def tails(samples,threshold):
    n=len(samples);k=int(np.count_nonzero(samples>=threshold))
    return dict(k=k,n=n,p=k/n,low95=0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
        high95=1. if k==n else float(beta.ppf(.975,k+1,n-k)),
        upper95=1. if k==n else float(beta.ppf(.95,k+1,n-k)))

def probabilities():
    o=pd.read_csv(bind(V16/'global/observed.csv')).sort_values('mass_MeV')
    old=pd.read_csv(bind(V16/'global/analysis/pvalue_curves.csv'));old=old[old.method=='profiled'].sort_values('mass_MeV')
    vec=np.load(bind(V16/'global/scan_vectors.npz'));maxima=np.load(bind(V16/'global/analysis/maxima.npz'))
    cov=np.load(bind(V16/'global/analysis/covariance.npz'))
    m=vec['masses_MeV'];r=o.profiled_r.to_numpy();a=vec['asimov_profiled'][0];D=vec['asimov_profiled'][1:]-a
    C=D.T@D;s=np.sqrt(np.diag(C));K=C/np.outer(s,s)
    z=(r-a)/s;score=np.where(r>0,z,-np.inf);local=np.where(r>0,norm.sf(z),1.)
    nominal=norm.sf(np.maximum(r,0));toys=vec['validation1000_profiled']
    toy_score=np.where(toys>0,(toys-a)/s,-np.inf);directmax=toy_score.max(axis=1)
    rawmax=np.maximum(toys,0).max(axis=1)
    assert np.array_equal(m,o.mass_MeV) and np.array_equal(m,old.mass_MeV) and len(m)==232
    assert np.allclose(K,cov['profiled_K'],atol=1e-12,rtol=1e-12)
    assert np.array_equal(directmax,maxima['profiled_direct'])
    assert np.array_equal(rawmax,maxima['profiled_direct_raw'])
    assert np.allclose(local,old.p_local_common_truth,atol=1e-15,rtol=2e-13)
    assert np.allclose(nominal,old.p_asymptotic,atol=1e-15,rtol=2e-13)
    rows=[]
    for j,mass in enumerate(m):
        row=dict(mass_MeV=int(mass),dataset_set=o.iloc[j].dataset_set,observed_r=float(r[j]),
            asimov_r=float(a[j]),response_sd=float(s[j]),z=float(z[j]),eligible=bool(r[j]>0),
            nominal_local_p=float(nominal[j]),conditional_local_gaussian=float(local[j]),
            ungated_signed_gaussian=float(norm.sf(z[j])))
        groups={'direct_local':tails(toy_score[:,j],score[j]),'direct_global':tails(directmax,score[j]),
            'gp_global':tails(maxima['profiled_gp'],score[j]),'gp_raw_global':tails(maxima['profiled_gp_raw'],max(0,r[j])),
            'direct_raw_global':tails(rawmax,max(0,r[j]))}
        for name,g in groups.items():row.update({name+'_'+key:v for key,v in g.items()})
        assert groups['gp_global']['k']==old.iloc[j].gp_k
        assert groups['direct_global']['k']==old.iloc[j].direct_k
        assert groups['gp_raw_global']['k']==old.iloc[j].raw_gp_k
        assert groups['direct_global']['k']>=groups['direct_local']['k']
        rows.append(row)
    d=pd.DataFrame(rows);d.to_csv(HERE/'derived/probability_grid.csv',index=False)
    changes=[]
    for j in range(1,len(m)):
        if (r[j]>0)!=(r[j-1]>0):changes.append(dict(left_mass_MeV=int(m[j-1]),right_mass_MeV=int(m[j]),
            left_r=float(r[j-1]),right_r=float(r[j]),left_conditional_p=float(local[j-1]),right_conditional_p=float(local[j])))
    extract=json.loads(bind(EXT/'derived/fit_closure.json').read_text());comparisons=[]
    individual=pd.read_csv(bind(HERE.parent/'v4p9p13_calibration_20260905/summary/observed_calibrated_limits.csv')).set_index(['scope_key','mass_MeV'])
    selected=pd.read_csv(bind(EXT/'derived/fit_summary.csv'))
    for fit in extract['checks']:
        mass=fit['mass_MeV'];fid=fit['fit_id'];scope=selected[selected.fit_id==fid].iloc[0].scope_key
        ref=float(o.set_index('mass_MeV').loc[mass,'profiled_r']) if fit['group']=='combined' else float(individual.loc[(scope,mass),'signed_r_profiled_asymptotic'])
        err=abs(fit['root']-ref);assert err<2e-5
        comparisons.append(dict(fit_id=fid,mass_MeV=mass,extraction_r=fit['root'],scan_r=ref,error=err,
            extraction_nominal_p=float(norm.sf(max(0,fit['root']))),scan_nominal_p=float(norm.sf(max(0,ref)))))
    pd.DataFrame(comparisons).to_csv(HERE/'derived/extraction_consistency.csv',index=False)
    corr=[]
    for left,right in [(65,71),(66,71),(66,72),(66,78),(71,78),(72,78),(78,85)]:
        i=int(np.flatnonzero(m==left)[0]);j=int(np.flatnonzero(m==right)[0])
        corr.append(dict(left_MeV=left,right_MeV=right,gp_combined_rho=float(K[i,j]),
            direct_combined_rho=float(np.corrcoef(toys[:,i],toys[:,j])[0,1])))
    pd.DataFrame(corr).to_csv(HERE/'derived/combined_correlations.csv',index=False)
    result=dict(passed=True,masses=232,n_direct=1000,n_gp=len(maxima['profiled_gp']),
        nonpositive_mass_count=int(np.sum(r<=0)),gate_crossings=changes,
        zero_gp_masses=d.loc[d.gp_global_k==0,'mass_MeV'].tolist(),
        zero_direct_global_masses=d.loc[d.direct_global_k==0,'mass_MeV'].tolist(),
        old_below_1e8_local_floor=d.loc[d.conditional_local_gaussian<1e-8,'mass_MeV'].tolist(),
        nominal_minimum_mass_MeV=int(m[np.argmin(nominal)]),nominal_minimum_p=float(nominal.min()),
        max_extraction_root_error=max(x['error'] for x in comparisons),
        all_existing_probabilities_reproduced=True,statistical_definition_changed=False)
    dump(HERE/'derived/probability_audit.json',result)
    print('Probability audit:',result['masses'],'masses,',len(changes),'sign transitions; max extraction root difference',result['max_extraction_root_error'],flush=True)

def echo_replay():
    rev=pd.read_csv(bind(OLD/'reverse_injection/derived/common_truth_and_signals.csv'))
    pair=pd.read_csv(bind(OLD/'double_peak_injection/derived/generating_spectrum_and_gp_response.csv'))
    oldscan=pd.read_csv(bind(OLD/'reverse_injection/derived/deterministic_scans.csv'))
    oldsummary=json.loads(bind(OLD/'reverse_injection/derived/summary.json').read_text())
    pairsummary=json.loads(bind(OLD/'double_peak_injection/derived/summary.json').read_text())
    for item in oldsummary['source_hashes'].values():
        p=Path(item['path']);assert sha(p)==item['sha256'];bind(p)
    cfg=c.production.load_config(c.production.DEFAULT_CARD);c.production.validate_card(cfg)
    c.production.validate_histogram_inputs(cfg);datasets=c.production.make_datasets(cfg)
    states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
    contract=json.loads(bind(V16/'global/contract.json').read_text())
    for name,digest in contract['source_sha256'].items():
        assert sha(ROOT/name)==digest;SOURCES[name]=digest
    sc=next(q for q in c.production.SCOPES if q[2]==('2021',))
    truth=rev.smooth_truth_counts.to_numpy();assert np.allclose(truth,pair.smooth_truth_counts,rtol=1e-14,atol=1e-7)
    injections=dict(background=np.zeros(len(truth)),inject_66=rev.signal_m66_counts.to_numpy(),
        inject_78=rev.signal_m78_counts.to_numpy(),double_65_78=pair.signal_pair_counts.to_numpy())
    records=[];backgrounds=[];arrays={};deltas=[]
    for mass in range(60,89):
        ctx=core.Context(sc,mass,cfg,datasets,states);p=ctx.parts[0]['p']
        assert len(truth)==len(p.x_full)==422
        assert np.allclose(p.x_full*1000,rev.mass_MeV,atol=1e-10,rtol=0)
        assert np.array_equal(p.y_full,rev.observed_counts)
        records.append(dict(mass_MeV=mass,lane='observed',r=ctx.ofit['profiled']['signed_r'],
            Ahat=ctx.ofit['profiled']['Ahat'],sigma_A=ctx.ofit['profiled']['sigma_A'],
            nll=ctx.ofit['profiled']['free']['nll'],null_nll=ctx.ofit['profiled']['null']['nll'],
            max_score=ctx.ofit['profiled']['max_score'],min_lambda=ctx.ofit['profiled']['min_lambda']))
        for lane,signal in injections.items():
            n=truth+signal;b,L=ctx.retrain(n);mod=c.Profile(b,L,ctx.w,'linear')
            ff=mod.fit(n[ctx.mask]);nn=mod.fit(n[ctx.mask],0.)
            q=2*(nn['nll']-ff['nll']);assert q>=-1e-7
            r=float(np.sign(ff['A'])*np.sqrt(max(0.,q)))
            row=dict(mass_MeV=mass,lane=lane,r=r,Ahat=ff['A'],sigma_A=ff['sigma'],nll=ff['nll'],null_nll=nn['nll'],
                max_score=mod.max_score,min_lambda=min(ff['min_lambda'],nn['min_lambda']))
            records.append(row)
            pref=f'm{mass:03d}__{lane}__'
            arrays.update({pref+k:v for k,v in dict(counts=n[ctx.mask],gp_mean=b,L=L,w=ctx.w,
                free_theta=ff['z'][1:],null_theta=nn['z'],free_lambda=ff['lam'],null_lambda=nn['lam']).items()})
            if mass in [71,72]:
                for x,bb,nnn,sig in zip(p.x_full[ctx.mask]*1000,b,n[ctx.mask],signal[ctx.mask]):
                    backgrounds.append(dict(test_mass_MeV=mass,lane=lane,bin_mass_MeV=float(x),gp_mean=float(bb),
                        generating_counts=float(nnn),injected_counts=float(sig)))
            old=oldscan[(oldscan.mass_MeV==mass)&(oldscan.lane==lane)]
            if len(old):deltas.append(dict(mass_MeV=mass,lane=lane,new_r=r,old_r=float(old.signed_r.iloc[0]),delta=r-float(old.signed_r.iloc[0])))
        print('Echo replay mass',mass,flush=True)
    data=pd.DataFrame(records);table=data.pivot(index='mass_MeV',columns='lane',values='r')
    differences=table[['inject_66','inject_78','double_65_78']].subtract(table.background,axis=0)
    data.to_csv(HERE/'derived/echo_dense_scans.csv',index=False)
    differences.to_csv(HERE/'derived/echo_injection_changes.csv')
    pd.DataFrame(backgrounds).to_csv(HERE/'derived/echo_background_response.csv',index=False)
    pd.DataFrame(deltas).to_csv(HERE/'derived/echo_legacy_comparison.csv',index=False)
    np.savez_compressed(HERE/'derived/echo_likelihood_components.npz',**arrays)
    outcomes={str(m):{str(k):float(v) for k,v in table.loc[m].items()} for m in [65,66,71,72,78,80,85]}
    changes={str(m):{str(k):float(v) for k,v in differences.loc[m].items()} for m in [65,66,71,72,78,80,85]}
    dump(HERE/'derived/echo_summary.json',dict(passed=True,new_random_toys=0,new_unblinded_events=0,
        masses=29,deterministic_profile_tests=116,observed_reconstructions=29,
        truth_description='same archived smooth 2021 spectrum trained outside60--86MeV, kernel anchored66MeV',
        injection_yields={k:float(v.sum()) for k,v in injections.items()},
        max_current_versus_legacy_r_difference=max(abs(x['delta']) for x in deltas),
        max_profile_score=float(data.max_score.max()),absolute_roots=outcomes,injection_changes=changes,
        scope='conditional mechanism demonstration; selected truth and amplitudes; no one-vs-two-particle probability'))
    print('Echo replay complete; max legacy discrepancy',max(abs(x['delta']) for x in deltas),flush=True)

def main():
    for folder in ['derived','figures','note','provenance','qa','review']:(HERE/folder).mkdir(exist_ok=True)
    bind(__file__);bind(HERE/'PROTOCOL.md')
    probabilities();echo_replay()
    dump(HERE/'provenance/numerical_inputs.json',dict(passed=True,input_sha256=SOURCES,
        output_sha256={str(p.relative_to(ROOT)):sha(p) for p in (HERE/'derived').iterdir() if p.is_file()}))
if __name__=='__main__':main()
