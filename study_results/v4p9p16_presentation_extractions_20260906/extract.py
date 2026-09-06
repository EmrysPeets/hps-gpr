#!/usr/bin/env python3
"""Reconstruct selected dense observed fits; display binning never enters inference."""
from pathlib import Path
import hashlib, json, os, sys, time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V16=HERE.parent/'v4p9p16_combined_global_20260906'
V13=HERE.parent/'v4p9p13_calibration_20260905'
sys.path.insert(0,str(V16))
import run_combined as parent
core,c,np,pd=parent.core,parent.c,parent.np,parent.pd
from scipy.stats import norm

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def select(frame, n, sigma, column='r'):
    rows=frame.sort_values('mass_MeV').reset_index(drop=True)
    vals=rows[column].to_numpy(float)
    candidates=[i for i,v in enumerate(vals) if v>0 and (i==0 or v>=vals[i-1])
                and (i==len(vals)-1 or v>=vals[i+1])]
    chosen=[]
    for i in sorted(candidates,key=lambda i:(-vals[i],int(rows.iloc[i].mass_MeV))):
        m=int(rows.iloc[i].mass_MeV)
        if all(abs(m-q)>2.25*max(sigma(m),sigma(q)) for q in chosen):chosen.append(m)
        if len(chosen)==n:break
    if len(chosen)!=n:raise RuntimeError('Insufficient separated peaks')
    return chosen

def scope(keys):return next(s for s in c.production.SCOPES if s[2]==tuple(keys))

def grouping(edges,mask,sigma):
    """Integer native-bin groups, anchored at the histogram's first edge."""
    step=np.diff(edges);assert np.allclose(step,step[0],rtol=0,atol=1e-9)
    k=max(1,int(np.floor(.5*sigma/step[0]+.5)))
    out=[];be=[]
    for start in range(0,len(mask)-k+1,k):
        idx=np.arange(start,start+k)
        if np.all(mask[idx]):out.append(idx);be.append([edges[start],edges[start+k]])
    W=np.zeros((len(out),len(mask)))
    for j,idx in enumerate(out):W[j,idx]=1.
    if len(out)<3:raise RuntimeError('Display has fewer than three bins')
    return W,np.array(be),k

def common_group(parts):
    low=max(p['edges'][np.flatnonzero(p['mask'])[0]] for p in parts)
    high=min(p['edges'][np.flatnonzero(p['mask'])[-1]+1] for p in parts)
    start=int(np.ceil((low-36)/1.25-1e-8));stop=int(np.floor((high-36)/1.25+1e-8))
    edges=36+1.25*np.arange(start,stop+1);maps=[]
    if len(edges)<4:raise RuntimeError('Insufficient common whole-bin support')
    for p in parts:
        W=np.zeros((len(edges)-1,len(p['mask'])))
        for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            a=np.flatnonzero(np.isclose(p['edges'],lo,rtol=0,atol=1e-8))
            b=np.flatnonzero(np.isclose(p['edges'],hi,rtol=0,atol=1e-8))
            if len(a)!=1 or len(b)!=1:raise RuntimeError('Common edges do not align')
            assert np.all(p['mask'][a[0]:b[0]])
            W[i,a[0]:b[0]]=1.
        maps.append(W)
    return maps,np.column_stack((edges[:-1],edges[1:]))

def main():
    for name in ('derived','figures','note','provenance','qa','review'):(HERE/name).mkdir(parents=True,exist_ok=True)
    contract=json.loads((V16/'global/contract.json').read_text())
    sources=contract['source_sha256']
    for p,h in sources.items():
        if sha(ROOT/p)!=h:raise RuntimeError('Changed parent input: '+p)
    cfg=c.production.load_config(c.production.DEFAULT_CARD)
    c.production.result_config=cfg
    c.production.validate_card(cfg);c.production.validate_histogram_inputs(cfg)
    datasets=c.production.make_datasets(cfg)
    states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
    dense=pd.read_csv(V13/'summary/observed_calibrated_limits.csv')
    dense=dense.set_index(['scope_key','mass_MeV'])
    joint=pd.read_csv(V16/'global/observed.csv').set_index('mass_MeV')
    plan=[];rank_records=[]
    def add(key,m,reason,rank=0):
        fid=f'{key}_m{m:03d}'
        if not any(x['fit_id']==fid for x in plan):plan.append(dict(fit_id=fid,group=key,mass_MeV=m,reason=reason,rank=rank))
    for year in ('2015','2016','2021'):
        sc=scope((year,));frame=dense.loc[sc[0]].reset_index().rename(columns={'signed_r_profiled_asymptotic':'r'})
        chosen=select(frame,2,lambda m:float(datasets[year].sigma(m/1000))*1000)
        for rank,m in enumerate(chosen,1):add(year,m,'individual positive local maximum',rank)
        m=int(frame.loc[frame.r.idxmin(),'mass_MeV']);add(year,m,'individual deepest deficit')
        rank_records.append(dict(group=year,positive_peaks_MeV=chosen,deepest_deficit_MeV=m))
    frame=joint.reset_index().rename(columns={'profiled_r':'r'})
    sigma=lambda m:max(float(datasets[k].sigma(m/1000))*1000 for k in parent.scope_for(m)[2])
    chosen=select(frame,2,sigma)
    for rank,m in enumerate(chosen,1):add('combined',m,'full-union positive local maximum',rank)
    multi=select(frame[frame.n_active>=2],2,sigma)
    for rank,m in enumerate(multi,1):add('combined',m,'multi-dataset positive local maximum',rank)
    m=int(frame.loc[frame.r.idxmin(),'mass_MeV']);add('combined',m,'combined deepest deficit')
    rank_records.append(dict(group='combined',positive_peaks_MeV=chosen,multidataset_positive_peaks_MeV=multi,deepest_deficit_MeV=m))
    for m in (76,83):add('combined',m,'stress-centering diagnostic')
    expected={'2015':[51,21],'2016':[90,117],'2021':[78,65],'combined':[66,21]}
    assert all(x['positive_peaks_MeV']==expected[x['group']] for x in rank_records)
    assert multi==[66,92]
    dump(HERE/'derived/selection.json',dict(rule='Observed profiled signed-root local maxima; frozen resolution separation',rankings=rank_records,fits=plan))
    rows=[];consistency=[];binrows=[];closure=[];arrays={};fitdata={};infos=[];retained=[]
    for target in plan:
        t0=time.time();m=target['mass_MeV'];fid=target['fit_id'];group=target['group']
        sc=parent.scope_for(m) if group=='combined' else scope((group,))
        ctx=core.Context(sc,m,cfg,datasets,states)
        fit=ctx.ofit['profiled'];free=fit['free'];null=fit['null']
        eps=fit['Ahat']/ctx.conversion;se=fit['sigma_A']/ctx.conversion
        ref=float(joint.loc[m,'profiled_r']) if group=='combined' else float(dense.loc[(sc[0],m),'signed_r_profiled_asymptotic'])
        if abs(fit['signed_r']-ref)>2e-5:raise RuntimeError('Dense observed root mismatch')
        if group=='combined' and pd.notna(joint.loc[m,'profiled_Ahat_window']):
            assert np.isclose(fit['Ahat'],joint.loc[m,'profiled_Ahat_window'],rtol=1e-9,atol=1e-5)
        ref_ul=float(joint.loc[m,'profiled_eps2_ee_raw']) if group=='combined' else float(dense.loc[(sc[0],m),'eps2_profiled_asymptotic_ee_raw'])
        assert abs(fit['A90']/ctx.conversion/ref_ul-1)<1e-8
        fitparts=[];start=0;ind_nll=0.;joint_nll=free['nll'];sum_signal=0.;sum_info=0.
        for j,part in enumerate(ctx.parts):
            key=part['key'];p=part['p'];mask=p.blind_mask;n=int(mask.sum());end=start+n
            edges=np.asarray(p.edges_full)*1000
            obs=np.asarray(p.y_full,float);prior=np.asarray(p.mu_full,float)
            fitted=np.full(len(obs),np.nan);fitted[mask]=free['bfit'][start:end]
            total=np.full(len(obs),np.nan);total[mask]=free['lam'][start:end]
            nullbg=np.full(len(obs),np.nan);nullbg[mask]=null['bfit'][start:end]
            sf=ctx.signal[ctx.offsets[j]:ctx.offsets[j+1]]*fit['Ahat']
            assert np.allclose((fitted+sf)[mask],total[mask],rtol=2e-12,atol=2e-6)
            sum_signal+=float(sf[mask].sum())
            signal_unit=ctx.w[start:end]*ctx.conversion
            L=ctx.L[start:end,start:end]
            C=L@L.T
            conversion=float(signal_unit.sum())
            mod=c.Profile(ctx.b[start:end],L,signal_unit/conversion,'linear')
            ff=mod.fit(ctx.obs[start:end]);nn=mod.fit(ctx.obs[start:end],0.)
            rr=float(np.sign(ff['A'])*np.sqrt(max(0,2*(nn['nll']-ff['nll']))))
            ind_nll+=ff['nll']
            if group=='combined' and (scope((key,))[0],m) in dense.index:
                assert abs(rr-float(dense.loc[(scope((key,))[0],m),'signed_r_profiled_asymptotic']))<2e-5
            cc=dict(fit_id=fid,dataset=key,mass_MeV=m,individual_eps2_hat=ff['A']/conversion,
                individual_sigma_eps2=ff['sigma']/conversion,individual_r=rr,common_eps2_hat=eps,
                common_sigma_eps2=se,individual_nll=ff['nll'],individual_score=ff['score'])
            consistency.append(cc)
            info=float(signal_unit@np.linalg.solve(np.diag(ctx.b[start:end])+C,signal_unit));sum_info+=info
            infos.append(dict(fit_id=fid,mass_MeV=m,dataset=key,information=info))
            W,be,k=grouping(edges,mask,float(p.sigma_val)*1000)
            item=dict(key=key,edges=edges,mask=mask,observed=obs,gp_mean=prior,
                profiled_background=fitted,signal=sf,total=total,null_background=nullbg,sigma=float(p.sigma_val)*1000)
            fitparts.append(item)
            arrays.update({fid+'__'+key+'__'+name:value for name,value in item.items() if isinstance(value,np.ndarray)})
            arrays.update({fid+'__'+key+'__'+name:value for name,value in dict(
                display_map=W,display_edges=be,fit_covariance=C,fit_factor=L,
                signal_unit=signal_unit,fit_gp_mean=ctx.b[start:end],fit_counts=ctx.obs[start:end],
                common_free_theta=free['z'][1+start:1+end],common_null_theta=null['z'][start:end],
                independent_free_theta=ff['z'][1:],independent_null_theta=nn['z'],
                independent_background=ff['bfit'],independent_total=ff['lam'],
                independent_null_background=nn['bfit']).items()})
            def binned_map(item,W,be,panel):
                values={name:W@np.nan_to_num(item[name],nan=0.) for name in
                    ('observed','gp_mean','profiled_background','signal','total','null_background')}
                for i,(lo,hi) in enumerate(be):binrows.append(dict(fit_id=fid,panel=panel,bin=i,low_MeV=lo,high_MeV=hi,
                    **{name:float(val[i]) for name,val in values.items()}))
            binned_map(item,W,be,key)
            one=dict(**target,dataset=key,dataset_set='+'.join(ctx.keys),sigma_MeV=float(p.sigma_val)*1000,
                fit_low_MeV=float(edges[np.flatnonzero(mask)[0]]),fit_high_MeV=float(edges[np.flatnonzero(mask)[-1]+1]),
                n_fit_bins=n,display_native_bins_per_bin=k,display_bin_width_MeV=float(be[0,1]-be[0,0]),
                display_bins=len(be),display_observed_fraction=float((W@obs).sum()/obs[mask].sum()),
                signed_r=float(fit['signed_r']),raw_reference_one_sided_p=float(norm.sf(abs(fit['signed_r']))),
                eps2_hat=eps,sigma_eps2=se,signal_window=float(sf[mask].sum()),signal_full=float(sf.sum()),
                signal_yield_per_eps2_window=conversion,signal_yield_per_eps2_full=float(ctx.signal[ctx.offsets[j]:ctx.offsets[j+1]].sum()*ctx.conversion),
                prediction_state_sha256=next(x['prediction_state_sha256'] for x in ctx.ledger if x['dataset']==key),
                profile_score=free['score'],min_lambda=free['min_lambda'],scope_key=sc[0],
                at_search_endpoint=m in (c.production.EXPECTED_DATASET_GRIDS[key][0],c.production.EXPECTED_DATASET_GRIDS[key][-1]))
            rows.append(one);start=end
        assert start==len(ctx.obs)
        assert np.isclose(sum_signal,fit['Ahat'],rtol=1e-12,atol=1e-6)
        if len(fitparts)>1:
            maps,be=common_group(fitparts)
            values={name:sum(W@np.nan_to_num(part[name],nan=0.) for W,part in zip(maps,fitparts)) for name in
                ('observed','gp_mean','profiled_background','signal','total','null_background')}
            for i,(lo,hi) in enumerate(be):binrows.append(dict(fit_id=fid,panel='sum',bin=i,low_MeV=lo,high_MeV=hi,
                **{name:float(val[i]) for name,val in values.items()}))
            for W,part in zip(maps,fitparts):
                arrays[fid+'__'+part['key']+'__common_map']=W
                retained.append(dict(fit_id=fid,dataset=part['key'],mass_MeV=m,
                    common_low_MeV=float(be[0,0]),common_high_MeV=float(be[-1,1]),
                    common_display_bins=len(be),native_fit_bins=int(part['mask'].sum()),
                    retained_native_bins=int(W.sum()),native_bin_fraction=float(W.sum()/part['mask'].sum()),
                    observed_fraction=float((W@part['observed']).sum()/part['observed'][part['mask']].sum()),
                    fitted_signal_fraction=float((W@part['signal']).sum()/part['signal'][part['mask']].sum()),
                    fitted_signal_window=float(part['signal'][part['mask']].sum()),
                    displayed_signal_sum=float((W@part['signal']).sum())))
            arrays[fid+'__sum__display_edges']=be
        qcompat=2*(joint_nll-ind_nll)
        assert qcompat>=-1e-7
        closure.append(dict(**target,root=float(fit['signed_r']),reference_root=ref,root_error=float(fit['signed_r']-ref),
            sum_signal_window=sum_signal,Ahat=fit['Ahat'],conversion=ctx.conversion,eps2_hat=eps,
            sigma_eps2=se,individual_common_deviance=max(0.,qcompat),compatibility_df=len(ctx.keys)-1,
            common_nll=joint_nll,sum_independent_nll=ind_nll,score=free['score'],null_score=null['score'],
            min_lambda=free['min_lambda'],sum_information=sum_info,passed=True))
        print(fid, 'r=',round(fit['signed_r'],5),'eps=',eps,'q_compat=',qcompat,'seconds=',round(time.time()-t0,2),flush=True)
    pd.DataFrame(rows).to_csv(HERE/'derived/fit_summary.csv',index=False)
    pd.DataFrame(consistency).to_csv(HERE/'derived/dataset_consistency.csv',index=False)
    pd.DataFrame(binrows).to_csv(HERE/'derived/display_bins.csv',index=False)
    pd.DataFrame(infos).to_csv(HERE/'derived/information.csv',index=False)
    pd.DataFrame(retained).to_csv(HERE/'derived/common_display_retention.csv',index=False)
    np.savez_compressed(HERE/'derived/fit_arrays.npz',**arrays)
    dump(HERE/'derived/fit_closure.json',dict(passed=True,checks=closure,n_fits=len(plan),n_dataset_fits=len(rows),
        max_root_error=max(abs(x['root_error']) for x in closure),new_toys=0,new_unblinded_data=False))
    sources={**sources,**{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),HERE/'PROTOCOL.md',
        V16/'global/observed.csv',V16/'MANIFEST.csv',HERE.parent/'v4p9p16_deficit_extension_20260906/MANIFEST.csv')}}
    output_names=['selection.json','fit_summary.csv','dataset_consistency.csv','display_bins.csv',
        'information.csv','common_display_retention.csv','fit_arrays.npz','fit_closure.json']
    dump(HERE/'provenance/extraction.json',dict(input_sha256=sources,
        output_sha256={str((HERE/'derived'/name).relative_to(ROOT)):sha(HERE/'derived'/name) for name in output_names},passed=True))

if __name__=='__main__':main()
