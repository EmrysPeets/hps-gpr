"""Independent bookkeeping and sampled likelihood checks; one numerical thread.

This validator does not generate study toys or alter fit results. Regenerated
seed draws are checked against saved arrays and discarded. The scalar oracle
uses a one-dimensional score root; GP likelihood checks use SciPy BFGS rather
than the inherited safeguarded Newton implementation.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, ast, hashlib, json, time
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import injection_core as I


def require(value,message):
    if not value:
        raise AssertionError(message)


def independent_probabilities(m):
    a=I.MC.MC[m]
    meta=json.loads(str(a['metadata']))
    require(meta['stats']['underflow']==0,'Unexpected source underflow needs explicit bookkeeping')
    require(abs(meta['sumw']-a['sumw'].sum()-meta['stats']['overflow'])<1e-9,
            'Source overflow bookkeeping changed')
    lo=a['edges_GeV'][:-1];hi=a['edges_GeV'][1:]
    edges=I.D['edges']
    overlap=np.maximum(0,np.minimum(edges[1:,None],hi[None,:])-
                          np.maximum(edges[:-1,None],lo[None,:]))
    p=(overlap/(hi-lo)[None,:])@a['sumw']/float(meta['sumw'])
    below=float(np.dot(np.clip((edges[0]-lo)/(hi-lo),0,1),a['sumw'])/meta['sumw'])
    above=float(np.dot(np.clip((hi-edges[-1])/(hi-lo),0,1),a['sumw'])/meta['sumw'])
    above+=float(meta['sumw']-a['sumw'].sum())/meta['sumw']
    return p,np.r_[below,p,above]


def objective(n,b,J,pen,z):
    lam=b+J@z
    if np.any(lam<=0):
        return float('inf'),np.zeros_like(z)
    pos=n>0;t=(lam[pos]-n[pos])/n[pos]
    value=float(np.sum(n[pos]*(t-np.log1p(t)))+np.sum(lam[~pos])+
                .5*np.dot(pen*z,z))
    gradient=J.T@(1-n/lam)+pen*z
    return value,gradient


def scalar_oracle(n,b,s):
    active=s>0
    lower=-float(np.min(b[active]/s[active]))
    lower+=max(1e-7,abs(lower)*1e-12)
    score=lambda a:float(np.dot(s,1-n/(b+a*s)))
    upper=max(1000.,float(np.sum(n)/np.sum(s)))
    while score(upper)<0:upper*=2
    require(score(lower)<0,'Oracle lower endpoint must bracket root')
    ahat=brentq(score,lower,upper,xtol=1e-8,rtol=1e-13)
    lam=b+ahat*s
    sigma=1/np.sqrt(np.sum(n*s*s/(lam*lam)))
    return float(ahat),float(sigma)


def verify_extension():
    """Compare the retained release with toys 0--19, without tolerances."""
    previous=I.B/'history/20_toy_release'
    if not previous.exists():
        return dict(checked=False,reason='The original release archive is absent.')
    old_protocol=json.loads((previous/'protocol.json').read_text())
    require(old_protocol['toys_per_cell']==20,'Reference release must contain 20 toys')
    require(old_protocol['seed']==I.SEED,'Master seed changed')
    code_blocks={}
    for folder in (previous,I.B):
        parsed=ast.parse((folder/'scripts/injection_core.py').read_text())
        blocks=[]
        for node in parsed.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('source','Context','run_mass'):
                if node.name=='run_mass':
                    node.args.defaults[0]=ast.Constant(value=20)
                blocks.append(ast.dump(node,include_attributes=False))
        code_blocks[str(folder)]=hashlib.sha256('\n'.join(blocks).encode()).hexdigest()
    require(len(set(code_blocks.values()))==1,'Scientific fit or generation code changed')
    for rec in json.loads((previous/'provenance/input_hashes.json').read_text()):
        require(hashlib.sha256((I.B/rec['bundled']).read_bytes()).hexdigest()==rec['sha256'],
                'Original fixed input changed')
    checks=[]
    for m in I.MASSES:
        old=previous/'results/checkpoints';new=I.B/'results/checkpoints'
        marker=json.loads((old/f'm{m:03d}.json').read_text())
        require(marker['toys']==20 and marker['complete'],'Incomplete original checkpoint')
        for name,digest in marker['output_hashes'].items():
            require(hashlib.sha256((old/name).read_bytes()).hexdigest()==digest,
                    'Original release output hash mismatch')
        old_draws=dict(np.load(old/f'm{m:03d}_draws.npz'))
        new_draws=dict(np.load(new/f'm{m:03d}_draws.npz'))
        for name,value in old_draws.items():
            current=new_draws[name][:20] if name in ('backgrounds','injections') else new_draws[name]
            require(np.array_equal(value,current),'Original draw or fixed model array changed: '+name)
        original=(old/f'm{m:03d}_toys.csv').read_bytes()
        prefix=b''.join((new/f'm{m:03d}_toys.csv').read_bytes().splitlines(keepends=True)[:561])
        require(original==prefix,'Original fit CSV rows changed')
        require((old/f'm{m:03d}_asimov.csv').read_bytes()==(new/f'm{m:03d}_asimov.csv').read_bytes(),
                'Deterministic control changed')
        old_backgrounds={I.array_hash(v) for v in old_draws['backgrounds']}
        added_backgrounds={I.array_hash(v) for v in new_draws['backgrounds'][20:]}
        require(len(added_backgrounds)==20 and not old_backgrounds & added_backgrounds,
                'Repeated background draw in added toys')
        for level in range(len(I.LEVELS)):
            old_signals={I.array_hash(v) for v in old_draws['injections'][:,level]}
            added_signals={I.array_hash(v) for v in new_draws['injections'][20:,level]}
            require(len(added_signals)==20 and not old_signals & added_signals,
                    'Repeated signal draw in added toys')
        checks.append(dict(mass_MeV=m,original_fit_csv_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
            original_checkpoint_outputs_verified=True,first20_fit_rows_byte_identical=True,
            first20_draws_identical=True,added_draws_disjoint=True,asimov_identical=True))
    result=dict(passed=True,checked=True,original_toys_per_cell=20,added_toys_per_cell=20,
        total_toys_per_cell=I.TOYS,first20_toy_indices=[0,19],added_toy_indices=[20,39],
        preserved_extraction_rows=6160,new_extraction_rows=6160,
        scientific_code_sha256=code_blocks[str(I.B)],
        scientific_code_note='AST of source, Context and run_mass unchanged except the default toy count.',
        independent_stream_definition='Unchanged SeedSequence([seed,mass,toy,injected_N]); added toy indices 20--39.',
        original_fixed_input_hashes_verified=True,method='Deterministic full rerun with exact retained-release comparisons',
        mass_checks=checks)
    I.write_json(I.B/'qa/toy_extension.json',result)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pilot',action='store_true')
    args=ap.parse_args();start=time.monotonic()
    folder=I.B/('qa/pilot' if args.pilot else 'results/checkpoints')
    markers=sorted(folder.glob('m[0-9][0-9][0-9].json'))
    require(bool(markers),'No completed mass checkpoints')
    masses=[int(p.stem[1:]) for p in markers]
    require(set(masses)<=set(I.MASSES),'Non-native mass in checkpoints')
    if not args.pilot:require(masses==list(I.MASSES),'Production mass grid incomplete')
    all_rows=[];rebin_errors=[];draw_counts=0;row_count=0;numerical=[]
    for marker in markers:
        meta=json.loads(marker.read_text());m=int(marker.stem[1:]);nt=int(meta['toys'])
        require(meta['complete'],'Incomplete checkpoint')
        if not args.pilot:
            require(nt==I.TOYS,f'Production requires exactly {I.TOYS} toys')
            require(meta['dependency_signature']==I.dependency_signature(),'Checkpoint dependency mismatch')
            require(len(meta['output_hashes'])==3,'Checkpoint output manifest incomplete')
            for name,digest in meta['output_hashes'].items():
                require(hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,
                        'Checkpoint output hash mismatch')
        saved=dict(np.load(folder/f'm{m:03d}_draws.npz'))
        frame=pd.read_csv(folder/f'm{m:03d}_toys.csv',float_precision='round_trip')
        require(len(frame)==28*nt,'Unexpected number of extraction rows')
        require(not frame.duplicated(['toy','injected_N','method','control']).any(),
                'Duplicate extraction identity')
        require(np.array_equal(saved['levels'],I.LEVELS),'Injection strengths changed')
        require(np.array_equal(saved['truth'],I.TRUTH),'Generator truth changed')
        require(np.array_equal(saved['edges_GeV'],I.D['edges']),'Analysis edges changed')
        require(saved['backgrounds'].shape==(nt,len(I.TRUTH)),'Background shape')
        require(saved['injections'].shape==(nt,4,len(I.TRUTH)+2),'Signal categories shape')
        require(np.issubdtype(saved['injections'].dtype,np.integer),'Integer signal draws')
        require(np.issubdtype(saved['backgrounds'].dtype,np.integer),'Integer background draws')
        pp,cc=independent_probabilities(m)
        rebin_errors.append(dict(mass_MeV=m,
            max_probability_error=float(np.max(abs(pp-saved['probability']))),
            max_category_error=float(np.max(abs(cc-saved['categories'])))))
        require(np.max(abs(cc-saved['categories']))<3e-14,'CDF rebin differs from overlap integral')
        require(abs(saved['categories'].sum()-1)<1e-12,'Categories not normalized')
        require(np.array_equal(saved['injections'].sum(axis=2),
                    np.broadcast_to(np.array(I.LEVELS),(nt,4))),'Fixed injected totals violated')
        p,_,center,_=I.source(m)
        contexts={method:I.Context(m,method,center,p) for method in I.METHODS}
        for method,ctx in contexts.items():
            key='pole_mask' if method=='pole_centered' else 'core_mask'
            require(np.array_equal(saved[key],ctx.mask),'Saved mask changed')
        for toy in range(nt):
            background=saved['backgrounds'][toy]
            rng=np.random.default_rng(np.random.SeedSequence([I.SEED,m,toy,0]))
            require(np.array_equal(background,rng.poisson(I.TRUTH)),'Background seed mismatch')
            for index,N in enumerate(I.LEVELS):
                draw=saved['injections'][toy,index]
                rng=np.random.default_rng(np.random.SeedSequence([I.SEED,m,toy,N]))
                require(np.array_equal(draw,rng.multinomial(N,saved['categories'])),
                        'Signal seed mismatch')
                draw_counts+=1
            for N in (0,*I.LEVELS):
                signal=(np.zeros_like(background) if N==0 else
                        saved['injections'][toy,list(I.LEVELS).index(N),1:-1])
                counts=background+signal
                q=frame[(frame.toy==toy)&(frame.injected_N==N)]
                require(len(q)==(4 if N==0 else 6),'Paired row family incomplete')
                require(q.background_draw_hash.eq(I.array_hash(background)).all(),
                        'Background hash pairing mismatch')
                require(q.injected_counts_hash.eq(I.array_hash(counts)).all(),
                        'Injected spectrum hash pairing mismatch')
                require(np.all(q.actual_support==signal.sum()),'Support count mismatch')
                require(np.all(q.actual_outside_support==N-signal.sum()),'Outside count mismatch')
                for method,ctx in contexts.items():
                    qq=q[q.method==method]
                    require(np.all(qq.actual_window==signal[ctx.mask].sum()),'Window count mismatch')
                    require(np.all(qq.actual_training==signal[~ctx.mask].sum()),'Training count mismatch')
                    require(np.max(abs(qq.expected_window_fraction-ctx.S.sum()))<1e-13,
                            'Window normalization changed')
        require(np.all(np.isfinite(frame[['Ahat','sigma_A','pull','q_true','fit_score','min_lambda']])),
                'Nonfinite fit statistic')
        require(np.all(frame.sigma_A>0),'Nonpositive fit uncertainty')
        require(float(frame.fit_score.max())<3e-5,'Fit score gate')
        require(float(frame.min_lambda.min())>0,'Fit positivity gate')
        require(np.max(abs(frame.pull-(frame.Ahat-frame.injected_N)/frame.sigma_A))<1e-12,
                'Signed pull identity')
        require(np.max(abs(frame.q_true-np.maximum(0,2*(frame.true_nll-frame.free_nll))))<1e-10,
                'Profile ratio identity')
        require((frame.true_nll-frame.free_nll).min()>-1e-6,'Profile nesting')
        if m in (60,260):
            for sample_toy in ((0,20) if nt>20 else (0,)):
                background=saved['backgrounds'][sample_toy]
                signal=saved['injections'][sample_toy,list(I.LEVELS).index(30000),1:-1]
                counts=background+signal
                for method,ctx in contexts.items():
                    b,L,diagnostic=ctx.predict(counts)
                    b0,cov0=I.C.predict(I.D['x'],counts,ctx.mask,ctx.const,ctx.ls)
                    L0,_=I.C.factor_cov(cov0,b0)
                    mean_error=float(np.max(abs(b-b0)/b0))
                    covariance_error=float(np.max(abs(L@L.T-L0@L0.T))/max(1.,np.max(abs(L0@L0.T))))
                    require(mean_error<1e-12 and covariance_error<1e-11,
                            'Cached GP differs from parent prediction')
                    row=frame[(frame.toy==sample_toy)&(frame.injected_N==30000)&
                              (frame.method==method)&(frame.control=='contaminated_gp')].iloc[0]
                    scale=1/np.sqrt(np.sum(ctx.S**2/b));J=np.column_stack([scale*ctx.S,L])
                    pen=np.r_[0.,np.ones(L.shape[1])];n=counts[ctx.mask].astype(float)
                    result=minimize(lambda z:objective(n,b,J,pen,z),np.zeros(J.shape[1]),
                                    jac=True,method='BFGS',options={'gtol':2e-7,'maxiter':300})
                    value,gradient=objective(n,b,J,pen,result.x)
                    score=float(np.max(abs(gradient)));lam=b+J@result.x
                    hessian=(J.T*(n/lam**2))@J+np.diag(pen)
                    sigma=float(scale*np.sqrt(np.linalg.inv(hessian)[0,0]))
                    ahat=float(result.x[0]*scale)
                    require(score<3e-5,'Independent BFGS score failure')
                    require(abs(ahat-row.Ahat)/row.sigma_A<2e-5,'Independent free yield mismatch')
                    require(abs(value-row.free_nll)<2e-6,'Independent minimum NLL mismatch')
                    require(abs(sigma/row.sigma_A-1)<2e-6,'Independent Hessian sigma mismatch')
                    fixed=minimize(lambda z:objective(n,b+30000*ctx.S,L,np.ones(L.shape[1]),z),
                                   np.zeros(L.shape[1]),jac=True,method='BFGS',
                                   options={'gtol':2e-7,'maxiter':300})
                    fixed_value,fixed_gradient=objective(n,b+30000*ctx.S,L,np.ones(L.shape[1]),fixed.x)
                    fixed_score=float(np.max(abs(fixed_gradient)))
                    require(fixed_score<3e-5,'Independent fixed-truth BFGS score failure')
                    require(abs(fixed_value-row.true_nll)<2e-6,'Independent true-N NLL mismatch')
                    orow=frame[(frame.toy==sample_toy)&(frame.injected_N==30000)&
                               (frame.method==method)&(frame.control=='known_background')].iloc[0]
                    oa,os=scalar_oracle(n,I.TRUTH[ctx.mask],ctx.S)
                    require(abs(oa-orow.Ahat)/orow.sigma_A<1e-6,'Independent oracle yield mismatch')
                    require(abs(os/orow.sigma_A-1)<1e-7,'Independent oracle sigma mismatch')
                    numerical.append(dict(mass_MeV=m,toy=sample_toy,method=method,GP_mean_relative_error=mean_error,
                        GP_covariance_relative_error=covariance_error,independent_BFGS_score=score,
                        independent_fixed_truth_score=fixed_score,
                        free_yield_difference_in_sigma=float((ahat-row.Ahat)/row.sigma_A),
                        NLL_difference=float(value-row.free_nll),Hessian_sigma_ratio=float(sigma/row.sigma_A),
                        fixed_truth_NLL_difference=float(fixed_value-row.true_nll),
                        oracle_yield_difference_in_sigma=float((oa-orow.Ahat)/orow.sigma_A),
                        oracle_sigma_ratio=float(os/orow.sigma_A)))
        row_count+=len(frame);all_rows.append(frame)
    all_rows=pd.concat(all_rows,ignore_index=True)
    aggregate_checks={}
    extension_check={}
    if not args.pilot:
        require(row_count==len(I.MASSES)*I.TOYS*28,'Production extraction row count')
        require(draw_counts==len(I.MASSES)*I.TOYS*len(I.LEVELS),'Production draw count')
        summary=pd.read_csv(I.B/'results/summary.csv',float_precision='round_trip')
        paired=pd.read_csv(I.B/'results/paired.csv',float_precision='round_trip')
        require(len(summary)==308,'Summary row count')
        require(len(paired)==44,'Paired summary row count')
        for _,row in summary.iterrows():
            g=all_rows[(all_rows.mass_MeV==row.mass_MeV)&
                       (all_rows.injected_N==row.injected_N)&
                       (all_rows.method==row.method)&(all_rows.control==row.control)]
            require(len(g)==I.TOYS,'Summary cell incomplete')
            for name,value in [('mean_A',g.Ahat.mean()),('sd_A',g.Ahat.std(ddof=1)),
                    ('mean_bias',g.Ahat.mean()-row.injected_N),('pull_mean',g.pull.mean()),
                    ('pull_width',g.pull.std(ddof=1))]:
                require(np.isclose(row[name],value,rtol=1e-11,atol=1e-8),'Summary '+name+' mismatch')
            for level in ('68','95'):
                require(row['profile'+level+'_count']==g['profile_contains'+level].sum(),
                        'Containment count mismatch')
        for _,row in paired.iterrows():
            g=all_rows[(all_rows.mass_MeV==row.mass_MeV)&
                       (all_rows.control=='contaminated_gp')]
            null=g[g.injected_N==0].pivot(index='toy',columns='method',values='Ahat')
            inject=g[g.injected_N==row.injected_N].pivot(index='toy',columns='method',values='Ahat')
            delta=inject.core_shifted-inject.pole_centered
            require(np.isclose(row.mean_core_minus_pole,delta.mean(),rtol=1e-11,atol=1e-8),
                    'Paired method difference mismatch')
            inc=(inject-null)/row.injected_N
            require(np.isclose(row.mean_incremental_recovery_pole,inc.pole_centered.mean(),rtol=1e-11),
                    'Paired pole increment mismatch')
            require(np.isclose(row.mean_incremental_recovery_core,inc.core_shifted.mean(),rtol=1e-11),
                    'Paired shifted increment mismatch')
        asimov=pd.read_csv(I.B/'results/asimov.csv',float_precision='round_trip')
        require(len(asimov)==308,'Asimov row count')
        oracle=asimov[(asimov.control=='known_background')&(asimov.injected_N>0)]
        oracle_error=float(np.max(abs(oracle.Ahat/oracle.injected_N-1)))
        require(oracle_error<1e-6,'Asimov oracle recovery failure')
        null=asimov[(asimov.control=='contaminated_gp')&(asimov.injected_N==0)].set_index(['mass_MeV','method']).Ahat
        mechanism={}
        for method in I.METHODS:
            for control in ('clean_sidebands','contaminated_gp'):
                selected=asimov[(asimov.method==method)&(asimov.control==control)&(asimov.injected_N==30000)]
                recovery=np.array([(row.Ahat-null.loc[(row.mass_MeV,method)])/30000
                                   for _,row in selected.iterrows()])
                mechanism[method+'_'+control]=[float(recovery.min()),float(recovery.max())]
        aggregate_checks=dict(summary_cells_verified=308,paired_cells_verified=44,
            Asimov_oracle_max_relative_yield_error=oracle_error,
            deterministic_30000_paired_null_subtracted_recovery=mechanism)
        extension_check=verify_extension()
    result=dict(passed=True,stage='pilot' if args.pilot else 'production',masses_MeV=masses,
        extraction_rows=row_count,exactN_seed_draws_verified=draw_counts,
        profile_and_pull_identities_verified=True,full_spectrum_pairing_verified=True,
        histogram_overlap_rebin_checks=rebin_errors,sampled_likelihood_checks=numerical,
        aggregate_checks=aggregate_checks,toy_extension_check=extension_check,
        maximum_saved_fit_score=float(all_rows.fit_score.max()),
        minimum_saved_expectation=float(all_rows.min_lambda.min()),
        seconds=time.monotonic()-start,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=I.B/'qa'/('independent_pilot.json' if args.pilot else 'independent_validation.json')
    I.write_json(path,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
