#!/usr/bin/env python3
"""Coherent shared-coupling scan over the complete 19--250 MeV union."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, subprocess
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V15=HERE.parent/'v4p9p15_global_2016_2021_20260906'
V14=HERE.parent/'v4p9p14_interpretation_global_20260906'
sys.path.insert(0,str(V15))
import run_global as previous
core,np,pd,c=previous.core,previous.np,previous.pd,previous.c
sha=previous.sha
PARENT=previous.PARENT
METHODS=('profiled','fixed')
ENSEMBLES=('pilot10','validation1000','asimov')
SENTINELS={39,49,50,90,91,180}
YEARS=('2015','2016','2021')
SIZES={'2015':484,'2016':720,'2021':422}
STARTS={'2015':0,'2016':484,'2021':1204}
FOLDER=HERE/'global'

def write_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def source_folder(year):
    return (V14/'global'/year) if year=='2015' else V15/'global_fast'/year

def scope_for(mass):
    keys=tuple(k for k in YEARS if c.production.EXPECTED_DATASET_GRIDS[k][0]
               <=mass<=c.production.EXPECTED_DATASET_GRIDS[k][-1])
    return next(s for s in c.production.SCOPES if s[2]==keys)

def active_indices(keys):
    return np.concatenate([np.arange(STARTS[k],STARTS[k]+SIZES[k]) for k in keys])

def dimuon_factor(mass):
    if mass<=2*105.6583745:return 1.
    ratio=(105.6583745/mass)**2
    return 1.+np.sqrt(1.-4.*ratio)*(1.+2.*ratio)

class MemoContext(core.Context):
    """Exact reuse of repeated predictions, especially unchanged Asimov parts."""
    def __init__(self,*args):
        self.prediction_cache={}
        self.prediction_cache_hits=0
        super().__init__(*args)

    def retrain(self,whole):
        bs=[];Ls=[]
        for j,part in enumerate(self.parts):
            y=whole[self.offsets[j]:self.offsets[j+1]][part['keep']]
            key=(j,id(part['predictor']),self.nuisance_cut,y.tobytes())
            if key in self.prediction_cache:
                b,L=self.prediction_cache[key]
                self.prediction_cache_hits+=1
            else:
                b,C=part['predictor'].predict(y)
                C,_=c.production.condition_covariance_block(C,b)
                if self.nuisance_cut:
                    sd=np.sqrt(b)
                    v,U=np.linalg.eigh(C/sd[:,None]/sd[None,:])
                    keep=v>self.nuisance_cut
                    L=sd[:,None]*U[:,keep]*np.sqrt(v[keep])
                    width=min(12,len(b))
                    if L.shape[1]>width:raise RuntimeError('Nuisance padding exceeded')
                    L=np.pad(L,((0,0),(0,width-L.shape[1])))
                else:
                    L=c._chol_with_jitter(C)
                self.prediction_cache[key]=(b,L)
            bs.append(b);Ls.append(L)
        return np.concatenate(bs),core.block_diag(*Ls)

def exact_backend(ctx):
    ctx.nuisance_cut=0.
    for p in ctx.parts:p['predictor']=p['exact_predictor']
    ctx.gp_backend='exact_cached_cholesky'

def evaluate(ctx,counts):
    ctx.scalar_checks=[];ctx.scalar_check_batches=0
    roots,checks=previous.evaluate(ctx,counts)
    passed=all(x['score']<2e-7 for x in checks) and all(x['passed'] for x in ctx.scalar_checks)
    if not passed:raise RuntimeError('Numerical score/scalar gate failed')
    return dict(values=roots,checks=checks,scalar_checks=list(ctx.scalar_checks),passed=True)

def observed_row(ctx,reference,old):
    mass=ctx.mass;factor=dimuon_factor(mass)
    row=dict(mass_MeV=mass,scope_key=ctx.scope[0],dataset_set='+'.join(ctx.keys),
             n_active=len(ctx.keys),dimuon_factor=factor,
             signal_yield_per_eps2_fitted_window=ctx.conversion,
             inherited_2016_exception='2016' in ctx.keys)
    checks={}
    for method in METHODS:
        fit=ctx.ofit[method]
        row.update({method+'_r':float(fit['signed_r']),
                    method+'_eps2_ee_raw':float(fit['A90']/ctx.conversion),
                    method+'_eps2_display':float(fit['A90']/ctx.conversion*factor),
                    method+'_Ahat_window':float(fit['Ahat']),
                    method+'_A90_window':float(fit['A90'])})
        checks[method]=dict(cls=fit['cls'],q_obs=fit['q_obs'],
            q_asimov=fit['q_asimov'],max_score=fit['max_score'],
            min_lambda=fit['min_lambda'],monotonicity_error=fit['monotonicity_error'])
        key=(ctx.scope[0],mass)
        if key in reference.index:
            ref=reference.loc[key]
            re=abs(row[method+'_r']-ref['signed_r_'+method+'_asymptotic'])
            le=abs(row[method+'_eps2_ee_raw']/ref['eps2_'+method+'_asymptotic_ee_raw']-1)
            checks[method].update(v13_root_error=float(re),v13_limit_relative_error=float(le))
            if re>2e-5 or le>1e-8:raise RuntimeError('Dense v13 observed mismatch')
    ref=old.loc[(ctx.scope[0],mass)]
    row['v12_eps2_ee_raw']=float(ref.eps2_90)
    row['v12_eps2_display']=float(ref.eps2_90*factor)
    row['v12_p0']=float(ref.p0_local_asymptotic)
    row['v12_limit_relative_difference']=row['profiled_eps2_ee_raw']/ref.eps2_90-1
    row['v12_bounded_root_difference']=max(0,row['profiled_r'])-float(ref.Z_local_asymptotic)
    checks['v12_investigation_required']=bool(
        abs(row['v12_limit_relative_difference'])>.03 or
        abs(row['v12_bounded_root_difference'])>.15)
    return row,checks

def reused_observed(scope,mass,reference,old):
    x=reference.loc[(scope[0],mass)];f=dimuon_factor(mass)
    row=dict(mass_MeV=mass,scope_key=scope[0],dataset_set='+'.join(scope[2]),
        n_active=1,dimuon_factor=f,
        signal_yield_per_eps2_fitted_window=float(x.signal_yield_per_eps2_fitted_window),
        inherited_2016_exception=False)
    for method in METHODS:
        row[method+'_r']=float(x['signed_r_'+method+'_asymptotic'])
        row[method+'_eps2_ee_raw']=float(x['eps2_'+method+'_asymptotic_ee_raw'])
        row[method+'_eps2_display']=row[method+'_eps2_ee_raw']*f
    ref=old.loc[(scope[0],mass)]
    row.update(v12_eps2_ee_raw=float(ref.eps2_90),v12_eps2_display=float(ref.eps2_90*f),
        v12_p0=float(ref.p0_local_asymptotic),
        v12_limit_relative_difference=row['profiled_eps2_ee_raw']/ref.eps2_90-1,
        v12_bounded_root_difference=max(0,row['profiled_r'])-float(ref.Z_local_asymptotic))
    return row

def setup():
    sources=json.loads((PARENT/'derived/contract.json').read_text())['hashes']
    import gp_lowrank_pilot
    extra=[Path(__file__),HERE/'PROTOCOL.md',Path(previous.__file__),Path(core.__file__),
        Path(sys.modules['gp_refit_pilot'].__file__),
        Path(sys.modules['batch_profile'].__file__),Path(gp_lowrank_pilot.__file__),
        PARENT/'summary/observed_calibrated_limits.csv',
        V14/'MANIFEST.csv',V15/'MANIFEST.csv']
    counts={};truths={};upstream={}
    for year in YEARS:
        base=source_folder(year);upstream[year]={}
        for ensemble in ENSEMBLES:
            p=base/ensemble
            summary=json.loads((p/'summary.json').read_text())
            contract=json.loads((p/'contract.json').read_text())
            assert summary['passed'] and summary['complete']
            for name,key in [('contract.json','contract_sha256'),
                             ('spectra.npz','spectra_sha256'),
                             ('scan_vectors.npz','vectors_sha256')]:
                if sha(p/name)!=summary[key]:raise RuntimeError('Upstream ensemble changed')
                extra.append(p/name)
            extra.append(p/'summary.json')
            with np.load(p/'spectra.npz') as x:
                b=x['truth'];n=x['counts']
            assert len(b)==SIZES[year]
            if year in truths:assert np.array_equal(truths[year],b)
            truths[year]=b
            if ensemble!='asimov':counts[year,ensemble]=n
            upstream[year][ensemble]=dict(contract_sha256=sha(p/'contract.json'),
                 seed_convention='v4p9p14-global, dataset, ensemble',
                 n_spectra=len(n),source=str(p.relative_to(ROOT)))
    for p in extra:sources[str(p.relative_to(ROOT))]=sha(p)
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Changed input: '+name)
    truth=np.concatenate([truths[y] for y in YEARS])
    joint={}
    for ensemble in ('pilot10','validation1000'):
        joint[ensemble]=np.concatenate([counts[y,ensemble] for y in YEARS],axis=1)
        assert joint[ensemble].shape==((10 if ensemble=='pilot10' else 1000),1626)
    n=len(truth)
    joint['asimov']=np.broadcast_to(truth,(n+1,n)).copy()
    i=np.arange(n);joint['asimov'][i+1,i]+=np.sqrt(truth)
    contract=dict(version='4.9.16',mass_grid_MeV=list(range(19,251)),
        source_sha256=sources,upstream=upstream,sentinels_MeV=sorted(SENTINELS),
        full_bin_sizes=SIZES,full_bin_starts=STARTS,
        truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
        methods=list(METHODS),ensembles={e:len(a) for e,a in joint.items()},
        membership=[dict(mass_MeV=m,datasets=list(scope_for(m)[2])) for m in range(19,251)])
    cp=FOLDER/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=contract:
        raise RuntimeError('Changed numerical contract; preserve the completed derivative')
    write_json(cp,contract)
    for ensemble,arr in joint.items():
        p=FOLDER/'spectra'/f'{ensemble}.npz'
        if p.exists():
            with np.load(p) as saved:
                if not np.array_equal(arr,saved['counts']) or not np.array_equal(truth,saved['truth']):
                    raise RuntimeError('Changed joint spectra')
        else:np.savez_compressed(p,counts=arr,truth=truth)
    return truth,joint,sha(cp)

def run_point(mass,truth,joint,cfg,datasets,states,reference,old,contract_sha):
    checkpoint=FOLDER/'points'/f'm{mass:03d}.npz'
    auditpath=checkpoint.with_name(f'm{mass:03d}_qa.json')
    if checkpoint.exists() and auditpath.exists():
        audit=json.loads(auditpath.read_text())
        if not audit['passed'] or audit['contract_sha256']!=contract_sha or audit['checkpoint_sha256']!=sha(checkpoint):
            raise RuntimeError('Invalid completed point')
        return audit
    tick=time.monotonic();scope=scope_for(mass);indices=active_indices(scope[2])
    audit=dict(mass_MeV=mass,scope_key=scope[0],active_datasets=list(scope[2]),
        contract_sha256=contract_sha,passed=False,source_reference_sha256={})
    arrays={}
    if len(scope[2])==1:
        year=scope[2][0]
        for ensemble in ENSEMBLES:
            p=source_folder(year)/ensemble/f'm{mass:03d}.npz'
            q=p.with_name(f'm{mass:03d}_qa.json')
            qa=json.loads(q.read_text())
            if not qa['passed'] or qa['checkpoint_sha256']!=sha(p):
                raise RuntimeError('Reused point failed its source audit')
            for path in (p,q):audit['source_reference_sha256'][str(path.relative_to(ROOT))]=sha(path)
            with np.load(p) as x:
                for method in METHODS:
                    r=x[method]
                    if ensemble=='asimov':
                        embedded=np.full(1627,r[0]);embedded[indices+1]=r[1:];r=embedded
                    arrays[ensemble+'_'+method]=r
        audit['observed']=reused_observed(scope,mass,reference,old)
        audit.update(numerical_backend='reused_validated_single_dataset',
                     numerical_checks_in_upstream=True)
    else:
        ctx=MemoContext(scope,mass,cfg,datasets,states)
        if not np.array_equal(ctx.truths['stress'],truth[indices]):
            raise RuntimeError('Combined truth does not match source spectra')
        audit['observed'],audit['observed_checks']=observed_row(ctx,reference,old)
        local={}
        for ensemble in ENSEMBLES:
            rows=np.r_[0,indices+1] if ensemble=='asimov' else np.arange(len(joint[ensemble]))
            local[ensemble]=joint[ensemble][np.ix_(rows,indices)]
        exact_backend(ctx)
        exact_pilot=evaluate(ctx,local['pilot10'])
        # Check memoization against the unchanged parent calculation.
        b0,L0=core.Context.retrain(ctx,local['asimov'][0])
        b1,L1=ctx.retrain(local['asimov'][0])
        if not np.array_equal(b0,b1) or not np.array_equal(L0,L1):
            raise RuntimeError('Memoized prediction differs from direct parent')
        audit['memoization_parent_baseline_exact']=True
        probe=[]
        for j,part in enumerate(ctx.parts):
            start=int(ctx.offsets[j]);size=int(part['n'])
            bins=np.flatnonzero(part['p'].blind_mask)
            probe.extend((start+np.unique(np.r_[np.linspace(0,size-1,16,dtype=int),
                bins[0],bins[len(bins)//2],bins[-1]])+1).tolist())
        probe_indices=np.arange(len(local['asimov'])) if mass in SENTINELS else np.r_[0,np.unique(probe)]
        exact_response=evaluate(ctx,local['asimov'][probe_indices])
        refpath=FOLDER/'references'/f'm{mass:03d}.npz'
        np.savez_compressed(refpath,probe_indices=probe_indices,active_full_bin_indices=indices,
            **{'pilot_'+m:exact_pilot['values'][m] for m in METHODS},
            **{'response_'+m:exact_response['values'][m] for m in METHODS})
        write_json(refpath.with_suffix('.json'),dict(passed=True,
            pilot_checks=exact_pilot['checks'],pilot_scalar_checks=exact_pilot['scalar_checks'],
            response_checks=exact_response['checks'],response_scalar_checks=exact_response['scalar_checks'],
            reference_sha256=sha(refpath)))
        parent_ok=core.enable_lowrank(ctx)
        audit.update(parent_gate_passed=parent_ok,parent_checks=list(ctx.numerical_checks),
                     parent_fallback=ctx.gp_fallback_reason,fallback_reasons=[],
                     response_checks={},pilot_checks={})
        if not parent_ok:audit['fallback_reasons'].append('Parent approximation gate')
        evaluations={}
        try:
            for ensemble in ENSEMBLES:
                if ensemble=='pilot10' and not parent_ok:evaluations[ensemble]=exact_pilot
                elif ensemble=='asimov' and not parent_ok and mass in SENTINELS:evaluations[ensemble]=exact_response
                else:evaluations[ensemble]=evaluate(ctx,local[ensemble])
            for method in METHODS:
                p=evaluations['pilot10']['values'][method]
                exact=exact_pilot['values'][method]
                error=float(np.max(abs(p-exact)));flips=int(np.count_nonzero((p>0)!=(exact>0)))
                audit['pilot_checks'][method]=dict(max_root_error=error,bounded_atom_flips=flips,passed=error<1e-3 and flips==0)
                if not audit['pilot_checks'][method]['passed']:audit['fallback_reasons'].append(method+' pilot mismatch')
                r=evaluations['asimov']['values'][method];expected=exact_response['values'][method]
                response=r[1:]-r[0];width=float(np.linalg.norm(response))
                delta=(r[probe_indices]-r[0])-(expected-expected[0])
                rec=dict(full_column=mass in SENTINELS,probed_spectra=len(probe_indices),
                    max_root_error=float(np.max(abs(r[probe_indices]-expected))),
                    baseline_error_over_width=float(abs(r[0]-expected[0])/width),
                    max_response_error=float(np.max(abs(delta))),
                    max_response_error_over_width=float(np.max(abs(delta))/width))
                passed=width>0 and rec['max_root_error']<1e-3 and rec['baseline_error_over_width']<1e-3 and rec['max_response_error']<1e-4 and rec['max_response_error_over_width']<1e-4
                if mass in SENTINELS:
                    expected_response=expected[1:]-expected[0]
                    rec['relative_l2_response_error']=float(np.linalg.norm(response-expected_response)/np.linalg.norm(expected_response))
                    rec['relative_width_error']=float(abs(width/np.linalg.norm(expected_response)-1))
                    passed &= rec['relative_l2_response_error']<1e-3 and rec['relative_width_error']<1e-3
                rec['passed']=bool(passed);audit['response_checks'][method]=rec
                if not passed:audit['fallback_reasons'].append(method+' response mismatch')
        except Exception as error:
            if ctx.gp_backend=='exact_cached_cholesky':raise
            audit['fallback_reasons'].append('Approximation exception: '+type(error).__name__+': '+str(error))
        if audit['fallback_reasons'] and ctx.gp_backend!='exact_cached_cholesky':
            exact_backend(ctx);evaluations={}
            for ensemble in ENSEMBLES:
                if ensemble=='pilot10':evaluations[ensemble]=exact_pilot
                elif ensemble=='asimov' and mass in SENTINELS:evaluations[ensemble]=exact_response
                else:evaluations[ensemble]=evaluate(ctx,local[ensemble])
        if set(evaluations)!=set(ENSEMBLES):raise RuntimeError('Incomplete coordinate')
        for ensemble,ev in evaluations.items():
            for method in METHODS:
                r=ev['values'][method]
                if ensemble=='asimov':
                    embedded=np.full(1627,r[0]);embedded[indices+1]=r[1:];r=embedded
                arrays[ensemble+'_'+method]=r
        audit.update(numerical_backend=ctx.gp_backend,prediction_cache_hits=ctx.prediction_cache_hits,
            phase_checks={e:dict(checks=v['checks'],scalar_checks=v['scalar_checks']) for e,v in evaluations.items()})
        for path in (refpath,refpath.with_suffix('.json')):
            audit['source_reference_sha256'][str(path.relative_to(ROOT))]=sha(path)
    if any(not np.all(np.isfinite(v)) for v in arrays.values()):raise RuntimeError('Nonfinite roots')
    np.savez_compressed(checkpoint,**arrays)
    audit.update(passed=True,seconds=time.monotonic()-tick,checkpoint_sha256=sha(checkpoint))
    write_json(auditpath,audit)
    return audit

def assemble(contract_sha):
    masses=np.arange(19,251);audits=[]
    for mass in masses:
        p=FOLDER/'points'/f'm{mass:03d}.npz';q=p.with_name(f'm{mass:03d}_qa.json')
        if not p.exists() or not q.exists():return False
        a=json.loads(q.read_text())
        if not a['passed'] or a['checkpoint_sha256']!=sha(p) or a['contract_sha256']!=contract_sha:
            raise RuntimeError('Final point integrity failure')
        audits.append(a)
    arrays={}
    for ensemble in ENSEMBLES:
        for method in METHODS:
            arrays[ensemble+'_'+method]=np.column_stack([
                np.load(FOLDER/'points'/f'm{mass:03d}.npz')[ensemble+'_'+method] for mass in masses])
    np.savez_compressed(FOLDER/'scan_vectors.npz',masses_MeV=masses,**arrays)
    pd.DataFrame([a['observed'] for a in audits]).to_csv(FOLDER/'observed.csv',index=False)
    investigations=[a['mass_MeV'] for a in audits if a.get('observed_checks',{}).get('v12_investigation_required')]
    write_json(FOLDER/'summary.json',dict(passed=True,complete=True,hypotheses=232,
        full_bins=1626,pilot_scans=10,validation_scans=1000,asimov_scans=1627,
        contract_sha256=contract_sha,vectors_sha256=sha(FOLDER/'scan_vectors.npz'),
        observed_sha256=sha(FOLDER/'observed.csv'),
        spectra_sha256={e:sha(FOLDER/'spectra'/f'{e}.npz') for e in ENSEMBLES},
        reused_single_coordinates=90,new_joint_coordinates=142,
        coordinate_seconds=sum(a['seconds'] for a in audits),
        exact_fallback_masses=[a['mass_MeV'] for a in audits if a['numerical_backend']=='exact_cached_cholesky'],
        v12_dense_investigation_masses=investigations))
    return True

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--masses',type=int,nargs='*')
    args=ap.parse_args();started=time.monotonic()
    truth,joint,contract_sha=setup()
    cfg=c.production.load_config(c.production.DEFAULT_CARD);c.production.validate_card(cfg)
    datasets=c.production.make_datasets(cfg)
    states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
    reference=pd.read_csv(PARENT/'summary/observed_calibrated_limits.csv').set_index(['scope_key','mass_MeV'])
    old=pd.read_csv(Path(c.production.__file__).parent/'derived/final_dataset_result_curves.csv').set_index(['scope_key','mass_MeV'])
    for mass in (args.masses if args.masses is not None else range(19,251)):
        try:audit=run_point(mass,truth,joint,cfg,datasets,states,reference,old,contract_sha)
        except Exception as error:
            write_json(FOLDER/'points'/f'm{mass:03d}_FAILURE.json',dict(type=type(error).__name__,error=str(error)))
            raise
        print(json.dumps(dict(mass=mass,scope=audit['scope_key'],backend=audit['numerical_backend'],
            seconds=round(audit['seconds'],2),elapsed_seconds=round(time.monotonic()-started,2))),flush=True)
    complete=assemble(contract_sha)
    print(json.dumps(dict(complete=complete,seconds_this_invocation=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
