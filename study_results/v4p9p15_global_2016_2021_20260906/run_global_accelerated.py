#!/usr/bin/env python3
"""Paired numerical derivative with per-coordinate exact fallbacks and response gates."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import run_global as dense
core,np,pd,c = dense.core,dense.np,dense.pd,dense.c
HERE,ROOT,PARENT = dense.HERE,dense.ROOT,dense.PARENT
sha,write_json = dense.sha,dense.write_json
METHODS = ('profiled','fixed')
ENSEMBLES = ('pilot10','validation1000','asimov')
SENTINELS = {'2016':{39,56,66,75,120,180},'2021':{50,78,100,150,200,250}}

def exact_backend(ctx):
    ctx.nuisance_cut = 0.
    for part in ctx.parts:
        part['predictor'] = part['exact_predictor']
    ctx.gp_backend = 'exact_cached_cholesky'

def evaluate(ctx,counts):
    ctx.scalar_checks = []
    ctx.scalar_check_batches = 0
    values,checks = dense.evaluate(ctx,counts)
    return dict(values=values,checks=checks,scalar_checks=list(ctx.scalar_checks))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True,choices=['2016','2021'])
    args = ap.parse_args()
    year = args.dataset
    folder = HERE/'global_fast'/year
    (folder/'numerical_gates').mkdir(parents=True,exist_ok=True)
    (folder/'response_audit').mkdir(exist_ok=True)
    start = time.monotonic()
    scope = next(s for s in core.SCOPES if s[2]==(year,))
    cfg = c.production.load_config(c.production.DEFAULT_CARD)
    c.production.validate_card(cfg)
    datasets = c.production.make_datasets(cfg)
    states = c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
    masses = np.arange(scope[3],scope[4]+1)
    first = core.Context(scope,int(masses[0]),cfg,datasets,states)
    truth = first.truths['stress']
    inputs = json.loads((PARENT/'derived/contract.json').read_text())['hashes']
    for name,digest in inputs.items():
        if sha(ROOT/name)!=digest:
            raise RuntimeError('Parent source changed: '+name)
    source = dict(inputs)
    import gp_lowrank_pilot
    for path in [Path(__file__),HERE/'run_global.py',HERE/'PROTOCOL.md',HERE/'ACCELERATION_PROTOCOL.md',
                 HERE/'ACCELERATION_RESPONSE_GATES.md',Path(core.__file__),
                 Path(sys.modules['gp_refit_pilot'].__file__),Path(sys.modules['batch_profile'].__file__),
                 Path(gp_lowrank_pilot.__file__)]:
        source[str(path.relative_to(ROOT))] = sha(path)
    n = dict(pilot10=10,validation1000=1000,asimov=len(truth)+1)
    spectra = {}
    for ensemble in ENSEMBLES:
        out = folder/ensemble
        out.mkdir(exist_ok=True)
        contract = dict(dataset=year,ensemble=ensemble,n_spectra=n[ensemble],masses_MeV=masses.tolist(),
            truth='archived_stress_common_full_spectrum',truth_array_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
            source_sha256=source,parent_contract_sha256=sha(PARENT/'derived/contract.json'),batch_size=32,
            kernel_optimized_per_toy=False,numerical_backend='gated_parent_lowrank_with_entire_coordinate_exact_fallback',
            seed_convention='v4p9p14-global, dataset, ensemble; paired with exact reference')
        cp = out/'contract.json'
        if cp.exists() and json.loads(cp.read_text())!=contract:
            raise RuntimeError('Changed contract; use a new derivative')
        write_json(cp,contract)
        if ensemble=='asimov':
            counts = np.broadcast_to(truth,(n[ensemble],len(truth))).copy()
            ii=np.arange(len(truth));counts[ii+1,ii]+=np.sqrt(truth)
        else:
            counts = core.seed('v4p9p14-global',year,ensemble).poisson(truth,size=(n[ensemble],len(truth))).astype(float)
        old = HERE/'global'/year/ensemble/'spectra.npz'
        if old.exists() and not np.array_equal(counts,np.load(old)['counts']):
            raise RuntimeError('Exact and fast spectra differ')
        spectra[ensemble] = counts
        np.savez_compressed(out/'spectra.npz',counts=counts,truth=truth,edges_GeV=first.parts[0]['p'].edges_full)
    obs = pd.read_csv(PARENT/'summary/observed_calibrated_limits.csv')
    obs = obs[obs.scope_key==scope[0]].set_index('mass_MeV')

    for j,mass in enumerate(masses):
        mass=int(mass)
        files = [(folder/e/f'm{mass:03d}.npz',folder/e/f'm{mass:03d}_qa.json') for e in ENSEMBLES]
        gatepath = folder/'numerical_gates'/f'm{mass:03d}.json'
        if all(p.exists() and q.exists() for p,q in files) and gatepath.exists():
            if not json.loads(gatepath.read_text())['passed']:
                raise RuntimeError('Failed saved gate')
            for p,q in files:
                audit=json.loads(q.read_text())
                if not audit['passed'] or audit['checkpoint_sha256']!=sha(p):
                    raise RuntimeError('Checkpoint mismatch')
            continue
        ctx = first if j==0 else core.Context(scope,mass,cfg,datasets,states)
        if not np.array_equal(ctx.truths['stress'],truth):
            raise RuntimeError('Truth changes with hypothesis')
        observed_error={m:abs(float(ctx.ofit[m]['signed_r'])-float(obs.loc[mass,'signed_r_'+m+'_asymptotic'])) for m in METHODS}
        if max(observed_error.values())>2e-5:
            raise RuntimeError('Observed parent mismatch')
        tick=time.monotonic()
        parent_ok=core.enable_lowrank(ctx)
        parent_records=list(ctx.numerical_checks)
        gate=dict(mass_MeV=mass,parent_gate_passed=parent_ok,parent_checks=parent_records,
                  parent_fallback=ctx.gp_fallback_reason,fallback_reasons=[],response_checks={},overlap_checks={},passed=False)
        if not parent_ok:
            gate['fallback_reasons'].append('Parent numerical gate selected exact backend')
        fast_predictors=[p['predictor'] for p in ctx.parts]
        fast_cut=ctx.nuisance_cut
        fast_backend=ctx.gp_backend
        bins=np.flatnonzero(ctx.mask)
        probe=np.unique(np.r_[np.linspace(0,len(truth)-1,16,dtype=int),bins[0],bins[len(bins)//2],bins[-1]])+1
        indices=np.arange(len(truth)+1) if mass in SENTINELS[year] else np.r_[0,probe]
        exact_backend(ctx)
        exact_probe=evaluate(ctx,spectra['asimov'][indices])
        ctx.nuisance_cut=fast_cut
        ctx.gp_backend=fast_backend
        for part,predictor in zip(ctx.parts,fast_predictors):
            part['predictor']=predictor
        if mass in SENTINELS[year]:
            np.savez_compressed(folder/'response_audit'/f'm{mass:03d}_exact_full.npz',**exact_probe['values'])
        evaluations={};phase_seconds={}
        try:
            for ensemble in ENSEMBLES:
                t=time.monotonic()
                evaluations[ensemble]=evaluate(ctx,spectra[ensemble])
                phase_seconds[ensemble]=time.monotonic()-t
            for method in METHODS:
                roots=evaluations['asimov']['values'][method]
                expected=exact_probe['values'][method]
                response=roots[1:]-roots[0]
                width=float(np.linalg.norm(response))
                delta=(roots[indices]-roots[0])-(expected-expected[0])
                record=dict(probed_spectra=len(indices),full_column=mass in SENTINELS[year],
                    max_root_error=float(np.max(abs(roots[indices]-expected))),
                    baseline_error_over_width=float(abs(roots[0]-expected[0])/width),
                    max_response_error=float(np.max(abs(delta))),max_response_error_over_width=float(np.max(abs(delta))/width))
                passed=record['max_root_error']<1e-3 and record['baseline_error_over_width']<1e-3 and record['max_response_error']<1e-4 and record['max_response_error_over_width']<1e-4
                if mass in SENTINELS[year]:
                    response_exact=expected[1:]-expected[0]
                    record['relative_l2_response_error']=float(np.linalg.norm(response-response_exact)/np.linalg.norm(response_exact))
                    record['relative_width_error']=float(abs(width/np.linalg.norm(response_exact)-1))
                    passed &= record['relative_l2_response_error']<1e-3 and record['relative_width_error']<1e-3
                record['passed']=bool(passed)
                gate['response_checks'][method]=record
                if not passed:
                    gate['fallback_reasons'].append(method+' response accuracy gate')
                for ensemble in ('pilot10','validation1000'):
                    old=HERE/'global'/year/ensemble/f'm{mass:03d}.npz'
                    if old.exists():
                        expected_root=np.load(old)[method]
                        error=float(np.max(abs(evaluations[ensemble]['values'][method]-expected_root)))
                        gate['overlap_checks'][ensemble+'_'+method]=dict(n=len(expected_root),max_root_error=error,passed=error<1e-3)
                        if error>=1e-3:
                            gate['fallback_reasons'].append(ensemble+' '+method+' exact overlap gate')
        except Exception as error:
            if ctx.gp_backend=='exact_cached_cholesky':
                write_json(folder/f'm{mass:03d}_FAILURE.json',dict(type=type(error).__name__,error=str(error)))
                raise
            gate['fallback_reasons'].append('Approximation execution exception: '+type(error).__name__+': '+str(error))
        if gate['fallback_reasons'] and fast_backend!='exact_cached_cholesky':
            exact_backend(ctx)
            evaluations={};phase_seconds={}
            for ensemble in ENSEMBLES:
                t=time.monotonic()
                evaluations[ensemble]=evaluate(ctx,spectra[ensemble])
                phase_seconds[ensemble]=time.monotonic()-t
        gate['final_backend']=ctx.gp_backend
        gate['passed']=all(all(row['score']<2e-7 for row in ev['checks']) and all(row['passed'] for row in ev['scalar_checks']) for ev in evaluations.values())
        gate['seconds_total_coordinate']=time.monotonic()-tick
        if not gate['passed']:
            write_json(gatepath,gate)
            raise RuntimeError('Coordinate numerical QA failed')
        write_json(gatepath,gate)
        for ensemble in ENSEMBLES:
            checkpoint=folder/ensemble/f'm{mass:03d}.npz'
            ev=evaluations[ensemble]
            np.savez_compressed(checkpoint,**ev['values'])
            write_json(folder/ensemble/f'm{mass:03d}_qa.json',dict(passed=True,mass_MeV=mass,
                seconds=phase_seconds[ensemble],observed_r_error=observed_error,checks=ev['checks'],
                scalar_checks=ev['scalar_checks'],checkpoint_sha256=sha(checkpoint),n_spectra=n[ensemble],
                numerical_gate_sha256=sha(gatepath),numerical_backend=ctx.gp_backend))
        print(json.dumps(dict(dataset=year,mass=mass,completed=j+1,total=len(masses),backend=ctx.gp_backend,
                              elapsed_seconds=round(time.monotonic()-start,2))),flush=True)
    for ensemble in ENSEMBLES:
        out=folder/ensemble
        arrays={m:np.column_stack([np.load(out/f'm{x:03d}.npz')[m] for x in masses]) for m in METHODS}
        np.savez_compressed(out/'scan_vectors.npz',masses_MeV=masses,**arrays)
        phase_total=sum(json.loads((out/f'm{x:03d}_qa.json').read_text())['seconds'] for x in masses)
        write_json(out/'summary.json',dict(passed=True,complete=True,n_spectra=n[ensemble],hypotheses=len(masses),
            full_bins=len(truth),seconds_this_invocation=phase_total,timing_note='summed ensemble evaluations; shared numerical gates and construction reported at dataset level',
            contract_sha256=sha(out/'contract.json'),spectra_sha256=sha(out/'spectra.npz'),vectors_sha256=sha(out/'scan_vectors.npz'),truth_min_count=float(truth.min())))
    gates=[json.loads((folder/'numerical_gates'/f'm{x:03d}.json').read_text()) for x in masses]
    write_json(folder/'execution_summary.json',dict(passed=True,seconds_this_invocation=time.monotonic()-start,
        complete_coordinates=len(masses),exact_fallback_masses=[g['mass_MeV'] for g in gates if g['final_backend']=='exact_cached_cholesky'],
        paired_exact_pilot_masses=len(masses),paired_exact_validation_masses=sum(any(k.startswith('validation1000') for k in g['overlap_checks']) for g in gates)))

if __name__=='__main__':
    main()
