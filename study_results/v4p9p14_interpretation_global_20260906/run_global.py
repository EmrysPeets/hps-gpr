#!/usr/bin/env python3
"""Coherent full-spectrum scan toys and Asimov-response covariance pilot."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=ROOT/'study_results/v4p9p13_calibration_20260905'
sys.path.insert(0,str(PARENT))
import calibration_core as core
np,pd,c=core.np,core.pd,core.c

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def evaluate(ctx,spectra,batch=32):
    values={m:[] for m in ('profiled','fixed')}; checks=[]
    for start in range(0,len(spectra),batch):
        models=ctx.make_models(spectra[start:start+batch])
        for method,model in models.items():
            values[method].extend(model.r.tolist())
            checks.append(dict(start=start,n=len(model.r),method=method,
                               score=model.max_score,scalar_fallbacks=model.fallbacks))
    if any(not np.all(np.isfinite(v)) for v in values.values()):
        raise RuntimeError('Nonfinite scan statistic')
    return {k:np.array(v) for k,v in values.items()},checks

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['2015','2016','2021'],default='2015')
    ap.add_argument('--ensemble',choices=['pilot10','validation1000','asimov'],required=True)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    out=args.output or HERE/'global'/args.dataset/args.ensemble
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    scope=next(s for s in core.SCOPES if s[2]==(args.dataset,))
    cfg=c.production.load_config(c.production.DEFAULT_CARD)
    c.production.validate_card(cfg)
    datasets=c.production.make_datasets(cfg)
    states=c.production.state_map(pd.read_csv(c.production.DEFAULT_STATES))
    masses=np.arange(scope[3],scope[4]+1)
    first=core.Context(scope,int(masses[0]),cfg,datasets,states)
    truth=first.truths['stress']
    input_hashes=json.loads((PARENT/'derived/contract.json').read_text())['hashes']
    for name,digest in input_hashes.items():
        if sha(ROOT/name)!=digest: raise RuntimeError('Parent source changed: '+name)
    source={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            str((HERE/'PROTOCOL.md').relative_to(ROOT)):sha(HERE/'PROTOCOL.md')}
    source.update(input_hashes)
    # Numerical modules outside the parent's main contract are bound explicitly.
    for module in (core,sys.modules['gp_refit_pilot'],sys.modules['batch_profile']):
        source[str(Path(module.__file__).relative_to(ROOT))]=sha(module.__file__)
    n={'pilot10':10,'validation1000':1000,'asimov':len(truth)+1}[args.ensemble]
    contract=dict(dataset=args.dataset,ensemble=args.ensemble,n_spectra=n,
        masses_MeV=masses.tolist(),truth='archived_stress_common_full_spectrum',
        truth_array_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),source_sha256=source,
        parent_contract_sha256=sha(PARENT/'derived/contract.json'),batch_size=32,
        kernel_optimized_per_toy=False)
    cp=out/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=contract:
        raise RuntimeError('Changed contract; use a new output directory')
    write_json(cp,contract)
    if args.ensemble=='asimov':
        spectra=np.broadcast_to(truth,(n,len(truth))).copy()
        ii=np.arange(len(truth));spectra[ii+1,ii]+=np.sqrt(truth)
    else:
        rng=core.seed('v4p9p14-global',args.dataset,args.ensemble)
        spectra=rng.poisson(truth,size=(n,len(truth))).astype(float)
    np.savez_compressed(out/'spectra.npz',counts=spectra,truth=truth,
                        edges_GeV=first.parts[0]['p'].edges_full)
    obs=pd.read_csv(PARENT/'summary/observed_calibrated_limits.csv')
    obs=obs[obs.scope_key==scope[0]].set_index('mass_MeV')
    for j,mass in enumerate(masses):
        checkpoint=out/f'm{mass:03d}.npz';audit=out/f'm{mass:03d}_qa.json'
        if checkpoint.exists() and audit.exists():
            old=json.loads(audit.read_text())
            if old['checkpoint_sha256']!=sha(checkpoint) or not old['passed']:
                raise RuntimeError('Checkpoint QA mismatch')
            continue
        ctx=first if j==0 else core.Context(scope,int(mass),cfg,datasets,states)
        if not np.array_equal(ctx.truths['stress'],truth):
            raise RuntimeError('Truth varies across hypotheses')
        observed_error={m:abs(float(ctx.ofit[m]['signed_r'])-float(obs.loc[mass,'signed_r_'+m+'_asymptotic'])) for m in ('profiled','fixed')}
        if max(observed_error.values())>2e-5: raise RuntimeError('Observed parent mismatch')
        tick=time.monotonic()
        try:
            values,checks=evaluate(ctx,spectra)
        except Exception as error:
            write_json(out/f'm{mass:03d}_FAILURE.json',dict(error=str(error),type=type(error).__name__))
            raise
        np.savez_compressed(checkpoint,**values)
        write_json(audit,dict(passed=all(x['score']<2e-7 for x in checks),
            mass_MeV=int(mass),seconds=time.monotonic()-tick,
            observed_r_error=observed_error,checks=checks,scalar_checks=ctx.scalar_checks,
            checkpoint_sha256=sha(checkpoint),n_spectra=n))
        print(json.dumps(dict(ensemble=args.ensemble,mass=int(mass),completed=j+1,total=len(masses),elapsed_seconds=round(time.monotonic()-started,2))),flush=True)
    arrays={m:np.column_stack([np.load(out/f'm{x:03d}.npz')[m] for x in masses]) for m in ('profiled','fixed')}
    np.savez_compressed(out/'scan_vectors.npz',masses_MeV=masses,**arrays)
    write_json(out/'summary.json',dict(passed=True,complete=True,n_spectra=n,
        hypotheses=len(masses),full_bins=len(truth),seconds_this_invocation=time.monotonic()-started,
        contract_sha256=sha(cp),spectra_sha256=sha(out/'spectra.npz'),
        vectors_sha256=sha(out/'scan_vectors.npz'),truth_min_count=float(truth.min())))

if __name__=='__main__':main()
