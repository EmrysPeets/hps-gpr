"""Independent ensemble B; retain every original ensemble-A scan unchanged."""
import run_global as R
from run_global import *
from concurrent.futures import ProcessPoolExecutor
import argparse,time
NVALID=1024;NAMESPACE=2

def validation_protocol():
    spec=dict(version='6.4.4',local_calibration='Frozen original1024 complete scans, namespace1',
        independent_validation_toys=NVALID,seed=SEED,namespace=NAMESPACE,
        windows={'2016':[-2.5,2.5],'2021':[-4,3],'2015':'unchanged Gaussian +/-2.25 sigma_ref'},
        shared_coupling='Exactly one psi=epsilon2/1e-8 at each generated mass; inherited fixed campaign conversions and independent GP nuisances',
        generation='Same frozen Poisson null sources and extraction as parent; one campaign draw reused at all masses and scopes',
        parent_protocol_sha256=sha(B/'provenance/global_protocol.json'),script_sha256=sha(__file__),
        engine_sha256=sha(B/'scripts/extraction.py'),parent_scan_sha256={s:sha(B/f'results/global_scans_{s}.npz') for s in SCOPES},
        observed_sha256=sha(B/'results/observed_scan.csv'))
    p=B/'provenance/validation_protocol.json'
    if p.exists():assert json.loads(p.read_text())==spec
    else:write(p,spec)
    return sha(p)

def initialize_validation():R.initialize(NVALID,NAMESPACE)

def job(args):
    m,signature=args;p=B/f'results/validation_checkpoints/m{m:03d}.npz'
    if p.exists():
        with np.load(p) as z:assert str(z['signature'])==signature and np.array_equal(z['toy_ids'],np.arange(NVALID))
        return dict(mass_MeV=m,cached=True)
    ctx=at_mass(m);out={};start=time.monotonic();score=0.;minimum=np.inf
    for t in range(NVALID):
        rows=evaluate(m,ctx,{y:R.CALIBRATION[y][t] for y in ctx})
        for scope,r in rows.items():
            if scope not in out:out[scope]=np.empty((NVALID,len(FIELDS)))
            out[scope][t]=[r[k] for k in FIELDS];score=max(score,r['max_score']);minimum=min(minimum,r['min_lambda'])
    assert all(np.isfinite(a).all() for a in out.values())
    tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,signature=signature,toy_ids=np.arange(NVALID),namespace=NAMESPACE,fields=np.array(FIELDS),mass_MeV=m,**out);tmp.replace(p)
    return dict(mass_MeV=m,cached=False,seconds=time.monotonic()-start,max_score=score,min_lambda=minimum)

def assemble(signature):
    initialize_validation();hashes={y:np.array([hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in a]) for y,a in R.CALIBRATION.items()}
    np.savez_compressed(B/'results/toy_draw_hashes_validation.npz',toy_ids=np.arange(NVALID),namespace=NAMESPACE,**hashes)
    for s in SCOPES:
        masses=np.arange(40 if s=='2016' else 60,176 if s=='2016' else 241);parts=[]
        for m in masses:
            with np.load(B/f'results/validation_checkpoints/m{m:03d}.npz') as z:
                assert str(z['signature'])==signature;parts.append(z[s])
        np.savez_compressed(B/f'results/global_validation_{s}.npz',masses_MeV=masses,toy_ids=np.arange(NVALID),namespace=NAMESPACE,fields=np.array(FIELDS),values=np.stack(parts,axis=1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assemble-only',action='store_true');args=ap.parse_args();signature=validation_protocol()
    if not args.assemble_only:
        start=time.monotonic();ledger=[]
        with ProcessPoolExecutor(max_workers=2,initializer=initialize_validation) as pool:
            for i,r in enumerate(pool.map(job,[(m,signature) for m in range(40,241)]),1):
                ledger.append(r);write(B/'qa/validation_progress.json',dict(completed_mass_jobs=i,total_mass_jobs=201,elapsed_seconds=time.monotonic()-start,last=r))
                if i%10==0:print('Completed',i,'/201 mass jobs',flush=True)
        write(B/'qa/validation_execution.json',dict(passed=True,toys=NVALID,workers=2,numerical_threads=1,elapsed_seconds=time.monotonic()-start,mass_jobs=ledger))
    assemble(signature)

if __name__=='__main__':main()
