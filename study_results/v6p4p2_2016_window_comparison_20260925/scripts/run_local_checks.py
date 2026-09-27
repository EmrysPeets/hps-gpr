"""Bounded fixed-mass source checks after observed region selection; not global."""
from extraction import *
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import beta
import argparse,time

NTOYS=1000;SEED=641250925;BLOCK=100

def job(args):
    scope,m,first,signature=args
    path=B/f'results/toy_checkpoints/{scope}_m{m:03d}_t{first:04d}.json'
    if path.exists():
        q=json.loads(path.read_text());assert q['signature']==signature;return q['rows']
    ys=['2016'] if scope=='2016' else years(m)
    truth={y:np.load(B/f'inputs/null_{y}.npz')['truth'] for y in ys}
    methods=['mc'] if scope=='2016' else list(METHODS)
    keys={('2016','mc')} if scope=='2016' else {(y,'mc' if y=='2021' and method!='all_gaussian' or y=='2016' and method=='mc2016_2021' else 'gaussian') for method in methods for y in ys}
    contexts={key:Context(key[0],m,key[1]) for key in keys};rows=[]
    for t in range(first,min(first+BLOCK,NTOYS)):
        counts={y:np.random.default_rng(np.random.SeedSequence([SEED,int(y),t])).poisson(truth[y]) for y in ys}
        count_hash=hashlib.sha256(b''.join(np.ascontiguousarray(counts[y]).tobytes() for y in sorted(ys))).hexdigest()
        parts={key:ctx.part(counts[key[0]]) for key,ctx in contexts.items()}
        for method in methods:
            take=[parts['2016','mc']] if scope=='2016' else [parts[y,'mc' if y=='2021' and method!='all_gaussian' or y=='2016' and method=='mc2016_2021' else 'gaussian'] for y in ys]
            r=solve(take,m,method,scope,limit=False);r.update(toy=t,counts_hash=count_hash)
            rows.append(r)
    write(path,dict(signature=signature,rows=rows));return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);args=ap.parse_args();assert 1<=args.workers<=2
    selected=json.loads((B/'results/selected_regions.json').read_text())['regions']
    targets=[(r['scope'],r['mass_MeV']) for r in selected if r['scope']=='2016' or r['region']=='excess_1']
    spec=dict(toys=NTOYS,seed=SEED,source='Fixed archived GP arithmetic mean; independent Poisson bins; recompute GP predictions and profiled nuisance fit',
        targets=targets,methods='MC only at both 2016 regions; all three methods at strongest combined region',
        paired='Identical year-specific counts across methods and selected masses; independent streams across campaigns',
        selection='After observed scan; fixed-mass diagnostic, no scan-wide correction',
        script_sha256=sha(__file__),observed_protocol_sha256=sha(B/'provenance/protocol.json'),
        truths={y:sha(B/f'inputs/null_{y}.npz') for y in ('2015','2016','2021')})
    spec=json.loads(json.dumps(spec));path=B/'provenance/local_check_protocol.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:write(path,spec)
    signature=sha(path);rows=[];start=time.monotonic()
    jobs=[(scope,m,t,signature) for scope,m in targets for t in range(0,NTOYS,BLOCK)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,chunk in enumerate(pool.map(job,jobs),1):
            rows+=chunk
            if i%5==0:print('Local check batches',i,'/',len(jobs),flush=True)
    csv(B/'results/local_toy_rows.csv',rows);toys=pd.DataFrame(rows)
    observed=pd.read_csv(B/'results/observed_scan.csv',float_precision='round_trip');summ=[]
    for (scope,m,method),q in toys.groupby(['scope','mass_MeV','method']):
        r=observed[(observed.scope==scope)&(observed.mass_MeV==m)&(observed.method==method)].iloc[0]
        assert len(q)==NTOYS and q.toy.nunique()==NTOYS and q.valid.all()
        k=int((q.q0>=r.q0).sum())
        summ.append(dict(scope=scope,mass_MeV=int(m),method=method,null_toys=NTOYS,tail_count=k,p_rank=(1+k)/(NTOYS+1),
            cp95_low=0. if k==0 else float(beta.ppf(.025,k,NTOYS-k+1)),
            cp95_high=1. if k==NTOYS else float(beta.ppf(.975,k+1,NTOYS-k)),
            observed_q0=float(r.q0),observed_Z=float(r.Z_local),p0_asymptotic=float(r.p0_asymptotic),
            background_mean_psi=float(q.psi_hat.mean()),background_sd_psi=float(q.psi_hat.std(ddof=1)),
            mean_signed_root=float(q.signed_root.mean()),sd_signed_root=float(q.signed_root.std(ddof=1))))
    combined=toys[toys.scope=='combined']
    assert (combined.groupby('toy').counts_hash.nunique()==1).all()
    csv(B/'results/local_checks.csv',summ)
    write(B/'qa/local_checks.json',dict(passed=True,rows=len(toys),distinct_checks=len(summ),all_expected_toy_ids=True,
        paired_combined_methods=True,max_score=float(toys.max_score.max()),minimum_expected_count=float(toys.min_lambda.min()),
        runtime_seconds=time.monotonic()-start,rows_sha256=sha(B/'results/local_toy_rows.csv'),protocol_sha256=signature))
    print(json.dumps(summ,indent=2),flush=True)

if __name__=='__main__':main()
