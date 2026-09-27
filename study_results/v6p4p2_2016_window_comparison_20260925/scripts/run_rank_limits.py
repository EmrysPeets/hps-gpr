"""Selected-point inversion with a common full-MC signal source across methods."""
from extraction import *
from concurrent.futures import ProcessPoolExecutor
import time

N=200;GRID=(0.,.5,1.,2.,3.,4.,5.,6.,8.,10.,12.,16.);SEED=641250925;BLOCK=20

def contexts(scope,m):
    ys=['2016'] if scope=='2016' else years(m)
    methods=list(KINDS2016) if scope=='2016' else list(METHODS)
    keys={('2016',k) for k in KINDS2016} if scope=='2016' else {(y,'mc' if y=='2021' and method!='all_gaussian' or y=='2016' and method=='mc2016_2021' else 'gaussian') for method in methods for y in ys}
    return ys,methods,{key:Context(*key[:1],m,key[1]) for key in keys}

def take_parts(scope,method,ys,parts):
    return [parts['2016',method]] if scope=='2016' else [parts[y,'mc' if y=='2021' and method!='all_gaussian' or y=='2016' and method=='mc2016_2021' else 'gaussian'] for y in ys]

def draw_hash(data):return hashlib.sha256(b''.join(np.ascontiguousarray(data[y]).tobytes() for y in sorted(data))).hexdigest()

def run_block(args):
    scope,m,first,s0,signature=args
    path=B/f'results/rank_checkpoints/{scope}_m{m:03d}_t{first:04d}.json'
    if path.exists():
        q=json.loads(path.read_text());assert q['signature']==signature;return q['rows']
    ys,methods,ctx=contexts(scope,m)
    truths={y:np.load(B/f'inputs/null_{y}.npz')['truth'] for y in ys}
    cats={y:Context(y,m,'gaussian' if y=='2015' else 'mc').categories for y in ys}
    factors={y:conversion(y,m) for y in ys};rows=[]
    for toy in range(first,min(first+BLOCK,N)):
        background={y:np.random.default_rng(np.random.SeedSequence([SEED,2,int(y),toy])).poisson(truths[y]) for y in ys}
        rng={y:np.random.default_rng(np.random.SeedSequence([SEED,3,m,int(y),toy])) for y in ys}
        signal={y:np.zeros(len(cats[y]),dtype=np.int64) for y in ys};previous=0.;bh=draw_hash(background)
        for z in GRID:
            for y in ys:signal[y]+=rng[y].poisson((z-previous)*s0*factors[y]*cats[y])
            previous=z;counts={y:background[y]+signal[y][1:-1] for y in ys}
            parts={key:context.part(counts[key[0]]) for key,context in ctx.items()}
            sh,ch=draw_hash(signal),draw_hash(counts)
            for method in methods:
                chosen=take_parts(scope,method,ys,parts);r=solve(chosen,m,method,scope,limit=False)
                r.update(toy=toy,z=z,psi_expected=z*s0,s0_psi=s0,background_hash=bh,signal_hash=sh,counts_hash=ch,
                    realized_full_signal=sum(int(signal[y].sum()) for y in ys),
                    realized_signal_in_fit=sum(int(signal[p['year']][1:-1][p['context'].fit].sum()) for p in chosen),
                    realized_signal_in_training=sum(int(signal[p['year']][1:-1][~p['context'].guard].sum()) for p in chosen),
                    realized_signal_outside_support=sum(int(signal[y][0]+signal[y][-1]) for y in ys))
                rows.append(r)
    write(path,dict(signature=signature,rows=rows));return rows

def main():
    (B/'results/rank_checkpoints').mkdir(exist_ok=True)
    selected=json.loads((B/'results/selected_regions.json').read_text())['regions']
    targets=[(r['scope'],r['mass_MeV']) for r in selected if r['scope']=='2016' or r['region']=='excess_1']
    scales={}
    for scope,m in targets:
        ys,methods,ctx=contexts(scope,m);truths={y:np.load(B/f'inputs/null_{y}.npz')['truth'] for y in ys}
        parts={key:context.part(truths[key[0]]) for key,context in ctx.items()}
        kind='gaussian' if scope=='2016' else 'all_gaussian'
        row=solve(take_parts(scope,kind,ys,parts),m,kind,scope,limit=False)
        scales[f'{scope}_{m}']=row['sigma_psi']
    spec=dict(calibration_toys=N,strength_grid=list(GRID),targets=targets,reference_scales_psi=scales,
        reference_scale='Gaussian baseline Hessian error on fixed GPmean-source counts, independent of observed fitted yield',
        source='Full neighboring MC for2016 and2021, retained Gaussian2015; shared psi normalization. Independent Poisson signal categories including both support losses; nested increments and paired backgrounds across extraction methods.',
        ordering='p_psi=(1+count(calibration fitted psi <= observed fitted psi))/201; accept if >0.10; retain all grid points, holes, empty and censor flags',
        seed=SEED,background_namespace=2,signal_namespace=3,
        scope='Pointwise conditional grid inversion on a fixed estimated background source; no coverage guarantee or global significance',
        script_sha256=sha(__file__),observed_protocol_sha256=sha(B/'provenance/protocol.json'))
    spec=json.loads(json.dumps(spec));path=B/'provenance/rank_limit_protocol.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:write(path,spec)
    signature=sha(path);jobs=[(scope,m,t,scales[f'{scope}_{m}'],signature) for scope,m in targets for t in range(0,N,BLOCK)]
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,q in enumerate(pool.map(run_block,jobs),1):
            rows+=q
            if i%5==0:print('Rank-limit batches',i,'/',len(jobs),flush=True)
    csv(B/'results/rank_calibration_rows.csv',rows);d=pd.DataFrame(rows)
    obs=pd.read_csv(B/'results/observed_scan.csv',float_precision='round_trip');summ=[];gridrows=[]
    for (scope,m,method),q in d.groupby(['scope','mass_MeV','method']):
        assert len(q)==N*len(GRID) and q.valid.all()
        o=obs[(obs.scope==scope)&(obs.mass_MeV==m)&(obs.method==method)].iloc[0]
        s0=scales[f'{scope}_{m}'];pv=[]
        for z in GRID:
            zz=q[q.z==z];assert zz.toy.nunique()==N
            k=int((zz.psi_hat<=o.psi_hat).sum());p=(1+k)/(N+1);pv.append(p)
            gridrows.append(dict(scope=scope,mass_MeV=int(m),method=method,z=z,psi_expected=z*s0,calibration_toys=N,
                lower_tail_count=k,p_rank=p,accepted=p>.1,observed_psi_hat=o.psi_hat))
        inds=np.flatnonzero(np.array(pv)>.1);empty=len(inds)==0
        endpoint=None if empty else GRID[inds[-1]]*s0
        next_rejected=None if empty or inds[-1]==len(GRID)-1 else GRID[inds[-1]+1]*s0
        summ.append(dict(scope=scope,mass_MeV=int(m),method=method,s0_psi=s0,
            accepted_z_json=json.dumps([GRID[i] for i in inds]),p_rank_json=json.dumps(pv),
            empty=empty,holes=False if empty else bool(len(inds)!=(inds[-1]-inds[0]+1)),
            upper_grid_censored=bool(not empty and inds[-1]==len(GRID)-1),
            largest_accepted_psi=endpoint,next_grid_psi=next_rejected,
            largest_accepted_epsilon2=None if endpoint is None else endpoint*1e-8*branch(m),
            next_grid_epsilon2=None if next_rejected is None else next_rejected*1e-8*branch(m),
            largest_accepted_full_yield_2016=None if scope!='2016' or endpoint is None else endpoint*conversion('2016',m),
            next_grid_full_yield_2016=None if scope!='2016' or next_rejected is None else next_rejected*conversion('2016',m)))
    csv(B/'results/rank_limits.csv',summ);csv(B/'results/rank_grid.csv',gridrows)
    groups=d.groupby(['scope','mass_MeV','toy','z'])
    assert (groups.counts_hash.nunique()==1).all() and (groups.signal_hash.nunique()==1).all()
    assert np.all(d.realized_full_signal==d.realized_signal_in_fit+d.realized_signal_in_training+d.realized_signal_outside_support)
    write(B/'qa/rank_limits.json',dict(passed=True,fit_rows=len(d),methods_and_regions=len(summ),
        full_probability_categories_retained=True,paired_across_methods=True,all_grid_cells_complete=True,
        rows_sha256=sha(B/'results/rank_calibration_rows.csv'),protocol_sha256=signature,runtime_seconds=time.monotonic()-start,
        any_empty=any(r['empty'] for r in summ),any_holes=any(r['holes'] for r in summ),any_censored=any(r['upper_grid_censored'] for r in summ)))
    print(json.dumps(summ,indent=2),flush=True)

if __name__=='__main__':main()
