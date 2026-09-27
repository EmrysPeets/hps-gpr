"""Complete fixed-source Poisson mass scans with the selected 2016 ±2.5u window."""
from extraction import *
from concurrent.futures import ProcessPoolExecutor
import argparse,time
NTOYS=1024;SEED=643250925;FIELDS=('signed_root','psi_hat','sigma_psi','max_score','min_lambda')
SCOPES=('2016','2021','combined');CALIBRATION=None

class SelectedContext(Context):
    def __init__(self,y,m):
        super().__init__(y,m,'gaussian' if y=='2015' else 'mc')
        if y!='2016':return
        self.kind='mc_2p5u';self.requested_low=self.center-2.5*self.width;self.requested_high=self.center+2.5*self.width
        d=self.data;self.fit=(d['x']*1000>=self.requested_low)&(d['x']*1000<=self.requested_high);self.guard=self.fit.copy()
        xt=d['x'][~self.guard];xq=d['x'][self.fit]
        self.K=C.kernel(xt,xt,self.const,self.ls);self.Kqt=C.kernel(xq,xt,self.const,self.ls);self.Kqq=C.kernel(xq,xq,self.const,self.ls)
        assert self.fit.sum()>3 and np.sum(d['x']*1000<self.requested_low)>=3 and np.sum(d['x']*1000>self.requested_high)>=3

def at_mass(m):
    ys=['2016'] if m<60 else years(m)
    return {y:SelectedContext(y,m) for y in ys}

def evaluate(m,ctx,counts=None,save=False):
    ps={y:c.part(None if counts is None else counts[y]) for y,c in ctx.items()};out={}
    for y in ('2016','2021'):
        if y in ps:
            dest=B/f'results/selected_fits/{y}_m{m:03d}.npz' if save else None
            out[y]=solve([ps[y]],m,'selected_mc',y,save=dest,limit=False)
    if m>=60:
        if len(ps)==1:out['combined']=dict(out['2021'],scope='combined')
        else:
            dest=B/f'results/selected_fits/combined_m{m:03d}.npz' if save else None
            out['combined']=solve([ps[y] for y in years(m)],m,'selected_mc','combined',save=dest,limit=False)
    return out

def protocol():
    spec=dict(version='6.4.3',toys=NTOYS,seed=SEED,calibration_namespace=1,
        spectra='Archived full2015, full2016 and2021 10percent',
        domains_MeV={'2016':[40,175],'2021':[60,240],'combined':[60,240]},step_MeV=1,
        windows={'2016':[-2.5,2.5],'2021':[-4,3],'2015':'retained Gaussian +/-2.25 reference sigma'},
        center_and_shape='Full selected neighboring MC, linear core center/width; same interpolated MC shapes as v6.4.1/2',
        fit_equals_exclusion=True,combined='Common inherited coupling coordinate; independent campaign GP constraints;2015 through100,2016 through175,2021 through240',
        statistic='T=max_m max(0,signed profile likelihood root(m))^2; no null-mean subtraction or masswise rescaling',
        background_source='Fixed archived GP arithmetic-mean spectrum per campaign; independent Poisson bins and campaigns; no GP-function draws',
        repeated_procedure='Same full campaign draw at all masses and scopes; recompute GP prediction,covariance and nuisance profiles for every mass and toy; fixed archived kernel settings',
        tail='Exceedances Ttoy>=Tobserved; add-one rank(k+1)/(N+1); exact two-sided95percent Clopper-Pearson interval for exceedance probability',
        scope='Mass-grid global conditional on fixed source, selected windows and templates; does not calibrate prior method/window/source selection or a continuous mass supremum',
        script_sha256=sha(__file__),engine_sha256=sha(B/'scripts/extraction.py'),
        inputs_sha256=sha(B/'provenance/input_manifest.sha256'))
    path=B/'provenance/global_protocol.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:write(path,spec)
    return sha(path)

def observed():
    rows=[];geometry=[];start=time.monotonic()
    for m in range(40,241):
        ctx=at_mass(m);rows.extend(evaluate(m,ctx).values());geometry.extend(c.geometry() for c in ctx.values())
    csv(B/'results/observed_scan.csv',rows);csv(B/'results/template_geometry.csv',geometry)
    frame=pd.DataFrame(rows);peaks=[]
    for scope,q in frame.groupby('scope'):
        r=q.loc[q.q0.idxmax()];peaks.append(dict(scope=scope,mass_MeV=int(r.mass_MeV),q0=float(r.q0),signed_root=float(r.signed_root),p0_asymptotic=float(r.p0_asymptotic)))
        evaluate(int(r.mass_MeV),at_mass(int(r.mass_MeV)),save=True)
    write(B/'results/observed_peaks.json',dict(peaks=peaks,runtime_seconds=time.monotonic()-start))
    print('Observed peaks',peaks,flush=True)

def initialize(n=NTOYS,namespace=1):
    global CALIBRATION
    CALIBRATION={}
    for y in ('2015','2016','2021'):
        truth=np.load(B/f'inputs/null_{y}.npz')['truth']
        CALIBRATION[y]=np.array([np.random.default_rng(np.random.SeedSequence([SEED,namespace,int(y),t])).poisson(truth) for t in range(n)])

def mass_job(args):
    m,signature=args;p=B/f'results/checkpoints/m{m:03d}.npz'
    if p.exists():
        with np.load(p) as z:
            assert str(z['signature'])==signature and len(z['toy_ids'])==NTOYS
        return dict(mass_MeV=m,cached=True)
    ctx=at_mass(m);start=time.monotonic();out={};score=0.;minimum=np.inf
    for t in range(NTOYS):
        rows=evaluate(m,ctx,{y:CALIBRATION[y][t] for y in ctx})
        for scope,r in rows.items():
            if scope not in out:out[scope]=np.empty((NTOYS,len(FIELDS)))
            out[scope][t]=[r[k] for k in FIELDS];score=max(score,r['max_score']);minimum=min(minimum,r['min_lambda'])
    for a in out.values():assert np.isfinite(a).all()
    payload=dict(signature=signature,toy_ids=np.arange(NTOYS),fields=np.array(FIELDS),mass_MeV=m,**out)
    tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,**payload);tmp.replace(p)
    return dict(mass_MeV=m,cached=False,seconds=time.monotonic()-start,max_score=score,min_lambda=minimum)

def assemble(signature):
    initialize();hashes={}
    for y,a in CALIBRATION.items():hashes[y]=np.array([hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in a])
    np.savez_compressed(B/'results/toy_draw_hashes.npz',toy_ids=np.arange(NTOYS),**hashes)
    for scope in SCOPES:
        masses=np.arange(40 if scope=='2016' else 60,176 if scope=='2016' else 241);a=[]
        for m in masses:
            with np.load(B/f'results/checkpoints/m{m:03d}.npz') as z:
                assert str(z['signature'])==signature;a.append(z[scope])
        a=np.stack(a,axis=1);np.savez_compressed(B/f'results/global_scans_{scope}.npz',masses_MeV=masses,toy_ids=np.arange(NTOYS),fields=np.array(FIELDS),values=a)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['observed','benchmark','calibrate','assemble','all'],default='all');args=ap.parse_args()
    signature=protocol()
    if args.mode in ('observed','all'):observed()
    if args.mode=='benchmark':
        initialize(4,0);timings=[]
        for m in (45,68,91,120,160,220):
            ctx=at_mass(m);t=time.monotonic()
            for j in range(4):evaluate(m,ctx,{y:CALIBRATION[y][j] for y in ctx})
            timings.append(dict(mass=m,seconds_per_full_mass=(time.monotonic()-t)/4))
        write(B/'qa/timing_pilot.json',dict(excluded_from_calibration=True,timings=timings));print(timings);return
    if args.mode in ('calibrate','all'):
        start=time.monotonic();ledger=[]
        with ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
            for i,r in enumerate(pool.map(mass_job,[(m,signature) for m in range(40,241)]),1):
                ledger.append(r);write(B/'qa/global_progress.json',dict(completed_mass_jobs=i,total_mass_jobs=201,elapsed_seconds=time.monotonic()-start,last=r))
                if i%5==0:print('Complete mass jobs',i,'/201; seconds',round(time.monotonic()-start,1),flush=True)
        write(B/'qa/global_execution.json',dict(passed=True,mass_jobs=ledger,elapsed_seconds=time.monotonic()-start,toys=NTOYS,workers=2,numerical_threads=1))
    if args.mode in ('assemble','calibrate','all'):assemble(signature)

if __name__=='__main__':main()
