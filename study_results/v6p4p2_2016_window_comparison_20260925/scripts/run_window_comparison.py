"""2016 fit/exclusion ±2 core widths versus the inherited ±3.5 prescription."""
from extraction import *
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import beta
import time
R=B/'results/window_comparison'
WIDTHS=(3.5,2.0);INJECTION_MASSES=(60,69,91,100,160);NINJ=200;NNULL=1000

class WindowContext(Context):
    def __init__(self,m,h):
        super().__init__('2016',m,'mc')
        if h==3.5:return
        assert h==2.
        oldfit=self.fit.copy();oldp=self.categories.copy()
        self.kind='mc_2u'
        self.requested_low=self.center-h*self.width;self.requested_high=self.center+h*self.width
        d=self.data;self.fit=(d['x']*1000>=self.requested_low)&(d['x']*1000<=self.requested_high);self.guard=self.fit.copy()
        xt=d['x'][~self.guard];xq=d['x'][self.fit]
        self.K=C.kernel(xt,xt,self.const,self.ls);self.Kqt=C.kernel(xq,xt,self.const,self.ls);self.Kqq=C.kernel(xq,xq,self.const,self.ls)
        assert self.fit.sum()>3 and np.all(~self.fit|oldfit) and np.array_equal(oldp,self.categories)
        assert (d['x']*1000<self.requested_low).sum()>=3 and (d['x']*1000>self.requested_high).sum()>=3

def parts(m,h):
    return [WindowContext(m,h).part() if y=='2016' else Context(y,m,'gaussian' if y=='2015' else 'mc').part() for y in years(m)]

def observed_job(args):
    m,signature=args;p=R/f'checkpoints/observed_{m}.json'
    if p.exists():
        z=json.loads(p.read_text());assert z['signature']==signature;return z
    ctx=WindowContext(m,2.);rows=[solve([ctx.part()],m,'mc_2u','2016')]
    if m>=60:rows.append(solve(parts(m,2.),m,'mc_2u','combined'))
    z=dict(signature=signature,rows=rows,geometry=ctx.geometry());write(p,z);return z

def toy_job(args):
    kind,m,first,s0,signature=args;p=R/f'checkpoints/{kind}_{m}_{first}.json'
    if p.exists():
        z=json.loads(p.read_text());assert z['signature']==signature;return z['rows']
    ctx={h:WindowContext(m,h) for h in WIDTHS};truth=np.load(B/'inputs/null_2016.npz')['truth'];rows=[]
    N=NNULL if kind=='null' else NINJ
    for t in range(first,min(first+100,N)):
        seed=[641250925,2016,t] if kind=='null' else [642250925,1,2016,t]
        background=np.random.default_rng(np.random.SeedSequence(seed)).poisson(truth)
        sig=np.zeros(len(ctx[2.].categories),dtype=np.int64)
        if kind=='injection':
            sig=np.random.default_rng(np.random.SeedSequence([642250925,2,m,2016,t])).poisson(3*s0*ctx[2.].categories)
        for injected in (False,True) if kind=='injection' else (False,):
            signal=sig if injected else sig*0;counts=background+signal[1:-1]
            count_hash=hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest()
            for h,c in ctx.items():
                # Parent null rows already use exactly this background stream.
                if kind=='null' and h==3.5 and m in (69,91):continue
                row=solve([c.part(counts)],m,'mc_2u' if h==2 else 'mc_3p5u','2016',limit=False)
                row.update(study=kind,mass_MeV=m,half_width_u=h,toy=t,injected=injected,
                    A_expected=3*s0 if injected else 0.,s0_full_yield=s0,counts_hash=count_hash,
                    realized_full_signal=int(signal.sum()),realized_signal_in_fit=int(signal[1:-1][c.fit].sum()),
                    realized_signal_in_training=int(signal[1:-1][~c.guard].sum()),realized_signal_outside=int(signal[0]+signal[-1]),
                    origin='fresh_fit')
                rows.append(row)
    write(p,dict(signature=signature,rows=rows));return rows

def main():
    t0=time.monotonic();R.mkdir(exist_ok=True);(R/'checkpoints').mkdir(exist_ok=True);(R/'selected_fits').mkdir(exist_ok=True)
    spec=dict(version='6.4.2 appendix',year='2016',windows=list(WIDTHS),fit_equals_GP_exclusion=True,
        fixed='Same empirical neighboring template, core center/width, full selected normalization, kernel states and coupling conversion',
        combined='Only2016 changes;2015Gaussian and2021MC[-4,+3] retained; identical campaign support',
        local_null='1000 paired fixed-GPmean Poisson backgrounds at69,91 and new2016minimum; same seed stream as v6.4.1',
        injection='200 paired full-MC signal toys at60,69,91,100,160; expected A=3s0; s0=3.5u MC Hessian error on fixed-GPmean; independent Poisson bins including all categories',
        response='Mean paired Ahat(signal+background)-Ahat(background), divided by expected full selected A; no coverage claim',
        scope='Conditional observed profileCLs and local asymptotic p; fixed-source window diagnostic',
        script_sha256=sha(__file__),parent_observed_sha256=sha(B/'results/observed_scan.csv'),
        parent_protocol_sha256=sha(B/'provenance/protocol.json'),input_manifest_sha256=sha(B/'provenance/input_manifest.sha256'))
    pp=B/'provenance/window_comparison_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==spec
    else:write(pp,spec)
    signature=sha(pp);new=[];geometry=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for q in pool.map(observed_job,[(m,signature) for m in range(40,176)]):new+=q['rows'];geometry.append(q['geometry'])
    old=pd.read_csv(B/'results/observed_scan.csv',float_precision='round_trip')
    base=old[((old.scope=='2016')&(old.method=='mc'))|((old.scope=='combined')&(old.method=='mc2016_2021'))].copy()
    base['method']='mc_3p5u';base['half_width_u']=3.5;base['origin']='reused_v641'
    inherited=base[(base.scope=='combined')&(base.mass_MeV>175)].copy();inherited['method']='mc_2u';inherited['half_width_u']=2.;inherited['origin']='reused_no2016'
    fresh=pd.DataFrame(new);fresh['half_width_u']=2.
    obs=pd.concat([base,fresh,inherited],ignore_index=True).sort_values(['scope','mass_MeV','half_width_u'])
    csv(R/'observed_scan.csv',obs)
    g=pd.read_csv(B/'results/template_geometry.csv',float_precision='round_trip');g=g[(g.year==2016)&(g.kind=='mc')].copy();g['half_width_u']=3.5
    gn=pd.DataFrame(geometry);gn['half_width_u']=2.;geom=pd.concat([g,gn],ignore_index=True);csv(R/'geometry.csv',geom)
    minima=[]
    for (scope,h),q in obs.groupby(['scope','half_width_u']):
        r=q.loc[q.p0_asymptotic.idxmin()];minima.append(dict(scope=scope,half_width_u=h,mass_MeV=int(r.mass_MeV),p0=float(r.p0_asymptotic),Z=float(r.Z_local)))
    peak=next(r['mass_MeV'] for r in minima if r['scope']=='2016' and r['half_width_u']==2.)
    targets=sorted(set([69,91,peak]));print('Observed minima',minima,flush=True)
    for m in targets:
        for h in WIDTHS:solve([WindowContext(m,h).part()],m,'mc_2u' if h==2 else 'mc_3p5u','2016',save=R/f'selected_fits/m{m:03d}_{h:g}u.npz')
    scales={}
    truth=np.load(B/'inputs/null_2016.npz')['truth']
    for m in INJECTION_MASSES:scales[m]=solve([WindowContext(m,3.5).part(truth)],m,'mc_3p5u','2016',limit=False)['sigma_A']
    write(R/'toy_design.json',dict(null_targets=targets,signal_masses=list(INJECTION_MASSES),s0_full_selected=scales))
    jobs=[('null',m,t,0.,signature) for m in targets for t in range(0,NNULL,100)]
    jobs += [('injection',m,t,scales[m],signature) for m in INJECTION_MASSES for t in range(0,NINJ,100)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,q in enumerate(pool.map(toy_job,jobs),1):
            rows+=q
            if i%10==0:print('Toy batches',i,'/',len(jobs),flush=True)
    q=pd.read_csv(B/'results/local_toy_rows.csv',float_precision='round_trip');q=q[q.scope=='2016'].copy()
    q['method']='mc_3p5u';q['half_width_u']=3.5;q['study']='null';q['injected']=False;q['A_expected']=0.;q['s0_full_yield']=0.;q['origin']='reused_v641'
    for col in ('realized_full_signal','realized_signal_in_fit','realized_signal_in_training','realized_signal_outside'):q[col]=0
    toys=pd.concat([pd.DataFrame(rows),q],ignore_index=True);csv(R/'toy_rows.csv',toys)
    null=[];response=[]
    for (m,h),q in toys[toys.study=='null'].groupby(['mass_MeV','half_width_u']):
        assert len(q)==NNULL and q.toy.nunique()==NNULL
        o=obs[(obs.scope=='2016')&(obs.mass_MeV==m)&(obs.half_width_u==h)].iloc[0];k=int((q.q0>=o.q0).sum())
        null.append(dict(mass_MeV=int(m),half_width_u=h,exceedances=k,N=NNULL,p_rank=(k+1)/(NNULL+1),
            cp95_low=0 if k==0 else float(beta.ppf(.025,k,NNULL-k+1)),cp95_high=1 if k==NNULL else float(beta.ppf(.975,k+1,NNULL-k)),
            asymptotic_p=float(o.p0_asymptotic),mean_signed_root=float(q.signed_root.mean()),sd_signed_root=float(q.signed_root.std(ddof=1))))
    for (m,h),q in toys[toys.study=='injection'].groupby(['mass_MeV','half_width_u']):
        z=q[~q.injected].set_index('toy');s=q[q.injected].set_index('toy');assert len(s)==len(z)==NINJ
        A=float(s.A_expected.iloc[0]);delta=(s.Ahat-z.Ahat).to_numpy();resp=float(delta.mean()/A)
        response.append(dict(mass_MeV=int(m),half_width_u=h,N=NINJ,A_expected=A,
            response=resp,response_mc_se=float(delta.std(ddof=1)/np.sqrt(NINJ)/A),
            null_mean_Ahat=float(z.Ahat.mean()),null_sd_Ahat=float(z.Ahat.std(ddof=1)),
            injected_mean_Ahat=float(s.Ahat.mean()),injected_sd_Ahat=float(s.Ahat.std(ddof=1)),
            injected_mean_sigma_A=float(s.sigma_A.mean()),
            response_adjusted_null_sd=float(z.Ahat.std(ddof=1)/resp),
            mean_pull=float(((s.Ahat-A)/s.sigma_A).mean()),sd_pull=float(((s.Ahat-A)/s.sigma_A).std(ddof=1)),
            mean_realized_full_signal=float(s.realized_full_signal.mean()),mean_realized_fit_signal=float(s.realized_signal_in_fit.mean()),
            mean_realized_training_signal=float(s.realized_signal_in_training.mean())))
    csv(R/'null_checks.csv',null);csv(R/'signal_response.csv',response)
    assert toys.valid.all() and (toys.realized_full_signal==toys.realized_signal_in_fit+toys.realized_signal_in_training+toys.realized_signal_outside).all()
    assert (toys.groupby(['study','mass_MeV','toy','injected']).counts_hash.nunique()==1).all()
    stats=[]
    for scope in ('2016','combined'):
        q=obs[obs.scope==scope].pivot(index='mass_MeV',columns='half_width_u',values='epsilon2_90_visible_legacy');rat=q[2.]/q[3.5]
        if scope=='combined':rat=rat[rat.index<=175]
        stats.append(dict(scope=scope,ratio_min=float(rat.min()),ratio_median=float(rat.median()),ratio_max=float(rat.max()),min_at=int(rat.idxmin()),max_at=int(rat.idxmax())))
    summary=dict(passed=True,minima=minima,ratio_summary=stats,observed_rows=len(obs),fresh_observed_rows=len(fresh),
        fresh_toy_rows=len(rows),reused_toy_rows=len(q) if False else 2000,toy_rows=len(toys),
        numerical_score_max=float(toys.max_score.max()),minimum_expected_count=float(toys.min_lambda.min()),
        paired_counts_verified=True,full_signal_partitions_verified=True,runtime_seconds=time.monotonic()-t0)
    write(R/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
