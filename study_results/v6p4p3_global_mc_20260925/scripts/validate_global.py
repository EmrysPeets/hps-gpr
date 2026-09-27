"""Audit the frozen inputs, complete scans, paired draws and empirical tails."""
import run_global as R
from run_global import *
from scipy.stats import beta

def main():
    checks=0
    def check(value):
        nonlocal checks
        assert value
        checks+=1
    signature=protocol()
    for line in (B/'provenance/input_manifest.sha256').read_text().splitlines():
        h,p=line.split('  ',1);check(sha(B/p)==h)
    old=json.loads((B/'provenance/previous_report_hashes.json').read_text());preserved=[]
    for p,h in old.items():
        if Path(p).exists():check(sha(p)==h);preserved.append(p)
    observed=pd.read_csv(B/'results/observed_scan.csv',dtype={'scope':str},float_precision='round_trip')
    parent=pd.read_csv(B/'inputs/parent_v641_observed_scan.csv',dtype={'scope':str},float_precision='round_trip')
    a=observed[observed.scope=='2021'].set_index('mass_MeV').signed_root
    b=parent[(parent.scope=='2021')&(parent.method=='mc')].set_index('mass_MeV').signed_root
    old_error=float(np.max(np.abs(a-b)));check(old_error<1e-8)
    check(len(observed)==498);check(observed.valid.all())
    geom=pd.read_csv(B/'results/template_geometry.csv',dtype={'year':str},float_precision='round_trip')
    check(len(geom)==358)
    for row in geom.itertuples():
        c=SelectedContext(row.year,row.mass_MeV);check(np.array_equal(c.fit,c.guard))
        low,high=(-2.5,2.5) if row.year=='2016' else (-4,3)
        if row.year!='2015':
            u=(c.data['x']*1000-c.center)/c.width
            check(np.array_equal(c.fit,(u>=low)&(u<=high)))
            baseline=Context(row.year,row.mass_MeV,'mc');check(np.array_equal(c.categories,baseline.categories))
        else:
            baseline=Context('2015',row.mass_MeV,'gaussian');check(np.array_equal(c.fit,baseline.fit))
            check(np.array_equal(c.categories,baseline.categories))
        check(abs(c.probability[c.fit].sum()-row.fit_fraction)<1e-13)
        check(abs(c.categories.sum()-1)<1e-12);check(c.categories.min()>=0)
    source=[]
    for year in ('2015','2016','2021'):
        z=np.load(B/f'inputs/null_{year}.npz');d=C.DATA[year]
        check(np.array_equal(z['observed'],d['n']));check(np.array_equal(z['edges_GeV'],d['edges']))
        const,ls=C.kernel_state(year,76);check(const==float(z['const']));check(ls==float(z['ls']))
        mean,_=C.predict(d['x'],d['n'],np.zeros(len(d['n']),bool),const,ls,query=d['x'])
        err=float(np.max(np.abs(mean/z['truth']-1)));check(err<1e-11)
        source.append(dict(year=year,all_bins_used=True,anchor_MeV=76,max_relative_reconstruction_error=err))
    R.initialize();draws=np.load(B/'results/toy_draw_hashes.npz')
    for year,counts in R.CALIBRATION.items():
        hashes=[hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in counts]
        check(np.array_equal(hashes,draws[year]));check(len(set(hashes))==NTOYS)
    roots={};maximum={};maximum_score=0.;minimum_lambda=np.inf;replays=0
    for scope in SCOPES:
        z=np.load(B/f'results/global_scans_{scope}.npz');v=z['values'];m=z['masses_MeV']
        check(np.array_equal(z['toy_ids'],np.arange(NTOYS)));check(list(z['fields'])==list(FIELDS))
        check(np.array_equal(m,np.arange(40 if scope=='2016' else 60,176 if scope=='2016' else 241)))
        check(v.shape==(NTOYS,len(m),5));check(np.isfinite(v).all())
        check(np.all(v[:,:,2]>0));check(np.all(v[:,:,3]<3e-5));check(np.all(v[:,:,4]>0))
        maximum_score=max(maximum_score,float(v[:,:,3].max()));minimum_lambda=min(minimum_lambda,float(v[:,:,4].min()))
        for j,mass in enumerate(m):
            p=np.load(B/f'results/checkpoints/m{mass:03d}.npz');check(str(p['signature'])==signature)
            check(np.array_equal(p['toy_ids'],np.arange(NTOYS)));check(np.array_equal(p[scope],v[:,j,:]))
        roots[scope]=v[:,:,0];maximum[scope]=np.maximum(v[:,:,0],0).max(axis=1)**2
    check(np.array_equal(roots['2021'][:,116:],roots['combined'][:,116:]))
    replay_masses=(40,60,67,91,100,175,176,240)
    for mass in replay_masses:
        ctx=at_mass(mass);stored=np.load(B/f'results/checkpoints/m{mass:03d}.npz')
        for c in ctx.values():
            b,v=c.prediction(c.data['n']);bb,vv=C.predict(c.data['x'],c.data['n'],c.guard,c.const,c.ls)
            check(np.allclose(b,bb,rtol=1e-11,atol=1e-8));check(np.allclose(v,vv,rtol=1e-10,atol=1e-7))
        for toy in (0,1,17):
            result=evaluate(mass,ctx,{y:R.CALIBRATION[y][toy] for y in ctx})
            for scope,r in result.items():
                want=stored[scope][toy];got=np.array([r[k] for k in FIELDS])
                check(np.allclose(got,want,rtol=1e-10,atol=1e-9));replays+=1
        result=evaluate(mass,ctx)
        for scope,r in result.items():
            saved=observed[(observed.scope==scope)&(observed.mass_MeV==mass)].iloc[0]
            check(abs(r['signed_root']-saved.signed_root)<1e-9)
    curves=pd.read_csv(B/'results/local_global_curves.csv',dtype={'scope':str},float_precision='round_trip')
    summary=pd.read_csv(B/'results/global_summary.csv',dtype={'scope':str},float_precision='round_trip')
    for scope in SCOPES:
        o=observed[observed.scope==scope].sort_values('mass_MeV');q=np.maximum(roots[scope],0)**2
        cc=curves[curves.scope==scope].sort_values('mass_MeV');check(len(cc)==len(o))
        for j,r in enumerate(cc.itertuples()):
            for prefix,k in [('local',int((q[:,j]>=r.q0).sum())),('global',int((maximum[scope]>=r.q0).sum()))]:
                check(k==getattr(r,prefix+'_exceedances'));check(abs((k+1)/(NTOYS+1)-getattr(r,prefix+'_p_rank'))<1e-15)
                lo=0 if k==0 else beta.ppf(.025,k,NTOYS-k+1);hi=1 if k==NTOYS else beta.ppf(.975,k+1,NTOYS-k)
                check(abs(lo-getattr(r,prefix+'_p95_low'))<1e-15);check(abs(hi-getattr(r,prefix+'_p95_high'))<1e-15)
            check(r.global_exceedances>=r.local_exceedances)
        peak=o.loc[o.q0.idxmax()];r=summary[summary.scope==scope].iloc[0]
        check(r.peak_mass_MeV==peak.mass_MeV);check(r.global_exceedances==int((maximum[scope]>=peak.q0).sum()))
    fam=json.loads((B/'results/family_global.json').read_text());fm=np.max(np.stack(list(maximum.values())),axis=0)
    check(fam['observed_q0']==float(observed.q0.max()));check(fam['exceedances']==int((fm>=fam['observed_q0']).sum()))
    for v in maximum.values():check(np.all(fm>=v))
    result=dict(passed=True,checks=checks,complete_toys=NTOYS,scope_mass_coordinates=498,
        unique_profile_pairs=433*NTOYS,maximum_score=maximum_score,minimum_expected_count=minimum_lambda,
        all_toy_draw_hashes_regenerated=True,paired_scans_verified=True,targeted_toy_profile_replays=replays,
        replay_masses_MeV=list(replay_masses),null_sources_reconstructed=source,unchanged_2021_max_root_difference=old_error,
        previous_reports_preserved=len(preserved),global_summary_sha256=sha(B/'results/global_summary.csv'))
    write(B/'qa/global_validation.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
