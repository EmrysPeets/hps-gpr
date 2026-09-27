"""Two-worker checkpointed observed scan; unchanged 2021 single-year fits reused."""
from extraction import *
from concurrent.futures import ProcessPoolExecutor
import argparse,time

def run_mass(args):
    m,signature=args;dest=B/f'results/checkpoints/m{m:03d}.json'
    if dest.exists():
        q=json.loads(dest.read_text());assert q['signature']==signature
        return q
    rows=[];geometry=[];parts={}
    if m<=175:
        for kind in KINDS2016:
            ctx=Context('2016',m,kind);part=ctx.part();parts['2016',kind]=part;geometry.append(ctx.geometry())
            rows.append(solve([part],m,kind,'2016'))
    if m>=60:
        if m<=100:
            ctx=Context('2015',m,'gaussian');parts['2015','gaussian']=ctx.part();geometry.append(ctx.geometry())
            rows.append(solve([parts['2015','gaussian']],m,'gaussian','2015'))
        for kind in ('gaussian','mc'):
            ctx=Context('2021',m,kind);parts['2021',kind]=ctx.part();geometry.append(ctx.geometry())
        for method in METHODS:
            selected=[]
            for y in years(m):
                kind='mc' if y=='2021' and method!='all_gaussian' or y=='2016' and method=='mc2016_2021' else 'gaussian'
                selected.append(parts[y,kind])
            rows.append(solve(selected,m,method,'combined'))
    out=dict(signature=signature,mass_MeV=m,rows=rows,geometry=geometry)
    write(dest,out);return out

def select_regions(frame,scope,method):
    q=frame[(frame.scope==scope)&(frame.method==method)].sort_values('mass_MeV').reset_index(drop=True)
    v=q.q0.to_numpy();candidates=[]
    for i,r in q.iterrows():
        if r.q0>0 and (i==0 or v[i]>v[i-1]) and (i==len(v)-1 or v[i]>=v[i+1]):candidates.append(r)
    chosen=[];used=[]
    for r in sorted(candidates,key=lambda r:(-r.q0,r.mass_MeV)):
        m=int(r.mass_MeV)
        masks={'2016':Context('2016',m,'mc').fit} if scope=='2016' else {p['year']:p['context'].fit for p in parts_for_combined(m,method)}
        if any(any(y in old and np.any(mask & old[y]) for y,mask in masks.items()) for old in used):continue
        chosen.append(dict(region=f'excess_{len(chosen)+1}',mass_MeV=m,scope=scope,method=method,
            p0_asymptotic=float(r.p0_asymptotic),Z_local=float(r.Z_local),signed_root=float(r.signed_root),q0=float(r.q0)))
        used.append(masks)
        if len(chosen)==2:break
    assert len(chosen)==2
    return chosen

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    assert 1<=args.workers<=2
    signature=protocol();rows=[];geometries=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for q in pool.map(run_mass,[(m,signature) for m in range(40,241)]):
            rows+=q['rows'];geometries+=q['geometry']
            if q['mass_MeV']%20==0:print('Observed mass',q['mass_MeV'],'complete',flush=True)
    old=pd.read_csv(B/'inputs/v638_observed_scan.csv',float_precision='round_trip')
    for r in old[old.policy.isin(['gaussian_baseline','morph_starter'])].itertuples():
        factor=conversion('2021',r.mass_MeV)
        rows.append(dict(mass_MeV=r.mass_MeV,scope='2021',method='gaussian' if r.policy=='gaussian_baseline' else 'mc',
            campaigns='2021',psi_hat=r.Ahat/factor,sigma_psi=r.sigma_A/factor,psi90=r.A90/factor,
            epsilon2_90_ee_proxy=r.A90/factor*1e-8,epsilon2_90_visible_legacy=r.A90/factor*1e-8*branch(r.mass_MeV),
            signed_root=r.signed_r,q0=max(0,r.signed_r)**2,Z_local=max(0,r.signed_r),p0_asymptotic=r.p0_fixed_mass,
            Ahat=r.Ahat,sigma_A=r.sigma_A,A90=r.A90,cls=r.cls,max_score=r.max_score,min_lambda=r.min_lambda,
            max_covariance_load=r.covariance_load,valid=True,origin='reused_unchanged_v638'))
    frame=pd.DataFrame(rows);csv(B/'results/observed_scan.csv',frame);csv(B/'results/template_geometry.csv',geometries)
    g=pd.DataFrame(geometries).drop_duplicates(['year','mass_MeV'])
    csv(B/'results/normalization.csv',g[['year','mass_MeV','conversion_events_per_psi']])
    regions=select_regions(frame,'2016','mc')+select_regions(frame,'combined','mc2016_2021')
    write(B/'results/selected_regions.json',dict(regions=regions,selection='Observed-selected positive local maxima, disjoint actual windows; not independent/global probabilities'))
    for r in regions:
        m=r['mass_MeV'];scope=r['scope']
        for method in KINDS2016 if scope=='2016' else METHODS:
            parts=[Context('2016',m,method).part()] if scope=='2016' else parts_for_combined(m,method)
            row=solve(parts,m,method,scope,save=B/f'results/selected_fits/{scope}_{r["region"]}_m{m:03d}_{method}.npz')
            ref=frame[(frame.scope==scope)&(frame.mass_MeV==m)&(frame.method==method)].iloc[0]
            assert abs(row['psi90']/ref.psi90-1)<2e-8 and abs(row['signed_root']-ref.signed_root)<2e-6
    cats=np.array([BANKS['2016'].categories(m,C.DATA['2016']['edges']*1000) for m in range(40,176)])
    np.savez_compressed(B/'results/template_grid_2016.npz',masses_MeV=np.arange(40,176),
        edges_GeV=C.DATA['2016']['edges'],full_MC_categories=cats,core_centers_MeV=np.array([BANKS['2016'].parameters(m)[0] for m in range(40,176)]),
        core_widths_MeV=np.array([BANKS['2016'].parameters(m)[1] for m in range(40,176)]))
    minima=[]
    for (scope,method),q in frame.groupby(['scope','method']):
        r=q.loc[q.p0_asymptotic.idxmin()]
        minima.append(dict(scope=scope,method=method,mass_MeV=int(r.mass_MeV),p0=float(r.p0_asymptotic),Z=float(r.Z_local),
            epsilon2_90=float(r.epsilon2_90_visible_legacy)))
    write(B/'results/summary.json',dict(version='6.4.1',rows=len(frame),fresh_fit_rows=int((frame.origin=='fresh_fit').sum()),
        reused_2021_rows=int((frame.origin!='fresh_fit').sum()),minima=minima,regions=regions,runtime_seconds=time.monotonic()-start))
    print(json.dumps(dict(minima=minima,selected=regions),indent=2),flush=True)

if __name__=='__main__':main()
