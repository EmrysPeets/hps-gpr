#!/usr/bin/env python3
"""Bounded independent joint calibration replay and inherited-campaign audit."""
from pathlib import Path
import os,sys,json,hashlib,datetime
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.special import ndtr
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'inputs/v6p1/scripts'))
import common as C
MASSES=tuple(range(60,241,20));POLICIES=('pole','logshift');YEARS=('2015','2016','2021');GRID=(0,1,2,3,4,5,6,8,10,12,16,20,24);MASTER=63520210925
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def jread(p):return json.loads(Path(p).read_text())
def read(p):return pd.read_csv(p,keep_default_na=False,float_precision='round_trip')
def close(a,b):assert np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=3e-11,atol=3e-11)
def years(m):return tuple(y for y in YEARS if y=='2021' or y=='2015' and m<=100 or y=='2016' and m<=180)
def factor(y,m):
    d=C.DATA[y];x=m/1000.;s=float(np.polynomial.polynomial.polyval(x,d['sigma_coeffs']));e=d['native_edges']
    overlap=np.maximum(0.,np.minimum(e[1:],x+1.64*s)-np.maximum(e[:-1],x-1.64*s))
    density=float(np.sum(d['native_counts']*overlap/np.diff(e))/(3.28*s))
    return float(3*np.pi*x*float(d['frad_effective'])*density/(2/137.))*1e-8
def categories(y,m):
    if y!='2021':return np.r_[0.,C.signal(y,m)/factor(y,m),0.]
    h=np.load(B/f'inputs/v6p1/histograms/m{m:03d}.npz');total=json.loads(str(h['metadata']))['sumw'];c=np.r_[0.,np.cumsum(h['sumw'])]/total
    p=np.interp(C.DATA[y]['edges'],h['edges_GeV'],c,left=0,right=c[-1]);return np.r_[p[0],np.diff(p),1-p[-1]]
def main():
    checks=[];co=np.load(B/'results/combined_cohorts.npz');protocol=jread(B/'qa/observed_protocol.json');ref=jread(B/'results/combined_reference.json');freeze=jread(B/'results/combined_calibration_freeze.json')
    for r in jread(B/'provenance/input_hashes.json'):assert sha(B/r['path'])==r['sha256']
    assert sha(B/'results/combined_cohorts.npz')==protocol['cohorts_sha256']
    hashes=[]
    for cohort,ns in [('pilot',1),('calibration',2),('evaluation',3)]:
        for yi,y in enumerate(YEARS):
            truth=np.load(B/f'inputs/null_{y}.npz')['truth']
            for t in range(100):
                bg=np.random.default_rng(np.random.SeedSequence([MASTER,ns,yi,t])).poisson(truth)
                assert np.array_equal(bg,co[cohort+'_'+y][t]);hashes.append(ahash(bg))
    assert len(set(hashes))==900;checks.append(dict(name='900 background spectra exact seed replay',passed=True))
    assert ref['pilot_sha256']==sha(B/'results/combined_pilot.csv')
    assert freeze['calibration_sha256']==sha(B/'results/combined_calibration.csv') and freeze['reference_sha256']==sha(B/'results/combined_reference.json')
    signature=ref['signature'];assert signature==freeze['signature']==protocol['signature']
    totals={};drawcount=0
    for cohort,levels in [('pilot',(0,)),('calibration',GRID),('evaluation',(0,1,3,5))]:
        data=read(B/f'results/combined_{cohort}.csv');totals[cohort]=len(data)
        assert len(data)==10*100*2*len(levels) and not data.duplicated(['mass_MeV','toy','z','policy']).any()
        assert data.valid.all() and (data.sigma_psi>0).all() and (data.score<3e-5).all() and (data.min_lambda>0).all()
        close(data.pull,(data.psi_hat-data.psi_expected)/data.sigma_psi)
        for m in MASSES:
            path=B/f'results/combined_{cohort}/m{m:03d}.csv';mark=jread(path.with_suffix('.json'));draws=jread(path.with_suffix('.draws.json'))
            assert mark['signature']==signature and mark['sha256']==sha(path) and mark['draws_sha256']==sha(path.with_suffix('.draws.json'))
            if cohort!='pilot':assert mark['reference_sha256']==sha(B/'results/combined_reference.json') and mark['started_utc']>=ref['frozen_utc']
            if cohort=='evaluation':assert mark['calibration_freeze_sha256']==sha(B/'results/combined_calibration_freeze.json') and mark['started_utc']>=freeze['frozen_utc']
            if cohort=='pilot':
                close(ref['masses'][str(m)]['s0_psi'],data[(data.mass_MeV==m)&(data.policy=='pole')].sigma_psi.mean())
            cats={y:categories(y,m) for y in years(m)};factors={y:factor(y,m) for y in years(m)};seen=set()
            for row in draws:
                t,z,y=row['toy'],row['z'],row['year'];assert (t,z,y) not in seen;seen.add((t,z,y))
                psi=z*ref['masses'][str(m)]['s0_psi'] if cohort!='pilot' else 0.
                expected=psi*factors[y];close(row['expected_total'],expected)
                a=np.zeros(len(cats[y]),dtype=np.int64) if z==0 else np.random.default_rng(np.random.SeedSequence([MASTER,20 if cohort=='calibration' else 30,YEARS.index(y),m,t,z])).poisson(expected*cats[y])
                assert ahash(a)==row['sha256'] and a.sum()==row['drawn_total'];drawcount+=1
            assert len(seen)==100*len(levels)*len(years(m))
            g=data[data.mass_MeV==m];assert (g.campaigns=='+'.join(years(m))).all()
            assert set(map(tuple,g[['toy','z','policy']].to_numpy()))=={(t,z,p) for t in range(100) for z in levels for p in POLICIES}
    checks.append(dict(name='joint rows signals hashes and freeze order',passed=True,rows=totals,signal_vectors_replayed=drawcount))
    obs=read(B/'results/observed_dense.csv');assert len(obs)==724 and not obs.duplicated(['mass_MeV','policy','scope']).any()
    assert obs.valid.all() and (obs.max_score<3e-5).all() and (obs.min_lambda>0).all();close(obs.epsilon2_90,obs.psi90*1e-8)
    for m in range(60,241):
        p=B/f'results/observed_chunks/m{m:03d}.csv';mark=jread(p.with_suffix('.json'));assert mark['signature']==signature and mark['sha256']==sha(p)
        assert set(obs[obs.mass_MeV==m].scope)=={'2021','combined'}
    checks.append(dict(name='724 dense conditional observed rows and checkpoint signatures',passed=True,electron_channel_proxy=True))
    cal=read(B/'results/combined_calibration.csv');ev=read(B/'results/combined_evaluation.csv');test=read(B/'results/combined_evaluation_rank.csv');observed=read(B/'results/combined_observed_rank.csv')
    topology=[]
    for m in MASSES:
        for policy in POLICIES:
            cell=cal[(cal.mass_MeV==m)&(cal.policy==policy)];arrays={z:cell[cell.z==z].psi_hat.to_numpy() for z in GRID};s0=ref['masses'][str(m)]['s0_psi']
            for row in ev[(ev.mass_MeV==m)&(ev.policy==policy)].itertuples(index=False):
                rank=np.array([(1+np.count_nonzero(arrays[z]<=row.psi_hat))/101 for z in GRID]);keep=rank>.1;upper=max(np.array(GRID)[keep],default=0)*s0
                r=test[(test.mass_MeV==m)&(test.policy==policy)&(test.toy==row.toy)&(test.z==row.z)].iloc[0]
                close(r.p_at_truth,rank[GRID.index(row.z)]);close(r.upper_psi,upper)
                assert r.truth_accepted==(rank[GRID.index(row.z)]>.1) and r.upper_covers==(upper>=row.psi_expected)
                assert r['empty']==(not keep.any()) and r.right_censored==keep[-1]
                accepted=np.flatnonzero(keep)
                internal=bool(len(accepted)>1 and (~keep[accepted[0]:accepted[-1]+1]).any())
                topology.append(dict(kind='evaluation',mass_MeV=m,policy=policy,toy=row.toy,z=row.z,leading_rejected=bool(keep.any() and not keep[0]),internal_holes=internal,legacy_holes=bool(r.holes)))
            o=obs[(obs.mass_MeV==m)&(obs.policy==policy)&(obs.scope=='combined')].iloc[0];r=observed[(observed.mass_MeV==m)&(observed.policy==policy)].iloc[0]
            rank=np.array([(1+np.count_nonzero(arrays[z]<=o.psi_hat))/101 for z in GRID]);close(json.loads(r.grid_rank_p),rank)
            k=np.count_nonzero(cell[cell.z==0].signed_root.astype(float)>=o.signed_root);assert r.null_exceedances==k;close(r.p0_rank,(k+1)/101)
            keep=rank>.1;accepted=np.flatnonzero(keep)
            topology.append(dict(kind='observed',mass_MeV=m,policy=policy,toy=-1,z=-1,leading_rejected=bool(keep.any() and not keep[0]),internal_holes=bool(len(accepted)>1 and (~keep[accepted[0]:accepted[-1]+1]).any()),legacy_holes=bool(r.holes)))
    checks.append(dict(name='independent8000 heldout and20 observed rank inversions',passed=True))
    topo=pd.DataFrame(topology);topo.to_csv(B/'results/combined_set_topology.csv',index=False)
    checks.append(dict(name='explicit accepted-set topology',passed=True,internal_holes=int(topo.internal_holes.sum()),leading_rejected=int(topo.leading_rejected.sum()),legacy_holes=int(topo.legacy_holes.sum()),definition='Legacy holes includes leading rejections; internal_holes counts missing points strictly between accepted endpoints.'))
    # Independently reconstruct inherited old-campaign contexts; no toy fitting.
    import core as K
    inherited=[]
    for y,m in [('2015',60),('2015',100),('2016',60),('2016',180)]:
        psi=3*ref['masses'][str(m)]['s0_psi'];draw=np.random.default_rng(np.random.SeedSequence([MASTER,30,YEARS.index(y),m,0,3])).poisson(psi*factor(y,m)*categories(y,m));counts=co['evaluation_'+y][0]+draw[1:-1]
        prior=C.moving_context(y,m,counts=counts);current=K.Context(m,'pole',year=y);alternate=K.Context(m,'logshift',year=y);b,L,d=current.predict(counts);ba,La,da=alternate.predict(counts)
        assert np.array_equal(prior['mask'],current.mask) and np.array_equal(current.mask,alternate.mask)
        close(prior['b'],b);close(prior['L']@prior['L'].T,L@L.T);assert np.array_equal(b,ba) and np.array_equal(L,La)
        close(prior['S'][:,0],C.signal(y,m,current.mask))
        inherited.append(dict(year=y,mass_MeV=m,cohort='evaluation',toy=0,z=3,counts_sha256=ahash(counts),mean_sha256=ahash(b),covariance_factor_sha256=ahash(L),signal_sha256=ahash(prior['S'][:,0]),policy_parts_exactly_equal=True))
    checks.append(dict(name='inherited2015/2016 fit masks means covariance and signal units',passed=True,representative_states=inherited))
    out=dict(passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,signature=signature,
        scientific_nonclosure_is_not_validation_failure=True,no_rare_tail_or_global_p0_calibration=True)
    (B/'qa/observed_independent_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
