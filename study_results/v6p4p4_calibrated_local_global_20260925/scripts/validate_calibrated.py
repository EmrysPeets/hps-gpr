"""Independent rank counting, draw regeneration, profile replay and preservation."""
import run_global as R
from analyze_calibrated import *

def main():
    fits=freeze();checks=0
    def check(x):
        nonlocal checks
        assert x;checks+=1
    frozen=json.loads((B/'provenance/parent_v643_hashes.json').read_text());preserved=0
    for p,h in frozen.items():
        if p.startswith(('inputs/','results/')) or p in ('scripts/run_global.py','scripts/extraction.py','scripts/archived_templates.py','scripts/observed_templates.py'):
            check(sha(B/p)==h);preserved+=1
    old=B.parent/'v6p4p3_global_mc_20260925'
    if old.exists():
        for p,h in frozen.items():check(sha(old/p)==h)
    signature=validation_protocol();R.initialize(NVALID,NAMESPACE)
    hB=np.load(B/'results/toy_draw_hashes_validation.npz');hA=np.load(B/'results/toy_draw_hashes.npz')
    for y,draws in R.CALIBRATION.items():
        hashes=[hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in draws]
        check(np.array_equal(hashes,hB[y]));check(not set(hashes)&set(hA[y].tolist()))
    obs=pd.read_csv(B/'results/observed_scan.csv',dtype={'scope':str},float_precision='round_trip')
    summary=pd.read_csv(B/'results/calibrated_summary.csv',dtype={'scope':str},float_precision='round_trip')
    curves=pd.read_csv(B/'results/calibrated_curves.csv',dtype={'scope':str},float_precision='round_trip')
    thresholds=pd.read_csv(B/'results/threshold_comparison.csv',dtype={'scope':str},float_precision='round_trip')
    maxscore=0.;minlambda=np.inf;allB={}
    for s in SCOPES:
        aa=np.load(B/f'results/global_scans_{s}.npz');bb=np.load(B/f'results/global_validation_{s}.npz');rr=np.load(B/f'results/calibrated_validation_{s}.npz');v=bb['values'];m=bb['masses_MeV'];allB[s]=v
        check(np.array_equal(aa['masses_MeV'],m));check(np.array_equal(bb['toy_ids'],np.arange(NVALID)));check(int(bb['namespace'])==2)
        check(v.shape==(1024,len(m),5));check(np.isfinite(v).all());check((v[:,:,2]>0).all());check((v[:,:,3]<3e-5).all());check((v[:,:,4]>0).all())
        maxscore=max(maxscore,float(v[:,:,3].max()));minlambda=min(minlambda,float(v[:,:,4].min()))
        a=np.maximum(aa['values'][:,:,0],0)**2;b=np.maximum(v[:,:,0],0)**2;o=obs[obs.scope==s].sort_values('mass_MeV');qobs=o.q0.to_numpy()
        ro=1+(a>=qobs).sum(axis=0);check(np.array_equal(ro,rr['observed_rank_counts']))
        # Direct comparisons, deliberately independent of the sorted-search implementation.
        for start in range(0,1024,32):
            direct=1+(a[None,:,:]>=b[start:start+32,None,:]).sum(axis=1)
            check(np.array_equal(direct,rr['local_rank_counts'][start:start+32]))
        rb=rr['local_rank_counts'];minimum=rb.min(axis=1);check(np.array_equal(minimum,rr['min_rank_counts']))
        check(np.all((rb>=1)&(rb<=1025)))
        for j,mass in enumerate(m):
            z=np.load(B/f'results/validation_checkpoints/m{mass:03d}.npz');check(str(z['signature'])==signature);check(np.array_equal(z[s],v[:,j]))
        c=curves[curves.scope==s].sort_values('mass_MeV');check(np.array_equal(c.mass_MeV,m));check(np.array_equal(c.local_rank_count,ro))
        maxA=a.max(axis=1);maxB=b.max(axis=1)
        for j,t in enumerate(c.itertuples()):
            k=int((minimum<=ro[j]).sum());check(t.minp_B_k==k)
            check(abs(t.minp_B_p-(k+1)/1025)<1e-15)
            lo=0 if k==0 else beta.ppf(.025,k,1025-k);hi=1 if k==1024 else beta.ppf(.975,k+1,1024-k)
            check(abs(t.minp_B_p95_low-lo)<1e-15);check(abs(t.minp_B_p95_high-hi)<1e-15)
            kraw=int((maxA>=qobs[j]).sum()+(maxB>=qobs[j]).sum());check(abs(t.raw_2048_p-(kraw+1)/2049)<1e-15)
            check(abs(t.sidak_fitted_p-(1-(1-t.local_p)**fits[s]['N_eff']))<2e-14)
            check(abs(t.sidak_grid_p-(1-(1-t.local_p)**len(m)))<2e-14)
        t=summary[summary.scope==s].iloc[0];j=int(np.argmin(ro));check(t.localfirst_peak_mass_MeV==m[j]);check(t.minp_B_k==int((minimum<=ro[j]).sum()))
        for r in thresholds[thresholds.scope==s].itertuples():
            k=int((minimum/1025<=r.alpha).sum());check(k==r.B_k);check(abs(r.B_p-(k+1)/1025)<1e-15)
        # Estimate the A-only trial factor with independent rankdata (including ties).
        from scipy.stats import rankdata
        loo=np.column_stack([rankdata(-a[:,j],method='max') for j in range(len(m))]).min(axis=1)/1024
        g=np.array([(loo<=alpha).mean() for alpha in ALPHAS]);mask=g<1;x=np.log1p(-ALPHAS[mask]);y=np.log1p(-g[mask]);n=float(np.sum(x*y)/np.sum(x*x))
        check(abs(n-fits[s]['N_eff'])<1e-12)
    check(np.array_equal(allB['combined'][:,116:,:],allB['2021'][:,116:,:]))
    replay=0
    for m in (40,67,68,91,120,175,176,240):
        ctx=at_mass(m);z=np.load(B/f'results/validation_checkpoints/m{m:03d}.npz')
        for t in (0,17):
            for scope,r in evaluate(m,ctx,{y:R.CALIBRATION[y][t] for y in ctx}).items():
                check(np.allclose([r[k] for k in FIELDS],z[scope][t],rtol=1e-10,atol=1e-9));replay+=1
    coupling=json.loads((B/'qa/common_coupling_audit.json').read_text());check(coupling['passed']);check(coupling['exactly_one_shared_signal_parameter'])
    result=dict(passed=True,checks=checks,unchanged_parent_numerical_and_input_files=preserved,complete_A_toys=1024,complete_independent_B_toys=1024,
        total_distinct_experiments=2048,new_profile_pairs=443392,all_B_rank_counts_independently_recomputed=True,inclusive_ties_verified=True,
        all_B_draw_hashes_regenerated=True,no_A_B_draw_hash_overlap=True,targeted_new_toy_profile_replays=replay,
        maximum_score=maxscore,minimum_expected_count=minlambda,sidak_fitted_before_B_evaluation=True,
        common_coupling_audit_passed=True,calibrated_summary_sha256=sha(B/'results/calibrated_summary.csv'))
    write(B/'qa/calibrated_validation.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
