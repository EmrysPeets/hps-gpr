"""Frozen empirical local maps, independent min-p tails and A-only Sidak fit."""
from run_validation import *
from scipy.stats import beta,norm
from scipy.special import ndtr
ALPHAS=np.array([.005,.01,.02,.03,.05])

def ranks(reference,query,add_one=True):
    query=np.atleast_2d(query);out=np.empty(query.shape,dtype=np.int32)
    for j in range(reference.shape[1]):
        a=np.sort(reference[:,j]);out[:,j]=len(a)-np.searchsorted(a,query[:,j],side='left')+int(add_one)
    return out

def tail(k,n):
    p=(int(k)+1)/(n+1);lo=0. if k==0 else float(beta.ppf(.025,k,n-k+1));hi=1. if k==n else float(beta.ppf(.975,k+1,n-k))
    return dict(k=int(k),N=n,p=p,p_fraction=k/n,p95_low=lo,p95_high=hi,Z=max(0.,float(norm.isf(p))))

def sidak(p,n):return float(-np.expm1(n*np.log1p(-p))) if p<1 else 1.
def ze(p):return max(0.,float(norm.isf(p)))
def equivalent(g,a):return float(np.log1p(-g)/np.log1p(-a)) if g<1 else None

def freeze():
    spec=dict(version='6.4.4',calibration_size=1024,validation_size=1024,
        local_map='p_A(q)=(1+number of A q>=q)/(1025); upper-tail inclusive ties',
        statistic='min_m p_A(q_m); smaller is more extreme; ties included in B tail; lower mass chosen for display ties',
        independence='Only A defines maps and Sidak fit; B does not update maps; same frozen map applied to observations and B',
        sidak='Two references: actual grid-size independence N=136/181, and fixed effectiveN learned from A only',
        sidak_fit='A leave-one-out ranks count(q_A>=q_i)/1024 including self; min across mass; least squares through origin of log(1-G_A(alpha)) on log(1-alpha)',
        fit_alphas=ALPHAS.tolist(),fit_weighting='equal weights in log-survival space; omit saturated G_A=1; require at least2 points',
        sidak_limits='A LOO minima are dependent and differ from B conditional on the full frozen map; no binomial CI for A fit; fit is an approximation tested on B',
        raw_comparison='Keep raw-maximum global tails for A,B,and A+B2048 separately; no B reuse in local calibration',
        uncertainties='Clopper-Pearson95 for B conditional on fixed A maps and fixed null source; calibration-map/source uncertainty not included',
        combined='Exactly one common psi at same mass, same null and alternative, independent GP nuisance blocks; no maximization over dataset selections',
        validation_protocol_sha256=validation_protocol(),script_sha256=sha(__file__))
    path=B/'provenance/calibrated_analysis_protocol.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:write(path,spec)
    fits=[]
    for s in SCOPES:
        z=np.load(B/f'results/global_scans_{s}.npz');q=np.maximum(z['values'][:,:,0],0)**2
        rr=ranks(q,q,False);minp=rr.min(axis=1)/NTOYS;g=np.array([(minp<=a).mean() for a in ALPHAS]);keep=g<1
        assert keep.sum()>=2
        x=np.log1p(-ALPHAS[keep]);y=np.log1p(-g[keep]);neff=float(x@y/(x@x));assert neff>0
        fits.append(dict(scope=s,N_eff=neff,grid_points=q.shape[1],fit_alphas=ALPHAS.tolist(),A_LOO_global_fractions=g.tolist(),fit_points_used=keep.tolist(),
            threshold_specific_A_N_eff=[equivalent(v,a) for v,a in zip(g,ALPHAS)]))
    target=B/'results/sidak_fit.json';payload=dict(protocol_sha256=sha(path),fits=fits)
    if target.exists():assert json.loads(target.read_text())==payload
    else:write(target,payload)
    return {r['scope']:r for r in fits}

def analyze():
    fits=freeze();obs=pd.read_csv(B/'results/observed_scan.csv',dtype={'scope':str},float_precision='round_trip')
    summaries=[];curves=[];thresholds=[];ledger=[]
    for s in SCOPES:
        aa=np.load(B/f'results/global_scans_{s}.npz');bb=np.load(B/f'results/global_validation_{s}.npz')
        assert np.array_equal(aa['masses_MeV'],bb['masses_MeV']);m=aa['masses_MeV'];a=np.maximum(aa['values'][:,:,0],0)**2;b=np.maximum(bb['values'][:,:,0],0)**2
        o=obs[obs.scope==s].sort_values('mass_MeV');assert np.array_equal(o.mass_MeV,m)
        qobs=o.q0.to_numpy();rankobs=ranks(a,qobs)[0];rankB=ranks(a,b);minB=rankB.min(axis=1);pobs=rankobs/(NTOYS+1)
        ma=a.max(axis=1);mb=b.max(axis=1);both=np.r_[ma,mb];n=fits[s]['N_eff']
        imin=int(np.argmin(rankobs));iraw=int(np.argmax(qobs));local_k=int(rankobs[imin]-1)
        for j,mass in enumerate(m):
            glob=tail(int((minB<=rankobs[j]).sum()),NVALID);rawA=tail(int((ma>=qobs[j]).sum()),NTOYS);rawB=tail(int((mb>=qobs[j]).sum()),NVALID);rawAB=tail(int((both>=qobs[j]).sum()),NTOYS+NVALID)
            sf=sidak(pobs[j],n);sg=sidak(pobs[j],len(m))
            curves.append(dict(scope=s,mass_MeV=int(mass),local_rank_count=int(rankobs[j]),local_p=float(pobs[j]),local_Z=ze(pobs[j]),q0=float(qobs[j]),
                raw_A_p=rawA['p'],raw_B_p=rawB['p'],raw_2048_p=rawAB['p'],raw_2048_Z=rawAB['Z'],
                **{'minp_B_'+k:v for k,v in glob.items()},sidak_fitted_p=sf,sidak_fitted_Z=ze(sf),sidak_grid_p=sg,sidak_grid_Z=ze(sg)))
        selected=curves[-len(m)+imin];raw=tail(int((both>=qobs[iraw]).sum()),NTOYS+NVALID)
        local=tail(local_k,NTOYS)
        summaries.append(dict(scope=s,raw_peak_mass_MeV=int(m[iraw]),localfirst_peak_mass_MeV=int(m[imin]),
            tied_peak_masses_MeV_json=json.dumps(m[rankobs==rankobs.min()].tolist()),local_A_exceedances=local_k,local_A_p=local['p'],local_A_p95_low=local['p95_low'],local_A_p95_high=local['p95_high'],local_A_Z=local['Z'],local_map_floor=bool(local_k==0),
            raw_1024_k=int((ma>=qobs[iraw]).sum()),raw_1024_p=tail(int((ma>=qobs[iraw]).sum()),NTOYS)['p'],raw_new1024_k=int((mb>=qobs[iraw]).sum()),raw_new1024_p=tail(int((mb>=qobs[iraw]).sum()),NVALID)['p'],
            **{'raw_2048_'+k:v for k,v in raw.items()},**{k:v for k,v in selected.items() if k.startswith('minp_B_') or k.startswith('sidak_')},sidak_N_eff=n))
        for alpha in sorted(set(np.r_[np.arange(1,104)/(NTOYS+1),ALPHAS,pobs[imin]])):
            t=tail(int((minB/(NTOYS+1)<=alpha).sum()),NVALID);fitp=sidak(alpha,n);gridp=sidak(alpha,len(m))
            thresholds.append(dict(scope=s,alpha=float(alpha),**{'B_'+k:v for k,v in t.items()},sidak_fitted_p=fitp,sidak_grid_p=gridp,
                effective_N=equivalent(t['p'],alpha),effective_N95_low=equivalent(t['p95_low'],alpha),effective_N95_high=equivalent(t['p95_high'],alpha),
                sidak_inside_B95=bool(t['p95_low']<=fitp<=t['p95_high']),inside_A_fit_range=bool(ALPHAS.min()<=alpha<=ALPHAS.max())))
        for t in range(NVALID):ledger.append(dict(scope=s,toy=t,min_local_rank_count=int(minB[t]),min_local_p=float(minB[t]/(NTOYS+1)),min_p_mass_MeV=int(m[np.argmin(rankB[t])]),raw_max_q0=float(mb[t])))
        np.savez_compressed(B/f'results/calibrated_validation_{s}.npz',masses_MeV=m,toy_ids=np.arange(NVALID),local_rank_counts=rankB,observed_rank_counts=rankobs,min_rank_counts=minB)
    csv(B/'results/calibrated_summary.csv',summaries);csv(B/'results/calibrated_curves.csv',curves);csv(B/'results/threshold_comparison.csv',thresholds);csv(B/'results/validation_minima.csv',ledger)
    write(B/'results/calibrated_results.json',dict(primary=summaries,local_calibration_toys=NTOYS,independent_global_validation_toys=NVALID,total_distinct_experiments=NTOYS+NVALID,
        common_coupling_test=True,dataset_choice_maximum_used=False,conditional_on_A_and_null_source=True,calibration_map_uncertainty_in_B_intervals=False))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
    if args.freeze_only:print(json.dumps(freeze(),indent=2))
    else:analyze()
