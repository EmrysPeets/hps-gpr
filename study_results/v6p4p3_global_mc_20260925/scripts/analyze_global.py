"""Mass-grid and displayed-search-family probabilities from complete Poisson scans."""
from run_global import *
from scipy.stats import beta,norm

def tail(k,n=NTOYS):
    p=(int(k)+1)/(n+1);lo=0. if k==0 else float(beta.ppf(.025,k,n-k+1));hi=1. if k==n else float(beta.ppf(.975,k+1,n-k))
    z=max(0.,float(norm.isf(p)))
    return dict(exceedances=int(k),toys=n,p_rank=p,p_fraction=float(k/n),p95_low=lo,p95_high=hi,
        Z_excess=z,Z95_low=max(0.,float(norm.isf(hi))),Z95_high=None if lo==0 else max(0.,float(norm.isf(lo))),
        zero_exceedances=bool(k==0),Z_convention='max(0,normal.isf(p)); report p even when Z clips to0')

def main():
    signature=protocol();spec=dict(primary='Separate mass-grid maxima for2016,2021 andcombined',
        supplementary='Maximum q0 over all three displayed searches and their respective mass grids, with their correlations preserved',
        local_reference='Pointwise tail from the same null scans at each observed mass; no recentering or rescaling of the statistic',
        prior_choices='Earlier window/template/source exploration is outside the calibrated family',
        null_protocol_sha256=signature,script_sha256=sha(__file__))
    path=B/'provenance/global_analysis_protocol.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:write(path,spec)
    observed=pd.read_csv(B/'results/observed_scan.csv',dtype={'scope':str},float_precision='round_trip')
    summaries=[];curves=[];maxima=[];means=[];mx={};obspeak={}
    for scope in SCOPES:
        z=np.load(B/f'results/global_scans_{scope}.npz');m=z['masses_MeV'];v=z['values'];fields=list(z['fields'])
        assert len(v)==NTOYS and np.array_equal(z['toy_ids'],np.arange(NTOYS)) and np.isfinite(v).all()
        roots=v[:,:,fields.index('signed_root')];q=np.maximum(roots,0)**2;maximum=q.max(axis=1);mx[scope]=maximum
        o=observed[observed.scope==scope].sort_values('mass_MeV');assert np.array_equal(o.mass_MeV.to_numpy(),m)
        oq=o.q0.to_numpy();peak=int(np.argmax(oq));obspeak[scope]=float(oq[peak])
        local_counts=(q>=oq[None,:]).sum(axis=0);global_counts=(maximum[:,None]>=oq[None,:]).sum(axis=0)
        assert np.all(global_counts>=local_counts)
        for j,oo in enumerate(o.itertuples()):
            local=tail(int(local_counts[j]));glob=tail(int(global_counts[j]))
            curves.append(dict(scope=scope,mass_MeV=int(m[j]),q0=float(oq[j]),signed_root=float(oo.signed_root),
                asymptotic_p=float(oo.p0_asymptotic),asymptotic_Z=float(oo.Z_local),
                **{'local_'+k:x for k,x in local.items()},**{'global_'+k:x for k,x in glob.items()}))
            means.append(dict(scope=scope,mass_MeV=int(m[j]),mean_signed_root=float(roots[:,j].mean()),sd_signed_root=float(roots[:,j].std(ddof=1))))
        for t in range(NTOYS):maxima.append(dict(scope=scope,toy=t,max_q0=float(maximum[t]),max_root=float(np.sqrt(maximum[t])),peak_mass_MeV=int(m[np.argmax(q[t])])) )
        local=tail(int(local_counts[peak]));glob=tail(int(global_counts[peak]));oo=o.iloc[peak]
        summaries.append(dict(scope=scope,mass_low_MeV=int(m[0]),mass_high_MeV=int(m[-1]),mass_points=len(m),peak_mass_MeV=int(m[peak]),
            observed_q0=float(oq[peak]),observed_Z_asymptotic=float(oo.Z_local),observed_p_asymptotic=float(oo.p0_asymptotic),
            **{'local_'+k:x for k,x in local.items()},**{'global_'+k:x for k,x in glob.items()},
            null_mean_root_at_peak=float(roots[:,peak].mean()),null_sd_root_at_peak=float(roots[:,peak].std(ddof=1))))
    family=np.max(np.stack([mx[s] for s in SCOPES]),axis=0);threshold=max(obspeak.values());fam=tail(int(np.count_nonzero(family>=threshold)))
    winner=max(obspeak,key=obspeak.get);fam.update(scope='displayed_search_family',observed_q0=threshold,winning_scope=winner,
        winning_mass_MeV=next(r['peak_mass_MeV'] for r in summaries if r['scope']==winner),
        search_family=list(SCOPES),calibrates_prior_window_choices=False)
    for t,x in enumerate(family):maxima.append(dict(scope='displayed_search_family',toy=t,max_q0=float(x),max_root=float(np.sqrt(x)),peak_mass_MeV=None))
    csv(B/'results/global_summary.csv',summaries);csv(B/'results/local_global_curves.csv',curves);csv(B/'results/toy_maxima.csv',maxima);csv(B/'results/null_root_moments.csv',means)
    write(B/'results/family_global.json',fam)
    write(B/'results/global_results.json',dict(primary=summaries,supplementary_family=fam,complete_toys=NTOYS,
        mass_grid_global=True,continuous_mass_global=False,window_template_selection_calibrated=False,
        source_estimation_uncertainty_calibrated=False))
    print(json.dumps(dict(primary=summaries,family=fam),indent=2))

if __name__=='__main__':main()
