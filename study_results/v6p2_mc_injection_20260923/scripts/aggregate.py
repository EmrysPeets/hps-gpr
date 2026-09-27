"""Aggregate finite-toy recovery diagnostics; do not calibrate by recentering."""
from pathlib import Path
import json,hashlib,sys
import injection_core as I
import numpy as np
import pandas as pd
from scipy.stats import beta,t
B=I.B

def cp(k,n):
    return (0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k)))

def main():
    frames=[];asimov=[];checks=[]
    for m in I.MASSES:
        p=B/f'results/checkpoints/m{m:03d}'
        marker=json.loads(p.with_suffix('.json').read_text())
        assert marker['complete'] and marker['toys']==I.TOYS
        d=pd.read_csv(str(p)+'_toys.csv');frames.append(d)
        asimov.append(pd.read_csv(str(p)+'_asimov.csv'))
        a=np.load(str(p)+'_draws.npz')
        assert np.array_equal(a['levels'],I.LEVELS)
        assert a['backgrounds'].shape==(I.TOYS,len(I.TRUTH))
        assert a['injections'].shape==(I.TOYS,len(I.LEVELS),len(I.TRUTH)+2)
        assert np.array_equal(a['injections'].sum(axis=2),np.tile(I.LEVELS,(I.TOYS,1)))
        assert np.array_equal(a['truth'],I.TRUTH)
        assert np.array_equal(a['edges_GeV'],I.D['edges'])
        for (N,toy),g in d.groupby(['injected_N','toy']):
            assert g.injected_counts_hash.nunique()==1 and g.background_draw_hash.nunique()==1
            assert g.actual_support.nunique()==1 and g.expected_support_fraction.nunique()==1
            assert np.allclose(g.actual_support+g.actual_outside_support,N,atol=1e-10)
            assert np.allclose(g.actual_window+g.actual_training,g.actual_support,atol=1e-10)
        assert d.groupby('toy').background_draw_hash.nunique().eq(1).all()
        checks.append(dict(mass_MeV=m,exact_N=True,paired_counts=True,null_control=True,rows=len(d)))
    df=pd.concat(frames,ignore_index=True)
    df['profile_pull']=np.sign(df.Ahat-df.injected_N)*np.sqrt(df.q_true)
    assert len(df)==len(I.MASSES)*I.TOYS*28 and not df.duplicated(['mass_MeV','injected_N','toy','method','control']).any()
    numeric=['Ahat','sigma_A','pull','q_true','fit_score','min_lambda']
    assert np.isfinite(df[numeric]).all().all() and df.sigma_A.gt(0).all()
    assert df.fit_score.max()<3e-5 and df.min_lambda.min()>0
    summary=[]
    for key,g in df.groupby(['mass_MeV','injected_N','method','control']):
        m,N,method,control=key;n=len(g);assert n==I.TOYS
        sd=float(g.Ahat.std(ddof=1));sem=sd/np.sqrt(n);t95=t.ppf(.975,n-1)
        r=dict(mass_MeV=m,injected_N=N,method=method,control=control,n=n,
            mean_A=float(g.Ahat.mean()),sd_A=sd,mean_sigma=float(g.sigma_A.mean()),
            mean_bias=float(g.Ahat.mean()-N),bias_se=sem,bias_ci95_low=float(g.Ahat.mean()-N-t95*sem),
            bias_ci95_high=float(g.Ahat.mean()-N+t95*sem),
            mean_recovery=float(g.Ahat.mean()/N) if N else np.nan,recovery_se=sem/N if N else np.nan,
            pull_mean=float(g.pull.mean()),pull_width=float(g.pull.std(ddof=1)),
            pull_mean_se=float(g.pull.std(ddof=1)/np.sqrt(n)),
            profile_pull_mean=float(g.profile_pull.mean()),profile_pull_width=float(g.profile_pull.std(ddof=1)),
            rmse=float(np.sqrt(np.mean((g.Ahat-N)**2))),
            empirical_spread_over_mean_error=float(sd/g.sigma_A.mean()),
            negative_yield_count=int(g.Ahat.lt(0).sum()),
            actual_support_mean=float(g.actual_support.mean()),actual_window_mean=float(g.actual_window.mean()),
            actual_training_mean=float(g.actual_training.mean()),
            expected_support_fraction=float(g.expected_support_fraction.iloc[0]),
            expected_window_fraction=float(g.expected_window_fraction.iloc[0]),
            kernel_anchor_MeV=int(g.kernel_anchor_MeV.iloc[0]),extension_260=bool(m==260))
        for tag in ('68','95'):
            k=int(g['profile_contains'+tag].sum());lo,hi=cp(k,n)
            r.update({f'profile{tag}_count':k,f'profile{tag}_fraction':k/n,
                f'profile{tag}_cp_low':lo,f'profile{tag}_cp_high':hi,
                f'wald{tag}_count':int(g['wald_contains'+tag].sum())})
        summary.append(r)
    sums=pd.DataFrame(summary)
    primary=df[df.control=='contaminated_gp']
    paired=[]
    for (m,N),g in primary[primary.injected_N.gt(0)].groupby(['mass_MeV','injected_N']):
        wide=g.pivot(index='toy',columns='method',values='Ahat')
        null=primary[(primary.mass_MeV==m)&primary.injected_N.eq(0)].pivot(index='toy',columns='method',values='Ahat')
        delta=wide.core_shifted-wide.pole_centered;increment=(wide-null)/N
        paired.append(dict(mass_MeV=m,injected_N=N,mean_core_minus_pole=float(delta.mean()),
            se_core_minus_pole=float(delta.std(ddof=1)/np.sqrt(len(wide))),
            mean_incremental_recovery_pole=float(increment.pole_centered.mean()),
            mean_incremental_recovery_core=float(increment.core_shifted.mean()),
            se_incremental_recovery_pole=float(increment.pole_centered.std(ddof=1)/np.sqrt(len(wide))),
            se_incremental_recovery_core=float(increment.core_shifted.std(ddof=1)/np.sqrt(len(wide))),
            covariance_methods=float(wide.cov().iloc[0,1]),
            definition='Incremental recovery subtracts paired N=0 fitted yield; diagnostic only, no correction to raw pulls.'))
    for name,data in [('toys',df),('summary',sums),('paired',pd.DataFrame(paired)),('asimov',pd.concat(asimov,ignore_index=True))]:
        data.to_csv(B/f'results/{name}.csv',index=False,float_format='%.17g')
    comparison=[]
    for key,g in df.groupby(['mass_MeV','injected_N','method','control']):
        m,N,method,control=key
        r=dict(mass_MeV=m,injected_N=N,method=method,control=control)
        for tag,subset in [('first20',g[g.toy<20]),('added20',g[g.toy>=20]),('all40',g)]:
            r.update({f'{tag}_n':len(subset),f'{tag}_mean_A':float(subset.Ahat.mean()),
                f'{tag}_mean_bias':float(subset.Ahat.mean()-N),
                f'{tag}_mean_recovery':float(subset.Ahat.mean()/N) if N else np.nan,
                f'{tag}_pull_mean':float(subset.pull.mean()),
                f'{tag}_pull_width':float(subset.pull.std(ddof=1)),
                f'{tag}_profile95_count':int(subset.profile_contains95.sum())})
            if N and control=='contaminated_gp':
                baseline=df[(df.mass_MeV==m)&(df.injected_N==0)&
                    (df.method==method)&(df.control==control)].set_index('toy').Ahat
                increment=(subset.set_index('toy').Ahat-baseline.loc[subset.toy])/N
                r[f'{tag}_incremental_recovery']=float(increment.mean())
        comparison.append(r)
    pd.DataFrame(comparison).to_csv(B/'results/toy_extension_comparison.csv',index=False,float_format='%.17g')
    ledger=[]
    for m in I.MASSES:
        marker=json.loads((B/f'results/checkpoints/m{m:03d}.json').read_text())
        ledger.append(dict(mass_MeV=m,core_center_MeV=marker['center']['core_center_MeV'],
            shift_MeV=marker['center']['center_shift_MeV'],nominal_sigma_MeV=marker['center']['nominal_sigma_MeV'],
            core_sigma_MeV=marker['center']['fitted_core_sigma_MeV'],support_fraction=marker['support_fraction'],
            source_selected_entries=marker['source_selected_entries']))
    pd.DataFrame(ledger).to_csv(B/'results/centers.csv',index=False,float_format='%.17g')
    for rec in json.loads((B/'provenance/input_hashes.json').read_text()):
        assert hashlib.sha256((B/rec['bundled']).read_bytes()).hexdigest()==rec['sha256']
    selected=sums[(sums.control=='contaminated_gp')&sums.injected_N.gt(0)]
    overview=dict(version='6.2',masses=list(I.MASSES),levels=list(I.LEVELS),toys_per_cell=I.TOYS,
        unique_injected_toys=len(I.MASSES)*len(I.LEVELS)*I.TOYS,
        unique_null_toys=len(I.MASSES)*I.TOYS,
        primary_fits=len(I.MASSES)*len(I.LEVELS)*I.TOYS*len(I.METHODS),
        total_extraction_fits=len(df),asimov_extraction_fits=sum(len(x) for x in asimov),
        optimizer_fits=2*(len(df)+sum(len(x) for x in asimov)),
        source='2021 10% nominal full-support GP truth; exactN selected native smeared MC draws',
        primary_recovery_range=[float(selected.mean_recovery.min()),float(selected.mean_recovery.max())],
        primary_pull_mean_range=[float(selected.pull_mean.min()),float(selected.pull_mean.max())],
        primary_pull_width_range=[float(selected.pull_width.min()),float(selected.pull_width.max())],
        primary_95_containment_count_range=[int(selected.profile95_count.min()),int(selected.profile95_count.max())],
        max_fit_score=float(df.fit_score.max()),min_lambda=float(df.min_lambda.min()),
        native_masses_in_original_domain=10,extension_mass=260,
        scope=f'Conditional recovery diagnostic; {I.TOYS} toys per cell do not establish calibrated coverage.')
    I.write_json(B/'results/overview.json',overview)
    I.write_json(B/'qa/validation.json',dict(passed=True,complete_matrix=True,rows=len(df),
        finite_fits=True,source_hashes_verified=True,maximum_score=float(df.fit_score.max()),
        minimum_lambda=float(df.min_lambda.min()),mass_checks=checks))
    protocol=json.loads((B/'protocol.json').read_text());protocol['status']='computations_complete'
    I.write_json(B/'protocol.json',protocol)
    print(json.dumps(overview,indent=2))

if __name__=='__main__':main()
