#!/usr/bin/env python3
"""Equal-mass sensitivity summaries with whole-toy, paired bootstrap errors."""
import run_study as R
import numpy as np
import pandas as pd

def main():
    rows=[]
    for cohort in ('tuning','evaluation'):
        frame=pd.read_csv(R.B/f'results/{cohort}_rows.csv',float_precision='round_trip')
        for source,q in frame.groupby('source'):
            policies=sorted(q.policy.unique());zero=[];signal=[];strength=[]
            for mass in R.MASSES:
                r=q[q.mass_MeV==mass]
                zero.append(r[r.z==0].pivot(index='toy',columns='policy',values='Ahat')[policies].to_numpy())
                signal.append(r[r.z==3].pivot(index='toy',columns='policy',values='Ahat')[policies].to_numpy())
                strength.append(float(r[r.z==3].A_expected.iloc[0]))
            bg=np.stack(zero,axis=1);inj=np.stack(signal,axis=1);A=np.array(strength)[None,:,None]
            delta=(inj-bg)/A
            sensitivity=bg.std(axis=0,ddof=1)/delta.mean(axis=0)
            global_score=np.exp(np.log(sensitivity).mean(axis=0))
            ns={'tuning':1,'evaluation':3}[cohort];sid=list(R.TRUTHS).index(source)
            indices=np.random.default_rng(np.random.SeedSequence([R.SEED,7,ns,sid])).integers(0,R.N,(2000,R.N))
            sampled=bg[indices].std(axis=1,ddof=1)/delta[indices].mean(axis=1)
            bootstrap=np.exp(np.log(sampled).mean(axis=1))
            baseline=policies.index('gaussian_baseline')
            for i,policy in enumerate(policies):
                ratios=bootstrap[:,i]/bootstrap[:,baseline];lo,hi=np.quantile(ratios,[.025,.975])
                rows.append(dict(cohort=cohort,source=source,policy=policy,masses=','.join(map(str,R.MASSES)),toys_per_mass=R.N,
                    equal_mass_geometric_SD_over_R=global_score[i],ratio_to_baseline=global_score[i]/global_score[baseline],
                    ratio95_low=lo,ratio95_high=hi,bootstrap_resamples=2000))
    R.atomic_csv(R.B/'results/global_response_summary.csv',pd.DataFrame(rows))

if __name__=='__main__':main()
