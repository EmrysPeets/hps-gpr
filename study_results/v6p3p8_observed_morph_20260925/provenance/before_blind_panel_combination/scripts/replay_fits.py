#!/usr/bin/env python3
"""Replay selected observed profiles and archived toy fits for numerical QA."""
import json
import numpy as np
import pandas as pd
import run_observed as R

def main():
    regions=json.loads((R.B/'results/selected_regions.json').read_text())['regions']
    rows=[];component_checks=[]
    for region in regions:
        m=region['mass_MeV']
        for policy in R.POLICIES:
            path=R.B/f'results/selected_fits/{region["region"]}_m{m:03d}_{policy}.npz'
            with np.load(path) as d:
                summary=json.loads(str(d['fit_summary_json']));mask=d['fit_mask'];ctx=R.Context(m,policy)
                assert np.array_equal(mask,ctx.fit) and not np.any(mask&~d['guard_mask'])
                assert np.array_equal(d['counts'],R.D['n'])
                assert np.allclose(d['profiled_background_signed']+d['profiled_signed_signal'],d['profiled_signed_total'],rtol=1e-12,atol=1e-8)
                assert np.allclose(d['profiled_signed_signal'],summary['Ahat']*d['signal_probability'][mask],rtol=1e-12,atol=1e-8)
                # Full-support and fit-only matrix products use different BLAS
                # reduction shapes. Compare their small floating-point residual
                # against the prediction and marginal covariance scales.
                mean_relative=float(np.max(np.abs(d['prefit_GP_mean'][mask]-d['fit_prefit_mean'])/d['fit_prefit_mean']))
                covariance_difference=d['prefit_GP_covariance'][np.ix_(mask,mask)]-d['fit_prefit_covariance']
                covariance_scale=np.sqrt(np.outer(np.diag(d['fit_prefit_covariance']),np.diag(d['fit_prefit_covariance'])))
                covariance_scaled=float(np.max(np.abs(covariance_difference)/covariance_scale))
                assert mean_relative<1e-8 and covariance_scaled<1e-5
                if summary['Ahat']<0:assert np.array_equal(d['profiled_bounded_total'],d['profiled_background_only'])
                component_checks.append(dict(region=region['region'],mass_MeV=m,policy=policy,sha256=R.sha(path),full_vs_fit_mean_relative_difference=mean_relative,full_vs_fit_covariance_marginal_scaled_difference=covariance_scaled))
                if policy=='morph_starter':
                    fresh=ctx.fit_counts(d['counts'],limit=True)
                    assert fresh['fit_valid']
                    for key in ('Ahat','sigma_A','A90','signed_r'):
                        assert np.isclose(fresh[key],summary[key],rtol=2e-10,atol=1e-6)
                    rows.append(dict(kind='observed',mass_MeV=m,policy=policy,Ahat_difference=fresh['Ahat']-summary['Ahat'],A90_difference=fresh['A90']-summary['A90'],sigma_difference=fresh['sigma_A']-summary['sigma_A']))
    for region,cohort,policy,toy,z in ((regions[0],'null','morph_starter',123,0.),(regions[1],'calibration','gaussian_baseline',76,5.),(regions[2],'calibration','morph_starter',198,3.),(regions[3],'null','gaussian_starter',917,0.)):
        m=region['mass_MeV'];first=(toy//50)*50
        path=R.B/f'results/calibration_checkpoints/{cohort}_m{m:03d}_t{first:04d}.csv'
        frame=pd.read_csv(path,float_precision='round_trip',keep_default_na=False)
        r=frame[(frame.toy==toy)&(frame.policy==policy)&(frame.z==z)].iloc[0]
        with np.load(path.with_suffix('.npz')) as d:
            ti=int(np.flatnonzero(d['toys']==toy)[0]);zi=int(np.flatnonzero(d['strengths']==z)[0]);bg=d['backgrounds'][ti];draw=d['signal_categories'][ti*len(d['strengths'])+zi];counts=bg+draw[1:-1]
            assert R.ahash(bg)==r.background_hash and R.ahash(draw)==r.signal_hash and R.ahash(counts)==r.counts_hash
            ctx=R.Context(m,policy);fresh=ctx.fit_counts(counts)
            assert fresh['fit_valid']
            assert np.isclose(fresh['Ahat'],r.Ahat,rtol=2e-10,atol=1e-6)
            assert np.isclose(fresh['sigma_A'],r.sigma_A,rtol=2e-10,atol=1e-6)
            assert int(draw.sum())==r.actual_full and int(draw[1:-1][~ctx.guard].sum())==r.actual_training
            rows.append(dict(kind=cohort,mass_MeV=m,policy=policy,toy=toy,z=z,Ahat_difference=float(fresh['Ahat']-r.Ahat),sigma_difference=float(fresh['sigma_A']-r.sigma_A)))
    R.write(R.B/'qa/fit_replay.json',dict(passed=True,replayed_observed_profiles=4,replayed_toy_extractions=4,selected_fit_component_files_verified=12,relative_tolerance=2e-10,amplitude_absolute_tolerance=1e-6,full_vs_fit_GP_mean_relative_tolerance=1e-8,full_vs_fit_GP_covariance_marginal_scaled_tolerance=1e-5,
        rows=rows,component_files=component_checks,script_sha256=R.sha(__file__)))
    print('Passed: 4 observed profile replays, 4 toy replays, 12 component files')

if __name__=='__main__':main()
