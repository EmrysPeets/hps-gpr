"""Signed Gaussian signal compression and matched fixed-mass combinations.

All primary Gaussian probabilities condition on V=diag(lambda_null)+L L^T.
This is the parent predictive/joint-auxiliary Gaussian reference, not a direct
Poisson or sideband-refit calibration. The full across-mass covariance is saved.
"""
from pathlib import Path
import os, sys, json, hashlib, math
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B/'engine'))
import experiment as X
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq
from scipy.special import ndtri_exp
from scipy.stats import norm, chi2


def gaussian_cls_upper(x, sigma, alpha=.1):
    """One-dimensional known-Gaussian CLs bound, stable in large deficits."""
    return float(x - sigma*ndtri_exp(math.log(alpha)+norm.logcdf(x/sigma)))


def positive_amplitude_p(q, dimensions=3):
    """Inclusive tail of a sum of independent positive standard-normal squares."""
    if q <= 0:
        return 1.
    return float(sum(math.comb(dimensions,k)/2**dimensions*chi2.sf(q,k)
                     for k in range(1,dimensions+1)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    bank_path=B/'derived/score_fields.npz'
    grid_path=B/'derived/individual_shape_grid.csv'
    bank=np.load(bank_path)
    means=bank['means_MeV']; widths=bank['width_t']
    assert np.array_equal(means,np.arange(90,94.0001,.25))
    k=int(np.flatnonzero(widths==1)[0])
    grid=pd.read_csv(grid_path)
    grid['dataset']=grid.dataset.astype(str)
    covariances={'masses_MeV':means,'years':np.array(X.YS)}
    scores=[]; errors=[]; amplitudes=[]; unit_yields=[]
    direct_errors=[]
    for i,y in enumerate(X.YS):
        units=bank['unit_vectors_'+y][:,k]
        sqrtI=bank['sqrt_information_'+y][:,k]
        z=units@bank['whitened_residual_'+y]
        sigma=1/sqrtI; amp=z*sigma
        signal=np.array([X.shape(i,m,1)[0] for m in means])
        p=X.parts[i]
        V=np.diag(X.nulls[i]['lam'])+p['L']@p['L'].T
        vinv=cho_factor(V,lower=True)
        info=np.einsum('ij,ji->i',signal,cho_solve(vinv,signal.T))
        direct=signal@cho_solve(vinv,p['n']-p['b'])/np.sqrt(info)
        direct_errors.append(float(np.max(abs(direct-z))))
        assert np.max(abs(direct-z))<1e-8
        assert np.max(abs(info/sqrtI**2-1))<1e-9
        covariance=(units@units.T)*np.outer(sigma,sigma)
        covariances['amplitude_covariance_'+y]=covariance
        covariances['score_correlation_'+y]=units@units.T
        scores.append(z);errors.append(sigma);amplitudes.append(amp)
        unit_yields.append(signal.sum(axis=1))
    scores=np.array(scores).T;errors=np.array(errors).T
    amplitudes=np.array(amplitudes).T;unit_yields=np.array(unit_yields).T
    covariances['total_yield_covariance']=sum(
        covariances['amplitude_covariance_'+y]*np.outer(unit_yields[:,i],unit_yields[:,i])
        for i,y in enumerate(X.YS))
    common_weights=(1/errors**2)/(1/errors**2).sum(axis=1)[:,None]
    covariances['common_amplitude_covariance']=sum(
        covariances['amplitude_covariance_'+y]*np.outer(common_weights[:,i],common_weights[:,i])
        for i,y in enumerate(X.YS))
    covariances['signed_stouffer_covariance']=sum(
        covariances['score_correlation_'+y] for y in X.YS)/3
    rows=[]; max_replay=0.; compression_error=0.; nesting_margin=[]
    for j,m in enumerate(means):
        a=amplitudes[j]; s=errors[j]; z=scores[j]; yieldfactor=unit_yields[j]
        variance=1/np.sum(1/s**2)
        mu=float(np.sum(a/s**2)*variance); err=float(np.sqrt(variance)); zc=mu/err
        common_Q=max(0.,zc)**2
        qi=float(np.sum(np.maximum(z,0)**2)); pi=positive_amplitude_p(qi)
        stouffer=float(np.sum(z)/np.sqrt(3))
        fisher_q=float(-2*np.sum(norm.logsf(z)))
        fisher_p=float(chi2.sf(fisher_q,6))
        total=float(a@yieldfactor); totalerr=float(np.linalg.norm(s*yieldfactor))
        cls_common=gaussian_cls_upper(mu,err)
        cls_total=gaussian_cls_upper(total,totalerr)
        templ=np.concatenate([X.shape(i,m,1)[0] for i in range(3)])
        exact=X.C.OneSignalProfile(X.b,X.L,templ).limit(X.n)
        individual=grid[(grid.mean_MeV==m)&(grid.t==1)].set_index('dataset').loc[X.YS]
        ap=individual.A_signed.to_numpy()
        sp=individual.error_epsilon2.to_numpy()/1e-8
        qp=float(individual.Q.sum())
        zp=ap/sp
        mp=float(np.sum(ap/sp**2)/np.sum(1/sp**2)); ep=float(1/np.sqrt(np.sum(1/sp**2)))
        row=dict(mass_MeV=float(m),common_GLS_A=mu,common_GLS_sigma_A=err,
          common_GLS_signed_Z=zc,common_GLS_Q=common_Q,common_GLS_p_local=float(norm.sf(zc)),
          common_GLS_A90=cls_common,common_GLS_epsilon2_90=cls_common*1e-8,
          common_GLS_total_yield_hat=mu*sum(yieldfactor),
          common_GLS_total_yield_sigma=err*sum(yieldfactor),
          common_GLS_total_yield_upper90=cls_common*sum(yieldfactor),
          common_Poisson_A=exact['Ahat'],common_Poisson_sigma_A=exact['sigma_A'],
          common_Poisson_signed_root=exact['signed_r'],common_Poisson_Q=exact['Z0']**2,
          common_Poisson_p_local_reference=exact['p_signed'],
          common_Poisson_A90=exact['A90'],common_Poisson_epsilon2_90=exact['A90']*1e-8,
          common_Poisson_total_yield_upper90=exact['A90']*sum(yieldfactor),
          common_Poisson_limit_max_score=exact['max_score'],common_Poisson_limit_CLs=exact['cls'],
          independent_GLS_Q=qi,independent_GLS_raw_root=np.sqrt(qi),
          independent_GLS_p_local=pi,independent_GLS_Z_equivalent=float(norm.isf(pi)),
          independent_Poisson_Q=qp,independent_Poisson_raw_root=np.sqrt(qp),
          independent_Poisson_p_chibar_reference=positive_amplitude_p(qp),
          independent_GLS_total_yield_hat=total,independent_GLS_total_yield_sigma=totalerr,
          independent_GLS_total_yield_upper90=cls_total,
          independent_GLS_total_yield_plain_upper90=max(0.,total+norm.ppf(.9)*totalerr),
          signed_Stouffer_Z=stouffer,signed_Stouffer_p_local=float(norm.sf(stouffer)),
          signed_Fisher_Q=fisher_q,signed_Fisher_p_local=fisher_p,
          signed_Fisher_Z_equivalent=float(norm.isf(fisher_p)),
          curvature_compressed_common_signed_Z=mp/ep,
          curvature_compressed_common_p_reference=float(norm.sf(mp/ep)),
          curvature_compressed_independent_Q=float(np.sum(np.maximum(zp,0)**2)),
          curvature_compressed_independent_p_reference=positive_amplitude_p(float(np.sum(np.maximum(zp,0)**2))),
          curvature_compressed_Stouffer_Z=float(np.sum(zp)/np.sqrt(3)))
        for i,y in enumerate(X.YS):
            row.update({f'GLS_A_{y}':a[i],f'GLS_sigma_A_{y}':s[i],f'GLS_z_{y}':z[i],
              f'unit_fitted_yield_{y}':yieldfactor[i],f'GLS_signal_yield_{y}':a[i]*yieldfactor[i],
              f'GLS_signal_yield_error_{y}':s[i]*yieldfactor[i],
              f'Poisson_A_{y}':ap[i],f'Poisson_sigma_A_{y}':sp[i],
              f'Poisson_signed_root_{y}':float(individual.loc[y,'signed_r'])})
        # Gaussian signal estimates retain exactly the amplitude-dependent joint
        # likelihood: compare an off-optimum common amplitude to direct bin GLS.
        trial=mu+.73*err; q_direct=0.; q_compressed=0.
        for i,y in enumerate(X.YS):
            p=X.parts[i];V=np.diag(X.nulls[i]['lam'])+p['L']@p['L'].T
            d=p['n']-p['b'];S=X.shape(i,m,1)[0]
            inv=cho_factor(V,lower=True)
            q_direct += float((d-trial*S)@cho_solve(inv,d-trial*S)-d@cho_solve(inv,d))
            q_compressed += ((a[i]-trial)/s[i])**2-(a[i]/s[i])**2
        compression_error=max(compression_error,abs(q_direct-q_compressed))
        max_replay=max(max_replay,abs(qi-float(individual.gaussian_Q.sum())))
        nesting_margin.append(qi-common_Q)
        assert qi>=common_Q-1e-10 and qp>=exact['Z0']**2-1e-6
        assert cls_total>=row['independent_GLS_total_yield_plain_upper90']-1e-7
        rows.append(row)
    out=pd.DataFrame(rows)
    out.to_csv(B/'derived/extracted_combination_scan.csv',index=False,float_format='%.17g')
    np.savez_compressed(B/'derived/extracted_combination_covariance.npz',**covariances)
    # Check the derived CLs formula and its Gaussian coverage without random toys.
    root_errors=[];coverage=[]
    for zz in (-20.,-5.,-1.,0.,1.,4.,10.):
        u=gaussian_cls_upper(zz,1.)
        root_errors.append(abs(np.exp(norm.logcdf(zz-u)-norm.logcdf(zz))-.1))
        assert u>0 and u>=zz+norm.ppf(.9)-1e-12
    for truth in (.001,.01,.1,.5,1.,2.,5.,10.):
        threshold=brentq(lambda x:gaussian_cls_upper(x,1.)-truth,-10000,truth)
        cov=float(norm.sf(threshold-truth));coverage.append(dict(true_total_over_sigma=truth,coverage=cov))
        assert cov>=.9-1e-12
    assert max(root_errors)<1e-11 and compression_error<1e-7 and max_replay<1e-9
    for q in (.01,1.,5.,20.):
        assert abs(positive_amplitude_p(q,1)-norm.sf(np.sqrt(q)))<1e-13
    assert positive_amplitude_p(0)==1.
    ninety_two=out[out.mass_MeV==92].iloc[0]
    parent=pd.read_csv(B/'derived/fits.csv').set_index('name')
    parent_common_error=abs(ninety_two.common_Poisson_Q-parent.loc['scaled_fixed_common','Q'])
    parent_independent_error=abs(ninety_two.independent_Poisson_Q-parent.loc['scaled_fixed_independent','Q'])
    assert max(parent_common_error,parent_independent_error)<1e-7
    report=dict(passed=True,masses=len(out),mass_range_MeV=[90,94],mass_step_MeV=.25,
      width_t=1,fixed_union_windows=True,source_score_fields_sha256=sha(bank_path),
      source_individual_grid_sha256=sha(grid_path),script_sha256=sha(__file__),
      experiment_sha256=sha(B/'engine/experiment.py'),
      maximum_direct_score_difference=max(direct_errors),maximum_gaussian_grid_Q_replay_difference=max_replay,
      maximum_compressed_joint_loglikelihood_difference=compression_error,
      parent_92_Poisson_common_Q_difference=float(parent_common_error),
      parent_92_Poisson_independent_Q_difference=float(parent_independent_error),
      minimum_independent_minus_common_Gaussian_Q=min(nesting_margin),
      maximum_exact_limit_score=float(out.common_Poisson_limit_max_score.max()),
      maximum_CLs_formula_root_error=max(root_errors),coverage_checks=coverage,
      covariance_across_masses_preserved=True,inter_dataset_covariance='block diagonal, as in parent likelihood',
      upper_limit_estimand='Sum of expected reconstructed signal rows inside three fixed fitting windows',
      upper_limit_claim='Conservative 90% Gaussian CLs bound conditional on known frozen predictive covariance; no signal-allocation law required for signed-sum bound',
      probability_claim='Pointwise predictive/joint-auxiliary Gaussian reference; no mass, shape, model-choice or sideband-refit calibration',
      inputs_fixed_before_this_comparison=True,full_search_global_significance=False,
      summary_at_92_MeV={k:float(ninety_two[k]) for k in [
       'common_GLS_signed_Z','common_GLS_p_local','independent_GLS_Q','independent_GLS_Z_equivalent',
       'independent_GLS_p_local','signed_Stouffer_Z','signed_Stouffer_p_local',
       'signed_Fisher_Q','signed_Fisher_p_local','signed_Fisher_Z_equivalent',
       'common_GLS_total_yield_upper90','independent_GLS_total_yield_upper90',
       'common_Poisson_signed_root','independent_Poisson_raw_root']})
    (B/'qa/extracted_combination_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary_at_92_MeV'],indent=2))
    print('Validated 17 fixed-mass Gaussian combinations and matched Poisson profiles.',flush=True)


if __name__=='__main__':
    main()
