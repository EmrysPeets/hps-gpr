#!/usr/bin/env python3
"""Independent numerical checks and frozen-input verification; no parent writes."""
import json
from pathlib import Path
import run_comparison as c
from scipy.optimize import minimize, brentq
from hps_gpr.statistics import fit_A_profiled_gaussian_details


def main():
    cfg=c.production.load_config(c.production.DEFAULT_CARD)
    ds=c.production.make_datasets(cfg)['2021']
    states=c.production.state_map(c.pd.read_csv(c.production.DEFAULT_STATES))
    records=[]
    for mass in (59,65,71,78,86,182,231):
        st=states['2021',mass];m=mass/1000
        p=c.production.estimate_background_for_dataset(ds,m,cfg,restarts=0,
            kernel=c.make_fixed_kernel(st['const_opt'],st['ls_opt']),optimize=False,train_exclude_nsigma=2.25)
        cov,_=c.production.condition_covariance_block(p.cov,p.mu)
        w,_=c.build_window_template_from_full(p.edges_full,p.blind_mask,m,p.sigma_val,config=cfg)
        w=w/w.sum()
        gp=c.fit_gpr(p.x_full[~p.blind_mask],p.y_full[~p.blind_mask],cfg,restarts=0,
            kernel=c.make_fixed_kernel(st['const_opt'],st['ls_opt']),optimize=False)
        g,K=gp.predict(c.np.log(p.x_full[p.blind_mask]).reshape(-1,1),return_cov=True)
        R,_,_=c.condition_log(K)
        models={'gaussian':c.Profile(p.mu,c._chol_with_jitter(cov),w,'linear'),
                'log_gp':c.Profile(c.np.exp(g),R,w,'log')}
        for name,model in models.items():
            fit=model.fit(p.obs)
            for sign in (-1,1):
                init=c.np.r_[sign*3.,sign*c.np.linspace(-.2,.2,model.npar)]
                other=minimize(lambda z:model.objective(z,p.obs)[0],init,
                    jac=lambda z:model.objective(z,p.obs)[1],method='BFGS',
                    options={'gtol':2e-7,'maxiter':500})
                val,grad,_,_,_=model.objective(other.x,p.obs)
                delta=float(val-fit['nll'])
                da=float(abs(other.x[0]*model.scale-fit['A']))
                if abs(delta)>1e-7 or da>1e-2 or c.np.max(abs(grad))>2e-5:
                    raise RuntimeError(f'Independent optimizer disagreement at {mass}/{name}: {delta}, {da}')
                records.append(dict(mass_MeV=mass,model=name,start_sign=sign,
                    delta_nll=delta,delta_A=da,score=float(c.np.max(abs(grad))),
                    scipy_status=int(other.status)))
            if name=='gaussian':
                old=fit_A_profiled_gaussian_details(p.obs,p.mu,cov,w,allow_negative=True)
                z=c.np.r_[old['A_hat']/model.scale,old['theta_hat']]
                value,gradient,_,_,_=model.objective(z,p.obs)
                records.append(dict(mass_MeV=mass,model='released_optimizer_diagnostic',
                    legacy_success=bool(old['success']),legacy_scaled_score=float(c.np.max(abs(gradient))),
                    centered_nll_improvement=float(value-fit['nll']),
                    legacy_A=float(old['A_hat']),stable_A=fit['A']))
    # Independently derive fixed-background bounded CLs in a one-bin example.
    n=c.np.array([93.]);b=c.np.array([100.]);w=c.np.ones(1)
    model=c.Profile(b,c.np.zeros((1,0)),w,'linear');res=model.limit(n)
    def analytic(A):
        qo=2*(A-93*c.np.log1p(A/100.))
        qa=2*(A-100*c.np.log1p(A/100.))
        return c.bounded_tildeq_asymptotic_tails(qo,qa).cls
    expected=brentq(lambda a:analytic(a)-.1,.01,100.)
    assert abs(expected-res['A90'])<1e-6
    summary=json.loads((c.OUT/'summary.json').read_text())
    assert summary['status']=='complete' and summary['all_prediction_hashes_exact']
    for p,h in summary['sources'].items():
        assert c.sha(p)==h,p
    frame=c.pd.read_csv(c.OUT/'observed_limits.csv')
    assert frame.mass_MeV.tolist()==list(range(50,251))
    for name in ('gaussian_control','log_gp','fixed'):
        assert c.np.all(c.np.isfinite(frame[f'eps2_{name}']))
        assert c.np.all(frame[f'eps2_{name}']>0)
        assert max(abs(frame[f'cls_{name}']-.1))<2e-6
        assert frame[f'min_lambda_{name}'].min()>0
    payload=dict(status='passed',mass_grid_points=201,new_profile_limits=603,
        source_hashes_unchanged=True,all_native_prediction_hashes_exact=True,
        independent_multistart_fits=28,one_bin_fixed_UL_abs_difference=abs(expected-res['A90']),
        checks=records,script_sha256=c.sha(Path(__file__)))
    (c.OUT/'validation.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='checks'},indent=2))


if __name__=='__main__':
    main()
