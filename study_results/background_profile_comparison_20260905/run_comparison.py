#!/usr/bin/env python3
"""Isolated observed Poisson/Gaussian, Poisson/log-GP and fixed-background scan."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARENT = REPO / 'study_results/v4p9p12_final_dataset_combinations_20260902'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PARENT))
import run_final_combinations as production
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from hps_gpr.gpr import fit_gpr, make_fixed_kernel, predict_counts_from_log_gpr
from hps_gpr.statistics import _chol_with_jitter
from hps_gpr.template import build_window_template_from_full
from bounded_tildeq_cls import bounded_tildeq_asymptotic_tails

OUT = HERE / 'derived'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def deviance_half(n, lam):
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
        return float('inf')
    positive = n > 0
    t = (lam[positive] - n[positive]) / n[positive]
    return float(np.sum(n[positive] * (t - np.log1p(t))) + lam[~positive].sum())


class Profile:
    def __init__(self, b, factor, w, mode):
        self.b = np.asarray(b, float)
        self.L = np.asarray(factor, float)
        self.w = np.asarray(w, float)
        self.mode = mode
        self.npar = self.L.shape[1]
        self.scale = float(np.sqrt(np.sum(self.b)))
        self.max_score = 0.
        self.max_iterations = 0
        self.hessian_shifts = 0

    def objective(self, z, n, fixed=None):
        a = z[0] if fixed is None else fixed / self.scale
        theta = z[1:] if fixed is None else z
        bfit = self.b * np.exp(self.L @ theta) if self.mode == 'log' else self.b + self.L @ theta
        lam = bfit + a * self.scale * self.w
        value = deviance_half(n, lam) + .5 * float(theta @ theta)
        if not np.isfinite(value):
            return value, None, None, bfit, lam
        J = bfit[:,None] * self.L if self.mode == 'log' else self.L
        if fixed is None:
            J = np.column_stack((self.scale*self.w,J))
        r = (lam - n) / lam
        gradient = J.T @ r
        hessian = (J.T * (n / lam**2)) @ J
        offset = int(fixed is None)
        gradient[offset:] += theta
        hessian[offset:,offset:] += np.eye(self.npar)
        if self.mode == 'log':
            hessian[offset:,offset:] += (self.L.T * (bfit*r)) @ self.L
        return value, gradient, hessian, bfit, lam

    def fit(self, n, fixed=None, initial=None):
        n = np.asarray(n,float)
        size = self.npar + int(fixed is None)
        z = np.zeros(size) if initial is None else np.asarray(initial,float).copy()
        for iteration in range(101):
            value,g,H,bfit,lam = self.objective(z,n,fixed)
            if not np.isfinite(value):
                raise RuntimeError('Invalid initial expectation')
            score = float(np.max(np.abs(g))) if size else 0.
            if score < 2e-7:
                break
            try:
                np.linalg.cholesky(H)
            except np.linalg.LinAlgError:
                H = H + (max(0.,-np.linalg.eigvalsh(H).min())+1e-7)*np.eye(size)
                self.hessian_shifts += 1
            step = np.linalg.solve(H, -g)
            descent = float(g@step)
            if descent >= 0:
                raise RuntimeError('Newton direction is not descent')
            alpha = 1.
            for _ in range(50):
                newvalue = self.objective(z+alpha*step,n,fixed)[0]
                if newvalue <= value + 1e-4*alpha*descent + 1e-12:
                    z += alpha*step
                    break
                alpha *= .5
            else:
                raise RuntimeError(f'Line search failed, score={score}')
        else:
            raise RuntimeError(f'Unconverged fit, score={score}')
        self.max_score = max(self.max_score,score)
        self.max_iterations = max(self.max_iterations,iteration)
        sigma = self.scale*np.sqrt(np.linalg.inv(H)[0,0]) if fixed is None else None
        return dict(A=float(z[0]*self.scale) if fixed is None else float(fixed),
                    nll=value, bfit=bfit, lam=lam, z=z, sigma=sigma,
                    score=score, iterations=iteration, min_lambda=float(lam.min()))

    def limit(self, n, alpha=.1):
        free = self.fit(n)
        null = self.fit(n,0.)
        if free['nll'] > null['nll'] + 1e-7:
            raise RuntimeError('Free/null nesting failure')
        denominator = free if free['A'] >= 0 else null
        trace = []
        def cls(A):
            # The null, theta=0, is exact on this model's Asimov spectrum.
            fp = self.fit(n,A)
            ap = self.fit(self.b,A)
            q = max(0.,2*(fp['nll']-denominator['nll'])) if free['A'] <= A else 0.
            qa = 2*ap['nll']
            if qa <= 0 or fp['nll'] < denominator['nll'] - 1e-7:
                raise RuntimeError('Invalid likelihood-ratio nesting')
            tails = bounded_tildeq_asymptotic_tails(q,qa)
            trace.append((float(A),float(tails.cls),q,qa))
            return float(tails.cls)
        lo = max(free['A'],0.) + self.scale*1e-5
        hi = max(free['A'],0.) + 3*free['sigma']
        clo = cls(lo)
        chi = cls(hi)
        while chi > alpha:
            hi *= 2
            chi = cls(hi)
        if clo < alpha:
            raise RuntimeError('Lower CLs bracket invalid')
        ul = brentq(lambda A:cls(A)-alpha,lo,hi,xtol=1e-6,rtol=2e-10)
        final_cls = cls(ul)
        ordered = sorted(trace)
        monotone = max([ordered[j+1][1]-ordered[j][1] for j in range(len(ordered)-1)]+[0.])
        if abs(final_cls-alpha)>2e-6 or monotone>5e-5:
            raise RuntimeError('CLs root/monotonicity failure')
        signed = float(np.sign(free['A'])*np.sqrt(max(0,2*(null['nll']-free['nll']))))
        return dict(A90=ul,Ahat=free['A'],sigma_A=free['sigma'],signed_r=signed,
                    cls=final_cls,q_obs=trace[-1][2],q_asimov=trace[-1][3],
                    max_score=self.max_score,max_iterations=self.max_iterations,
                    hessian_shifts=self.hessian_shifts,monotonicity_error=monotone,
                    min_lambda=min(free['min_lambda'],null['min_lambda']),
                    free=free,null=null,trace=trace)


def condition_log(K):
    K = .5*(K+K.T)
    scale = max(float(np.max(np.abs(np.diag(K)))),np.finfo(float).tiny)
    for rel in (1e-10,1e-9,1e-8,1e-7,1e-6,1e-5):
        loaded = K+rel*scale*np.eye(len(K))
        try:
            return np.linalg.cholesky(loaded),loaded,rel
        except np.linalg.LinAlgError:
            continue
    raise RuntimeError('Log covariance needs a load >=1e-4')


def derivative_checks(model,n):
    z = np.linspace(-.03,.03,model.npar+1)
    f,g,H,_,_ = model.objective(z,n)
    h = 1e-4
    numeric = np.array([(model.objective(z+np.eye(len(z))[j]*h,n)[0]
                       -model.objective(z-np.eye(len(z))[j]*h,n)[0])/(2*h)
                       for j in range(len(z))])
    numericH = np.column_stack([(model.objective(z+np.eye(len(z))[j]*h,n)[1]
                              -model.objective(z-np.eye(len(z))[j]*h,n)[1])/(2*h)
                              for j in range(len(z))])
    ge = float(np.max(np.abs(numeric-g)))
    he = float(np.max(np.abs(numericH-H)))
    if ge > 5e-6 or he > 5e-6:
        raise RuntimeError(f'Analytic derivative mismatch: {ge}, {he}')
    return dict(gradient_max_abs=ge,hessian_max_abs=he)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--masses',type=int,nargs='*')
    args=parser.parse_args()
    masses=args.masses or list(range(50,251))
    started=time.monotonic()
    OUT.mkdir(exist_ok=True)
    (OUT/'checkpoints').mkdir(exist_ok=True)
    sources=[production.DEFAULT_CARD,production.DEFAULT_STATES,
        PARENT/'derived/all_three_peak_extraction_plot_data.csv',
        PARENT/'derived/final_dataset_result_curves.csv',
        PARENT/'derived/prediction_state_ledger.csv',
        PARENT/'piecewise_cached_solver.py',HERE/'PROTOCOL.md',Path(__file__)]
    sources += list((production.RUNTIME_CAMPAIGN/'runtime_combined/hps_gpr').glob('*.py'))
    cfg=production.load_config(production.DEFAULT_CARD)
    production.validate_card(cfg)
    sources.append(Path(cfg.path_2021))
    protected={str(p):sha(p) for p in sources}
    if protected[str(Path(cfg.path_2021))] != production.EXPECTED_HISTOGRAM_SHA256['2021']:
        raise RuntimeError('Native histogram file hash mismatch')
    datasets=production.make_datasets(cfg)
    states=production.state_map(pd.read_csv(production.DEFAULT_STATES))
    data=pd.read_csv(sources[2]);data=data[data.dataset.astype(str)=='2021'].sort_values('bin_center_GeV')
    x=data.bin_center_GeV.to_numpy(float);y=data.observed_events.to_numpy(float)
    widths=data.bin_width_MeV.to_numpy(float)/1000
    edges=np.r_[x-widths/2,x[-1]+widths[-1]/2]
    assert np.all(y>0) and np.array_equal(y,np.rint(y))
    saved=pd.read_csv(sources[3]);saved=saved[saved.scope_key=='individual_2021_10pct'].set_index('mass_MeV')
    ledger=pd.read_csv(sources[4]);ledger=ledger[ledger.dataset.astype(str)=='2021'].set_index('mass_MeV')
    native=production.estimate_background_for_dataset(datasets['2021'],.071,cfg,restarts=0,
        kernel=make_fixed_kernel(states['2021',71]['const_opt'],states['2021',71]['ls_opt']),
        optimize=False,train_exclude_nsigma=2.25)
    if not np.array_equal(y,native.y_full) or not np.allclose(x,native.x_full,rtol=0,atol=1e-15):
        raise RuntimeError('Archived spectrum does not match the native histogram')
    # Preserve bit-exact mass coordinates: CSV decimal parsing can perturb the
    # ill-conditioned GP algebra even when it changes x by only one ulp.
    csv_x_delta=float(np.max(np.abs(x-native.x_full)))
    x=native.x_full.copy(); y=native.y_full.copy(); edges=native.edges_full.copy()
    widths=np.diff(edges)
    records=[];checks=[];plotrows=[];details=[]
    for mass in masses:
        m=mass/1000;sig=float(np.polynomial.polynomial.polyval(m,cfg.sigma_coeffs_2021))
        st=states['2021',mass]
        pred=production.estimate_background_for_dataset(datasets['2021'],m,cfg,restarts=0,
            kernel=make_fixed_kernel(st['const_opt'],st['ls_opt']),optimize=False,train_exclude_nsigma=2.25)
        if production.prediction_state_sha256(pred) != ledger.loc[mass,'prediction_state_sha256']:
            raise RuntimeError(f'Released prediction hash mismatch at {mass}')
        mask=pred.blind_mask
        gp=fit_gpr(x[~mask],y[~mask],cfg,restarts=0,
                   kernel=make_fixed_kernel(st['const_opt'],st['ls_opt']),optimize=False)
        lml_delta=float(gp.log_marginal_likelihood_value_-st['lml'])
        if abs(lml_delta)>5e-5:
            raise RuntimeError(f'LML reconstruction failed at {mass}: {lml_delta}')
        g,K=gp.predict(np.log(x[mask]).reshape(-1,1),return_cov=True)
        b,C=predict_counts_from_log_gpr(gp,x[mask],cfg)
        if not np.array_equal(b,pred.mu) or not np.array_equal(C,pred.cov):
            raise RuntimeError('Direct GP reconstruction differs from native release prediction')
        cov,cond=production.condition_covariance_block(C,b)
        L=_chol_with_jitter(cov)
        R,KL,load=condition_log(K)
        w,_=build_window_template_from_full(edges,mask,m,sig,config=cfg)
        fraction=float(w.sum());w=np.asarray(w,float)/fraction
        conversion=float(saved.loc[mass,'signal_yield_per_eps2_total'])*fraction
        models={'gaussian_control':Profile(b,L,w,'linear'),
                'log_gp':Profile(np.exp(g),R,w,'log'),
                'fixed':Profile(b,np.zeros((len(b),0)),w,'linear')}
        results={name:model.limit(y[mask]) for name,model in models.items()}
        reference=json.loads(saved.loc[mass,'limit_profile_status'])['observed']['base']
        rf=reference['fit_unbounded']
        reference_r=float(np.sign(rf['A_hat'])*np.sqrt(max(0,2*(reference['null']['nll']-rf['nll']))))
        current=float(saved.loc[mass,'eps2_90'])
        record=dict(mass_MeV=mass,sigma_MeV=sig*1000,n_fit_bins=int(mask.sum()),
                    eps2_current=current,r_current=reference_r,
                    current_Ahat_window=float(rf['A_hat']),
                    conversion_window_per_eps2=conversion,
                    template_window_fraction=fraction,lml_delta=lml_delta,
                    log_cov_relative_load=load,count_cov_relative_load=cond['selected_diagonal_load_relative'],
                    prediction_hash_exact=True,
                    log_mean_median_relative=float(np.max(b/np.exp(g)-1)),
                    gp_relative_sd_max=float(np.max(np.sqrt(np.diag(C))/b)))
        logmom_cov=np.outer(b,b)*np.expm1(KL)
        record['log_loading_count_cov_max_relative']=float(np.max(np.abs(logmom_cov-C))/np.max(np.diag(C)))
        for name,res in results.items():
            record.update({f'eps2_{name}':res['A90']/conversion,f'r_{name}':res['signed_r'],
                           f'Ahat_{name}':res['Ahat'],f'sigma_A_{name}':res['sigma_A'],
                           f'cls_{name}':res['cls'],f'score_{name}':res['max_score'],
                           f'min_lambda_{name}':res['min_lambda']})
            details.append(dict(mass_MeV=mass,model=name,
                **{k:v for k,v in res.items() if k not in ('free','null')}))
        record['gaussian_control_relative_to_release']=record['eps2_gaussian_control']/current-1
        records.append(record)
        if mass in (65,71,78,182):
            for name,model in models.items():
                checks.append(dict(mass_MeV=mass,model=name,**derivative_checks(model,y[mask])))
            fixed=results['fixed'];low=-float(np.min(b/w))*(1-1e-10)
            score=lambda a:float(np.sum(w*(1-y[mask]/(b+a*w))))
            independent=brentq(score,low,max(fixed['Ahat']+10*fixed['sigma_A'],10*fixed['sigma_A']),xtol=1e-6)
            if abs(independent-fixed['Ahat'])>1e-3:
                raise RuntimeError('Independent fixed-background MLE did not close')
            checks.append(dict(mass_MeV=mass,model='fixed_independent',Ahat_abs_difference=abs(independent-fixed['Ahat'])))
            bf,_=predict_counts_from_log_gpr(gp,x,cfg)
            index=0
            for j in np.flatnonzero(abs(x-m)<=4*sig):
                p=dict(mass_MeV=mass,bin_center_MeV=x[j]*1000,bin_width_MeV=widths[j]*1000,
                       observed=y[j],gp_mean=bf[j],in_fit=bool(mask[j]))
                if mask[j]:
                    ii=int(np.sum(mask[:j]))
                    p['gp_sd']=float(np.sqrt(C[ii,ii]))
                    for name,res in results.items():
                        p[f'b_{name}']=float(res['free']['bfit'][ii])
                        p[f's_{name}']=float(res['Ahat']*w[ii])
                        p[f'total_{name}']=float(res['free']['lam'][ii])
                plotrows.append(p)
        (OUT/'checkpoints'/f'm{mass:03d}.json').write_text(json.dumps(record,indent=2)+'\n')
        if mass%25==0 or mass==masses[-1]:
            print(f'{mass} MeV: log/current={record["eps2_log_gp"]/current:.6f}, fixed/current={record["eps2_fixed"]/current:.4f}',flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(OUT/'observed_limits.csv',index=False)
    pd.DataFrame(plotrows).to_csv(OUT/'fit_plot_data.csv',index=False)
    (OUT/'numerical_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (OUT/'profile_diagnostics.json').write_text(json.dumps(details,indent=2)+'\n')
    drift=[p for p,h in protected.items() if sha(p)!=h]
    if drift:raise RuntimeError(f'Source drift: {drift}')
    summary=dict(status='complete' if len(frame)==201 else 'pilot',mass_points=len(frame),new_toys=0,
       max_abs_gaussian_release_relative=float(abs(frame.gaussian_control_relative_to_release).max()),
       max_abs_r_gaussian_release=float(abs(frame.r_gaussian_control-frame.r_current).max()),
       max_abs_log_current_relative=float(abs(frame.eps2_log_gp/frame.eps2_current-1).max()),
       median_log_current=float((frame.eps2_log_gp/frame.eps2_current).median()),
       median_fixed_current=float((frame.eps2_fixed/frame.eps2_current).median()),
       min_fixed_current=float((frame.eps2_fixed/frame.eps2_current).min()),
       max_fixed_current=float((frame.eps2_fixed/frame.eps2_current).max()),
       maximum_gp_fractional_sd=float(frame.gp_relative_sd_max.max()),
       maximum_log_mean_median_relative=float(frame.log_mean_median_relative.max()),
       native_histogram_sha256=protected[str(Path(cfg.path_2021))],
       archived_counts_exactly_match_native=True,csv_coordinate_max_abs_delta=csv_x_delta,
       all_prediction_hashes_exact=bool(frame.prediction_hash_exact.all()),
       sources=protected,parent_sources_unchanged=True,
       runtime_provenance=production.RUNTIME_PROVENANCE,elapsed_seconds=time.monotonic()-started)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('sources','runtime_provenance')},indent=2))


if __name__=='__main__':
    main()
