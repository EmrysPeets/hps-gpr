"""Read-only use of frozen scan arrays; calculate correlation and LEE diagnostics."""
from pathlib import Path
import hashlib
import json
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
from scipy.stats import beta, norm

B = Path(__file__).resolve().parents[1]
SCOPES = ('2016', '2021', 'combined')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv(name, rows):
    pd.DataFrame(rows).to_csv(B/'results'/name, index=False, float_format='%.17g')


def tail(k, n):
    return dict(k=int(k), N=int(n), p=(k+1)/(n+1), fraction=k/n,
                low=0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
                high=1. if k == n else float(beta.ppf(.975, k+1, n-k)),
                Z=max(0., float(norm.isf((k+1)/(n+1)))))


def sidak(p, n):
    return float(-np.expm1(n*np.log1p(-p))) if p < 1 else 1.


def equivalent(p, alpha):
    return float(np.log1p(-p)/np.log1p(-alpha)) if p < 1 else None


def independent_marginals(ranks, threshold):
    f = (ranks <= threshold).mean(axis=0)
    return float(-np.expm1(np.log1p(-f).sum())) if np.all(f < 1) else 1.


def main():
    for line in (B/'provenance/input_manifest.sha256').read_text().splitlines():
        h, p = line.split('  ', 1)
        assert sha(B/p) == h, p
    geometry = pd.read_csv(B/'inputs/template_geometry.csv', dtype={'year': str}, float_precision='round_trip')
    old = pd.read_csv(B/'inputs/calibrated_summary.csv', dtype={'scope': str}, float_precision='round_trip').set_index('scope')
    obs = pd.read_csv(B/'inputs/observed_scan.csv', dtype={'scope': str}, float_precision='round_trip')
    protocol = dict(version='6.4.5', new_toys=0, reused_A_scans=1024, reused_B_scans=1024,
        primary='Unchanged frozen-A local rank map and complete-row B minimum; inclusive ties, add-one global rank.',
        correlation='Pearson correlations of signed likelihood roots, centered and standardized only for this diagnostic.',
        resolution_distance='|c_i-c_j|/sqrt((s_i^2+s_j^2)/2) using stored shifted MC core centers and widths.',
        pair_bins='0 to6 resolution units in steps0.25; median and16/84percentiles across mass pairs, not confidence intervals.',
        resolution_count='HPS-style N=W/mean_s; W=m_max-m_min, mean_s=integral s(m) dm/W by trapezoids on stored1MeV grid. Not fitted to toys.',
        legacy_resolution_count='Same formula using inherited Gaussian sigma_ref, provided for comparison only.',
        independence_control='1-product_m(1-F_B,m(alpha)); exact product of empirical marginal distributions, equivalent to independently resampling toy IDs at each mass. Counterfactual, not a calibrated result.',
        sidak_equivalent='N_eq(alpha)=log(1-p_B(alpha))/log(1-alpha); re-expression of the same B tail, not a fitted independent validation.',
        combined='Reuse archived joint fits with one shared psi; no single campaign width assigned to the combined search.',
        interpretation='Retrospective explanation and diagnostics; no new method selection, independent validation claim or change to the previously defined primary test.',
        limits='Conditional on A maps, fixed observed-derived null sources and declared1MeV grids; no map/source/earlier-selection uncertainty.',
        script_sha256=sha(__file__))
    write(B/'provenance/correlation_protocol.json', protocol)
    selected=[];curves=[];pairs=[];correlation_summary=[];widths=[];checks=0
    for scope in SCOPES:
        with np.load(B/f'inputs/global_scans_{scope}.npz') as a, np.load(B/f'inputs/global_validation_{scope}.npz') as v, np.load(B/f'inputs/calibrated_validation_{scope}.npz') as saved:
            m=a['masses_MeV'];ra=a['values'][:,:,0];rb=v['values'][:,:,0]
            assert np.array_equal(m,v['masses_MeV']); assert len(ra)==len(rb)==1024
            qa=np.maximum(ra,0)**2;qb=np.maximum(rb,0)**2
            qo=obs[obs.scope==scope].sort_values('mass_MeV').q0.to_numpy()
            ranks=np.empty(qb.shape,dtype=np.int32)
            for start in range(0,1024,32):
                ranks[start:start+32] = 1+(qa[None,:,:]>=qb[start:start+32,None,:]).sum(axis=1)
            ranks_obs=1+(qa>=qo).sum(axis=0)
            assert np.array_equal(ranks,saved['local_rank_counts'])
            assert np.array_equal(ranks_obs,saved['observed_rank_counts'])
            minimum=ranks.min(axis=1); assert np.array_equal(minimum,saved['min_rank_counts'])
            checks += 5
        ca=np.corrcoef(ra,rowvar=False);cb=np.corrcoef(rb,rowvar=False)
        assert np.allclose(ca,ca.T) and np.allclose(np.diag(ca),1)
        assert np.isfinite(ca).all() and np.isfinite(cb).all()
        # Independently verify representative correlation coefficients.
        for i,j in ((0,1),(0,len(m)-1),(len(m)//2,len(m)//2+1)):
            x=ra[:,i]-ra[:,i].mean();y=ra[:,j]-ra[:,j].mean()
            assert abs(x@y/np.sqrt((x@x)*(y@y))-ca[i,j])<1e-12
            checks += 1
        np.savez_compressed(B/f'results/correlations_{scope}.npz',masses_MeV=m,A=ca,B=cb)
        adjacent=np.diag(ca,1)
        correlation_summary.append(dict(scope=scope,grid_points=len(m),mass_low=int(m[0]),mass_high=int(m[-1]),
            adjacent_A_median=float(np.median(adjacent)),adjacent_A_min=float(adjacent.min()),adjacent_A_max=float(adjacent.max()),
            adjacent_B_median=float(np.median(np.diag(cb,1))),mean_absolute_A_B_difference=float(np.mean(abs(ca-cb)))))
        nres=nlegacy=None
        if scope!='combined':
            g=geometry[geometry.year==scope].sort_values('mass_MeV');assert np.array_equal(g.mass_MeV,m)
            s=g.core_width_MeV.to_numpy();c=g.center_MeV.to_numpy();ref=g.sigma_ref_MeV.to_numpy();W=float(m[-1]-m[0])
            mean=float(np.trapezoid(s,m)/W);meanref=float(np.trapezoid(ref,m)/W)
            nres=W/mean;nlegacy=W/meanref
            widths.append(dict(scope=scope,mass_span_MeV=W,mean_MC_core_width_MeV=mean,mean_legacy_width_MeV=meanref,
                N_MC_resolution=nres,N_legacy_resolution=nlegacy))
            i,j=np.triu_indices(len(m),1);du=abs(c[i]-c[j])/np.sqrt((s[i]**2+s[j]**2)/2)
            gaussian=np.sqrt(2*s[i]*s[j]/(s[i]**2+s[j]**2))*np.exp(-(c[i]-c[j])**2/(2*(s[i]**2+s[j]**2)))
            for lo in np.arange(0,6,.25):
                keep=(du>=lo)&(du<lo+.25)
                if not keep.any():continue
                v=ca[i[keep],j[keep]];w=cb[i[keep],j[keep]]
                pairs.append(dict(scope=scope,u_low=float(lo),u_high=float(lo+.25),u_center=float(np.median(du[keep])),pairs=int(keep.sum()),
                    A_median=float(np.median(v)),A_p16=float(np.quantile(v,.16)),A_p84=float(np.quantile(v,.84)),
                    B_median=float(np.median(w)),gaussian_overlap_median=float(np.median(gaussian[keep]))))
        j=int(np.argmin(ranks_obs));threshold=int(ranks_obs[j]);alpha=threshold/1025
        t=tail(int((minimum<=threshold).sum()),1024);p_ind=independent_marginals(ranks,threshold)
        assert t['k']==old.loc[scope,'minp_B_k'];assert abs(t['p']-old.loc[scope,'minp_B_p'])<1e-15
        neq=equivalent(t['p'],alpha)
        assert abs(sidak(alpha,neq)-t['p'])<1e-14
        selected.append(dict(scope=scope,mass_MeV=int(m[j]),local_rank_count=threshold,local_p=alpha,**t,
            independent_empirical_p=p_ind,correlation_reduction_vs_independent_fraction=p_ind-t['fraction'],
            N_MC_resolution=nres,N_legacy_resolution=nlegacy,N_toy_equivalent=neq,
            N_toy_equivalent_low=equivalent(t['low'],alpha),N_toy_equivalent_high=equivalent(t['high'],alpha),
            MC_resolution_linear_p=None if nres is None else min(1,nres*alpha),
            MC_resolution_sidak_p=None if nres is None else sidak(alpha,nres),
            legacy_resolution_linear_p=None if nlegacy is None else min(1,nlegacy*alpha),
            local_map_floor=bool(threshold==1)))
        for threshold in range(1,104):
            alpha=threshold/1025;t=tail(int((minimum<=threshold).sum()),1024)
            curves.append(dict(scope=scope,rank_threshold=threshold,alpha=alpha,**t,
                independent_empirical_p=independent_marginals(ranks,threshold),
                N_toy_equivalent=equivalent(t['p'],alpha),
                N_toy_equivalent_low=equivalent(t['low'],alpha),N_toy_equivalent_high=equivalent(t['high'],alpha),
                MC_resolution_linear_p=None if nres is None else min(1,nres*alpha),
                MC_resolution_sidak_p=None if nres is None else sidak(alpha,nres)))
        checks += 5
    csv('correlation_summary.csv',correlation_summary);csv('correlation_by_resolution_distance.csv',pairs)
    csv('resolution_counts.csv',widths);csv('correlation_global_summary.csv',selected)
    csv('correlation_global_curves.csv',curves)
    audit=json.loads((B/'inputs/common_coupling_audit.json').read_text())
    assert audit['passed'] and audit['exactly_one_shared_signal_parameter']
    write(B/'qa/numerical_validation.json',dict(passed=True,checks=checks,all_509952_B_local_ranks_recomputed=True,
        old_global_counts_and_probabilities_unchanged=True,common_coupling_audit_retained=True,
        every_B_scan_retains_original_mass_correlation=True,new_likelihood_fits=0,new_toys=0,
        primary_result='Same global probabilities as v6.4.4; correlation is already included, not an additional multiplier.'))
    print(pd.DataFrame(selected).to_string(index=False))


if __name__=='__main__':main()
