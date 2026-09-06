#!/usr/bin/env python3
"""Predeclared conventional fits; default invocation only prints the plan.

Execution requires an explicit persistence trigger. No production fitting module
is imported: inputs are the sealed, already released native histogram arrays.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev, legendre
from scipy.stats import chi2, norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT/'study_results/v4p9p16_presentation_extractions_20260906'
SELECTED = [('2015', 51), ('2015', 21), ('2016', 90), ('2016', 117),
            ('2021', 78), ('2021', 65)]
VARIANTS = [('baseline', 0, 0), ('degree_minus', -1, 0),
            ('degree_plus', 1, 0), ('width_minus', 0, -2), ('width_plus', 0, 2)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_inputs():
    files = [SOURCE/'derived/fit_arrays.npz', SOURCE/'derived/fit_summary.csv']
    manifest = {row['path']: row for row in csv.DictReader((SOURCE/'MANIFEST.csv').open())}
    for path in files:
        row = manifest[str(path.relative_to(ROOT))]
        if sha(path) != row['sha256'] or path.stat().st_size != int(row['bytes']):
            raise RuntimeError('Frozen source mismatch: '+str(path))
    arrays = np.load(files[0])
    summary = pd.read_csv(files[1], dtype={'dataset':str}, float_precision='round_trip')
    sources = {str(path.relative_to(ROOT)):sha(path) for path in files+[SOURCE/'MANIFEST.csv']}
    selected = []
    for year, mass in SELECTED:
        fid = f'{year}_m{mass:03d}'
        rows = summary[(summary.fit_id == fid) & (summary.dataset == year)]
        if len(rows) != 1:
            raise RuntimeError('Ambiguous source fit: '+fid)
        row = rows.iloc[0]; prefix = fid+'__'+year+'__'
        edges = arrays[prefix+'edges']; counts = arrays[prefix+'observed']
        if len(edges) != len(counts)+1 or not np.array_equal(counts, np.rint(counts)) or np.any(counts<0):
            raise RuntimeError('Invalid original histogram: '+fid)
        baseline_degree = 5 if year == '2015' and mass < 39 else 3
        baseline_width = (14 if mass < 39 else 13) if year == '2015' else 8
        selected.append(dict(fit_id=fid, dataset=year, mass_MeV=mass,
              sigma_MeV=float(row.sigma_MeV), edges=edges, counts=counts,
              baseline_degree=baseline_degree, baseline_total_width_sigma=baseline_width,
              basis='chebyshev' if year == '2015' else 'legendre',
              gp_root=float(row.signed_r), gp_full_yield=float(row.signal_full),
              source_scope=row.scope_key))
    return selected, sources


def window(item, width):
    edges = item['edges']; mass = item['mass_MeV']; sigma = item['sigma_MeV']
    span = width*sigma; lo = mass-span/2; hi = mass+span/2
    requested = [lo, hi]
    if span > edges[-1]-edges[0]:
        raise RuntimeError('Requested fit window exceeds spectrum support')
    if lo < edges[0]:
        lo, hi = edges[0], edges[0]+span
    if hi > edges[-1]:
        lo, hi = edges[-1]-span, edges[-1]
    centers = .5*(edges[:-1]+edges[1:])
    index = np.flatnonzero((centers >= lo) & (centers <= hi))
    if len(index) == 0 or not np.array_equal(index, np.arange(index[0],index[-1]+1)):
        raise RuntimeError('Invalid whole-bin fit window')
    return index, dict(requested_bounds_MeV=requested, shifted_bounds_MeV=[float(lo),float(hi)],
          effective_bounds_MeV=[float(edges[index[0]]),float(edges[index[-1]+1])],
          shifted_at_support=bool(lo != requested[0] or hi != requested[1]),
          bin_rule='Native bins with centers in the declared shifted window; no fractional counts.')


def plan(selected, sources):
    rows = []
    for item in selected:
        for variant, dn, dw in VARIANTS:
            degree = item['baseline_degree']+dn; width = item['baseline_total_width_sigma']+dw
            index, geometry = window(item,width)
            rows.append(dict(fit_id=item['fit_id'],dataset=item['dataset'],mass_MeV=item['mass_MeV'],
                  variant=variant,degree=degree,total_width_sigma=width,basis=item['basis'],
                  sigma_MeV=item['sigma_MeV'],n_bins=len(index),**geometry))
    return dict(status='prepared; execution requires a documented persistence trigger',
          new_random_toys=0, new_unblinded_events=0, fit_count=len(rows), fits=rows,
          source_sha256=sources, methods='Joint binned Poisson exponential-polynomial background and fixed Gaussian; signed auxiliary full-Gaussian yield.',
          interpretation='Five retained variants at GP-selected masses; nominal local references only, no global correction, coverage, or independent discovery claim.')


class LocalModel:
    def __init__(self, edges, counts, mass, sigma, degree, basis, quadrature=16):
        self.edges = np.asarray(edges); self.n = np.asarray(counts); self.mass = mass
        self.sigma = sigma; self.degree = degree; self.basis = basis
        self.scale = float(np.sqrt(self.n.sum()))
        if self.scale <= 0 or len(counts) <= degree+2:
            raise RuntimeError('Insufficient positive-count information')
        nodes, weights = legendre.leggauss(quadrature)
        center = .5*(edges[:-1]+edges[1:]); half = .5*np.diff(edges)
        self.nodes = center[:,None]+half[:,None]*nodes
        self.weights = half[:,None]*weights
        t = 2*(self.nodes-edges[0])/(edges[-1]-edges[0])-1
        self.P = (chebyshev.chebvander if basis == 'chebyshev' else legendre.legvander)(t,degree)
        self.signal = np.diff(norm.cdf((edges-mass)/sigma))
        self.signal_scaled = self.scale*self.signal
        self.quadrature = quadrature

    def background(self, coefficients):
        eta = self.P@coefficients
        if np.any(eta > 700) or np.any(eta < -700):
            raise FloatingPointError('Exponential polynomial outside numerical domain')
        weighted = self.weights*np.exp(eta)
        b = weighted.sum(axis=1)
        first = np.einsum('iq,iqk->ik',weighted,self.P)
        second = np.einsum('iq,iqj,iqk->ijk',weighted,self.P,self.P)
        return b,first,second

    def evaluate(self, parameters, free):
        coefficients = parameters[1:] if free else parameters
        alpha = parameters[0] if free else 0.
        b, first, second = self.background(coefficients)
        lam = b+alpha*self.signal_scaled
        if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
            raise FloatingPointError('Nonpositive or nonfinite Poisson expectation')
        positive = self.n > 0
        # High-count spectra make t-log1p(t) much smaller than its terms.
        # Accumulate this exact deviance in extended precision so a final
        # Newton step is not rejected by double-precision cancellation.
        n_long = self.n[positive].astype(np.longdouble)
        t = (lam[positive].astype(np.longdouble)-n_long)/n_long
        value = float(np.sum(n_long*(t-np.log1p(t)),dtype=np.longdouble)
                      +lam[~positive].sum(dtype=np.longdouble))
        residual = (lam-self.n)/lam; weight = self.n/lam**2
        J = np.column_stack((self.signal_scaled,first)) if free else first
        gradient = J.T@residual
        hessian = (J.T*weight)@J
        offset = int(free)
        hessian[offset:,offset:] += np.einsum('i,ijk->jk',residual,second)
        return value,gradient,.5*(hessian+hessian.T),b,lam

    def initial(self):
        center = .5*(self.edges[:-1]+self.edges[1:])
        t = 2*(center-self.edges[0])/(self.edges[-1]-self.edges[0])-1
        P = (chebyshev.chebvander if self.basis == 'chebyshev' else legendre.legvander)(t,self.degree)
        weight = np.sqrt(np.maximum(self.n,1.))
        return np.linalg.lstsq(P*weight[:,None],
               np.log(np.maximum(self.n,.5)/np.diff(self.edges))*weight,rcond=None)[0]

    def fit(self, initial, free):
        parameters = np.asarray(initial,float).copy(); shifts = 0; refinements = 0
        for iteration in range(200):
            value,g,H,b,lam = self.evaluate(parameters,free)
            eigen = np.linalg.eigvalsh(H)
            diag = np.maximum(np.abs(np.diag(H)),np.finfo(float).tiny)
            scaled_score = float(np.max(abs(g)/np.sqrt(diag)))
            if eigen[0] > 0 and scaled_score < 1e-7:
                break
            work = H
            if eigen[0] <= max(1.,eigen[-1])*1e-12:
                work = H+(max(1.,eigen[-1])*1e-9-min(0.,eigen[0]))*np.eye(len(g)); shifts += 1
            step = np.linalg.solve(work,-g); descent = float(g@step)
            if descent >= 0 or not np.isfinite(descent):
                raise RuntimeError('Invalid Newton descent')
            # At a stationary point the predicted NLL improvement can be
            # smaller than floating-point changes in the bin predictions.
            # Permit a full Newton refinement only when its score improves
            # by at least a factor of two and the NLL agrees to 1e-9.
            # The final score/covariance gates below remain unchanged.
            if eigen[0] > 0 and np.sqrt(max(0.,-descent)) < 1e-5:
                try:
                    refined = self.evaluate(parameters+step,free)
                    refined_score = np.max(abs(refined[1])/np.sqrt(np.maximum(abs(np.diag(refined[2])),np.finfo(float).tiny)))
                    if refined[0] <= value+1e-9 and refined_score < .5*scaled_score:
                        parameters += step; refinements += 1; continue
                except FloatingPointError:
                    pass
            fraction = 1.
            for _ in range(60):
                trial = parameters+fraction*step
                try:
                    new_value = self.evaluate(trial,free)[0]
                except FloatingPointError:
                    new_value = np.inf
                if new_value <= value+1e-4*fraction*descent+1e-12:
                    parameters = trial; break
                fraction *= .5
            else:
                raise RuntimeError('Line search failed')
        else:
            raise RuntimeError('Maximum iterations exceeded')
        covariance = np.linalg.inv(H)
        decrement = float(np.sqrt(max(0.,g@covariance@g)))
        if decrement > 1e-6 or np.any(np.linalg.eigvalsh(covariance) <= 0):
            raise RuntimeError('Unqualified local covariance or stationarity')
        return dict(parameters=parameters,nll=value,gradient=g,hessian=H,covariance=covariance,
              background=b,total=lam,iterations=iteration,scaled_score=scaled_score,
              newton_decrement=decrement,hessian_shifts=shifts,min_lambda=float(lam.min()),
              stationary_refinements=refinements,
              amplitude=float(parameters[0]*self.scale) if free else 0.)


def fit_coordinate(item, variant, dn, dw):
    degree = item['baseline_degree']+dn; width = item['baseline_total_width_sigma']+dw
    index, geometry = window(item,width)
    edges = item['edges'][index[0]:index[-1]+2]; counts = item['counts'][index]
    model = LocalModel(edges,counts,item['mass_MeV'],item['sigma_MeV'],degree,item['basis'])
    null = model.fit(model.initial(),False)
    # All starts are fixed before seeing the result. Retain their diagnostics.
    starts = [model.fit(np.r_[alpha,null['parameters']],True) for alpha in (0.,1.,-1.)]
    free = min(starts,key=lambda f:f['nll'])
    sigma_A = model.scale*np.sqrt(free['covariance'][0,0])
    amplitude_spread = float(np.ptp([f['amplitude'] for f in starts])/sigma_A)
    nll_spread = float(np.ptp([f['nll'] for f in starts]))
    if amplitude_spread > 1e-4 or nll_spread > 1e-6:
        raise RuntimeError('Fixed multistarts disagree')
    q = 2*(null['nll']-free['nll'])
    if q < -1e-7:
        raise RuntimeError('Free/null nesting failure')
    r = float(np.sign(free['amplitude'])*np.sqrt(max(0.,q)))
    # Higher quadrature evaluates saved parameters; it does not refit or select a model.
    finer = LocalModel(edges,counts,item['mass_MeV'],item['sigma_MeV'],degree,item['basis'],32)
    ff = finer.evaluate(free['parameters'],True); nn = finer.evaluate(null['parameters'],False)
    quadrature_relative = float(max(np.max(abs(ff[4]/free['total']-1)),np.max(abs(nn[4]/null['total']-1))))
    quadrature_q_error = float(abs(2*(nn[0]-ff[0])-q))
    if quadrature_relative > 1e-9 or quadrature_q_error > 1e-6:
        raise RuntimeError('Quadrature precision gate failed')
    full_cov = free['covariance'].copy(); full_cov[0,:] *= model.scale; full_cov[:,0] *= model.scale
    ndof = int(len(counts)-degree-2); null_ndof = int(len(counts)-degree-1)
    record = dict(fit_id=item['fit_id'],dataset=item['dataset'],mass_MeV=item['mass_MeV'],variant=variant,
          status='passed',basis=item['basis'],degree=degree,total_width_sigma=width,sigma_MeV=item['sigma_MeV'],
          n_bins=len(counts),low_MeV=float(edges[0]),high_MeV=float(edges[-1]),
          shifted_at_support=geometry['shifted_at_support'],amplitude_full=free['amplitude'],sigma_amplitude_full=float(sigma_A),
          signal_window_fraction=float(model.signal.sum()),signal_window_yield=float(free['amplitude']*model.signal.sum()),
          root=r,p0_nominal=float(norm.sf(max(0.,r))),q_raw=float(q),nll=free['nll'],null_nll=null['nll'],
          deviance=2*free['nll'],ndof=ndof,deviance_per_dof=2*free['nll']/ndof,
          nominal_GOF_p=float(chi2.sf(2*free['nll'],ndof)),null_deviance=2*null['nll'],null_ndof=null_ndof,
          gp_root=item['gp_root'],gp_full_yield=item['gp_full_yield'],max_scaled_score=max(f['scaled_score'] for f in starts+[null]),
          max_newton_decrement=max(f['newton_decrement'] for f in starts+[null]),min_lambda=min(f['min_lambda'] for f in starts+[null]),
          multistart_NLL_spread=nll_spread,multistart_amplitude_spread_in_SE=amplitude_spread,
          quadrature_relative_error=quadrature_relative,quadrature_q_error=quadrature_q_error,
          amplitude_scale=model.scale)
    metadata = dict(**record,geometry=geometry,starts=[{k:f[k] for k in ('nll','amplitude','iterations','scaled_score','newton_decrement','hessian_shifts','stationary_refinements')} for f in starts],
          null_diagnostics={k:null[k] for k in ('iterations','scaled_score','newton_decrement','hessian_shifts','stationary_refinements')},
          amplitude_definition='Total untruncated unit-normal Gaussian yield; expected window signal equals amplitude times stored signal_bin_probability sum.',
          polynomial_definition='exp(sum beta_k P_k(t)), integrated in each native bin; t maps the actual fit window to [-1,1].',
          probability_scope='Nominal asymptotic local reference at a GP-selected mass; five variants retained without selecting a preferred p-value.')
    arrays = dict(edges_MeV=edges,native_indices=index,counts=counts,signal_bin_probability=model.signal,
          background_free=free['background'],background_null=null['background'],total_free=free['total'],total_null=null['total'],
          free_parameters_scaled=free['parameters'],null_coefficients=null['parameters'],
          free_coefficients=free['parameters'][1:],free_gradient_scaled=free['gradient'],null_gradient=null['gradient'],
          free_hessian_scaled=free['hessian'],null_hessian=null['hessian'],free_covariance_full_amplitude=full_cov,
          null_covariance=null['covariance'],quadrature_nodes_MeV=model.nodes,quadrature_weights_MeV=model.weights,
          polynomial_basis_at_nodes=model.P)
    return record,metadata,arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--trigger-note',default='')
    parser.add_argument('--trigger-file',type=Path)
    args = parser.parse_args()
    selected,sources = load_inputs(); proposed = plan(selected,sources)
    if not args.execute:
        print(json.dumps(proposed,indent=2));return
    if not args.trigger_note.strip() or args.trigger_file is None or not args.trigger_file.is_file():
        parser.error('Execution requires --trigger-note and an existing --trigger-file documenting measured persistence.')
    if json.loads(args.trigger_file.read_text()).get('traditional_fits_triggered') is not True:
        parser.error('The saved persistence trigger must explicitly authorize conventional fits.')
    output = HERE/'derived'; points = output/'points'; points.mkdir(parents=True,exist_ok=True)
    started = time.monotonic(); records = []; failures = []
    dump(output/'execution_contract.json',dict(plan=proposed,trigger_note=args.trigger_note,
          trigger_file=str(args.trigger_file.resolve()),trigger_sha256=sha(args.trigger_file),script_sha256=sha(__file__)))
    for item in selected:
        for variant,dn,dw in VARIANTS:
            label = item['fit_id']+'__'+variant
            try:
                record,metadata,arrays = fit_coordinate(item,variant,dn,dw)
                np.savez_compressed(points/(label+'.npz'),**arrays);dump(points/(label+'.json'),metadata)
                records.append(record)
                print(label,'r=',round(record['root'],6),'D/ndof=',round(record['deviance_per_dof'],4),flush=True)
            except Exception as exc:
                failure = dict(fit_id=item['fit_id'],dataset=item['dataset'],mass_MeV=item['mass_MeV'],variant=variant,status='failed',error=str(exc))
                failures.append(failure);dump(points/(label+'_failure.json'),failure)
                print(label,'FAILED:',str(exc),flush=True)
    pd.DataFrame(records).to_csv(output/'fit_summary.csv',index=False)
    dump(output/'summary.json',dict(passed=len(records)==30 and not failures,completed=len(records),failures=failures,
          seconds=time.monotonic()-started,new_random_toys=0,new_unblinded_events=0,
          source_sha256=sources,scope=proposed['interpretation'],
          output_sha256={str(p.relative_to(ROOT)):sha(p) for p in points.iterdir() if p.is_file()}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
