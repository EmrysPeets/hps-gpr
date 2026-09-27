"""One-signal Poisson/Gaussian profile limits; deterministic, no random draws.

Contract: ``OneSignalProfile(b, L, S).limit(n)`` with
``lambda = b + L @ theta + A * S`` and penalty ``theta @ theta / 2``.
S is the full-yield template restricted to the fitted bins, in counts per chosen
A unit. It is NEVER normalized here. L is any supplied covariance square root.
Negative unconstrained A is allowed while Poisson means remain nonnegative.

Observed AND background-Asimov likelihood ratios are explicitly profiled; only
their sampling tails are asymptotic. See Cowan et al., arXiv:1007.1727, Eqs.
(16), (65)-(67). This is a conditional, fixed-mass reference, not calibration.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize, nnls
from scipy.special import log_ndtr, ndtr


def bounded_tildeq_tails(q: float, qa: float) -> dict:
    """CLs = right tail under A / right tail under 0, including deficit branch.

    qa is the *profiled* background-Asimov q at this tested A. At q=0 the
    inclusive-tail convention is CLs=1. Log tails prevent underflow in deficits.
    """
    q, qa = float(q), float(qa)
    if not np.isfinite(q + qa) or q < 0 or qa <= 0:
        raise ValueError("q must be finite/nonnegative and qa finite/positive")
    if q <= 1e-14:
        return dict(cls=1., cl_sb=1., cl_b=1., branch="zero")
    a = math.sqrt(qa)
    if q <= qa:
        zsb = math.sqrt(q)
        zb = zsb - a
        branch = "interior"
    else:
        zsb = (q + qa) / (2 * a)
        zb = (q - qa) / (2 * a)
        branch = "deficit"
    lsb, lb = float(log_ndtr(-zsb)), float(log_ndtr(-zb))
    return dict(cls=math.exp(min(0., lsb - lb)), cl_sb=math.exp(lsb),
                cl_b=math.exp(lb), log_cl_sb=lsb, log_cl_b=lb, branch=branch)


def poisson_deviance_half(n: np.ndarray, lam: np.ndarray) -> float:
    """Half Poisson deviance, with exact n=0 contribution lambda."""
    positive = n > 0
    if np.any(lam < 0) or np.any(lam[positive] <= 0) or not np.all(np.isfinite(lam)):
        return math.inf
    np_, lp = n[positive], lam[positive]
    t = (lp - np_) / np_
    close = np.abs(t) < .5
    value = np.sum(np_[close] * (t[close] - np.log1p(t[close])))
    value += np.sum(lp[~close] - np_[~close] + np_[~close] * np.log(np_[~close] / lp[~close]))
    return float(value + np.sum(lam[~positive]))


class OneSignalProfile:
    """Convex Poisson/Gaussian profiles with safeguarded Newton optimization.

    Fast path uses Cholesky Newton solves and warm starts. Linear positivity
    constraints use SLSQP only if an empty bin reaches a physical boundary.
    ``fit`` raises on failed numerical/KKT checks; no failed fit becomes a limit.
    """
    def __init__(self, b, L, S, *, score_tolerance=2e-7):
        self.b = np.asarray(b, dtype=float).reshape(-1)
        self.L = np.asarray(L, dtype=float)
        self.S = np.asarray(S, dtype=float).reshape(-1)
        if not len(self.b) or self.L.ndim != 2 or self.L.shape[0] != len(self.b) or self.S.shape != self.b.shape:
            raise ValueError("Expected b,S vectors and L with matching rows")
        if not all(np.all(np.isfinite(a)) for a in (self.b, self.L, self.S)):
            raise ValueError("Nonfinite model inputs")
        if np.any(self.b <= 0) or np.any(self.S < 0) or not np.any(self.S > 0):
            raise ValueError("Require b>0 and a nonzero nonnegative signal template")
        self.rank = self.L.shape[1]
        self.scale = float(1 / np.sqrt(np.sum(self.S ** 2 / self.b)))
        self.Jfree = np.column_stack((self.scale * self.S, self.L))
        self.penfree = np.r_[0., np.ones(self.rank)]
        self.score_tolerance = float(score_tolerance)
        self._asimov_cache = {}

    def _counts(self, n):
        n = np.asarray(n, dtype=float).reshape(-1)
        if n.shape != self.b.shape or np.any(n < 0) or not np.all(np.isfinite(n)):
            raise ValueError("Counts must match b and be finite/nonnegative")
        return n

    def _objective(self, z, n, J, base, pen, hessian=True):
        lam = base + J @ z
        value = poisson_deviance_half(n, lam) + .5 * float(np.dot(pen * z, z))
        if not np.isfinite(value):
            return value, None, None, lam
        positive = n > 0
        ratio = np.zeros_like(lam)
        np.divide(n, lam, out=ratio, where=positive)
        gradient = J.T @ (1 - ratio) + pen * z
        H = None
        if hessian:
            curvature = np.zeros_like(lam)
            np.divide(n, lam ** 2, out=curvature, where=positive)
            H = (J.T * curvature) @ J
            H.flat[::len(H) + 1] += pen
        return value, gradient, H, lam

    def _boundary_fit(self, n, J, base, pen, initial):
        """Rare exact linear-boundary fallback; report KKT residual, not raw g."""
        # A zero expectation is permitted precisely where n=0. The tiny floor
        # for n>0 only protects log evaluation during the constrained search.
        floor = np.where(n > 0, np.maximum(self.b, 1.) * 1e-14, 0.)
        def fg(z):
            f, g, _, _ = self._objective(z, n, J, base, pen, False)
            if not np.isfinite(f):
                # SLSQP may evaluate infinitesimally outside a linear bound.
                lam = np.maximum(base + J @ z, np.maximum(floor, 1e-15))
                f = poisson_deviance_half(n, lam) + .5 * float(np.dot(pen*z, z))
                g = J.T @ (1 - n / lam) + pen * z
            return f, g
        result = minimize(fg, initial, jac=True, method="SLSQP",
            constraints={"type": "ineq", "fun": lambda z: base + J@z - floor,
                         "jac": lambda z: J},
            options={"ftol": 2e-12, "maxiter": 300})
        z = result.x
        f, g, H, lam = self._objective(z, n, J, base, pen)
        # Move negligible infeasibility to an interior feasible point, then
        # re-evaluate the original objective; never clip the reported lambda.
        if not np.isfinite(f) and np.min(lam - floor) > -1e-8:
            for fraction in (1e-12, 1e-11, 1e-10, 1e-9):
                trial = (1 - fraction) * z + fraction * initial
                f, g, H, lam = self._objective(trial, n, J, base, pen)
                if np.isfinite(f):
                    z = trial
                    break
        if not np.isfinite(f):
            raise RuntimeError(f"Boundary fit infeasible: {result.message}")
        active = (lam - floor) <= 1e-7 * np.maximum(self.b, 1.)
        if np.any(active):
            dual, _ = nnls(J[active].T, g)
            residual = g - J[active].T @ dual
            complementarity = float(np.max(np.abs(dual * (lam[active]-floor[active]))))
        else:
            residual = g
            complementarity = 0.
        score = max(float(np.max(np.abs(residual), initial=0.)), complementarity)
        if score > 3e-5 or np.min(lam) < 0:
            raise RuntimeError(f"Boundary KKT failure: {score:g}; {result.message}")
        return z, f, g, H, lam, score, int(result.nit), "boundary_slsqp"

    def fit(self, n, fixed=None, initial=None):
        n = self._counts(n)
        free = fixed is None
        if not free and (not np.isfinite(fixed) or fixed < 0):
            raise ValueError("A fixed for upper-limit profiles must be nonnegative")
        J = self.Jfree if free else self.L
        pen = self.penfree if free else np.ones(self.rank)
        base = self.b if free else self.b + float(fixed) * self.S
        z = np.zeros(J.shape[1]) if initial is None else np.asarray(initial, float).copy()
        if z.shape != (J.shape[1],):
            raise ValueError("Initial fit vector has wrong dimension")
        if not np.isfinite(self._objective(z, n, J, base, pen, False)[0]):
            z = np.zeros(J.shape[1])
        # The origin is strictly feasible, even when the warm start lies on
        # an empty-bin boundary and cannot support numerical backtracking.
        feasible_start = np.zeros(J.shape[1])
        method = "newton"
        for iteration in range(70):
            value, g, H, lam = self._objective(z, n, J, base, pen)
            score = float(np.max(np.abs(g), initial=0.))
            if score <= self.score_tolerance:
                break
            try:
                step = cho_solve(cho_factor(H, lower=True, check_finite=False), -g, check_finite=False)
            except np.linalg.LinAlgError:
                break
            descent = float(g @ step)
            if not np.isfinite(descent) or descent >= 0:
                break
            dlam = J @ step
            falling = dlam < 0
            fraction = min(1., .995 * float(np.min(-lam[falling] / dlam[falling]))) if np.any(falling) else 1.
            accepted = False
            for _ in range(35):
                newvalue = self._objective(z + fraction * step, n, J, base, pen, False)[0]
                if newvalue <= value + 1e-4 * fraction * descent + 2e-12:
                    z += fraction * step
                    accepted = True
                    break
                fraction *= .5
            if not accepted or fraction < 1e-12:
                break
        else:
            score = math.inf
        # Refresh after an accepted step, including early boundary detection.
        value, g, H, lam = self._objective(z, n, J, base, pen)
        score = float(np.max(np.abs(g), initial=0.))
        if score > self.score_tolerance:
            z, value, g, H, lam, score, iteration, method = self._boundary_fit(n, J, base, pen, feasible_start)
        sigma = None
        if free:
            try:
                unit = np.zeros(len(z)); unit[0] = 1.
                sigma = self.scale * math.sqrt(float(cho_solve(cho_factor(H, lower=True, check_finite=False), unit, check_finite=False)[0]))
            except (np.linalg.LinAlgError, ValueError):
                # Curvature need not exist at a zero-count boundary. This
                # Fisher scale is a root-bracketing aid only, never the limit.
                sigma = self.fisher_sigma()
        theta = z[1:] if free else z
        return dict(A=float(z[0] * self.scale) if free else float(fixed),
                    nll=float(value), z=z, theta=theta, lam=lam,
                    bfit=self.b + self.L @ theta, sigma=sigma, score=score,
                    iterations=iteration, method=method,
                    min_lambda=float(np.min(lam)),
                    min_background=float(np.min(self.b + self.L @ theta)))

    def fisher_sigma(self):
        H = (self.Jfree.T / self.b) @ self.Jfree
        H.flat[::len(H)+1] += self.penfree
        unit = np.zeros(len(H)); unit[0] = 1.
        return self.scale * math.sqrt(float(cho_solve(cho_factor(H, lower=True, check_finite=False), unit, check_finite=False)[0]))

    def limit(self, n, alpha=.1, *, details=False):
        """Solve CLs=alpha, returning scalars by default and optional fit trace."""
        if not 0 < alpha < .5:
            raise ValueError("alpha must lie between 0 and 0.5")
        n = self._counts(n)
        null = self.fit(n, 0.)
        free = self.fit(n, initial=np.r_[0., null["z"]])
        if free["nll"] > null["nll"] + 2e-6:
            raise RuntimeError("Free/null likelihood nesting failure")
        denominator = free if free["A"] >= 0 else null
        fits = [free, null]
        cache = {}
        obs_initial = denominator["theta"].copy()
        asimov_initial = np.zeros(self.rank)
        def cls(A):
            nonlocal obs_initial, asimov_initial
            A = float(A)
            if A <= max(0., free["A"]):
                return 1.
            if A in cache:
                return cache[A]["cls"]
            fp = self.fit(n, A, initial=obs_initial)
            # The Asimov null has theta=0 and deviance exactly zero.
            ap = self._asimov_cache.get(A)
            if ap is None:
                ap = self.fit(self.b, A, initial=asimov_initial)
                self._asimov_cache[A] = ap
            obs_initial, asimov_initial = fp["z"], ap["z"]
            fits.extend((fp, ap))
            q = 2 * (fp["nll"] - denominator["nll"])
            qa = 2 * ap["nll"]
            if q < -2e-6 or qa <= 0:
                raise RuntimeError("Invalid profiled likelihood ratio")
            tails = bounded_tildeq_tails(max(0., q), qa)
            cache[A] = dict(A=A, q_obs=max(0., q), q_asimov=qa, **tails)
            return tails["cls"]
        lo = max(free["A"], 0.)
        hi = lo + 3 * max(free["sigma"], self.scale)
        for _ in range(35):
            if cls(hi) <= alpha:
                break
            hi = lo + 2 * (hi - lo)
        else:
            raise RuntimeError("CLs root could not be bracketed")
        # Solve in scaled units so eps^2 and count amplitudes use the same
        # relative numerical accuracy without an arbitrary physical-unit xtol.
        root = self.scale * brentq(lambda x: cls(x * self.scale) - alpha,
            lo / self.scale, hi / self.scale, xtol=2e-7, rtol=2e-9)
        final_cls = cls(root)
        trace = [cache[k] for k in sorted(cache)]
        monotone = max([trace[i+1]["cls"] - trace[i]["cls"] for i in range(len(trace)-1)] + [0.])
        if abs(final_cls - alpha) > 2e-6 or monotone > 5e-5:
            raise RuntimeError(f"CLs root/monotonicity check failed: {final_cls}, {monotone}")
        r = math.copysign(math.sqrt(max(0., 2*(null["nll"]-free["nll"]))), free["A"])
        row = dict(A90=float(root), Ahat=free["A"], Ahat_bounded=max(free["A"], 0.),
            sigma_A=free["sigma"], signed_r=r, Z0=max(0., r),
            p0_fixed_mass=float(ndtr(-max(0., r))), p_signed=float(ndtr(-r)),
            cls=float(final_cls), q_obs=cache[root]["q_obs"], q_asimov=cache[root]["q_asimov"],
            cls_branch=cache[root]["branch"], max_score=max(f["score"] for f in fits),
            max_iterations=max(f["iterations"] for f in fits),
            boundary_fits=sum(f["method"] != "newton" for f in fits),
            min_lambda=min(f["min_lambda"] for f in fits),
            min_background=min(f["min_background"] for f in fits),
            n_profiles=len(fits), n_cls_evaluations=len(trace), monotonicity_error=monotone,
            nll_null=null["nll"], nll_free=free["nll"],
            n_bins=len(n), nuisance_rank=self.rank, zero_bins=int(np.count_nonzero(n == 0)),
            ok=True, status="converged", calibration="fixed-mass asymptotic reference")
        if details:
            row.update(free=free, null=null, trace=trace)
        return row


def upper_limit(n, b, L, S, alpha=.1, *, details=False):
    """Convenience wrapper; reuse OneSignalProfile when b,L,S are unchanged."""
    return OneSignalProfile(b, L, S).limit(n, alpha=alpha, details=details)


def validate(output_path=None):
    """Focused deterministic validation; no pseudoexperiments are generated."""
    from pathlib import Path
    import hashlib
    import importlib.util
    import json
    import time
    from scipy.stats import norm

    checks = []
    analytic_rows = []
    # Independent closed solution for a single Gaussian-constrained bin.
    def closed_nll(n, b, variance, signal):
        c = b + signal - variance
        lam = .5*(c + math.sqrt(c*c + 4*n*variance)) if variance else b + signal
        if variance and c < 0 and n > 0:
            lam = 2*n*variance/(math.sqrt(c*c + 4*n*variance)-c)
        pois = lam - n + n*math.log(n/lam) if n > 0 else lam
        return pois + ((lam-b-signal)**2/(2*variance) if variance else 0.)
    def independent_cls(q, qa):
        sq = math.sqrt(qa)
        zsb, zb = (math.sqrt(q), math.sqrt(q)-sq) if q <= qa else ((q+qa)/(2*sq), (q-qa)/(2*sq))
        return math.exp(norm.logsf(zsb)-norm.logsf(zb))
    for n in (0., 1., 60., 100., 200.):
        for sd in (0., 20.):
            b = 100.
            model = OneSignalProfile([b], [[sd]] if sd else np.empty((1,0)), [1.])
            result = model.limit([n])
            denominator = 0. if n >= b else closed_nll(n, b, sd*sd, 0.)
            def fn(a):
                q = max(0., 2*(closed_nll(n,b,sd*sd,a)-denominator))
                qa = 2*closed_nll(b,b,sd*sd,a)
                return independent_cls(q,qa)-.1
            expected = brentq(fn, max(n-b,0.)+1e-5, max(n-b,0.)+500., xtol=1e-10)
            rel = abs(result["A90"]/expected-1)
            aerr = abs(result["Ahat"] - (n-b))
            if rel > 3e-7 or aerr > 2e-5:
                raise AssertionError(f"Closed one-bin profile mismatch: {n}, {sd}, {rel}, {aerr}")
            analytic_rows.append(dict(n=n, b=b, sd=sd, A90=result["A90"],
                independent_A90=expected, relative_error=rel,
                boundary_fits=result["boundary_fits"], cls_branch=result["cls_branch"]))
    checks.append("10 independent closed one-bin limits, both CLs branches, including all-zero counts")

    # Formula branch boundary and a large deficit for log-tail stability.
    for qa in (.01, 1., 4., 100.):
        left = bounded_tildeq_tails(qa*(1-1e-8),qa)["cls"]
        right = bounded_tildeq_tails(qa*(1+1e-8),qa)["cls"]
        if not abs(left-right) < 1e-7:
            raise AssertionError("Discontinuous tilde-q branches")
    if not 0 < bounded_tildeq_tails(500.,4.)["cls"] < 1:
        raise AssertionError("Log-tail instability")
    checks.append("Cowan piecewise-tail continuity and large-deficit log-tail stability")

    b = np.array([40., 70., 50., 90.])
    L = np.array([[2.,0.], [1.,3.], [.5,2.], [-.3,1.]])
    S = np.array([.1,.7,.2,.03])
    n = np.array([38.,79.,53.,86.])
    model = OneSignalProfile(b,L,S)
    z = np.array([.08,-.04,.07]); h = 1e-4
    f,g,H,_ = model._objective(z,n,model.Jfree,b,model.penfree)
    eye = np.eye(len(z))
    ng = np.array([(model._objective(z+h*v,n,model.Jfree,b,model.penfree,False)[0]
                   -model._objective(z-h*v,n,model.Jfree,b,model.penfree,False)[0])/(2*h) for v in eye])
    nH = np.column_stack([(model._objective(z+h*v,n,model.Jfree,b,model.penfree,False)[1]
                         -model._objective(z-h*v,n,model.Jfree,b,model.penfree,False)[1])/(2*h) for v in eye])
    derivative = dict(gradient_max_abs=float(np.max(abs(ng-g))), hessian_max_abs=float(np.max(abs(nH-H))))
    if max(derivative.values()) > 2e-7:
        raise AssertionError(f"Derivative failure {derivative}")
    checks.append("Independent finite-difference gradient and Hessian")
    nominal = model.limit(n)
    unit_scaled = OneSignalProfile(b,L,S*1e8).limit(n)
    if abs(nominal["A90"]/(unit_scaled["A90"]*1e8)-1) > 2e-8:
        raise AssertionError("Physical-unit scaling failure")
    u = np.array([[.6,-.8],[.8,.6]])
    rotated = OneSignalProfile(b,L@u,S).limit(n)
    if abs(nominal["A90"]/rotated["A90"]-1) > 2e-8:
        raise AssertionError("Covariance-factor rotation failure")
    checks.append("Physical amplitude-unit rescaling and covariance-factor rotation invariance")
    empty_rows=[]
    for counts in (np.array([0.,79.,0.,86.]),np.zeros(4)):
        r=model.limit(counts)
        empty_rows.append({k:r[k] for k in ("A90","Ahat","zero_bins","boundary_fits","max_score","min_lambda")})
    checks.append("Correlated four-bin profiles with mixed empty bins and an all-empty spectrum")

    # Parent study regression is optional so the solver remains portable.
    parent = Path(__file__).resolve().parents[2]/"v5p2p0_two_peak_structure_20260910/scripts/core.py"
    benchmarks=[]
    if parent.exists():
        spec=importlib.util.spec_from_file_location("v520_solver_validation_core",parent)
        core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
        for year,m1,m2 in (("2015",65,74),("2016",81,92),("2021",90,117)):
            part=core.context(year,m1,m2)
            before=core.Model([part]); old=before.fit([0]); oldnull=before.fit()
            profile=OneSignalProfile(part["b"],part["L"],part["S"][:,0])
            start=time.perf_counter(); row=profile.limit(part["n"],details=True); elapsed=time.perf_counter()-start
            errors=dict(free_nll=abs(row["nll_free"]-old["nll"]),null_nll=abs(row["nll_null"]-oldnull["nll"]),
                        Ahat_relative=abs(row["Ahat"]-old["amp"][0])/max(1.,abs(row["Ahat"])))
            if max(errors.values()) > 2e-6:
                raise AssertionError(f"v5.2.0 regression failure {year}: {errors}")
            benchmarks.append(dict(year=year,bins=len(part["n"]),rank=profile.rank,
                seconds=elapsed,A90=row["A90"],max_score=row["max_score"],errors=errors))
        checks.append("Three native dataset free/null profile regressions against frozen v5.2.0 core")
    # Validation requires bad inputs to fail explicitly.
    invalid_rejected=0
    for counts in ([-1.,2.,3.,4.],[math.nan,2.,3.,4.],[1.,2.]):
        try:model.limit(counts)
        except ValueError:invalid_rejected+=1
    if invalid_rejected != 3:raise AssertionError("Invalid-count validation failure")
    checks.append("Negative, nonfinite, and wrong-shape spectra rejected")
    payload=dict(ok=True, checks=checks, n_checks=len(checks),
        source_url="https://arxiv.org/pdf/1007.1727", source_equations=[16,65,66,67],
        solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        random_draws=0, analytic_cases=analytic_rows, derivative=derivative,
        empty_bin_cases=empty_rows, native_regression_benchmarks=benchmarks,
        boundary="Numerical validation only; no coverage or global-significance calibration")
    if output_path is not None:
        Path(output_path).write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    return payload


if __name__ == "__main__":
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-output",required=True)
    args=parser.parse_args()
    report=validate(args.validate_output)
    print(json.dumps({k:report[k] for k in ("ok","n_checks","random_draws")}))
