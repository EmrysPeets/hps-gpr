"""Portable legacy-optimizer diagnostic; six cells, no scans or data changes.

The five marked functions are copied verbatim from the attested v4.9.7 runtime
inherited by the v5.0.5 published curves. The v5.9 study's context and Newton
solver are compared on exactly the same b, C, and signal. Legacy functions are
used solely to explain the historical numerical discrepancy, not for new limits.
"""
from common import *
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from scipy.stats import norm
import scipy
import platform
LEGACY_SOURCE = {'path': '/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5p0p2-20260909/study_results/v4p9p7_2016_support_combined_100toy_20260902/runtime_combined/hps_gpr/statistics.py', 'sha256': 'b8cbd484056925d64bed4d9a4ad3294fbac07d51079e5cb9ed565150b73c1ff2', 'functions': [{'name': '_chol_with_jitter', 'start_line': 121, 'end_line': 153, 'sha256': '33a6e2308f7133922572a699ddbc2b91959fc03a502cb3d1cf5c8261a87178f3'}, {'name': '_profile_theta_given_A', 'start_line': 165, 'end_line': 211, 'sha256': '3323c6115c9e43d82ddf756ae9cfb2f11c5d1612119660efa4d1e790552322cc'}, {'name': 'fit_A_profiled_gaussian_details', 'start_line': 282, 'end_line': 387, 'sha256': '80a6c0fbd8239f96652b9e18bc5206572adacfc96f1e563d59272fe2b330a022'}, {'name': 'fit_A_profiled_gaussian', 'start_line': 390, 'end_line': 409, 'sha256': '4c652bdd25e73c33d46c203b25711084d528aace1fac98b9d15a18bddc5aea83'}, {'name': 'p0_profiled_gaussian_LRT', 'start_line': 524, 'end_line': 570, 'sha256': 'a701422d262cb6bc2e5e392adf6004e5fd50d34c2e803a772d3c21148fea1b63'}]}

# BEGIN verbatim attested legacy functions

def _chol_with_jitter(C: np.ndarray, jitter0: float = 1e-10, max_tries: int = 8) -> np.ndarray:
    """Return a numerically safe square-root factor of a (nearly) covariance matrix.

    Attempts Cholesky with progressively larger diagonal jitter, then falls back to
    an eigenvalue-clipped symmetric square root. The returned L satisfies L L^T ≈ C.
    """
    C = np.asarray(C, float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Covariance must be square; got shape={C.shape}")
    if not np.all(np.isfinite(C)):
        raise ValueError("Covariance contains non-finite entries (NaN/inf).")
    C = 0.5 * (C + C.T)
    B = C.shape[0]
    if B == 0:
        return np.zeros((0, 0), float)
    diag = np.diag(C)
    scale = float(np.max(np.abs(diag))) if diag.size else 1.0
    scale = max(scale, 1.0)
    I = np.eye(B)
    jitter = float(jitter0) * scale
    for _ in range(int(max_tries)):
        try:
            return np.linalg.cholesky(C + jitter * I)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    try:
        w, V = np.linalg.eigh(C)
        floor = max(1e-12 * scale, float(jitter0) * scale)
        w = np.clip(w, floor, None)
        return V @ np.diag(np.sqrt(w)) @ V.T
    except np.linalg.LinAlgError:
        d = np.clip(np.diag(C), 1e-12 * scale, None)
        return np.diag(np.sqrt(d))


def _get_chol(C):
    """Legacy routing helper; the attested code selects this same factor."""
    return _chol_with_jitter(C)


def _profile_theta_given_A(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    A_fixed: float,
    th0: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Profile over θ with A fixed (used internally by p0_profiled_gaussian_LRT).

    Model: n_i ~ Poisson(λ_i),  λ = b_mean + L θ + A_fixed * w,  θ ~ N(0, I).
    """
    n = np.clip(np.asarray(n_obs, float), 0.0, None)
    b = np.clip(np.asarray(b_mean, float), 1e-12, None)
    C = np.asarray(b_cov, float)
    w = np.asarray(template, float)
    B = b.size
    if C.shape != (B, B):
        raise ValueError(f"Cov shape mismatch: {C.shape} vs {(B, B)}")
    if w.shape != (B,):
        raise ValueError(f"Template shape mismatch: {w.shape} vs {(B,)}")
    L = _get_chol(C)
    eps = 1e-9 * max(1.0, float(np.median(b)))
    th0 = np.zeros(B, float) if th0 is None else np.asarray(th0, float).reshape(B)
    A_fixed = float(A_fixed)

    def nll_and_grad(th):
        lam = b + L @ th + A_fixed * w
        lam_eff = np.maximum(lam, eps)
        ll = np.sum(n * np.log(lam_eff) - lam_eff) - 0.5 * float(np.dot(th, th))
        r = (n / lam_eff) - 1.0
        return -float(ll), -(L.T @ r) + th

    res = minimize(
        fun=lambda th: nll_and_grad(th)[0],
        x0=th0,
        jac=lambda th: nll_and_grad(th)[1],
        method="L-BFGS-B",
        options=dict(maxiter=400, ftol=1e-10),
    )
    return dict(
        theta_hat=np.asarray(res.x, float),
        nll=float(res.fun) if np.isfinite(getattr(res, "fun", np.nan)) else float("nan"),
        success=bool(res.success),
        status=int(getattr(res, "status", -1)),
        message=str(getattr(res, "message", "")),
    )


def fit_A_profiled_gaussian_details(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    *,
    allow_negative: bool = True,
    lam_floor: float = 1e-12,
) -> Dict[str, object]:
    """Profile-likelihood fit for A with Gaussian nuisance parameters for the background.

    Returns dict with: A_hat, sigma_A, theta_hat, delta_b_hat, b_fit, lambda_hat, success, nll.
    """
    n = np.clip(np.asarray(n_obs, float), 0.0, None)
    b = np.clip(np.asarray(b_mean, float), 1e-12, None)
    C = np.asarray(b_cov, float)
    w = np.asarray(template, float)
    B = b.size
    if B == 0:
        return dict(A_hat=float("nan"), sigma_A=float("nan"), theta_hat=np.array([]),
                    delta_b_hat=np.array([]), b_fit=np.array([]), lambda_hat=np.array([]),
                    success=False, nll=float("nan"))
    L = _chol_with_jitter(C)
    eps = max(float(lam_floor), 1e-9 * max(1.0, float(np.median(b))))

    # GLS initialisation for A
    def _gls_start():
        V = C + np.diag(np.clip(b, 1.0, None)) + (eps * 100.0) * np.eye(B)
        try:
            Vinv_w = np.linalg.solve(V, w)
            denom = float(np.dot(w, Vinv_w))
            if not np.isfinite(denom) or denom <= 0:
                raise ValueError
            A0 = float(np.dot(w, np.linalg.solve(V, n - b))) / denom
            sig0 = float(np.sqrt(1.0 / denom))
        except Exception:
            A0 = float(np.sum(n - b))
            sig0 = float(np.sqrt(np.sum(np.clip(b, 1.0, None))))
        return float(A0), float(sig0)

    A0, sig0 = _gls_start()
    if not allow_negative:
        A0 = max(0.0, A0)

    def nll_and_grad(x: np.ndarray):
        A = float(x[0])
        th = np.asarray(x[1:], float)
        lam = b + L @ th + A * w
        lam_eff = np.maximum(lam, eps)
        ll = np.sum(n * np.log(lam_eff) - lam_eff) - 0.5 * float(np.dot(th, th))
        r = (n / lam_eff) - 1.0
        gA = -float(np.dot(w, r))
        gth = -(L.T @ r - th)
        bad = lam < eps
        if np.any(bad):
            delta = eps - lam[bad]
            k = 1e6
            penalty = float(k * np.dot(delta, delta))
            dpen = -2.0 * k * delta
            gA += float(np.dot(w[bad], dpen))
            gth += L[bad].T @ dpen
            return -float(ll) + penalty, np.concatenate(([gA], np.asarray(gth, float)))
        return -float(ll), np.concatenate(([gA], np.asarray(gth, float)))

    bounds = None if allow_negative else [(0.0, None)] + [(None, None)] * B
    res = minimize(
        fun=lambda x: nll_and_grad(x)[0],
        x0=np.concatenate(([A0], np.zeros(B))),
        jac=lambda x: nll_and_grad(x)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=500, ftol=1e-10),
    )

    Ahat = float(res.x[0])
    thhat = np.asarray(res.x[1:], float)
    delta_b = (L @ thhat).astype(float)
    lamhat = (b + delta_b + Ahat * w).astype(float)
    lam_eff = np.maximum(lamhat, eps)

    # Profile information (Schur complement) for σ_A
    W = n / (lam_eff ** 2)
    I_AA = float(np.sum(W * (w ** 2)))
    I_Ath = (w * W) @ L
    I_thth = (L.T * W) @ L + np.eye(B)
    sigA = float("nan")
    try:
        sol = np.linalg.solve(I_thth, I_Ath.reshape(-1, 1)).reshape(-1)
        I_prof = I_AA - float(I_Ath @ sol)
        sigA = float(np.sqrt(1.0 / max(I_prof, 1e-18)))
    except Exception:
        sigA = float(sig0)

    if not allow_negative and Ahat < 0:
        Ahat = 0.0

    return dict(
        A_hat=float(Ahat),
        sigma_A=float(sigA),
        theta_hat=thhat,
        delta_b_hat=delta_b,
        b_fit=(b + delta_b).astype(float),
        lambda_hat=lamhat,
        success=bool(getattr(res, "success", False)),
        nll=float(getattr(res, "fun", float("nan"))),
    )


def fit_A_profiled_gaussian(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
    allow_negative: bool = True,
) -> Dict[str, float]:
    """Fit signal amplitude with profiled Gaussian prior on background.

    Thin wrapper around fit_A_profiled_gaussian_details returning only the essentials.
    """
    d = fit_A_profiled_gaussian_details(
        n_obs, b_mean, b_cov, template, allow_negative=allow_negative
    )
    return dict(
        A_hat=float(d["A_hat"]),
        sigma_A=float(d["sigma_A"]),
        success=bool(d["success"]),
        nll=float(d.get("nll", np.nan)),
    )


def p0_profiled_gaussian_LRT(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: np.ndarray,
    template: np.ndarray,
) -> Tuple[float, float, float, Dict[str, object]]:
    """Asymptotic profiled LRT p0 for A >= 0 vs A = 0.

    Returns (p0, Z, q0, info) where q0 = -2 ln Λ and Z = sqrt(q0).

    Uses the same Poisson + Gaussian-prior additive model as fit_A_profiled_gaussian.
    The null is maximized over θ with A=0; the alternative over (A>=0, θ).
    """
    alt = fit_A_profiled_gaussian(
        n_obs=np.asarray(n_obs, int),
        b_mean=b_mean,
        b_cov=b_cov,
        template=template,
        allow_negative=False,
    )
    nll_alt = float(alt.get("nll", float("nan")))
    A_hat = float(alt.get("A_hat", float("nan")))
    sigma_A = float(alt.get("sigma_A", float("nan")))
    ok_alt = bool(alt.get("success", False))

    null = _profile_theta_given_A(
        n_obs=n_obs, b_mean=b_mean, b_cov=b_cov,
        template=template, A_fixed=0.0,
    )
    nll0 = float(null.get("nll", float("nan")))
    ok_null = bool(null.get("success", False))

    q0 = 0.0
    if np.isfinite(nll0) and np.isfinite(nll_alt):
        q0 = max(0.0, 2.0 * (nll0 - nll_alt))

    Z = float(np.sqrt(q0)) if q0 > 0 else 0.0
    p0 = min(max(float(norm.sf(Z)), 0.0), 1.0)

    info = dict(
        q0=float(q0), Z=float(Z), p0=float(p0),
        A_hat=float(A_hat), sigma_A=float(sigma_A),
        nll_alt=float(nll_alt), nll0=float(nll0),
        ok_alt=bool(ok_alt), ok_null=bool(ok_null),
        ok=bool(ok_alt and ok_null),
    )
    return float(p0), float(Z), float(q0), info

# END verbatim attested legacy functions


def audit():
    published = pd.read_csv(B/'inputs/nominal_v505_curves.csv',dtype={'dataset_set':str})
    rows=[]
    for year,mass in [('2015',51),('2016',90),('2021',78),('2016',83),('2016',59),('2021',86)]:
        part=context(year,[mass],padding=2.25,anchor=mass)
        n,b,S=part['n'],part['b'],part['S'][:,0]
        # Exact same preconditioned covariance supplied to both solvers.
        C=part['C']+part['diagnostic']['load']*max(float(np.diag(part['C']).max()),1.)*np.eye(len(b))
        L=_chol_with_jitter(C)
        exact=OneSignalProfile(b,L,S).limit(n,details=True)
        reduced=OneSignalProfile(b,part['L'],S).limit(n)
        w=S/S.sum()
        legacy_null=_profile_theta_given_A(n,b,C,w,0.)
        legacy_alt=fit_A_profiled_gaussian_details(n,b,C,w,allow_negative=False)
        legacy_p,legacy_z,legacy_q,legacy_info=p0_profiled_gaussian_LRT(n,b,C,w)
        theta=legacy_null['theta_hat'];lam=b+L@theta
        grad=L.T@(1-n/lam)+theta
        positive=n>0
        constant=float(np.sum(n[positive]-n[positive]*np.log(n[positive])))
        null_deviance=poisson_deviance_half(n,lam)+.5*float(theta@theta)
        exact_alt=exact['free'] if exact['Ahat']>=0 else exact['null']
        q=published[(published.dataset_set==year)&(published.mass_MeV==mass)].iloc[0]
        # At the archived observed limit, profile both observed and B-Asimov
        # spectra with the exact solver; CLs drift quantifies why a reroot differs.
        old_A=float(q.eps2_90)*1e8
        mod=OneSignalProfile(b,L,S)
        f=mod.fit(n,old_A);fa=mod.fit(b,old_A)
        tails=__import__('limit_solver').bounded_tildeq_tails(
            max(0.,2*(f['nll']-exact_alt['nll'])),2*fa['nll'])
        legacy_fixed=_profile_theta_given_A(n,b,C,w,old_A*S.sum())
        lt=legacy_fixed['theta_hat'];ll=b+L@lt+old_A*S
        legacy_fixed_deviance=poisson_deviance_half(n,ll)+.5*float(lt@lt)
        legacy_fixed_gradient=L.T@(1-n/ll)+lt
        row=dict(scope=year,mass_MeV=mass,
            legacy_fixed_at_published_limit_deviance=legacy_fixed_deviance,
            exact_fixed_at_published_limit_deviance=f['nll'],
            fixed_nll_improvement=legacy_fixed_deviance-f['nll'],
            legacy_fixed_max_abs_gradient=float(abs(legacy_fixed_gradient).max()),
            published_Z=float(q.Z_local_asymptotic),legacy_replayed_Z=legacy_z,
            exact_Z=exact['Z0'],legacy_minus_exact_Z=legacy_z-exact['Z0'],
            published_epsilon2_90=float(q.eps2_90),exact_epsilon2_90=exact['A90']*1e-8,
            exact_limit_relative_to_published=exact['A90']*1e-8/float(q.eps2_90)-1.,
            exact_cls_at_published_limit=tails['cls'],
            legacy_null_raw_nll=legacy_null['nll'],legacy_null_stable_deviance=null_deviance,
            exact_null_deviance=exact['nll_null'],null_nll_improvement=null_deviance-exact['nll_null'],
            legacy_null_max_abs_gradient=float(abs(grad).max()),
            legacy_null_message=legacy_null['message'],legacy_null_success=legacy_null['success'],
            legacy_bounded_alt_deviance=poisson_deviance_half(n,legacy_alt['lambda_hat'])+.5*float(legacy_alt['theta_hat']@legacy_alt['theta_hat']),
            exact_bounded_alt_deviance=exact_alt['nll'],
            exact_max_score=exact['max_score'],raw_objective_constant=constant,
            full_vs_reduced_limit_relative=exact['A90']/reduced['A90']-1.,
            full_vs_reduced_Z=exact['Z0']-reduced['Z0'],
            minimum_legacy_lambda=float(min(lam.min(),legacy_alt['lambda_hat'].min())),
            maximum_legacy_Z_replay_error=abs(legacy_z-float(q.Z_local_asymptotic)))
        rows.append(row)
    check=next(r for r in rows if r['scope']=='2016' and r['mass_MeV']==83)
    assert check['legacy_null_max_abs_gradient']>.1
    assert check['null_nll_improvement']>.02
    assert abs(check['legacy_minus_exact_Z']-.12366045)<2e-5
    assert max(abs(r['full_vs_reduced_limit_relative']) for r in rows)<2e-7
    assert max(abs(r['full_vs_reduced_Z']) for r in rows)<2e-6
    assert max(r['maximum_legacy_Z_replay_error'] for r in rows if r['published_Z']>.001)<2e-5
    # At zero significance, subtraction of ~1e8 absolute NLLs can leave one
    # floating-point ulp (Z=2^-12); record this instead of calling it an excess.
    assert max(r['maximum_legacy_Z_replay_error'] for r in rows if r['published_Z']<=.001)<3e-4
    return dict(ok=True,source=LEGACY_SOURCE,script_sha256=sha(Path(__file__)),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        cells=rows,
        conclusion='The historical drift is reproduced by the inherited large-absolute-NLL L-BFGS stopping rule. The 2016 83 MeV null reports success while its gradient is non-negligible; the same-input Newton fit reaches lower NLL. Full legacy Cholesky versus the retained covariance modes is negligible in these controls.',
        scope='Six deterministic fixed-mass cells; no new coverage, statistical calibration, or scans. The same model and inputs are used in each same-cell comparison. The source file is provenance only and is not required at runtime.')

if __name__=='__main__':
    payload=audit()
    write(B/'qa/legacy_optimizer_audit.json',payload)
    print(json.dumps(dict(ok=payload['ok'],cells=len(payload['cells']))))
