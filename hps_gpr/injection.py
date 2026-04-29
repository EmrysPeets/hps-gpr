"""Signal injection and extraction closure tests."""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    import joblib
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:
    import contextlib
    _threadpool_limits = contextlib.nullcontext  # type: ignore[assignment]

from .io import estimate_background_for_dataset
from .template import build_template, build_window_template_from_full
from .conversion import A_from_epsilon2
from .statistics import (
    fit_A_profiled_gaussian,
    fit_A_profiled_gaussian_details,
    qmu_tilde_profiled_gaussian,
    draw_bkg_mvn_nonneg,
)
from .gpr import (
    make_kernel_for_dataset,
    fit_gpr,
    predict_counts_from_log_gpr,
    compute_kernel_ls_bounds,
    _extract_rbf_bounds_and_scale,
)
from .plotting import ensure_dir

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig
    from .io import BlindPrediction


# ---------------------------------------------------------------------------
# Low-level injection helpers
# ---------------------------------------------------------------------------

def _inject_counts_weighted(
    weights: np.ndarray, total: int, rng: np.random.Generator
) -> np.ndarray:
    """Multinomial draw across bins (weights must sum to ~1)."""
    total = int(total)
    if total <= 0:
        return np.zeros_like(weights, dtype=int)
    w = np.asarray(weights, float)
    wsum = float(np.sum(w))
    if wsum <= 0:
        return np.zeros_like(weights, dtype=int)
    return rng.multinomial(total, w / wsum)


def _inject_counts_from_template(
    template: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    mode: str = "multinomial",
) -> Tuple[np.ndarray, int, float]:
    """Inject a signal distributed like `template` at amplitude `strength`.

    Returns:
        (sig_counts, Nsig_realized, f_sumw)
    where f_sumw = Σw (the sum of template weights, useful for leakage diagnostics).
    """
    w = np.asarray(template, float)
    f = float(np.sum(w))
    if not np.isfinite(f) or f <= 0 or not np.isfinite(strength) or strength <= 0:
        return np.zeros_like(w, dtype=int), 0, f

    if mode == "poisson":
        sig = rng.poisson(float(strength) * w)
        return sig.astype(int), int(np.sum(sig)), f

    # multinomial: interpret strength as total yield parameter
    Ntot = int(np.round(float(strength)))
    if Ntot <= 0:
        return np.zeros_like(w, dtype=int), 0, f

    if f < 1.0:
        Nwin = int(rng.binomial(Ntot, f))
        if Nwin <= 0:
            return np.zeros_like(w, dtype=int), 0, f
        sig = _inject_counts_weighted(w / f, Nwin, rng)
    else:
        sig = _inject_counts_weighted(w / f, Ntot, rng)

    return sig.astype(int), int(np.sum(sig)), f


def inject_counts(
    edges: np.ndarray,
    mass: float,
    sigma_val: float,
    strength: int,
    rng: np.random.Generator,
    mode: str = "multinomial",
) -> np.ndarray:
    """Inject signal counts into bins (legacy interface, kept for back-compat)."""
    if strength <= 0:
        return np.zeros(len(edges) - 1, dtype=int)
    w = build_template(edges, mass, sigma_val)
    if mode == "multinomial":
        return rng.multinomial(int(strength), w).astype(int)
    return rng.poisson(float(strength) * w).astype(int)


def _prediction_blind_mask(pred: Any) -> np.ndarray:
    """Return the blind-window mask, deriving it when older mocks omit the field."""
    blind_mask = getattr(pred, "blind_mask", None)
    if blind_mask is not None:
        return np.asarray(blind_mask, bool).reshape(-1)

    blind = getattr(pred, "blind", None)
    if blind is None:
        raise AttributeError("Prediction is missing both 'blind_mask' and 'blind'.")

    x_full = getattr(pred, "x_full", None)
    if x_full is None:
        edges_full = np.asarray(getattr(pred, "edges_full"), float).reshape(-1)
        if edges_full.size < 2:
            raise AttributeError(
                "Prediction cannot derive 'blind_mask' without 'x_full' or usable 'edges_full'."
            )
        x_full = 0.5 * (edges_full[:-1] + edges_full[1:])

    x_full = np.asarray(x_full, float).reshape(-1)
    return np.asarray(
        (x_full >= float(blind[0])) & (x_full <= float(blind[1])),
        bool,
    )


def _prediction_integral_density(pred: Any) -> float:
    """Return a finite signal-density scale, with a simple fallback for test doubles."""
    integral_density = getattr(pred, "integral_density", None)
    if integral_density is not None:
        val = float(integral_density)
        if np.isfinite(val) and val > 0:
            return val

    mu_full = getattr(pred, "mu_full", None)
    if mu_full is None:
        mu_full = getattr(pred, "mu", None)
    arr = np.asarray(mu_full, float).reshape(-1)

    edges_full = getattr(pred, "edges_full", None)
    if edges_full is not None:
        edges = np.asarray(edges_full, float).reshape(-1)
        if edges.size >= 2:
            width = float(edges[-1] - edges[0])
            if np.isfinite(width) and width > 0:
                return float(np.sum(np.clip(arr, 0.0, None)) / width)

    return float(np.sum(np.clip(arr, 0.0, None)))


def _dataset_kernel_factor(config: "Config", ds_key: str, base_attr: str, by_attr: str) -> float:
    """Resolve a scalar kernel-factor knob after per-dataset overrides."""
    by_ds = dict(getattr(config, by_attr, {}) or {})
    return float(by_ds.get(str(ds_key), getattr(config, base_attr)))


def _kernel_static_diagnostics(ds: "DatasetConfig", config: "Config", mass: float) -> Dict[str, Any]:
    """Return configuration-level kernel diagnostics for one dataset/mass point."""
    out: Dict[str, Any] = dict(
        kernel_ls_policy=str(getattr(config, "kernel_ls_policy", "")),
        kernel_ls_res_lower_factor=_dataset_kernel_factor(
            config,
            str(ds.key),
            "kernel_ls_res_lower_factor",
            "kernel_ls_res_lower_factor_by_dataset",
        ),
        kernel_ls_res_upper_factor=_dataset_kernel_factor(
            config,
            str(ds.key),
            "kernel_ls_res_upper_factor",
            "kernel_ls_res_upper_factor_by_dataset",
        ),
        ls_lo=float("nan"),
        ls_hi=float("nan"),
        ls_init=float("nan"),
    )
    try:
        info = compute_kernel_ls_bounds(ds, config, mass=float(mass))
        out.update(
            ls_lo=float(info.get("ls_lo", float("nan"))),
            ls_hi=float(info.get("ls_hi", float("nan"))),
            ls_init=float(info.get("ls_init", float("nan"))),
        )
    except Exception:
        pass
    return out


def _gpr_fit_diagnostics(gpr: Any) -> Dict[str, float]:
    """Extract optimized kernel diagnostics from a fitted sklearn GP."""
    out = dict(ls_opt=float("nan"), const_opt=float("nan"))
    try:
        kopt = getattr(gpr, "kernel_", None)
        if kopt is not None and hasattr(kopt, "k1"):
            out["const_opt"] = float(getattr(kopt.k1, "constant_value", float("nan")))
        _, _, ls_opt = _extract_rbf_bounds_and_scale(kopt if kopt is not None else getattr(gpr, "kernel", None))
        out["ls_opt"] = float(ls_opt)
    except Exception:
        pass
    return out


def _handle_refit_failure(config: "Config", label: str, exc: Exception) -> str:
    """Return a compact error string, optionally escalating failed toy refits."""
    msg = f"{type(exc).__name__}: {exc}"
    print(f"[inj][refit] WARNING {label}: {msg}")
    if bool(getattr(config, "inj_refit_fail_on_error", False)) or bool(getattr(config, "fail_fast", False)):
        raise RuntimeError(f"[inj][refit] failed for {label}: {msg}") from exc
    return msg[:240]


def _dataset_source_metadata(ds: "DatasetConfig") -> Dict[str, Any]:
    """Return optional pseudo-data source metadata attached to a dataset clone."""
    out: Dict[str, Any] = {}
    for name in [
        "source_model",
        "source_label",
        "source_root",
        "container",
        "function_tag",
        "toy_hist",
        "toy_index",
    ]:
        if hasattr(ds, name):
            out[name] = getattr(ds, name)
    return out


def _qmu_tilde_fields(
    config: "Config",
    obs: np.ndarray,
    mu_fit: np.ndarray,
    cov_fit: np.ndarray,
    tmpl_win: np.ndarray,
    A_test: float,
) -> Dict[str, Any]:
    """Compute optional exact profiled ``tilde q_mu`` diagnostics for toy rows."""
    if not bool(getattr(config, "inj_write_qmu", False)):
        return {}

    try:
        qmu, info = qmu_tilde_profiled_gaussian(
            obs,
            np.asarray(mu_fit, float),
            np.asarray(cov_fit, float),
            np.asarray(tmpl_win, float),
            float(A_test),
        )
        qmu = float(qmu)
        sqrt_qmu = float(np.sqrt(max(qmu, 0.0))) if np.isfinite(qmu) else float("nan")
        return dict(
            qmu_tilde=float(qmu),
            tmu_tilde=float(qmu),
            sqrt_qmu_tilde=float(sqrt_qmu),
            sqrt_tmu_tilde=float(sqrt_qmu),
            qmu_A_test=float(info.get("A_test", A_test)),
            qmu_branch=str(info.get("branch", "")),
            qmu_ok=bool(info.get("ok", False)),
            qmu_nll_fixed=float(info.get("nll_fixed", float("nan"))),
            qmu_nll_unbounded=float(info.get("nll_unbounded", float("nan"))),
            qmu_nll_null=float(info.get("nll_null", float("nan"))),
        )
    except Exception as exc:
        return dict(
            qmu_tilde=float("nan"),
            tmu_tilde=float("nan"),
            sqrt_qmu_tilde=float("nan"),
            sqrt_tmu_tilde=float("nan"),
            qmu_A_test=float(A_test),
            qmu_branch="error",
            qmu_ok=False,
            qmu_nll_fixed=float("nan"),
            qmu_nll_unbounded=float("nan"),
            qmu_nll_null=float("nan"),
            qmu_error=f"{type(exc).__name__}: {exc}"[:240],
        )


# ---------------------------------------------------------------------------
# Reference sigma_A for strength scaling
# ---------------------------------------------------------------------------

def _sigmaA_reference(
    pred: "BlindPrediction",
    mass: float,
    *,
    source: str = "asimov",
    rng: Optional[np.random.Generator] = None,
    config: Optional["Config"] = None,
) -> float:
    """Estimate σ_A(m) under B-only for 'sigma-level' injection scaling."""
    blind_mask = _prediction_blind_mask(pred)
    tmpl, _ = build_window_template_from_full(
        pred.edges_full, blind_mask, mass, pred.sigma_val, config=config
    )
    b = np.asarray(pred.mu, float)
    if source == "poisson":
        if rng is None:
            rng = np.random.default_rng(12345)
        n_ref = rng.poisson(b)
    else:
        n_ref = b  # Asimov
    d = fit_A_profiled_gaussian_details(n_ref, b, pred.cov, tmpl, allow_negative=True)
    return float(d.get("sigma_A", np.nan))


# ---------------------------------------------------------------------------
# Main injection/extraction study
# ---------------------------------------------------------------------------

def run_injection_extraction_toys(
    ds: "DatasetConfig",
    config: "Config",
    *,
    masses: List[float],
    strengths: Optional[List[float]] = None,
    n_toys: int = 200,
    outdir: str = None,
    seed: int = 314159,
    inj_mode: Optional[str] = None,
    strengths_mode: Optional[str] = None,
    sigma_multipliers: Optional[List[float]] = None,
    sigma_source: Optional[str] = None,
    refit_gp_on_toy: Optional[bool] = None,
    refit_restarts: Optional[int] = None,
    refit_optimize: Optional[bool] = None,
    inj_shape_mode: Optional[str] = None,
    train_exclude_nsigma: Optional[float] = None,
    write_toy_csv: Optional[bool] = None,
) -> pd.DataFrame:
    """Run injection+extraction closure tests for one dataset.

    Modes
    -----
    (1) Conditional blind-window toys (fast, default):
        Draw background from the GP posterior MVN, Poisson-sample obs, inject in window only.

    (2) Full procedural toys with GP refit (slow):
        Full-range pseudo-dataset, GP refit on toy sidebands, extract in blind window.
        Controlled by refit_gp_on_toy=True.

    strengths_mode:
        'absolute' — inject `strengths` counts (or use config.inj_strengths)
        'sigmaA'   — inject multiples of σ_A (use sigma_multipliers)
    """
    if outdir is None:
        outdir = os.path.join(config.output_dir, "injection_extraction")
    ensure_dir(outdir)

    if write_toy_csv is None:
        write_toy_csv = bool(getattr(config, "inj_write_toy_csv", True))

    rng = np.random.default_rng(int(seed))

    # Resolve mode toggles from config or arguments
    strengths_mode = (strengths_mode or getattr(config, "inj_strength_mode", "absolute")).lower().strip()
    sigma_source = (sigma_source or getattr(config, "inj_sigma_a_source", "asimov")).lower().strip()
    inj_mode = (inj_mode or config.inj_mode).lower().strip()

    if refit_gp_on_toy is None:
        refit_gp_on_toy = bool(getattr(config, "inj_refit_gp_on_toy", False))
    if refit_restarts is None:
        refit_restarts = int(getattr(config, "inj_refit_gp_restarts", 0))
    if refit_optimize is None:
        refit_optimize = bool(getattr(config, "inj_refit_gp_optimize", True))
    inj_shape_mode = (inj_shape_mode or getattr(config, "inj_shape_mode", "full")).lower().strip()
    if inj_shape_mode not in ("full", "window"):
        inj_shape_mode = "full"
    if train_exclude_nsigma is None:
        train_exclude_nsigma = getattr(config, "inj_train_exclude_nsigma", None)

    mvn_method = str(getattr(config, "mvn_trunc_method", "reject_then_clip"))
    mvn_max_tries = int(getattr(config, "mvn_trunc_max_tries", 80))

    # Strength grid
    if strengths_mode == "sigmaa":
        # In sigmaA mode, explicit `strengths` (e.g. from CLI/SLURM job point)
        # override config defaults so one job maps to one requested sigma level.
        if strengths is not None:
            mults = strengths
        elif sigma_multipliers is not None:
            mults = sigma_multipliers
        else:
            mults = getattr(config, "inj_sigma_multipliers", [0.0, 1.0, 2.0, 3.0])
        strength_tags = [float(x) for x in mults]
    else:
        strength_tags = [float(x) for x in (strengths or config.inj_strengths)]

    train_nsig_default = float(
        getattr(config, "gp_train_exclude_nsigma", None) or config.blind_nsigma
    )

    rows: List[Dict] = []

    for m in masses:
        m = float(m)
        pred = estimate_background_for_dataset(ds, m, config)
        blind_mask = _prediction_blind_mask(pred)
        integral_density = _prediction_integral_density(pred)

        tmpl_win, tmpl_full = build_window_template_from_full(
            pred.edges_full, blind_mask, m, pred.sigma_val, config=config
        )
        f_win = float(np.sum(tmpl_win))

        # Full-range template for leakage diagnostics
        x_full = np.asarray(pred.x_full, float).reshape(-1)
        f_full = float(np.sum(tmpl_full))
        kernel_diag = _kernel_static_diagnostics(ds, config, m)
        initial_ls_opt = float(getattr(pred, "ls_opt", float("nan")))
        initial_const_opt = float(getattr(pred, "const_opt", float("nan")))

        sigmaA_ref = _sigmaA_reference(pred, m, source=sigma_source, rng=rng, config=config)

        if strengths_mode == "sigmaa":
            A_inj_list = [float(t) * float(sigmaA_ref) for t in strength_tags]
            inj_nsigma_list = list(strength_tags)
        else:
            A_inj_list = list(strength_tags)
            inj_nsigma_list = [
                float(A) / float(sigmaA_ref) if np.isfinite(sigmaA_ref) and sigmaA_ref > 0 else np.nan
                for A in A_inj_list
            ]

        blind = tuple(pred.blind)
        msk_blind = (x_full >= blind[0]) & (x_full <= blind[1])
        if int(np.sum(msk_blind)) == 0:
            raise RuntimeError(f"[inj][{ds.key}] m={m:.6g}: blind window has no bins")

        if train_exclude_nsigma is not None:
            tn = float(train_exclude_nsigma)
        else:
            tn = float(getattr(pred, "train_exclude_nsigma", train_nsig_default))
        blind_train = (m - tn * float(pred.sigma_val), m + tn * float(pred.sigma_val))
        msk_train = (x_full < blind_train[0]) | (x_full > blind_train[1])
        n_train_low = int(np.count_nonzero(x_full < blind_train[0]))
        n_train_high = int(np.count_nonzero(x_full > blind_train[1]))
        n_train = int(np.count_nonzero(msk_train))
        n_blind = int(np.count_nonzero(msk_blind))

        f_train = float(np.sum(tmpl_full[msk_train])) if tmpl_full.shape[0] == x_full.shape[0] else float("nan")
        f_train_frac = float(f_train / f_full) if np.isfinite(f_train) and f_full > 0 else float("nan")

        toy_mode = "full_refit" if refit_gp_on_toy else "conditional_gp"

        if refit_gp_on_toy:
            mu_full_arr = np.asarray(pred.mu_full, float).reshape(-1)
            ker = make_kernel_for_dataset(ds, config, mass=m)
            x_win = x_full[msk_blind]
        else:
            b_draws = draw_bkg_mvn_nonneg(
                pred.mu, pred.cov, int(n_toys) * len(strength_tags),
                rng, method=mvn_method, max_tries=mvn_max_tries,
            )
            draw_idx = 0

        for A_inj, inj_nsigma in zip(A_inj_list, inj_nsigma_list):
            A_inj = float(A_inj)

            for i in range(int(n_toys)):
                refit_ok = float("nan")
                refit_fallback_used = False
                refit_error = ""
                refit_diag = dict(refit_ls_opt=float("nan"), refit_const_opt=float("nan"))

                if refit_gp_on_toy:
                    bkg_full = rng.poisson(np.clip(mu_full_arr, 0.0, None)).astype(int)
                    if inj_shape_mode == "window":
                        sig_full = np.zeros_like(bkg_full, dtype=int)
                        s_win, Nsig_win, _ = _inject_counts_from_template(tmpl_win, A_inj, rng, inj_mode)
                        idx_blind = np.where(msk_blind)[0]
                        n = min(len(s_win), len(idx_blind))
                        sig_full[idx_blind[:n]] = s_win[:n]
                        Nsig_full = int(np.sum(sig_full))
                    else:
                        s_full, Nsig_full, _ = _inject_counts_from_template(tmpl_full, A_inj, rng, inj_mode)
                        sig_full = np.asarray(s_full, dtype=int)
                        Nsig_win = int(np.sum(sig_full[msk_blind]))

                    y_toy = (bkg_full + sig_full).astype(int)
                    obs = y_toy[msk_blind].astype(int)
                    Nsig_train = int(np.sum(sig_full[msk_train]))

                    mu_fit = np.asarray(pred.mu, float)
                    cov_fit = np.asarray(pred.cov, float)
                    try:
                        X_tr = x_full[msk_train]
                        y_tr = y_toy[msk_train].astype(float)
                        gpr = fit_gpr(X_tr, y_tr, config, restarts=refit_restarts, kernel=ker, optimize=refit_optimize)
                        mu_fit, cov_fit = predict_counts_from_log_gpr(gpr, x_win, config)
                        diag = _gpr_fit_diagnostics(gpr)
                        refit_diag.update(
                            refit_ls_opt=float(diag["ls_opt"]),
                            refit_const_opt=float(diag["const_opt"]),
                        )
                        refit_ok = 1.0
                    except Exception as exc:
                        refit_ok = 0.0
                        refit_fallback_used = True
                        refit_error = _handle_refit_failure(
                            config,
                            f"{ds.key} m={m:.6g} toy={i} A={A_inj:.6g}",
                            exc,
                        )

                else:
                    b = b_draws[draw_idx % b_draws.shape[0]]
                    draw_idx += 1
                    sig, Nsig_win, _ = _inject_counts_from_template(tmpl_win, A_inj, rng, inj_mode)
                    lam = np.clip(b, 0.0, None) + np.clip(sig.astype(float), 0.0, None)
                    obs = rng.poisson(lam).astype(int)
                    Nsig_train = 0
                    mu_fit, cov_fit = pred.mu, pred.cov

                ls_opt_effective = (
                    float(refit_diag["refit_ls_opt"])
                    if bool(refit_gp_on_toy) and refit_ok == 1.0
                    else float(initial_ls_opt)
                )
                const_opt_effective = (
                    float(refit_diag["refit_const_opt"])
                    if bool(refit_gp_on_toy) and refit_ok == 1.0
                    else float(initial_const_opt)
                )

                fit = fit_A_profiled_gaussian(
                    obs, mu_fit, cov_fit, tmpl_win,
                    allow_negative=bool(getattr(config, "extract_allow_negative", True)),
                )
                A_hat = float(fit["A_hat"])
                sigma_A = float(fit["sigma_A"])
                pull = (A_hat - A_inj) / sigma_A if np.isfinite(sigma_A) and sigma_A > 0 else float("nan")
                Zhat = A_hat / sigma_A if np.isfinite(sigma_A) and sigma_A > 0 else float("nan")

                row = dict(
                    dataset=ds.key, mass_GeV=m, toy=int(i),
                    strength=float(A_inj), inj_nsigma=float(inj_nsigma),
                    sigmaA_ref=float(sigmaA_ref), sigma_val=float(pred.sigma_val),
                    integral_density=float(integral_density),
                    A_per_eps2_unit=float(A_from_epsilon2(ds, float(m), 1.0, integral_density)),
                    sigma_x=float(getattr(pred, "sigma_x", float("nan"))),
                    kernel_ls_policy=str(kernel_diag["kernel_ls_policy"]),
                    kernel_ls_res_lower_factor=float(kernel_diag["kernel_ls_res_lower_factor"]),
                    kernel_ls_res_upper_factor=float(kernel_diag["kernel_ls_res_upper_factor"]),
                    ls_lo=float(kernel_diag["ls_lo"]),
                    ls_hi=float(kernel_diag["ls_hi"]),
                    ls_init=float(kernel_diag["ls_init"]),
                    initial_ls_opt=float(initial_ls_opt),
                    initial_const_opt=float(initial_const_opt),
                    refit_ls_opt=float(refit_diag["refit_ls_opt"]),
                    refit_const_opt=float(refit_diag["refit_const_opt"]),
                    ls_opt=float(ls_opt_effective),
                    const_opt=float(const_opt_effective),
                    f_win=float(f_win), f_full=float(f_full),
                    f_train=float(f_train), f_train_frac=float(f_train_frac),
                    n_train=int(n_train), n_train_low=int(n_train_low),
                    n_train_high=int(n_train_high), n_blind=int(n_blind),
                    blind_nsigma=float(config.blind_nsigma), train_exclude_nsigma=float(tn),
                    signal_model=str(getattr(config, "signal_model", "default")),
                    inj_shape_mode=inj_shape_mode,
                    A_hat=float(A_hat), sigma_A=float(sigma_A),
                    Zhat=float(Zhat), pull_param=float(pull),
                    Nsig_win=int(Nsig_win), Nsig_train=int(Nsig_train),
                    success=bool(fit["success"]), nll=float(fit.get("nll", float("nan"))),
                    toy_mode=toy_mode, refit_gp_on_toy=bool(refit_gp_on_toy),
                    refit_ok=float(refit_ok), refit_restarts=int(refit_restarts),
                    refit_optimize=bool(refit_optimize),
                    refit_fallback_used=bool(refit_fallback_used),
                    refit_error=str(refit_error),
                )
                row.update(_dataset_source_metadata(ds))
                row.update(_qmu_tilde_fields(config, obs, mu_fit, cov_fit, tmpl_win, A_inj))
                rows.append(row)

        print(f"[inj] {ds.key} m={m:.6g} GeV done ({toy_mode}; {len(strength_tags)} strengths × {n_toys} toys)")

    df = pd.DataFrame(rows)
    if bool(write_toy_csv):
        out_csv = os.path.join(outdir, f"inj_extract_toys_{ds.key}.csv")
        df.to_csv(out_csv, index=False)
        print("[inj] wrote", out_csv)
    else:
        print(f"[inj] skipped writing toy CSV for {ds.key} (write_toy_csv=False)")
    return df


@dataclass
class _InjectionMassContext:
    """Precomputed per-mass context used by streaming toy workers."""

    ds: "DatasetConfig"
    mass: float
    mu: np.ndarray
    cov: np.ndarray
    mu_full: np.ndarray
    x_full: np.ndarray
    msk_blind: np.ndarray
    msk_train: np.ndarray
    tmpl_win: np.ndarray
    tmpl_full: np.ndarray
    sigmaA_ref: float
    sigma_val: float
    sigma_x: float
    kernel_ls_policy: str
    kernel_ls_res_lower_factor: float
    kernel_ls_res_upper_factor: float
    ls_lo: float
    ls_hi: float
    ls_init: float
    initial_ls_opt: float
    initial_const_opt: float
    integral_density: float
    A_per_eps2_unit: float
    f_win: float
    f_full: float
    f_train: float
    f_train_frac: float
    n_train: int
    n_train_low: int
    n_train_high: int
    n_blind: int
    blind_nsigma: float
    train_exclude_nsigma: float
    signal_model: str
    inj_mode: str
    inj_shape_mode: str
    refit_gp_on_toy: bool
    refit_restarts: int
    refit_optimize: bool
    allow_negative: bool
    mvn_method: str
    mvn_max_tries: int


@dataclass
class _ToyPointAccumulator:
    """Compact per-point accumulator for summary-level statistics."""

    dataset: str
    mass_GeV: float
    strength: float
    sigmaA_ref: float
    integral_density: float
    A_per_eps2_unit: float
    f_win: float
    f_train_frac: float
    kernel_ls_policy: str = ""
    signal_model: str = ""
    kernel_ls_res_lower_factor: float = float("nan")
    kernel_ls_res_upper_factor: float = float("nan")
    ls_lo: float = float("nan")
    ls_hi: float = float("nan")
    ls_init: float = float("nan")
    initial_ls_opt: float = float("nan")
    initial_const_opt: float = float("nan")
    n_train: int = 0
    n_train_low: int = 0
    n_train_high: int = 0
    n_blind: int = 0
    pull_vals: List[float] = field(default_factory=list)
    zhat_vals: List[float] = field(default_factory=list)
    ahat_vals: List[float] = field(default_factory=list)
    sigma_a_vals: List[float] = field(default_factory=list)
    inj_nsigma_vals: List[float] = field(default_factory=list)
    success_vals: List[float] = field(default_factory=list)
    refit_ok_vals: List[float] = field(default_factory=list)
    refit_fallback_vals: List[float] = field(default_factory=list)
    refit_ls_opt_vals: List[float] = field(default_factory=list)
    refit_const_opt_vals: List[float] = field(default_factory=list)
    ls_opt_vals: List[float] = field(default_factory=list)
    const_opt_vals: List[float] = field(default_factory=list)
    qmu_vals: List[float] = field(default_factory=list)

    def update(self, row: Dict[str, Any]) -> None:
        self.pull_vals.append(float(row.get("pull_param", np.nan)))
        self.zhat_vals.append(float(row.get("Zhat", np.nan)))
        self.ahat_vals.append(float(row.get("A_hat", np.nan)))
        self.sigma_a_vals.append(float(row.get("sigma_A", np.nan)))
        self.inj_nsigma_vals.append(float(row.get("inj_nsigma", np.nan)))
        self.success_vals.append(1.0 if bool(row.get("success", False)) else 0.0)
        self.refit_ok_vals.append(float(row.get("refit_ok", np.nan)))
        self.refit_fallback_vals.append(1.0 if bool(row.get("refit_fallback_used", False)) else 0.0)
        self.refit_ls_opt_vals.append(float(row.get("refit_ls_opt", np.nan)))
        self.refit_const_opt_vals.append(float(row.get("refit_const_opt", np.nan)))
        self.ls_opt_vals.append(float(row.get("ls_opt", np.nan)))
        self.const_opt_vals.append(float(row.get("const_opt", np.nan)))
        self.qmu_vals.append(float(row.get("qmu_tilde", np.nan)))

    def update_many(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            self.update(row)

    def finalize(self) -> Dict[str, Any]:
        pull = np.asarray(self.pull_vals, float)
        zhat = np.asarray(self.zhat_vals, float)
        ahat = np.asarray(self.ahat_vals, float)
        siga = np.asarray(self.sigma_a_vals, float)
        inj = np.asarray(self.inj_nsigma_vals, float)
        succ = np.asarray(self.success_vals, float)
        refit_ok = np.asarray(self.refit_ok_vals, float)
        refit_fb = np.asarray(self.refit_fallback_vals, float)
        refit_ls_opt = np.asarray(self.refit_ls_opt_vals, float)
        refit_const_opt = np.asarray(self.refit_const_opt_vals, float)
        ls_opt = np.asarray(self.ls_opt_vals, float)
        const_opt = np.asarray(self.const_opt_vals, float)
        qmu = np.asarray(self.qmu_vals, float)
        n_toys = int(len(pull))
        pull_finite = pull[np.isfinite(pull)]

        def q(x: np.ndarray, p: float) -> float:
            v = np.asarray(x, float)
            return float(np.nanquantile(v, p)) if np.any(np.isfinite(v)) else float("nan")

        inj_std = float(np.nanstd(inj, ddof=1)) if np.sum(np.isfinite(inj)) > 1 else 0.0
        inj_level = float(np.nanmean(inj)) if n_toys else float("nan")
        pull_mean = float(np.nanmean(pull)) if n_toys else float("nan")
        zhat_mean = float(np.nanmean(zhat)) if n_toys else float("nan")

        return dict(
            dataset=str(self.dataset),
            mass_GeV=float(self.mass_GeV),
            strength=float(self.strength),
            n_toys=n_toys,
            inj_nsigma=inj_level,
            inj_nsigma_xerr=float(inj_std),
            sigmaA_ref=float(self.sigmaA_ref),
            integral_density=float(self.integral_density),
            A_per_eps2_unit=float(self.A_per_eps2_unit),
            f_win=float(self.f_win),
            f_train_frac=float(self.f_train_frac),
            kernel_ls_policy=str(self.kernel_ls_policy),
            signal_model=str(getattr(self, "signal_model", "")),
            kernel_ls_res_lower_factor=float(self.kernel_ls_res_lower_factor),
            kernel_ls_res_upper_factor=float(self.kernel_ls_res_upper_factor),
            ls_lo=float(self.ls_lo),
            ls_hi=float(self.ls_hi),
            ls_init=float(self.ls_init),
            initial_ls_opt=float(self.initial_ls_opt),
            initial_const_opt=float(self.initial_const_opt),
            refit_ls_opt=float(np.nanmean(refit_ls_opt)) if np.any(np.isfinite(refit_ls_opt)) else float("nan"),
            refit_const_opt=float(np.nanmean(refit_const_opt)) if np.any(np.isfinite(refit_const_opt)) else float("nan"),
            ls_opt=float(np.nanmean(ls_opt)) if np.any(np.isfinite(ls_opt)) else float("nan"),
            const_opt=float(np.nanmean(const_opt)) if np.any(np.isfinite(const_opt)) else float("nan"),
            n_train=int(self.n_train),
            n_train_low=int(self.n_train_low),
            n_train_high=int(self.n_train_high),
            n_blind=int(self.n_blind),
            A_hat_mean=float(np.nanmean(ahat)) if n_toys else float("nan"),
            A_hat_std=float(np.nanstd(ahat, ddof=1)) if n_toys > 1 else float("nan"),
            sigma_A_mean=float(np.nanmean(siga)) if n_toys else float("nan"),
            Ahat_over_Ainj=(
                float(np.nanmean(ahat)) / float(self.strength)
                if n_toys and np.isfinite(float(self.strength)) and abs(float(self.strength)) > 0
                else float("nan")
            ),
            sigmaA_over_sigmaAref=(
                float(np.nanmean(siga)) / float(self.sigmaA_ref)
                if n_toys and np.isfinite(float(self.sigmaA_ref)) and float(self.sigmaA_ref) > 0
                else float("nan")
            ),
            pull_mean=float(pull_mean),
            pull_std=float(np.nanstd(pull, ddof=1)) if n_toys > 1 else float("nan"),
            pull_std_err=(
                float(np.nanstd(pull_finite, ddof=1) / np.sqrt(max(1, 2 * (len(pull_finite) - 1))))
                if len(pull_finite) > 1
                else float("nan")
            ),
            pull_q16=q(pull, 0.16),
            pull_q84=q(pull, 0.84),
            pull_q02=q(pull, 0.025),
            pull_q97=q(pull, 0.975),
            cov_1sigma=float(np.nanmean(np.abs(pull) < 1.0)) if n_toys else float("nan"),
            cov_2sigma=float(np.nanmean(np.abs(pull) < 2.0)) if n_toys else float("nan"),
            Zhat_mean=float(zhat_mean),
            delta_z_minus_pull=(
                float((zhat_mean - inj_level) - pull_mean)
                if np.isfinite(zhat_mean) and np.isfinite(inj_level) and np.isfinite(pull_mean)
                else float("nan")
            ),
            ainj_over_sigmaAref=(
                float(self.strength) / float(self.sigmaA_ref)
                if np.isfinite(self.sigmaA_ref) and float(self.sigmaA_ref) > 0
                else float("nan")
            ),
            Zhat_q16=q(zhat, 0.16),
            Zhat_q84=q(zhat, 0.84),
            qmu_tilde_mean=float(np.nanmean(qmu)) if np.any(np.isfinite(qmu)) else float("nan"),
            qmu_tilde_q16=q(qmu, 0.16),
            qmu_tilde_q84=q(qmu, 0.84),
            n_source_toys=0,
            source_model="",
            source_label="",
            source_root="",
            container="",
            function_tag="",
            success_rate=float(np.nanmean(succ)) if succ.size else float("nan"),
            refit_ok_rate=float(np.nanmean(refit_ok)) if np.any(np.isfinite(refit_ok)) else float("nan"),
            refit_fallback_rate=float(np.nanmean(refit_fb)) if refit_fb.size else float("nan"),
        )


def _stable_point_seed(base_seed: int, dataset_key: str, mass: float, strength: float) -> int:
    """Stable point seed independent of runtime ordering/scheduling."""
    payload = f"{int(base_seed)}|{str(dataset_key)}|{float(mass):.9f}|{float(strength):.12g}"
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") & 0xFFFFFFFF


def _stable_toy_seed(point_seed: int, toy_index: int) -> int:
    """Stable toy seed from point seed + toy index."""
    return int((int(point_seed) + int(toy_index) * 2654435761) & 0xFFFFFFFF)


def _chunk_indices(indices: List[int], n_chunks: int) -> List[List[int]]:
    """Split integer indices into approximately equal contiguous chunks."""
    if not indices:
        return []
    n_chunks = int(max(1, n_chunks))
    if n_chunks <= 1 or len(indices) <= 1:
        return [indices]
    chunks: List[List[int]] = []
    step = int(np.ceil(len(indices) / float(n_chunks)))
    for i in range(0, len(indices), step):
        chunks.append(indices[i:i + step])
    return [c for c in chunks if c]


def _append_toy_rows_csv(path: str, rows: List[Dict[str, Any]], *, header_written: bool) -> bool:
    """Append toy rows to a CSV file, writing header only once."""
    if not rows:
        return bool(header_written)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", index=False, header=not bool(header_written))
    return True


def _simulate_toy_rows_chunk(
    ctx: _InjectionMassContext,
    config: "Config",
    *,
    toy_indices: List[int],
    A_inj: float,
    inj_nsigma: float,
    point_seed: int,
    threads_per_worker: int,
) -> List[Dict[str, Any]]:
    """Simulate one chunk of toys for one (dataset, mass, strength) point."""
    if not toy_indices:
        return []

    out_rows: List[Dict[str, Any]] = []
    toy_mode = "full_refit" if bool(ctx.refit_gp_on_toy) else "conditional_gp"

    with _threadpool_limits(limits=int(max(1, threads_per_worker))):
        ker = make_kernel_for_dataset(ctx.ds, config, mass=float(ctx.mass)) if bool(ctx.refit_gp_on_toy) else None
        x_win = ctx.x_full[ctx.msk_blind]
        for toy_idx in toy_indices:
            rng = np.random.default_rng(_stable_toy_seed(point_seed, int(toy_idx)))
            refit_ok = float("nan")
            refit_fallback_used = False
            refit_error = ""
            refit_diag = dict(refit_ls_opt=float("nan"), refit_const_opt=float("nan"))

            if bool(ctx.refit_gp_on_toy):
                bkg_full = rng.poisson(np.clip(ctx.mu_full, 0.0, None)).astype(int)
                if str(ctx.inj_shape_mode) == "window":
                    sig_full = np.zeros_like(bkg_full, dtype=int)
                    s_win, Nsig_win, _ = _inject_counts_from_template(ctx.tmpl_win, A_inj, rng, ctx.inj_mode)
                    idx_blind = np.where(ctx.msk_blind)[0]
                    n = min(len(s_win), len(idx_blind))
                    sig_full[idx_blind[:n]] = s_win[:n]
                else:
                    s_full, _, _ = _inject_counts_from_template(ctx.tmpl_full, A_inj, rng, ctx.inj_mode)
                    sig_full = np.asarray(s_full, dtype=int)
                    Nsig_win = int(np.sum(sig_full[ctx.msk_blind]))

                y_toy = (bkg_full + sig_full).astype(int)
                obs = y_toy[ctx.msk_blind].astype(int)
                Nsig_train = int(np.sum(sig_full[ctx.msk_train]))
                mu_fit = np.asarray(ctx.mu, float)
                cov_fit = np.asarray(ctx.cov, float)
                try:
                    X_tr = ctx.x_full[ctx.msk_train]
                    y_tr = y_toy[ctx.msk_train].astype(float)
                    gpr = fit_gpr(
                        X_tr,
                        y_tr,
                        config,
                        restarts=int(ctx.refit_restarts),
                        kernel=ker,
                        optimize=bool(ctx.refit_optimize),
                    )
                    mu_fit, cov_fit = predict_counts_from_log_gpr(gpr, x_win, config)
                    diag = _gpr_fit_diagnostics(gpr)
                    refit_diag.update(
                        refit_ls_opt=float(diag["ls_opt"]),
                        refit_const_opt=float(diag["const_opt"]),
                    )
                    refit_ok = 1.0
                except Exception as exc:
                    refit_ok = 0.0
                    refit_fallback_used = True
                    refit_error = _handle_refit_failure(
                        config,
                        f"{ctx.ds.key} m={float(ctx.mass):.6g} toy={int(toy_idx)} A={float(A_inj):.6g}",
                        exc,
                    )
            else:
                b = draw_bkg_mvn_nonneg(
                    ctx.mu,
                    ctx.cov,
                    1,
                    rng,
                    method=str(ctx.mvn_method),
                    max_tries=int(ctx.mvn_max_tries),
                )[0]
                sig, Nsig_win, _ = _inject_counts_from_template(ctx.tmpl_win, A_inj, rng, ctx.inj_mode)
                lam = np.clip(b, 0.0, None) + np.clip(sig.astype(float), 0.0, None)
                obs = rng.poisson(lam).astype(int)
                Nsig_train = 0
                mu_fit, cov_fit = ctx.mu, ctx.cov

            ls_opt_effective = (
                float(refit_diag["refit_ls_opt"])
                if bool(ctx.refit_gp_on_toy) and refit_ok == 1.0
                else float(ctx.initial_ls_opt)
            )
            const_opt_effective = (
                float(refit_diag["refit_const_opt"])
                if bool(ctx.refit_gp_on_toy) and refit_ok == 1.0
                else float(ctx.initial_const_opt)
            )

            fit = fit_A_profiled_gaussian(
                obs,
                np.asarray(mu_fit, float),
                np.asarray(cov_fit, float),
                ctx.tmpl_win,
                allow_negative=bool(ctx.allow_negative),
            )
            A_hat = float(fit["A_hat"])
            sigma_A = float(fit["sigma_A"])
            pull = (A_hat - float(A_inj)) / sigma_A if np.isfinite(sigma_A) and sigma_A > 0 else float("nan")
            Zhat = A_hat / sigma_A if np.isfinite(sigma_A) and sigma_A > 0 else float("nan")

            row = dict(
                dataset=str(ctx.ds.key),
                mass_GeV=float(ctx.mass),
                toy=int(toy_idx),
                strength=float(A_inj),
                inj_nsigma=float(inj_nsigma),
                sigmaA_ref=float(ctx.sigmaA_ref),
                integral_density=float(ctx.integral_density),
                A_per_eps2_unit=float(ctx.A_per_eps2_unit),
                sigma_val=float(ctx.sigma_val),
                sigma_x=float(ctx.sigma_x),
                kernel_ls_policy=str(ctx.kernel_ls_policy),
                kernel_ls_res_lower_factor=float(ctx.kernel_ls_res_lower_factor),
                kernel_ls_res_upper_factor=float(ctx.kernel_ls_res_upper_factor),
                ls_lo=float(ctx.ls_lo),
                ls_hi=float(ctx.ls_hi),
                ls_init=float(ctx.ls_init),
                initial_ls_opt=float(ctx.initial_ls_opt),
                initial_const_opt=float(ctx.initial_const_opt),
                refit_ls_opt=float(refit_diag["refit_ls_opt"]),
                refit_const_opt=float(refit_diag["refit_const_opt"]),
                ls_opt=float(ls_opt_effective),
                const_opt=float(const_opt_effective),
                f_win=float(ctx.f_win),
                f_full=float(ctx.f_full),
                f_train=float(ctx.f_train),
                f_train_frac=float(ctx.f_train_frac),
                n_train=int(ctx.n_train),
                n_train_low=int(ctx.n_train_low),
                n_train_high=int(ctx.n_train_high),
                n_blind=int(ctx.n_blind),
                blind_nsigma=float(ctx.blind_nsigma),
                train_exclude_nsigma=float(ctx.train_exclude_nsigma),
                signal_model=str(ctx.signal_model),
                inj_shape_mode=str(ctx.inj_shape_mode),
                A_hat=float(A_hat),
                sigma_A=float(sigma_A),
                Zhat=float(Zhat),
                pull_param=float(pull),
                Nsig_win=int(Nsig_win),
                Nsig_train=int(Nsig_train),
                success=bool(fit["success"]),
                nll=float(fit.get("nll", float("nan"))),
                toy_mode=str(toy_mode),
                refit_gp_on_toy=bool(ctx.refit_gp_on_toy),
                refit_ok=float(refit_ok),
                refit_restarts=int(ctx.refit_restarts),
                refit_optimize=bool(ctx.refit_optimize),
                refit_fallback_used=bool(refit_fallback_used),
                refit_error=str(refit_error),
            )
            row.update(_dataset_source_metadata(ctx.ds))
            row.update(_qmu_tilde_fields(config, obs, mu_fit, cov_fit, ctx.tmpl_win, A_inj))
            out_rows.append(row)

    return out_rows


def _simulate_toy_rows_batch(
    ctx: _InjectionMassContext,
    config: "Config",
    *,
    toy_indices: List[int],
    A_inj: float,
    inj_nsigma: float,
    point_seed: int,
    n_workers: int,
    parallel_backend: str,
    threads_per_worker: int,
) -> List[Dict[str, Any]]:
    """Simulate a toy batch with optional per-point worker parallelism."""
    chunks = _chunk_indices(list(toy_indices), int(max(1, n_workers)))
    if not chunks:
        return []
    if int(n_workers) <= 1 or not _HAVE_JOBLIB:
        rows = _simulate_toy_rows_chunk(
            ctx,
            config,
            toy_indices=chunks[0],
            A_inj=float(A_inj),
            inj_nsigma=float(inj_nsigma),
            point_seed=int(point_seed),
            threads_per_worker=int(threads_per_worker),
        )
        rows.sort(key=lambda r: int(r["toy"]))
        return rows

    parts = joblib.Parallel(n_jobs=int(n_workers), backend=str(parallel_backend))(
        joblib.delayed(_simulate_toy_rows_chunk)(
            ctx,
            config,
            toy_indices=chunk,
            A_inj=float(A_inj),
            inj_nsigma=float(inj_nsigma),
            point_seed=int(point_seed),
            threads_per_worker=int(threads_per_worker),
        )
        for chunk in chunks
    )
    rows = [r for part in parts for r in part]
    rows.sort(key=lambda r: int(r["toy"]))
    return rows


def _resolve_injection_strength_tags(
    *,
    config: "Config",
    strengths: Optional[List[float]] = None,
    strengths_mode: Optional[str] = None,
    sigma_multipliers: Optional[List[float]] = None,
) -> Tuple[str, List[float]]:
    """Resolve injection strength mode + requested strength tags."""
    mode = (strengths_mode or getattr(config, "inj_strength_mode", "absolute")).lower().strip()
    if mode == "sigmaa":
        if strengths is not None:
            tags = [float(x) for x in strengths]
        elif sigma_multipliers is not None:
            tags = [float(x) for x in sigma_multipliers]
        else:
            tags = [float(x) for x in getattr(config, "inj_sigma_multipliers", [0.0, 1.0, 2.0, 3.0])]
    else:
        tags = [float(x) for x in (strengths or getattr(config, "inj_strengths", []))]
    return mode, tags


def _build_injection_mass_context(
    ds: "DatasetConfig",
    config: "Config",
    *,
    mass: float,
    seed: int,
    inj_mode: str,
    sigma_source: str,
    refit_gp_on_toy: bool,
    refit_restarts: int,
    refit_optimize: bool,
    inj_shape_mode: str,
    train_exclude_nsigma: Optional[float],
    mvn_method: str,
    mvn_max_tries: int,
) -> _InjectionMassContext:
    """Build per-mass context used by the streaming toy runner."""
    pred = estimate_background_for_dataset(ds, float(mass), config)
    blind_mask = _prediction_blind_mask(pred)

    tmpl_win, tmpl_full = build_window_template_from_full(
        pred.edges_full, blind_mask, float(mass), pred.sigma_val, config=config
    )
    f_win = float(np.sum(tmpl_win))
    x_full = np.asarray(pred.x_full, float).reshape(-1)
    f_full = float(np.sum(tmpl_full))
    kernel_diag = _kernel_static_diagnostics(ds, config, float(mass))
    initial_ls_opt = float(getattr(pred, "ls_opt", float("nan")))
    initial_const_opt = float(getattr(pred, "const_opt", float("nan")))

    sigma_seed = _stable_point_seed(int(seed), str(ds.key), float(mass), -1.0)
    sigma_rng = np.random.default_rng(int(sigma_seed))
    sigmaA_ref = _sigmaA_reference(pred, float(mass), source=str(sigma_source), rng=sigma_rng, config=config)
    integral_density = _prediction_integral_density(pred)
    A_per_eps2_unit = float(A_from_epsilon2(ds, float(mass), 1.0, integral_density))

    train_nsig_default = float(getattr(config, "gp_train_exclude_nsigma", None) or config.blind_nsigma)
    if train_exclude_nsigma is not None:
        tn = float(train_exclude_nsigma)
    else:
        tn = float(getattr(pred, "train_exclude_nsigma", train_nsig_default))

    blind = tuple(pred.blind)
    msk_blind = (x_full >= float(blind[0])) & (x_full <= float(blind[1]))
    if int(np.sum(msk_blind)) == 0:
        raise RuntimeError(f"[inj][{ds.key}] m={float(mass):.6g}: blind window has no bins")
    blind_train = (float(mass) - tn * float(pred.sigma_val), float(mass) + tn * float(pred.sigma_val))
    msk_train = (x_full < blind_train[0]) | (x_full > blind_train[1])
    n_train_low = int(np.count_nonzero(x_full < blind_train[0]))
    n_train_high = int(np.count_nonzero(x_full > blind_train[1]))
    n_train = int(np.count_nonzero(msk_train))
    n_blind = int(np.count_nonzero(msk_blind))
    f_train = float(np.sum(tmpl_full[msk_train])) if tmpl_full.shape[0] == x_full.shape[0] else float("nan")
    f_train_frac = float(f_train / f_full) if np.isfinite(f_train) and f_full > 0 else float("nan")

    return _InjectionMassContext(
        ds=ds,
        mass=float(mass),
        mu=np.asarray(pred.mu, float),
        cov=np.asarray(pred.cov, float),
        mu_full=np.asarray(pred.mu_full, float).reshape(-1),
        x_full=x_full,
        msk_blind=msk_blind,
        msk_train=msk_train,
        tmpl_win=np.asarray(tmpl_win, float),
        tmpl_full=np.asarray(tmpl_full, float),
        sigmaA_ref=float(sigmaA_ref),
        sigma_val=float(pred.sigma_val),
        sigma_x=float(getattr(pred, "sigma_x", float("nan"))),
        kernel_ls_policy=str(kernel_diag["kernel_ls_policy"]),
        kernel_ls_res_lower_factor=float(kernel_diag["kernel_ls_res_lower_factor"]),
        kernel_ls_res_upper_factor=float(kernel_diag["kernel_ls_res_upper_factor"]),
        ls_lo=float(kernel_diag["ls_lo"]),
        ls_hi=float(kernel_diag["ls_hi"]),
        ls_init=float(kernel_diag["ls_init"]),
        initial_ls_opt=float(initial_ls_opt),
        initial_const_opt=float(initial_const_opt),
        integral_density=float(integral_density),
        A_per_eps2_unit=float(A_per_eps2_unit),
        f_win=float(f_win),
        f_full=float(f_full),
        f_train=float(f_train),
        f_train_frac=float(f_train_frac),
        n_train=int(n_train),
        n_train_low=int(n_train_low),
        n_train_high=int(n_train_high),
        n_blind=int(n_blind),
        blind_nsigma=float(config.blind_nsigma),
        train_exclude_nsigma=float(tn),
        signal_model=str(getattr(config, "signal_model", "default")),
        inj_mode=str(inj_mode),
        inj_shape_mode=str(inj_shape_mode),
        refit_gp_on_toy=bool(refit_gp_on_toy),
        refit_restarts=int(refit_restarts),
        refit_optimize=bool(refit_optimize),
        allow_negative=bool(getattr(config, "extract_allow_negative", True)),
        mvn_method=str(mvn_method),
        mvn_max_tries=int(mvn_max_tries),
    )


def run_injection_extraction_streaming(
    ds: "DatasetConfig",
    config: "Config",
    *,
    masses: List[float],
    strengths: Optional[List[float]] = None,
    n_toys: int = 200,
    outdir: Optional[str] = None,
    seed: int = 314159,
    inj_mode: Optional[str] = None,
    strengths_mode: Optional[str] = None,
    sigma_multipliers: Optional[List[float]] = None,
    sigma_source: Optional[str] = None,
    refit_gp_on_toy: Optional[bool] = None,
    refit_restarts: Optional[int] = None,
    refit_optimize: Optional[bool] = None,
    inj_shape_mode: Optional[str] = None,
    train_exclude_nsigma: Optional[float] = None,
    write_toy_csv: Optional[bool] = None,
    aggregate_every: Optional[int] = None,
    n_workers: Optional[int] = None,
    parallel_backend: Optional[str] = None,
    threads_per_worker: Optional[int] = None,
) -> pd.DataFrame:
    """Streaming injection runner for one dataset.

    Processes toys in small batches and stores only per-point aggregate statistics
    (plus optional toy CSV append output).
    """
    if outdir is None:
        outdir = os.path.join(config.output_dir, "injection_extraction")
    ensure_dir(outdir)

    if write_toy_csv is None:
        write_toy_csv = bool(getattr(config, "inj_write_toy_csv", True))
    if aggregate_every is None:
        aggregate_every = int(getattr(config, "inj_aggregate_every", 100))
    if n_workers is None:
        n_workers = int(getattr(config, "inj_n_workers", 5))
    if parallel_backend is None:
        parallel_backend = str(getattr(config, "inj_parallel_backend", "loky"))
    if threads_per_worker is None:
        threads_per_worker = int(getattr(config, "inj_threads_per_worker", 1))

    aggregate_every = int(max(1, aggregate_every))
    n_workers = int(max(1, n_workers))
    threads_per_worker = int(max(1, threads_per_worker))

    inj_mode = (inj_mode or getattr(config, "inj_mode", "poisson")).lower().strip()
    sigma_source = (sigma_source or getattr(config, "inj_sigma_a_source", "asimov")).lower().strip()
    if refit_gp_on_toy is None:
        refit_gp_on_toy = bool(getattr(config, "inj_refit_gp_on_toy", False))
    if refit_restarts is None:
        refit_restarts = int(getattr(config, "inj_refit_gp_restarts", 0))
    if refit_optimize is None:
        refit_optimize = bool(getattr(config, "inj_refit_gp_optimize", True))
    inj_shape_mode = (inj_shape_mode or getattr(config, "inj_shape_mode", "full")).lower().strip()
    if inj_shape_mode not in ("full", "window"):
        inj_shape_mode = "full"
    if train_exclude_nsigma is None:
        train_exclude_nsigma = getattr(config, "inj_train_exclude_nsigma", None)

    mvn_method = str(getattr(config, "mvn_trunc_method", "reject_then_clip"))
    mvn_max_tries = int(getattr(config, "mvn_trunc_max_tries", 80))
    strengths_mode_resolved, strength_tags = _resolve_injection_strength_tags(
        config=config,
        strengths=strengths,
        strengths_mode=strengths_mode,
        sigma_multipliers=sigma_multipliers,
    )

    toy_csv_path = os.path.join(outdir, f"inj_extract_toys_{ds.key}.csv")
    toy_header_written = False
    if bool(write_toy_csv) and os.path.exists(toy_csv_path):
        os.remove(toy_csv_path)

    summary_rows: List[Dict[str, Any]] = []
    for m in [float(x) for x in masses]:
        ctx = _build_injection_mass_context(
            ds,
            config,
            mass=float(m),
            seed=int(seed),
            inj_mode=str(inj_mode),
            sigma_source=str(sigma_source),
            refit_gp_on_toy=bool(refit_gp_on_toy),
            refit_restarts=int(refit_restarts),
            refit_optimize=bool(refit_optimize),
            inj_shape_mode=str(inj_shape_mode),
            train_exclude_nsigma=train_exclude_nsigma,
            mvn_method=str(mvn_method),
            mvn_max_tries=int(mvn_max_tries),
        )

        if strengths_mode_resolved == "sigmaa":
            A_inj_list = [float(t) * float(ctx.sigmaA_ref) for t in strength_tags]
            inj_nsigma_list = [float(t) for t in strength_tags]
        else:
            A_inj_list = [float(t) for t in strength_tags]
            inj_nsigma_list = [
                float(A) / float(ctx.sigmaA_ref) if np.isfinite(ctx.sigmaA_ref) and ctx.sigmaA_ref > 0 else float("nan")
                for A in A_inj_list
            ]

        for A_inj, inj_nsigma_val in zip(A_inj_list, inj_nsigma_list):
            acc = _ToyPointAccumulator(
                dataset=str(ds.key),
                mass_GeV=float(m),
                strength=float(A_inj),
                sigmaA_ref=float(ctx.sigmaA_ref),
                integral_density=float(ctx.integral_density),
                A_per_eps2_unit=float(ctx.A_per_eps2_unit),
                f_win=float(ctx.f_win),
                f_train_frac=float(ctx.f_train_frac),
                kernel_ls_policy=str(ctx.kernel_ls_policy),
                signal_model=str(ctx.signal_model),
                kernel_ls_res_lower_factor=float(ctx.kernel_ls_res_lower_factor),
                kernel_ls_res_upper_factor=float(ctx.kernel_ls_res_upper_factor),
                ls_lo=float(ctx.ls_lo),
                ls_hi=float(ctx.ls_hi),
                ls_init=float(ctx.ls_init),
                initial_ls_opt=float(ctx.initial_ls_opt),
                initial_const_opt=float(ctx.initial_const_opt),
                n_train=int(ctx.n_train),
                n_train_low=int(ctx.n_train_low),
                n_train_high=int(ctx.n_train_high),
                n_blind=int(ctx.n_blind),
            )
            point_seed = _stable_point_seed(int(seed), str(ds.key), float(m), float(A_inj))
            done = 0
            while done < int(n_toys):
                end = min(done + aggregate_every, int(n_toys))
                toy_indices = list(range(done, end))
                batch_rows = _simulate_toy_rows_batch(
                    ctx,
                    config,
                    toy_indices=toy_indices,
                    A_inj=float(A_inj),
                    inj_nsigma=float(inj_nsigma_val),
                    point_seed=int(point_seed),
                    n_workers=int(n_workers),
                    parallel_backend=str(parallel_backend),
                    threads_per_worker=int(threads_per_worker),
                )
                acc.update_many(batch_rows)
                if bool(write_toy_csv):
                    toy_header_written = _append_toy_rows_csv(
                        toy_csv_path, batch_rows, header_written=bool(toy_header_written)
                    )
                done = end
                print(
                    "[inj][stream] "
                    f"{ds.key} m={float(m):.6g} GeV strength={float(A_inj):.6g}: "
                    f"{done}/{int(n_toys)} toys"
                )

            summary_rows.append(acc.finalize())

    df_sum = pd.DataFrame(summary_rows)
    if df_sum.empty:
        return df_sum
    return df_sum.sort_values(["dataset", "mass_GeV", "strength"]).reset_index(drop=True)


def _combine_toy_rows(rows: List[Dict[str, Any]], *, mass: float, toy_idx: int) -> Optional[Dict[str, Any]]:
    """Combine per-dataset toy rows with inverse-variance weighting."""
    valid = []
    for r in rows:
        s = float(r.get("sigma_A", np.nan))
        if np.isfinite(s) and s > 0:
            valid.append(r)
    if not valid:
        return None

    sigma = np.array([float(r["sigma_A"]) for r in valid], float)
    w = 1.0 / np.square(sigma)
    if not np.any(np.isfinite(w)) or float(np.sum(w)) <= 0:
        return None
    sum_w = float(np.sum(w))

    ah = np.array([float(r.get("A_hat", np.nan)) for r in valid], float)
    st = np.array([float(r.get("strength", np.nan)) for r in valid], float)
    zinj = np.array([float(r.get("inj_nsigma", np.nan)) for r in valid], float)
    sref = np.array([float(r.get("sigmaA_ref", np.nan)) for r in valid], float)

    A_hat = float(np.nansum(w * ah) / sum_w)
    sigma_A = float(1.0 / np.sqrt(sum_w))
    strength = float(np.nansum(w * st) / sum_w)
    inj_nsigma = float(np.nanmean(zinj)) if np.any(np.isfinite(zinj)) else float("nan")
    sigmaA_ref = float(np.nanmean(sref)) if np.any(np.isfinite(sref)) else float("nan")
    pull = (A_hat - strength) / sigma_A if np.isfinite(sigma_A) and sigma_A > 0 else float("nan")
    zhat = A_hat / sigma_A if np.isfinite(sigma_A) and sigma_A > 0 else float("nan")

    return dict(
        dataset="combined",
        mass_GeV=float(mass),
        toy=int(toy_idx),
        strength=float(strength),
        inj_nsigma=float(inj_nsigma),
        sigmaA_ref=float(sigmaA_ref),
        A_hat=float(A_hat),
        sigma_A=float(sigma_A),
        pull_param=float(pull),
        Zhat=float(zhat),
        n_contrib=int(len(valid)),
        contrib_datasets="+".join(sorted({str(r.get("dataset", "")) for r in valid})),
        success=bool(all(bool(r.get("success", False)) for r in valid)),
    )


def run_injection_extraction_streaming_combined(
    datasets: Dict[str, "DatasetConfig"],
    config: "Config",
    *,
    masses: List[float],
    strengths: Optional[List[float]] = None,
    n_toys: int = 200,
    outdir: Optional[str] = None,
    seed: int = 314159,
    inj_mode: Optional[str] = None,
    strengths_mode: Optional[str] = None,
    sigma_multipliers: Optional[List[float]] = None,
    sigma_source: Optional[str] = None,
    refit_gp_on_toy: Optional[bool] = None,
    refit_restarts: Optional[int] = None,
    refit_optimize: Optional[bool] = None,
    inj_shape_mode: Optional[str] = None,
    train_exclude_nsigma: Optional[float] = None,
    write_toy_csv: Optional[bool] = None,
    aggregate_every: Optional[int] = None,
    n_workers: Optional[int] = None,
    parallel_backend: Optional[str] = None,
    threads_per_worker: Optional[int] = None,
    mass_policy: Optional[str] = None,
    min_n_contrib: Optional[int] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Streaming injection runner across multiple datasets + on-the-fly combined toy aggregation."""
    if outdir is None:
        outdir = os.path.join(config.output_dir, "injection_extraction")
    ensure_dir(outdir)

    ds_keys = [str(k) for k in datasets.keys()]
    if not ds_keys:
        return {}, pd.DataFrame()

    if write_toy_csv is None:
        write_toy_csv = bool(getattr(config, "inj_write_toy_csv", True))
    if aggregate_every is None:
        aggregate_every = int(getattr(config, "inj_aggregate_every", 100))
    if n_workers is None:
        n_workers = int(getattr(config, "inj_n_workers", 5))
    if parallel_backend is None:
        parallel_backend = str(getattr(config, "inj_parallel_backend", "loky"))
    if threads_per_worker is None:
        threads_per_worker = int(getattr(config, "inj_threads_per_worker", 1))
    mass_policy = str(mass_policy or getattr(config, "inj_combined_mass_policy", "intersection")).strip().lower()
    min_n_contrib = int(min_n_contrib if min_n_contrib is not None else getattr(config, "inj_combined_min_n_contrib", 2))

    aggregate_every = int(max(1, aggregate_every))
    n_workers = int(max(1, n_workers))
    threads_per_worker = int(max(1, threads_per_worker))

    inj_mode = (inj_mode or getattr(config, "inj_mode", "poisson")).lower().strip()
    sigma_source = (sigma_source or getattr(config, "inj_sigma_a_source", "asimov")).lower().strip()
    if refit_gp_on_toy is None:
        refit_gp_on_toy = bool(getattr(config, "inj_refit_gp_on_toy", False))
    if refit_restarts is None:
        refit_restarts = int(getattr(config, "inj_refit_gp_restarts", 0))
    if refit_optimize is None:
        refit_optimize = bool(getattr(config, "inj_refit_gp_optimize", True))
    inj_shape_mode = (inj_shape_mode or getattr(config, "inj_shape_mode", "full")).lower().strip()
    if inj_shape_mode not in ("full", "window"):
        inj_shape_mode = "full"
    if train_exclude_nsigma is None:
        train_exclude_nsigma = getattr(config, "inj_train_exclude_nsigma", None)

    mvn_method = str(getattr(config, "mvn_trunc_method", "reject_then_clip"))
    mvn_max_tries = int(getattr(config, "mvn_trunc_max_tries", 80))
    strengths_mode_resolved, strength_tags = _resolve_injection_strength_tags(
        config=config,
        strengths=strengths,
        strengths_mode=strengths_mode,
        sigma_multipliers=sigma_multipliers,
    )

    contexts_by_ds: Dict[str, Dict[float, _InjectionMassContext]] = {k: {} for k in ds_keys}
    for ds_key, ds in datasets.items():
        for m in [float(x) for x in masses]:
            contexts_by_ds[str(ds_key)][float(m)] = _build_injection_mass_context(
                ds,
                config,
                mass=float(m),
                seed=int(seed),
                inj_mode=str(inj_mode),
                sigma_source=str(sigma_source),
                refit_gp_on_toy=bool(refit_gp_on_toy),
                refit_restarts=int(refit_restarts),
                refit_optimize=bool(refit_optimize),
                inj_shape_mode=str(inj_shape_mode),
                train_exclude_nsigma=train_exclude_nsigma,
                mvn_method=str(mvn_method),
                mvn_max_tries=int(mvn_max_tries),
            )

    support_input = {
        k: pd.DataFrame({"mass_GeV": sorted(list(v.keys()))}) for k, v in contexts_by_ds.items()
    }
    support = _combined_mass_support_summary(
        support_input,
        mass_policy=mass_policy,
        min_n_contrib=int(min_n_contrib),
    )
    print(format_combined_mass_support_summary(support))
    accepted_masses = set(float(x) for x in support.get("accepted_masses", set()))

    toy_paths = {k: os.path.join(outdir, f"inj_extract_toys_{k}.csv") for k in ds_keys}
    toy_header_written = {k: False for k in ds_keys}
    comb_toy_path = os.path.join(outdir, "inj_extract_toys_combined.csv")
    comb_toy_header_written = False
    if bool(write_toy_csv):
        for p in [*toy_paths.values(), comb_toy_path]:
            if os.path.exists(p):
                os.remove(p)

    summary_rows_by_ds: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ds_keys}
    summary_rows_combined: List[Dict[str, Any]] = []

    for m in [float(x) for x in masses]:
        ctxs_m = {k: contexts_by_ds[k][float(m)] for k in ds_keys if float(m) in contexts_by_ds[k]}
        if not ctxs_m:
            continue

        for strength_tag in strength_tags:
            A_inj_by_ds: Dict[str, float] = {}
            inj_nsigma_by_ds: Dict[str, float] = {}
            for ds_key, ctx in ctxs_m.items():
                if strengths_mode_resolved == "sigmaa":
                    A_inj_by_ds[ds_key] = float(strength_tag) * float(ctx.sigmaA_ref)
                    inj_nsigma_by_ds[ds_key] = float(strength_tag)
                else:
                    A_inj_by_ds[ds_key] = float(strength_tag)
                    inj_nsigma_by_ds[ds_key] = (
                        float(strength_tag) / float(ctx.sigmaA_ref)
                        if np.isfinite(ctx.sigmaA_ref) and float(ctx.sigmaA_ref) > 0
                        else float("nan")
                    )

            acc_by_ds = {
                ds_key: _ToyPointAccumulator(
                    dataset=str(ds_key),
                    mass_GeV=float(m),
                    strength=float(A_inj_by_ds[ds_key]),
                    sigmaA_ref=float(ctxs_m[ds_key].sigmaA_ref),
                    integral_density=float(ctxs_m[ds_key].integral_density),
                    A_per_eps2_unit=float(ctxs_m[ds_key].A_per_eps2_unit),
                    f_win=float(ctxs_m[ds_key].f_win),
                    f_train_frac=float(ctxs_m[ds_key].f_train_frac),
                    kernel_ls_policy=str(ctxs_m[ds_key].kernel_ls_policy),
                    kernel_ls_res_lower_factor=float(ctxs_m[ds_key].kernel_ls_res_lower_factor),
                    kernel_ls_res_upper_factor=float(ctxs_m[ds_key].kernel_ls_res_upper_factor),
                    ls_lo=float(ctxs_m[ds_key].ls_lo),
                    ls_hi=float(ctxs_m[ds_key].ls_hi),
                    ls_init=float(ctxs_m[ds_key].ls_init),
                    initial_ls_opt=float(ctxs_m[ds_key].initial_ls_opt),
                    initial_const_opt=float(ctxs_m[ds_key].initial_const_opt),
                    n_train=int(ctxs_m[ds_key].n_train),
                    n_train_low=int(ctxs_m[ds_key].n_train_low),
                    n_train_high=int(ctxs_m[ds_key].n_train_high),
                    n_blind=int(ctxs_m[ds_key].n_blind),
                    signal_model=str(ctxs_m[ds_key].signal_model),
                )
                for ds_key in ctxs_m.keys()
            }

            can_combine = float(m) in accepted_masses
            if can_combine:
                w_ref = []
                for ds_key, ctx in ctxs_m.items():
                    sref = float(ctx.sigmaA_ref)
                    w_ref.append(1.0 / (sref * sref) if np.isfinite(sref) and sref > 0 else 0.0)
                w_ref_arr = np.asarray(w_ref, float)
                if np.any(np.isfinite(w_ref_arr)) and float(np.sum(w_ref_arr)) > 0:
                    Ain_arr = np.asarray([A_inj_by_ds[k] for k in ctxs_m.keys()], float)
                    sref_arr = np.asarray([ctxs_m[k].sigmaA_ref for k in ctxs_m.keys()], float)
                    z_arr = np.asarray([inj_nsigma_by_ds[k] for k in ctxs_m.keys()], float)
                    comb_strength_nom = float(np.nansum(w_ref_arr * Ain_arr) / np.sum(w_ref_arr))
                    comb_sigma_ref = float(np.nanmean(sref_arr)) if np.any(np.isfinite(sref_arr)) else float("nan")
                    comb_inj = float(np.nanmean(z_arr)) if np.any(np.isfinite(z_arr)) else float("nan")
                    acc_comb = _ToyPointAccumulator(
                        dataset="combined",
                        mass_GeV=float(m),
                        strength=float(comb_strength_nom),
                        sigmaA_ref=float(comb_sigma_ref),
                        integral_density=float("nan"),
                        A_per_eps2_unit=float("nan"),
                        f_win=float("nan"),
                        f_train_frac=float("nan"),
                    )
                else:
                    can_combine = False
                    acc_comb = None
            else:
                acc_comb = None

            done = 0
            while done < int(n_toys):
                end = min(done + aggregate_every, int(n_toys))
                toy_indices = list(range(done, end))
                rows_batch_by_ds: Dict[str, List[Dict[str, Any]]] = {}
                for ds_key, ctx in ctxs_m.items():
                    point_seed = _stable_point_seed(int(seed), str(ds_key), float(m), float(A_inj_by_ds[ds_key]))
                    rows_ds = _simulate_toy_rows_batch(
                        ctx,
                        config,
                        toy_indices=toy_indices,
                        A_inj=float(A_inj_by_ds[ds_key]),
                        inj_nsigma=float(inj_nsigma_by_ds[ds_key]),
                        point_seed=int(point_seed),
                        n_workers=int(n_workers),
                        parallel_backend=str(parallel_backend),
                        threads_per_worker=int(threads_per_worker),
                    )
                    rows_batch_by_ds[ds_key] = rows_ds
                    acc_by_ds[ds_key].update_many(rows_ds)
                    if bool(write_toy_csv):
                        toy_header_written[ds_key] = _append_toy_rows_csv(
                            toy_paths[ds_key],
                            rows_ds,
                            header_written=bool(toy_header_written[ds_key]),
                        )

                if can_combine and acc_comb is not None:
                    rows_map = {
                        ds_key: {int(r["toy"]): r for r in rows}
                        for ds_key, rows in rows_batch_by_ds.items()
                    }
                    comb_rows: List[Dict[str, Any]] = []
                    for toy_idx in toy_indices:
                        contrib = [rows_map[k].get(int(toy_idx)) for k in ctxs_m.keys() if int(toy_idx) in rows_map[k]]
                        contrib = [r for r in contrib if r is not None]
                        if mass_policy == "intersection" and len(contrib) != len(ctxs_m):
                            continue
                        if mass_policy == "union_min_n" and len(contrib) < int(min_n_contrib):
                            continue
                        comb_row = _combine_toy_rows(contrib, mass=float(m), toy_idx=int(toy_idx))
                        if comb_row is None:
                            continue
                        comb_rows.append(comb_row)
                    acc_comb.update_many(comb_rows)
                    if bool(write_toy_csv):
                        comb_toy_header_written = _append_toy_rows_csv(
                            comb_toy_path,
                            comb_rows,
                            header_written=bool(comb_toy_header_written),
                        )

                done = end
                print(
                    "[inj][stream][combined] "
                    f"m={float(m):.6g} GeV tag={float(strength_tag):.6g}: "
                    f"{done}/{int(n_toys)} toys"
                )

            for ds_key, acc in acc_by_ds.items():
                summary_rows_by_ds[ds_key].append(acc.finalize())
            if can_combine and acc_comb is not None and len(acc_comb.pull_vals):
                summary_rows_combined.append(acc_comb.finalize())

    out_by_ds = {
        k: (
            pd.DataFrame(rows).sort_values(["dataset", "mass_GeV", "strength"]).reset_index(drop=True)
            if rows else pd.DataFrame()
        )
        for k, rows in summary_rows_by_ds.items()
    }
    out_comb = (
        pd.DataFrame(summary_rows_combined).sort_values(["dataset", "mass_GeV", "strength"]).reset_index(drop=True)
        if summary_rows_combined
        else pd.DataFrame()
    )
    return out_by_ds, out_comb


def run_injection_extraction(
    ds: "DatasetConfig",
    masses: List[float],
    strengths: List[int],
    config: "Config",
    outdir: str = None,
    seed: int = 314159,
) -> pd.DataFrame:
    """Legacy single-toy injection/extraction (kept for back-compat)."""
    return run_injection_extraction_toys(
        ds, config, masses=masses, strengths=[float(s) for s in strengths],
        n_toys=1, outdir=outdir, seed=seed, strengths_mode="absolute",
    )






def _combined_mass_support_summary(df_toys_by_dataset: Dict[str, pd.DataFrame], *, mass_policy: str = "intersection", min_n_contrib: int = 2) -> Dict[str, Any]:
    """Compute accepted/rejected combined mass support under the configured policy."""
    dataset_mass_sets: Dict[str, set] = {}
    for key, df in (df_toys_by_dataset or {}).items():
        if df is None or df.empty or "mass_GeV" not in df.columns:
            continue
        masses = pd.to_numeric(df["mass_GeV"], errors="coerce").to_numpy(float)
        mass_set = {float(m) for m in masses[np.isfinite(masses)]}
        if mass_set:
            dataset_mass_sets[str(key)] = mass_set

    if not dataset_mass_sets:
        return {
            "mass_policy": str(mass_policy),
            "min_n_contrib": int(min_n_contrib),
            "dataset_mass_sets": {},
            "all_masses": set(),
            "accepted_masses": set(),
            "rejected_masses": set(),
            "n_contrib_by_mass": {},
            "accepted_count": 0,
            "rejected_count": 0,
        }

    all_masses = set().union(*dataset_mass_sets.values())
    n_contrib_by_mass = {m: sum(m in mset for mset in dataset_mass_sets.values()) for m in all_masses}

    pol = str(mass_policy or "intersection").strip().lower()
    if pol == "union_min_n":
        accepted_masses = {m for m, n in n_contrib_by_mass.items() if int(n) >= int(min_n_contrib)}
    else:
        accepted_masses = set.intersection(*dataset_mass_sets.values())

    rejected_masses = all_masses - accepted_masses
    return {
        "mass_policy": pol,
        "min_n_contrib": int(min_n_contrib),
        "dataset_mass_sets": dataset_mass_sets,
        "all_masses": all_masses,
        "accepted_masses": accepted_masses,
        "rejected_masses": rejected_masses,
        "n_contrib_by_mass": n_contrib_by_mass,
        "accepted_count": len(accepted_masses),
        "rejected_count": len(rejected_masses),
    }


def format_combined_mass_support_summary(summary: Dict[str, Any]) -> str:
    """Format one-line summary for combined mass-support acceptance."""
    pol = str(summary.get("mass_policy", "intersection"))
    min_n = int(summary.get("min_n_contrib", 2))
    acc = int(summary.get("accepted_count", 0))
    rej = int(summary.get("rejected_count", 0))
    if pol == "union_min_n":
        return f"[inj] combined mass support: accepted={acc} rejected={rej} policy=union_min_n(min_n_contrib>={min_n})"
    return f"[inj] combined mass support: accepted={acc} rejected={rej} policy=intersection"

def combine_injection_toy_tables(
    df_toys_by_dataset: Dict[str, pd.DataFrame],
    *,
    mass_policy: str = "intersection",
    min_n_contrib: int = 2,
) -> pd.DataFrame:
    """Build toy-level combined injection results from per-dataset toy tables.

    Combines available datasets at each toy point using inverse-variance
    weighting of extracted amplitudes:

      Ahat_comb = sum_i (Ahat_i / sigma_i^2) / sum_i (1/sigma_i^2)

    with sigma_comb = 1 / sqrt(sum_i 1/sigma_i^2).

    Grouping mode is selected from available columns:
      - absolute mode: (mass_GeV, strength, toy)
      - sigma-scaled mode (when finite `inj_nsigma` is present and strengths are
        inconsistent across datasets at fixed (mass, inj_nsigma, toy)):
        (mass_GeV, inj_nsigma_key, toy), where inj_nsigma_key is rounded to avoid
        floating-point mismatch across files.

    In sigma-scaled mode, output `strength` is defined as the inverse-variance
    weighted mean of input injected amplitudes for each combined group.

    Combined masses are filtered by `mass_policy`:
      - "intersection" (default): mass must be present in every enabled dataset.
      - "union_min_n": mass may be present in any dataset but must have at least
        `min_n_contrib` contributing datasets.
    """
    frames = []
    for key, df in (df_toys_by_dataset or {}).items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["dataset"] = str(key)
        frames.append(d)
    if not frames:
        return pd.DataFrame()

    n_expected_datasets = len(frames)

    all_df = pd.concat(frames, ignore_index=True)
    support = _combined_mass_support_summary(
        df_toys_by_dataset,
        mass_policy=mass_policy,
        min_n_contrib=min_n_contrib,
    )
    accepted_masses = support["accepted_masses"]

    print("[inj] combine_injection_toy_tables: begin aggregation")
    t0 = pd.Timestamp.now()

    valid = all_df[
        all_df["mass_GeV"].isin(accepted_masses)
        & np.isfinite(pd.to_numeric(all_df["sigma_A"], errors="coerce").to_numpy(float))
        & (pd.to_numeric(all_df["sigma_A"], errors="coerce").to_numpy(float) > 0)
    ].copy()
    if valid.empty:
        print("[inj] combine_injection_toy_tables: no valid rows after mass/sigma filtering")
        return pd.DataFrame()

    valid["w"] = 1.0 / np.square(valid["sigma_A"].to_numpy(float))
    valid["wA"] = valid["w"].to_numpy(float) * valid["A_hat"].to_numpy(float)
    valid["dataset"] = valid["dataset"].astype(str)
    if "strength" in valid.columns:
        valid["wS"] = valid["w"].to_numpy(float) * pd.to_numeric(valid["strength"], errors="coerce").to_numpy(float)
    else:
        valid["wS"] = np.nan

    inj_vals = pd.to_numeric(valid.get("inj_nsigma", np.nan), errors="coerce").to_numpy(float)
    has_finite_inj_nsigma = bool(np.any(np.isfinite(inj_vals)))
    sigma_group_col = "inj_nsigma_key"
    valid[sigma_group_col] = np.round(inj_vals, 8)

    sigma_scaled_mode = False
    if has_finite_inj_nsigma and "strength" in valid.columns:
        probe = (
            valid[np.isfinite(valid[sigma_group_col])]
            .groupby(["mass_GeV", sigma_group_col, "toy"], dropna=False)["strength"]
            .nunique(dropna=True)
        )
        sigma_scaled_mode = bool((probe > 1).any())

    grp_cols = ["mass_GeV", sigma_group_col, "toy"] if sigma_scaled_mode else ["mass_GeV", "strength", "toy"]
    agg = (
        valid.groupby(grp_cols, dropna=False)
        .agg(
            sum_w=("w", "sum"),
            sum_wA=("wA", "sum"),
            sum_wS=("wS", "sum"),
            inj_nsigma=("inj_nsigma", "mean"),
            sigmaA_ref=("sigmaA_ref", "mean"),
            success=("success", "all"),
            n_contrib=("dataset", "nunique"),
            contrib_datasets=("dataset", lambda s: "+".join(sorted(set(s)))),
        )
        .reset_index()
    )

    n_groups_before = len(agg)
    if support["mass_policy"] == "intersection":
        agg = agg[agg["n_contrib"] == int(n_expected_datasets)]
    elif support["mass_policy"] == "union_min_n":
        agg = agg[agg["n_contrib"] >= int(min_n_contrib)]
    n_groups_after = len(agg)
    print(
        "[inj] combine_injection_toy_tables: "
        f"dropped {n_groups_before - n_groups_after} groups due to missing contributors "
        f"(policy={support['mass_policy']}, expected_datasets={n_expected_datasets}, min_n_contrib={int(min_n_contrib)})"
    )

    if agg.empty:
        print("[inj] combine_injection_toy_tables: no groups after contribution filtering")
        return pd.DataFrame()

    agg["A_hat"] = agg["sum_wA"] / agg["sum_w"]
    agg["sigma_A"] = 1.0 / np.sqrt(agg["sum_w"])
    agg["strength"] = agg["sum_wS"] / agg["sum_w"]
    if sigma_scaled_mode:
        agg["inj_nsigma"] = agg[sigma_group_col]
    agg["pull_param"] = (agg["A_hat"] - agg["strength"]) / agg["sigma_A"]
    agg["Zhat"] = agg["A_hat"] / agg["sigma_A"]
    agg["dataset"] = "combined"
    agg["toy"] = agg["toy"].astype(int)
    out = agg[
        [
            "dataset",
            "mass_GeV",
            "toy",
            "strength",
            "inj_nsigma",
            "sigmaA_ref",
            "A_hat",
            "sigma_A",
            "pull_param",
            "Zhat",
            "n_contrib",
            "contrib_datasets",
            "success",
        ]
    ]
    dt_s = (pd.Timestamp.now() - t0).total_seconds()
    print(f"[inj] combine_injection_toy_tables: completed aggregation in {dt_s:.3f} s for {len(out)} groups")

    if out.empty:
        return out
    sort_cols = ["mass_GeV", "inj_nsigma", "toy"] if sigma_scaled_mode else ["mass_GeV", "strength", "toy"]
    return out.sort_values(sort_cols).reset_index(drop=True)

def summarize_injection_grid(df_toys: pd.DataFrame) -> pd.DataFrame:
    """Summarize injection toys by (dataset, mass, strength)."""
    if df_toys.empty:
        return pd.DataFrame()

    def q(x, p):
        v = np.asarray(x, float)
        return float(np.nanquantile(v, p)) if np.any(np.isfinite(v)) else float("nan")

    def _mean_col(sub: pd.DataFrame, name: str) -> float:
        if name not in sub.columns:
            return float("nan")
        arr = sub[name].to_numpy(float)
        return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")

    def _first_col(sub: pd.DataFrame, name: str) -> str:
        if name not in sub.columns or sub[name].empty:
            return ""
        vals = sub[name].dropna().astype(str)
        return str(vals.iloc[0]) if len(vals) else ""

    def _mean_bool_col(sub: pd.DataFrame, name: str) -> float:
        if name not in sub.columns:
            return float("nan")
        return float(np.nanmean(sub[name].astype(bool).to_numpy(float))) if len(sub) else float("nan")

    rows = []
    for (ds, m, A), sub in df_toys.groupby(["dataset", "mass_GeV", "strength"], dropna=False):
        Ahat = sub["A_hat"].to_numpy(float)
        sigA = sub["sigma_A"].to_numpy(float)
        pull = sub["pull_param"].to_numpy(float)
        Zhat = sub["Zhat"].to_numpy(float)

        n_toys = int(len(sub))
        pull_finite = pull[np.isfinite(pull)]
        inj_nsigma_vals = sub["inj_nsigma"].to_numpy(float)
        inj_nsigma_std = float(np.nanstd(inj_nsigma_vals, ddof=1)) if np.sum(np.isfinite(inj_nsigma_vals)) > 1 else 0.0
        sigmaA_ref_mean = _mean_col(sub, "sigmaA_ref")
        pull_mean = float(np.nanmean(pull))
        zhat_mean = float(np.nanmean(Zhat))

        rows.append(dict(
            dataset=ds, mass_GeV=float(m), strength=float(A),
            n_toys=n_toys,
            inj_nsigma=float(np.nanmean(inj_nsigma_vals)),
            inj_nsigma_xerr=float(inj_nsigma_std),
            sigmaA_ref=sigmaA_ref_mean,
            integral_density=_mean_col(sub, "integral_density"),
            A_per_eps2_unit=_mean_col(sub, "A_per_eps2_unit"),
            f_win=_mean_col(sub, "f_win"),
            f_train_frac=_mean_col(sub, "f_train_frac"),
            kernel_ls_policy=_first_col(sub, "kernel_ls_policy"),
            signal_model=_first_col(sub, "signal_model"),
            kernel_ls_res_lower_factor=_mean_col(sub, "kernel_ls_res_lower_factor"),
            kernel_ls_res_upper_factor=_mean_col(sub, "kernel_ls_res_upper_factor"),
            ls_lo=_mean_col(sub, "ls_lo"),
            ls_hi=_mean_col(sub, "ls_hi"),
            ls_init=_mean_col(sub, "ls_init"),
            initial_ls_opt=_mean_col(sub, "initial_ls_opt"),
            initial_const_opt=_mean_col(sub, "initial_const_opt"),
            refit_ls_opt=_mean_col(sub, "refit_ls_opt"),
            refit_const_opt=_mean_col(sub, "refit_const_opt"),
            ls_opt=_mean_col(sub, "ls_opt"),
            const_opt=_mean_col(sub, "const_opt"),
            n_train=int(round(_mean_col(sub, "n_train"))) if np.isfinite(_mean_col(sub, "n_train")) else 0,
            n_train_low=int(round(_mean_col(sub, "n_train_low"))) if np.isfinite(_mean_col(sub, "n_train_low")) else 0,
            n_train_high=int(round(_mean_col(sub, "n_train_high"))) if np.isfinite(_mean_col(sub, "n_train_high")) else 0,
            n_blind=int(round(_mean_col(sub, "n_blind"))) if np.isfinite(_mean_col(sub, "n_blind")) else 0,
            A_hat_mean=float(np.nanmean(Ahat)),
            A_hat_std=float(np.nanstd(Ahat, ddof=1)),
            sigma_A_mean=float(np.nanmean(sigA)),
            Ahat_over_Ainj=(
                float(np.nanmean(Ahat)) / float(A)
                if np.isfinite(float(A)) and abs(float(A)) > 0
                else float("nan")
            ),
            sigmaA_over_sigmaAref=(
                float(np.nanmean(sigA)) / sigmaA_ref_mean
                if np.isfinite(sigmaA_ref_mean) and sigmaA_ref_mean > 0
                else float("nan")
            ),
            pull_mean=pull_mean,
            pull_std=float(np.nanstd(pull, ddof=1)),
            pull_std_err=float(np.nanstd(pull_finite, ddof=1) / np.sqrt(max(1, 2*(len(pull_finite)-1)))) if len(pull_finite) > 1 else float("nan"),
            pull_q16=q(pull, 0.16), pull_q84=q(pull, 0.84),
            pull_q02=q(pull, 0.025), pull_q97=q(pull, 0.975),
            cov_1sigma=float(np.nanmean(np.abs(pull) < 1.0)),
            cov_2sigma=float(np.nanmean(np.abs(pull) < 2.0)),
            Zhat_mean=zhat_mean,
            delta_z_minus_pull=float((zhat_mean - float(np.nanmean(inj_nsigma_vals))) - pull_mean),
            ainj_over_sigmaAref=(float(A) / sigmaA_ref_mean if np.isfinite(sigmaA_ref_mean) and sigmaA_ref_mean > 0 else float("nan")),
            Zhat_q16=q(Zhat, 0.16), Zhat_q84=q(Zhat, 0.84),
            qmu_tilde_mean=_mean_col(sub, "qmu_tilde"),
            qmu_tilde_q16=q(sub["qmu_tilde"].to_numpy(float), 0.16) if "qmu_tilde" in sub.columns else float("nan"),
            qmu_tilde_q84=q(sub["qmu_tilde"].to_numpy(float), 0.84) if "qmu_tilde" in sub.columns else float("nan"),
            n_source_toys=int(sub["toy_index"].nunique()) if "toy_index" in sub.columns else 0,
            source_model=_first_col(sub, "source_model"),
            source_label=_first_col(sub, "source_label"),
            source_root=_first_col(sub, "source_root"),
            container=_first_col(sub, "container"),
            function_tag=_first_col(sub, "function_tag"),
            success_rate=_mean_col(sub, "success"),
            refit_ok_rate=_mean_col(sub, "refit_ok"),
            refit_fallback_rate=_mean_bool_col(sub, "refit_fallback_used"),
        ))

    return pd.DataFrame(rows).sort_values(["dataset", "mass_GeV", "strength"]).reset_index(drop=True)


def collapse_fragmented_injection_summary(df_sum: pd.DataFrame) -> pd.DataFrame:
    """Collapse one-toy summary fragments into ordinary injection summary rows.

    Distributed refit studies sometimes concatenate many ``inj_extract_summary``
    files produced with one toy per job.  Those rows are summary-shaped, but
    statistically they are toy-level fragments: ``pull_mean`` and ``Zhat_mean``
    are single-toy values and ``strength`` can jitter when sigma-scaled
    injections use a stochastic ``sigmaA_ref``.  Collapse by injected sigma level
    so downstream plots see one row per (dataset, mass, injection level).
    """
    if df_sum is None or df_sum.empty:
        return pd.DataFrame() if df_sum is None else df_sum
    if "n_toys" not in df_sum.columns:
        return df_sum

    work = df_sum.copy()
    if "dataset" not in work.columns:
        work["dataset"] = "all"
    if "mass_GeV" not in work.columns:
        return work
    work["mass_GeV"] = pd.to_numeric(work["mass_GeV"], errors="coerce")
    work["_mass_key"] = np.round(work["mass_GeV"].to_numpy(float), 10)
    internal_cols = ["_mass_key", "_inj_key", "_strength_key"]
    if "inj_nsigma" in work.columns:
        work["inj_nsigma"] = pd.to_numeric(work["inj_nsigma"], errors="coerce")
        work["_inj_key"] = np.round(work["inj_nsigma"].to_numpy(float), 10)
        gcols = ["dataset", "_mass_key", "_inj_key"]
        sort_cols = ["dataset", "mass_GeV", "inj_nsigma"]
    elif "strength" in work.columns:
        work["strength"] = pd.to_numeric(work["strength"], errors="coerce")
        work["_strength_key"] = np.round(work["strength"].to_numpy(float), 10)
        gcols = ["dataset", "_mass_key", "_strength_key"]
        sort_cols = ["dataset", "mass_GeV", "strength"]
    else:
        return work.drop(columns=[c for c in internal_cols if c in work.columns])

    n_toys_vals = pd.to_numeric(work["n_toys"], errors="coerce")
    grouped_sizes = work.groupby(gcols, dropna=False).size()
    repeated_groups = bool((grouped_sizes > 1).any())
    if not repeated_groups or not np.any(np.isfinite(n_toys_vals.to_numpy(float))):
        out = work.copy()
        out["mass_GeV"] = np.round(pd.to_numeric(out["mass_GeV"], errors="coerce").to_numpy(float), 10)
        if "inj_nsigma" in out.columns:
            out["inj_nsigma"] = np.round(pd.to_numeric(out["inj_nsigma"], errors="coerce").to_numpy(float), 10)
        return out.drop(columns=[c for c in internal_cols if c in out.columns])

    def _mean_col(sub: pd.DataFrame, name: str) -> float:
        if name not in sub.columns:
            return float("nan")
        arr = pd.to_numeric(sub[name], errors="coerce").to_numpy(float)
        return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")

    def _sum_col(sub: pd.DataFrame, name: str) -> float:
        if name not in sub.columns:
            return float("nan")
        arr = pd.to_numeric(sub[name], errors="coerce").to_numpy(float)
        return float(np.nansum(arr)) if np.any(np.isfinite(arr)) else float("nan")

    def _first_col(sub: pd.DataFrame, name: str, default: Any = "") -> Any:
        if name not in sub.columns or sub[name].empty:
            return default
        vals = sub[name].dropna()
        return vals.iloc[0] if len(vals) else default

    def _q(values: np.ndarray, p: float) -> float:
        arr = np.asarray(values, float)
        return float(np.nanquantile(arr, p)) if np.any(np.isfinite(arr)) else float("nan")

    def _std(values: np.ndarray) -> float:
        arr = np.asarray(values, float)
        arr = arr[np.isfinite(arr)]
        return float(np.nanstd(arr, ddof=1)) if arr.size > 1 else float("nan")

    passthrough: List[pd.DataFrame] = []
    rows: List[Dict[str, Any]] = []

    for key, sub in work.groupby(gcols, dropna=False):
        sub = sub.copy()
        nt = pd.to_numeric(sub["n_toys"], errors="coerce").to_numpy(float)
        is_fragment_group = len(sub) > 1 and np.any(np.isfinite(nt)) and bool(np.nanmax(nt) <= 1.0)
        if not is_fragment_group:
            passthrough.append(sub)
            continue

        pulls = pd.to_numeric(sub.get("pull_mean", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(float)
        zhats = pd.to_numeric(sub.get("Zhat_mean", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(float)
        ahats = pd.to_numeric(sub.get("A_hat_mean", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(float)
        sigas = pd.to_numeric(sub.get("sigma_A_mean", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(float)
        inj_vals = pd.to_numeric(sub.get("inj_nsigma", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(float)
        inj_level = float(np.nanmean(inj_vals)) if np.any(np.isfinite(inj_vals)) else float("nan")
        strength = _mean_col(sub, "strength")
        sigma_ref = _mean_col(sub, "sigmaA_ref")
        zhat_mean = float(np.nanmean(zhats)) if np.any(np.isfinite(zhats)) else float("nan")
        pull_mean = float(np.nanmean(pulls)) if np.any(np.isfinite(pulls)) else float("nan")
        n_frag = int(len(sub))

        row: Dict[str, Any] = dict(
            dataset=str(_first_col(sub, "dataset", key[0] if isinstance(key, tuple) else "all")),
            mass_GeV=float(_mean_col(sub, "mass_GeV")),
            strength=float(strength),
            n_toys=n_frag,
            inj_nsigma=inj_level,
            inj_nsigma_xerr=_std(inj_vals) if np.any(np.isfinite(inj_vals)) else float("nan"),
            sigmaA_ref=float(sigma_ref),
            integral_density=_mean_col(sub, "integral_density"),
            A_per_eps2_unit=_mean_col(sub, "A_per_eps2_unit"),
            f_win=_mean_col(sub, "f_win"),
            f_train_frac=_mean_col(sub, "f_train_frac"),
            A_hat_mean=float(np.nanmean(ahats)) if np.any(np.isfinite(ahats)) else float("nan"),
            A_hat_std=_std(ahats),
            sigma_A_mean=float(np.nanmean(sigas)) if np.any(np.isfinite(sigas)) else float("nan"),
            pull_mean=pull_mean,
            pull_std=_std(pulls),
            pull_std_err=(
                _std(pulls) / np.sqrt(max(1, 2 * (n_frag - 1)))
                if n_frag > 2 and np.isfinite(_std(pulls))
                else float("nan")
            ),
            pull_q16=_q(pulls, 0.16),
            pull_q84=_q(pulls, 0.84),
            pull_q02=_q(pulls, 0.025),
            pull_q97=_q(pulls, 0.975),
            cov_1sigma=_mean_col(sub, "cov_1sigma"),
            cov_2sigma=_mean_col(sub, "cov_2sigma"),
            Zhat_mean=zhat_mean,
            delta_z_minus_pull=(
                float((zhat_mean - inj_level) - pull_mean)
                if np.isfinite(zhat_mean) and np.isfinite(inj_level) and np.isfinite(pull_mean)
                else float("nan")
            ),
            ainj_over_sigmaAref=(
                float(strength) / float(sigma_ref)
                if np.isfinite(strength) and np.isfinite(sigma_ref) and sigma_ref > 0
                else inj_level
            ),
            Zhat_q16=_q(zhats, 0.16),
            Zhat_q84=_q(zhats, 0.84),
            success_rate=_mean_col(sub, "success_rate"),
        )

        for col in [
            "kernel_ls_res_lower_factor",
            "kernel_ls_res_upper_factor",
            "ls_lo",
            "ls_hi",
            "ls_init",
            "initial_ls_opt",
            "initial_const_opt",
            "refit_ls_opt",
            "refit_const_opt",
            "ls_opt",
            "const_opt",
            "refit_ok_rate",
            "refit_fallback_rate",
        ]:
            if col in sub.columns:
                row[col] = _mean_col(sub, col)
        if "kernel_ls_policy" in sub.columns:
            row["kernel_ls_policy"] = str(_first_col(sub, "kernel_ls_policy", ""))
        if "signal_model" in sub.columns:
            row["signal_model"] = str(_first_col(sub, "signal_model", ""))
        for col in ["n_train", "n_train_low", "n_train_high", "n_blind"]:
            if col in sub.columns:
                val = _mean_col(sub, col)
                row[col] = int(round(val)) if np.isfinite(val) else 0

        rows.append(row)

    out_parts = passthrough[:]
    if rows:
        out_parts.append(pd.DataFrame(rows))
    if not out_parts:
        return work
    out = pd.concat(out_parts, ignore_index=True, sort=False)
    out["mass_GeV"] = np.round(pd.to_numeric(out["mass_GeV"], errors="coerce").to_numpy(float), 10)
    if "inj_nsigma" in out.columns:
        out["inj_nsigma"] = np.round(pd.to_numeric(out["inj_nsigma"], errors="coerce").to_numpy(float), 10)
    out = out.drop(columns=[c for c in internal_cols if c in out.columns])
    return out.sort_values([c for c in sort_cols if c in out.columns]).reset_index(drop=True)
