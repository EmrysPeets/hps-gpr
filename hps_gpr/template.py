"""Signal template and CLs calculations."""

import math
from math import erf, sqrt
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .statistics import (
    asymptotic_cls_profiled_gaussian,
    toy_cls_profiled_gaussian,
)

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


def gaussian_bin_integrals(
    edges: np.ndarray, m0: float, sigma: float
) -> np.ndarray:
    """Compute Gaussian CDF integrals over bins.

    Args:
        edges: Bin edges
        m0: Gaussian mean
        sigma: Gaussian width

    Returns:
        Array of integrals for each bin
    """
    e = np.asarray(edges, dtype=float)
    z = (e - m0) / (sqrt(2.0) * float(sigma))
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z))
    integ = np.diff(cdf)
    return np.clip(integ, 0.0, None)


def normalize_template(w: np.ndarray) -> np.ndarray:
    """Normalize a template to sum to 1."""
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        return np.full_like(w, 1.0 / max(1, w.size))
    return w / s


def normalize_signal_model(signal_model: str) -> str:
    """Normalize the configured signal model name."""
    mode = str(signal_model or "default").lower().strip()
    if mode in ("default", "gaussian", "template"):
        return "default"
    if mode in ("kernel", "signal_kernel", "gp_signal_kernel"):
        return "kernel"
    raise ValueError(f"Unknown signal_model={signal_model!r}; expected 'default' or 'kernel'.")


def signal_model_from_config(config: Optional["Config"]) -> str:
    """Return the normalized signal model configured for this analysis."""
    return normalize_signal_model(getattr(config, "signal_model", "default") if config is not None else "default")


def _signal_kernel_hyperparams(
    sigma_val: float,
    *,
    width_factor: float = 1.0,
    length_scale_factor: float = 1.0,
) -> Tuple[float, float]:
    """Return (t, ell) for the localized signal kernel."""
    sig = max(float(sigma_val), 1e-12)
    t = max(float(width_factor) * sig, 1e-12)
    ell = max(float(length_scale_factor) * sig, 1e-12)
    return t, ell


def signal_kernel_covariance(
    x: np.ndarray,
    mass: float,
    sigma_val: float,
    *,
    width_factor: float = 1.0,
    length_scale_factor: float = 1.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    r"""Localized signal covariance from Frate et al. Eq. 14.

    The kernel is
    ``A exp[-0.5 (x-x')^2/ell^2] exp[-0.5 ((x-m)^2 + (x'-m)^2)/t^2]``.
    The HPS implementation fixes the localization width ``t`` and correlation
    length ``ell`` from the mass resolution through configurable scale factors.
    """
    xv = np.asarray(x, float).reshape(-1)
    t, ell = _signal_kernel_hyperparams(
        sigma_val,
        width_factor=width_factor,
        length_scale_factor=length_scale_factor,
    )
    dx = xv[:, None] - xv[None, :]
    env = (xv - float(mass)) ** 2
    return (
        float(amplitude)
        * np.exp(-0.5 * (dx ** 2) / (ell ** 2))
        * np.exp(-0.5 * (env[:, None] + env[None, :]) / (t ** 2))
    )


def signal_kernel_bin_weights(
    edges: np.ndarray,
    mass: float,
    sigma_val: float,
    *,
    width_factor: float = 1.0,
    length_scale_factor: float = 1.0,
) -> np.ndarray:
    """Build a positive signal template from the leading signal-kernel mode."""
    e = np.asarray(edges, float).reshape(-1)
    if e.size < 2:
        return np.asarray([], float)
    centers = 0.5 * (e[:-1] + e[1:])
    widths = np.clip(np.diff(e), 0.0, None)
    K = signal_kernel_covariance(
        centers,
        mass,
        sigma_val,
        width_factor=width_factor,
        length_scale_factor=length_scale_factor,
    )
    try:
        vals, vecs = np.linalg.eigh(0.5 * (K + K.T))
        v = np.asarray(vecs[:, int(np.argmax(vals))], float)
        if np.sum(v) < 0:
            v = -v
        w = np.clip(v, 0.0, None) * widths
    except Exception:
        t, _ = _signal_kernel_hyperparams(
            sigma_val,
            width_factor=width_factor,
            length_scale_factor=length_scale_factor,
        )
        w = np.exp(-0.5 * ((centers - float(mass)) / t) ** 2) * widths
    if not np.any(np.isfinite(w)) or float(np.nansum(w)) <= 0:
        return gaussian_bin_integrals(e, mass, sigma_val)
    return np.asarray(w, float)


def build_template(
    edges: np.ndarray,
    mass: float,
    sigma_val: float,
    *,
    signal_model: str = "default",
    signal_kernel_width_factor: float = 1.0,
    signal_kernel_length_scale_factor: float = 1.0,
) -> np.ndarray:
    """Build a normalized signal template.

    Args:
        edges: Bin edges
        mass: Signal mass hypothesis
        sigma_val: Mass resolution
        signal_model: "default" Gaussian-bin template or "kernel" signal-kernel template

    Returns:
        Normalized template array
    """
    model = normalize_signal_model(signal_model)
    if model == "kernel":
        return normalize_template(
            signal_kernel_bin_weights(
                edges,
                mass,
                sigma_val,
                width_factor=float(signal_kernel_width_factor),
                length_scale_factor=float(signal_kernel_length_scale_factor),
            )
        )
    return normalize_template(gaussian_bin_integrals(edges, mass, sigma_val))


def build_full_template(
    edges_full: np.ndarray,
    mass: float,
    sigma_val: float,
    *,
    config: Optional["Config"] = None,
    signal_model: Optional[str] = None,
) -> np.ndarray:
    """Build a signal template normalized on the full histogram range."""
    model = signal_model_from_config(config) if signal_model is None else normalize_signal_model(signal_model)
    return build_template(
        edges_full,
        mass,
        sigma_val,
        signal_model=model,
        signal_kernel_width_factor=float(getattr(config, "signal_kernel_width_factor", 1.0)),
        signal_kernel_length_scale_factor=float(getattr(config, "signal_kernel_length_scale_factor", 1.0)),
    )


def slice_template_to_window(
    template_full: np.ndarray,
    window_mask: np.ndarray,
) -> np.ndarray:
    """Extract the blinded-bin slice from a full-range normalized template."""
    w = np.asarray(template_full, float).reshape(-1)
    m = np.asarray(window_mask, bool).reshape(-1)
    if w.size != m.size:
        raise ValueError(
            "slice_template_to_window requires template_full and window_mask "
            "to have the same number of bins."
        )
    return np.asarray(w[m], float)


def build_window_template_from_full(
    edges_full: np.ndarray,
    window_mask: np.ndarray,
    mass: float,
    sigma_val: float,
    *,
    config: Optional["Config"] = None,
    signal_model: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a full-range signal template and return its blinded-bin slice.

    The returned window template is not renormalized inside the blinded region.
    Its sum therefore equals the signal fraction contained inside the blind window.
    """
    w_full = build_full_template(
        edges_full,
        mass,
        sigma_val,
        config=config,
        signal_model=signal_model,
    )
    w_window = slice_template_to_window(w_full, window_mask)
    return w_window, w_full


def _safe_mvn_draw(
    mean: np.ndarray,
    cov: Optional[np.ndarray],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Safely draw from multivariate normal, with fallbacks."""
    m = np.asarray(mean, dtype=float)

    if cov is None:
        draws = np.tile(m, (size, 1))
    else:
        C = np.asarray(cov, dtype=float)
        try:
            draws = rng.multivariate_normal(
                m, C, size=size, check_valid="ignore", tol=1e-8
            )
        except Exception:
            diag = np.clip(np.diag(C), 0.0, None)
            draws = rng.normal(loc=m, scale=np.sqrt(diag), size=(size, m.size))

    return np.clip(draws, 0.0, None)


def _log_lr(
    n: np.ndarray, b: np.ndarray, s: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Compute log likelihood ratio."""
    n = np.asarray(n, dtype=float)
    b = np.asarray(b, dtype=float)
    s = np.asarray(s, dtype=float)

    if b.ndim == 1 and n.ndim > 1:
        b = np.broadcast_to(b, n.shape)

    b_eff = np.clip(b, eps, None)
    sb_eff = np.clip(b + s, eps, None)
    term = -s + n * (np.log(sb_eff) - np.log(b_eff))
    return np.sum(term, axis=-1)


def cls_amplitude_asymptotic(
    A: float,
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    cov: Optional[np.ndarray],
    template: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """Compute CLs using asymptotic bounded profile-likelihood calibration.

    Args:
        A: Signal amplitude
        n_obs: Observed counts
        b_mean: Background mean prediction
        cov: Background covariance matrix
        template: Signal template on the fitted bins. This may be the blinded-bin
            slice of a full-range normalized template, so its sum need not be 1.
        eps: Small value to prevent division by zero

    Returns:
        Tuple of (CLs, CL_sb, CL_b)
    """
    del eps
    b = np.asarray(b_mean, dtype=float)
    C = (
        np.asarray(cov, dtype=float)
        if cov is not None
        else np.zeros((b.size, b.size), dtype=float)
    )
    cls, CL_sb, CL_b, _ = asymptotic_cls_profiled_gaussian(
        float(A),
        np.asarray(n_obs, dtype=float),
        b,
        C,
        np.asarray(template, dtype=float),
    )
    return float(cls), float(CL_sb), float(CL_b)


def cls_amplitude_toys(
    A: float,
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    cov: Optional[np.ndarray],
    template: np.ndarray,
    rng: np.random.Generator,
    num_toys: int,
    floor: float = 1e-12,
) -> Tuple[float, float, float]:
    """Compute CLs using toy calibration of the bounded profile-likelihood test.

    Args:
        A: Signal amplitude
        n_obs: Observed counts
        b_mean: Background mean prediction
        cov: Background covariance matrix
        template: Signal template on the fitted bins. This may be the blinded-bin
            slice of a full-range normalized template, so its sum need not be 1.
        rng: Random number generator
        num_toys: Number of toys to generate
        floor: Small value to prevent division by zero

    Returns:
        Tuple of (CLs, CL_sb, CL_b)
    """
    del floor
    b = np.asarray(b_mean, dtype=float)
    C = (
        np.asarray(cov, dtype=float)
        if cov is not None
        else np.zeros((b.size, b.size), dtype=float)
    )
    cls, CL_sb, CL_b, _ = toy_cls_profiled_gaussian(
        float(A),
        np.asarray(n_obs, dtype=float),
        b,
        C,
        np.asarray(template, dtype=float),
        rng,
        int(num_toys),
    )
    return float(cls), float(CL_sb), float(CL_b)


def cls_limit_for_amplitude(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: Optional[np.ndarray],
    edges: np.ndarray,
    mass: float,
    sigma_val: float,
    config: "Config",
    seed: int = 1,
    *,
    full_edges: Optional[np.ndarray] = None,
    window_mask: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Find the CLs upper limit on total signal amplitude.

    Uses bisection to find the amplitude A such that CLs = alpha.

    Args:
        n_obs: Observed counts
        b_mean: Background mean prediction
        b_cov: Background covariance matrix
        edges: Bin edges for the fitted window
        mass: Signal mass hypothesis
        sigma_val: Mass resolution
        config: Global configuration
        seed: Random seed
        full_edges: Full histogram edges for the signal template normalization
        window_mask: Blind-window mask in the full histogram binning

    Returns:
        Tuple of (A_upper_limit, debug_dict), where A_upper_limit is the total local
        Gaussian signal yield associated with the supplied template definition.
    """
    if full_edges is not None and window_mask is not None:
        template, _ = build_window_template_from_full(
            np.asarray(full_edges, float),
            np.asarray(window_mask, bool),
            mass,
            sigma_val,
            config=config,
        )
    else:
        template = build_template(
            edges,
            mass,
            sigma_val,
            signal_model=signal_model_from_config(config),
            signal_kernel_width_factor=float(getattr(config, "signal_kernel_width_factor", 1.0)),
            signal_kernel_length_scale_factor=float(getattr(config, "signal_kernel_length_scale_factor", 1.0)),
        )
    rng = np.random.default_rng(seed)
    alpha = config.cls_alpha
    mode = config.cls_mode
    num_toys = config.cls_num_toys

    def cls_at(A):
        if mode == "asymptotic":
            return cls_amplitude_asymptotic(A, n_obs, b_mean, b_cov, template)[0]
        return cls_amplitude_toys(
            A, n_obs, b_mean, b_cov, template, rng, max(1, int(num_toys))
        )[0]

    b_sum = float(np.sum(b_mean))
    A_lo = 0.0
    A_hi = max(1.0, 3.0 * math.sqrt(max(b_sum, 1.0)))

    cls_hi = cls_at(A_hi)
    it = 0
    while cls_hi > alpha and A_hi < 1e7 and it < 40:
        A_hi *= 2.0
        cls_hi = cls_at(A_hi)
        it += 1

    gridA = [A_lo, A_hi]
    gridC = [cls_at(A_lo), cls_hi]

    for _ in range(40):
        Amid = 0.5 * (A_lo + A_hi)
        cls_mid = cls_at(Amid)
        gridA.append(Amid)
        gridC.append(cls_mid)

        if abs(cls_mid - alpha) < 1e-6:
            A_lo = A_hi = Amid
            break
        if cls_mid > alpha:
            A_lo = Amid
        else:
            A_hi = Amid
        if abs(A_hi - A_lo) <= max(1e-12, 1e-6 * max(abs(A_hi), abs(A_lo))):
            break

    return 0.5 * (A_lo + A_hi), {
        "A_grid": np.array(gridA),
        "CLs_grid": np.array(gridC),
    }


def cls_limit_for_template(
    n_obs: np.ndarray,
    b_mean: np.ndarray,
    b_cov: Optional[np.ndarray],
    template: np.ndarray,
    *,
    ds: Optional["DatasetConfig"] = None,
    mass: Optional[float] = None,
    integral_density: Optional[float] = None,
    alpha: float = 0.05,
    mode: str = "asymptotic",
    use_eps2: bool = False,
    num_toys: int = 100,
    seed: int = 1,
    A_hi0: Optional[float] = None,
) -> Tuple[float, float]:
    """CLs upper limit for a pre-built signal template.

    More flexible than cls_limit_for_amplitude: accepts a pre-built template
    and can optionally convert the amplitude limit to epsilon^2.

    Args:
        n_obs: Observed counts
        b_mean: Background mean prediction
        b_cov: Background covariance matrix
        template: Pre-built signal template on the fitted bins. It may be the
            blinded-bin slice of a full-range normalized template, so its sum can
            be less than 1.
        ds: Dataset config (required when use_eps2=True)
        mass: Signal mass in GeV (required when use_eps2=True)
        integral_density: Counts per GeV (required when use_eps2=True)
        alpha: CL level (default 0.05 → 95% UL)
        mode: "asymptotic" or "toys"
        use_eps2: If True, convert A_up to eps2_up and return (eps2_up, A_up)
        num_toys: Number of CLs toys (only used when mode="toys")
        seed: Random seed
        A_hi0: Initial upper bracket for bisection (auto-set if None)

    Returns:
        (limit, A_up) where limit = eps2_up if use_eps2=True else A_up, and A_up is
        the total local Gaussian signal yield associated with the template.
    """
    template = np.asarray(template, float)
    rng = np.random.default_rng(int(seed))

    def cls_at(A: float) -> float:
        if mode == "asymptotic":
            return cls_amplitude_asymptotic(float(A), n_obs, b_mean, b_cov, template)[0]
        return cls_amplitude_toys(
            float(A), n_obs, b_mean, b_cov, template, rng, max(1, int(num_toys))
        )[0]

    b_sum = float(np.sum(b_mean))
    A_lo = 0.0
    if A_hi0 is None:
        A_hi = max(1.0, 3.0 * math.sqrt(max(b_sum, 1.0)))
    else:
        A_hi = float(A_hi0)

    cls_hi = cls_at(A_hi)
    it = 0
    while cls_hi > alpha and A_hi < 1e7 and it < 40:
        A_hi *= 2.0
        cls_hi = cls_at(A_hi)
        it += 1

    for _ in range(60):
        Amid = 0.5 * (A_lo + A_hi)
        cls_mid = cls_at(Amid)
        if abs(cls_mid - alpha) < 1e-8:
            A_lo = A_hi = Amid
            break
        if cls_mid > alpha:
            A_lo = Amid
        else:
            A_hi = Amid
        if abs(A_hi - A_lo) <= max(1e-12, 1e-6 * max(abs(A_hi), abs(A_lo))):
            break

    A_up = float(0.5 * (A_lo + A_hi))

    if not bool(use_eps2):
        return A_up, A_up

    if ds is None or mass is None or integral_density is None:
        raise ValueError(
            "cls_limit_for_template(use_eps2=True) requires ds, mass, and integral_density."
        )

    from .conversion import epsilon2_from_A

    eps2_up = float(epsilon2_from_A(ds, float(mass), A_up, float(integral_density)))
    return eps2_up, A_up
