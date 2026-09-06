"""Histogram loading and per-dataset background estimation."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from .gpr import (
    fit_gpr,
    predict_counts_from_log_gpr,
    predict_counts_mean_from_log_gpr,
    make_kernel_for_dataset,
    compute_kernel_ls_bounds,
    _extract_rbf_bounds_and_scale,
)

if TYPE_CHECKING:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from .config import Config
    from .dataset import DatasetConfig


@dataclass
class BlindPrediction:
    """Background prediction results for a blind window."""

    mu: np.ndarray  # Background mean in blind window
    cov: np.ndarray  # Background covariance in blind window
    obs: np.ndarray  # Observed counts in blind window
    edges: np.ndarray  # Bin edges in blind window
    sigma_val: float  # Mass resolution
    blind: Tuple[float, float]  # Blind window bounds

    x_full: np.ndarray  # All bin centers
    y_full: np.ndarray  # All observed counts
    mu_full: np.ndarray  # Background prediction for all bins
    edges_full: np.ndarray  # All bin edges
    blind_mask: np.ndarray  # Mask of the blind window in the full binning

    integral_density: float  # Counts per GeV in signal region
    density_nsigma: float = float("nan")
    density_window_lo: float = float("nan")
    density_window_hi: float = float("nan")
    density_window_width: float = float("nan")
    density_source_lo: float = float("nan")
    density_source_hi: float = float("nan")
    density_source_n_bins: int = 0
    density_source_bin_width_median: float = float("nan")
    density_window_fully_covered: bool = False
    blind_train: Optional[Tuple[float, float]] = None  # GP training exclusion window

    # GP/kernel diagnostics for summary CSV/plots
    kernel_str: str = ""
    ls_lo: float = float("nan")
    ls_hi: float = float("nan")
    ls_init: float = float("nan")
    ls_opt: float = float("nan")
    sigma_x: float = float("nan")
    const_opt: float = float("nan")
    lml: float = float("nan")
    n_train: int = 0
    n_train_low: int = 0
    n_train_high: int = 0
    n_full: int = 0
    n_blind: int = 0
    train_domain_lo: float = float("nan")
    train_domain_hi: float = float("nan")
    bin_width_median: float = float("nan")
    const_init: float = float("nan")
    const_lo: float = float("nan")
    const_hi: float = float("nan")
    const_at_lower: bool = False
    const_at_upper: bool = False
    ls_at_lower: bool = False
    ls_at_upper: bool = False
    optimizer_restarts: int = 0
    optimizer_random_state: int = -1
    optimizer_warning_count: int = 0
    optimizer_warnings: str = ""


def _extract_constant_bounds_and_value(kernel) -> Tuple[float, float, float]:
    """Return ConstantKernel (lower, upper, value), if present."""
    try:
        const = kernel.k1 if hasattr(kernel, "k1") else kernel
        value = float(getattr(const, "constant_value"))
        bounds = getattr(const, "constant_value_bounds", None)
        if bounds is None or (isinstance(bounds, str) and bounds == "fixed"):
            return value, value, value
        lo, hi = np.asarray(bounds, dtype=float).reshape(-1)[:2]
        return float(lo), float(hi), value
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _at_kernel_bound(value: float, bound: float) -> bool:
    """Use the scan convention: within 0.1% of a positive kernel bound."""
    if not np.isfinite(value) or not np.isfinite(bound) or bound <= 0.0:
        return False
    return bool(np.isclose(float(value), float(bound), rtol=1.0e-3, atol=1.0e-12))


def _gp_model(h, kernel, **kwargs):
    """Compatibility wrapper for gp.GaussianProcessModel.

    Some versions of the local gp package require kernel as a mandatory
    (possibly positional) argument.
    """
    import gp

    # Try keyword kernel + keyword h
    try:
        return gp.GaussianProcessModel(h=h, kernel=kernel, **kwargs)
    except TypeError as e1:
        # Try positional kernel, keyword h
        try:
            return gp.GaussianProcessModel(kernel, h=h, **kwargs)
        except TypeError:
            # Try positional kernel, positional h
            try:
                return gp.GaussianProcessModel(kernel, h, **kwargs)
            except TypeError:
                raise e1


def _build_model(
    ds: "DatasetConfig",
    blind: Tuple[float, float],
    rebin: int,
    config: "Config",
    *,
    mass: Optional[float] = None,
):
    """Prepare the rebinned/limited histogram used by the modern sklearn path."""
    import gp

    hist_source = ds.hist_override if getattr(ds, "hist_override", None) is not None else (ds.root_path, ds.hist_name)
    histogram = gp._hist.io._deduce_histogram(hist_source)

    del blind, config, mass

    edges_all = np.asarray(histogram.axes[0].edges, float)
    first_edge = float(edges_all[0])
    last_edge = float(edges_all[-1])

    # Prefer explicit data-range overrides for training/fits; otherwise use
    # full histogram extent to avoid edge instabilities near the search range.
    data_lo = getattr(ds, "data_low", None)
    data_hi = getattr(ds, "data_high", None)
    lower = max(first_edge, float(data_lo)) if data_lo is not None else first_edge
    upper = min(last_edge, float(data_hi)) if data_hi is not None else last_edge

    manip = gp._hist.manipulation.rebin_and_limit(int(rebin), lower, upper)
    return SimpleNamespace(
        histogram=manip(histogram),
        density_histogram=histogram,
    )


def _blind_pred_detail(
    model,
    gpr: "GaussianProcessRegressor",
    blind: Tuple[float, float],
    config: "Config",
):
    """Extract prediction details for the blind window."""
    Xc = model.histogram.axes[0].centers
    vals = model.histogram.values().astype(int)
    edges = np.asarray(model.histogram.axes[0].edges, dtype=float)

    msk = (Xc >= blind[0]) & (Xc <= blind[1])
    idx = np.where(msk)[0]
    if idx.size == 0:
        raise RuntimeError("Blind window has no bins")

    Xb = Xc[msk]
    e_slice = edges[idx[0] : idx[-1] + 2]

    mu, cov = predict_counts_from_log_gpr(gpr, Xb, config)
    obs = vals[msk]

    return (
        np.asarray(mu, float),
        np.asarray(cov, float),
        np.asarray(obs, int),
        np.asarray(e_slice, float),
    )


def _compute_integral_density(
    model,
    mass: float,
    sigma_val: float,
    *,
    density_nsigma: float,
    return_metadata: bool = False,
):
    """Compute density in the exact physical window using the uncropped histogram.

    Boundary bins are weighted by their fractional overlap with
    ``mass +/- density_nsigma * sigma_val``.  The conversion must never inherit
    the cropped/rebinned GP support, and a source histogram that does not cover
    the requested physical interval is rejected rather than silently clipped.
    """
    density_histogram = getattr(model, "density_histogram", None)
    if density_histogram is None:
        raise ValueError(
            "Density computation requires an uncropped density_histogram"
        )

    nsigma = float(density_nsigma)
    sigma = float(sigma_val)
    center = float(mass)
    if not np.isfinite(nsigma) or nsigma <= 0.0:
        raise ValueError("density_nsigma must be finite and positive")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_val must be finite and positive")
    if not np.isfinite(center):
        raise ValueError("mass must be finite")

    half_width = nsigma * sigma
    lo, hi = center - half_width, center + half_width
    window_width = hi - lo

    ax = density_histogram.axes[0]
    edges = np.asarray(ax.edges, dtype=float)
    vals = np.asarray(density_histogram.values(), dtype=float)
    widths = np.diff(edges)
    if (
        edges.ndim != 1
        or vals.ndim != 1
        or len(edges) != len(vals) + 1
        or len(vals) == 0
    ):
        raise ValueError("Density source must be a non-empty one-dimensional histogram")
    if (
        not np.all(np.isfinite(edges))
        or not np.all(np.isfinite(widths))
        or np.any(widths <= 0.0)
    ):
        raise ValueError("Density source has invalid bin edges")

    source_lo = float(edges[0])
    source_hi = float(edges[-1])
    coverage_tolerance = 32.0 * np.finfo(float).eps * max(
        1.0, abs(source_lo), abs(source_hi), abs(lo), abs(hi)
    )
    fully_covered = bool(
        lo >= source_lo - coverage_tolerance
        and hi <= source_hi + coverage_tolerance
    )
    if not fully_covered:
        raise ValueError(
            "Density source does not fully cover physical window "
            f"[{lo:.12g}, {hi:.12g}] GeV; source is "
            f"[{source_lo:.12g}, {source_hi:.12g}] GeV"
        )

    overlap = np.maximum(
        0.0,
        np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo),
    )
    covered_width = float(np.sum(overlap))
    if not np.isclose(
        covered_width,
        window_width,
        rtol=1.0e-12,
        atol=coverage_tolerance,
    ):
        raise ValueError(
            "Density histogram binning does not cover the full physical window"
        )

    integral_counts = float(np.sum(vals * (overlap / widths)))
    if not np.isfinite(integral_counts):
        raise ValueError("Density source produced non-finite integral counts")

    density = float(integral_counts / window_width)
    metadata = {
        "density_nsigma": nsigma,
        "density_window_lo": float(lo),
        "density_window_hi": float(hi),
        "density_window_width": float(window_width),
        "density_source_lo": source_lo,
        "density_source_hi": source_hi,
        "density_source_n_bins": int(len(vals)),
        "density_source_bin_width_median": float(np.median(widths)),
        "density_window_fully_covered": fully_covered,
    }
    if return_metadata:
        return density, metadata
    return density


def estimate_background_for_dataset(
    ds: "DatasetConfig",
    mass: float,
    config: "Config",
    rebin: int = None,
    restarts: int = None,
    train_exclude_nsigma: Optional[float] = None,
    *,
    kernel=None,
    optimize: bool = True,
) -> BlindPrediction:
    """Estimate background for a dataset at a given mass.

    Args:
        ds: Dataset configuration
        mass: Signal mass hypothesis (GeV)
        config: Global configuration
        rebin: Rebinning factor (defaults to config.neighborhood_rebin)
        restarts: Number of GPR restarts (defaults to config.n_restarts)
        train_exclude_nsigma: Half-width of GP training exclusion in sigma units.
            Defaults to config.gp_train_exclude_nsigma (or config.blind_nsigma if
            gp_train_exclude_nsigma is None). The extraction blind window always uses
            config.blind_nsigma; only the GP training mask is affected.
        kernel: Optional explicit kernel. This is used to reconstruct a reviewed
            observed GP state for conditional ensembles.
        optimize: If False, keep the explicit kernel hyperparameters fixed.

    Returns:
        BlindPrediction with background estimates
    """
    if rebin is None:
        rebin = config.neighborhood_rebin
    if restarts is None:
        restarts = config.n_restarts

    sigma_val = ds.sigma(mass)
    blind = (
        mass - config.blind_nsigma * sigma_val,
        mass + config.blind_nsigma * sigma_val,
    )

    # GP training exclusion window (may differ from extraction blind window)
    if train_exclude_nsigma is None:
        train_exclude_nsigma = float(
            getattr(config, "gp_train_exclude_nsigma", None) or config.blind_nsigma
        )
    blind_train = (
        mass - float(train_exclude_nsigma) * sigma_val,
        mass + float(train_exclude_nsigma) * sigma_val,
    )

    model = _build_model(ds, blind, rebin=rebin, config=config, mass=float(mass))

    X = model.histogram.axes[0].centers
    y = model.histogram.values().astype(float)

    mask_train = (X < blind_train[0]) | (X > blind_train[1])
    X_train = X[mask_train]
    y_train = y[mask_train]

    kernel_used = (
        make_kernel_for_dataset(ds, config, mass=float(mass))
        if kernel is None
        else kernel
    )
    gpr = fit_gpr(
        X_train,
        y_train,
        config,
        restarts=int(restarts),
        kernel=kernel_used,
        optimize=bool(optimize),
    )

    mu_blind, cov_blind, obs_blind, edges_blind = _blind_pred_detail(
        model, gpr, blind, config
    )

    mu_full = predict_counts_mean_from_log_gpr(gpr, X, config)

    density_nsigma = float(
        getattr(config, "eps2_density_nsigma", None) or config.blind_nsigma
    )
    integral_density, density_metadata = _compute_integral_density(
        model,
        mass,
        sigma_val,
        density_nsigma=density_nsigma,
        return_metadata=True,
    )

    # Kernel diagnostics for scan summary outputs. Read the actual kernel used
    # by sklearn so explicit per-dataset overrides cannot be misreported.
    kernel_initial = getattr(gpr, "kernel", None)
    ls_lo, ls_hi, ls_init = _extract_rbf_bounds_and_scale(kernel_initial)
    sigma_x = float("nan")
    try:
        ls_info = compute_kernel_ls_bounds(ds, config, mass=float(mass))
        sigma_x = float(ls_info.get("sigma_x", float("nan")))
    except Exception:
        pass

    const_lo, const_hi, const_init = _extract_constant_bounds_and_value(
        kernel_initial
    )
    const_opt = float("nan")
    ls_opt = float("nan")
    lml = float("nan")
    try:
        kopt = getattr(gpr, "kernel_", None)
        if kopt is not None and hasattr(kopt, "k1"):
            const_opt = float(getattr(kopt.k1, "constant_value", float("nan")))
        _, _, ls_opt = _extract_rbf_bounds_and_scale(
            kopt if kopt is not None else gpr.kernel
        )
    except Exception:
        pass

    try:
        lml = float(gpr.log_marginal_likelihood_value_)
    except Exception:
        pass

    edges_full = np.asarray(model.histogram.axes[0].edges, float)
    widths_full = np.diff(edges_full)
    blind_mask = np.asarray((X >= blind[0]) & (X <= blind[1]), bool)
    n_train_low = int(np.count_nonzero(X < blind_train[0]))
    n_train_high = int(np.count_nonzero(X > blind_train[1]))

    return BlindPrediction(
        mu=mu_blind,
        cov=cov_blind,
        obs=obs_blind,
        edges=edges_blind,
        sigma_val=sigma_val,
        blind=blind,
        x_full=np.asarray(X, float),
        y_full=np.asarray(y, float),
        mu_full=np.asarray(mu_full, float),
        edges_full=edges_full,
        blind_mask=blind_mask,
        integral_density=integral_density,
        **density_metadata,
        blind_train=blind_train,
        kernel_str=str(getattr(gpr, "kernel_", "")),
        ls_lo=ls_lo,
        ls_hi=ls_hi,
        ls_init=ls_init,
        ls_opt=float(ls_opt),
        sigma_x=sigma_x,
        const_opt=const_opt,
        lml=lml,
        n_train=int(np.count_nonzero(mask_train)),
        n_train_low=n_train_low,
        n_train_high=n_train_high,
        n_full=int(len(X)),
        n_blind=int(np.count_nonzero(blind_mask)),
        train_domain_lo=float(edges_full[0]) if edges_full.size else float("nan"),
        train_domain_hi=float(edges_full[-1]) if edges_full.size else float("nan"),
        bin_width_median=(
            float(np.median(widths_full)) if widths_full.size else float("nan")
        ),
        const_init=const_init,
        const_lo=const_lo,
        const_hi=const_hi,
        const_at_lower=_at_kernel_bound(const_opt, const_lo),
        const_at_upper=_at_kernel_bound(const_opt, const_hi),
        ls_at_lower=_at_kernel_bound(ls_opt, ls_lo),
        ls_at_upper=_at_kernel_bound(ls_opt, ls_hi),
        optimizer_restarts=int(restarts) if optimize else 0,
        optimizer_random_state=int(
            getattr(gpr, "_hps_optimizer_random_state", -1)
            if getattr(gpr, "_hps_optimizer_random_state", None) is not None
            else -1
        ),
        optimizer_warning_count=len(getattr(gpr, "_hps_optimizer_warnings", ())),
        optimizer_warnings=" | ".join(getattr(gpr, "_hps_optimizer_warnings", ())),
    )
