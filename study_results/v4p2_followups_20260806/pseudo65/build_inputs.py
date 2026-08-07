#!/usr/bin/env python3
"""Build the two conditional 65 MeV background-only replacement spectra.

Only the native 2021 bins in [60, 70) MeV are replaced.  The two lanes are:

1. independent Poisson counts around the exact accepted v4.2 fixed-GP mean;
2. independent Poisson counts around a sideband-only fGenGammaThresh fit.

No signal is injected (Ainj = 0).  The original spectrum is retained exactly
outside the replacement interval.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import scipy
import sklearn
import uproot
import yaml
from scipy.optimize import minimize
from scipy.stats import chi2

from hps_gpr.config import load_config
from hps_gpr.dataset import make_datasets
from hps_gpr.gpr import (
    fit_gpr,
    make_fixed_kernel,
    predict_counts_mean_from_log_gpr,
)
from hps_gpr.io import _build_model


MASS_GEV = 0.065
REPLACE_LO_GEV = 0.060
REPLACE_HI_GEV = 0.070
MASTER_SEED = 20260806
SOURCE_ROOT = Path(
    "/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root"
)
SOURCE_KEY = "preselection/h_invM_8000"
PARENT_CONFIG = (
    REPO
    / "study_configs"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "config_obsUL90_combined_wide_support_v4p2_2016k12_combined300.yaml"
)
STATE_LEDGER = (
    REPO
    / "study_results"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "derived"
    / "observed_gp_states_k12_reviewed.csv"
)
OUT_ROOT = HERE / "inputs" / "pseudo65_background_replacements.root"
PROVENANCE_JSON = HERE / "derived" / "input_provenance.json"
FIT_QC_JSON = HERE / "derived" / "functional_fit_qc.json"

GP_KEY = "gp_mean/preselection/h_invM_8000"
FUNC_KEY = "functional_form_fGenGammaThresh/preselection/h_invM_8000"
SOURCE_COPY_KEY = "source/preselection/h_invM_8000"
GP_EXPECTATION_KEY = "expectations/gp_mean_m065"
FUNC_EXPECTATION_KEY = "expectations/fGenGammaThresh_m065"

EXPECTED_SOURCE_SHA256 = (
    "3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4"
)
EXPECTED_V42_2021_STATE_SHA256 = (
    "c02fb8a3fc4bbe27ec9021f61d0eb0bd2f405538aa2a6379fa7dde71dc0102b4"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def provenance_path(path: Path) -> str:
    """Use a portable repo-relative path for in-repository artifacts."""
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def gengamma_shape_average(
    centers_lo: np.ndarray,
    centers_hi: np.ndarray,
    shape_parameters: np.ndarray,
    *,
    quadrature_order: int = 8,
) -> np.ndarray:
    """Return the bin-average fGenGammaThresh shape without amplitude."""
    a, log_lambda, power, x0, xt, log_w = np.asarray(shape_parameters, float)
    lam = float(np.exp(log_lambda))
    width = float(np.exp(log_w))
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    x = (
        0.5 * (centers_lo[:, None] + centers_hi[:, None])
        + 0.5 * (centers_hi - centers_lo)[:, None] * nodes[None, :]
    )
    z = x - x0
    safe_z = np.maximum(z, 1.0e-300)
    sigmoid = 1.0 / (
        1.0 + np.exp(-np.clip((x - xt) / width, -100.0, 100.0))
    )
    shape = np.where(
        z > 0.0,
        sigmoid * safe_z**a * np.exp(-((safe_z / lam) ** power)),
        0.0,
    )
    return np.asarray(shape @ (0.5 * weights), float)


def poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    obs = np.asarray(observed, float)
    exp = np.asarray(expected, float)
    if (
        obs.shape != exp.shape
        or not np.all(np.isfinite(exp))
        or np.any(exp <= 0.0)
    ):
        return float("inf")
    term = np.where(
        obs > 0.0,
        exp - obs + obs * np.log(obs / exp),
        exp,
    )
    return float(2.0 * np.sum(term))


def fit_functional_sidebands(
    values: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit fGenGammaThresh to 50--85 MeV excluding [60,70) MeV."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    fit_lo, fit_hi = 0.050, 0.085
    fit_mask = (
        (centers >= fit_lo)
        & (centers < fit_hi)
        & ~((centers >= REPLACE_LO_GEV) & (centers < REPLACE_HI_GEV))
    )
    low_mask = fit_mask & (centers < REPLACE_LO_GEV)
    high_mask = fit_mask & (centers >= REPLACE_HI_GEV)
    if np.count_nonzero(low_mask) == 0 or np.count_nonzero(high_mask) == 0:
        raise RuntimeError("Functional fit requires non-empty sidebands on both sides")

    observed = np.asarray(values[fit_mask], float)
    bin_lo = np.asarray(edges[:-1][fit_mask], float)
    bin_hi = np.asarray(edges[1:][fit_mask], float)
    weighted_mean = float(
        np.average(centers[fit_mask], weights=np.clip(observed, 0.0, None))
    )
    bounds = [
        (0.5, 20.0),
        (float(np.log(1.0e-4)), float(np.log(max(0.02, 2.0 * (fit_hi - fit_lo))))),
        (0.2, 10.0),
        (max(0.0, fit_lo - 0.030), min(weighted_mean, fit_lo + 0.020)),
        (max(0.0, fit_lo - 0.010), min(fit_lo + 0.050, fit_hi - 0.001)),
        (float(np.log(0.001)), float(np.log(0.020))),
    ]

    starts_physical = [
        (3.0344, 0.003781, 0.6524, 0.03747, 0.05151, 0.004268),
        (1.2, 0.0050, 1.2, 0.0400, 0.0480, 0.0030),
        (3.5, 0.0026, 0.62, 0.0337, 0.0496, 0.0064),
        (5.6, 0.00032, 0.49, 0.0275, 0.0509, 0.0063),
        (1.3, 0.00366, 0.509, 0.0427, 0.0565, 0.00767),
    ]

    def objective(parameters: np.ndarray, *, details: bool = False):
        shape = gengamma_shape_average(bin_lo, bin_hi, parameters)
        if (
            not np.all(np.isfinite(shape))
            or np.any(shape <= 0.0)
            or float(np.sum(shape)) <= 0.0
        ):
            return (float("inf"), float("nan"), np.full_like(observed, np.nan)) if details else 1.0e100
        amplitude = float(np.sum(observed) / np.sum(shape))
        expected = amplitude * shape
        deviance = poisson_deviance(observed, expected)
        if details:
            return deviance, amplitude, expected
        return deviance / float(len(observed))

    trials: list[dict[str, Any]] = []
    best_result = None
    for trial_index, physical in enumerate(starts_physical):
        a, lam, power, x0, xt, width = physical
        initial = np.array([a, np.log(lam), power, x0, xt, np.log(width)])
        initial = np.array(
            [np.clip(initial[i], bounds[i][0], bounds[i][1]) for i in range(6)]
        )
        result = minimize(
            objective,
            initial,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxiter": 10000,
                "xatol": 1.0e-11,
                "fatol": 1.0e-11,
            },
        )
        deviance, amplitude, expected = objective(result.x, details=True)
        central_mask = (centers >= REPLACE_LO_GEV) & (centers < REPLACE_HI_GEV)
        central_shape = gengamma_shape_average(
            edges[:-1][central_mask],
            edges[1:][central_mask],
            result.x,
        )
        trials.append(
            {
                "trial_index": trial_index,
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "objective_deviance_per_bin": float(result.fun),
                "poisson_deviance": float(deviance),
                "amplitude": float(amplitude),
                "central_expectation_sum": float(amplitude * np.sum(central_shape)),
                "parameters_internal": result.x.tolist(),
            }
        )
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result

    if best_result is None:
        raise RuntimeError("No functional-form fit trial was produced")

    deviance, amplitude, expected = objective(best_result.x, details=True)
    n_parameters = 7  # six shape parameters plus profiled amplitude
    ndf = int(len(observed) - n_parameters)
    pearson = float(np.sum((observed - expected) ** 2 / expected))
    pull = (observed - expected) / np.sqrt(expected)

    names = ["a", "lambda", "power", "x0", "xt", "w"]
    physical_parameters = [
        float(best_result.x[0]),
        float(np.exp(best_result.x[1])),
        float(best_result.x[2]),
        float(best_result.x[3]),
        float(best_result.x[4]),
        float(np.exp(best_result.x[5])),
    ]
    physical_bounds = [
        bounds[0],
        (float(np.exp(bounds[1][0])), float(np.exp(bounds[1][1]))),
        bounds[2],
        bounds[3],
        bounds[4],
        (float(np.exp(bounds[5][0])), float(np.exp(bounds[5][1]))),
    ]
    bound_fraction = {}
    at_bound = {}
    for name, value, (lower, upper) in zip(
        names, physical_parameters, physical_bounds
    ):
        fraction = float((value - lower) / (upper - lower))
        bound_fraction[name] = fraction
        at_bound[name] = bool(fraction <= 1.0e-3 or fraction >= 1.0 - 1.0e-3)

    near_optimum_trials = [
        trial
        for trial in trials
        if np.isfinite(trial["poisson_deviance"])
        and trial["poisson_deviance"] <= deviance + 1.0
    ]
    central_sums = [
        float(trial["central_expectation_sum"]) for trial in near_optimum_trials
    ]
    fit_pass = bool(
        best_result.success
        and ndf > 0
        and deviance / ndf < 1.5
        and pearson / ndf < 1.5
        and chi2.sf(deviance, ndf) > 0.01
        and not any(at_bound.values())
        and len(near_optimum_trials) >= 2
    )
    fit_info = {
        "model": "fGenGammaThresh",
        "interpretation": (
            "Local smooth sideband interpolation truth; not a physical generator"
        ),
        "fit_range_GeV": [fit_lo, fit_hi],
        "excluded_interval_GeV": [REPLACE_LO_GEV, REPLACE_HI_GEV],
        "likelihood": "binned Poisson; amplitude profiled analytically",
        "bin_expectation": "8-point Gauss-Legendre bin average",
        "n_bins_low_sideband": int(np.count_nonzero(low_mask)),
        "n_bins_high_sideband": int(np.count_nonzero(high_mask)),
        "n_bins_fit": int(len(observed)),
        "n_parameters": n_parameters,
        "ndf": ndf,
        "optimizer": {
            "method": "Nelder-Mead",
            "success": bool(best_result.success),
            "message": str(best_result.message),
            "iterations": int(best_result.nit),
            "evaluations": int(best_result.nfev),
            "n_deterministic_starts": len(starts_physical),
        },
        "amplitude": float(amplitude),
        "parameters": dict(zip(names, physical_parameters)),
        "parameter_bounds": {
            name: [float(lower), float(upper)]
            for name, (lower, upper) in zip(names, physical_bounds)
        },
        "parameter_bound_fraction": bound_fraction,
        "parameter_at_bound": at_bound,
        "poisson_deviance": float(deviance),
        "poisson_deviance_per_ndf": float(deviance / ndf),
        "poisson_deviance_pvalue": float(chi2.sf(deviance, ndf)),
        "pearson_chi2": pearson,
        "pearson_chi2_per_ndf": float(pearson / ndf),
        "pull_mean": float(np.mean(pull)),
        "pull_rms": float(np.sqrt(np.mean(pull**2))),
        "pull_max_abs": float(np.max(np.abs(pull))),
        "near_optimum_trial_count_delta_deviance_le_1": len(
            near_optimum_trials
        ),
        "near_optimum_central_expectation_range": [
            float(min(central_sums)),
            float(max(central_sums)),
        ],
        "trials": trials,
        "fit_qc_pass": fit_pass,
        "fit_qc_criteria": {
            "optimizer_success": True,
            "deviance_per_ndf_lt": 1.5,
            "pearson_per_ndf_lt": 1.5,
            "deviance_pvalue_gt": 0.01,
            "no_parameter_within_fraction_of_bound": 0.001,
            "minimum_near_optimum_starts": 2,
        },
    }
    if not fit_pass:
        raise RuntimeError(
            "Sideband-only fGenGammaThresh fit failed predeclared QC; "
            f"deviance/ndf={deviance / ndf:.4f}, "
            f"Pearson/ndf={pearson / ndf:.4f}"
        )
    return np.array([amplitude, *physical_parameters], float), fit_info


def evaluate_functional_native(
    edges: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    amplitude, a, lam, power, x0, xt, width = np.asarray(parameters, float)
    internal = np.array([a, np.log(lam), power, x0, xt, np.log(width)])
    return amplitude * gengamma_shape_average(edges[:-1], edges[1:], internal)


def reconstruct_gp_native_expectation(
    values: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = load_config(str(PARENT_CONFIG))
    datasets = make_datasets(cfg)
    ds = datasets["2021"]
    states = pd.read_csv(STATE_LEDGER)
    rows = states[
        (states["dataset"].astype(str) == "2021")
        & np.isclose(states["mass_GeV"].to_numpy(float), MASS_GEV, atol=5.0e-10)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one accepted 2021 state at 65 MeV, found {len(rows)}")
    row = rows.iloc[0]

    sigma = float(ds.sigma(MASS_GEV))
    blind = (
        MASS_GEV - float(cfg.blind_nsigma) * sigma,
        MASS_GEV + float(cfg.blind_nsigma) * sigma,
    )
    model = _build_model(
        ds,
        blind,
        rebin=int(cfg.neighborhood_rebin),
        config=cfg,
        mass=MASS_GEV,
    )
    coarse_centers = np.asarray(model.histogram.axes[0].centers, float)
    coarse_values = np.asarray(model.histogram.values(), float)
    train_mask = (coarse_centers < blind[0]) | (coarse_centers > blind[1])
    fixed_kernel = make_fixed_kernel(float(row["const_opt"]), float(row["ls_opt"]))
    gpr = fit_gpr(
        coarse_centers[train_mask],
        coarse_values[train_mask],
        cfg,
        restarts=0,
        kernel=fixed_kernel,
        optimize=False,
    )
    reconstructed_lml = float(gpr.log_marginal_likelihood_value_)
    if not np.isclose(
        reconstructed_lml, float(row["lml"]), atol=3.0e-5, rtol=0.0
    ):
        raise RuntimeError(
            "Exact fixed-state LML mismatch: "
            f"{reconstructed_lml:.12g} vs {float(row['lml']):.12g}"
        )
    coarse_mean = predict_counts_mean_from_log_gpr(gpr, coarse_centers, cfg)

    native_centers = 0.5 * (edges[:-1] + edges[1:])
    native_mask = (
        (native_centers >= REPLACE_LO_GEV)
        & (native_centers < REPLACE_HI_GEV)
    )
    coarse_mask = (
        (coarse_centers >= REPLACE_LO_GEV)
        & (coarse_centers < REPLACE_HI_GEV)
    )
    if np.count_nonzero(native_mask) != 80 or np.count_nonzero(coarse_mask) != 16:
        raise RuntimeError("Unexpected native/coarse replacement geometry")

    expectation = np.zeros_like(values, dtype=float)
    native_indices = np.where(native_mask)[0]
    coarse_indices = np.where(coarse_mask)[0]
    group_records = []
    for coarse_index in coarse_indices:
        coarse_lo = float(model.histogram.axes[0].edges[coarse_index])
        coarse_hi = float(model.histogram.axes[0].edges[coarse_index + 1])
        group = native_indices[
            (native_centers[native_indices] >= coarse_lo - 1.0e-14)
            & (native_centers[native_indices] < coarse_hi - 1.0e-14)
        ]
        if len(group) != 5:
            raise RuntimeError(
                f"Expected five native bins in coarse bin [{coarse_lo},{coarse_hi}), "
                f"found {len(group)}"
            )
        relative = predict_counts_mean_from_log_gpr(
            gpr, native_centers[group], cfg
        )
        relative = np.asarray(relative, float)
        native_means = float(coarse_mean[coarse_index]) * relative / np.sum(relative)
        expectation[group] = native_means
        group_records.append(
            {
                "coarse_lo_GeV": coarse_lo,
                "coarse_hi_GeV": coarse_hi,
                "coarse_mean_count": float(coarse_mean[coarse_index]),
                "native_mean_sum": float(np.sum(native_means)),
            }
        )

    gp_info = {
        "source": "exact accepted v4.2 2021 fixed GP state at 65 MeV",
        "state_ledger": provenance_path(STATE_LEDGER),
        "state_ledger_sha256": sha256_file(STATE_LEDGER),
        "expected_state_sha256": EXPECTED_V42_2021_STATE_SHA256,
        "mass_GeV": MASS_GEV,
        "sigma_GeV": sigma,
        "v42_blind_interval_physical_GeV": [float(blind[0]), float(blind[1])],
        "replacement_interval_GeV": [REPLACE_LO_GEV, REPLACE_HI_GEV],
        "const_opt": float(row["const_opt"]),
        "ls_opt": float(row["ls_opt"]),
        "reviewed_lml": float(row["lml"]),
        "reconstructed_lml": reconstructed_lml,
        "selected_source": str(row["selected_source"]),
        "selected_source_sha256": str(row["selected_source_sha256"]),
        "selected_review_status": str(row["review_status"]),
        "n_coarse_bins_total": int(len(coarse_centers)),
        "n_coarse_bins_training": int(np.count_nonzero(train_mask)),
        "n_coarse_bins_blind": int(np.count_nonzero(~train_mask)),
        "n_native_bins_replaced": int(np.count_nonzero(native_mask)),
        "coarse_group_checks": group_records,
        "central_expectation_sum": float(np.sum(expectation[native_mask])),
    }
    return expectation, gp_info


def main() -> None:
    (HERE / "inputs").mkdir(parents=True, exist_ok=True)
    (HERE / "derived").mkdir(parents=True, exist_ok=True)

    source_sha = sha256_file(SOURCE_ROOT)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"2021 input SHA256 mismatch: {source_sha} != {EXPECTED_SOURCE_SHA256}"
        )
    source_hist = uproot.open(SOURCE_ROOT)[SOURCE_KEY]
    values, edges = source_hist.to_numpy(flow=False)
    values = np.asarray(values, float)
    edges = np.asarray(edges, float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    replace_mask = (
        (centers >= REPLACE_LO_GEV) & (centers < REPLACE_HI_GEV)
    )
    if (
        len(values) != 8000
        or not np.isclose(edges[0], 0.0)
        or not np.isclose(edges[-1], 1.0)
        or np.count_nonzero(replace_mask) != 80
        or not np.allclose(np.diff(edges), 0.000125, atol=2.0e-16, rtol=0.0)
    ):
        raise RuntimeError("Unexpected source histogram geometry")
    if not np.all(values == np.rint(values)):
        raise RuntimeError("Source histogram contains non-integer bin contents")

    gp_expectation, gp_info = reconstruct_gp_native_expectation(values, edges)
    functional_parameters, fit_info = fit_functional_sidebands(values, edges)
    functional_full = evaluate_functional_native(edges, functional_parameters)
    functional_expectation = np.zeros_like(values, dtype=float)
    functional_expectation[replace_mask] = functional_full[replace_mask]

    seed_sequence = np.random.SeedSequence(MASTER_SEED)
    gp_seed, functional_seed = seed_sequence.spawn(2)
    gp_rng = np.random.Generator(np.random.PCG64(gp_seed))
    functional_rng = np.random.Generator(np.random.PCG64(functional_seed))

    gp_values = values.copy()
    func_values = values.copy()
    gp_draw = gp_rng.poisson(gp_expectation[replace_mask]).astype(np.int64)
    func_draw = functional_rng.poisson(
        functional_expectation[replace_mask]
    ).astype(np.int64)
    gp_values[replace_mask] = gp_draw
    func_values[replace_mask] = func_draw

    if not np.array_equal(gp_values[~replace_mask], values[~replace_mask]):
        raise RuntimeError("GP lane changed a bin outside the replacement interval")
    if not np.array_equal(func_values[~replace_mask], values[~replace_mask]):
        raise RuntimeError(
            "Functional lane changed a bin outside the replacement interval"
        )

    metadata_in_root = {
        "study": "v4p2 2021 10% conditional 65 MeV background replacement",
        "Ainj": 0.0,
        "signal_injected": False,
        "replacement_interval_GeV": [REPLACE_LO_GEV, REPLACE_HI_GEV],
        "source_sha256": source_sha,
        "master_seed": MASTER_SEED,
        "keys": {
            "source": SOURCE_COPY_KEY,
            "gp_mean": GP_KEY,
            "functional_form": FUNC_KEY,
            "gp_expectation": GP_EXPECTATION_KEY,
            "functional_expectation": FUNC_EXPECTATION_KEY,
        },
    }
    with uproot.recreate(OUT_ROOT) as root_file:
        root_file[SOURCE_COPY_KEY] = (values, edges)
        root_file[GP_KEY] = (gp_values, edges)
        root_file[FUNC_KEY] = (func_values, edges)
        root_file[GP_EXPECTATION_KEY] = (gp_expectation, edges)
        root_file[FUNC_EXPECTATION_KEY] = (functional_expectation, edges)
        root_file["metadata/json"] = json.dumps(
            metadata_in_root, sort_keys=True, separators=(",", ":")
        )

    root_sha = sha256_file(OUT_ROOT)
    function_central_expected = float(
        np.sum(functional_expectation[replace_mask])
    )
    fit_info["withheld_central_diagnostic_not_used_in_fit"] = {
        "original_observed_sum": float(np.sum(values[replace_mask])),
        "functional_expectation_sum": function_central_expected,
        "difference_observed_minus_expectation": float(
            np.sum(values[replace_mask]) - function_central_expected
        ),
    }
    fit_info["central_expectation_sha256"] = sha256_array(
        functional_expectation[replace_mask]
    )
    FIT_QC_JSON.write_text(
        json.dumps(json_safe(fit_info), indent=2, sort_keys=True) + "\n"
    )

    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git_commit(),
        "parent_analysis": "HPS-GPR analysis note v4.2 (2026-08-05)",
        "parent_config": provenance_path(PARENT_CONFIG),
        "parent_config_sha256": sha256_file(PARENT_CONFIG),
        "inference_from_ambiguous_request": {
            "Ainj": 0.0,
            "interpretation": (
                "background-only conditional replacements; no nonzero signal "
                "strength was supplied"
            ),
        },
        "source": {
            "root_path": str(SOURCE_ROOT),
            "histogram_key": SOURCE_KEY,
            "sha256": source_sha,
            "class": str(source_hist.classname),
            "n_bins": int(len(values)),
            "range_GeV": [float(edges[0]), float(edges[-1])],
            "native_bin_width_GeV": float(np.median(np.diff(edges))),
            "total_count": float(np.sum(values)),
            "replacement_original_count": float(np.sum(values[replace_mask])),
        },
        "replacement_geometry": {
            "center_mass_GeV": MASS_GEV,
            "requested_half_width_sigma": 2.5,
            "sigma_GeV": float(gp_info["sigma_GeV"]),
            "requested_physical_interval_GeV": [
                float(MASS_GEV - 2.5 * gp_info["sigma_GeV"]),
                float(MASS_GEV + 2.5 * gp_info["sigma_GeV"]),
            ],
            "implemented_complete_native_bin_interval_GeV": [
                REPLACE_LO_GEV,
                REPLACE_HI_GEV,
            ],
            "n_native_bins": int(np.count_nonzero(replace_mask)),
            "analysis_rebin_factor": 5,
            "analysis_bin_width_GeV": 0.000625,
            "n_analysis_bins": 16,
            "v42_analysis_blind_nsigma": 2.25,
            "geometry_note": (
                "At 65 MeV, both +/-2.25 sigma and +/-2.5 sigma select the same "
                "sixteen 0.625 MeV analysis-bin centers, whose complete native-bin "
                "edges are [60,70) MeV."
            ),
        },
        "randomization": {
            "distribution": "independent binwise Poisson",
            "master_seed": MASTER_SEED,
            "bit_generator": "PCG64",
            "numpy_version": np.__version__,
            "gp_child_seed_state": gp_seed.state,
            "functional_child_seed_state": functional_seed.state,
            "lanes_are_independent_child_streams": True,
        },
        "gp": {
            **gp_info,
            "central_expectation_sha256": sha256_array(
                gp_expectation[replace_mask]
            ),
            "central_draw_sha256": sha256_array(gp_draw),
            "central_draw_sum": int(np.sum(gp_draw)),
        },
        "functional_form": {
            "fit_qc_json": provenance_path(FIT_QC_JSON),
            "fit_qc_json_sha256": sha256_file(FIT_QC_JSON),
            "fit_qc_pass": bool(fit_info["fit_qc_pass"]),
            "fit_range_GeV": fit_info["fit_range_GeV"],
            "excluded_interval_GeV": fit_info["excluded_interval_GeV"],
            "parameters": fit_info["parameters"],
            "amplitude": fit_info["amplitude"],
            "central_expectation_sum": function_central_expected,
            "central_expectation_sha256": sha256_array(
                functional_expectation[replace_mask]
            ),
            "central_draw_sha256": sha256_array(func_draw),
            "central_draw_sum": int(np.sum(func_draw)),
        },
        "output": {
            "root_path": provenance_path(OUT_ROOT),
            "root_sha256": root_sha,
            "keys": metadata_in_root["keys"],
            "gp_total_count": float(np.sum(gp_values)),
            "functional_total_count": float(np.sum(func_values)),
            "outside_replacement_bitwise_equal": True,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "uproot": uproot.__version__,
            "pyyaml": yaml.__version__,
        },
        "interpretation_boundary": (
            "Each lane is one conditional background-only draw.  The observed "
            "spectrum outside [60,70) MeV is retained, so these are not independent "
            "global-null pseudoexperiments and do not establish expected sensitivity, "
            "coverage, or a global p-value."
        ),
    }
    PROVENANCE_JSON.write_text(
        json.dumps(json_safe(provenance), indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {OUT_ROOT}")
    print(f"Wrote {PROVENANCE_JSON}")
    print(f"Wrote {FIT_QC_JSON}")
    print(
        "Central sums: "
        f"source={np.sum(values[replace_mask]):.0f}, "
        f"GP expectation/draw={np.sum(gp_expectation[replace_mask]):.3f}/"
        f"{np.sum(gp_draw)}, "
        f"functional expectation/draw={function_central_expected:.3f}/"
        f"{np.sum(func_draw)}"
    )


if __name__ == "__main__":
    main()
