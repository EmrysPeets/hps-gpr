#!/usr/bin/env python3
"""Fit and cache the predeclared v4.9 fSigPowExpQ-anchored stress truth.

The local threshold correction is a logistic turn-on multiplied by the
exponential of a Chebyshev polynomial.  Candidate degree is chosen without
reference to any GP extraction: the lowest common degree that passes fixed
goodness-of-fit and stability gates in both native source spectra is retained.
The selected local fit is joined to the archived broad fSigPowExpQ anchor with
a quintic smootherstep.  The archive's redundant ``theta`` and ``c1``
coordinates are evaluated through the identifiable combination
``beta = c1 - 1/theta``.  This is deliberately labelled an anchored residual
stress truth, never a pure fSigPowExpQ fit or an independent physical model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import uproot
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from scipy.special import expit


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
REFERENCE = HERE / "reference"
DERIVED = HERE / "derived"
QA = HERE / "qa"

OUTPUT_ROOT = INPUTS / "fsig_anchor_background_toys_25.root"
OUTPUT_MANIFEST = INPUTS / "fsig_anchor_background_toys_25.manifest.json"
FIT_SUMMARY = DERIVED / "fsig_anchor_fit_summary.json"
QA_SUMMARY = QA / "truth_product_validation.json"

HIST_NAME = "preselection/h_invM_8000"
LOCAL_RANGE = (0.030, 0.080)
SUPPORT_RANGE = (0.030, 0.300)
BLEND_RANGE = (0.075, 0.085)
CANDIDATE_DEGREES = (3, 4, 5, 6)
SELECTED_DEGREE = 6
QUADRATURE_ORDER = 16
BASE_SEED = 20260817

SOURCES = {
    "one_pct": {
        "path": INPUTS / "source_2021_1pct.root",
        "sha256": "eecb5b8f40820c8f3c5d9673d2cddc155f2e2282474e20eb4f8930fd0a561ada",
        "reference": REFERENCE / "fsig_seed_2021_1pct_support040_300.root",
        "reference_sha256": "3baf09030b73147d2ddbab7bca922c56dbdf08a2b75e7010f73b89996d733f4b",
        "reference_metadata": REFERENCE / "fsig_seed_2021_1pct_support040_300.root.metadata.json",
        "reference_metadata_sha256": "ac4455ebe5ff8ca1d944f878970b9b20dfd07a246441a9824cfe0a434141584e",
        "scenario": "2021_1pct_x10",
        "exposure_multiplier": 10,
    },
    "ten_pct": {
        "path": INPUTS / "source_2021_10pct.root",
        "sha256": "3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4",
        "reference": REFERENCE / "fsig_seed_2021_10pct_support040_300.root",
        "reference_sha256": "4f85e102540c496ca32ee2728a2fbf9a2a5494241a53e4dc2351a4ad56a9ce98",
        "reference_metadata": REFERENCE / "fsig_seed_2021_10pct_support040_300.root.metadata.json",
        "reference_metadata_sha256": "aa9fb4f3c75a7c089bd9254bdc795e54dc1780484d8cfd412f3f64ed0d566c71",
        "scenario": "2021_10pct",
        "exposure_multiplier": 1,
    },
}


class TruthBuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: Any, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def stable_seed_words(namespace: str, *parts: object) -> list[int]:
    material = "|".join([str(BASE_SEED), namespace, *map(str, parts)]).encode("utf-8")
    raw = hashlib.sha256(material).digest()[:16]
    return [int.from_bytes(raw[index:index + 4], "little") for index in range(0, 16, 4)]


def deviance_residual(expected: np.ndarray, observed: np.ndarray) -> np.ndarray:
    mu = np.clip(np.asarray(expected, dtype=float), 1.0e-12, None)
    obs = np.asarray(observed, dtype=float)
    term = np.where(obs > 0.0, 2.0 * (mu - obs + obs * np.log(obs / mu)), 2.0 * mu)
    return np.sign(obs - mu) * np.sqrt(np.maximum(term, 0.0))


def chebyshev_logistic_density(x: np.ndarray, params: np.ndarray, degree: int) -> np.ndarray:
    # Fixed coordinate makes coefficients comparable between fit variants.
    t = (np.asarray(x, dtype=float) - 0.0550) / 0.0250
    log_shape = np.polynomial.chebyshev.chebval(t, params[: degree + 1])
    turnon_mass = params[-2]
    turnon_width = math.exp(params[-1])
    return np.exp(np.clip(log_shape, -40.0, 40.0)) * expit((x - turnon_mass) / turnon_width)


def integrate_density(
    func: Callable[[np.ndarray], np.ndarray],
    low_edges: np.ndarray,
    high_edges: np.ndarray,
    order: int,
) -> np.ndarray:
    nodes, weights = leggauss(int(order))
    center = 0.5 * (low_edges + high_edges)
    half = 0.5 * (high_edges - low_edges)
    x = center[:, None] + half[:, None] * nodes[None, :]
    # ``func`` is an intensity in expected counts per GeV; the result is an
    # actual expected count integrated over each histogram bin.
    return half * np.sum(weights[None, :] * func(x), axis=1)


@dataclass
class FitResult:
    degree: int
    params: np.ndarray
    expected: np.ndarray
    deviance: float
    ndf: int
    deviance_ndf: float
    rebin5_deviance_ndf: float
    max_abs_rebin5_pull: float
    at_bound: bool
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_optimality: float
    parameter_lower_bounds: np.ndarray
    parameter_upper_bounds: np.ndarray


def fit_candidate(
    low_edges: np.ndarray,
    high_edges: np.ndarray,
    observed: np.ndarray,
    degree: int,
) -> FitResult:
    bin_width = float(np.median(high_edges - low_edges))
    p0 = np.r_[math.log(max(float(observed.max()) / bin_width, 1.0)), np.zeros(degree), 0.048, math.log(0.004)]
    lower = np.r_[0.0, np.full(degree, -20.0), 0.035, math.log(0.0003)]
    upper = np.r_[30.0, np.full(degree, 20.0), 0.070, math.log(0.0200)]

    def expected(params: np.ndarray, order: int = QUADRATURE_ORDER) -> np.ndarray:
        return integrate_density(
            lambda x: chebyshev_logistic_density(x, params, degree),
            low_edges,
            high_edges,
            order,
        )

    best: tuple[float, Any] | None = None
    span = upper - lower
    for trial in range(12):
        rng = np.random.default_rng(np.random.SeedSequence(stable_seed_words("fit_start", degree, trial)))
        start = p0.copy()
        if trial:
            start += rng.normal(0.0, 0.025, start.size) * span
        start = np.clip(start, lower + 1.0e-7 * span, upper - 1.0e-7 * span)
        fit = least_squares(
            lambda p: deviance_residual(expected(p), observed),
            start,
            bounds=(lower, upper),
            max_nfev=20000,
            ftol=1.0e-11,
            xtol=1.0e-11,
            gtol=1.0e-11,
        )
        score = float(np.sum(deviance_residual(expected(fit.x), observed) ** 2))
        if best is None or score < best[0]:
            best = (score, fit)
    if best is None:
        raise TruthBuildError(f"no fit result for degree {degree}")
    deviance, fit = best
    mu = expected(fit.x)
    ndf = int(observed.size - fit.x.size)
    obs5 = observed.reshape(-1, 5).sum(axis=1)
    mu5 = mu.reshape(-1, 5).sum(axis=1)
    dev5 = float(np.sum(deviance_residual(mu5, obs5) ** 2))
    pull5 = (obs5 - mu5) / np.sqrt(np.clip(mu5, 1.0, None))
    near = np.minimum(np.abs((fit.x - lower) / span), np.abs((upper - fit.x) / span))
    return FitResult(
        degree=degree,
        params=np.asarray(fit.x, dtype=float),
        expected=mu,
        deviance=deviance,
        ndf=ndf,
        deviance_ndf=deviance / ndf,
        rebin5_deviance_ndf=dev5 / (obs5.size - fit.x.size),
        max_abs_rebin5_pull=float(np.max(np.abs(pull5))),
        at_bound=bool(np.any(near < 1.0e-4)),
        optimizer_success=bool(fit.success),
        optimizer_status=int(fit.status),
        optimizer_message=str(fit.message),
        optimizer_optimality=float(fit.optimality),
        parameter_lower_bounds=lower,
        parameter_upper_bounds=upper,
    )


def c2_smootherstep(x: np.ndarray) -> np.ndarray:
    u = np.clip((np.asarray(x) - BLEND_RANGE[0]) / (BLEND_RANGE[1] - BLEND_RANGE[0]), 0.0, 1.0)
    return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def load_reference_tail(record: dict[str, Any], native_bin_width: float) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, float]]:
    metadata = json.loads(Path(record["reference_metadata"]).read_text())
    fit = next(item for item in metadata["fits"] if item["tag"] == "fSigPowExpQ")
    parameters = {item["name"]: float(item["value"]) for item in fit["parameters"]}
    parameters["beta_identified"] = parameters["c1"] - 1.0 / parameters["theta"]

    def intensity(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        positive = x > 0.0
        safe = np.maximum(x, 1.0e-12)
        old_count_per_bin = (
            parameters["A"]
            * expit((x - parameters["xt"]) / parameters["w"])
            * np.power(safe, parameters["a"])
            * np.exp(np.clip(
                parameters["beta_identified"] * x + parameters["c2"] * x * x,
                -700.0,
                700.0,
            ))
        )
        return np.where(positive, old_count_per_bin / native_bin_width, 0.0)

    return intensity, parameters


def residual_diagnostics(expected: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    residual = deviance_residual(expected, observed)
    signs = residual >= 0.0
    runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size else 0
    npos = int(np.count_nonzero(signs))
    nneg = int(signs.size - npos)
    if npos > 0 and nneg > 0:
        mean_runs = 1.0 + 2.0 * npos * nneg / (npos + nneg)
        var_runs = (
            2.0 * npos * nneg * (2.0 * npos * nneg - npos - nneg)
            / (((npos + nneg) ** 2) * (npos + nneg - 1.0))
        )
        runs_z = (runs - mean_runs) / math.sqrt(max(var_runs, 1.0e-12))
    else:
        runs_z = float("nan")
    lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1]) if residual.size > 2 else float("nan")
    return {"n_runs": runs, "runs_z": runs_z, "lag1_deviance_residual_correlation": lag1}


def value_slope_curvature(func: Callable[[np.ndarray], np.ndarray], x: float, h: float = 2.0e-6) -> list[float]:
    fm = float(np.asarray(func(np.asarray([x - h])))[0])
    f0 = float(np.asarray(func(np.asarray([x])))[0])
    fp = float(np.asarray(func(np.asarray([x + h])))[0])
    return [f0, (fp - fm) / (2.0 * h), (fp - 2.0 * f0 + fm) / (h * h)]


def load_histogram(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    with uproot.open(path) as root:
        values, edges = root[key].to_numpy()
    return np.asarray(values, dtype=float), np.asarray(edges, dtype=float)


def fit_without_window(
    low: np.ndarray,
    high: np.ndarray,
    observed: np.ndarray,
    mass: float,
    sigma: float,
) -> dict[str, Any]:
    center = 0.5 * (low + high)
    keep = np.abs(center - mass) > 2.25 * sigma
    # Fit routine requires the five-bin grouping only for its reporting.  The
    # LO-window diagnostic therefore uses raw-bin deviance and prediction drift.
    bin_width = float(np.median(high - low))
    p0 = np.r_[math.log(max(float(observed[keep].max()) / bin_width, 1.0)), np.zeros(SELECTED_DEGREE), 0.048, math.log(0.004)]
    lower = np.r_[0.0, np.full(SELECTED_DEGREE, -20.0), 0.035, math.log(0.0003)]
    upper = np.r_[30.0, np.full(SELECTED_DEGREE, 20.0), 0.070, math.log(0.0200)]

    def expected(params: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return integrate_density(
            lambda x: chebyshev_logistic_density(x, params, SELECTED_DEGREE), lo, hi, QUADRATURE_ORDER
        )

    best: tuple[float, Any] | None = None
    span = upper - lower
    for trial in range(8):
        rng = np.random.default_rng(np.random.SeedSequence(stable_seed_words("loo_start", mass, trial)))
        start = p0 + (rng.normal(0.0, 0.02, p0.size) * span if trial else 0.0)
        start = np.clip(start, lower + 1.0e-7 * span, upper - 1.0e-7 * span)
        result = least_squares(
            lambda p: deviance_residual(expected(p, low[keep], high[keep]), observed[keep]),
            start,
            bounds=(lower, upper),
            max_nfev=20000,
        )
        score = float(np.sum(deviance_residual(expected(result.x, low[keep], high[keep]), observed[keep]) ** 2))
        if best is None or score < best[0]:
            best = (score, result)
    assert best is not None
    pred = expected(best[1].x, low, high)
    return {
        "mass_gev": mass,
        "excluded_half_width_gev": 2.25 * sigma,
        "n_excluded_raw_bins": int(np.count_nonzero(~keep)),
        "params": best[1].x.tolist(),
        "prediction_sha256_float64": array_sha256(pred, "<f8"),
        "prediction": pred,
    }


def build() -> dict[str, Any]:
    for record in SOURCES.values():
        if sha256_file(record["path"]) != record["sha256"]:
            raise TruthBuildError(f"source hash mismatch: {record['path']}")
        if sha256_file(record["reference"]) != record["reference_sha256"]:
            raise TruthBuildError(f"reference hash mismatch: {record['reference']}")
        if sha256_file(record["reference_metadata"]) != record["reference_metadata_sha256"]:
            raise TruthBuildError(f"reference metadata hash mismatch: {record['reference_metadata']}")

    fit_records: dict[str, Any] = {}
    output_payload: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    toy_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []

    for family, record in SOURCES.items():
        source_values, edges = load_histogram(record["path"], HIST_NAME)
        centers = 0.5 * (edges[:-1] + edges[1:])
        local = (centers >= LOCAL_RANGE[0]) & (centers < LOCAL_RANGE[1])
        support = (centers >= SUPPORT_RANGE[0]) & (centers < SUPPORT_RANGE[1])
        low_local, high_local = edges[:-1][local], edges[1:][local]
        observed_local = source_values[local]

        candidates: dict[int, FitResult] = {
            degree: fit_candidate(low_local, high_local, observed_local, degree)
            for degree in CANDIDATE_DEGREES
        }
        # Predeclared gates use data GOF only; no extraction or injected-signal
        # result can enter model selection.
        def passes(candidate: FitResult) -> bool:
            return (
                candidate.deviance_ndf <= 1.50
                and candidate.rebin5_deviance_ndf <= 2.00
                and candidate.max_abs_rebin5_pull <= 5.0
                and not candidate.at_bound
            )

        passing = [degree for degree in CANDIDATE_DEGREES if passes(candidates[degree])]
        if not passing:
            raise TruthBuildError(f"no candidate passes fixed GOF gates for {family}")
        # Degree is common across source families.  Degrees four and five can
        # pass the lower-statistics 1% source alone, but fail the native-10%
        # five-bin GOF gate once 30--35 MeV is included.  Degree six is the
        # lowest degree that passes both sources under the frozen gates.
        selected = SELECTED_DEGREE
        if not passes(candidates[selected]):
            raise TruthBuildError(f"common selected degree {selected} fails {family}")
        fit = candidates[selected]

        support_low = edges[:-1][support]
        support_high = edges[1:][support]
        support_centers = centers[support]
        local_intensity = lambda x: chebyshev_logistic_density(x, fit.params, selected)
        local_full = integrate_density(
            local_intensity,
            support_low,
            support_high,
            QUADRATURE_ORDER,
        )
        local_full_32 = integrate_density(
            local_intensity,
            support_low,
            support_high,
            2 * QUADRATURE_ORDER,
        )
        quadrature_rel = float(np.max(np.abs(local_full - local_full_32) / np.maximum(local_full_32, 1.0e-12)))

        baseline, baseline_edges = load_histogram(
            record["reference"], "fSigPowExpQ/fSigPowExpQ_analytic_seed_lumi_scaled"
        )
        if not np.array_equal(edges, baseline_edges):
            raise TruthBuildError(f"reference edge mismatch for {family}")
        baseline_support = baseline[support]
        tail_intensity, tail_parameters = load_reference_tail(record, float(edges[1] - edges[0]))
        fixed_local_bins = integrate_density(
            lambda x: (1.0 - c2_smootherstep(x)) * local_intensity(x),
            support_low,
            support_high,
            QUADRATURE_ORDER,
        )
        tail_basis_bins = integrate_density(
            lambda x: c2_smootherstep(x) * tail_intensity(x),
            support_low,
            support_high,
            QUADRATURE_ORDER,
        )
        target_total = float(np.sum(source_values[support]))
        fixed_local = float(np.sum(fixed_local_bins))
        tail_basis = float(np.sum(tail_basis_bins))
        tail_scale = (target_total - fixed_local) / tail_basis
        if not math.isfinite(tail_scale) or tail_scale <= 0.0:
            raise TruthBuildError(f"nonpositive tail scale for {family}: {tail_scale}")
        native_mean_support = fixed_local_bins + tail_scale * tail_basis_bins
        if np.any(~np.isfinite(native_mean_support)) or np.any(native_mean_support <= 0.0):
            raise TruthBuildError(f"full-support truth is not strictly positive for {family}")
        if abs(float(native_mean_support.sum()) - target_total) > 1.0e-6:
            raise TruthBuildError(f"observed-total constraint failed for {family}")
        tail_region = support_centers >= BLEND_RANGE[1]
        tail_baseline = baseline_support[tail_region]
        tail_new = native_mean_support[tail_region]
        tail_ratio = tail_new / np.maximum(tail_baseline, 1.0e-12)
        tail_obs = source_values[support][tail_region]
        tail_deviance = float(np.sum(deviance_residual(tail_new, tail_obs) ** 2))
        blended_intensity = lambda x: (
            (1.0 - c2_smootherstep(x)) * local_intensity(x)
            + c2_smootherstep(x) * tail_scale * tail_intensity(x)
        )
        continuity = {}
        for label, boundary, component in (
            ("low_75mev", BLEND_RANGE[0], local_intensity),
            ("high_80mev", BLEND_RANGE[1], lambda x: tail_scale * tail_intensity(x)),
        ):
            blend_v = value_slope_curvature(blended_intensity, boundary)
            component_v = value_slope_curvature(component, boundary)
            relative = [
                abs(a - b) / max(abs(b), 1.0)
                for a, b in zip(blend_v, component_v)
            ]
            continuity[label] = {
                "blended_value_slope_curvature": blend_v,
                "endpoint_component_value_slope_curvature": component_v,
                "relative_differences": relative,
                "finite_difference_step_gev": 2.0e-6,
            }

        # Retain native 8000-bin geometry with strictly positive values only on
        # the declared 40--300 MeV GP support and zeros outside that support.
        native_mean = np.zeros_like(source_values, dtype=float)
        native_mean[support] = native_mean_support
        scenario = record["scenario"]
        multiplier = int(record["exposure_multiplier"])
        scenario_mean = native_mean * multiplier
        baseline_scenario_mean = baseline * multiplier
        mean_key = f"truth/fsig_anchor/{scenario}_mean"
        local_display_key = f"truth/fsig_anchor_local_fit/{scenario}_mean"
        baseline_key = f"truth/baseline_fSigPowExpQ/{scenario}_mean"
        output_payload[mean_key] = (scenario_mean, edges)
        local_display = np.zeros_like(source_values, dtype=float)
        local_display[local] = fit.expected * multiplier
        output_payload[local_display_key] = (local_display, edges)
        output_payload[baseline_key] = (baseline_scenario_mean, edges)

        # Leave-one-signal-window-out is diagnostic only and cannot change the
        # selected degree or fitted production parameters.
        loo = []
        for mass in (0.055, 0.060, 0.065, 0.070):
            sigma = 0.00184825 - 0.001375 * mass + 0.085875 * mass * mass
            item = fit_without_window(low_local, high_local, observed_local, mass, sigma)
            full_pred = fit.expected
            loo_pred = item.pop("prediction")
            item["max_abs_fractional_prediction_change"] = float(
                np.max(np.abs(loo_pred - full_pred) / np.maximum(full_pred, 1.0e-12))
            )
            item["rms_fractional_prediction_change"] = float(
                np.sqrt(np.mean(((loo_pred - full_pred) / np.maximum(full_pred, 1.0e-12)) ** 2))
            )
            loo.append(item)

        candidate_json = []
        for degree, candidate in candidates.items():
            candidate_json.append({
                "degree": degree,
                "n_parameters": int(candidate.params.size),
                "parameters": candidate.params.tolist(),
                "deviance": candidate.deviance,
                "ndf": candidate.ndf,
                "deviance_ndf": candidate.deviance_ndf,
                "rebin5_deviance_ndf": candidate.rebin5_deviance_ndf,
                "max_abs_rebin5_pull": candidate.max_abs_rebin5_pull,
                "aic": candidate.deviance + 2.0 * candidate.params.size,
                "bic": candidate.deviance + candidate.params.size * math.log(observed_local.size),
                "residual_diagnostics": residual_diagnostics(candidate.expected, observed_local),
                "at_bound": candidate.at_bound,
                "optimizer_success": candidate.optimizer_success,
                "optimizer_status": candidate.optimizer_status,
                "optimizer_message": candidate.optimizer_message,
                "optimizer_optimality": candidate.optimizer_optimality,
                "parameter_lower_bounds": candidate.parameter_lower_bounds.tolist(),
                "parameter_upper_bounds": candidate.parameter_upper_bounds.tolist(),
                "passes_fixed_gates": passes(candidate),
            })
        fit_records[family] = {
            "fit_source": str(record["path"].relative_to(HERE)),
            "fit_source_sha256": record["sha256"],
            "fit_histogram": HIST_NAME,
            "fit_range_gev": list(LOCAL_RANGE),
            "fit_bin_width_gev": float(edges[1] - edges[0]),
            "fit_observed_count": int(observed_local.sum()),
            "candidate_family": "logistic_times_exp_Chebyshev",
            "candidate_degrees": candidate_json,
            "selected_degree": selected,
            "selected_parameters": fit.params.tolist(),
            "selection_rule": "lowest common degree passing fixed raw and five-bin Poisson-deviance, pull, and parameter-bound gates in both native sources",
            "selection_uses_injection_results": False,
            "quadrature": {
                "rule": "fixed Gauss-Legendre on each native bin",
                "order": QUADRATURE_ORDER,
                "doubled_order": 2 * QUADRATURE_ORDER,
                "max_relative_change": quadrature_rel,
            },
            "full_support_construction": {
                "support_range_gev": list(SUPPORT_RANGE),
                "local_model_range_gev": [SUPPORT_RANGE[0], BLEND_RANGE[0]],
                "blend_range_gev": list(BLEND_RANGE),
                "blend_weight": "u^3*(10-15u+6u^2), C2 with zero first and second derivatives at both endpoints",
                "high_tail_source": str(record["reference"].relative_to(HERE)),
                "high_tail_source_sha256": record["reference_sha256"],
                "high_tail_function": "fSigPowExpQ_identified_beta_anchor",
                "high_tail_parameters": tail_parameters,
                "blend_integration": "C2 weight and both intensities integrated jointly in every native bin with fixed Gauss-Legendre quadrature",
                "c2_numerical_continuity": continuity,
                "tail_scale_equation": "s=(Nobs_30_300-sum_i((1-w_i)*L_i))/sum_i(w_i*T_i)",
                "tail_scale": tail_scale,
                "native_observed_total_30_300": target_total,
                "native_mean_total_30_300": float(native_mean_support.sum()),
                "strictly_positive_on_support": True,
                "tail_85_300_validation": {
                    "n_bins": int(np.count_nonzero(tail_region)),
                    "observed_count": float(tail_obs.sum()),
                    "new_expected_count": float(tail_new.sum()),
                    "baseline_expected_count": float(tail_baseline.sum()),
                    "new_poisson_deviance_ndf": tail_deviance / max(int(np.count_nonzero(tail_region)) - 1, 1),
                    "new_over_baseline_min": float(tail_ratio.min()),
                    "new_over_baseline_max": float(tail_ratio.max()),
                    "new_over_baseline_rms_from_one": float(np.sqrt(np.mean((tail_ratio - 1.0) ** 2))),
                },
            },
            "toy_scenario": scenario,
            "toy_exposure_multiplier": multiplier,
            "toy_mean_total_30_300": float(scenario_mean.sum()),
            "leave_one_signal_window_out_diagnostic": loo,
        }

        truth_rows.append({
            "source_family": family,
            "fit_source": str(record["path"].relative_to(HERE)),
            "scenario": scenario,
            "exposure_multiplier": multiplier,
            "analytic_mean_key": mean_key,
            "local_fit_display_key": local_display_key,
            "baseline_analytic_mean_key": baseline_key,
            "mean_sha256_float64": array_sha256(scenario_mean, "<f8"),
            "mean_total": float(scenario_mean.sum()),
        })

        for toy_index in range(25):
            seed_words = stable_seed_words("fsig_anchor_poisson", scenario, toy_index)
            rng = np.random.default_rng(np.random.SeedSequence(seed_words))
            counts = rng.poisson(scenario_mean).astype(np.int64)
            key = f"toys/fsig_anchor/{scenario}/toy_{toy_index:04d}"
            output_payload[key] = (counts, edges)
            toy_rows.append({
                "source_family": family,
                "fit_source": str(record["path"].relative_to(HERE)),
                "scenario": scenario,
                "exposure_multiplier": multiplier,
                "toy_index": toy_index,
                "output_histogram": key,
                "seed_namespace": "fsig_anchor_poisson",
                "seed_words": seed_words,
                "counts_sha256": array_sha256(counts, "<i8"),
                "total_count": int(counts.sum()),
                "expected_mean_total": float(scenario_mean.sum()),
            })

    fit_payload = {
        "schema_version": 1,
        "study_id": HERE.name,
        "model_selection_frozen_before_injection": True,
        "smoothness_statement": "degree-six local log-spectrum over 50 MeV is C2-joined to an identified fSigPowExpQ anchor; no signal-scale knot is fitted, and leave-one-window-out checks are diagnostic only",
        "broadness_metric": {
            "n_internal_knots": 0,
            "smallest_basis_variation_scale_gev": (LOCAL_RANGE[1] - LOCAL_RANGE[0]) / SELECTED_DEGREE,
            "comparison_to_signal_sigma_gev": "nominal basis scale is at least 3.8 times sigma_m over 55-70 MeV",
            "maximum_leave_one_2p25sigma_window_out_drift_fraction": max(
                item["max_abs_fractional_prediction_change"]
                for record in fit_records.values()
                for item in record["leave_one_signal_window_out_diagnostic"]
            ),
        },
        "gates": {
            "raw_deviance_ndf_max": 1.50,
            "rebin5_deviance_ndf_max": 2.00,
            "max_abs_rebin5_pull": 5.0,
            "parameter_bound_contact_forbidden": True,
        },
        "fits": fit_records,
    }
    atomic_json(FIT_SUMMARY, fit_payload)

    temporary = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.{os.getpid()}.tmp")
    with uproot.recreate(temporary) as root:
        for key, histogram in output_payload.items():
            root[key] = histogram
    os.replace(temporary, OUTPUT_ROOT)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study_id": HERE.name,
        "generation": "independent native-source fSigPowExpQ-anchored threshold fits; 1pct mean x10 and native 10pct Poisson toys",
        "ensemble_semantics": "50 independent background spectra total (25 per scenario), reused across later mass, signal-strength, and paired-support fits; not independently regenerated per extraction state",
        "base_seed": BASE_SEED,
        "n_toys_per_scenario": 25,
        "scenarios": ["2021_1pct_x10", "2021_10pct"],
        "root": str(OUTPUT_ROOT.relative_to(HERE)),
        "root_sha256": sha256_file(OUTPUT_ROOT),
        "toy_key_template": "toys/fsig_anchor/{scenario}/toy_{toy_index:04d}",
        "truths": truth_rows,
        "toys": toy_rows,
        "fit_summary": str(FIT_SUMMARY.relative_to(HERE)),
        "fit_summary_sha256": sha256_file(FIT_SUMMARY),
    }
    content = dict(manifest)
    manifest["manifest_content_sha256"] = canonical_sha256(content)
    atomic_json(OUTPUT_MANIFEST, manifest)
    qa = validate()
    return {"status": "pass", "root": str(OUTPUT_ROOT), "manifest": str(OUTPUT_MANIFEST), "qa": qa}


def validate() -> dict[str, Any]:
    if not OUTPUT_ROOT.is_file() or not OUTPUT_MANIFEST.is_file() or not FIT_SUMMARY.is_file():
        raise TruthBuildError("truth products are incomplete")
    manifest = json.loads(OUTPUT_MANIFEST.read_text())
    content = dict(manifest)
    recorded = content.pop("manifest_content_sha256")
    if canonical_sha256(content) != recorded:
        raise TruthBuildError("manifest content hash mismatch")
    if sha256_file(OUTPUT_ROOT) != manifest["root_sha256"]:
        raise TruthBuildError("ROOT hash mismatch")
    if sha256_file(FIT_SUMMARY) != manifest["fit_summary_sha256"]:
        raise TruthBuildError("fit summary hash mismatch")
    if len(manifest["toys"]) != 50 or len(manifest["truths"]) != 2:
        raise TruthBuildError("unexpected toy/truth record count")
    checks: list[dict[str, Any]] = []
    with uproot.open(OUTPUT_ROOT) as root:
        for truth in manifest["truths"]:
            values, edges = root[truth["analytic_mean_key"]].to_numpy()
            centers = 0.5 * (edges[:-1] + edges[1:])
            support = (centers >= SUPPORT_RANGE[0]) & (centers < SUPPORT_RANGE[1])
            ok = (
                array_sha256(values, "<f8") == truth["mean_sha256_float64"]
                and np.all(values[support] > 0.0)
                and np.all(values[~support] == 0.0)
            )
            if not ok:
                raise TruthBuildError(f"analytic truth validation failed: {truth['scenario']}")
            root[truth["baseline_analytic_mean_key"]].to_numpy()
        for row in manifest["toys"]:
            values, edges = root[row["output_histogram"]].to_numpy()
            counts = np.asarray(values, dtype=np.int64)
            if array_sha256(counts, "<i8") != row["counts_sha256"]:
                raise TruthBuildError(f"toy count hash mismatch: {row['output_histogram']}")
            if int(counts.sum()) != row["total_count"] or np.any(counts < 0):
                raise TruthBuildError(f"toy total/nonnegative validation failed: {row['output_histogram']}")
    fit = json.loads(FIT_SUMMARY.read_text())
    for family, record in fit["fits"].items():
        qrel = record["quadrature"]["max_relative_change"]
        if qrel > 1.0e-8:
            raise TruthBuildError(f"quadrature convergence failed for {family}: {qrel}")
        checks.append({"family": family, "selected_degree": record["selected_degree"], "quadrature_max_rel": qrel})
    qa = {
        "status": "pass",
        "root_sha256": sha256_file(OUTPUT_ROOT),
        "manifest_sha256": sha256_file(OUTPUT_MANIFEST),
        "fit_summary_sha256": sha256_file(FIT_SUMMARY),
        "n_truths": len(manifest["truths"]),
        "n_toys": len(manifest["toys"]),
        "checks": checks,
    }
    atomic_json(QA_SUMMARY, qa)
    return qa


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    arguments = parser.parse_args()
    result = build() if arguments.command == "build" else validate()
    print(json.dumps(result, indent=2, sort_keys=True))
