#!/usr/bin/env python3
"""Build the frozen 2016 full-statistics threshold stress truth.

The low-mass shape is fit only to the independent 2016 10% development
spectrum. Its normalization uses one scalar count from the 2016 full sample.
No GP extraction, observed local p-value, or upper limit enters model choice.
"""

from __future__ import annotations

import argparse
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
QA = HERE / "qa"

SOURCE_10 = INPUTS / "source_2016_10pct.root"
SOURCE_FULL = INPUTS / "source_2016_full.root"
BASELINE_ROOT = INPUTS / "2016_thresholdfit_shape_x10_background_toys_100.root"
OUTPUT_ROOT = INPUTS / "2016_threshold_qualified_background_toys_100.root"
OUTPUT_MANIFEST = INPUTS / "2016_threshold_qualified_background_toys_100.manifest.json"
FIT_SUMMARY = REFERENCE / "2016_threshold_truth_fit_summary.json"
QA_SUMMARY = QA / "truth_product_validation.json"

SOURCE_10_SHA256 = "789e619fcbeb5e81f9193d3e224bc17919983477a037bf3d79692327555f9fd4"
SOURCE_FULL_SHA256 = "c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301"
HIST_NAME = "h_Minv_General_Final_1"
BASELINE_MEAN_KEY = "validation/fShiftSigPowTail_expected_counts"
LOCAL_RANGE = (0.026, 0.080)
SUPPORT_RANGE = (0.026, 0.210)
BLEND_RANGE = (0.075, 0.085)
CANDIDATE_DEGREES = tuple(range(4, 11))
QUADRATURE_ORDER = 16
BASE_SEED = 20260902
N_TOYS = 100
FULL_TARGET_COUNT = 73_145_594
EXPECTED_SOURCE_COUNT = 7_475_607
SCENARIO = "2016_full"


class TruthBuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: Any, dtype: str) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=dtype).tobytes(order="C")
    ).hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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
    material = "|".join(
        [str(BASE_SEED), namespace, *map(str, parts)]
    ).encode("utf-8")
    raw = hashlib.sha256(material).digest()[:16]
    return [
        int.from_bytes(raw[index : index + 4], "little")
        for index in range(0, 16, 4)
    ]


def deviance_residual(expected: np.ndarray, observed: np.ndarray) -> np.ndarray:
    mu = np.clip(np.asarray(expected, dtype=float), 1.0e-12, None)
    obs = np.asarray(observed, dtype=float)
    term = np.full_like(mu, 2.0 * mu)
    positive = obs > 0.0
    term[positive] = 2.0 * (
        mu[positive]
        - obs[positive]
        + obs[positive] * np.log(obs[positive] / mu[positive])
    )
    return np.sign(obs - mu) * np.sqrt(np.maximum(term, 0.0))


def logistic_chebyshev_density(
    x: np.ndarray, params: np.ndarray, degree: int
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    coordinate = (values - 0.053) / 0.027
    log_shape = np.polynomial.chebyshev.chebval(
        coordinate, params[: degree + 1]
    )
    turnon_mass = params[-2]
    turnon_width = math.exp(params[-1])
    return np.exp(np.clip(log_shape, -40.0, 40.0)) * expit(
        (values - turnon_mass) / turnon_width
    )


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
    lower: np.ndarray
    upper: np.ndarray


def parameter_bounds(degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = np.r_[
        math.log(1.0e8), np.zeros(degree), 0.034, math.log(0.003)
    ]
    lower = np.r_[
        0.0, np.full(degree, -20.0), 0.022, math.log(0.0002)
    ]
    upper = np.r_[
        35.0, np.full(degree, 20.0), 0.045, math.log(0.0200)
    ]
    return initial, lower, upper


def fit_candidate(
    low_edges: np.ndarray,
    high_edges: np.ndarray,
    observed: np.ndarray,
    degree: int,
    *,
    keep: np.ndarray | None = None,
    namespace: str = "fit_start",
    starts: int = 10,
) -> FitResult:
    initial, lower, upper = parameter_bounds(degree)
    bin_width = float(np.median(high_edges - low_edges))
    initial[0] = math.log(max(float(observed.max()) / bin_width, 1.0))
    if keep is None:
        keep = np.ones(observed.size, dtype=bool)

    def expected(params: np.ndarray, order: int = QUADRATURE_ORDER) -> np.ndarray:
        return integrate_density(
            lambda x: logistic_chebyshev_density(x, params, degree),
            low_edges,
            high_edges,
            order,
        )

    best: tuple[float, Any] | None = None
    span = upper - lower
    for trial in range(int(starts)):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                stable_seed_words(namespace, degree, trial, int(keep.sum()))
            )
        )
        start = initial.copy()
        if trial:
            start += rng.normal(0.0, 0.02, start.size) * span
        start = np.clip(
            start, lower + 1.0e-7 * span, upper - 1.0e-7 * span
        )
        result = least_squares(
            lambda p: deviance_residual(expected(p)[keep], observed[keep]),
            start,
            bounds=(lower, upper),
            max_nfev=15_000,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        score = float(
            np.sum(
                deviance_residual(
                    expected(result.x)[keep], observed[keep]
                )
                ** 2
            )
        )
        if best is None or score < best[0]:
            best = (score, result)
    if best is None:
        raise TruthBuildError(f"no fit result for degree {degree}")

    deviance, result = best
    mu = expected(result.x)
    ndf = int(np.count_nonzero(keep) - result.x.size)
    if observed.size % 5:
        raise TruthBuildError("local fit geometry is not divisible by five")
    obs5 = observed.reshape(-1, 5).sum(axis=1)
    mu5 = mu.reshape(-1, 5).sum(axis=1)
    dev5 = float(np.sum(deviance_residual(mu5, obs5) ** 2))
    pull5 = (obs5 - mu5) / np.sqrt(np.clip(mu5, 1.0, None))
    near = np.minimum(
        np.abs((result.x - lower) / span),
        np.abs((upper - result.x) / span),
    )
    return FitResult(
        degree=degree,
        params=np.asarray(result.x, dtype=float),
        expected=mu,
        deviance=deviance,
        ndf=ndf,
        deviance_ndf=deviance / ndf,
        rebin5_deviance_ndf=dev5 / (obs5.size - result.x.size),
        max_abs_rebin5_pull=float(np.max(np.abs(pull5))),
        at_bound=bool(np.any(near < 1.0e-4)),
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        optimizer_optimality=float(result.optimality),
        lower=lower,
        upper=upper,
    )


def passes(candidate: FitResult) -> bool:
    return bool(
        candidate.optimizer_success
        and candidate.deviance_ndf <= 1.50
        and candidate.rebin5_deviance_ndf <= 2.00
        and candidate.max_abs_rebin5_pull <= 5.0
        and not candidate.at_bound
    )


def residual_diagnostics(
    expected: np.ndarray, observed: np.ndarray
) -> dict[str, float]:
    residual = deviance_residual(expected, observed)
    signs = residual >= 0.0
    runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1]))
    npos = int(np.count_nonzero(signs))
    nneg = int(signs.size - npos)
    if npos > 0 and nneg > 0:
        mean_runs = 1.0 + 2.0 * npos * nneg / (npos + nneg)
        var_runs = (
            2.0
            * npos
            * nneg
            * (2.0 * npos * nneg - npos - nneg)
            / (((npos + nneg) ** 2) * (npos + nneg - 1.0))
        )
        runs_z = (runs - mean_runs) / math.sqrt(max(var_runs, 1.0e-12))
    else:
        runs_z = float("nan")
    lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    return {
        "n_runs": runs,
        "runs_z": runs_z,
        "lag1_deviance_residual_correlation": lag1,
    }


def c2_smootherstep(x: np.ndarray) -> np.ndarray:
    u = np.clip(
        (np.asarray(x) - BLEND_RANGE[0])
        / (BLEND_RANGE[1] - BLEND_RANGE[0]),
        0.0,
        1.0,
    )
    return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def load_histogram(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    with uproot.open(path) as root:
        values, edges = root[key].to_numpy()
    return np.asarray(values, dtype=float), np.asarray(edges, dtype=float)


def leave_window_out(
    low_edges: np.ndarray,
    high_edges: np.ndarray,
    observed: np.ndarray,
    nominal: FitResult,
    mass: float,
) -> dict[str, Any]:
    sigma = (
        0.00038
        + 0.041 * mass
        - 0.27 * mass**2
        + 3.49 * mass**3
        - 11.11 * mass**4
    )
    center = 0.5 * (low_edges + high_edges)
    keep = np.abs(center - mass) > 2.25 * sigma
    result = fit_candidate(
        low_edges,
        high_edges,
        observed,
        nominal.degree,
        keep=keep,
        namespace=f"loo_{mass:.6f}",
        starts=6,
    )
    fractional = (result.expected - nominal.expected) / np.maximum(
        nominal.expected, 1.0e-12
    )
    return {
        "mass_gev": mass,
        "sigma_gev": sigma,
        "excluded_half_width_gev": 2.25 * sigma,
        "n_excluded_raw_bins": int(np.count_nonzero(~keep)),
        "parameters": result.params.tolist(),
        "prediction_sha256_float64": array_sha256(result.expected, "<f8"),
        "max_abs_fractional_prediction_change": float(
            np.max(np.abs(fractional))
        ),
        "rms_fractional_prediction_change": float(
            np.sqrt(np.mean(fractional**2))
        ),
    }


def build() -> dict[str, Any]:
    for path, expected in (
        (SOURCE_10, SOURCE_10_SHA256),
        (SOURCE_FULL, SOURCE_FULL_SHA256),
    ):
        if sha256_file(path) != expected:
            raise TruthBuildError(f"source hash mismatch: {path}")
    if not BASELINE_ROOT.is_file():
        raise TruthBuildError(f"missing broad-tail source: {BASELINE_ROOT}")

    source, edges = load_histogram(SOURCE_10, HIST_NAME)
    full, full_edges = load_histogram(SOURCE_FULL, HIST_NAME)
    baseline, baseline_edges = load_histogram(BASELINE_ROOT, BASELINE_MEAN_KEY)
    if not np.array_equal(edges, full_edges) or not np.array_equal(
        edges, baseline_edges
    ):
        raise TruthBuildError("source/baseline histogram edge mismatch")

    centers = 0.5 * (edges[:-1] + edges[1:])
    local_mask = (centers >= LOCAL_RANGE[0]) & (centers < LOCAL_RANGE[1])
    support_mask = (centers >= SUPPORT_RANGE[0]) & (
        centers < SUPPORT_RANGE[1]
    )
    source_count = int(np.rint(source[support_mask].sum()))
    full_count = int(np.rint(full[support_mask].sum()))
    if source_count != EXPECTED_SOURCE_COUNT or full_count != FULL_TARGET_COUNT:
        raise TruthBuildError(
            f"common-envelope counts drift: source={source_count}, full={full_count}"
        )

    low_local = edges[:-1][local_mask]
    high_local = edges[1:][local_mask]
    observed_local = source[local_mask]
    candidates = {
        degree: fit_candidate(
            low_local, high_local, observed_local, degree
        )
        for degree in CANDIDATE_DEGREES
    }
    passing = [degree for degree in CANDIDATE_DEGREES if passes(candidates[degree])]
    if not passing:
        summary = {
            degree: {
                "deviance_ndf": value.deviance_ndf,
                "rebin5_deviance_ndf": value.rebin5_deviance_ndf,
                "max_abs_rebin5_pull": value.max_abs_rebin5_pull,
                "at_bound": value.at_bound,
            }
            for degree, value in candidates.items()
        }
        raise TruthBuildError(f"no degree passes frozen gates: {summary}")
    selected_degree = min(passing)
    selected = candidates[selected_degree]

    support_low = edges[:-1][support_mask]
    support_high = edges[1:][support_mask]
    support_centers = centers[support_mask]
    local_mean = integrate_density(
        lambda x: logistic_chebyshev_density(
            x, selected.params, selected_degree
        ),
        support_low,
        support_high,
        QUADRATURE_ORDER,
    )
    local_mean_32 = integrate_density(
        lambda x: logistic_chebyshev_density(
            x, selected.params, selected_degree
        ),
        support_low,
        support_high,
        2 * QUADRATURE_ORDER,
    )
    baseline_support = baseline[support_mask]
    weight = c2_smootherstep(support_centers)
    blended_source_shape = (
        (1.0 - weight) * local_mean + weight * baseline_support
    )
    blended_source_shape_32 = (
        (1.0 - weight) * local_mean_32 + weight * baseline_support
    )
    quadrature_max_rel = float(
        np.max(
            np.abs(blended_source_shape - blended_source_shape_32)
            / np.maximum(blended_source_shape_32, 1.0e-12)
        )
    )
    if np.any(~np.isfinite(blended_source_shape)) or np.any(
        blended_source_shape <= 0.0
    ):
        raise TruthBuildError("blended source shape is not finite and positive")
    scale = FULL_TARGET_COUNT / float(blended_source_shape.sum())
    scenario_mean_support = scale * blended_source_shape
    scenario_mean = np.zeros_like(source, dtype=float)
    scenario_mean[support_mask] = scenario_mean_support
    if not math.isclose(
        float(scenario_mean.sum()),
        float(FULL_TARGET_COUNT),
        rel_tol=0.0,
        abs_tol=1.0e-5,
    ):
        raise TruthBuildError("full-count normalization failed")

    candidate_rows = []
    for degree, candidate in candidates.items():
        candidate_rows.append(
            {
                "degree": degree,
                "n_parameters": int(candidate.params.size),
                "parameters": candidate.params.tolist(),
                "parameter_lower_bounds": candidate.lower.tolist(),
                "parameter_upper_bounds": candidate.upper.tolist(),
                "deviance": candidate.deviance,
                "ndf": candidate.ndf,
                "deviance_ndf": candidate.deviance_ndf,
                "rebin5_deviance_ndf": candidate.rebin5_deviance_ndf,
                "max_abs_rebin5_pull": candidate.max_abs_rebin5_pull,
                "at_bound": candidate.at_bound,
                "optimizer_success": candidate.optimizer_success,
                "optimizer_status": candidate.optimizer_status,
                "optimizer_message": candidate.optimizer_message,
                "optimizer_optimality": candidate.optimizer_optimality,
                "aic": candidate.deviance + 2.0 * candidate.params.size,
                "bic": candidate.deviance
                + candidate.params.size * math.log(observed_local.size),
                "residual_diagnostics": residual_diagnostics(
                    candidate.expected, observed_local
                ),
                "passes_fixed_gates": passes(candidate),
            }
        )

    loo = [
        leave_window_out(
            low_local, high_local, observed_local, selected, mass
        )
        for mass in (0.044, 0.049, 0.054, 0.059, 0.065)
    ]
    fit_summary = {
        "schema_version": 1,
        "study_id": HERE.name,
        "model_selection_frozen_before_support_extraction": True,
        "selection_uses_gp_or_observed_full_shape": False,
        "source_10pct": str(SOURCE_10.relative_to(HERE)),
        "source_10pct_sha256": SOURCE_10_SHA256,
        "source_full": str(SOURCE_FULL.relative_to(HERE)),
        "source_full_sha256": SOURCE_FULL_SHA256,
        "source_full_use": "one scalar common-envelope normalization only",
        "histogram": HIST_NAME,
        "local_fit_range_gev": list(LOCAL_RANGE),
        "truth_support_range_gev": list(SUPPORT_RANGE),
        "blend_range_gev": list(BLEND_RANGE),
        "candidate_family": "logistic_times_exp_Chebyshev",
        "candidate_degrees": candidate_rows,
        "selected_degree": selected_degree,
        "selected_parameters": selected.params.tolist(),
        "selection_rule": "lowest degree passing the frozen source-GOF gates",
        "gates": {
            "raw_deviance_ndf_max": 1.5,
            "rebin5_deviance_ndf_max": 2.0,
            "max_abs_rebin5_pull": 5.0,
            "parameter_bound_contact_forbidden": True,
        },
        "source_common_envelope_count": source_count,
        "full_common_envelope_count": full_count,
        "source_to_full_normalization": FULL_TARGET_COUNT / source_count,
        "final_blend_normalization": scale,
        "broad_tail_source": str(BASELINE_ROOT.relative_to(HERE)),
        "broad_tail_source_sha256": sha256_file(BASELINE_ROOT),
        "broad_tail_key": BASELINE_MEAN_KEY,
        "quadrature": {
            "order": QUADRATURE_ORDER,
            "doubled_order": 2 * QUADRATURE_ORDER,
            "max_relative_change": quadrature_max_rel,
        },
        "leave_one_signal_window_out_diagnostic": loo,
        "holdout_mass_gev": 0.065,
        "claim_boundary": (
            "Source-conditioned smooth stress truth; not a physical background "
            "generator, observed-data bias measurement, or coverage model."
        ),
    }
    atomic_json(FIT_SUMMARY, fit_summary)

    local_display = np.zeros_like(source, dtype=float)
    local_display[local_mask] = selected.expected
    output_payload: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "truth/threshold_qualified/2016_full_mean": (scenario_mean, edges),
        "truth/local_threshold_fit/2016_10pct_mean": (
            local_display,
            edges,
        ),
        "truth/broad_tail_baseline/2016_10pct_mean": (baseline, edges),
    }
    toy_rows = []
    for toy_index in range(N_TOYS):
        seed_words = stable_seed_words(
            "2016_threshold_poisson", SCENARIO, toy_index
        )
        rng = np.random.default_rng(np.random.SeedSequence(seed_words))
        counts = rng.poisson(scenario_mean).astype(np.int64)
        key = f"toys/threshold_qualified/{SCENARIO}/toy_{toy_index:04d}"
        output_payload[key] = (counts, edges)
        toy_rows.append(
            {
                "scenario": SCENARIO,
                "toy_index": toy_index,
                "output_histogram": key,
                "seed_namespace": "2016_threshold_poisson",
                "seed_words": seed_words,
                "counts_sha256": array_sha256(counts, "<i8"),
                "total_count": int(counts.sum()),
                "expected_mean_total": float(scenario_mean.sum()),
            }
        )

    temporary = OUTPUT_ROOT.with_name(
        f".{OUTPUT_ROOT.name}.{os.getpid()}.tmp"
    )
    with uproot.recreate(temporary) as root:
        for key, histogram in output_payload.items():
            root[key] = histogram
    os.replace(temporary, OUTPUT_ROOT)

    truth_row = {
        "scenario": SCENARIO,
        "analytic_mean_key": "truth/threshold_qualified/2016_full_mean",
        "mean_sha256_float64": array_sha256(scenario_mean, "<f8"),
        "mean_total": float(scenario_mean.sum()),
    }
    manifest = {
        "schema_version": 1,
        "study_id": HERE.name,
        "generation": (
            "2016 10pct threshold shape with one scalar 2016-full envelope "
            "normalization and 100 independent Poisson draws"
        ),
        "ensemble_semantics": (
            "The same 100 background spectra are paired across support edges, "
            "masses, and injection strengths."
        ),
        "base_seed": BASE_SEED,
        "n_toys_per_scenario": N_TOYS,
        "scenarios": [SCENARIO],
        "root": str(OUTPUT_ROOT.relative_to(HERE)),
        "root_sha256": sha256_file(OUTPUT_ROOT),
        "toy_key_template": (
            "toys/threshold_qualified/{scenario}/toy_{toy_index:04d}"
        ),
        "truths": [truth_row],
        "toys": toy_rows,
        "fit_summary": str(FIT_SUMMARY.relative_to(HERE)),
        "fit_summary_sha256": sha256_file(FIT_SUMMARY),
    }
    content = dict(manifest)
    manifest["manifest_content_sha256"] = canonical_sha256(content)
    atomic_json(OUTPUT_MANIFEST, manifest)
    qa = validate()
    return {
        "status": "pass",
        "root": str(OUTPUT_ROOT),
        "manifest": str(OUTPUT_MANIFEST),
        "qa": qa,
    }


def validate() -> dict[str, Any]:
    for path in (OUTPUT_ROOT, OUTPUT_MANIFEST, FIT_SUMMARY):
        if not path.is_file():
            raise TruthBuildError(f"missing truth product: {path}")
    manifest = json.loads(OUTPUT_MANIFEST.read_text(encoding="utf-8"))
    content = dict(manifest)
    recorded = content.pop("manifest_content_sha256")
    if canonical_sha256(content) != recorded:
        raise TruthBuildError("manifest content hash mismatch")
    if sha256_file(OUTPUT_ROOT) != manifest["root_sha256"]:
        raise TruthBuildError("ROOT hash mismatch")
    if sha256_file(FIT_SUMMARY) != manifest["fit_summary_sha256"]:
        raise TruthBuildError("fit-summary hash mismatch")
    if len(manifest["toys"]) != N_TOYS:
        raise TruthBuildError("toy inventory length mismatch")

    truth = manifest["truths"][0]
    with uproot.open(OUTPUT_ROOT) as root:
        mean, edges = root[truth["analytic_mean_key"]].to_numpy()
        centers = 0.5 * (edges[:-1] + edges[1:])
        support = (centers >= SUPPORT_RANGE[0]) & (
            centers < SUPPORT_RANGE[1]
        )
        if array_sha256(mean, "<f8") != truth["mean_sha256_float64"]:
            raise TruthBuildError("analytic-mean hash mismatch")
        if not np.all(mean[support] > 0.0) or not np.all(mean[~support] == 0.0):
            raise TruthBuildError("analytic-mean support/positivity failure")
        if not math.isclose(
            float(mean.sum()), FULL_TARGET_COUNT, rel_tol=0.0, abs_tol=1.0e-5
        ):
            raise TruthBuildError("analytic-mean total mismatch")
        for row in manifest["toys"]:
            values, toy_edges = root[row["output_histogram"]].to_numpy()
            counts = np.rint(values).astype(np.int64)
            if not np.array_equal(edges, toy_edges):
                raise TruthBuildError("toy edge mismatch")
            if array_sha256(counts, "<i8") != row["counts_sha256"]:
                raise TruthBuildError("toy count hash mismatch")
            if int(counts.sum()) != int(row["total_count"]):
                raise TruthBuildError("toy total mismatch")

    fit = json.loads(FIT_SUMMARY.read_text(encoding="utf-8"))
    selected = int(fit["selected_degree"])
    selected_row = next(
        row for row in fit["candidate_degrees"] if int(row["degree"]) == selected
    )
    if not bool(selected_row["passes_fixed_gates"]):
        raise TruthBuildError("selected degree does not pass frozen gates")
    if fit["quadrature"]["max_relative_change"] > 1.0e-8:
        raise TruthBuildError("quadrature convergence failure")
    qa = {
        "status": "pass",
        "root_sha256": sha256_file(OUTPUT_ROOT),
        "manifest_sha256": sha256_file(OUTPUT_MANIFEST),
        "fit_summary_sha256": sha256_file(FIT_SUMMARY),
        "selected_degree": selected,
        "n_toys": N_TOYS,
        "full_target_count": FULL_TARGET_COUNT,
    }
    atomic_json(QA_SUMMARY, qa)
    return qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    args = parser.parse_args()
    result = build() if args.command == "build" else validate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
