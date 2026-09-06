#!/usr/bin/env python3
"""Independent, prospectively frozen validator for v4.9.11p1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import pandas as pd
import uproot
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC_PATH = REPO / "study_configs/v4p9p11p1_2016_reference30_uniform_optimizer_remediation_20260902/study_spec.json"
PROTOCOL_PATH = HERE / "STUDY_PROTOCOL.md"
RUNNER_PATH = HERE / "run_uniform_remediation.py"
PREFLIGHT_PATH = HERE / "qa/preflight.json"
PATHS_PATH = HERE / "derived/optimizer_paths.csv"
STATES_PATH = HERE / "derived/observed_2016_gp_states_reviewed.csv"
DECISION_PATH = HERE / "derived/state_certification_decision.json"
VALIDATION_PATH = HERE / "qa/final_validation.json"
EXPECTED_PROTOCOL_SHA = "ae3f0bde2978f07c3e135d1e978fe9528482556b12cecef3f8dc5173776a9235"
EXPECTED_SPEC_SHA = "4a90696216afbd78d41d4a5f0e249a70488fae823ee32a1b0004d31d327557b6"
EXPECTED_RUNNER_SHA = "9327afde235ef655ab8895a305972fba65e89200529c07bd2377cba38cba5a27"
EXPECTED_PREFLIGHT_SHA = "6cbebed114511021389b766787b883a7c7462684245bed139555ac08500d2b6e"
EXPECTED_MASSES = list(range(39, 181))
FORBIDDEN_TOKENS = (
    "a_hat",
    "signal_pull",
    "p0",
    "z_local",
    "upper_limit",
    "epsilon2",
    "expected_band",
    "toy_limit",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def histogram_hash(values: np.ndarray, edges: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype=np.float64).tobytes())
    digest.update(np.asarray(edges, dtype=np.float64).tobytes())
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        if value.strip().lower() == "true":
            return True
        if value.strip().lower() == "false":
            return False
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"not a canonical boolean: {value!r}")


def blank(value: Any) -> bool:
    return bool(pd.isna(value) or (isinstance(value, str) and not value.strip()))


def close(a: Any, b: Any, atol: float = 1e-10, rtol: float = 1e-10) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), atol=atol, rtol=rtol, equal_nan=True))
    except (TypeError, ValueError):
        return False


def read_freeze(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2 or pieces[1] in result:
            raise RuntimeError(f"malformed freeze {path}")
        result[pieces[1]] = pieces[0]
    return result


def sigma_2016(mass: float, spec: dict[str, Any]) -> float:
    coefficients = [float(value) for value in spec["sigma_coeffs_2016"]]
    transition = float(spec["sigma_tail_m0_2016"])
    if mass <= transition:
        return float(sum(coefficient * mass**power for power, coefficient in enumerate(coefficients)))
    at_transition = float(sum(coefficient * transition**power for power, coefficient in enumerate(coefficients)))
    return at_transition + float(spec["sigma_tail_slope_override_2016"]) * (mass - transition)


def sigma_x(mass: float, spec: dict[str, Any]) -> float:
    return float(np.log((mass + sigma_2016(mass, spec)) / mass))


def length_bounds(mass: float, spec: dict[str, Any]) -> tuple[float, float]:
    local = sigma_x(mass, spec)
    resolution_grid = np.linspace(0.039, 0.180, int(spec["kernel_ls_res_npts"]))
    global_base = float(np.median([sigma_x(float(point), spec) for point in resolution_grid]))
    lower = float(spec["kernel_ls_res_lower_factor_2016"]) * local
    factor = float(spec["kernel_ls_res_upper_factor_2016"])
    upper = max(factor * local, factor * global_base * float(spec["kernel_ls_local_hi_floor_factor"]))
    return lower, upper


def interior(constant: float, length: float, mass: float, spec: dict[str, Any]) -> bool:
    lower, upper = length_bounds(mass, spec)
    constant_lower, constant_upper = map(float, spec["kernel_constant_bounds"])
    tolerance = float(spec["kernel_bound_rel_tolerance"])
    return bool(
        np.isfinite([constant, length]).all()
        and not np.isclose(constant, constant_lower, rtol=tolerance, atol=1e-12)
        and not np.isclose(constant, constant_upper, rtol=tolerance, atol=1e-12)
        and not np.isclose(length, lower, rtol=tolerance, atol=1e-12)
        and not np.isclose(length, upper, rtol=tolerance, atol=1e-12)
    )


def load_histogram(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    declaration = spec["full_input"]
    source = REPO / declaration["path"]
    if sha256_file(source) != declaration["file_sha256"]:
        raise RuntimeError("full input hash mismatch")
    with uproot.open(source) as handle:
        values, edges = handle[declaration["histogram"]].to_numpy(flow=False)
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if histogram_hash(values, edges) != declaration["histogram_sha256"]:
        raise RuntimeError("histogram hash mismatch")
    return values, edges


def rebin(values: np.ndarray, edges: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lower = float(spec["support_lower_MeV"]) / 1000.0
    upper = float(spec["support_upper_MeV"]) / 1000.0
    selected_bins = np.flatnonzero((edges[:-1] >= lower - 1e-12) & (edges[1:] <= upper + 1e-12))
    if not selected_bins.size or not np.all(np.diff(selected_bins) == 1):
        raise RuntimeError("support slice failure")
    selected_values = values[selected_bins]
    factor = int(spec["rebin"])
    if selected_values.size % factor:
        raise RuntimeError("rebin phase failure")
    counts = selected_values.reshape(-1, factor).sum(axis=1)
    native_edges = edges[selected_bins[0] : selected_bins[-1] + 2]
    coarse_edges = native_edges[::factor]
    if coarse_edges.size != counts.size + 1:
        coarse_edges = np.append(coarse_edges, native_edges[-1])
    centers = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    return np.asarray(centers), np.asarray(counts)


def expected_path_ids() -> dict[str, str]:
    direct_labels = [
        "archived",
        "v4p9p11_seed_2711",
        "v4p9p11_seed_6043",
        "v4p9p11_seed_9151",
        "card_initializer",
    ]
    for constant in (10, 100, 1000):
        for fraction in ("0p1", "0p5", "0p9"):
            direct_labels.append(f"lattice_c{constant}_f{fraction}")
    result = {f"direct_lbfgsb__{label}": "direct_lbfgsb" for label in direct_labels}
    result.update(
        {
            "powell_lbfgsb__card_initializer": "powell_lbfgsb",
            "powell_lbfgsb__best_prior_source": "powell_lbfgsb",
            "trust_lbfgsb__fixed_lattice_center": "trust_lbfgsb",
            "trust_lbfgsb__best_prior_source": "trust_lbfgsb",
        }
    )
    return result


def recompute_path_row(row: pd.Series, spec: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    family = str(row["method_family"])
    invoked = 1 if family == "direct_lbfgsb" else 2
    stage_success = True
    warning_count = 0
    no_errors = True
    for stage_number in range(1, invoked + 1):
        stage_success = stage_success and as_bool(row[f"stage{stage_number}_success"])
        warning_count += int(row[f"stage{stage_number}_warning_count"])
        no_errors = no_errors and blank(row[f"stage{stage_number}_error"])
    warning_free = bool(blank(row["setup_warnings"]) and warning_count == 0 and int(row["exact_warning_count"]) == 0)
    no_errors = bool(no_errors and blank(row["exact_error"]))
    constant = float(row["postpolish_constant"])
    length = float(row["postpolish_length"])
    finite = bool(np.isfinite([constant, length, float(row["fixed_lml"])]).all())
    inside = interior(constant, length, float(row["mass_GeV"]), spec) if finite else False
    objective_lml = -float(row["optimizer_objective"])
    signed_lml_logged = float(row["optimizer_lml_explicit_negative_objective"])
    objective_difference = objective_lml - float(row["fixed_lml"])
    objective_sign = close(signed_lml_logged, objective_lml, atol=1e-10, rtol=1e-12)
    difference_logged = close(row["optimizer_lml_minus_fixed_lml"], objective_difference, atol=1e-10, rtol=1e-12)
    lml_closure = bool(np.isfinite(objective_difference) and abs(objective_difference) <= float(spec["fixed_lml_abs_tolerance"]))
    gradient_infinity = max(abs(float(row["gradient_constant_log"])), abs(float(row["gradient_length_log"])))
    gradient_logged = close(row["gradient_infinity"], gradient_infinity, atol=1e-12, rtol=1e-12)
    gradient_pass = bool(np.isfinite(gradient_infinity) and gradient_infinity < float(spec["analytic_gradient_infinity_max"]))
    eligible = bool(stage_success and warning_free and no_errors and finite and inside and objective_sign and difference_logged and lml_closure and gradient_logged and gradient_pass)
    comparisons = {
        "stage_success_all": stage_success,
        "stage_warning_count": warning_count,
        "warning_free_all": warning_free,
        "no_stage_or_exact_error": no_errors,
        "coordinates_finite": finite,
        "coordinates_interior": inside,
        "objective_fixed_lml_closure_pass": lml_closure,
        "gradient_pass": gradient_pass,
        "path_eligible": eligible,
    }
    for column, expected in comparisons.items():
        observed = int(row[column]) if column == "stage_warning_count" else as_bool(row[column])
        if observed != expected:
            failures.append(column)
    if not objective_sign:
        failures.append("objective_sign")
    if not difference_logged:
        failures.append("objective_difference")
    if not gradient_logged:
        failures.append("gradient_infinity")
    return eligible, failures


def lognormal_counts(mean_log: np.ndarray, covariance_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diagonal = np.clip(np.diag(covariance_log), 0.0, None)
    mean = np.exp(mean_log + 0.5 * diagonal)
    covariance = np.outer(mean, mean) * (np.exp(np.clip(covariance_log, -40.0, 40.0)) - 1.0)
    return np.asarray(mean), np.asarray(covariance)


def covariance_replay(mean: np.ndarray, covariance: np.ndarray, spec: dict[str, Any]) -> dict[str, Any]:
    total = 0.5 * (covariance + covariance.T) + np.diag(np.clip(mean, 1e-12, None))
    total = 0.5 * (total + total.T)
    diagonal = np.diag(total)
    max_diagonal = float(np.max(diagonal))
    median_diagonal = float(np.median(diagonal))
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(total)))
    negative_ok = min_eigenvalue >= -float(spec["covariance_negative_eigen_rel_tolerance"]) * max_diagonal
    cholesky = None
    jitter = math.nan
    for relative in (0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        try:
            cholesky = np.linalg.cholesky(total + np.eye(len(mean)) * relative * median_diagonal)
            jitter = float(relative)
            break
        except np.linalg.LinAlgError:
            continue
    passed = bool(
        negative_ok
        and cholesky is not None
        and jitter <= float(spec["covariance_max_jitter_rel_median_diag"])
    )
    return {
        "covariance_ok": passed,
        "minimum_eigenvalue": min_eigenvalue,
        "maximum_diagonal": max_diagonal,
        "median_diagonal": median_diagonal,
        "jitter_relative_median_diagonal": jitter,
    }


def replay_selected_state(
    state: pd.Series,
    centers: np.ndarray,
    counts: np.ndarray,
    spec: dict[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    mass = float(state["mass_GeV"])
    half_width = float(spec["blind_nsigma"]) * sigma_2016(mass, spec)
    low = centers < mass - half_width
    high = centers > mass + half_width
    train = low | high
    query = (centers >= mass - half_width) & (centers <= mass + half_width)
    geometry = {
        "n_train": int(np.count_nonzero(train)),
        "n_train_low": int(np.count_nonzero(low)),
        "n_train_high": int(np.count_nonzero(high)),
        "n_query": int(np.count_nonzero(query)),
        "train_centers_sha256": array_hash(centers[train]),
        "train_counts_sha256": array_hash(counts[train]),
        "query_centers_sha256": array_hash(centers[query]),
    }
    for column, expected in geometry.items():
        observed = state[column]
        if (isinstance(expected, int) and int(observed) != expected) or (isinstance(expected, str) and str(observed) != expected):
            failures.append(column)
    two_sidebands = bool(np.count_nonzero(low) > 0 and np.count_nonzero(high) > 0)
    if as_bool(state["two_training_sidebands"]) != two_sidebands:
        failures.append("two_training_sidebands")

    if blank(state["selected_path_id"]):
        expected_certificate = False
        if as_bool(state["selected_state_certificate_pass"]):
            failures.append("no_path_certificate")
        return expected_certificate, failures

    constant = float(state["const_opt"])
    length = float(state["ls_opt"])
    lower, upper = length_bounds(mass, spec)
    kernel = ConstantKernel(constant, tuple(map(float, spec["kernel_constant_bounds"]))) * RBF(length, (lower, upper))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1.0 / counts[train],
        optimizer=None,
        normalize_y=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(np.log(centers[train]).reshape(-1, 1), np.log(counts[train]))
        fixed_lml = float(model.log_marginal_likelihood_value_)
        analytic_lml, gradient = model.log_marginal_likelihood(
            theta=np.log([constant, length]), eval_gradient=True, clone_kernel=True
        )
        mean_sklearn, cov_sklearn = model.predict(np.log(centers[query]).reshape(-1, 1), return_cov=True)

    train_log = np.log(centers[train])
    query_log = np.log(centers[query])
    train_kernel = constant * np.exp(-0.5 * ((train_log[:, None] - train_log[None, :]) / length) ** 2)
    train_kernel = train_kernel + np.diag(1.0 / counts[train])
    cholesky = np.linalg.cholesky(train_kernel)
    cross = constant * np.exp(-0.5 * ((train_log[:, None] - query_log[None, :]) / length) ** 2)
    alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, np.log(counts[train])))
    mean_direct = cross.T @ alpha
    projected = np.linalg.solve(cholesky, cross)
    query_kernel = constant * np.exp(-0.5 * ((query_log[:, None] - query_log[None, :]) / length) ** 2)
    direct_covariance = query_kernel - projected.T @ projected
    direct_covariance = 0.5 * (direct_covariance + direct_covariance.T)
    prediction_pass = bool(
        np.allclose(
            mean_sklearn,
            mean_direct,
            rtol=float(spec["direct_prediction_relative_tolerance"]),
            atol=float(spec["direct_prediction_absolute_tolerance"]),
        )
        and np.allclose(
            cov_sklearn,
            direct_covariance,
            rtol=float(spec["direct_prediction_relative_tolerance"]),
            atol=float(spec["direct_prediction_absolute_tolerance"]),
        )
    )
    mean_counts, covariance_counts = lognormal_counts(np.asarray(mean_sklearn), np.asarray(cov_sklearn))
    covariance = covariance_replay(mean_counts, covariance_counts, spec)
    gradient = np.asarray(gradient, dtype=float)
    gradient_infinity = float(np.max(np.abs(gradient)))
    fixed_closure = abs(fixed_lml - float(state["lml"])) <= float(spec["fixed_lml_abs_tolerance"])
    analytic_closure = abs(float(analytic_lml) - fixed_lml) <= float(spec["fixed_lml_abs_tolerance"])
    inside = interior(constant, length, mass, spec)
    expected_certificate = bool(
        not caught
        and fixed_closure
        and analytic_closure
        and prediction_pass
        and covariance["covariance_ok"]
        and inside
        and gradient_infinity < float(spec["analytic_gradient_infinity_max"])
    )
    exact_values = {
        "reconstruction_warning_count": len(caught),
        "reconstructed_lml": fixed_lml,
        "reconstructed_minus_selected_fixed_lml": fixed_lml - float(state["lml"]),
        "analytic_lml_minus_reconstructed_lml": float(analytic_lml) - fixed_lml,
        "selected_gradient_constant_log": float(gradient[0]),
        "selected_gradient_length_log": float(gradient[1]),
        "selected_gradient_infinity": gradient_infinity,
        "prediction_mean_max_abs_difference": float(np.max(np.abs(mean_sklearn - mean_direct))),
        "prediction_covariance_max_abs_difference": float(np.max(np.abs(cov_sklearn - direct_covariance))),
        "prediction_mean_sha256": array_hash(mean_counts),
        "prediction_covariance_sha256": array_hash(covariance_counts),
        **covariance,
    }
    for column, expected in exact_values.items():
        observed = state[column]
        if isinstance(expected, bool):
            matches = as_bool(observed) == expected
        elif isinstance(expected, str):
            matches = str(observed) == expected
        elif isinstance(expected, int):
            matches = int(observed) == expected
        else:
            matches = close(observed, expected, atol=1e-8, rtol=1e-10)
        if not matches:
            failures.append(column)
    booleans = {
        "reconstruction_success": True,
        "selected_coordinates_interior": inside,
        "prediction_closure_pass": prediction_pass,
        "covariance_ok": covariance["covariance_ok"],
        "selected_state_certificate_pass": expected_certificate,
    }
    for column, expected in booleans.items():
        if as_bool(state[column]) != expected:
            failures.append(column)
    return expected_certificate, failures


def validate() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    record("protocol_hash", sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA)
    record("spec_hash", sha256_file(SPEC_PATH) == EXPECTED_SPEC_SHA)
    record("runner_hash", sha256_file(RUNNER_PATH) == EXPECTED_RUNNER_SHA)
    record("preflight_hash", sha256_file(PREFLIGHT_PATH) == EXPECTED_PREFLIGHT_SHA)
    protocol_freeze = read_freeze(HERE / "FROZEN_PROTOCOL_SHA256")
    execution_freeze = read_freeze(HERE / "FROZEN_EXECUTION_SHA256")
    record(
        "protocol_freeze",
        protocol_freeze
        == {
            "STUDY_PROTOCOL.md": EXPECTED_PROTOCOL_SHA,
            "../../study_configs/v4p9p11p1_2016_reference30_uniform_optimizer_remediation_20260902/study_spec.json": EXPECTED_SPEC_SHA,
        },
        protocol_freeze,
    )
    for relative, digest in execution_freeze.items():
        target = (HERE / relative).resolve()
        if not target.is_file() or sha256_file(target) != digest:
            record("execution_freeze_file_hashes", False, relative)
            break
    else:
        record("execution_freeze_file_hashes", True, execution_freeze)
    record(
        "execution_freeze_required_entries",
        execution_freeze.get("run_uniform_remediation.py") == EXPECTED_RUNNER_SHA
        and execution_freeze.get("qa/preflight.json") == EXPECTED_PREFLIGHT_SHA
        and execution_freeze.get("validate_release.py") == sha256_file(Path(__file__))
        and execution_freeze.get("STUDY_PROTOCOL.md") == EXPECTED_PROTOCOL_SHA
        and execution_freeze.get("../../study_configs/v4p9p11p1_2016_reference30_uniform_optimizer_remediation_20260902/study_spec.json") == EXPECTED_SPEC_SHA,
    )

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    input_failures = []
    declarations = [spec["reviewed_reference_card"], spec["archived_states"], *spec["v4p9p11_inputs"].values()]
    for declaration in declarations:
        path = REPO / declaration["path"]
        if not path.is_file() or sha256_file(path) != declaration["sha256"]:
            input_failures.append(declaration["path"])
    record("declared_input_hashes", not input_failures, input_failures)
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    record(
        "preflight_semantics",
        preflight.get("status") == "preflight_pass"
        and preflight.get("mass_rows") == 142
        and preflight.get("planned_path_rows") == 2556
        and preflight.get("mass_grid_MeV") == EXPECTED_MASSES
        and preflight.get("per_mass_method_multiplicity")
        == {"direct_lbfgsb": 14, "powell_lbfgsb": 2, "trust_lbfgsb": 2}
        and preflight.get("best_prior_pool_exact")
        == ["archived", "v4p9p11_seed_2711", "v4p9p11_seed_6043", "v4p9p11_seed_9151"]
        and preflight.get("lattice_center") == {"constant": 100.0, "log_length_fraction": 0.5}
        and preflight.get("forbidden_inference_fields_accessed") == [],
    )

    paths = pd.read_csv(PATHS_PATH)
    states = pd.read_csv(STATES_PATH)
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    forbidden_columns = [
        column
        for column in list(paths.columns) + list(states.columns)
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    ]
    record("no_forbidden_inference_columns", not forbidden_columns, forbidden_columns)
    record(
        "no_inference_decision_fields",
        decision.get("inference_fields_accessed") == [] and decision.get("inference_artifacts_produced") == [],
    )

    expected_ids = expected_path_ids()
    grid_ok = bool(len(paths) == 2556 and len(states) == 142 and states["mass_MeV"].astype(int).tolist() == EXPECTED_MASSES)
    id_failures: list[int] = []
    for mass in EXPECTED_MASSES:
        group = paths.loc[paths["mass_MeV"].astype(int) == mass]
        actual = dict(zip(group["path_id"].astype(str), group["method_family"].astype(str)))
        if len(group) != 18 or actual != expected_ids:
            id_failures.append(mass)
    record("exact_142_2556_grid", grid_ok and not id_failures, id_failures)

    replay_eligible: dict[int, dict[str, bool]] = {}
    path_failures: list[dict[str, Any]] = []
    for index, row in paths.iterrows():
        eligible, failures = recompute_path_row(row, spec)
        mass = int(row["mass_MeV"])
        replay_eligible.setdefault(mass, {})[str(row["path_id"])] = eligible
        if failures:
            path_failures.append({"row": int(index), "mass_MeV": mass, "path_id": str(row["path_id"]), "fields": failures})
    record("path_eligibility_replay", not path_failures, path_failures[:20])

    values, edges = load_histogram(spec)
    centers, counts = rebin(values, edges, spec)
    selection_failures: list[dict[str, Any]] = []
    state_replay_failures: list[dict[str, Any]] = []
    replay_state_resolved: dict[int, bool] = {}
    for mass in EXPECTED_MASSES:
        group = paths.loc[paths["mass_MeV"].astype(int) == mass].copy()
        state = states.loc[states["mass_MeV"].astype(int) == mass].iloc[0]
        eligible = group.loc[
            group["path_id"].astype(str).map(replay_eligible[mass]).astype(bool)
        ].copy()
        selection_fields: list[str] = []
        if len(eligible):
            eligible = eligible.sort_values(["fixed_lml", "path_id"], ascending=[False, True])
            selected = eligible.iloc[0]
            selected_lml = float(selected["fixed_lml"])
            selected_constant = float(selected["postpolish_constant"])
            selected_length = float(selected["postpolish_length"])
            cluster = eligible.loc[
                (eligible["fixed_lml"].astype(float).sub(selected_lml).abs() <= float(spec["cluster_lml_abs_tolerance"]))
                & (eligible["postpolish_constant"].astype(float).sub(selected_constant).abs() <= float(spec["cluster_coordinate_rel_tolerance"]) * abs(selected_constant))
                & (eligible["postpolish_length"].astype(float).sub(selected_length).abs() <= float(spec["cluster_coordinate_rel_tolerance"]) * abs(selected_length))
            ]
            cluster_ids = sorted(cluster["path_id"].astype(str).tolist())
            cluster_families = sorted(cluster["method_family"].astype(str).unique().tolist())
            cluster_pass = bool(
                len(cluster) >= int(spec["cluster_minimum_paths"])
                and len(cluster_families) >= int(spec["cluster_minimum_method_families"])
            )
            joins = {
                "selected_path_id": str(selected["path_id"]),
                "selected_method_family": str(selected["method_family"]),
                "const_opt": selected_constant,
                "ls_opt": selected_length,
                "lml": selected_lml,
                "eligible_path_count": len(eligible),
                "selected_cluster_path_count": len(cluster),
                "selected_cluster_method_family_count": len(cluster_families),
                "selected_cluster_method_families": "|".join(cluster_families),
                "selected_cluster_path_ids": "|".join(cluster_ids),
            }
            for column, expected in joins.items():
                observed = state[column]
                if isinstance(expected, str):
                    matches = str(observed) == expected
                elif isinstance(expected, int):
                    matches = int(observed) == expected
                else:
                    matches = close(observed, expected, atol=1e-9, rtol=1e-10)
                if not matches:
                    selection_fields.append(column)
        else:
            cluster_pass = False
            if not blank(state["selected_path_id"]) or int(state["eligible_path_count"]) != 0:
                selection_fields.append("no_eligible_selection")
        if as_bool(state["selected_cluster_pass"]) != cluster_pass:
            selection_fields.append("selected_cluster_pass")
        if selection_fields:
            selection_failures.append({"mass_MeV": mass, "fields": selection_fields})

        certificate, replay_failures = replay_selected_state(state, centers, counts, spec)
        resolved = bool(cluster_pass and certificate and as_bool(state["two_training_sidebands"]))
        replay_state_resolved[mass] = resolved
        if as_bool(state["state_resolved"]) != resolved:
            replay_failures.append("state_resolved")
        if replay_failures:
            state_replay_failures.append({"mass_MeV": mass, "fields": replay_failures})
    record("global_max_and_cluster_replay", not selection_failures, selection_failures[:20])
    record("selected_state_prediction_covariance_replay", not state_replay_failures, state_replay_failures[:20])

    resolved_masses = [mass for mass in EXPECTED_MASSES if replay_state_resolved[mass]]
    unresolved_masses = [mass for mass in EXPECTED_MASSES if not replay_state_resolved[mass]]
    all_resolved = len(resolved_masses) == 142
    decision_expected = bool(
        decision.get("status") == ("all_142_states_certified" if all_resolved else "stopped_unresolved_state")
        and decision.get("combination_authorized") is all_resolved
        and decision.get("state_rows") == 142
        and decision.get("resolved_rows") == len(resolved_masses)
        and decision.get("unresolved_masses_MeV") == unresolved_masses
        and decision.get("optimizer_paths") == {"rows": 2556, "sha256": sha256_file(PATHS_PATH)}
        and decision.get("states") == {"rows": 142, "sha256": sha256_file(STATES_PATH)}
        and decision.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA
        and decision.get("spec_sha256") == EXPECTED_SPEC_SHA
        and decision.get("script_sha256") == EXPECTED_RUNNER_SHA
        and decision.get("preflight_sha256") == EXPECTED_PREFLIGHT_SHA
        and decision.get("lower_lml_branch_substitution_permitted") is False
    )
    record("decision_semantics_and_hashes", decision_expected, {"resolved": len(resolved_masses), "unresolved": unresolved_masses})
    record("global_stop_or_complete_rule", decision.get("combination_authorized") is bool(all_resolved))

    passed = bool(all(item["pass"] for item in checks))
    return {
        "status": "validation_pass" if passed else "validation_failure",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(int(item["pass"]) for item in checks),
        "checks_total": len(checks),
        "all_checks_pass": passed,
        "canonical_outcome": decision.get("status"),
        "canonical_decision_sha256": sha256_file(DECISION_PATH),
        "optimizer_paths_sha256": sha256_file(PATHS_PATH),
        "states_sha256": sha256_file(STATES_PATH),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "spec_sha256": EXPECTED_SPEC_SHA,
        "runner_sha256": EXPECTED_RUNNER_SHA,
        "preflight_sha256": EXPECTED_PREFLIGHT_SHA,
        "validator_sha256": sha256_file(Path(__file__)),
        "inference_fields_accessed": [],
        "checks": checks,
    }


def main() -> None:
    try:
        payload = validate()
    except Exception as exception:
        payload = {
            "status": "validation_exception",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "all_checks_pass": False,
            "error": repr(exception),
            "validator_sha256": sha256_file(Path(__file__)),
            "inference_fields_accessed": [],
        }
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload.get("all_checks_pass"):
        sys.exit(1)


if __name__ == "__main__":
    main()
