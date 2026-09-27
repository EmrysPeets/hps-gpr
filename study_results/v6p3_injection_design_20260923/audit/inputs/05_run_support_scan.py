#!/usr/bin/env python3
"""Run the v4.9.5 native-10% lower-support-edge scan.

This driver preserves the frozen v4.2 2021 analysis card and the effective
v4.5 refmatched, fixed-histogram signal-injection semantics.  It uses the
v4.6 deterministic optimizer-repeat gate so that optimizer selection is based
only on likelihood and branch reproducibility, never on the fitted signal,
pull, recovery, or epsilon-squared coordinate.

The functional family used to construct the cached pseudo-histograms is not
encoded here.  The only required ROOT layout is

    toys/fsig_anchor/{scenario}/toy_NNNN

for the native-10% scenario at toy indices 0--99.  Indices 0--24 reproduce the
v4.9 pilot exactly and 25--99 are an independently seeded continuation.  This
runner extracts the background-only and 2- and 5-sigma injection states at
55, 60, 65, and 70 MeV for one explicitly selected lower support edge.  All
scientific inputs,
the frozen card, and the narrowly scoped runtime instrumentation are verified
against SHA-256 declarations in ``study_spec.json`` before production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RUNTIME_ROOT = HERE / "runtime_overlay"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

SPEC_PATH = HERE / "study_spec.json"
INPUTS = HERE / "inputs"
SUPPORT_RANGES = {
    "030_300": (0.030, 0.300),
    "032_300": (0.032, 0.300),
    "034_300": (0.034, 0.300),
    "036_300": (0.036, 0.300),
    "038_300": (0.038, 0.300),
    "040_300": (0.040, 0.300),
}
SUPPORT_MODE = os.environ.get("V4P9P5_SUPPORT", "")
if SUPPORT_MODE not in SUPPORT_RANGES:
    raise SystemExit(
        "Set V4P9P5_SUPPORT to one of " + ", ".join(SUPPORT_RANGES)
    )
SUPPORT_RANGE = SUPPORT_RANGES[SUPPORT_MODE]
RUNS = HERE / "runs" / f"native10_fsig_{SUPPORT_MODE}"
DERIVED = HERE / "derived" / f"native10_fsig_{SUPPORT_MODE}"
QA = HERE / "qa"

SCENARIOS = ("2021_10pct",)
MASS_GRID = (0.055, 0.060, 0.065, 0.070)
STRENGTH_GRID = (0.0, 2.0, 5.0)
N_TOYS = 100
TOY_CONTAINER_PREFIX = "toys/fsig_anchor"
BASE_SEED = 20260817
LEDGER_FILES = (
    "optimizer_attempts.csv",
    "accepted_rows.csv",
    "raw_primary_rows.csv",
    "exclusions.csv",
)
V4P6_COMPATIBILITY_CARD_SHA256 = (
    "5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055"
)
V4P6_COMPATIBILITY_RUNTIME_SHA256 = {
    "hps_gpr/gpr.py": "1c83cae238e87a4e94928c97fb737947c22a3f88b16dfaf955d48ab6b4771dd5",
    "hps_gpr/io.py": "b36f8da7671a0fc0958b663e11d83a1a4421e90d1aab9b10e40c31ce078035db",
}


class StudyError(RuntimeError):
    """Raised when the frozen study contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_spec() -> dict[str, Any]:
    if not SPEC_PATH.is_file():
        raise StudyError(f"missing study specification: {SPEC_PATH}")
    payload = load_json(SPEC_PATH)
    if int(payload.get("schema_version", -1)) != 1:
        raise StudyError("unsupported or missing study specification schema_version")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: Any, dtype: str) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=dtype).tobytes(order="C")
    ).hexdigest()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(namespace: str, *parts: object) -> int:
    material = "|".join(
        [str(BASE_SEED), str(namespace), *[str(part) for part in parts]]
    )
    return int.from_bytes(
        hashlib.sha256(material.encode("utf-8")).digest()[:4], "little"
    )


def wald_cls_upper_limit(
    a_hat: float, sigma_a: float, *, alpha: float = 0.10
) -> float:
    """Return the bounded Wald-asymptotic CLs upper limit on signal yield.

    This is a fast per-pseudo-dataset diagnostic derived from the accepted
    profiled amplitude and uncertainty.  It uses the same Cowan-style
    ``tilde_q_mu`` branches and CLs tail ratio as the full likelihood code, but
    it does not rerun the nuisance profiling at each tested amplitude.  The
    final observed-data scan is evaluated with the full profiled likelihood.
    """
    a_hat = float(a_hat)
    sigma_a = float(sigma_a)
    alpha = float(alpha)
    if not np.isfinite(a_hat) or not np.isfinite(sigma_a) or sigma_a <= 0:
        return float("nan")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    def cls_at(a_test: float) -> float:
        a_test = max(float(a_test), 0.0)
        if a_test <= 0.0:
            return 1.0
        if a_hat > a_test:
            q_obs = 0.0
        elif a_hat >= 0.0:
            q_obs = ((a_test - a_hat) / sigma_a) ** 2
        else:
            q_obs = (a_test * a_test - 2.0 * a_test * a_hat) / (sigma_a**2)
        q_asimov = (a_test / sigma_a) ** 2
        sqrt_obs = math.sqrt(max(q_obs, 0.0))
        sqrt_asimov = math.sqrt(max(q_asimov, 0.0))
        cl_sb = float(norm.sf(sqrt_obs))
        cl_b = max(float(norm.cdf(sqrt_asimov - sqrt_obs)), 1.0e-12)
        return cl_sb / cl_b

    lo = 0.0
    hi = max(1.0, max(a_hat, 0.0) + 5.0 * sigma_a)
    for _ in range(40):
        if cls_at(hi) <= alpha:
            break
        hi *= 2.0
    else:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cls_at(mid) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo <= max(1.0e-10, 1.0e-9 * hi):
            break
    return 0.5 * (lo + hi)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
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


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def configure_process() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"


def resolve_study_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return HERE / path


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise StudyError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != str(expected):
        raise StudyError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}: {path}"
        )


def _paired_path_key(record: Mapping[str, Any], hash_key: str) -> str | None:
    if hash_key == "sha256":
        for candidate in (
            "archived_path",
            "path",
            "root",
            "manifest",
            "metadata",
            "file",
            "production_driver",
        ):
            if candidate in record:
                return candidate
        return None
    if not hash_key.endswith("_sha256"):
        return None
    stem = hash_key[: -len("_sha256")]
    for candidate in (stem, f"{stem}_path", f"archived_{stem}_path"):
        if candidate in record:
            return candidate
    if stem == "config" and "archived_config_path" in record:
        return "archived_config_path"
    return None


def verify_declared_hashes(
    payload: Any,
    *,
    label: str,
    checks: dict[str, bool],
) -> None:
    """Recursively verify adjacent path/SHA-256 declarations.

    Declarations such as ``base_sha256`` that have no adjacent path are
    provenance-only and are intentionally skipped.
    """

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            path_key = _paired_path_key(payload, str(key))
            if path_key is None:
                continue
            declared_path = payload.get(path_key)
            if not isinstance(declared_path, str) or not declared_path:
                continue
            path = resolve_study_path(declared_path)
            check_name = f"{label}.{path_key}"
            require_hash(path, value, check_name)
            checks[check_name] = True
        for key, value in payload.items():
            if isinstance(value, (Mapping, list, tuple)):
                verify_declared_hashes(
                    value, label=f"{label}.{key}", checks=checks
                )
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            if isinstance(value, (Mapping, list, tuple)):
                verify_declared_hashes(
                    value, label=f"{label}[{index}]", checks=checks
                )


def background_root(spec: Mapping[str, Any]) -> Path:
    product = spec.get("background_toy_product", {})
    value = product.get("root")
    if not isinstance(value, str) or not value:
        raise StudyError("study_spec background_toy_product.root is missing")
    return resolve_study_path(value)


def toy_key(scenario: str, toy_index: int) -> str:
    return f"{TOY_CONTAINER_PREFIX}/{scenario}/toy_{int(toy_index):04d}"


def assert_spec_contract(spec: Mapping[str, Any]) -> None:
    scenarios = spec.get("scenarios", {})
    if not isinstance(scenarios, Mapping) or set(scenarios) != set(SCENARIOS):
        raise StudyError(
            f"scenario content must be exactly {SCENARIOS}, got {tuple(scenarios)}"
        )
    expected_scenario_semantics = {
        "2021_10pct": ("ten_pct", 1, 141_321_937),
    }
    for scenario, (family, multiplier, target_count) in expected_scenario_semantics.items():
        record = scenarios.get(scenario, {})
        try:
            actual_multiplier = int(record.get("source_multiplier", -1))
            actual_target = int(record.get("normalization_target_count", -1))
        except (TypeError, ValueError):
            actual_multiplier = -1
            actual_target = -1
        if (
            str(record.get("source_family")) != family
            or actual_multiplier != multiplier
            or actual_target != target_count
        ):
            raise StudyError(
                f"scenario semantics drift for {scenario}: expected "
                f"source_family={family}, source_multiplier={multiplier}, "
                f"normalization_target_count={target_count}"
            )
    masses = tuple(float(value) for value in spec.get("masses_gev", ()))
    strengths = tuple(float(value) for value in spec.get("sigma_strengths", ()))
    toy_indices = tuple(int(value) for value in spec.get("toy_indices", ()))
    if masses != MASS_GRID:
        raise StudyError(f"mass grid drift: expected {MASS_GRID}, got {masses}")
    if strengths != STRENGTH_GRID:
        raise StudyError(
            f"injection-strength grid drift: expected {STRENGTH_GRID}, got {strengths}"
        )
    if toy_indices != tuple(range(N_TOYS)):
        raise StudyError("toy_indices must be the predeclared contiguous range 0--99")

    declared_supports = tuple(
        float(value)
        for value in spec.get("analysis_card", {}).get(
            "candidate_gp_support_low_edges_gev", ()
        )
    )
    expected_supports = tuple(value[0] for value in SUPPORT_RANGES.values())
    if declared_supports != expected_supports:
        raise StudyError(
            "support-candidate drift: expected "
            f"{expected_supports}, got {declared_supports}"
        )

    product = spec.get("background_toy_product", {})
    if str(product.get("container_prefix", TOY_CONTAINER_PREFIX)) != TOY_CONTAINER_PREFIX:
        raise StudyError(
            f"background ROOT container must be {TOY_CONTAINER_PREFIX}"
        )

    expected_gate = {
        "version": "v4p7p1_reference_relative_v1",
        "reference_initial_attempts": 3,
        "maximum_attempts": 5,
        "top_branch_min_replicates": 2,
        "delta_lml_per_train_max": 0.001,
        "abs_log_length_ratio_max": 0.01,
        "abs_log_constant_ratio_max": 0.05,
        "abs_log_sigma_ratio_max": 0.02,
        "exact_start_abs_log_theta_max": 1e-8,
        "bound_ratio_window": 0.02,
        "sigma_over_reference_trigger": [0.5, 2.0],
        "reference_relative_lml_per_train_trigger": 0.02,
        "reference_relative_abs_log_length_trigger": 0.05,
        "reference_relative_abs_log_constant_trigger": 0.10,
        "covariance_min_eigenvalue_relative": -0.01,
        "minimum_accepted_per_cell_for_closure_claim": 95,
    }
    gate = spec.get("optimizer_gate", {})
    mismatches = []
    for key, expected in expected_gate.items():
        actual = gate.get(key)
        if isinstance(expected, float):
            valid = actual is not None and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-15
            )
        elif isinstance(expected, list):
            valid = [float(v) for v in actual or ()] == expected
        else:
            valid = actual == expected
        if not valid:
            mismatches.append(f"{key}={actual!r} (expected {expected!r})")
    if mismatches:
        raise StudyError("optimizer-gate drift: " + "; ".join(mismatches))


def build_config() -> Any:
    from hps_gpr.config import load_config

    spec = load_spec()
    state = spec.get("declared_result_state", {})
    card_value = state.get(
        "archived_config_path", "inputs/frozen_v4p2_analysis_card.yaml"
    )
    card_path = resolve_study_path(str(card_value))
    cfg = load_config(str(card_path))
    cfg.enable_2015 = False
    cfg.enable_2016 = False
    cfg.enable_2021 = True
    cfg.do_combined = False
    cfg.make_ul_bands = False
    cfg.ul_bands_toys = 0
    cfg.do_combined_bands = False
    cfg.combined_bands_n_toys = 0
    cfg.make_eps2_bands = False
    cfg.cls_mode = "asymptotic"
    cfg.cls_num_toys = 0
    cfg.kernel_ls_res_lower_factor_by_dataset = dict(
        cfg.kernel_ls_res_lower_factor_by_dataset
    )
    cfg.kernel_ls_res_upper_factor_by_dataset = dict(
        cfg.kernel_ls_res_upper_factor_by_dataset
    )
    cfg.kernel_ls_res_lower_factor_by_dataset["2021"] = 1.1
    cfg.kernel_ls_res_upper_factor_by_dataset["2021"] = 15.0
    cfg.data_range_2021 = list(SUPPORT_RANGE)
    cfg.blind_nsigma = 2.25
    cfg.gp_train_exclude_nsigma = 2.25
    cfg.scan_edge_guard_nsigma = 2.25
    cfg.scan_require_two_sidebands = True
    cfg.neighborhood_rebin = 5
    cfg.n_restarts = 12
    cfg.extract_allow_negative = True
    cfg.extract_background_mode = "profiled"
    cfg.eps2_density_nsigma = 1.64
    cfg.signal_model = "default"
    cfg.fail_fast = True
    cfg.debug_print = False
    cfg.save_plots = False
    return cfg


def assert_config(cfg: Any) -> None:
    checks = {
        "range_2021": tuple(map(float, cfg.range_2021)) == (0.05, 0.25),
        "data_range_2021": tuple(map(float, cfg.data_range_2021))
        == SUPPORT_RANGE,
        "pre_log": bool(cfg.pre_log),
        "alpha_model": str(cfg.alpha_model) == "1/y",
        "pre_zero_alpha": float(cfg.pre_zero_alpha) == 1.0,
        "lower_factor": float(
            cfg.kernel_ls_res_lower_factor_by_dataset["2021"]
        )
        == 1.1,
        "upper_factor": float(
            cfg.kernel_ls_res_upper_factor_by_dataset["2021"]
        )
        == 15.0,
        "blind_nsigma": float(cfg.blind_nsigma) == 2.25,
        "gp_train_exclude_nsigma": float(cfg.gp_train_exclude_nsigma) == 2.25,
        "scan_edge_guard_nsigma": float(cfg.scan_edge_guard_nsigma) == 2.25,
        "two_sidebands": bool(cfg.scan_require_two_sidebands),
        "neighborhood_rebin": int(cfg.neighborhood_rebin) == 5,
        "n_restarts": int(cfg.n_restarts) == 12,
        "signed_amplitude": bool(cfg.extract_allow_negative),
        "profiled_background": str(cfg.extract_background_mode) == "profiled",
        "density": float(cfg.eps2_density_nsigma) == 1.64,
        "signal_model": str(cfg.signal_model) == "default",
        "no_limit_bands": not bool(cfg.make_ul_bands)
        and not bool(cfg.do_combined_bands)
        and not bool(cfg.make_eps2_bands),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise StudyError("frozen-card assertion failed: " + ", ".join(failed))


def validate_toy_product(spec: Mapping[str, Any]) -> dict[str, Any]:
    import uproot

    path = background_root(spec)
    product = spec["background_toy_product"]
    manifest_path = resolve_study_path(str(product["manifest"]))
    manifest = load_json(manifest_path)
    manifest_content = dict(manifest)
    recorded_content_hash = manifest_content.pop("manifest_content_sha256", None)
    if not isinstance(recorded_content_hash, str) or canonical_json_hash(
        manifest_content
    ) != recorded_content_hash:
        raise StudyError("background toy manifest content hash mismatch")
    if int(manifest.get("n_toys_per_scenario", -1)) != N_TOYS:
        raise StudyError("background toy manifest n_toys_per_scenario drift")
    if set(manifest.get("scenarios", ())) != set(SCENARIOS):
        raise StudyError("background toy manifest scenario drift")
    expected_template = "toys/fsig_anchor/{scenario}/toy_{toy_index:04d}"
    if str(manifest.get("toy_key_template")) != expected_template:
        raise StudyError("background toy manifest key-template drift")
    manifest_toys: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in manifest.get("toys", ()):
        key = (str(row.get("scenario")), int(row.get("toy_index", -1)))
        if key in manifest_toys:
            raise StudyError(f"duplicate background toy manifest key: {key}")
        manifest_toys[key] = row
    expected_manifest_keys = {
        (scenario, toy_index)
        for scenario in SCENARIOS
        for toy_index in range(N_TOYS)
    }
    if set(manifest_toys) != expected_manifest_keys:
        raise StudyError("background toy manifest inventory is incomplete")

    expected_edges: np.ndarray | None = None
    count_hashes: dict[str, str] = {}
    with uproot.open(path) as root_file:
        for scenario in SCENARIOS:
            for toy_index in range(N_TOYS):
                key = toy_key(scenario, toy_index)
                if key not in root_file:
                    raise StudyError(f"missing predeclared toy histogram: {key}")
                values, edges = root_file[key].to_numpy()
                values = np.asarray(values, dtype=float)
                edges = np.asarray(edges, dtype=float)
                if values.ndim != 1 or edges.shape != (values.size + 1,):
                    raise StudyError(f"invalid one-dimensional histogram: {key}")
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise StudyError(f"nonfinite or negative toy counts: {key}")
                rounded = np.rint(values)
                if not np.allclose(values, rounded, rtol=0.0, atol=1e-6):
                    raise StudyError(f"toy counts are not integer-like: {key}")
                if float(edges[0]) > 0.03 or float(edges[-1]) < 0.30:
                    raise StudyError(f"toy histogram does not cover 30--300 MeV: {key}")
                if expected_edges is None:
                    expected_edges = edges.copy()
                elif not np.array_equal(edges, expected_edges):
                    raise StudyError(f"toy edge mismatch: {key}")
                centers = 0.5 * (edges[:-1] + edges[1:])
                outside_support = (centers < 0.03) | (centers >= 0.30)
                if np.any(rounded[outside_support] != 0):
                    raise StudyError(f"toy has nonzero counts outside 30--300 MeV: {key}")
                counts_digest = array_hash(rounded, "<i8")
                manifest_row = manifest_toys[(scenario, toy_index)]
                if str(manifest_row.get("output_histogram")) != key:
                    raise StudyError(f"manifest ROOT key mismatch: {key}")
                if str(manifest_row.get("counts_sha256")) != counts_digest:
                    raise StudyError(f"manifest count hash mismatch: {key}")
                if int(manifest_row.get("total_count", -1)) != int(np.sum(rounded)):
                    raise StudyError(f"manifest count total mismatch: {key}")
                count_hashes[key] = counts_digest

        manifest_truths = {
            str(row.get("scenario")): row for row in manifest.get("truths", ())
        }
        if set(manifest_truths) != set(SCENARIOS):
            raise StudyError("background toy manifest truth inventory drift")
        analytic_keys = product.get("analytic_mean_keys", {})
        baseline_keys = product.get("baseline_analytic_mean_keys", {})
        for scenario in SCENARIOS:
            truth_row = manifest_truths[scenario]
            analytic_key = str(truth_row.get("analytic_mean_key", ""))
            baseline_key = str(truth_row.get("baseline_analytic_mean_key", ""))
            if str(analytic_keys.get(scenario, "")) != analytic_key:
                raise StudyError(f"analytic-mean key mismatch for {scenario}")
            if str(baseline_keys.get(scenario, "")) != baseline_key:
                raise StudyError(f"baseline analytic-mean key mismatch for {scenario}")
            values, edges = root_file[analytic_key].to_numpy()
            if array_hash(values, "<f8") != str(
                truth_row.get("mean_sha256_float64", "")
            ):
                raise StudyError(f"analytic-mean hash mismatch for {scenario}")
            expected_total = float(
                spec["scenarios"][scenario]["normalization_target_count"]
            )
            if not math.isclose(
                float(np.sum(values)),
                expected_total,
                rel_tol=0.0,
                abs_tol=max(1e-3, expected_total * 1e-9),
            ):
                raise StudyError(f"analytic-mean total mismatch for {scenario}")
            if baseline_key not in root_file:
                raise StudyError(f"missing baseline analytic mean: {baseline_key}")
    return {
        "status": "pass",
        "root": str(path),
        "histograms": len(count_hashes),
        "bins_per_histogram": int(len(expected_edges) - 1)
        if expected_edges is not None
        else 0,
        "histogram_range_gev": [
            float(expected_edges[0]), float(expected_edges[-1])
        ]
        if expected_edges is not None
        else [],
        "toy_truth_support_gev": [0.03, 0.30],
        "extraction_gp_support_gev": list(SUPPORT_RANGE),
        "manifest_content_sha256": recorded_content_hash,
        "counts_sha256": count_hashes,
    }


def preflight(*, validate_inventory: bool = True) -> dict[str, Any]:
    spec = load_spec()
    assert_spec_contract(spec)
    checks: dict[str, bool] = {}

    state = spec.get("declared_result_state", {})
    card_path = resolve_study_path(
        str(state.get("archived_config_path", "inputs/frozen_v4p2_analysis_card.yaml"))
    )
    config_sha = state.get("config_sha256")
    if not isinstance(config_sha, str) or not config_sha:
        raise StudyError("declared_result_state.config_sha256 is missing")
    require_hash(card_path, config_sha, "frozen v4.2 card")
    if config_sha != V4P6_COMPATIBILITY_CARD_SHA256:
        raise StudyError("card is not hash-compatible with the archived v4.6 state")
    checks["frozen_v4p2_card"] = True

    product = spec.get("background_toy_product", {})
    root_path = background_root(spec)
    root_sha = product.get("root_sha256")
    if not isinstance(root_sha, str) or not root_sha:
        raise StudyError("background_toy_product.root_sha256 is missing")
    require_hash(root_path, root_sha, "near-threshold background toys")
    checks["background_toy_root"] = True
    manifest_value = product.get("manifest")
    manifest_sha = product.get("manifest_sha256")
    if not isinstance(manifest_value, str) or not isinstance(manifest_sha, str):
        raise StudyError("background toy manifest path/hash declarations are missing")
    require_hash(
        resolve_study_path(manifest_value), manifest_sha, "background toy manifest"
    )
    checks["background_toy_manifest"] = True

    for section_name in (
        "source_inputs",
        "functional_form_models",
        "fit_products",
        "model_products",
    ):
        section = spec.get(section_name)
        if section is not None:
            verify_declared_hashes(
                section, label=section_name, checks=checks
            )

    fit_record = spec.get("fit_product", spec.get("fit_summary"))
    if isinstance(fit_record, Mapping):
        fit_path_value = fit_record.get("path", fit_record.get("file"))
        fit_sha = fit_record.get("sha256")
        if isinstance(fit_path_value, str) and isinstance(fit_sha, str):
            fit_path = resolve_study_path(fit_path_value)
            require_hash(fit_path, fit_sha, "near-threshold fit summary")
            fit_payload = load_json(fit_path)
            if not bool(fit_payload.get("model_selection_frozen_before_injection")):
                raise StudyError(
                    "near-threshold model selection was not frozen before injection"
                )
            for family in ("one_pct", "ten_pct"):
                record = fit_payload.get("fits", {}).get(family, {})
                selected_degree = int(record.get("selected_degree", -1))
                selected_candidates = [
                    candidate
                    for candidate in record.get("candidate_degrees", ())
                    if int(candidate.get("degree", -2)) == selected_degree
                ]
                if (
                    len(selected_candidates) != 1
                    or not bool(selected_candidates[0].get("passes_fixed_gates"))
                    or bool(record.get("selection_uses_injection_results", True))
                ):
                    raise StudyError(
                        f"selected near-threshold fit fails fixed gates: {family}"
                    )
            checks["fit_summary_scientific_gate"] = True

    runtime = spec.get("runtime_instrumentation", {})
    modules = runtime.get("modules", {})
    if not isinstance(modules, Mapping) or not modules:
        raise StudyError("runtime_instrumentation.modules is missing")
    for module_path_text, record in modules.items():
        if not isinstance(record, Mapping):
            raise StudyError(f"invalid runtime record for {module_path_text}")
        expected = record.get("sha256")
        archived = record.get("archived_path")
        if not isinstance(expected, str) or not isinstance(archived, str):
            raise StudyError(f"runtime hash/path is missing for {module_path_text}")
        require_hash(
            resolve_study_path(archived), expected, f"archived runtime {module_path_text}"
        )
        require_hash(
            RUNTIME_ROOT / str(module_path_text),
            expected,
            f"attested runtime overlay {module_path_text}",
        )
        compatibility_hash = V4P6_COMPATIBILITY_RUNTIME_SHA256.get(
            str(module_path_text)
        )
        if compatibility_hash is None or expected != compatibility_hash:
            raise StudyError(
                f"runtime is not hash-compatible with archived v4.6: {module_path_text}"
            )
        checks[f"runtime.{module_path_text}"] = True
    driver_hash = runtime.get("production_driver_sha256")
    if isinstance(driver_hash, str) and driver_hash:
        require_hash(Path(__file__).resolve(), driver_hash, "production driver")
        checks["production_driver"] = True

    cfg = build_config()
    assert_config(cfg)
    checks["frozen_card_assertions"] = True
    inventory = validate_toy_product(spec) if validate_inventory else None
    if inventory is not None:
        checks["toy_inventory"] = True
    return {
        "status": "pass",
        "validated_utc": utc_now(),
        "checks": checks,
        "toy_inventory": inventory,
    }


def _load_histogram(path: Path, key: str) -> Any:
    from hps_gpr.funcform_toys import load_funcform_toy_hist

    container, name = key.rsplit("/", 1)
    return load_funcform_toy_hist(
        str(path), container=container, toy_name=name
    )


def make_toy_dataset(scenario: str, toy_index: int, cfg: Any) -> Any:
    from hps_gpr.dataset import make_datasets
    from hps_gpr.funcform_toys import FuncFormToySpec, build_funcform_toy_dataset

    spec = load_spec()
    path = background_root(spec)
    key = toy_key(scenario, int(toy_index))
    histogram = _load_histogram(path, key)
    base = make_datasets(cfg)["2021"]
    scenario_record = spec.get("scenarios", {}).get(scenario, {})
    truth_label = str(
        scenario_record.get(
            "function_tag",
            spec.get("background_toy_product", {}).get("truth_model", ""),
        )
    )
    if not truth_label:
        raise StudyError(f"missing functional-truth label for {scenario}")
    toy_spec = FuncFormToySpec(
        source_root=str(path),
        container=f"{TOY_CONTAINER_PREFIX}/{scenario}",
        function_tag=truth_label,
        toy_name=f"toy_{int(toy_index):04d}",
        toy_index=int(toy_index),
    )
    return build_funcform_toy_dataset(base, histogram, toy_spec)


def make_analytic_mean_dataset(
    scenario: str,
    cfg: Any,
    *,
    key_mapping_name: str = "analytic_mean_keys",
    function_label: str | None = None,
) -> Any:
    from hps_gpr.dataset import make_datasets
    from hps_gpr.funcform_toys import FuncFormToySpec, build_funcform_toy_dataset

    spec = load_spec()
    product = spec.get("background_toy_product", {})
    mapping = product.get(key_mapping_name, {})
    key = mapping.get(scenario) if isinstance(mapping, Mapping) else None
    if not isinstance(key, str) or not key:
        raise StudyError(
            "analytic closure command requires "
            f"background_toy_product.{key_mapping_name}.{scenario}"
        )
    path = background_root(spec)
    histogram = _load_histogram(path, key)
    base = make_datasets(cfg)["2021"]
    container, name = key.rsplit("/", 1)
    scenario_record = spec.get("scenarios", {}).get(scenario, {})
    expected_total = float(scenario_record["normalization_target_count"])
    actual_total = float(np.sum(np.asarray(histogram.values(), dtype=float)))
    total_tolerance = max(1e-3, expected_total * 1e-9)
    if not math.isclose(
        actual_total, expected_total, rel_tol=0.0, abs_tol=total_tolerance
    ):
        raise StudyError(
            f"analytic mean normalization mismatch for {scenario}: "
            f"expected {expected_total:.12g}, found {actual_total:.12g}"
        )
    truth_label = str(
        function_label
        or scenario_record.get(
            "function_tag",
            spec.get("background_toy_product", {}).get("truth_model", ""),
        )
    )
    if not truth_label:
        raise StudyError(f"missing functional-truth label for {scenario}")
    toy_spec = FuncFormToySpec(
        source_root=str(path),
        container=container,
        function_tag=truth_label,
        toy_name=name,
        toy_index=-1,
    )
    return build_funcform_toy_dataset(base, histogram, toy_spec)


def covariance_diagnostics(covariance: Any) -> dict[str, Any]:
    matrix = np.asarray(covariance, dtype=float)
    finite = bool(
        matrix.ndim == 2
        and matrix.shape[0] == matrix.shape[1]
        and matrix.size > 0
        and np.isfinite(matrix).all()
    )
    if not finite:
        return {
            "covariance_valid": False,
            "covariance_min_eigenvalue": float("nan"),
            "covariance_min_eigenvalue_relative": float("nan"),
        }
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.diag(symmetric))), 1.0)
    minimum = float(np.min(eigenvalues))
    relative = minimum / scale
    return {
        "covariance_valid": bool(
            np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-8 * scale)
            and relative >= -1e-2
        ),
        "covariance_min_eigenvalue": minimum,
        "covariance_min_eigenvalue_relative": relative,
    }


def training_geometry(
    x_full: Any,
    y_full: Any,
    train_mask: Any,
    edges_full: Any,
    *,
    mass: float,
) -> dict[str, Any]:
    x = np.asarray(x_full, dtype=float).reshape(-1)
    y = np.asarray(y_full, dtype=float).reshape(-1)
    mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    edges = np.asarray(edges_full, dtype=float).reshape(-1)
    if x.shape != y.shape or x.shape != mask.shape:
        raise StudyError(f"training geometry shape mismatch at {mass:.6g} GeV")
    if edges.shape != (x.size + 1,):
        raise StudyError(f"training edge shape mismatch at {mass:.6g} GeV")
    selected_x = x[mask]
    selected_y = y[mask]
    if selected_x.size == 0:
        raise StudyError(f"empty GP training set at {mass:.6g} GeV")
    if not np.all(np.isfinite(selected_y)) or np.any(selected_y <= 0):
        raise StudyError(
            f"pre_log requires strictly positive finite GP training counts at "
            f"{mass:.6g} GeV"
        )
    widths = np.diff(edges)
    return {
        "n_train": int(np.count_nonzero(mask)),
        "n_train_low": int(np.count_nonzero(mask & (x < float(mass)))),
        "n_train_high": int(np.count_nonzero(mask & (x > float(mass)))),
        "train_domain_lo": float(edges[0]),
        "train_domain_hi": float(edges[-1]),
        "bin_width_median": float(np.median(widths)),
        "n_zero_train": int(np.count_nonzero(selected_y <= 0)),
        "min_y_train": float(np.min(selected_y)),
        "max_y_train": float(np.max(selected_y)),
        "training_counts_sha256": array_hash(selected_y, "<f8"),
    }


def kernel_bound_diagnostics(
    *,
    ls_value: float,
    ls_lower: float,
    ls_upper: float,
    const_value: float,
    const_lower: float,
    const_upper: float,
) -> dict[str, Any]:
    def near(value: float, bound: float) -> bool:
        return bool(
            np.isfinite(value)
            and np.isfinite(bound)
            and bound > 0
            and np.isclose(value, bound, rtol=1e-3, atol=1e-12)
        )

    return {
        "ls_at_lower": near(ls_value, ls_lower),
        "ls_at_upper": near(ls_value, ls_upper),
        "const_at_lower": near(const_value, const_lower),
        "const_at_upper": near(const_value, const_upper),
    }


def branch_match(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> bool:
    required = ("gp_lml", "gp_ls", "gp_const", "sigma_A", "n_train")
    if not all(
        np.isfinite(float(first.get(key, np.nan)))
        and np.isfinite(float(second.get(key, np.nan)))
        for key in required
    ):
        return False
    n_train = max(1.0, min(float(first["n_train"]), float(second["n_train"])))
    if (
        abs(float(first["gp_lml"]) - float(second["gp_lml"])) / n_train
        > float(gate["delta_lml_per_train_max"])
    ):
        return False
    for key, limit in (
        ("gp_ls", gate["abs_log_length_ratio_max"]),
        ("gp_const", gate["abs_log_constant_ratio_max"]),
        ("sigma_A", gate["abs_log_sigma_ratio_max"]),
    ):
        left, right = float(first[key]), float(second[key])
        if left <= 0 or right <= 0:
            return False
        if abs(math.log(left / right)) > float(limit):
            return False
    return True


def select_branch(
    records: list[dict[str, Any]],
    gate: Mapping[str, Any],
    *,
    require_replication: bool,
) -> tuple[dict[str, Any] | None, int]:
    usable = [
        row
        for row in records
        if bool(row.get("fit_ok"))
        and bool(row.get("covariance_valid"))
        and np.isfinite(float(row.get("gp_lml", np.nan)))
        and np.isfinite(float(row.get("sigma_A", np.nan)))
        and float(row.get("sigma_A", 0.0)) > 0
    ]
    if not usable:
        return None, 0
    selected = max(usable, key=lambda row: float(row["gp_lml"]))
    replicates = sum(branch_match(selected, row, gate) for row in usable)
    if require_replication and replicates < int(gate["top_branch_min_replicates"]):
        return None, replicates
    return selected, replicates


def reference_attempt(
    ds: Any,
    cfg: Any,
    scenario: str,
    toy_index: int,
    mass: float,
    attempt: int,
) -> tuple[dict[str, Any], Any | None]:
    from hps_gpr.conversion import A_from_epsilon2
    from hps_gpr.injection import (
        _fit_A_for_extraction,
        _prediction_blind_mask,
        _prediction_y_full_bonly,
        _sigmaA_reference,
    )
    from hps_gpr.io import estimate_background_for_dataset
    from hps_gpr.template import build_window_template_from_full

    optimizer_seed = stable_seed(
        "v4p7_restart_v1",
        scenario,
        int(toy_index),
        f"{float(mass):.9f}",
        "reference",
        int(attempt),
    )
    cfg.gp_optimizer_random_state = int(optimizer_seed)
    base = {
        "scenario": scenario,
        "background_toy_index": int(toy_index),
        "mass_GeV": float(mass),
        "inj_nsigma": 0.0,
        "strength": 0.0,
        "role": "reference_bonly",
        "attempt": int(attempt),
        "optimizer_seed": int(optimizer_seed),
        "optimizer_restarts": 12,
        "fit_ok": False,
        "refit_fallback_used": False,
        "error": "",
    }
    try:
        pred = estimate_background_for_dataset(
            ds, float(mass), cfg, restarts=12, optimize=True
        )
        blind_mask = _prediction_blind_mask(pred)
        x_full = np.asarray(pred.x_full, dtype=float)
        y_full = np.asarray(_prediction_y_full_bonly(pred), dtype=float)
        train_half_width = (
            float(cfg.gp_train_exclude_nsigma) * float(pred.sigma_val)
        )
        train_mask = (x_full < float(mass) - train_half_width) | (
            x_full > float(mass) + train_half_width
        )
        geometry = training_geometry(
            x_full,
            y_full,
            train_mask,
            pred.edges_full,
            mass=float(mass),
        )
        if int(geometry["n_train"]) != int(pred.n_train):
            raise StudyError(
                f"reference training-mask cardinality mismatch at {mass:.6g} GeV"
            )
        template_window, _ = build_window_template_from_full(
            pred.edges_full,
            blind_mask,
            float(mass),
            pred.sigma_val,
            config=cfg,
        )
        observed = y_full[blind_mask]
        fit = _fit_A_for_extraction(
            cfg,
            observed,
            pred.mu,
            pred.cov,
            template_window,
            allow_negative=True,
        )
        sigma_a = float(fit["sigma_A"])
        sigma_reference = float(
            _sigmaA_reference(pred, float(mass), source="asimov", config=cfg)
        )
        density = float(pred.integral_density)
        record = {
            **base,
            "fit_ok": bool(fit.get("success", False)),
            "gp_lml": float(pred.lml),
            "gp_ls": float(pred.ls_opt),
            "gp_const": float(pred.const_opt),
            "gp_const_lo": float(pred.const_lo),
            "gp_const_hi": float(pred.const_hi),
            "gp_ls_lo": float(pred.ls_lo),
            "gp_ls_hi": float(pred.ls_hi),
            "gp_ls_init": float(pred.ls_init),
            "gp_const_init": float(pred.const_init),
            "sigma_A": sigma_a,
            "sigmaA_reference": sigma_reference,
            "A_hat": float(fit["A_hat"]),
            "Zhat": float(fit["A_hat"]) / sigma_a,
            "pull": float(fit["A_hat"]) / sigma_a,
            "amplitude_nll": float(fit.get("nll", np.nan)),
            "n_blind": int(pred.n_blind),
            "integral_density": density,
            "A_per_eps2_unit": float(
                A_from_epsilon2(ds, float(mass), 1.0, density)
            ),
            "optimizer_warning_count": int(pred.optimizer_warning_count),
            "optimizer_warnings": str(pred.optimizer_warnings),
            **geometry,
            **kernel_bound_diagnostics(
                ls_value=float(pred.ls_opt),
                ls_lower=float(pred.ls_lo),
                ls_upper=float(pred.ls_hi),
                const_value=float(pred.const_opt),
                const_lower=float(pred.const_lo),
                const_upper=float(pred.const_hi),
            ),
            **covariance_diagnostics(pred.cov),
        }
        return record, pred
    except Exception as exc:
        return {
            **base,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }, None


def refit_attempt(
    ds: Any,
    cfg: Any,
    reference_pred: Any,
    reference_row: Mapping[str, Any],
    scenario: str,
    toy_index: int,
    mass: float,
    z_value: float,
    attempt: int,
) -> dict[str, Any]:
    from hps_gpr.gpr import (
        fit_gpr,
        make_kernel_for_dataset,
        predict_counts_from_log_gpr,
    )
    from hps_gpr.injection import (
        _fit_A_for_extraction,
        _fixed_hist_background_counts,
        _gpr_fit_diagnostics,
        _inject_counts_from_template,
        _prediction_blind_mask,
        _prediction_y_full_bonly,
    )
    from hps_gpr.template import build_window_template_from_full

    optimizer_seed = stable_seed(
        "v4p7_restart_v1",
        scenario,
        int(toy_index),
        f"{float(mass):.9f}",
        f"z{float(z_value):.1f}",
        int(attempt),
    )
    signal_seed = stable_seed(
        "v4p7_signal_v1",
        scenario,
        int(toy_index),
        f"{float(mass):.9f}",
        f"z{float(z_value):.1f}",
    )
    sigma_reference = float(reference_row["sigma_A"])
    injected = float(z_value) * sigma_reference
    base = {
        "scenario": scenario,
        "background_toy_index": int(toy_index),
        "mass_GeV": float(mass),
        "inj_nsigma": float(z_value),
        "role": "injected_refit",
        "attempt": int(attempt),
        "optimizer_seed": int(optimizer_seed),
        "signal_seed": int(signal_seed),
        "optimizer_restarts": 12,
        "fit_ok": False,
        "refit_fallback_used": False,
        "error": "",
        "strength": injected,
        "sigmaA_reference": sigma_reference,
        "reference_attempt_selected": int(reference_row["attempt"]),
        "reference_gp_lml": float(reference_row["gp_lml"]),
        "reference_gp_ls": float(reference_row["gp_ls"]),
        "reference_gp_const": float(reference_row["gp_const"]),
        "A_per_eps2_unit": float(reference_row["A_per_eps2_unit"]),
        "integral_density": float(reference_row["integral_density"]),
    }
    try:
        blind_mask = _prediction_blind_mask(reference_pred)
        x_full = np.asarray(reference_pred.x_full, dtype=float)
        background = _fixed_hist_background_counts(
            _prediction_y_full_bonly(reference_pred),
            dataset_key="2021",
            mass=float(mass),
        )
        template_window, template_full = build_window_template_from_full(
            reference_pred.edges_full,
            blind_mask,
            float(mass),
            reference_pred.sigma_val,
            config=cfg,
        )
        rng = np.random.default_rng(int(signal_seed))
        signal_full, n_signal_full, _ = _inject_counts_from_template(
            template_full, injected, rng, "poisson"
        )
        signal_full = np.asarray(signal_full, dtype=int)
        y_toy = np.asarray(background, dtype=int) + signal_full

        # Keep this tied to the effective card.  A literal 2.25 here would let
        # the refit mask silently drift away from a future audited card value.
        train_half_width = (
            float(cfg.gp_train_exclude_nsigma)
            * float(reference_pred.sigma_val)
        )
        train_mask = (x_full < float(mass) - train_half_width) | (
            x_full > float(mass) + train_half_width
        )
        kernel = make_kernel_for_dataset(ds, cfg, mass=float(mass))
        gpr = fit_gpr(
            x_full[train_mask],
            y_toy[train_mask].astype(float),
            cfg,
            restarts=12,
            kernel=kernel,
            optimize=True,
            random_state=int(optimizer_seed),
        )
        mu, covariance = predict_counts_from_log_gpr(
            gpr, x_full[blind_mask], cfg
        )
        fit = _fit_A_for_extraction(
            cfg,
            y_toy[blind_mask],
            mu,
            covariance,
            template_window,
            allow_negative=True,
        )
        diagnostics = _gpr_fit_diagnostics(gpr)
        sigma_a = float(fit["sigma_A"])
        a_hat = float(fit["A_hat"])
        initial_kernel = getattr(gpr, "kernel", None)
        initial_const = float(
            getattr(
                getattr(initial_kernel, "k1", None),
                "constant_value",
                np.nan,
            )
        )
        initial_ls = float(
            getattr(
                getattr(initial_kernel, "k2", None),
                "length_scale",
                np.nan,
            )
        )
        const_lower = float(reference_row["gp_const_lo"])
        const_upper = float(reference_row["gp_const_hi"])
        geometry = training_geometry(
            x_full,
            y_toy,
            train_mask,
            reference_pred.edges_full,
            mass=float(mass),
        )
        return {
            **base,
            "fit_ok": bool(fit.get("success", False)),
            "gp_lml": float(gpr.log_marginal_likelihood_value_),
            "gp_ls": float(diagnostics["ls_opt"]),
            "gp_const": float(diagnostics["const_opt"]),
            "gp_const_lo": const_lower,
            "gp_const_hi": const_upper,
            "gp_ls_lo": float(reference_row["gp_ls_lo"]),
            "gp_ls_hi": float(reference_row["gp_ls_hi"]),
            "gp_ls_init": initial_ls,
            "gp_const_init": initial_const,
            "sigma_A": sigma_a,
            "A_hat": a_hat,
            "Zhat": a_hat / sigma_a,
            "pull": (a_hat - injected) / sigma_a,
            "amplitude_nll": float(fit.get("nll", np.nan)),
            "n_blind": int(np.count_nonzero(blind_mask)),
            "Nsig_full": int(n_signal_full),
            "Nsig_win": int(np.sum(signal_full[blind_mask])),
            "Nsig_train": int(np.sum(signal_full[train_mask])),
            "signal_counts_sha256": array_hash(signal_full, "<i8"),
            "optimizer_warning_count": len(
                getattr(gpr, "_hps_optimizer_warnings", ())
            ),
            "optimizer_warnings": " | ".join(
                getattr(gpr, "_hps_optimizer_warnings", ())
            ),
            **geometry,
            **kernel_bound_diagnostics(
                ls_value=float(diagnostics["ls_opt"]),
                ls_lower=float(reference_row["gp_ls_lo"]),
                ls_upper=float(reference_row["gp_ls_hi"]),
                const_value=float(diagnostics["const_opt"]),
                const_lower=const_lower,
                const_upper=const_upper,
            ),
            **covariance_diagnostics(covariance),
        }
    except Exception as exc:
        return {
            **base,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }


def refit_triggers(
    row: Mapping[str, Any], gate: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if not bool(row.get("fit_ok")) or not bool(row.get("covariance_valid")):
        reasons.append("invalid_or_nonfinite")
        return reasons
    ls_value, const_value = float(row["gp_ls"]), float(row["gp_const"])
    ls_initial, const_initial = float(row["gp_ls_init"]), float(
        row["gp_const_init"]
    )
    exact_tolerance = float(gate["exact_start_abs_log_theta_max"])
    if all(
        value > 0
        for value in (ls_value, const_value, ls_initial, const_initial)
    ):
        if max(
            abs(math.log(ls_value / ls_initial)),
            abs(math.log(const_value / const_initial)),
        ) < exact_tolerance:
            reasons.append("exact_start_signature")
    lower, upper = float(row["gp_ls_lo"]), float(row["gp_ls_hi"])
    bound_window = float(gate["bound_ratio_window"])
    if ls_value > 0 and lower > 0 and ls_value / lower <= 1.0 + bound_window:
        reasons.append("near_lower_length_bound")
    if ls_value > 0 and upper > 0 and ls_value / upper >= 1.0 - bound_window:
        reasons.append("near_upper_length_bound")
    ratio = float(row["sigma_A"]) / float(row["sigmaA_reference"])
    ratio_low, ratio_high = map(float, gate["sigma_over_reference_trigger"])
    if not np.isfinite(ratio) or ratio < ratio_low or ratio > ratio_high:
        reasons.append("sigma_reference_ratio")

    # v4.7.1: apply the amended reference-relative thresholds frozen after the
    # first full-ledger numerical audit and before the uniform rerun.  This is
    # a pull-blind repeat trigger only; the final branch is still selected by
    # LML and reproducibility.  No fitted amplitude, pull, recovery, or
    # epsilon-squared coordinate enters this decision.
    reference_values = {
        "lml": float(row.get("reference_gp_lml", np.nan)),
        "ls": float(row.get("reference_gp_ls", np.nan)),
        "const": float(row.get("reference_gp_const", np.nan)),
    }
    comparable = (
        np.isfinite(float(row["gp_lml"]))
        and np.isfinite(reference_values["lml"])
        and int(row["n_train"]) > 0
        and all(
            np.isfinite(value) and value > 0
            for value in (
                float(row["gp_ls"]),
                reference_values["ls"],
                float(row["gp_const"]),
                reference_values["const"],
            )
        )
    )
    if not comparable:
        reasons.append("reference_relative_nonfinite")
        return reasons
    if (
        abs(float(row["gp_lml"]) - reference_values["lml"])
        / float(row["n_train"])
        > float(gate["reference_relative_lml_per_train_trigger"])
    ):
        reasons.append("reference_relative_lml")
    if abs(math.log(float(row["gp_ls"]) / reference_values["ls"])) > float(
        gate["reference_relative_abs_log_length_trigger"]
    ):
        reasons.append("reference_relative_length")
    if abs(
        math.log(float(row["gp_const"]) / reference_values["const"])
    ) > float(gate["reference_relative_abs_log_constant_trigger"]):
        reasons.append("reference_relative_constant")
    return reasons


def accepted_row(
    selected: Mapping[str, Any],
    reference_row: Mapping[str, Any],
    scenario: str,
    toy_index: int,
    mass: float,
    z_value: float,
    attempts: int,
    replicates: int,
    gate_status: str,
    trigger_reasons: Iterable[str],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(selected)
    scenario_record = spec["scenarios"][scenario]
    function_tag = str(
        scenario_record.get(
            "function_tag",
            spec.get("background_toy_product", {}).get("truth_model", ""),
        )
    )
    truth_model = str(
        spec.get("background_toy_product", {}).get("truth_model", "")
    )
    if not function_tag or not truth_model:
        raise StudyError("functional-truth metadata is missing from study_spec")
    a_up_wald = wald_cls_upper_limit(
        float(selected["A_hat"]), float(selected["sigma_A"]), alpha=0.10
    )
    a_per_eps2 = float(reference_row["A_per_eps2_unit"])
    injected = float(selected.get("strength", 0.0))
    sigma_reference = float(reference_row["sigma_A"])
    row.update(
        {
            "study_id": spec["study_id"],
            "scenario": scenario,
            "scenario_label": scenario_record["label"],
            "source_family": scenario_record["source_family"],
            "truth_model": truth_model,
            "truth_function_tag": function_tag,
            "background_toy_index": int(toy_index),
            "mass_GeV": float(mass),
            "mass_MeV": 1000.0 * float(mass),
            "inj_nsigma": float(z_value),
            "n_attempts": int(attempts),
            "top_branch_replicates": int(replicates),
            "optimizer_gate_status": gate_status,
            "optimizer_trigger_reasons": ";".join(trigger_reasons),
            "accepted": True,
            "sigmaA_ref": sigma_reference,
            "sigmaA_ref_mode": "matched_refit_bonly_multistart_v1",
            "reference_top_branch_replicates": int(
                reference_row["top_branch_replicates"]
            ),
            "eps2_hat_signed": float(selected["A_hat"])
            / float(reference_row["A_per_eps2_unit"]),
            "eps2_injected": float(selected.get("strength", 0.0))
            / float(reference_row["A_per_eps2_unit"]),
            "eps2_sigma": float(selected["sigma_A"])
            / a_per_eps2,
            "A_up_wald90": a_up_wald,
            "eps2_up_wald90": a_up_wald / a_per_eps2,
            "A_up_wald90_over_sigmaA_ref": a_up_wald / sigma_reference,
            "A_up_wald90_minus_injected_over_sigmaA_ref":
                (a_up_wald - injected) / sigma_reference,
            "p0_wald_local": float(norm.sf(max(float(selected["Zhat"]), 0.0))),
            "upper_limit_diagnostic":
                "90pct_CLs_Wald_tilde_q_mu_from_profiled_Ahat_sigmaA",
            "nominal_Z_residual": float(selected["Zhat"]) - float(z_value),
            "pull_identity_residual": float(selected["pull"])
            - (
                (float(selected["A_hat"]) - float(selected.get("strength", 0.0)))
                / float(selected["sigma_A"])
            ),
            "refit_ls_over_hi": float(selected["gp_ls"])
            / float(selected["gp_ls_hi"]),
            "refit_ls_over_lo": float(selected["gp_ls"])
            / float(selected["gp_ls_lo"]),
            "refit_upper_boundary": float(selected["gp_ls"])
            / float(selected["gp_ls_hi"])
            >= 0.999,
            "refit_lower_boundary": float(selected["gp_ls"])
            / float(selected["gp_ls_lo"])
            <= 1.001,
            "refit_constant_lower_boundary": bool(
                selected.get("const_at_lower", False)
            ),
            "refit_constant_upper_boundary": bool(
                selected.get("const_at_upper", False)
            ),
            "analysis_partition":
                f"v4p9p5_native10_support_edge_scan_{SUPPORT_MODE}",
            "gp_support_mode": SUPPORT_MODE,
            "gp_support_low_GeV": SUPPORT_RANGE[0],
            "gp_support_high_GeV": SUPPORT_RANGE[1],
            "declared_result_commit": spec.get("declared_result_state", {}).get(
                "result_commit", ""
            ),
            "declared_integration_commit": spec.get(
                "declared_result_state", {}
            ).get("integration_commit", ""),
        }
    )
    return row


def task_directory(scenario: str, toy_index: int) -> Path:
    return RUNS / scenario / f"toy_{int(toy_index):04d}"


def successful_task(scenario: str, toy_index: int) -> Path | None:
    directory = task_directory(scenario, toy_index)
    marker_path = directory / "_SUCCESS.json"
    if not marker_path.is_file():
        return None
    try:
        payload = load_json(marker_path)
    except Exception:
        return None
    if payload.get("status") != "pass":
        return None
    if payload.get("study_spec_sha256") != sha256_file(SPEC_PATH):
        return None
    spec = load_spec()
    product = spec.get("background_toy_product", {})
    if payload.get("background_toy_root_sha256") != product.get("root_sha256"):
        return None
    if payload.get("background_toy_manifest_sha256") != product.get(
        "manifest_sha256"
    ):
        return None
    declared = payload.get("ledger_sha256")
    if not isinstance(declared, Mapping) or set(declared) != set(LEDGER_FILES):
        return None
    for name in LEDGER_FILES:
        path = directory / name
        if not path.is_file() or sha256_file(path) != str(declared[name]):
            return None
    return directory / "accepted_rows.csv"


def _empty_exclusions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scenario",
            "background_toy_index",
            "mass_GeV",
            "inj_nsigma",
            "exclusion_scope",
            "reason",
            "n_attempts",
            "trigger_reasons",
        ]
    )


def run_task(
    scenario: str, toy_index: int, *, force: bool = False
) -> dict[str, Any]:
    configure_process()
    spec = load_spec()
    assert_spec_contract(spec)
    gate = spec["optimizer_gate"]
    if scenario not in SCENARIOS or not 0 <= int(toy_index) < N_TOYS:
        raise StudyError("invalid headline scenario or toy index")
    preflight(validate_inventory=False)
    existing = successful_task(scenario, int(toy_index))
    if existing is not None and not force:
        return {
            "status": "already_complete",
            "scenario": scenario,
            "toy_index": int(toy_index),
        }

    final_directory = task_directory(scenario, int(toy_index))
    if final_directory.exists():
        if not force:
            raise StudyError(
                f"incomplete task exists; inspect or use --force: {final_directory}"
            )
        archived = final_directory.with_name(
            final_directory.name
            + ".superseded_"
            + datetime.now().strftime("%Y%m%dT%H%M%S")
        )
        if archived.exists():
            raise StudyError(f"superseded task destination already exists: {archived}")
        os.replace(final_directory, archived)

    final_directory.parent.mkdir(parents=True, exist_ok=True)
    work_directory = Path(
        tempfile.mkdtemp(prefix=f".{final_directory.name}.", dir=final_directory.parent)
    )
    cfg = build_config()
    assert_config(cfg)
    dataset = make_toy_dataset(scenario, int(toy_index), cfg)
    attempt_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    try:
        for mass in MASS_GRID:
            reference_records: list[dict[str, Any]] = []
            reference_predictions: dict[int, Any] = {}
            for attempt in range(int(gate["reference_initial_attempts"])):
                record, prediction = reference_attempt(
                    dataset,
                    cfg,
                    scenario,
                    int(toy_index),
                    float(mass),
                    attempt,
                )
                reference_records.append(record)
                if prediction is not None:
                    reference_predictions[int(attempt)] = prediction
            selected_reference, reference_replicates = select_branch(
                reference_records, gate, require_replication=True
            )
            if selected_reference is None:
                for attempt in range(
                    int(gate["reference_initial_attempts"]),
                    int(gate["maximum_attempts"]),
                ):
                    record, prediction = reference_attempt(
                        dataset,
                        cfg,
                        scenario,
                        int(toy_index),
                        float(mass),
                        attempt,
                    )
                    reference_records.append(record)
                    if prediction is not None:
                        reference_predictions[int(attempt)] = prediction
                selected_reference, reference_replicates = select_branch(
                    reference_records, gate, require_replication=True
                )
            attempt_rows.extend(reference_records)

            if selected_reference is None:
                for z_value in STRENGTH_GRID:
                    placeholder = (
                        dict(reference_records[0]) if reference_records else {}
                    )
                    placeholder.update(
                        {
                            "scenario": scenario,
                            "background_toy_index": int(toy_index),
                            "mass_GeV": float(mass),
                            "inj_nsigma": float(z_value),
                            "strength": float("nan"),
                            "A_hat": float("nan"),
                            "sigma_A": float("nan"),
                            "Zhat": float("nan"),
                            "pull": float("nan"),
                            "role": "raw_reference_invalid_placeholder",
                            "accepted": False,
                            "optimizer_gate_status": "exclude_irreproducible_reference",
                        }
                    )
                    raw_rows.append(placeholder)
                    exclusions.append(
                        {
                            "scenario": scenario,
                            "background_toy_index": int(toy_index),
                            "mass_GeV": float(mass),
                            "inj_nsigma": float(z_value),
                            "exclusion_scope": "scenario_toy_mass_all_strengths",
                            "reason": "irreproducible_background_reference_top_branch",
                            "n_attempts": len(reference_records),
                            "trigger_reasons": "",
                        }
                    )
                continue

            raw_rows.append(dict(reference_records[0]))
            selected_reference = dict(selected_reference)
            selected_reference["top_branch_replicates"] = int(reference_replicates)
            reference_prediction = reference_predictions[
                int(selected_reference["attempt"])
            ]
            reference_status = (
                "pass_replicated_initial3"
                if len(reference_records) == 3
                else "pass_replicated_after5"
            )
            accepted_rows.append(
                accepted_row(
                    selected_reference,
                    selected_reference,
                    scenario,
                    int(toy_index),
                    float(mass),
                    0.0,
                    len(reference_records),
                    reference_replicates,
                    reference_status,
                    (),
                    spec,
                )
            )

            for z_value in tuple(value for value in STRENGTH_GRID if value != 0.0):
                records = [
                    refit_attempt(
                        dataset,
                        cfg,
                        reference_prediction,
                        selected_reference,
                        scenario,
                        int(toy_index),
                        float(mass),
                        z_value,
                        0,
                    )
                ]
                trigger_reasons = refit_triggers(records[0], gate)
                if trigger_reasons:
                    records.extend(
                        refit_attempt(
                            dataset,
                            cfg,
                            reference_prediction,
                            selected_reference,
                            scenario,
                            int(toy_index),
                            float(mass),
                            z_value,
                            attempt,
                        )
                        for attempt in (1, 2)
                    )
                    selected, replicates = select_branch(
                        records, gate, require_replication=True
                    )
                    if selected is None:
                        records.extend(
                            refit_attempt(
                                dataset,
                                cfg,
                                reference_prediction,
                                selected_reference,
                                scenario,
                                int(toy_index),
                                float(mass),
                                z_value,
                                attempt,
                            )
                            for attempt in (3, 4)
                        )
                        selected, replicates = select_branch(
                            records, gate, require_replication=True
                        )
                    gate_status = (
                        "pass_trigger_replicated_after3"
                        if len(records) == 3
                        else "pass_trigger_replicated_after5"
                    )
                else:
                    selected, replicates = select_branch(
                        records, gate, require_replication=False
                    )
                    gate_status = "pass_single_untriggered"
                attempt_rows.extend(records)
                raw_rows.append(dict(records[0]))
                if selected is None:
                    exclusions.append(
                        {
                            "scenario": scenario,
                            "background_toy_index": int(toy_index),
                            "mass_GeV": float(mass),
                            "inj_nsigma": float(z_value),
                            "exclusion_scope": "single_injected_fit_row",
                            "reason": "irreproducible_injected_refit_top_branch",
                            "n_attempts": len(records),
                            "trigger_reasons": ";".join(trigger_reasons),
                        }
                    )
                    continue
                accepted_rows.append(
                    accepted_row(
                        selected,
                        selected_reference,
                        scenario,
                        int(toy_index),
                        float(mass),
                        z_value,
                        len(records),
                        replicates,
                        gate_status,
                        trigger_reasons,
                        spec,
                    )
                )

        attempts_frame = pd.DataFrame(attempt_rows)
        accepted_frame = pd.DataFrame(accepted_rows)
        raw_frame = pd.DataFrame(raw_rows)
        exclusions_frame = (
            pd.DataFrame(exclusions) if exclusions else _empty_exclusions()
        )
        expected_rows = len(MASS_GRID) * len(STRENGTH_GRID)
        if len(raw_frame) != expected_rows:
            raise StudyError(
                f"raw-primary cardinality is {len(raw_frame)}, expected {expected_rows}"
            )
        frames = {
            "optimizer_attempts.csv": attempts_frame,
            "accepted_rows.csv": accepted_frame,
            "raw_primary_rows.csv": raw_frame,
            "exclusions.csv": exclusions_frame,
        }
        for name, frame in frames.items():
            frame.to_csv(work_directory / name, index=False)
        ledger_hashes = {
            name: sha256_file(work_directory / name) for name in LEDGER_FILES
        }
        marker = {
            "status": "pass",
            "completed_utc": utc_now(),
            "scenario": scenario,
            "toy_index": int(toy_index),
            "attempt_rows": len(attempts_frame),
            "accepted_rows": len(accepted_frame),
            "raw_primary_rows": len(raw_frame),
            "excluded_rows": len(exclusions_frame),
            "study_spec_sha256": sha256_file(SPEC_PATH),
            "background_toy_root_sha256": spec["background_toy_product"][
                "root_sha256"
            ],
            "background_toy_manifest_sha256": spec["background_toy_product"][
                "manifest_sha256"
            ],
            "ledger_sha256": ledger_hashes,
        }
        atomic_json(work_directory / "_SUCCESS.json", marker)
        os.replace(work_directory, final_directory)
        return marker
    except Exception:
        shutil.rmtree(work_directory, ignore_errors=True)
        raise


def run_task_subprocess(
    scenario: str, toy_index: int, force: bool
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-task",
        scenario,
        str(int(toy_index)),
    ]
    if force:
        command.append("--force")
    environment = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[key] = "1"
    result = subprocess.run(
        command,
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise StudyError(
            f"task {scenario} toy {toy_index} failed:\n{result.stdout}"
        )
    return {
        "scenario": scenario,
        "toy_index": int(toy_index),
        "output": result.stdout,
    }


def run_many(
    toy_start: int,
    toy_stop: int,
    workers: int,
    *,
    force: bool = False,
) -> dict[str, Any]:
    preflight(validate_inventory=True)
    if not 0 <= int(toy_start) < int(toy_stop) <= N_TOYS:
        raise StudyError("toy interval must satisfy 0 <= start < stop <= 100")
    if not 1 <= int(workers) <= 2:
        raise StudyError("CPU-conscious production permits one or two workers")
    tasks = [
        (scenario, toy_index)
        for scenario in SCENARIOS
        for toy_index in range(int(toy_start), int(toy_stop))
    ]
    completed = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {
            pool.submit(run_task_subprocess, scenario, toy_index, force): (
                scenario,
                toy_index,
            )
            for scenario, toy_index in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                f"PASS {result['scenario']} toy {result['toy_index']:04d}",
                flush=True,
            )
    return {
        "status": "pass",
        "tasks": len(tasks),
        "completed": len(completed),
        "toy_start": int(toy_start),
        "toy_stop": int(toy_stop),
        "workers": int(workers),
    }


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _moments(group: pd.DataFrame, prefix: str) -> dict[str, Any]:
    from scipy.stats import chi2, median_abs_deviation, t, trim_mean

    pull_series = (
        group["pull"] if "pull" in group.columns else pd.Series(dtype=float)
    )
    values = pd.to_numeric(pull_series, errors="coerce").dropna().to_numpy(
        dtype=float
    )
    count = len(values)
    mean = float(np.mean(values)) if count else float("nan")
    width = float(np.std(values, ddof=1)) if count > 1 else float("nan")
    t_critical = float(t.ppf(0.95, count - 1)) if count > 1 else float("nan")
    chi_low = float(chi2.ppf(0.05, count - 1)) if count > 1 else float("nan")
    chi_high = float(chi2.ppf(0.95, count - 1)) if count > 1 else float("nan")
    leave_one_out = (
        [abs(float(np.mean(np.delete(values, index))) - mean) for index in range(count)]
        if count > 1
        else []
    )
    return {
        f"{prefix}_n": count,
        f"{prefix}_pull_mean": mean,
        f"{prefix}_pull_width": width,
        f"{prefix}_pull_median": float(np.median(values))
        if count
        else float("nan"),
        f"{prefix}_pull_mad_scaled": float(
            median_abs_deviation(values, scale="normal")
        )
        if count
        else float("nan"),
        f"{prefix}_pull_trimmed_mean_10pct": float(trim_mean(values, 0.1))
        if count
        else float("nan"),
        f"{prefix}_pull_mean_ci90_low": mean
        - t_critical * width / math.sqrt(count)
        if count > 1
        else float("nan"),
        f"{prefix}_pull_mean_ci90_high": mean
        + t_critical * width / math.sqrt(count)
        if count > 1
        else float("nan"),
        f"{prefix}_pull_width_ci90_low": math.sqrt(
            (count - 1) * width * width / chi_high
        )
        if count > 1
        else float("nan"),
        f"{prefix}_pull_width_ci90_high": math.sqrt(
            (count - 1) * width * width / chi_low
        )
        if count > 1
        else float("nan"),
        f"{prefix}_max_leave_one_out_mean_change": max(leave_one_out)
        if leave_one_out
        else float("nan"),
    }


def collect() -> dict[str, Any]:
    from scipy.stats import ttest_1samp

    spec = load_spec()
    assert_spec_contract(spec)
    accepted_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    attempt_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []
    task_audit: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for toy_index in range(N_TOYS):
            if successful_task(scenario, toy_index) is None:
                raise StudyError(
                    f"missing or hash-invalid task {scenario} toy {toy_index:04d}"
                )
            directory = task_directory(scenario, toy_index)
            accepted = _read_optional_csv(directory / "accepted_rows.csv")
            raw = _read_optional_csv(directory / "raw_primary_rows.csv")
            attempts = _read_optional_csv(directory / "optimizer_attempts.csv")
            exclusions = _read_optional_csv(directory / "exclusions.csv")
            if raw is None or attempts is None:
                raise StudyError(
                    f"required task ledger is empty: {scenario} toy {toy_index:04d}"
                )
            if accepted is not None:
                accepted_frames.append(accepted)
            raw_frames.append(raw)
            attempt_frames.append(attempts)
            if exclusions is not None:
                exclusion_frames.append(exclusions)
            marker = load_json(directory / "_SUCCESS.json")
            task_audit.append(
                {
                    "scenario": scenario,
                    "toy_index": toy_index,
                    "status": marker["status"],
                    "accepted_rows": marker["accepted_rows"],
                    "raw_primary_rows": marker["raw_primary_rows"],
                    "excluded_rows": marker["excluded_rows"],
                    "success_marker_sha256": sha256_file(
                        directory / "_SUCCESS.json"
                    ),
                }
            )

    accepted = (
        pd.concat(accepted_frames, ignore_index=True, sort=False)
        if accepted_frames
        else pd.DataFrame()
    )
    raw = pd.concat(raw_frames, ignore_index=True, sort=False)
    attempts = pd.concat(attempt_frames, ignore_index=True, sort=False)
    exclusions = (
        pd.concat(exclusion_frames, ignore_index=True, sort=False)
        if exclusion_frames
        else _empty_exclusions()
    )
    key_columns = [
        "scenario",
        "background_toy_index",
        "mass_GeV",
        "inj_nsigma",
    ]
    raw = raw.sort_values(key_columns).reset_index(drop=True)
    if not accepted.empty:
        accepted = accepted.sort_values(key_columns).reset_index(drop=True)
    expected_raw = len(SCENARIOS) * N_TOYS * len(MASS_GRID) * len(STRENGTH_GRID)
    if len(raw) != expected_raw or raw.duplicated(key_columns).any():
        raise StudyError(
            f"raw ledger must contain {expected_raw} unique states"
        )
    if not accepted.empty and accepted.duplicated(key_columns).any():
        raise StudyError("accepted ledger contains duplicate states")

    summaries: list[dict[str, Any]] = []
    minimum_required = int(
        spec["optimizer_gate"]["minimum_accepted_per_cell_for_closure_claim"]
    )
    for scenario in SCENARIOS:
        for mass in MASS_GRID:
            for z_value in STRENGTH_GRID:
                raw_group = raw[
                    (raw.scenario == scenario)
                    & np.isclose(raw.mass_GeV, mass)
                    & np.isclose(raw.inj_nsigma, z_value)
                ]
                accepted_group = (
                    accepted[
                        (accepted.scenario == scenario)
                        & np.isclose(accepted.mass_GeV, mass)
                        & np.isclose(accepted.inj_nsigma, z_value)
                    ]
                    if not accepted.empty
                    else pd.DataFrame()
                )
                record = {
                    "scenario": scenario,
                    "mass_GeV": float(mass),
                    "mass_MeV": 1000.0 * float(mass),
                    "inj_nsigma": float(z_value),
                    "n_generated": N_TOYS,
                    **_moments(raw_group, "raw"),
                    **_moments(accepted_group, "accepted"),
                    "n_excluded": N_TOYS - len(accepted_group),
                    "sample_size_eligible": len(accepted_group)
                    >= minimum_required,
                    "accepted_nominal_Z_residual_mean": float(
                        pd.to_numeric(
                            accepted_group["nominal_Z_residual"]
                        ).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_upper_boundary_fraction": float(
                        pd.to_numeric(
                            accepted_group["refit_upper_boundary"]
                        ).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_lower_boundary_fraction": float(
                        pd.to_numeric(
                            accepted_group["refit_lower_boundary"]
                        ).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_constant_lower_boundary_fraction": float(
                        pd.to_numeric(
                            accepted_group["refit_constant_lower_boundary"]
                        ).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_constant_upper_boundary_fraction": float(
                        pd.to_numeric(
                            accepted_group["refit_constant_upper_boundary"]
                        ).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_pull_identity_max_abs_residual": float(
                        pd.to_numeric(
                            accepted_group["pull_identity_residual"]
                        ).abs().max()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_ls_at_lower_fraction": float(
                        pd.to_numeric(accepted_group["ls_at_lower"]).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_ls_at_upper_fraction": float(
                        pd.to_numeric(accepted_group["ls_at_upper"]).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_const_at_lower_fraction": float(
                        pd.to_numeric(accepted_group["const_at_lower"]).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_const_at_upper_fraction": float(
                        pd.to_numeric(accepted_group["const_at_upper"]).mean()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_min_y_train": float(
                        pd.to_numeric(accepted_group["min_y_train"]).min()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_max_y_train": float(
                        pd.to_numeric(accepted_group["max_y_train"]).max()
                    )
                    if len(accepted_group)
                    else float("nan"),
                    "accepted_max_n_zero_train": int(
                        pd.to_numeric(accepted_group["n_zero_train"]).max()
                    )
                    if len(accepted_group)
                    else -1,
                }
                if float(z_value) > 0 and len(accepted_group):
                    recovery = (
                        pd.to_numeric(accepted_group["A_hat"])
                        / pd.to_numeric(accepted_group["strength"])
                    )
                    record.update(
                        {
                            "accepted_median_recovery": float(
                                np.median(recovery)
                            ),
                            "accepted_recovery_q16": float(
                                np.quantile(recovery, 0.16)
                            ),
                            "accepted_recovery_q84": float(
                                np.quantile(recovery, 0.84)
                            ),
                        }
                    )
                else:
                    record.update(
                        {
                            "accepted_median_recovery": float("nan"),
                            "accepted_recovery_q16": float("nan"),
                            "accepted_recovery_q84": float("nan"),
                        }
                    )
                summaries.append(record)

    summary = pd.DataFrame(summaries).sort_values(
        ["scenario", "mass_GeV", "inj_nsigma"]
    )
    zero_records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for mass in MASS_GRID:
            group = (
                accepted[
                    (accepted.scenario == scenario)
                    & np.isclose(accepted.mass_GeV, mass)
                    & np.isclose(accepted.inj_nsigma, 0.0)
                ]
                if not accepted.empty
                else pd.DataFrame()
            )
            values = (
                pd.to_numeric(group["pull"], errors="coerce")
                .dropna()
                .to_numpy(float)
                if len(group)
                else np.array([], dtype=float)
            )
            p_value = (
                float(ttest_1samp(values, 0.0).pvalue)
                if len(values) > 1
                else float("nan")
            )
            zero_records.append(
                {
                    "scenario": scenario,
                    "mass_GeV": float(mass),
                    "mass_MeV": 1000.0 * float(mass),
                    "n": len(values),
                    "mean_pull": float(np.mean(values))
                    if len(values)
                    else float("nan"),
                    "width": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else float("nan"),
                    "exploratory_ttest_p": p_value,
                }
            )
    zero = pd.DataFrame(zero_records)
    finite_positions = np.where(
        np.isfinite(zero["exploratory_ttest_p"].to_numpy(float))
    )[0]
    adjusted = np.full(len(zero), np.nan, dtype=float)
    if len(finite_positions):
        p_values = zero.loc[
            finite_positions, "exploratory_ttest_p"
        ].to_numpy(float)
        order = np.argsort(p_values)
        running = 0.0
        for rank, ordered_position in enumerate(order):
            candidate = (len(p_values) - rank) * float(
                p_values[ordered_position]
            )
            running = max(running, candidate)
            adjusted[finite_positions[ordered_position]] = min(1.0, running)
    zero["exploratory_holm_p"] = adjusted
    zero["exploratory_material_bias_flag"] = (
        (zero.exploratory_holm_p < 0.05) & (zero.mean_pull.abs() >= 0.2)
    )

    DERIVED.mkdir(parents=True, exist_ok=True)
    products = {
        "accepted_extraction_rows.csv": accepted,
        "raw_primary_extraction_rows.csv": raw,
        "optimizer_attempt_ledger.csv": attempts,
        "exclusion_ledger.csv": exclusions,
        "closure_summary.csv": summary,
        "zero_signal_bias_tests.csv": zero,
        "task_product_audit.csv": pd.DataFrame(task_audit),
    }
    for name, frame in products.items():
        atomic_csv(DERIVED / name, frame)
    product_hashes = {
        name: sha256_file(DERIVED / name) for name in products
    }
    result = {
        "status": "pass",
        "collected_utc": utc_now(),
        "study_spec_sha256": sha256_file(SPEC_PATH),
        "raw_rows": len(raw),
        "accepted_rows": len(accepted),
        "excluded_rows": len(exclusions),
        "optimizer_attempt_rows": len(attempts),
        "summary_cells": len(summary),
        "minimum_accepted_per_cell": int(summary.accepted_n.min()),
        "all_cells_sample_size_eligible": bool(
            summary.sample_size_eligible.all()
        ),
        "scientific_diagnostics": {
            "bias_endpoint": "cellwise accepted mean pull with two-sided 90% Student-t interval",
            "width_endpoint": "cellwise accepted sample pull width with two-sided 90% chi-square interval",
            "sample_size_gate_is_not_closure": True,
            "maximum_abs_pull_identity_residual": float(
                summary.accepted_pull_identity_max_abs_residual.max()
            ),
        },
        "interpretation": (
            "One-hundred-toy conditional near-threshold background-only "
            "validation; not coverage, expected limits, exclusion, or a "
            "scan-wise significance calibration."
        ),
        "gp_support_mode": SUPPORT_MODE,
        "gp_support_range_gev": list(SUPPORT_RANGE),
        "derived_sha256": product_hashes,
    }
    atomic_json(DERIVED / "collection_summary.json", result)
    return result


def analytic_mean_closure(*, baseline: bool = False) -> dict[str, Any]:
    spec = load_spec()
    assert_spec_contract(spec)
    preflight(validate_inventory=False)
    cfg = build_config()
    assert_config(cfg)
    gate = spec["optimizer_gate"]
    lane = "baseline_fsig" if baseline else "fsig_anchor"
    mapping_name = (
        "baseline_analytic_mean_keys" if baseline else "analytic_mean_keys"
    )
    selected_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        dataset = make_analytic_mean_dataset(
            scenario,
            cfg,
            key_mapping_name=mapping_name,
            function_label="fSigPowExpQ_identified_beta_anchor" if baseline else None,
        )
        for mass in MASS_GRID:
            records: list[dict[str, Any]] = []
            for attempt in range(int(gate["reference_initial_attempts"])):
                record, _ = reference_attempt(
                    dataset, cfg, scenario, -1, float(mass), attempt
                )
                records.append(record)
            selected, replicates = select_branch(
                records, gate, require_replication=True
            )
            if selected is None:
                for attempt in range(
                    int(gate["reference_initial_attempts"]),
                    int(gate["maximum_attempts"]),
                ):
                    record, _ = reference_attempt(
                        dataset, cfg, scenario, -1, float(mass), attempt
                    )
                    records.append(record)
                selected, replicates = select_branch(
                    records, gate, require_replication=True
                )
            attempt_rows.extend(records)
            if selected is None:
                raise StudyError(
                    f"analytic-mean optimizer branch failed for {scenario} at {mass}"
                )
            row = dict(selected)
            row.update(
                {
                    "study_id": spec["study_id"],
                    "scenario": scenario,
                    "mass_GeV": float(mass),
                    "mass_MeV": 1000.0 * float(mass),
                    "top_branch_replicates": int(replicates),
                    "n_attempts": len(records),
                    "analytic_mean": True,
                    "analytic_lane": lane,
                    "truth_model": "baseline_fSigPowExpQ_identified_beta"
                    if baseline
                    else str(
                        spec.get("background_toy_product", {}).get(
                            "truth_model", ""
                        )
                    ),
                    "truth_function_tag": "fSigPowExpQ_identified_beta_anchor"
                    if baseline
                    else str(
                        spec.get("scenarios", {})
                        .get(scenario, {})
                        .get(
                            "function_tag",
                            spec.get("background_toy_product", {}).get(
                                "truth_model", ""
                            ),
                        )
                    ),
                }
            )
            selected_rows.append(row)

    selected_frame = pd.DataFrame(selected_rows).sort_values(
        ["scenario", "mass_GeV"]
    )
    attempts_frame = pd.DataFrame(attempt_rows).sort_values(
        ["scenario", "mass_GeV", "attempt"]
    )
    DERIVED.mkdir(parents=True, exist_ok=True)
    prefix = "baseline_analytic_mean" if baseline else "analytic_mean"
    selected_path = DERIVED / f"{prefix}_zero_signal_closure.csv"
    attempts_path = DERIVED / f"{prefix}_optimizer_attempts.csv"
    atomic_csv(selected_path, selected_frame)
    atomic_csv(attempts_path, attempts_frame)
    result = {
        "status": "pass",
        "completed_utc": utc_now(),
        "rows": len(selected_frame),
        "attempt_rows": len(attempts_frame),
        "selected_sha256": sha256_file(selected_path),
        "attempts_sha256": sha256_file(attempts_path),
    }
    atomic_json(DERIVED / f"{prefix}_closure_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    task_parser = subparsers.add_parser("run-task")
    task_parser.add_argument("scenario", choices=SCENARIOS)
    task_parser.add_argument("toy_index", type=int)
    task_parser.add_argument("--force", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--toy-start", type=int, required=True)
    run_parser.add_argument("--toy-stop", type=int, required=True)
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("collect")
    subparsers.add_parser("analytic-mean")
    subparsers.add_parser("analytic-baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        result = preflight(validate_inventory=True)
    elif args.command == "run-task":
        result = run_task(
            args.scenario, int(args.toy_index), force=bool(args.force)
        )
    elif args.command == "run":
        result = run_many(
            int(args.toy_start),
            int(args.toy_stop),
            int(args.workers),
            force=bool(args.force),
        )
    elif args.command == "collect":
        result = collect()
    elif args.command == "analytic-mean":
        result = analytic_mean_closure(baseline=False)
    elif args.command == "analytic-baseline":
        result = analytic_mean_closure(baseline=True)
    else:
        raise StudyError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
