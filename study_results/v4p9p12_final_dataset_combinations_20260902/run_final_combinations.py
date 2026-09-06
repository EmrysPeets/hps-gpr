#!/usr/bin/env python3
"""Build all seven observed final-dataset limits from reviewed fixed GP states.

The output contains the three standalone scans, all three pairwise scans, and
the common three-dataset overlap for 2015 full, 2016 full, and 2021 10%.
No pseudoexperiments or expected-limit bands are evaluated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-gpr-v4p9p12-mpl")

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNTIME_CAMPAIGN = (
    REPO / "study_results/v4p9p7_2016_support_combined_100toy_20260902"
)
sys.path.insert(0, str(RUNTIME_CAMPAIGN))
from runtime_guard import activate_and_verify, assert_import_origins  # noqa: E402


RUNTIME_PROVENANCE = activate_and_verify()
sys.path.insert(0, str(HERE / "runtime"))
sys.path.insert(0, str(HERE))
# The attested hps_gpr snapshot imports the repository's companion ``gp``
# package.  A directly-invoked release script does not otherwise inherit the
# repository root on sys.path, so add it after the frozen runtime paths.
sys.path.append(str(REPO))

import gp as _gp  # noqa: E402

GP_PACKAGE_ORIGIN = str(Path(_gp.__file__).resolve())
try:
    Path(GP_PACKAGE_ORIGIN).relative_to((REPO / "gp").resolve())
except ValueError as exc:
    raise RuntimeError(
        f"gp package escaped the repository checkout: {GP_PACKAGE_ORIGIN}"
    ) from exc

from hps_gpr.config import load_config  # noqa: E402
from hps_gpr.conversion import A_from_epsilon2  # noqa: E402
from hps_gpr.dataset import make_datasets  # noqa: E402
from hps_gpr.evaluation import (  # noqa: E402
    _dataset_visibility,
    active_datasets_for_mass,
    build_combined_components,
)
from hps_gpr.gpr import make_fixed_kernel  # noqa: E402
from hps_gpr.io import estimate_background_for_dataset  # noqa: E402
from hps_gpr.statistics import (  # noqa: E402
    _chol_with_jitter,
    p0_profiled_gaussian_LRT,
)
from piecewise_cached_solver import (  # noqa: E402
    SOLVER_VERSION,
    CachedPiecewiseBoundedLimit,
)


REQUIRED_RUNTIME_MODULES = (
    "hps_gpr",
    "hps_gpr.config",
    "hps_gpr.conversion",
    "hps_gpr.dataset",
    "hps_gpr.evaluation",
    "hps_gpr.gpr",
    "hps_gpr.io",
    "hps_gpr.statistics",
    "hps_gpr.template",
)
RUNTIME_IMPORT_ORIGINS = assert_import_origins(REQUIRED_RUNTIME_MODULES)

DEFAULT_CARD = HERE / "inputs" / "analysis_card.yaml"
DEFAULT_STATES = HERE / "inputs" / "reviewed_gp_states.csv"
DEFAULT_INPUT_PROVENANCE = HERE / "inputs" / "analysis_input_provenance.json"
DEFAULT_OUTPUT = HERE / "derived"
LML_CLOSURE_ATOL = 5.0e-5

EXPECTED_HISTOGRAM_SHA256 = {
    "2015": "58ce717cde753d8566c754a73cb056560ed19e781fe9a43e8634111cc746531f",
    "2016": "c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301",
    "2021": "3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4",
}

DATASET_ORDER = ("2015", "2016", "2021")
EXPECTED_DATASET_GRIDS = {
    "2015": tuple(range(19, 91)),
    "2016": tuple(range(39, 181)),
    "2021": tuple(range(50, 251)),
}
EXPECTED_SUPPORTS = {
    "2015": (14, 135),
    "2021": (36, 300),
}
SCOPES = (
    ("individual_2015_full", "2015 full", ("2015",), 19, 90),
    ("individual_2016_full", "2016 full", ("2016",), 39, 180),
    ("individual_2021_10pct", "2021 10%", ("2021",), 50, 250),
    ("pair_2015_2016", "2015 full + 2016 full", ("2015", "2016"), 39, 90),
    ("pair_2015_2021", "2015 full + 2021 10%", ("2015", "2021"), 50, 90),
    ("pair_2016_2021", "2016 full + 2021 10%", ("2016", "2021"), 50, 180),
    (
        "all_2015_2016_2021",
        "2015 full + 2016 full + 2021 10%",
        ("2015", "2016", "2021"),
        50,
        90,
    ),
)
EXPECTED_SCOPE_ROWS = {
    key: high - low + 1 for key, _, _, low, high in SCOPES
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_sha256(frame: pd.DataFrame) -> str:
    """Hash mass and GP coordinates independent of CSV column layout."""

    columns = ["mass_GeV", "const_opt", "ls_opt", "lml"]
    ordered = frame.sort_values("mass_GeV")[columns].astype(float)
    payload = [
        {key: float(value) for key, value in row.items()}
        for row in ordered.to_dict(orient="records")
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
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


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def prediction_state_sha256(prediction: object) -> str:
    digest = hashlib.sha256()
    for array in (prediction.mu, prediction.cov, prediction.obs):
        value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    digest.update(
        np.asarray(
            [prediction.sigma_val, prediction.integral_density], dtype="<f8"
        ).tobytes()
    )
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def condition_covariance_block(
    covariance: np.ndarray,
    background_mean: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply the smallest predeclared diagonal load giving Cholesky closure."""

    raw = np.asarray(covariance, dtype=float)
    mean = np.asarray(background_mean, dtype=float)
    if raw.ndim != 2 or raw.shape != (mean.size, mean.size):
        raise RuntimeError("covariance block has the wrong shape")
    if not np.isfinite(raw).all() or not np.isfinite(mean).all():
        raise RuntimeError("covariance block or background mean is non-finite")
    symmetric = 0.5 * (raw + raw.T)
    if not np.allclose(raw, raw.T, rtol=1.0e-10, atol=1.0e-10):
        raise RuntimeError("covariance asymmetry exceeds the frozen tolerance")
    diagonal = np.diag(symmetric)
    if np.any(diagonal < 0.0):
        raise RuntimeError("covariance block has a negative diagonal variance")
    scale = max(float(np.max(np.abs(diagonal))), 1.0)
    raw_eigenvalues = np.linalg.eigvalsh(symmetric)
    raw_minimum = float(np.min(raw_eigenvalues))
    identity = np.eye(mean.size)
    conditioned = None
    relative_load = float("nan")
    for candidate_relative in (1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4):
        candidate = symmetric + candidate_relative * scale * identity
        try:
            np.linalg.cholesky(candidate)
        except np.linalg.LinAlgError:
            continue
        conditioned = candidate
        relative_load = candidate_relative
        break
    if conditioned is None:
        raise RuntimeError("covariance block needs diagonal loading beyond 1e-4")
    if relative_load >= 1.0e-4:
        raise RuntimeError(
            "covariance block reaches the forbidden 1e-4 diagonal-loading cap"
        )

    # This is the factor the attested likelihood will deterministically form
    # after receiving the conditioned matrix.  Recording L L^T makes its extra
    # baseline regularization explicit rather than silently attributing it to C.
    core_factor = _chol_with_jitter(conditioned)
    core_effective = core_factor @ core_factor.T
    effective_v = core_effective + np.diag(mean)
    v_eigenvalues = np.linalg.eigvalsh(0.5 * (effective_v + effective_v.T))
    v_scale = max(float(np.max(np.abs(np.diag(effective_v)))), 1.0)
    if float(np.min(v_eigenvalues)) <= 0.0:
        raise RuntimeError("conditioned background-plus-Poisson covariance is not SPD")
    relative_diagonal_change = float(
        np.max(np.abs(np.diag(core_effective) - diagonal)) / scale
    )
    record = {
        "raw_covariance_sha256": array_sha256(symmetric),
        "raw_min_eigenvalue": raw_minimum,
        "raw_min_eigenvalue_relative": raw_minimum / scale,
        "covariance_scale": scale,
        "selected_diagonal_load_relative": relative_load,
        "selected_diagonal_load_absolute": relative_load * scale,
        "core_effective_diagonal_change_relative": relative_diagonal_change,
        "conditioned_covariance_sha256": array_sha256(conditioned),
        "core_effective_covariance_sha256": array_sha256(core_effective),
        "effective_v_sha256": array_sha256(effective_v),
        "effective_v_min_eigenvalue": float(np.min(v_eigenvalues)),
        "effective_v_min_eigenvalue_relative": float(np.min(v_eigenvalues)) / v_scale,
        "eigen_clipping_used": False,
    }
    return conditioned, record


def block_diagonal(arrays: Sequence[np.ndarray]) -> np.ndarray:
    size = sum(array.shape[0] for array in arrays)
    output = np.zeros((size, size), dtype=float)
    start = 0
    for array in arrays:
        stop = start + array.shape[0]
        output[start:stop, start:stop] = array
        start = stop
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument(
        "--input-provenance", type=Path, default=DEFAULT_INPUT_PROVENANCE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--mass-min-mev", type=int, default=19)
    parser.add_argument("--mass-max-mev", type=int, default=250)
    return parser.parse_args(argv)


def validate_card(config: object) -> None:
    expected_ranges = {
        "range_2015": (0.019, 0.090),
        "range_2016": (0.039, 0.180),
        "range_2021": (0.050, 0.250),
        "data_range_2015": (0.014, 0.135),
        "data_range_2021": (0.036, 0.300),
    }
    for key, expected in expected_ranges.items():
        found = tuple(float(item) for item in getattr(config, key))
        if found != expected:
            raise RuntimeError(f"analysis-card drift: {key}={found} != {expected}")
    support_2016 = tuple(
        int(round(1000.0 * float(item))) for item in config.data_range_2016
    )
    if support_2016 != (30, 210):
        raise RuntimeError(f"invalid frozen 2016 support {support_2016}")
    exact = {
        "cls_mode": "asymptotic",
        "cls_alpha": 0.1,
        "cls_num_toys": 0,
        "combined_mode": "count_scale",
        "gp_train_exclude_nsigma": 2.25,
        "scan_require_two_sidebands": True,
        "neighborhood_rebin": 5,
        "make_ul_bands": False,
        "ul_bands_toys": 0,
        "do_combined_bands": False,
        "combined_bands_n_toys": 0,
        "make_eps2_bands": False,
    }
    for key, expected in exact.items():
        if getattr(config, key) != expected:
            raise RuntimeError(
                f"analysis-card drift: {key}={getattr(config, key)!r} != {expected!r}"
            )
    if float(config.kernel_ls_res_upper_factor_by_dataset["2016"]) != 12.0:
        raise RuntimeError("the frozen 2016 upper length-scale factor is not 12")
    if any(
        _dataset_visibility_key != "observed"
        for _dataset_visibility_key in dict(config.data_visibility).values()
    ):
        raise RuntimeError("all three final inputs must be marked observed")


def load_states(path: Path, config: object) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "mass_GeV",
        "const_opt",
        "ls_opt",
        "lml",
        "interpolated",
        "gp_support_low_MeV",
        "gp_support_high_MeV",
        "source_state",
        "source_ledger_path",
        "source_ledger_sha256",
        "combination_authorization_sha256",
        "dataset_support_decision_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"reviewed state ledger lacks columns: {missing}")
    frame["dataset"] = frame["dataset"].astype(str)
    if len(frame) != 415 or frame.duplicated(["dataset", "mass_GeV"]).any():
        raise RuntimeError("reviewed state ledger is not the exact 415 unique states")
    interpolated = frame["interpolated"].astype(str).str.lower().str.strip()
    if not interpolated.isin({"false", "0", "no"}).all():
        raise RuntimeError("interpolated GP states are forbidden")
    expected_supports = dict(EXPECTED_SUPPORTS)
    expected_supports["2016"] = tuple(
        int(round(1000.0 * float(item))) for item in config.data_range_2016
    )
    for dataset in DATASET_ORDER:
        rows = frame[frame["dataset"] == dataset].sort_values("mass_GeV")
        masses = tuple(int(round(1000.0 * value)) for value in rows.mass_GeV)
        if masses != EXPECTED_DATASET_GRIDS[dataset]:
            raise RuntimeError(f"reviewed {dataset} mass grid is not exact")
        supports = set(
            zip(
                rows.gp_support_low_MeV.astype(int),
                rows.gp_support_high_MeV.astype(int),
            )
        )
        if supports != {expected_supports[dataset]}:
            raise RuntimeError(
                f"reviewed {dataset} supports {supports} != {expected_supports[dataset]}"
            )
        numeric = rows[["const_opt", "ls_opt", "lml"]].to_numpy(float)
        if not np.isfinite(numeric).all() or np.any(rows["const_opt"] <= 0.0) or np.any(rows["ls_opt"] <= 0.0):
            raise RuntimeError(f"reviewed {dataset} GP coordinates are invalid")
        source_paths = set(rows.source_ledger_path.astype(str))
        source_hashes = set(rows.source_ledger_sha256.astype(str))
        if len(source_paths) != 1 or len(source_hashes) != 1:
            raise RuntimeError(f"reviewed {dataset} source provenance is not unique")
        source_path = Path(next(iter(source_paths))).expanduser().resolve()
        if not source_path.is_file() or sha256(source_path) != next(iter(source_hashes)):
            raise RuntimeError(f"reviewed {dataset} source ledger hash does not close")
        source = pd.read_csv(
            source_path,
            usecols=lambda column: column
            in {"dataset", "mass_GeV", "const_opt", "ls_opt", "lml", "interpolated"},
        )
        if "dataset" in source.columns:
            source = source[source.dataset.astype(str) == dataset].copy()
        required_source = {"mass_GeV", "const_opt", "ls_opt", "lml"}
        if not required_source <= set(source.columns):
            raise RuntimeError(
                f"reviewed {dataset} source ledger lacks exact GP coordinates"
            )
        source["mass_MeV_join"] = np.rint(
            1000.0 * source.mass_GeV.astype(float)
        ).astype(int)
        if source.duplicated("mass_MeV_join").any():
            raise RuntimeError(f"reviewed {dataset} source mass grid is duplicated")
        selected = rows.copy()
        selected["mass_MeV_join"] = np.rint(
            1000.0 * selected.mass_GeV.astype(float)
        ).astype(int)
        joined = selected.merge(
            source[["mass_MeV_join", "mass_GeV", "const_opt", "ls_opt", "lml"]],
            on="mass_MeV_join",
            how="left",
            validate="one_to_one",
            suffixes=("_selected", "_source"),
        )
        if len(joined) != len(rows) or joined["mass_GeV_source"].isna().any():
            raise RuntimeError(f"reviewed {dataset} states do not join to their source")
        for coordinate in ("mass_GeV", "const_opt", "ls_opt", "lml"):
            if not np.allclose(
                joined[f"{coordinate}_selected"],
                joined[f"{coordinate}_source"],
                rtol=2.0e-13,
                atol=2.0e-13,
            ):
                raise RuntimeError(
                    f"reviewed {dataset} {coordinate} differs from its source ledger"
                )
        if "interpolated" in source.columns:
            source_interpolated = (
                source.set_index("mass_MeV_join")
                .loc[list(masses), "interpolated"]
                .astype(str)
                .str.lower()
                .str.strip()
            )
            if not source_interpolated.isin({"false", "0", "no"}).all():
                raise RuntimeError(f"reviewed {dataset} source contains interpolation")
    return frame


def validate_input_provenance(
    path: Path, card: Path, states: Path, config: object
) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "status": "phase_c_conditional_inputs_frozen_with_numerical_exception",
        "analysis_card_sha256": sha256(card),
        "reviewed_gp_states_sha256": sha256(states),
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"analysis-input provenance drift: {key}={payload.get(key)!r}"
            )
    decision_path = Path(str(payload["combination_authorization_path"])).expanduser().resolve()
    decision_sha = str(payload["combination_authorization_sha256"])
    if not decision_path.is_file() or sha256(decision_path) != decision_sha:
        raise RuntimeError("combination-authorization hash does not close")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not (
        decision.get("status") == "all_142_states_certified"
        and decision.get("combination_authorized") is True
        and int(decision.get("state_rows", -1)) == 142
        and int(decision.get("resolved_rows", -1)) == 142
        and decision.get("unresolved_masses_MeV") == []
        and int(decision.get("support_lower_MeV", -1)) == 30
        and int(decision.get("support_upper_MeV", -1)) == 210
        and float(decision.get("upper_length_factor_2016", float("nan"))) == 12.0
    ):
        raise RuntimeError("2016 support/state decision did not authorize combination")
    exception_path = Path(str(payload["numerical_exception_path"])).expanduser().resolve()
    if not exception_path.is_file() or sha256(exception_path) != payload.get("numerical_exception_sha256"):
        raise RuntimeError("2016 numerical-exception hash does not close")
    exception = json.loads(exception_path.read_text(encoding="utf-8"))
    if not (
        exception.get("status") == "conditional_user_accepted_numerical_exception"
        and exception.get("p1_combination_authorized") is False
        and exception.get("independent_state_certification") is False
        and exception.get("exception_accepts_exact_frozen_coordinates_without_reoptimization") is True
        and payload.get("p1_combination_authorized") is False
        and payload.get("independent_state_certification_2016") is False
    ):
        raise RuntimeError("2016 numerical-exception semantics drift")
    support = tuple(
        int(round(1000.0 * float(item))) for item in config.data_range_2016
    )
    if tuple(payload.get("selected_support_2016_MeV", [])) != support:
        raise RuntimeError("card support does not match the frozen support decision")
    factor = float(config.kernel_ls_res_upper_factor_by_dataset["2016"])
    if not np.isclose(
        float(payload.get("selected_ls_upper_factor_2016", float("nan"))),
        factor,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("card length-scale cap does not match support provenance")
    state_frame = pd.read_csv(states)
    state_decisions = set(state_frame.combination_authorization_sha256.astype(str))
    if state_decisions != {decision_sha}:
        raise RuntimeError("reviewed states do not bind the exact combination authorization")
    support_decisions = dict(payload.get("dataset_support_decisions", {}))
    if set(support_decisions) != set(DATASET_ORDER):
        raise RuntimeError("dataset support-decision set is not exact")
    for dataset, record in support_decisions.items():
        support_path = Path(str(record["path"])).expanduser().resolve()
        support_sha = str(record["sha256"])
        if not support_path.is_file() or sha256(support_path) != support_sha:
            raise RuntimeError(f"{dataset} support-decision hash does not close")
        selected_support_shas = set(
            state_frame.loc[
                state_frame.dataset.astype(str) == dataset,
                "dataset_support_decision_sha256",
            ].astype(str)
        )
        if selected_support_shas != {support_sha}:
            raise RuntimeError(f"{dataset} states do not bind their own support decision")
    certifications = dict(payload.get("dataset_certifications", {}))
    if set(certifications) != set(DATASET_ORDER):
        raise RuntimeError("dataset certification set is not exact")
    for dataset in DATASET_ORDER:
        entry = dict(certifications[dataset])
        certificate_path = Path(str(entry["certificate_path"])).expanduser().resolve()
        source_path = Path(str(entry["source_ledger_path"])).expanduser().resolve()
        if not certificate_path.is_file() or sha256(certificate_path) != entry.get(
            "certificate_sha256"
        ):
            raise RuntimeError(f"{dataset} certification artifact hash does not close")
        if not source_path.is_file() or sha256(source_path) != entry.get(
            "source_ledger_sha256"
        ):
            raise RuntimeError(f"{dataset} certified source ledger hash does not close")
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        expected_certificate_status = (
            "conditional_user_accepted_numerical_exception"
            if dataset == "2016"
            else "qualified_for_final_inference"
        )
        if not (
            certificate.get("dataset") == dataset
            and certificate.get("status") == expected_certificate_status
            and certificate.get("passed") is True
            and certificate.get("source_ledger_sha256") == sha256(source_path)
        ):
            raise RuntimeError(f"{dataset} certification did not pass")
        if dataset == "2016" and certificate.get("independent_state_certification") is not False:
            raise RuntimeError("2016 certificate obscures the failed independent replay")
        selected = state_frame[state_frame.dataset.astype(str) == dataset]
        if certificate.get("certified_coordinate_sha256") != coordinate_sha256(
            selected
        ):
            raise RuntimeError(f"{dataset} certified coordinate hash does not close")
        selected_paths = set(selected.source_ledger_path.astype(str))
        selected_hashes = set(selected.source_ledger_sha256.astype(str))
        if selected_paths != {str(source_path)} or selected_hashes != {sha256(source_path)}:
            raise RuntimeError(f"{dataset} states do not bind the certified source")
        if dataset == "2016" and not (
            int(decision["states"]["rows"]) == 142
            and decision["states"]["sha256"] == sha256(source_path)
        ):
            raise RuntimeError("2016 final-state ledger does not match its decision")
    return payload


def validate_histogram_inputs(config: object) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    expected_hist_names = {
        "2015": "invariant_mass",
        "2016": "h_Minv_General_Final_1",
        "2021": "preselection/h_invM_8000",
    }
    for dataset in DATASET_ORDER:
        path = Path(getattr(config, f"path_{dataset}")).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"missing immutable {dataset} histogram file: {path}")
        digest = sha256(path)
        if digest != EXPECTED_HISTOGRAM_SHA256[dataset]:
            raise RuntimeError(
                f"{dataset} histogram file SHA-256 {digest} is not frozen"
            )
        hist_name = str(getattr(config, f"hist_{dataset}"))
        if hist_name != expected_hist_names[dataset]:
            raise RuntimeError(f"unexpected {dataset} histogram name {hist_name!r}")
        result[dataset] = {
            "path": str(path),
            "sha256": digest,
            "histogram": hist_name,
        }
    return result


def state_map(frame: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, object]]:
    result: Dict[Tuple[str, int], Dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        key = (str(row["dataset"]), int(round(1000.0 * float(row["mass_GeV"]))))
        result[key] = row
    return result


def reconstruct_predictions(
    mass_gev: float,
    datasets: Dict[str, object],
    config: object,
    states: Dict[Tuple[str, int], Dict[str, object]],
) -> Tuple[
    Dict[str, object],
    Dict[str, np.ndarray],
    Dict[str, Dict[str, object]],
    List[Dict[str, object]],
]:
    mass_mev = int(round(1000.0 * mass_gev))
    active = active_datasets_for_mass(mass_gev, datasets, config)
    found_keys = tuple(dataset.key for dataset in active)
    expected_keys = tuple(
        key
        for key in DATASET_ORDER
        if mass_mev in EXPECTED_DATASET_GRIDS[key]
    )
    if found_keys != expected_keys:
        raise RuntimeError(
            f"active-set drift at {mass_mev} MeV: {found_keys} != {expected_keys}"
        )
    predictions: Dict[str, object] = {}
    conditioned_covariances: Dict[str, np.ndarray] = {}
    conditioning_records: Dict[str, Dict[str, object]] = {}
    records: List[Dict[str, object]] = []
    with threadpool_limits(limits=1):
        for dataset in active:
            reviewed = states[(dataset.key, mass_mev)]
            prediction = estimate_background_for_dataset(
                dataset,
                mass_gev,
                config,
                restarts=0,
                train_exclude_nsigma=float(config.gp_train_exclude_nsigma),
                kernel=make_fixed_kernel(
                    float(reviewed["const_opt"]), float(reviewed["ls_opt"])
                ),
                optimize=False,
            )
            delta_lml = float(prediction.lml - float(reviewed["lml"]))
            if not np.isfinite(delta_lml) or abs(delta_lml) > LML_CLOSURE_ATOL:
                raise RuntimeError(
                    f"LML closure failed for {dataset.key} at {mass_mev} MeV: "
                    f"delta={delta_lml:.12g}"
                )
            if not (
                np.isfinite(prediction.mu).all()
                and np.all(np.asarray(prediction.mu) > 0.0)
                and np.isfinite(prediction.obs).all()
                and np.all(np.asarray(prediction.obs) >= 0.0)
            ):
                raise RuntimeError(
                    f"invalid prediction vectors for {dataset.key} at {mass_mev} MeV"
                )
            conditioned_covariance, conditioning = condition_covariance_block(
                np.asarray(prediction.cov, dtype=float),
                np.asarray(prediction.mu, dtype=float),
            )
            state_sha = prediction_state_sha256(prediction)
            k_factor = float(
                A_from_epsilon2(
                    dataset,
                    mass_gev,
                    1.0,
                    prediction.integral_density,
                )
            )
            if not np.isfinite(k_factor) or k_factor <= 0.0:
                raise RuntimeError(
                    f"invalid signal normalization for {dataset.key} at {mass_mev} MeV"
                )
            predictions[dataset.key] = prediction
            conditioned_covariances[dataset.key] = conditioned_covariance
            conditioning_records[dataset.key] = conditioning
            records.append(
                {
                    "dataset": dataset.key,
                    "mass_GeV": mass_gev,
                    "mass_MeV": mass_mev,
                    "const_opt": float(reviewed["const_opt"]),
                    "ls_opt": float(reviewed["ls_opt"]),
                    "reviewed_lml": float(reviewed["lml"]),
                    "recomputed_lml": float(prediction.lml),
                    "lml_delta": delta_lml,
                    "prediction_state_sha256": state_sha,
                    "sigma_mass_res_GeV": float(prediction.sigma_val),
                    "integral_density_events_per_GeV": float(
                        prediction.integral_density
                    ),
                    "signal_yield_per_eps2": k_factor,
                    "n_signal_window_bins": int(len(prediction.obs)),
                    "n_training_bins": int(prediction.n_train),
                    "covariance_symmetric": True,
                    **conditioning,
                    "gp_support_low_MeV": int(reviewed["gp_support_low_MeV"]),
                    "gp_support_high_MeV": int(reviewed["gp_support_high_MeV"]),
                    "source_state": str(reviewed["source_state"]),
                    "source_ledger_path": str(reviewed["source_ledger_path"]),
                    "source_ledger_sha256": str(reviewed["source_ledger_sha256"]),
                    "combination_authorization_sha256": str(
                        reviewed["combination_authorization_sha256"]
                    ),
                    "dataset_support_decision_sha256": str(
                        reviewed["dataset_support_decision_sha256"]
                    ),
                }
            )
    return predictions, conditioned_covariances, conditioning_records, records


def scope_rows_at_mass(
    mass_gev: float,
    datasets: Dict[str, object],
    predictions: Dict[str, object],
    conditioned_covariances: Dict[str, np.ndarray],
    conditioning_records: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    mass_mev = int(round(1000.0 * mass_gev))
    rows: List[Dict[str, object]] = []
    for scope_key, scope_label, keys, low, high in SCOPES:
        if not low <= mass_mev <= high:
            continue
        ds_here = [datasets[key] for key in keys]
        pred_here = [predictions[key] for key in keys]
        if not all(_dataset_visibility(ds, result_config) == "observed" for ds in ds_here):
            raise RuntimeError(f"non-observed input entered {scope_key}")
        obs, bkg, _raw_cov, s_unit = build_combined_components(
            mass_gev, ds_here, pred_here, config=result_config
        )
        cov = block_diagonal([conditioned_covariances[key] for key in keys])
        core_factor = _chol_with_jitter(cov)
        core_effective_covariance = core_factor @ core_factor.T
        effective_v = core_effective_covariance + np.diag(bkg)
        effective_v_eigenvalues = np.linalg.eigvalsh(
            0.5 * (effective_v + effective_v.T)
        )
        effective_v_scale = max(
            float(np.max(np.abs(np.diag(effective_v)))), 1.0
        )
        if float(np.min(effective_v_eigenvalues)) <= 0.0:
            raise RuntimeError(f"combined effective covariance is not SPD for {scope_key}")
        solver = CachedPiecewiseBoundedLimit(
            bkg,
            cov,
            s_unit,
            alpha=float(result_config.cls_alpha),
            combined_mode=str(result_config.combined_mode),
        )
        limit = solver.limit(obs)
        signal_scale = float(np.sum(s_unit))
        if not np.isclose(
            signal_scale,
            limit.signal_scale_counts_per_eps2,
            rtol=2.0e-15,
            atol=0.0,
        ):
            raise RuntimeError("limit solver count-scale normalization drift")
        signal_template = s_unit / signal_scale
        p0, z_value, q0, p0_info = p0_profiled_gaussian_LRT(
            obs,
            bkg,
            cov,
            signal_template,
        )
        if not bool(p0_info.get("ok", False)):
            raise RuntimeError(
                f"local discovery profile did not converge for {scope_key} at {mass_mev} MeV"
            )
        nll_alt = float(p0_info.get("nll_alt", float("nan")))
        nll0 = float(p0_info.get("nll0", float("nan")))
        p0_null_feasible_fallback = False
        p0_raw_nll_alt = nll_alt
        # A=0 is an exact feasible point of the A>=0 alternative.  If the
        # bounded optimizer stops just above the independently profiled null,
        # retain the better known feasible point.  The runtime already clips
        # q0 to zero in this case, so this only reconciles the fitted state.
        if np.isfinite(nll_alt) and np.isfinite(nll0) and nll0 < nll_alt:
            p0_null_feasible_fallback = True
            nll_alt = nll0
            p0_info = dict(p0_info)
            p0_info.update(
                A_hat=0.0,
                nll_alt=nll0,
                q0=0.0,
                Z=0.0,
                p0=0.5,
                null_feasible_fallback=True,
                raw_nll_alt_before_null_fallback=p0_raw_nll_alt,
            )
            p0, z_value, q0 = 0.5, 0.0, 0.0
        nll_difference_scale = max(1.0, abs(nll0 - nll_alt))
        nll_tolerance = 1.0e-6 + 1.0e-8 * nll_difference_scale
        if not (
            np.isfinite(nll_alt)
            and np.isfinite(nll0)
            and nll_alt <= nll0 + nll_tolerance
        ):
            raise RuntimeError(
                f"local-p0 likelihood nesting failed for {scope_key} at {mass_mev} MeV"
            )
        if not (
            np.isfinite(limit.eps2_90)
            and limit.eps2_90 > 0.0
            and np.isfinite(p0)
            and 0.0 < p0 <= 0.5
            and np.isfinite(z_value)
            and z_value >= 0.0
        ):
            raise RuntimeError(f"invalid result for {scope_key} at {mass_mev} MeV")
        k_by_dataset = {
            dataset.key: float(
                A_from_epsilon2(
                    dataset,
                    mass_gev,
                    1.0,
                    prediction.integral_density,
                )
            )
            for dataset, prediction in zip(ds_here, pred_here)
        }
        state_hashes = {
            key: prediction_state_sha256(predictions[key]) for key in keys
        }
        eps2_hat = float(p0_info.get("A_hat", float("nan"))) / signal_scale
        sigma_eps2 = float(p0_info.get("sigma_A", float("nan"))) / signal_scale
        fitted_window_A90 = float(limit.eps2_90 * signal_scale)
        rows.append(
            {
                "group": "individual" if len(keys) == 1 else "combination",
                "scope_key": scope_key,
                "scope_label": scope_label,
                "dataset_set": "+".join(keys),
                "mass_GeV": mass_gev,
                "mass_MeV": mass_mev,
                "A90_events": float(limit.eps2_90 * sum(k_by_dataset.values())),
                "A90_full_template_events": float(
                    limit.eps2_90 * sum(k_by_dataset.values())
                ),
                "A90_fitted_window_events": fitted_window_A90,
                "yield_coordinate": "summed full-template signal yield",
                "eps2_90": float(limit.eps2_90),
                "cls_alpha": float(limit.alpha),
                "confidence_level": float(limit.confidence_level),
                "p0_local_asymptotic": float(p0),
                "Z_local_asymptotic": float(z_value),
                "q0_local_asymptotic": float(q0),
                "eps2_hat_bounded_for_p0": eps2_hat,
                "sigma_eps2_hat_bounded_for_p0": sigma_eps2,
                "p0_null_feasible_fallback": p0_null_feasible_fallback,
                "p0_raw_nll_alt_before_null_fallback": p0_raw_nll_alt,
                "p0_profile_status": json.dumps(
                    {
                        "ok": bool(p0_info.get("ok", False)),
                        "ok_alt": bool(p0_info.get("ok_alt", False)),
                        "ok_null": bool(p0_info.get("ok_null", False)),
                        "nll_alt": nll_alt,
                        "nll0": nll0,
                        "raw_nll_alt_before_null_fallback": p0_raw_nll_alt,
                        "null_feasible_fallback": p0_null_feasible_fallback,
                        "fallback_nll_improvement": float(
                            p0_raw_nll_alt - nll_alt
                        ),
                        "nll_nesting_tolerance": nll_tolerance,
                        "A_hat_fitted_window_counts": float(
                            p0_info.get("A_hat", float("nan"))
                        ),
                        "sigma_A_fitted_window_counts": float(
                            p0_info.get("sigma_A", float("nan"))
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "signal_yield_per_eps2_total": float(sum(k_by_dataset.values())),
                "signal_yield_per_eps2_fitted_window": signal_scale,
                "signal_yield_per_eps2_by_dataset": json.dumps(
                    k_by_dataset, sort_keys=True, separators=(",", ":")
                ),
                "gp_state_sha256_by_dataset": json.dumps(
                    state_hashes, sort_keys=True, separators=(",", ":")
                ),
                "gp_support_by_dataset": json.dumps(
                    {
                        key: [
                            int(round(1000.0 * float(getattr(result_config, f"data_range_{key}")[0]))),
                            int(round(1000.0 * float(getattr(result_config, f"data_range_{key}")[1]))),
                        ]
                        for key in keys
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "covariance_conditioning_by_dataset": json.dumps(
                    {key: conditioning_records[key] for key in keys},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "conditioned_combined_covariance_sha256": array_sha256(cov),
                "core_effective_combined_covariance_sha256": array_sha256(
                    core_effective_covariance
                ),
                "effective_combined_v_sha256": array_sha256(effective_v),
                "effective_v_min_eigenvalue_relative": float(
                    np.min(effective_v_eigenvalues) / effective_v_scale
                ),
                "cls_at_limit": float(limit.cls_at_limit),
                "cl_sb_at_limit": float(limit.cl_sb_at_limit),
                "cl_b_at_limit": float(limit.cl_b_at_limit),
                "log_cls_at_limit": float(limit.log_cls_at_limit),
                "log_cl_sb_at_limit": float(limit.log_cl_sb_at_limit),
                "log_cl_b_at_limit": float(limit.log_cl_b_at_limit),
                "qmu_obs_at_limit": float(limit.qmu_obs_at_limit),
                "qmu_asimov_b_at_limit": float(limit.qmu_asimov_b_at_limit),
                "tail_branch_at_limit": limit.tail_branch_at_limit,
                "z_sb_at_limit": float(limit.z_sb_at_limit),
                "z_b_at_limit": float(limit.z_b_at_limit),
                "qmu_profile_branch_at_limit": limit.observed_qmu_branch_at_limit,
                "limit_fit_unconstrained_eps2": float(
                    limit.observed_unconstrained_strength
                    / limit.signal_scale_counts_per_eps2
                ),
                "limit_profile_optimizer_ok": bool(limit.optimizer_ok),
                "limit_profile_status": json.dumps(
                    limit.profile_status, sort_keys=True, separators=(",", ":")
                ),
                "limit_solver_counters": json.dumps(
                    limit.counters, sort_keys=True, separators=(",", ":")
                ),
                "limit_solver": limit.solver_version,
                "limit_method": "observed 90% CLs, bounded tilde-q_mu, piecewise asymptotic",
                "pvalue_method": "fixed-mass local asymptotic one-sided profile LRT",
                "combined_mode": "count_scale",
                "conditional_on_frozen_gp": True,
                "cls_bisection_iterations": int(limit.bisection_iterations),
                "cls_bracket_expansions": int(limit.bracket_expansions),
                "cls_bracket_low_eps2": float(limit.bracket_low_eps2),
                "cls_bracket_high_eps2": float(limit.bracket_high_eps2),
                "cls_bracket_low_value": float(limit.bracket_low_cls),
                "cls_bracket_high_value": float(limit.bracket_high_cls),
                "cls_convergence_reason": limit.convergence_reason,
            }
        )
    return rows


def run_mass(
    mass_mev: int,
    datasets: Dict[str, object],
    config: object,
    states: Dict[Tuple[str, int], Dict[str, object]],
    contract_sha256: str,
    work_dir: Path,
) -> Path:
    if assert_import_origins(REQUIRED_RUNTIME_MODULES) != RUNTIME_IMPORT_ORIGINS:
        raise RuntimeError("worker imported a different hps_gpr runtime")
    mass_gev = mass_mev / 1000.0
    (
        predictions,
        conditioned_covariances,
        conditioning_records,
        prediction_records,
    ) = reconstruct_predictions(
        mass_gev, datasets, config, states
    )
    result_rows = scope_rows_at_mass(
        mass_gev,
        datasets,
        predictions,
        conditioned_covariances,
        conditioning_records,
    )
    payload = {
        "schema_version": 1,
        "mass_MeV": mass_mev,
        "contract_sha256": contract_sha256,
        "runtime_import_origins": RUNTIME_IMPORT_ORIGINS,
        "gp_package_origin": GP_PACKAGE_ORIGIN,
        "prediction_rows": prediction_records,
        "result_rows": result_rows,
    }
    path = work_dir / f"m{mass_mev:03d}.json"
    atomic_json(path, payload)
    return path


def checkpoint_valid(path: Path, contract_sha256: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        payload.get("contract_sha256") == contract_sha256
        and int(payload.get("mass_MeV", -1)) == int(path.stem[1:])
    )


def contract_digest(card: Path, states: Path) -> Tuple[str, Dict[str, str]]:
    paths = {
        "analysis_card": card,
        "reviewed_gp_states": states,
        "runner": Path(__file__).resolve(),
        "cached_solver": HERE / "piecewise_cached_solver.py",
        "tail_mapper": HERE / "runtime" / "bounded_tildeq_cls.py",
        "statistical_protocol": HERE / "STATISTICAL_PROTOCOL.md",
        "conditioning_audit_protocol": (
            HERE / "NUMERICAL_CONDITIONING_AUDIT_PROTOCOL.md"
        ),
        "frozen_protocol_hash_ledger": HERE / "FROZEN_STATISTICAL_PROTOCOL_SHA256",
        "attested_runtime_manifest": Path(
            str(RUNTIME_PROVENANCE["runtime_manifest"])
        ),
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), hashes


def collect(
    paths: Iterable[Path], contract_sha256: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"stale checkpoint: {path}")
        result_rows.extend(payload["result_rows"])
        prediction_rows.extend(payload["prediction_rows"])
    results = pd.DataFrame(result_rows).sort_values(
        ["scope_key", "mass_MeV"]
    ).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["dataset", "mass_MeV"]
    ).reset_index(drop=True)
    return results, predictions


def validate_outputs(results: pd.DataFrame, predictions: pd.DataFrame) -> None:
    if len(results) != sum(EXPECTED_SCOPE_ROWS.values()):
        raise RuntimeError(f"expected 680 result rows, found {len(results)}")
    if len(predictions) != 415:
        raise RuntimeError(f"expected 415 prediction rows, found {len(predictions)}")
    if results.duplicated(["scope_key", "mass_MeV"]).any():
        raise RuntimeError("duplicate result coordinate")
    if predictions.duplicated(["dataset", "mass_MeV"]).any():
        raise RuntimeError("duplicate prediction coordinate")
    for scope, count in EXPECTED_SCOPE_ROWS.items():
        here = results[results.scope_key == scope].sort_values("mass_MeV")
        spec = next(item for item in SCOPES if item[0] == scope)
        if len(here) != count or not np.array_equal(
            here.mass_MeV.to_numpy(int), np.arange(spec[3], spec[4] + 1)
        ):
            raise RuntimeError(f"scope grid is not exact: {scope}")
    if not np.allclose(
        results.p0_local_asymptotic,
        __import__("scipy.stats", fromlist=["norm"]).norm.sf(
            results.Z_local_asymptotic
        ),
        rtol=2.0e-11,
        atol=1.0e-300,
    ):
        raise RuntimeError("p0/Z identity failed")
    if not np.allclose(results.cls_at_limit, 0.1, rtol=0.0, atol=2.0e-6):
        raise RuntimeError("one or more CLs roots do not close at 0.1")
    if set(results.limit_solver) != {SOLVER_VERSION}:
        raise RuntimeError("unexpected limit solver label")
    if set(results.combined_mode) != {"count_scale"}:
        raise RuntimeError("unexpected combined mode")
    if not results.limit_profile_optimizer_ok.astype(bool).all():
        raise RuntimeError("one or more limit profiles failed")
    if not (
        (results.cls_bracket_low_value > 0.1).all()
        and (results.cls_bracket_high_value <= 0.1).all()
        and (results.cls_bracket_low_eps2 < results.cls_bracket_high_eps2).all()
        and (
            (results.eps2_90 >= results.cls_bracket_low_eps2)
            & (results.eps2_90 <= results.cls_bracket_high_eps2)
        ).all()
    ):
        raise RuntimeError("one or more saved CLs brackets are not oriented")
    hash_columns = (
        predictions["effective_v_sha256"],
        results["effective_combined_v_sha256"],
    )
    for hashes in hash_columns:
        if not hashes.astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise RuntimeError("effective covariance hash is missing or malformed")
    loads = predictions.selected_diagonal_load_relative.to_numpy(float)
    if not (np.isfinite(loads).all() and np.all(loads < 1.0e-4)):
        raise RuntimeError("a covariance block reaches the forbidden 1e-4 load")
    allowed_sets = {
        "2015",
        "2016",
        "2021",
        "2015+2016",
        "2015+2021",
        "2016+2021",
        "2015+2016+2021",
    }
    if set(results.dataset_set.astype(str)) != allowed_sets:
        raise RuntimeError("non-final dataset leaked into result ledger")


def main(argv: Sequence[str] | None = None) -> None:
    global result_config
    args = parse_args(argv)
    card = args.card.expanduser().resolve()
    states_path = args.states.expanduser().resolve()
    input_provenance_path = args.input_provenance.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not card.is_file() or not states_path.is_file() or not input_provenance_path.is_file():
        raise SystemExit(
            "analysis card, reviewed state ledger, and input provenance are required"
        )
    result_config = load_config(card)
    validate_card(result_config)
    input_provenance = validate_input_provenance(
        input_provenance_path, card, states_path, result_config
    )
    histogram_inputs = validate_histogram_inputs(result_config)
    datasets = make_datasets(result_config)
    states_frame = load_states(states_path, result_config)
    states = state_map(states_frame)
    digest, input_hashes = contract_digest(card, states_path)
    input_hashes["analysis_input_provenance"] = sha256(input_provenance_path)
    encoded_hashes = json.dumps(
        input_hashes, sort_keys=True, separators=(",", ":")
    ).encode()
    digest = hashlib.sha256(encoded_hashes).hexdigest()
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)

    requested = list(range(args.mass_min_mev, args.mass_max_mev + 1))
    if requested != list(range(19, 251)):
        print("Running a partial diagnostic grid; final aggregation is disabled.")
    missing = [
        mass
        for mass in requested
        if not checkpoint_valid(work / f"m{mass:03d}.json", digest)
    ]
    if missing:
        joblib.Parallel(
            n_jobs=max(1, int(args.workers)),
            backend="threading",
            verbose=10,
        )(
            joblib.delayed(run_mass)(
                mass,
                datasets,
                result_config,
                states,
                digest,
                work,
            )
            for mass in missing
        )
    if requested != list(range(19, 251)):
        return

    paths = [work / f"m{mass:03d}.json" for mass in requested]
    if not all(checkpoint_valid(path, digest) for path in paths):
        raise RuntimeError("full-grid checkpoint set is incomplete")
    results, predictions = collect(paths, digest)
    validate_outputs(results, predictions)
    atomic_csv(output / "final_dataset_result_curves.csv", results)
    atomic_csv(output / "prediction_state_ledger.csv", predictions)

    minima = (
        results.loc[
            results.groupby("scope_key")["p0_local_asymptotic"].idxmin()
        ]
        .sort_values("scope_key")
        .reset_index(drop=True)
    )
    atomic_csv(output / "local_p0_minima.csv", minima)
    branch_counts = {
        str(key): int(value)
        for key, value in results.tail_branch_at_limit.value_counts().items()
    }
    summary = {
        "schema_version": 1,
        "status": "computed",
        "result_rows": int(len(results)),
        "prediction_rows": int(len(predictions)),
        "scope_rows": {
            key: int(value)
            for key, value in results.scope_key.value_counts().sort_index().items()
        },
        "tail_branch_counts": branch_counts,
        "all_three_minimum": minima[
            minima.scope_key == "all_2015_2016_2021"
        ].iloc[0][
            [
                "mass_MeV",
                "p0_local_asymptotic",
                "Z_local_asymptotic",
                "eps2_hat_bounded_for_p0",
                "sigma_eps2_hat_bounded_for_p0",
                "eps2_90",
            ]
        ].to_dict(),
        "contract_sha256": digest,
        "input_and_code_sha256": input_hashes,
        "analysis_input_provenance": input_provenance,
        "immutable_histogram_inputs": histogram_inputs,
        "runtime_provenance": RUNTIME_PROVENANCE,
        "runtime_import_origins": RUNTIME_IMPORT_ORIGINS,
        "gp_package_origin": GP_PACKAGE_ORIGIN,
        "claim_boundary": (
            "Observed fixed-mass asymptotic results conditional on frozen GP "
            "supports, reviewed GP states, and a disclosed 2016 cross-process "
            "numerical reproducibility exception; no scan-wide calibration, "
            "expected bands, toys, or coverage claim."
        ),
        "maximum_abs_gp_lml_replay_difference": float(
            predictions.lml_delta.astype(float).abs().max()
        ),
    }
    atomic_json(output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
