#!/usr/bin/env python3
"""Fail-closed structural validation of the v4.9.7 combined release products."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from runtime_guard import (  # noqa: E402
    activate_and_verify,
    assert_import_origins,
)

RUNTIME_PROVENANCE = activate_and_verify()

from hps_gpr.config import load_config

from run_combined_bands_cached_fixed_reviewed import (
    CACHE_ALGORITHM_VERSION,
    EXPECTED_DATASET_MASS_GRIDS_MEV,
    N_FULL_GRID_MASSES,
    N_TOYS_PER_MASS,
    SEED,
    full_mass_grid,
    load_support_freeze,
    sha256,
    validate_closure_report,
    validate_config_provenance,
    validate_input_hashes,
    validate_reviewed_provenance,
    validate_v4_geometry,
)

RUNTIME_IMPORT_ORIGINS = assert_import_origins(
    (
        "hps_gpr",
        "hps_gpr.config",
        "hps_gpr.conversion",
        "hps_gpr.dataset",
        "hps_gpr.evaluation",
        "hps_gpr.gpr",
        "hps_gpr.io",
        "hps_gpr.statistics",
    )
)


EXPECTED_ACTIVE_SETS = {
    "2015": 20,
    "2015+2016": 11,
    "2015+2016+2021": 41,
    "2016+2021": 90,
    "2021": 70,
}
QUANTILE_COLUMNS = (
    "eps2_lo2",
    "eps2_lo1",
    "eps2_med",
    "eps2_hi1",
    "eps2_hi2",
)
ALIASES = {
    "eps2_obs": "ul_eps2_obs",
    "eps2_lo2": "toy_eps2_uls_q02",
    "eps2_lo1": "toy_eps2_uls_q16",
    "eps2_med": "toy_eps2_uls_q50",
    "eps2_hi1": "toy_eps2_uls_q84",
    "eps2_hi2": "toy_eps2_uls_q97",
    "eps2_mean": "toy_eps2_uls_mean",
}


class ValidationError(RuntimeError):
    """Raised when a release invariant is violated."""


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def equal_float_columns(left: pd.Series, right: pd.Series) -> bool:
    a = left.to_numpy(float)
    b = right.to_numpy(float)
    return bool(np.array_equal(a, b, equal_nan=True))


def validate_reviewed_csv(path: Path, support: tuple[int, int]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "mass_GeV",
        "const_opt",
        "ls_opt",
        "lml",
        "interpolated",
        "source_sha256",
        "source_role",
        "source_row_index",
        "selected_support_low_MeV",
        "support_high_MeV",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValidationError(f"Reviewed ledger is missing columns: {missing}")
    if len(frame) != 415:
        raise ValidationError(f"Expected 415 reviewed rows, found {len(frame)}.")
    datasets = frame["dataset"].astype(str).str.replace(r"\.0$", "", regex=True)
    if frame.duplicated(["dataset", "mass_GeV"]).any():
        raise ValidationError("Reviewed ledger contains duplicate states.")
    normalized = frame["interpolated"].fillna("").astype(str).str.lower()
    if not normalized.isin({"false", "0", "no"}).all():
        raise ValidationError(
            "Reviewed interpolation flags are not explicitly all false."
        )
    pairs = set(
        zip(
            frame["selected_support_low_MeV"].astype(int),
            frame["support_high_MeV"].astype(int),
        )
    )
    if pairs != {support}:
        raise ValidationError("Reviewed ledger support metadata is inconsistent.")
    for dataset, expected_grid in EXPECTED_DATASET_MASS_GRIDS_MEV.items():
        rows = frame.loc[datasets == dataset]
        found = tuple(int(round(value * 1000.0)) for value in rows.mass_GeV)
        if found != expected_grid:
            raise ValidationError(f"Reviewed {dataset} grid is not exact.")
        values = rows.loc[:, ["const_opt", "ls_opt", "lml"]].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValidationError(f"Reviewed {dataset} coordinates are non-finite.")
    return frame


def validate_bands(path: Path) -> pd.DataFrame:
    bands = pd.read_csv(path)
    required = {
        "dataset_set",
        "mass_GeV",
        "eps2_obs",
        *QUANTILE_COLUMNS,
        *ALIASES.values(),
        "n_toys_requested",
        "n_toys_finite",
        "bands_seed_sequence_index",
        "tail_count_strong_le_observed",
        "tail_count_weak_ge_observed",
        "tail_count_equal_observed",
        "tail_count_two_sided_min",
        "p_strong",
        "p_weak",
        "p_two",
        "empirical_tail_resolution",
        "cls_statistic",
        "cls_calibration",
        "combined_mode",
        "global_method",
        "p0_scope",
        "empirical_tail_scope",
        "bands_refit_gp_on_toy",
        "limit_solver",
        "meta",
        "toy_eps2_uls_json",
        "toy_eps2_uls_sha256",
    }
    missing = sorted(required.difference(bands.columns))
    if missing:
        raise ValidationError(f"Band CSV is missing columns: {missing}")
    if len(bands) != N_FULL_GRID_MASSES:
        raise ValidationError(f"Expected 232 mass rows, found {len(bands)}.")
    expected_masses = np.asarray(full_mass_grid(), dtype=float)
    if not np.array_equal(bands["mass_GeV"].to_numpy(float), expected_masses):
        raise ValidationError("Band CSV does not contain the exact ordered mass grid.")
    expected_indices = np.arange(N_FULL_GRID_MASSES, dtype=int)
    if not np.array_equal(
        bands["bands_seed_sequence_index"].to_numpy(int), expected_indices
    ):
        raise ValidationError("SeedSequence child indices are not 0 through 231.")
    if not (bands["n_toys_requested"].astype(int) == N_TOYS_PER_MASS).all():
        raise ValidationError("A mass does not declare exactly 100 toys.")
    if not (bands["n_toys_finite"].astype(int) == N_TOYS_PER_MASS).all():
        raise ValidationError("A mass has fewer than 100 finite toy limits.")
    if int(bands["n_toys_requested"].sum()) != 23_200:
        raise ValidationError("Expected exactly 23,200 mass-local toy limits.")
    if bands["dataset_set"].value_counts().to_dict() != EXPECTED_ACTIVE_SETS:
        raise ValidationError("Active-dataset partition differs from the protocol.")

    numerical = bands.loc[:, ["eps2_obs", *QUANTILE_COLUMNS, "eps2_mean"]]
    if not np.isfinite(numerical.to_numpy(float)).all():
        raise ValidationError("Observed or band limits contain non-finite values.")
    if not (numerical.to_numpy(float) >= 0.0).all():
        raise ValidationError("Observed or band limits contain negative values.")
    quantiles = bands.loc[:, QUANTILE_COLUMNS].to_numpy(float)
    if not (np.diff(quantiles, axis=1) >= 0.0).all():
        raise ValidationError("Band quantiles are not ordered.")
    for canonical, alias in ALIASES.items():
        if not equal_float_columns(bands[canonical], bands[alias]):
            raise ValidationError(f"Alias {alias} does not equal {canonical}.")

    stored_toy_count = 0
    quantile_fields = dict(
        zip(QUANTILE_COLUMNS, (0.025, 0.16, 0.50, 0.84, 0.975))
    )
    for row in bands.itertuples(index=False):
        n = int(row.n_toys_finite)
        toys = np.asarray(json.loads(row.toy_eps2_uls_json), dtype="<f8")
        if toys.shape != (N_TOYS_PER_MASS,) or not np.isfinite(toys).all():
            raise ValidationError(
                f"Stored toy limits fail at {row.mass_GeV:.3f}."
            )
        stored_toy_count += int(toys.size)
        toy_hash = hashlib.sha256(
            np.ascontiguousarray(toys, dtype="<f8").tobytes()
        ).hexdigest()
        if toy_hash != str(row.toy_eps2_uls_sha256):
            raise ValidationError(
                f"Stored toy-limit hash fails at {row.mass_GeV:.3f}."
            )
        for field, probability in quantile_fields.items():
            expected = float(np.quantile(toys, probability))
            if not np.isclose(
                float(getattr(row, field)),
                expected,
                rtol=5.0e-14,
                atol=0.0,
            ):
                raise ValidationError(
                    f"Stored-toy quantile {field} fails at {row.mass_GeV:.3f}."
                )
        if not np.isclose(
            float(row.eps2_mean),
            float(np.mean(toys)),
            rtol=5.0e-14,
            atol=0.0,
        ):
            raise ValidationError(
                f"Stored-toy mean fails at {row.mass_GeV:.3f}."
            )
        strong = int(row.tail_count_strong_le_observed)
        weak = int(row.tail_count_weak_ge_observed)
        equal = int(row.tail_count_equal_observed)
        two_min = int(row.tail_count_two_sided_min)
        if strong + weak - equal != n:
            raise ValidationError(f"Tail partition fails at {row.mass_GeV:.3f}.")
        if two_min != min(strong, weak):
            raise ValidationError(f"Two-sided tail count fails at {row.mass_GeV:.3f}.")
        if not np.isclose(
            float(row.p_strong), float(strong / n), rtol=0.0, atol=1.0e-15
        ):
            raise ValidationError(f"Strong tail fraction fails at {row.mass_GeV:.3f}.")
        if not np.isclose(
            float(row.p_weak), float(weak / n), rtol=0.0, atol=1.0e-15
        ):
            raise ValidationError(f"Weak tail fraction fails at {row.mass_GeV:.3f}.")
        expected_two = min(1.0, 2.0 * min(strong / n, weak / n))
        if not np.isclose(
            float(row.p_two), float(expected_two), rtol=0.0, atol=1.0e-15
        ):
            raise ValidationError(f"Two-sided tail fraction fails at {row.mass_GeV:.3f}.")
        if float(row.empirical_tail_resolution) != 0.01:
            raise ValidationError(f"Tail resolution is not 0.01 at {row.mass_GeV:.3f}.")
        expected_keys = str(row.dataset_set).split("+")
        metadata = json.loads(row.meta)
        if [str(item["key"]) for item in metadata] != expected_keys:
            raise ValidationError(f"Per-state metadata fails at {row.mass_GeV:.3f}.")
        if strong != int(np.count_nonzero(toys <= float(row.eps2_obs))):
            raise ValidationError(
                f"Stored-toy strong count fails at {row.mass_GeV:.3f}."
            )
        if weak != int(np.count_nonzero(toys >= float(row.eps2_obs))):
            raise ValidationError(
                f"Stored-toy weak count fails at {row.mass_GeV:.3f}."
            )
        if equal != int(np.count_nonzero(toys == float(row.eps2_obs))):
            raise ValidationError(
                f"Stored-toy equality count fails at {row.mass_GeV:.3f}."
            )

    if stored_toy_count != 23_200:
        raise ValidationError(
            f"Expected 23,200 stored toy limits, found {stored_toy_count}."
        )

    if set(bands["cls_statistic"].astype(str)) != {"tilde_q_mu"}:
        raise ValidationError("Unexpected CLs statistic.")
    if set(bands["cls_calibration"].astype(str)) != {"asymptotic"}:
        raise ValidationError("Inner CLs is not asymptotic.")
    if set(bands["combined_mode"].astype(str)) != {"count_scale"}:
        raise ValidationError("Combined mode is not count_scale.")
    if set(bands["global_method"].astype(str)) != {"not_evaluated"}:
        raise ValidationError("The result incorrectly declares a global method.")
    if set(bands["p0_scope"].astype(str)) != {
        "mass_local_analytic_diagnostic"
    }:
        raise ValidationError("Analytic p0 scope is not explicitly mass-local.")
    if set(bands["empirical_tail_scope"].astype(str)) != {
        "observed_limit_rank_diagnostic_not_significance"
    }:
        raise ValidationError("Toy tail ranks are not bounded away from significance.")
    if set(bands["limit_solver"].astype(str)) != {CACHE_ALGORITHM_VERSION}:
        raise ValidationError("Unexpected cached solver version.")
    refit = bands["bands_refit_gp_on_toy"].astype(str).str.lower()
    if not refit.isin({"false", "0"}).all():
        raise ValidationError("At least one toy refits the GP.")
    return bands


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-provenance-json", type=Path, required=True)
    parser.add_argument("--reviewed-state-csv", type=Path, required=True)
    parser.add_argument("--reviewed-state-provenance-json", type=Path, required=True)
    parser.add_argument("--support-freeze-json", type=Path, required=True)
    parser.add_argument("--support-2016-low-mev", type=int, required=True)
    parser.add_argument("--support-2016-high-mev", type=int, required=True)
    parser.add_argument("--closure-report", type=Path, required=True)
    parser.add_argument("--bands-csv", type=Path, required=True)
    parser.add_argument("--bands-provenance-json", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    paths = {
        key: Path(value).expanduser().resolve()
        for key, value in {
            "config": args.config,
            "config_provenance": args.config_provenance_json,
            "reviewed_csv": args.reviewed_state_csv,
            "reviewed_provenance": args.reviewed_state_provenance_json,
            "support_freeze": args.support_freeze_json,
            "closure": args.closure_report,
            "bands": args.bands_csv,
            "bands_provenance": args.bands_provenance_json,
        }.items()
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")
    report_out = args.report_out.expanduser().resolve()
    if report_out.exists():
        raise SystemExit(f"Refusing to overwrite validation report: {report_out}")

    support = (
        int(args.support_2016_low_mev),
        int(args.support_2016_high_mev),
    )
    load_support_freeze(paths["support_freeze"], *support)
    validate_config_provenance(
        paths["config_provenance"],
        paths["config"],
        paths["support_freeze"],
        *support,
    )
    config = load_config(str(paths["config"]))
    validate_v4_geometry(config, *support)
    input_provenance = validate_input_hashes(config)
    validate_reviewed_provenance(
        paths["reviewed_provenance"],
        paths["reviewed_csv"],
        paths["support_freeze"],
        *support,
    )
    validate_reviewed_csv(paths["reviewed_csv"], support)
    validate_closure_report(
        paths["closure"],
        paths["config"],
        paths["config_provenance"],
        paths["reviewed_csv"],
        paths["reviewed_provenance"],
        paths["support_freeze"],
        *support,
    )
    bands = validate_bands(paths["bands"])

    provenance = json.loads(
        paths["bands_provenance"].read_text(encoding="utf-8")
    )
    exact_hash_fields = {
        "physics_config_sha256": sha256(paths["config"]),
        "physics_config_provenance_sha256": sha256(paths["config_provenance"]),
        "reviewed_csv_sha256": sha256(paths["reviewed_csv"]),
        "reviewed_provenance_sha256": sha256(paths["reviewed_provenance"]),
        "support_freeze_sha256": sha256(paths["support_freeze"]),
        "closure_report_sha256": sha256(paths["closure"]),
        "output_csv_sha256": sha256(paths["bands"]),
        "runner_sha256": sha256(HERE / "run_combined_bands_cached_fixed_reviewed.py"),
        "cached_solver_sha256": sha256(HERE / "cached_profile_solver.py"),
    }
    for key, expected in exact_hash_fields.items():
        if provenance.get(key) != expected:
            raise ValidationError(f"Band provenance field {key} does not match.")
    scalar_expectations = {
        "n_toys_per_mass": 100,
        "seed": SEED,
        "n_masses": 232,
        "n_mass_local_toy_limits": 23_200,
        "refit_gp_on_toy": False,
        "observed_gp_fit_mode": "fixed_reviewed_max_lml",
        "support_2016_low_MeV": support[0],
        "support_2016_high_MeV": support[1],
    }
    for key, expected in scalar_expectations.items():
        if provenance.get(key) != expected:
            raise ValidationError(
                f"Band provenance field {key} is {provenance.get(key)!r}, "
                f"expected {expected!r}."
            )
    if (
        provenance.get("runtime", {}).get("runtime_manifest_sha256")
        != RUNTIME_PROVENANCE["runtime_manifest_sha256"]
    ):
        raise ValidationError("Band provenance used another runtime snapshot.")
    if provenance.get("runtime_import_origins") != RUNTIME_IMPORT_ORIGINS:
        raise ValidationError("Band provenance has different import origins.")

    report = {
        "schema_version": 1,
        "status": "pass",
        "support_2016_low_MeV": support[0],
        "support_2016_high_MeV": support[1],
        "mass_rows": int(len(bands)),
        "mass_local_toy_limits": int(bands["n_toys_requested"].sum()),
        "stored_mass_local_toy_limits": 23_200,
        "active_set_rows": EXPECTED_ACTIVE_SETS,
        "inputs": input_provenance,
        "artifacts": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in paths.items()
        },
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "runtime": RUNTIME_PROVENANCE,
        "runtime_import_origins": RUNTIME_IMPORT_ORIGINS,
        "claim_boundary": (
            "The 100-toy quantiles are conditional fixed-GP expected-limit "
            "bands with an asymptotic inner CLs construction. This validation "
            "does not establish direct coverage or global significance."
        ),
    }
    atomic_text(report_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS: 232 masses and 23,200 finite mass-local toy limits")
    print(f"Wrote {report_out}")


if __name__ == "__main__":
    main()
