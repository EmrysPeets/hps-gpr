#!/usr/bin/env python3
"""Validate and compare the 2016 length-scale upper-bound grid.

The factor-8 reference is *only* the reviewed 2016 subset of the frozen v4
wide-support observed campaign.  Factors 10, 12, 15, and 20 are required;
factor 25 is included only when its complete raw result exists.  For each
higher factor, the script reviews the full raw scan together with any
``repairs/mMMM_attempt_NN/results_single.csv`` candidates and selects the
highest-LML *actual fit row* at each mass.  It never interpolates.

This is an observed/asymptotic diagnostic.  It does not run pseudoexperiments,
construct expected-limit bands, calculate toy-calibrated p-values, or select a
length-scale factor from favorable observed limits or p-values.  A settings
decision requires separate, predeclared closure/coverage evidence.

Generated products remain below this study directory:

* ``derived/*.csv``: source/config/input/LML audits and pointwise comparisons;
* ``derived/run_summary.json`` and ``derived/product_manifest.json``;
* ``plots/*.pdf`` and ``plots/*.png``: three uncluttered comparison figures.

Run from the repository root only after all factor scans have completed:

    python3 \
      study_results/v4p1_2016_ls_upper_optimization_20260804/\
postprocess_2016_ls_upper_grid.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

# Keep Matplotlib cache writes out of the repository and the user's home.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hps_gpr_v4p1_2016_ls_postprocess"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DERIVED = HERE / "derived"
PLOTS = HERE / "plots"

CONFIG_DIR = (
    REPO / "study_configs" / "v4p1_2016_ls_upper_optimization_20260804"
)
CONFIG_MANIFEST = CONFIG_DIR / "config_manifest.json"
V4_CAMPAIGN = (
    REPO
    / "study_results"
    / "v4_wide_support_2015full_2016full_2021_10pct_20260803"
)
V4_REVIEW = V4_CAMPAIGN / "derived" / "observed_gp_states_reviewed.csv"
V4_CONFIG = (
    REPO
    / "study_configs"
    / "v4_wide_support_2015full_2016full_2021_10pct_20260803"
    / "config_obsUL90_combined_wide_support_v4_observed_only.yaml"
)
V4_VALIDATION_REPORTS = tuple(
    V4_CAMPAIGN / f"observed_attempt_{index:02d}" / "validation_report.json"
    for index in (1, 2, 3)
)

REQUIRED_FACTORS = (8, 10, 12, 15, 20)
OPTIONAL_FACTORS = (25,)
DECLARED_FACTORS = REQUIRED_FACTORS + OPTIONAL_FACTORS
REFERENCE_FACTOR = 8
EXPECTED_MASS_MEV = np.arange(39, 181, dtype=int)
EXPECTED_MASS_GEV = EXPECTED_MASS_MEV.astype(float) / 1000.0
EXPECTED_ROWS = len(EXPECTED_MASS_MEV)
BOUNDARY_THRESHOLD = 0.999
LML_TOLERANCE = 1.0e-4
REPAIR_REPRODUCTION_TOLERANCE = 2.0e-5
STATIC_NUMERIC_ATOL = 1.0e-12
RATIO_RTOL = 2.0e-10

COLORS = {
    8: "#111111",
    10: "#0072B2",
    12: "#D55E00",
    15: "#CC79A7",
    20: "#009E73",
    25: "#E69F00",
}
LINESTYLES = {
    8: "-",
    10: (0, (5, 2)),
    12: (0, (2, 1)),
    15: (0, (6, 2, 1, 2)),
    20: (0, (1, 1)),
    25: (0, (8, 2)),
}

REQUIRED_COLUMNS = {
    "dataset",
    "mass_GeV",
    "sigma_val",
    "blind_lo",
    "blind_hi",
    "A_up",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
    "A_hat",
    "sigma_A",
    "extract_success",
    "cls_statistic",
    "cls_calibration",
    "signal_model",
    "global_method",
    "visibility",
    "ls_lo",
    "ls_hi",
    "ls_init",
    "ls_opt",
    "sigma_x",
    "const_opt",
    "lml",
    "n_train",
    "ls_lo_over_sigma_x",
    "ls_hi_over_sigma_x",
    "ls_opt_over_sigma_x",
}

REQUIRED_FINITE_COLUMNS = (
    "mass_GeV",
    "sigma_val",
    "blind_lo",
    "blind_hi",
    "A_up",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
    "A_hat",
    "sigma_A",
    "ls_lo",
    "ls_hi",
    "ls_init",
    "ls_opt",
    "sigma_x",
    "const_opt",
    "lml",
    "n_train",
    "ls_lo_over_sigma_x",
    "ls_hi_over_sigma_x",
    "ls_opt_over_sigma_x",
)

# Fields fixed by the dataset, card, and bin geometry.  The upper bound,
# upper-bound-dependent initializer, optimizer result, and inference outputs
# are deliberately absent.
SAME_INPUT_NUMERIC_COLUMNS = (
    "mass_GeV",
    "sigma_val",
    "blind_lo",
    "blind_hi",
    "ls_lo",
    "sigma_x",
    "n_train",
    "ls_lo_over_sigma_x",
)
SAME_INPUT_TEXT_COLUMNS = (
    "dataset",
    "cls_statistic",
    "cls_calibration",
    "signal_model",
    "global_method",
    "visibility",
)
SAME_INPUT_BOOLEAN_COLUMNS: tuple[str, ...] = ()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    return path.resolve()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(child) for child in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, label))
        return result
    return {prefix: value}


def parsed_differences(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    left = flatten(reference)
    right = flatten(candidate)
    return sorted(
        key
        for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    )


def normalize_boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        if bool(series.isna().any()):
            raise RuntimeError(f"{label} contains a missing boolean")
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    invalid = normalized.isna() | ~normalized.isin(mapping)
    if bool(invalid.any()):
        raise RuntimeError(
            f"{label} contains invalid booleans: "
            f"{series.loc[invalid].head(5).tolist()}"
        )
    return normalized.map(mapping).astype(bool)


def card_dataset_value(card: dict[str, Any], key: str, dataset: str) -> Any:
    values = card.get(key)
    if not isinstance(values, dict):
        raise RuntimeError(f"config field {key!r} is not a mapping")
    if dataset in values:
        return values[dataset]
    if int(dataset) in values:
        return values[int(dataset)]
    raise RuntimeError(f"config field {key!r} has no dataset {dataset!r}")


def require_close(
    actual: float,
    expected: float,
    label: str,
    *,
    atol: float = STATIC_NUMERIC_ATOL,
) -> None:
    if not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=atol
    ):
        raise RuntimeError(f"{label} is {actual!r}; expected {expected!r}")


def validate_no_toy_or_band_switches(card: dict[str, Any], factor: int) -> None:
    false_fields = (
        "make_ul_bands",
        "do_combined_bands",
        "make_eps2_bands",
        "funcform_closure_enable",
        "inject_signal",
    )
    nonzero_fields = ("cls_num_toys", "ul_bands_toys", "combined_bands_n_toys")
    active = [field for field in false_fields if bool(card.get(field))]
    nonzero = [
        field for field in nonzero_fields if int(card.get(field, 0)) != 0
    ]
    if active or nonzero:
        raise RuntimeError(
            f"factor {factor}: toy/band switches are active: {active + nonzero}"
        )
    if str(card.get("cls_mode", "")).strip().lower() != "asymptotic":
        raise RuntimeError(f"factor {factor}: cls_mode is not asymptotic")
    if bool(card.get("do_combined")):
        raise RuntimeError(f"factor {factor}: diagnostic card is combined")
    enabled = {
        dataset: bool(card.get(f"enable_{dataset}"))
        for dataset in ("2015", "2016", "2021")
    }
    if enabled != {"2015": False, "2016": True, "2021": False}:
        raise RuntimeError(
            f"factor {factor}: active datasets are {enabled}, expected 2016 only"
        )


def audit_configs() -> tuple[dict[int, Path], pd.DataFrame, dict[str, Any]]:
    if not CONFIG_MANIFEST.is_file():
        raise RuntimeError(f"missing config manifest: {CONFIG_MANIFEST}")
    if not V4_CONFIG.is_file():
        raise RuntimeError(f"missing reviewed v4 config: {V4_CONFIG}")

    payload = json.loads(CONFIG_MANIFEST.read_text(encoding="utf-8"))
    declared_factors = tuple(int(value) for value in payload.get("factors", []))
    if declared_factors != DECLARED_FACTORS:
        raise RuntimeError(
            "config manifest factors are "
            f"{declared_factors}; expected {DECLARED_FACTORS}"
        )
    source_path = resolve_recorded_path(str(payload.get("source_config", "")))
    if source_path != V4_CONFIG.resolve():
        raise RuntimeError(
            f"config manifest source is {source_path}; expected {V4_CONFIG}"
        )
    declared_source_hash = str(payload.get("source_config_sha256", ""))
    actual_source_hash = sha256(V4_CONFIG)
    if declared_source_hash != actual_source_hash:
        raise RuntimeError("reviewed v4 source-config checksum has changed")

    cards = payload.get("cards", [])
    by_factor = {int(item["upper_factor"]): item for item in cards}
    if (
        set(by_factor) != set(DECLARED_FACTORS)
        or len(cards) != len(DECLARED_FACTORS)
    ):
        raise RuntimeError("config manifest does not contain one card per factor")

    v4_card = yaml.safe_load(V4_CONFIG.read_text(encoding="utf-8"))
    config_paths: dict[int, Path] = {}
    parsed_cards: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for factor in DECLARED_FACTORS:
        entry = by_factor[factor]
        path = resolve_recorded_path(str(entry["config"]))
        if not path.is_file():
            raise RuntimeError(f"factor {factor}: missing config {path}")
        actual_hash = sha256(path)
        if actual_hash != str(entry.get("config_sha256", "")):
            raise RuntimeError(f"factor {factor}: config checksum mismatch")
        card = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate_no_toy_or_band_switches(card, factor)
        realized = float(
            card_dataset_value(
                card, "kernel_ls_res_upper_factor_by_dataset", "2016"
            )
        )
        require_close(
            realized,
            factor,
            f"factor {factor} configured upper factor",
            atol=0.0,
        )
        require_close(
            float(card_dataset_value(
                card, "kernel_ls_res_lower_factor_by_dataset", "2016"
            )),
            0.9,
            f"factor {factor} configured lower factor",
        )
        if list(map(float, card.get("range_2016", []))) != [0.039, 0.18]:
            raise RuntimeError(f"factor {factor}: unexpected 2016 search range")
        if list(map(float, card.get("data_range_2016", []))) != [0.03, 0.21]:
            raise RuntimeError(f"factor {factor}: unexpected 2016 support range")
        require_close(
            float(card.get("mass_step_gev")),
            0.001,
            f"factor {factor} mass step",
        )
        require_close(
            float(card.get("blind_nsigma")),
            2.25,
            f"factor {factor} blind window",
        )
        require_close(
            float(card.get("gp_train_exclude_nsigma")),
            2.25,
            f"factor {factor} training exclusion",
        )
        require_close(
            float(card.get("eps2_density_nsigma")),
            1.64,
            f"factor {factor} density window",
        )
        require_close(
            float(card.get("cls_alpha")),
            0.1,
            f"factor {factor} CLs alpha",
        )
        expected_output = (HERE / f"k{factor:02d}" / "raw_attempt_01").resolve()
        configured_output = Path(str(card.get("output_dir", ""))).resolve()
        if configured_output != expected_output:
            raise RuntimeError(
                f"factor {factor}: output_dir is {configured_output}; "
                f"expected {expected_output}"
            )
        config_paths[factor] = path
        parsed_cards[factor] = card

    source_to_k8 = parsed_differences(v4_card, parsed_cards[REFERENCE_FACTOR])
    expected_source_to_k8 = [
        "do_combined",
        "enable_2015",
        "enable_2021",
        "output_dir",
    ]
    if source_to_k8 != expected_source_to_k8:
        raise RuntimeError(
            "factor 8 diagnostic card differs unexpectedly from reviewed v4: "
            f"{source_to_k8}"
        )

    reference_card = parsed_cards[REFERENCE_FACTOR]
    for factor in DECLARED_FACTORS:
        differences = parsed_differences(reference_card, parsed_cards[factor])
        expected = (
            []
            if factor == REFERENCE_FACTOR
            else [
                "kernel_ls_res_upper_factor_by_dataset.2016",
                "output_dir",
            ]
        )
        if differences != expected:
            raise RuntimeError(
                f"factor {factor}: parsed differences from factor 8 are "
                f"{differences}; expected {expected}"
            )
        declared_from_v4 = sorted(
            str(value)
            for value in by_factor[factor].get(
                "parsed_differences_from_v4", []
            )
        )
        computed_from_v4 = parsed_differences(v4_card, parsed_cards[factor])
        if declared_from_v4 != computed_from_v4:
            raise RuntimeError(
                f"factor {factor}: manifest/config difference audit disagrees"
            )
        records.append(
            {
                "upper_factor": factor,
                "config": repo_path(config_paths[factor]),
                "config_sha256": sha256(config_paths[factor]),
                "configured_upper_factor_2016": float(
                    card_dataset_value(
                        parsed_cards[factor],
                        "kernel_ls_res_upper_factor_by_dataset",
                        "2016",
                    )
                ),
                "differences_from_v4": "|".join(computed_from_v4),
                "differences_from_factor8": "|".join(
                    parsed_differences(reference_card, parsed_cards[factor])
                ),
                "toys_or_bands_enabled": False,
            }
        )
    return config_paths, pd.DataFrame(records), payload


def result_path(factor: int) -> Path:
    if factor == REFERENCE_FACTOR:
        return V4_REVIEW
    return HERE / f"k{factor:02d}" / "raw_attempt_01" / "results_single.csv"


def discover_complete_sources() -> tuple[tuple[int, ...], dict[int, Path]]:
    required_paths = {
        factor: result_path(factor) for factor in REQUIRED_FACTORS
    }
    missing = [
        factor for factor, path in required_paths.items() if not path.is_file()
    ]
    if missing:
        details: list[str] = []
        for factor in missing:
            directory = HERE / f"k{factor:02d}" / "raw_attempt_01"
            present = (
                sorted(item.name for item in directory.iterdir())
                if directory.is_dir()
                else []
            )
            details.append(f"k{factor:02d}: present={present}")
        raise RuntimeError(
            "factor grid is incomplete; do not postprocess until every scan "
            f"has written results_single.csv. Missing factors={missing}; "
            + "; ".join(details)
        )
    active = list(REQUIRED_FACTORS)
    paths = dict(required_paths)
    for factor in OPTIONAL_FACTORS:
        path = result_path(factor)
        if path.is_file():
            active.append(factor)
            paths[factor] = path
    return tuple(active), paths


def validate_review_provenance(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "review_status",
        "branch_multiplicity",
        "selected_source",
        "selected_source_sha256",
        "row_source",
        "interpolated",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"factor 8 review lacks provenance columns: {missing}")
    statuses = set(frame["review_status"].astype(str))
    if statuses != {"resolved_reproduced_max_lml"}:
        raise RuntimeError(f"factor 8 review has unresolved statuses: {statuses}")
    if bool((frame["branch_multiplicity"].to_numpy(float) < 2.0).any()):
        raise RuntimeError("factor 8 review contains an unreproduced state")
    interpolated = normalize_boolean(frame["interpolated"], "factor8.interpolated")
    if bool(interpolated.any()):
        raise RuntimeError("factor 8 review contains interpolated rows")
    if not frame["row_source"].astype(str).str.startswith(
        "unchanged_card_max_lml:"
    ).all():
        raise RuntimeError("factor 8 review has an unexpected row_source")

    checked: dict[str, str] = {}
    for source, group in frame.groupby(frame["selected_source"].astype(str)):
        path = resolve_recorded_path(str(source))
        if not path.is_file():
            raise RuntimeError(f"factor 8 selected source is missing: {path}")
        actual = sha256(path)
        declared = set(group["selected_source_sha256"].astype(str))
        if declared != {actual}:
            raise RuntimeError(
                f"factor 8 selected-source checksum mismatch for {path}"
            )
        checked[repo_path(path)] = actual
    return {
        "review_status": "resolved_reproduced_max_lml",
        "minimum_branch_multiplicity": int(
            frame["branch_multiplicity"].min()
        ),
        "interpolation_used": False,
        "selected_source_checksums": checked,
    }


def validate_frame(
    raw: pd.DataFrame,
    factor: int,
    source: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = raw.copy()
    frame["dataset"] = frame["dataset"].astype(str).str.strip()
    if factor == REFERENCE_FACTOR:
        frame = frame.loc[frame["dataset"] == "2016"].copy()

    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise RuntimeError(f"factor {factor}: missing columns {missing}")
    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"factor {factor}: found {len(frame)} rows; expected {EXPECTED_ROWS}"
        )
    if set(frame["dataset"]) != {"2016"}:
        raise RuntimeError(f"factor {factor}: results are not 2016-only")

    frame["extract_success"] = normalize_boolean(
        frame["extract_success"], f"factor{factor}.extract_success"
    )
    if not bool(frame["extract_success"].all()):
        failed = frame.loc[~frame["extract_success"], "mass_GeV"].head().tolist()
        raise RuntimeError(
            f"factor {factor}: failed extractions at masses {failed}"
        )
    finite = frame.loc[:, REQUIRED_FINITE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    bad = [
        column
        for column in finite.columns
        if not np.isfinite(finite[column].to_numpy(float)).all()
    ]
    if bad:
        raise RuntimeError(
            f"factor {factor}: non-finite or nonnumeric values in {bad}"
        )

    mass_gev = frame["mass_GeV"].to_numpy(float)
    mass_mev = np.rint(1000.0 * mass_gev).astype(int)
    if not np.allclose(
        mass_gev, mass_mev.astype(float) / 1000.0, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError(f"factor {factor}: contains an off-grid mass")
    frame["mass_MeV"] = mass_mev
    if bool(frame["mass_MeV"].duplicated().any()):
        raise RuntimeError(f"factor {factor}: duplicate mass hypotheses")
    frame = frame.sort_values("mass_MeV").reset_index(drop=True)
    if not np.array_equal(frame["mass_MeV"].to_numpy(int), EXPECTED_MASS_MEV):
        missing_mass = sorted(
            set(EXPECTED_MASS_MEV).difference(frame["mass_MeV"].astype(int))
        )
        extra_mass = sorted(
            set(frame["mass_MeV"].astype(int)).difference(EXPECTED_MASS_MEV)
        )
        raise RuntimeError(
            f"factor {factor}: grid mismatch; "
            f"missing={missing_mass[:10]}, extra={extra_mass[:10]}"
        )

    expected_text = {
        "cls_statistic": {"tilde_q_mu"},
        "cls_calibration": {"asymptotic"},
        "visibility": {"observed"},
    }
    for column, expected in expected_text.items():
        actual = set(frame[column].astype(str).str.strip())
        if actual != expected:
            raise RuntimeError(
                f"factor {factor}: {column} values are {sorted(actual)}"
            )
    if bool((frame["A_up"].to_numpy(float) <= 0.0).any()):
        raise RuntimeError(f"factor {factor}: non-positive A_up")
    if bool((frame["eps2_up"].to_numpy(float) <= 0.0).any()):
        raise RuntimeError(f"factor {factor}: non-positive eps2_up")
    if bool((frame["sigma_A"].to_numpy(float) <= 0.0).any()):
        raise RuntimeError(f"factor {factor}: non-positive sigma_A")
    p0 = frame["p0_analytic"].to_numpy(float)
    if bool(((p0 < 0.0) | (p0 > 0.5 + 1.0e-12)).any()):
        raise RuntimeError(f"factor {factor}: p0_analytic is outside [0, 0.5]")
    if bool((frame["Z_analytic"].to_numpy(float) < -1.0e-12).any()):
        raise RuntimeError(f"factor {factor}: negative one-sided Z values")

    n_train = frame["n_train"].to_numpy(float)
    if not np.allclose(n_train, np.rint(n_train), rtol=0.0, atol=0.0):
        raise RuntimeError(f"factor {factor}: n_train is not integer-valued")
    if bool((n_train <= 0).any()):
        raise RuntimeError(f"factor {factor}: non-positive n_train")

    ls_lo = frame["ls_lo"].to_numpy(float)
    ls_hi = frame["ls_hi"].to_numpy(float)
    ls_opt = frame["ls_opt"].to_numpy(float)
    sigma_x = frame["sigma_x"].to_numpy(float)
    if bool(((ls_lo <= 0.0) | (ls_hi <= ls_lo) | (sigma_x <= 0.0)).any()):
        raise RuntimeError(f"factor {factor}: invalid length-scale domain")
    if bool((ls_opt < ls_lo * (1.0 - 1.0e-10) - 1.0e-14).any()):
        raise RuntimeError(f"factor {factor}: ls_opt is below its lower bound")
    if bool((ls_opt > ls_hi * (1.0 + 1.0e-10) + 1.0e-14).any()):
        raise RuntimeError(f"factor {factor}: ls_opt is above its upper bound")
    realized = ls_hi / sigma_x
    if not np.allclose(realized, factor, rtol=0.0, atol=1.0e-8):
        sample = realized[~np.isclose(
            realized, factor, rtol=0.0, atol=1.0e-8
        )][:5]
        raise RuntimeError(
            f"factor {factor}: realized ls_hi/sigma_x does not match card; "
            f"sample={sample.tolist()}"
        )
    recorded = frame["ls_hi_over_sigma_x"].to_numpy(float)
    if not np.allclose(recorded, realized, rtol=0.0, atol=1.0e-10):
        raise RuntimeError(
            f"factor {factor}: recorded ls_hi_over_sigma_x is inconsistent"
        )
    recorded_opt = frame["ls_opt_over_sigma_x"].to_numpy(float)
    if not np.allclose(
        recorded_opt, ls_opt / sigma_x, rtol=0.0, atol=1.0e-10
    ):
        raise RuntimeError(
            f"factor {factor}: recorded ls_opt_over_sigma_x is inconsistent"
        )

    frame["upper_factor"] = int(factor)
    frame["ls_opt_over_ls_hi"] = ls_opt / ls_hi
    frame["at_upper_boundary"] = (
        frame["ls_opt_over_ls_hi"].to_numpy(float) >= BOUNDARY_THRESHOLD
    )
    frame["result_source"] = repo_path(source)
    frame["result_source_sha256"] = sha256(source)

    provenance: dict[str, Any] = {}
    if factor == REFERENCE_FACTOR:
        provenance = validate_review_provenance(frame)
    metadata = {
        "factor": factor,
        "rows": int(len(frame)),
        "result_source": repo_path(source),
        "result_source_sha256": sha256(source),
        "source_kind": (
            "reviewed_v4_unchanged_card_max_lml"
            if factor == REFERENCE_FACTOR
            else "raw_attempt_01"
        ),
        "mass_min_MeV": int(frame["mass_MeV"].min()),
        "mass_max_MeV": int(frame["mass_MeV"].max()),
        "extract_success_rows": int(frame["extract_success"].sum()),
        "finite_rows": int(len(frame)),
        "realized_upper_factor_min": float(realized.min()),
        "realized_upper_factor_max": float(realized.max()),
        "boundary_rows": int(frame["at_upper_boundary"].sum()),
        "review_provenance": provenance,
    }
    return frame, metadata


def repair_result_paths(factor: int) -> list[Path]:
    if factor == REFERENCE_FACTOR:
        return []
    return sorted(
        (HERE / f"k{factor:02d}" / "repairs").glob(
            "m*_attempt_*/results_single.csv"
        )
    )


def compare_candidate_static_fields(
    candidate: pd.Series,
    raw_row: pd.Series,
    *,
    factor: int,
    mass_mev: int,
    source: Path,
) -> None:
    for column in SAME_INPUT_NUMERIC_COLUMNS:
        if not math.isclose(
            float(candidate[column]),
            float(raw_row[column]),
            rel_tol=0.0,
            abs_tol=STATIC_NUMERIC_ATOL,
        ):
            raise RuntimeError(
                f"factor {factor}, {mass_mev} MeV repair {source}: "
                f"static field {column} differs from raw"
            )
    for column in SAME_INPUT_TEXT_COLUMNS:
        if str(candidate[column]) != str(raw_row[column]):
            raise RuntimeError(
                f"factor {factor}, {mass_mev} MeV repair {source}: "
                f"static field {column} differs from raw"
            )


def validated_repair_row(
    raw_input: pd.DataFrame,
    raw_frame: pd.DataFrame,
    factor: int,
    path: Path,
) -> pd.Series:
    candidate_input = pd.read_csv(path)
    if len(candidate_input) != 1:
        raise RuntimeError(
            f"factor {factor}: repair {path} has {len(candidate_input)} rows"
        )
    missing = sorted(REQUIRED_COLUMNS.difference(candidate_input.columns))
    if missing:
        raise RuntimeError(
            f"factor {factor}: repair {path} lacks columns {missing}"
        )
    mass_gev = float(candidate_input.iloc[0]["mass_GeV"])
    mass_mev = int(round(1000.0 * mass_gev))
    if not math.isclose(
        mass_gev, mass_mev / 1000.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise RuntimeError(f"factor {factor}: repair {path} is off-grid")
    expected_prefix = f"m{mass_mev:03d}_attempt_"
    if not path.parent.name.startswith(expected_prefix):
        raise RuntimeError(
            f"factor {factor}: repair path {path.parent.name} disagrees "
            f"with its {mass_mev} MeV row"
        )
    match = np.isclose(
        raw_input["mass_GeV"].to_numpy(float),
        mass_gev,
        rtol=0.0,
        atol=1.0e-12,
    )
    if int(match.sum()) != 1:
        raise RuntimeError(
            f"factor {factor}: repair mass {mass_mev} is absent/duplicated in raw"
        )

    # Reuse the exact full-grid validator by replacing only this actual row.
    trial = raw_input.copy()
    index = trial.index[match][0]
    for column in candidate_input.columns:
        trial.loc[index, column] = candidate_input.iloc[0][column]
    validated, _ = validate_frame(trial, factor, result_path(factor))
    candidate = validated.loc[validated["mass_MeV"] == mass_mev].iloc[0].copy()
    raw_row = raw_frame.loc[raw_frame["mass_MeV"] == mass_mev].iloc[0]
    compare_candidate_static_fields(
        candidate,
        raw_row,
        factor=factor,
        mass_mev=mass_mev,
        source=path,
    )
    candidate["row_source"] = repo_path(path)
    candidate["row_source_sha256"] = sha256(path)
    candidate["row_source_kind"] = "repair"
    candidate["repair_attempt"] = path.parent.name.rsplit("_", 1)[-1]
    return candidate


def stitch_factor_candidates(
    factor: int,
    raw_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, list[Path]]:
    raw_input = pd.read_csv(raw_path)
    raw_frame, metadata = validate_frame(raw_input, factor, raw_path)
    raw_frame["row_source"] = repo_path(raw_path)
    raw_frame["row_source_sha256"] = sha256(raw_path)
    raw_frame["row_source_kind"] = "raw"
    raw_frame["repair_attempt"] = ""

    repairs = repair_result_paths(factor)
    repair_rows: dict[int, list[pd.Series]] = {}
    for path in repairs:
        row = validated_repair_row(
            raw_input, raw_frame, factor, path
        )
        repair_rows.setdefault(int(row["mass_MeV"]), []).append(row)

    selected_rows: list[pd.Series] = []
    ledger_records: list[dict[str, Any]] = []
    selected_repair_rows = 0
    reproduced_selected_repairs = 0
    pending_selected_repairs = 0
    for mass_mev in EXPECTED_MASS_MEV:
        raw_row = raw_frame.loc[
            raw_frame["mass_MeV"] == mass_mev
        ].iloc[0].copy()
        candidates = [raw_row, *repair_rows.get(int(mass_mev), [])]
        selected_index = int(
            np.argmax([float(row["lml"]) for row in candidates])
        )
        selected = candidates[selected_index].copy()
        selected_lml = float(selected["lml"])
        within = [
            abs(float(row["lml"]) - selected_lml)
            <= REPAIR_REPRODUCTION_TOLERANCE
            for row in candidates
        ]
        reproducing_other = [
            row
            for index, (row, agrees) in enumerate(zip(candidates, within))
            if index != selected_index and agrees
        ]
        selected_is_repair = str(selected["row_source_kind"]) == "repair"
        repair_candidates = [
            row for row in candidates if str(row["row_source_kind"]) == "repair"
        ]
        reproduction_pending = False
        if selected_is_repair:
            selected_repair_rows += 1
            if reproducing_other:
                reproduced_selected_repairs += 1
            elif len(repair_candidates) >= 2:
                raise RuntimeError(
                    f"factor {factor}, {mass_mev} MeV: selected repair has "
                    "available unchanged-card repeats but none reproduces its "
                    f"LML within {REPAIR_REPRODUCTION_TOLERANCE:g}"
                )
            else:
                reproduction_pending = True
                pending_selected_repairs += 1

        if selected_is_repair and reproducing_other:
            review_status = "repair_selected_reproduced_max_lml"
        elif selected_is_repair:
            review_status = "repair_selected_reproduction_pending"
        elif len(candidates) > 1:
            review_status = "raw_selected_after_candidate_review"
        else:
            review_status = "raw_scan_row"
        reproducing_sources = [
            str(row["row_source"])
            for row, agrees in zip(candidates, within)
            if agrees
        ]
        selected["selected_source"] = str(selected["row_source"])
        selected["selected_source_sha256"] = str(
            selected["row_source_sha256"]
        )
        selected["optimizer_repair_applied"] = bool(selected_is_repair)
        selected["review_status"] = review_status
        selected["candidate_count"] = len(candidates)
        selected["repair_candidate_count"] = len(repair_candidates)
        selected["branch_multiplicity"] = int(sum(within))
        selected["selected_repair_reproduced"] = bool(
            selected_is_repair and bool(reproducing_other)
        )
        selected["repair_reproduction_pending"] = bool(reproduction_pending)
        selected["reproducing_sources"] = "|".join(reproducing_sources)
        selected["raw_lml"] = float(raw_row["lml"])
        selected["delta_lml_selected_minus_raw"] = (
            selected_lml - float(raw_row["lml"])
        )
        selected["interpolated"] = False
        selected_rows.append(selected)

        for index, (candidate, agrees) in enumerate(zip(candidates, within)):
            ledger_records.append(
                {
                    "upper_factor": factor,
                    "mass_MeV": int(mass_mev),
                    "mass_GeV": float(mass_mev / 1000.0),
                    "candidate_source": str(candidate["row_source"]),
                    "candidate_source_sha256": str(
                        candidate["row_source_sha256"]
                    ),
                    "candidate_kind": str(candidate["row_source_kind"]),
                    "repair_attempt": str(candidate["repair_attempt"]),
                    "lml": float(candidate["lml"]),
                    "selected_lml": selected_lml,
                    "delta_lml_from_selected": float(
                        candidate["lml"] - selected_lml
                    ),
                    "within_reproduction_tolerance": bool(agrees),
                    "is_selected_maximum": bool(index == selected_index),
                    "selected_source": str(selected["selected_source"]),
                    "selected_source_kind": str(
                        selected["row_source_kind"]
                    ),
                    "selected_repair_reproduced": bool(
                        selected["selected_repair_reproduced"]
                    ),
                    "repair_reproduction_pending": bool(
                        selected["repair_reproduction_pending"]
                    ),
                    "review_status": review_status,
                    "selection_rule": "highest_lml_actual_fit_row",
                    "interpolated": False,
                }
            )

    stitched = pd.DataFrame(selected_rows).sort_values(
        "mass_MeV"
    ).reset_index(drop=True)
    # Validate the selected inference/optimizer columns once more as a grid.
    validated, validated_metadata = validate_frame(
        stitched.loc[:, raw_input.columns], factor, raw_path
    )
    provenance_columns = [
        column
        for column in stitched.columns
        if column not in validated.columns
        or column
        in {
            "row_source",
            "row_source_sha256",
            "row_source_kind",
            "repair_attempt",
        }
    ]
    for column in provenance_columns:
        validated[column] = stitched[column].to_numpy()
    metadata.update(validated_metadata)
    metadata.update(
        {
            "repair_candidate_files": len(repairs),
            "masses_with_repair_candidates": len(repair_rows),
            "selected_repair_rows": selected_repair_rows,
            "selected_repair_rows_reproduced": reproduced_selected_repairs,
            "selected_repair_rows_reproduction_pending": pending_selected_repairs,
            "row_selection_rule": "highest_lml_actual_fit_row",
            "interpolation_used": False,
        }
    )
    return validated, metadata, pd.DataFrame(ledger_records), repairs


def validation_report_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing validation report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "2016" not in payload or not isinstance(payload["2016"], dict):
        raise RuntimeError(f"{path} has no 2016 validation payload")
    report = payload["2016"]
    if not bool(report.get("ok")):
        raise RuntimeError(f"{path} does not validate the 2016 input")
    if report.get("messages") not in ([], None):
        raise RuntimeError(f"{path} contains validation messages")
    return report


def audit_validation_reports(
    factors: tuple[int, ...],
    repair_paths: dict[int, list[Path]],
) -> tuple[dict[int, list[Path]], dict[str, Any]]:
    reports: dict[int, list[Path]] = {
        REFERENCE_FACTOR: list(V4_VALIDATION_REPORTS)
    }
    for factor in factors:
        if factor == REFERENCE_FACTOR:
            continue
        candidate_results = [
            result_path(factor),
            *repair_paths.get(factor, []),
        ]
        reports[factor] = sorted(
            {path.parent / "validation_report.json" for path in candidate_results}
        )

    canonical: dict[int, list[str]] = {}
    details: dict[str, Any] = {}
    for factor, paths in reports.items():
        payloads = [validation_report_payload(path) for path in paths]
        canonical[factor] = [
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in payloads
        ]
        if len(set(canonical[factor])) != 1:
            raise RuntimeError(
                f"factor {factor}: unchanged-card validation reports differ"
            )
        details[str(factor)] = [
            {
                "path": repo_path(path),
                "sha256": sha256(path),
                "payload_2016": payload,
            }
            for path, payload in zip(paths, payloads)
        ]

    reference = canonical[REFERENCE_FACTOR][0]
    mismatch = [
        factor
        for factor in factors
        if canonical[factor][0] != reference
    ]
    if mismatch:
        raise RuntimeError(
            f"input validation payload differs from v4 for factors {mismatch}"
        )
    return reports, {
        "same_2016_validation_payload_all_factors": True,
        "reports": details,
    }


def compare_same_inputs(
    frames: dict[int, pd.DataFrame],
    factors: tuple[int, ...],
) -> pd.DataFrame:
    reference = frames[REFERENCE_FACTOR]
    records: list[dict[str, Any]] = []
    for factor in factors:
        candidate = frames[factor]
        numeric_max_delta = 0.0
        for column in SAME_INPUT_NUMERIC_COLUMNS:
            left = reference[column].to_numpy(float)
            right = candidate[column].to_numpy(float)
            delta = float(np.max(np.abs(right - left)))
            numeric_max_delta = max(numeric_max_delta, delta)
            if not np.allclose(
                right, left, rtol=0.0, atol=STATIC_NUMERIC_ATOL
            ):
                index = int(np.argmax(np.abs(right - left)))
                raise RuntimeError(
                    f"factor {factor}: same-input check fails for {column} at "
                    f"{int(candidate.iloc[index]['mass_MeV'])} MeV "
                    f"(delta={right[index] - left[index]:.12g})"
                )
        for column in SAME_INPUT_TEXT_COLUMNS:
            if not np.array_equal(
                candidate[column].astype(str).to_numpy(),
                reference[column].astype(str).to_numpy(),
            ):
                raise RuntimeError(
                    f"factor {factor}: same-input text field {column} differs"
                )
        for column in SAME_INPUT_BOOLEAN_COLUMNS:
            if not np.array_equal(
                candidate[column].to_numpy(bool),
                reference[column].to_numpy(bool),
            ):
                raise RuntimeError(
                    f"factor {factor}: same-input boolean {column} differs"
                )
        records.append(
            {
                "upper_factor": factor,
                "reference_factor": REFERENCE_FACTOR,
                "same_numeric_input_geometry": True,
                "same_text_semantics": True,
                "same_boolean_geometry": True,
                "numeric_columns_checked": len(SAME_INPUT_NUMERIC_COLUMNS),
                "text_columns_checked": len(SAME_INPUT_TEXT_COLUMNS),
                "boolean_columns_checked": len(SAME_INPUT_BOOLEAN_COLUMNS),
                "maximum_absolute_numeric_delta": numeric_max_delta,
                "numeric_tolerance": STATIC_NUMERIC_ATOL,
            }
        )
    return pd.DataFrame(records)


def audit_nested_lml(
    frames: dict[int, pd.DataFrame],
    factors: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pointwise_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for lower, upper in zip(factors[:-1], factors[1:]):
        lower_frame = frames[lower]
        upper_frame = frames[upper]
        delta = (
            upper_frame["lml"].to_numpy(float)
            - lower_frame["lml"].to_numpy(float)
        )
        violation = delta < -LML_TOLERANCE
        for mass_mev, value, failed in zip(
            EXPECTED_MASS_MEV, delta, violation
        ):
            pointwise_records.append(
                {
                    "lower_factor": lower,
                    "upper_factor": upper,
                    "mass_MeV": int(mass_mev),
                    "mass_GeV": float(mass_mev / 1000.0),
                    "delta_lml_upper_minus_lower": float(value),
                    "tolerance": LML_TOLERANCE,
                    "nested_order_violation": bool(failed),
                }
            )
        summary_records.append(
            {
                "lower_factor": lower,
                "upper_factor": upper,
                "rows": EXPECTED_ROWS,
                "delta_lml_min": float(np.min(delta)),
                "delta_lml_median": float(np.median(delta)),
                "delta_lml_max": float(np.max(delta)),
                "nested_order_violations": int(np.sum(violation)),
                "tolerance": LML_TOLERANCE,
            }
        )
    pointwise = pd.DataFrame(pointwise_records)
    summary = pd.DataFrame(summary_records)
    total = int(summary["nested_order_violations"].sum())
    if total:
        failed = pointwise.loc[
            pointwise["nested_order_violation"],
            ["lower_factor", "upper_factor", "mass_MeV", "delta_lml_upper_minus_lower"],
        ]
        raise RuntimeError(
            f"nested LML audit has {total} violations at tolerance "
            f"{LML_TOLERANCE:g}; unchanged-card reruns are required. "
            f"First rows={failed.head(12).to_dict(orient='records')}"
        )
    return pointwise, summary


def max_abs_mass(values: np.ndarray) -> tuple[float, int]:
    index = int(np.argmax(np.abs(values)))
    return float(np.abs(values[index])), int(EXPECTED_MASS_MEV[index])


def quantile_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_max": float(np.max(values)),
    }


def build_comparison_tables(
    frames: dict[int, pd.DataFrame],
    factors: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = frames[REFERENCE_FACTOR]
    reference_a = reference["A_up"].to_numpy(float)
    reference_eps2 = reference["eps2_up"].to_numpy(float)
    reference_z = reference["Z_analytic"].to_numpy(float)
    reference_lml = reference["lml"].to_numpy(float)

    long_frames: list[pd.DataFrame] = []
    factor_records: list[dict[str, Any]] = []
    for factor in factors:
        frame = frames[factor].copy()
        a_ratio = frame["A_up"].to_numpy(float) / reference_a
        eps2_ratio = frame["eps2_up"].to_numpy(float) / reference_eps2
        ratio_difference = a_ratio - eps2_ratio
        delta_z = frame["Z_analytic"].to_numpy(float) - reference_z
        delta_lml = frame["lml"].to_numpy(float) - reference_lml
        frame["A_up_ratio_to_k8"] = a_ratio
        frame["eps2_up_ratio_to_k8"] = eps2_ratio
        frame["A_eps2_ratio_difference"] = ratio_difference
        frame["delta_Z_to_k8"] = delta_z
        frame["delta_lml_to_k8"] = delta_lml
        selected_columns = [
            "upper_factor",
            "mass_MeV",
            "mass_GeV",
            "A_up",
            "A_up_ratio_to_k8",
            "eps2_up",
            "eps2_up_ratio_to_k8",
            "A_eps2_ratio_difference",
            "p0_analytic",
            "Z_analytic",
            "delta_Z_to_k8",
            "lml",
            "delta_lml_to_k8",
            "ls_lo",
            "ls_hi",
            "ls_opt",
            "sigma_x",
            "ls_lo_over_sigma_x",
            "ls_hi_over_sigma_x",
            "ls_opt_over_sigma_x",
            "ls_opt_over_ls_hi",
            "at_upper_boundary",
            "n_train",
            "blind_lo",
            "blind_hi",
            "sigma_val",
            "row_source",
            "row_source_sha256",
            "row_source_kind",
            "optimizer_repair_applied",
            "review_status",
            "branch_multiplicity",
            "selected_repair_reproduced",
            "repair_reproduction_pending",
            "delta_lml_selected_minus_raw",
        ]
        long_frames.append(frame.loc[:, selected_columns])

        min_p_index = int(np.argmin(frame["p0_analytic"].to_numpy(float)))
        max_delta_z, max_delta_z_mass = max_abs_mass(delta_z)
        record: dict[str, Any] = {
            "upper_factor": factor,
            "rows": EXPECTED_ROWS,
            "upper_boundary_threshold": BOUNDARY_THRESHOLD,
            "upper_boundary_rows": int(frame["at_upper_boundary"].sum()),
            "upper_boundary_fraction": float(
                frame["at_upper_boundary"].mean()
            ),
            "ls_opt_over_sigma_x_min": float(
                frame["ls_opt_over_sigma_x"].min()
            ),
            "ls_opt_over_sigma_x_median": float(
                frame["ls_opt_over_sigma_x"].median()
            ),
            "ls_opt_over_sigma_x_max": float(
                frame["ls_opt_over_sigma_x"].max()
            ),
            "delta_lml_to_k8_min": float(np.min(delta_lml)),
            "delta_lml_to_k8_median": float(np.median(delta_lml)),
            "delta_lml_to_k8_max": float(np.max(delta_lml)),
            "minimum_local_asymptotic_p0": float(
                frame.iloc[min_p_index]["p0_analytic"]
            ),
            "minimum_local_asymptotic_p0_mass_MeV": int(
                frame.iloc[min_p_index]["mass_MeV"]
            ),
            "maximum_local_asymptotic_Z": float(
                frame.iloc[min_p_index]["Z_analytic"]
            ),
            "max_abs_delta_Z_to_k8": max_delta_z,
            "max_abs_delta_Z_to_k8_mass_MeV": max_delta_z_mass,
            "p0_zero_rows": int(
                np.sum(frame["p0_analytic"].to_numpy(float) == 0.0)
            ),
            "max_abs_A_eps2_ratio_difference": float(
                np.max(np.abs(ratio_difference))
            ),
        }
        record.update(quantile_summary("A_up_ratio_to_k8", a_ratio))
        record.update(quantile_summary("eps2_up_ratio_to_k8", eps2_ratio))
        factor_records.append(record)

    adjacent_frames: list[pd.DataFrame] = []
    adjacent_records: list[dict[str, Any]] = []
    for lower, upper in zip(factors[:-1], factors[1:]):
        lower_frame = frames[lower]
        upper_frame = frames[upper]
        a_ratio = (
            upper_frame["A_up"].to_numpy(float)
            / lower_frame["A_up"].to_numpy(float)
        )
        eps2_ratio = (
            upper_frame["eps2_up"].to_numpy(float)
            / lower_frame["eps2_up"].to_numpy(float)
        )
        ratio_difference = a_ratio - eps2_ratio
        delta_z = (
            upper_frame["Z_analytic"].to_numpy(float)
            - lower_frame["Z_analytic"].to_numpy(float)
        )
        delta_lml = (
            upper_frame["lml"].to_numpy(float)
            - lower_frame["lml"].to_numpy(float)
        )
        pointwise = pd.DataFrame(
            {
                "lower_factor": lower,
                "upper_factor": upper,
                "mass_MeV": EXPECTED_MASS_MEV,
                "mass_GeV": EXPECTED_MASS_GEV,
                "A_up_upper_over_lower": a_ratio,
                "eps2_up_upper_over_lower": eps2_ratio,
                "A_eps2_ratio_difference": ratio_difference,
                "delta_Z_upper_minus_lower": delta_z,
                "delta_lml_upper_minus_lower": delta_lml,
                "lower_at_upper_boundary": lower_frame[
                    "at_upper_boundary"
                ].to_numpy(bool),
                "upper_at_upper_boundary": upper_frame[
                    "at_upper_boundary"
                ].to_numpy(bool),
            }
        )
        adjacent_frames.append(pointwise)
        max_delta_z, max_delta_z_mass = max_abs_mass(delta_z)
        max_ratio_index = int(np.argmax(np.abs(a_ratio - 1.0)))
        record = {
            "lower_factor": lower,
            "upper_factor": upper,
            "rows": EXPECTED_ROWS,
            "A_up_outside_5pct_rows": int(
                np.sum(np.abs(a_ratio - 1.0) > 0.05)
            ),
            "A_up_outside_10pct_rows": int(
                np.sum(np.abs(a_ratio - 1.0) > 0.10)
            ),
            "A_up_max_abs_deviation_mass_MeV": int(
                EXPECTED_MASS_MEV[max_ratio_index]
            ),
            "max_abs_delta_Z": max_delta_z,
            "max_abs_delta_Z_mass_MeV": max_delta_z_mass,
            "delta_lml_min": float(np.min(delta_lml)),
            "delta_lml_median": float(np.median(delta_lml)),
            "delta_lml_max": float(np.max(delta_lml)),
            "lower_boundary_rows": int(
                lower_frame["at_upper_boundary"].sum()
            ),
            "upper_boundary_rows": int(
                upper_frame["at_upper_boundary"].sum()
            ),
            "max_abs_A_eps2_ratio_difference": float(
                np.max(np.abs(ratio_difference))
            ),
        }
        record.update(
            quantile_summary("A_up_upper_over_lower", a_ratio)
        )
        record.update(
            quantile_summary("eps2_up_upper_over_lower", eps2_ratio)
        )
        adjacent_records.append(record)

    return (
        pd.concat(long_frames, ignore_index=True),
        pd.DataFrame(factor_records),
        pd.concat(adjacent_frames, ignore_index=True),
        pd.DataFrame(adjacent_records),
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, which="major", alpha=0.23, linewidth=0.7)
    axis.grid(True, which="minor", alpha=0.08, linewidth=0.5)
    axis.tick_params(direction="in", which="both", top=True, right=True)
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))


def factor_label(factor: int) -> str:
    if factor == REFERENCE_FACTOR:
        return rf"$k_{{\max}}={factor}$ (v4 reviewed)"
    return rf"$k_{{\max}}={factor}$"


def draw_curves(
    axis: plt.Axes,
    frames: dict[int, pd.DataFrame],
    factors: tuple[int, ...],
    column: str,
    *,
    log_y: bool = False,
) -> None:
    for factor in factors:
        values = frames[factor][column].to_numpy(float)
        if log_y:
            positive = values[values > 0.0]
            floor = (
                max(float(np.min(positive)) * 0.1, np.finfo(float).tiny)
                if len(positive)
                else np.finfo(float).tiny
            )
            values = np.clip(values, floor, None)
        axis.plot(
            frames[factor]["mass_MeV"],
            values,
            color=COLORS[factor],
            linestyle=LINESTYLES[factor],
            linewidth=2.1 if factor == REFERENCE_FACTOR else 1.25,
            label=factor_label(factor),
        )
    if log_y:
        axis.set_yscale("log")
        axis.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
        )
        axis.yaxis.set_minor_formatter(NullFormatter())
    style_axis(axis)


def add_legend(axis: plt.Axes, y: float = 1.17) -> None:
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=3,
        frameon=False,
        fontsize=8.8,
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    # Titles, labels, legends, and reference lines are structural.  No
    # explanatory ax.text/annotate/figtext blocks are placed on the figures.
    fig.align_ylabels()
    paths = [PLOTS / f"{stem}.png", PLOTS / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=260, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def make_plots(
    frames: dict[int, pd.DataFrame],
    factors: tuple[int, ...],
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    products: list[Path] = []

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.0),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    draw_curves(axes[0], frames, factors, "A_up", log_y=True)
    axes[0].set_ylabel(r"Observed 90% CLs $A_{\rm up}$ [events]")
    add_legend(axes[0])
    draw_curves(axes[1], frames, factors, "eps2_up", log_y=True)
    axes[1].set_ylabel(r"Observed 90% CLs $\epsilon^2_{\rm up}$")
    axes[1].set_xlabel(r"Mass hypothesis $m_{A'}$ [MeV]")
    products.extend(
        save_figure(fig, "observed_2016_limits_by_ls_upper_factor")
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.35), "hspace": 0.08},
    )
    draw_curves(axes[0], frames, factors, "p0_analytic", log_y=True)
    axes[0].set_ylabel(r"Local asymptotic $p_0$")
    add_legend(axes[0])
    reference_z = frames[REFERENCE_FACTOR]["Z_analytic"].to_numpy(float)
    all_delta: list[np.ndarray] = []
    for factor in factors:
        delta = frames[factor]["Z_analytic"].to_numpy(float) - reference_z
        all_delta.append(delta)
        axes[1].plot(
            frames[factor]["mass_MeV"],
            delta,
            color=COLORS[factor],
            linestyle=LINESTYLES[factor],
            linewidth=2.1 if factor == REFERENCE_FACTOR else 1.25,
        )
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    merged = np.concatenate(all_delta)
    span = max(float(np.ptp(merged)), 0.5)
    axes[1].set_ylim(
        float(np.min(merged)) - 0.07 * span,
        float(np.max(merged)) + 0.07 * span,
    )
    axes[1].set_ylabel(r"$\Delta Z_{\rm local}$ vs k8")
    axes[1].set_xlabel(r"Mass hypothesis $m_{A'}$ [MeV]")
    style_axis(axes[1])
    products.extend(
        save_figure(fig, "local_asymptotic_p0_by_ls_upper_factor")
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.2, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": (2.1, 1.5, 2.0), "hspace": 0.08},
    )
    draw_curves(axes[0], frames, factors, "ls_opt_over_sigma_x")
    axes[0].set_ylabel(r"$\ell_{\rm opt}/\sigma_x$")
    add_legend(axes[0], y=1.23)
    draw_curves(axes[1], frames, factors, "ls_opt_over_ls_hi")
    axes[1].axhline(
        BOUNDARY_THRESHOLD, color="0.35", linewidth=0.8, linestyle="--"
    )
    axes[1].set_ylabel(r"$\ell_{\rm opt}/\ell_{\rm hi}$")
    reference_lml = frames[REFERENCE_FACTOR]["lml"].to_numpy(float)
    for factor in factors:
        axes[2].plot(
            frames[factor]["mass_MeV"],
            frames[factor]["lml"].to_numpy(float) - reference_lml,
            color=COLORS[factor],
            linestyle=LINESTYLES[factor],
            linewidth=2.1 if factor == REFERENCE_FACTOR else 1.25,
        )
    axes[2].axhline(0.0, color="0.35", linewidth=0.8)
    axes[2].set_ylabel(r"$\Delta\log\mathcal{L}$ vs k8")
    axes[2].set_xlabel(r"Mass hypothesis $m_{A'}$ [MeV]")
    style_axis(axes[2])
    products.extend(
        save_figure(fig, "lml_and_length_scale_boundary_occupancy")
    )
    return products


def build_source_manifest(
    frames: dict[int, pd.DataFrame],
    frame_metadata: dict[int, dict[str, Any]],
    config_paths: dict[int, Path],
    report_paths: dict[int, list[Path]],
    factors: tuple[int, ...],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for factor in factors:
        metadata = frame_metadata[factor]
        records.append(
            {
                "upper_factor": factor,
                "source_kind": metadata["source_kind"],
                "result": metadata["result_source"],
                "result_sha256": metadata["result_source_sha256"],
                "config": repo_path(config_paths[factor]),
                "config_sha256": sha256(config_paths[factor]),
                "validation_reports": "|".join(
                    repo_path(path) for path in report_paths[factor]
                ),
                "validation_report_sha256s": "|".join(
                    sha256(path) for path in report_paths[factor]
                ),
                "rows": int(len(frames[factor])),
                "mass_min_MeV": int(frames[factor]["mass_MeV"].min()),
                "mass_max_MeV": int(frames[factor]["mass_MeV"].max()),
                "realized_upper_factor_min": float(
                    frames[factor]["ls_hi_over_sigma_x"].min()
                ),
                "realized_upper_factor_max": float(
                    frames[factor]["ls_hi_over_sigma_x"].max()
                ),
                "upper_boundary_rows": int(
                    frames[factor]["at_upper_boundary"].sum()
                ),
            }
        )
    return pd.DataFrame(records)


def build_k12_reviewed_overlay(k12: pd.DataFrame) -> pd.DataFrame:
    """Replace the v4 2016 states with reviewed factor-12 actual fit rows."""
    v4 = pd.read_csv(V4_REVIEW)
    v4["dataset"] = v4["dataset"].astype(str).str.strip()
    expected_counts = {"2015": 72, "2016": 142, "2021": 201}
    if v4.groupby("dataset").size().to_dict() != expected_counts:
        raise RuntimeError("v4 reviewed state table has unexpected dataset counts")

    core = [
        "dataset",
        "mass_GeV",
        "const_opt",
        "ls_opt",
        "lml",
        "ls_hi",
        "ls_hi_over_sigma_x",
    ]
    provenance = [
        "interpolated",
        "selected_source",
        "selected_source_sha256",
        "row_source",
        "optimizer_repair_applied",
        "review_status",
        "branch_multiplicity",
        "reproducing_sources",
        "selected_repair_reproduced",
        "repair_reproduction_pending",
        "candidate_count",
        "repair_candidate_count",
        "delta_lml_selected_minus_raw",
    ]
    non2016 = v4.loc[v4["dataset"] != "2016"].copy()
    non2016["selected_repair_reproduced"] = False
    non2016["repair_reproduction_pending"] = False
    non2016["candidate_count"] = non2016["branch_multiplicity"].astype(int)
    non2016["repair_candidate_count"] = 0
    non2016["delta_lml_selected_minus_raw"] = 0.0

    replacement = k12.copy()
    replacement["dataset"] = "2016"
    replacement["interpolated"] = False
    overlay = pd.concat(
        [non2016.loc[:, core + provenance], replacement.loc[:, core + provenance]],
        ignore_index=True,
    )
    order = {"2015": 0, "2016": 1, "2021": 2}
    overlay["_order"] = overlay["dataset"].map(order)
    overlay = overlay.sort_values(["_order", "mass_GeV"]).drop(
        columns="_order"
    ).reset_index(drop=True)
    if overlay.groupby("dataset").size().to_dict() != expected_counts:
        raise RuntimeError("factor-12 overlay has unexpected dataset counts")
    if bool(overlay.duplicated(["dataset", "mass_GeV"]).any()):
        raise RuntimeError("factor-12 overlay has duplicate states")
    numeric = overlay.loc[
        :, ["mass_GeV", "const_opt", "ls_opt", "lml", "ls_hi", "ls_hi_over_sigma_x"]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("factor-12 overlay has non-finite core values")
    if bool(
        (
            overlay["ls_opt"].to_numpy(float)
            > overlay["ls_hi"].to_numpy(float) * (1.0 + 1.0e-10) + 1.0e-14
        ).any()
    ):
        raise RuntimeError("factor-12 overlay has ls_opt above ls_hi")
    expected_upper = {"2015": 8.0, "2016": 12.0, "2021": 15.0}
    for dataset, factor in expected_upper.items():
        values = overlay.loc[
            overlay["dataset"] == dataset, "ls_hi_over_sigma_x"
        ].to_numpy(float)
        if not np.allclose(values, factor, rtol=0.0, atol=1.0e-8):
            raise RuntimeError(
                f"factor-12 overlay has wrong realized upper factor for {dataset}"
            )
    return overlay


def product_record(path: Path, kind: str) -> dict[str, Any]:
    return {
        "path": repo_path(path),
        "kind": kind,
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def write_product_manifest(
    generated_csv: Iterable[Path],
    generated_json: Iterable[Path],
    generated_plots: Iterable[Path],
) -> Path:
    records: list[dict[str, Any]] = []
    records.extend(product_record(path, "derived_csv") for path in generated_csv)
    records.extend(
        product_record(path, "derived_json") for path in generated_json
    )
    records.extend(product_record(path, "plot") for path in generated_plots)
    path = DERIVED / "product_manifest.json"
    write_json(
        path,
        {
            "study": "v4p1_2016_length_scale_upper_bound_grid",
            "selection_performed": False,
            "toys_generated": False,
            "limit_bands_generated": False,
            "products": records,
        },
    )
    return path


def write_checksums(
    generated: Iterable[Path],
    source_paths: Iterable[Path],
) -> Path:
    unique = sorted(
        {path.resolve() for path in [*generated, *source_paths]},
        key=lambda path: repo_path(path),
    )
    output = DERIVED / "sha256sums.txt"
    output.write_text(
        "\n".join(f"{sha256(path)}  {repo_path(path)}" for path in unique) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    config_paths, config_audit, config_manifest = audit_configs()
    factors, source_paths = discover_complete_sources()

    frames: dict[int, pd.DataFrame] = {}
    frame_metadata: dict[int, dict[str, Any]] = {}
    repair_paths: dict[int, list[Path]] = {factor: [] for factor in factors}
    repair_ledgers: list[pd.DataFrame] = []

    reference_raw = pd.read_csv(source_paths[REFERENCE_FACTOR])
    frames[REFERENCE_FACTOR], frame_metadata[REFERENCE_FACTOR] = validate_frame(
        reference_raw, REFERENCE_FACTOR, source_paths[REFERENCE_FACTOR]
    )
    reference = frames[REFERENCE_FACTOR]
    reference["row_source"] = reference["selected_source"].astype(str)
    reference["row_source_sha256"] = reference[
        "selected_source_sha256"
    ].astype(str)
    reference["row_source_kind"] = "reviewed_v4"
    reference["optimizer_repair_applied"] = normalize_boolean(
        reference["optimizer_repair_applied"],
        "factor8.optimizer_repair_applied",
    )
    reference["selected_repair_reproduced"] = False
    reference["repair_reproduction_pending"] = False
    reference["candidate_count"] = reference["branch_multiplicity"].astype(int)
    reference["repair_candidate_count"] = 0
    reference["delta_lml_selected_minus_raw"] = 0.0
    reference["interpolated"] = False

    for factor in factors:
        if factor == REFERENCE_FACTOR:
            continue
        (
            frames[factor],
            frame_metadata[factor],
            ledger,
            repair_paths[factor],
        ) = stitch_factor_candidates(
            factor, source_paths[factor]
        )
        repair_ledgers.append(ledger)

    report_paths, validation_report_audit = audit_validation_reports(
        factors, repair_paths
    )
    same_input_audit = compare_same_inputs(frames, factors)
    nested_pointwise, nested_summary = audit_nested_lml(frames, factors)
    (
        pointwise,
        factor_summary,
        adjacent_pointwise,
        adjacent_summary,
    ) = build_comparison_tables(frames, factors)
    source_manifest = build_source_manifest(
        frames, frame_metadata, config_paths, report_paths, factors
    )
    repair_ledger = pd.concat(repair_ledgers, ignore_index=True)
    k12_overlay = build_k12_reviewed_overlay(frames[12])

    DERIVED.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    csv_products = {
        "pointwise_factor_grid.csv": pointwise,
        "factor_summary.csv": factor_summary,
        "adjacent_factor_comparison.csv": adjacent_pointwise,
        "adjacent_factor_summary.csv": adjacent_summary,
        "nested_lml_audit.csv": nested_pointwise,
        "nested_lml_summary.csv": nested_summary,
        "same_input_audit.csv": same_input_audit,
        "config_audit.csv": config_audit.loc[
            config_audit["upper_factor"].isin(factors)
        ].reset_index(drop=True),
        "source_manifest.csv": source_manifest,
        "repair_candidate_ledger.csv": repair_ledger,
        "observed_gp_states_k12_reviewed.csv": k12_overlay,
    }
    csv_paths: list[Path] = []
    for name, table in csv_products.items():
        path = DERIVED / name
        table.to_csv(path, index=False)
        csv_paths.append(path)

    plot_paths = make_plots(frames, factors)
    run_summary_path = DERIVED / "run_summary.json"
    run_summary = {
        "study": "v4p1_2016_length_scale_upper_bound_grid",
        "scope": "observed_2016_asymptotic_diagnostic_only",
        "reference": {
            "upper_factor": REFERENCE_FACTOR,
            "source": repo_path(V4_REVIEW),
            "state_policy": "reviewed_unchanged_card_reproduced_max_lml",
        },
        "factors": list(factors),
        "required_factors": list(REQUIRED_FACTORS),
        "optional_factors_included": [
            factor for factor in OPTIONAL_FACTORS if factor in factors
        ],
        "mass_grid": {
            "min_MeV": int(EXPECTED_MASS_MEV.min()),
            "max_MeV": int(EXPECTED_MASS_MEV.max()),
            "step_MeV": 1,
            "rows_per_factor": EXPECTED_ROWS,
        },
        "boundary_criterion": {
            "definition": "ls_opt / ls_hi >= threshold",
            "threshold": BOUNDARY_THRESHOLD,
        },
        "nested_lml": {
            "definition": "lml(upper factor) - lml(lower factor)",
            "tolerance": LML_TOLERANCE,
            "violations": int(
                nested_summary["nested_order_violations"].sum()
            ),
        },
        "validation": {
            "all_exact_142_row_grids": True,
            "all_required_values_finite": True,
            "all_extractions_successful": True,
            "all_configured_and_realized_factors_match": True,
            "same_input_geometry_and_semantics": True,
            "same_input_validation_payload": True,
            "A_up_and_eps2_ratios_computed_separately": True,
            "A_up_and_eps2_ratio_identity_assumed": False,
            "interpolation_used": False,
            "free_form_plot_text_blocks": False,
            "repair_row_selection": "highest_lml_actual_fit_row",
            "selected_repairs_reproduced_or_explicitly_flagged": True,
            "repair_reproduction_tolerance": REPAIR_REPRODUCTION_TOLERANCE,
            "factor12_415_state_overlay_validated": True,
        },
        "interpretation_policy": {
            "selection_performed": False,
            "selection_from_observed_limits_or_pvalues_allowed": False,
            "factor_summary_is_diagnostic_not_a_ranking": True,
            "requires_separate_predeclared_closure_and_coverage_for_change": True,
        },
        "products_excluded": {
            "pseudoexperiments": True,
            "expected_limit_bands": True,
            "toy_calibrated_local_pvalues": True,
            "toy_calibrated_global_pvalues": True,
        },
        "config_manifest": {
            "path": repo_path(CONFIG_MANIFEST),
            "sha256": sha256(CONFIG_MANIFEST),
            "declared_source_config": config_manifest["source_config"],
            "declared_factors": config_manifest["factors"],
        },
        "validation_report_audit": validation_report_audit,
        "frame_metadata": frame_metadata,
        "factor_summary": factor_summary.to_dict(orient="records"),
        "adjacent_factor_summary": adjacent_summary.to_dict(orient="records"),
        "plots": [
            {
                "stem": "observed_2016_limits_by_ls_upper_factor",
                "content": "observed 90% asymptotic CLs A_up and eps2_up; no bands",
            },
            {
                "stem": "local_asymptotic_p0_by_ls_upper_factor",
                "content": "local asymptotic p0 and delta Z relative to reviewed k8",
            },
            {
                "stem": "lml_and_length_scale_boundary_occupancy",
                "content": "optimized length scale, boundary occupancy, and delta LML",
            },
        ],
    }
    write_json(run_summary_path, run_summary)

    product_manifest_path = write_product_manifest(
        csv_paths, [run_summary_path], plot_paths
    )
    generated = [
        Path(__file__).resolve(),
        *csv_paths,
        run_summary_path,
        product_manifest_path,
        *plot_paths,
    ]
    all_source_paths = [
        CONFIG_MANIFEST,
        V4_CONFIG,
        *config_paths.values(),
        *source_paths.values(),
        *(path for paths in repair_paths.values() for path in paths),
        *(path for paths in report_paths.values() for path in paths),
    ]
    checksum_path = write_checksums(generated, all_source_paths)
    print(
        f"Validated {len(factors)} factors x {EXPECTED_ROWS} masses; "
        f"wrote {len(csv_paths)} CSVs, 2 JSON manifests, "
        f"{len(plot_paths)} plots, and {repo_path(checksum_path)}."
    )
    print(
        "No factor selection, pseudoexperiments, expected bands, or "
        "toy-calibrated p-values were produced."
    )


if __name__ == "__main__":
    main()
