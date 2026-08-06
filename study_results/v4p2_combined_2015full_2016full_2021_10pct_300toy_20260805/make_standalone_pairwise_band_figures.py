#!/usr/bin/env python3
"""Validate and plot the v4.2 standalone and pairwise 100-toy bands.

This script is downstream-only.  It reads the six-scope fixed-GP band table,
validates the accompanying production validation and provenance records, applies
the minimal-visible branching reinterpretation, assembles union-range pairwise
display tables, and writes publication figures.

The input pairwise scopes contain only masses where both named datasets are
active.  For display, each pairwise panel is extended over the union of the two
campaign search ranges with the authoritative standalone rows outside the
overlap.  The active-set rail makes that construction explicit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-gpr-v4p2-bands100-plot-mpl")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1785898800")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, LogLocator, MultipleLocator, NullFormatter


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO = CAMPAIGN_DIR.parents[1]
INPUT_DIR = CAMPAIGN_DIR / "standalone_pairwise_bands100_fixed"
INPUT_CSV = INPUT_DIR / "ul_bands_standalone_pairwise_100.csv"
INPUT_VALIDATION = INPUT_DIR / "validation_standalone_pairwise_100.json"
INPUT_PROVENANCE = INPUT_DIR / "provenance_standalone_pairwise_100.json"

DERIVED_DIR = CAMPAIGN_DIR / "derived"
FIGURE_DIR = CAMPAIGN_DIR / "note_figures"
NOTE_FIGURE_DIR = (
    REPO
    / "hps_gpr_analysis_note"
    / "final_limit_projection_figs"
    / "v4p2_20260805_combined300"
)
DELIVERY_DIR = REPO / "output" / "pdf"

PAIRWISE_UNION_CSV = (
    DERIVED_DIR / "pairwise_union_bands100_reviewed_v4p2.csv"
)
PLOT_VALIDATION = (
    DERIVED_DIR / "standalone_pairwise_band_figures_validation_v4p2.json"
)
PLOT_PROVENANCE = (
    DERIVED_DIR / "standalone_pairwise_band_figures_provenance_v4p2.json"
)

STANDALONE_STEM = "standalone_observed_bands100_minimal_visible"
PAIRWISE_STEM = "pairwise_observed_bands100_minimal_visible"

N_INPUT_ROWS = 639
N_PAIRWISE_UNION_ROWS = 606
N_TOYS = 100
ACCEPTED_PARENT_DRAW_COUNT = 300
SELECTED_PARENT_INDICES = (0, 99)
STANDALONE_OBSERVED_CLOSURE_RTOL = 6.0e-6
M_MU_GEV = 0.1056583745
DIMUON_THRESHOLD_GEV = 2.0 * M_MU_GEV
DIMUON_THRESHOLD_MEV = 1000.0 * DIMUON_THRESHOLD_GEV

COLORS = {
    "observed": "#B42318",
    "expected": "#202124",
    "band1": "#4C956C",
    "band2": "#F2C14E",
    "threshold": "#6B7280",
}
ACTIVE_COLORS = {
    "2015": "#DCEAF7",
    "2016": "#F4DDD8",
    "2021": "#D9EEE7",
    "2015+2016": "#E7E0F2",
    "2015+2021": "#DDEFE8",
    "2016+2021": "#F2E4D8",
}
ACTIVE_LABELS = {
    "2015": "15",
    "2016": "16",
    "2021": "21",
    "2015+2016": "15+16",
    "2015+2021": "15+21",
    "2016+2021": "16+21",
}

SCOPE_SPECS: Dict[str, Dict[str, Any]] = {
    "individual_2015": {
        "scope_type": "standalone",
        "dataset_set": "2015",
        "dataset_keys": ("2015",),
        "label": "2015 100%",
        "mass_low_MeV": 19,
        "mass_high_MeV": 90,
        "n_rows": 72,
    },
    "individual_2016": {
        "scope_type": "standalone",
        "dataset_set": "2016",
        "dataset_keys": ("2016",),
        "label": "2016 100%",
        "mass_low_MeV": 39,
        "mass_high_MeV": 180,
        "n_rows": 142,
    },
    "individual_2021": {
        "scope_type": "standalone",
        "dataset_set": "2021",
        "dataset_keys": ("2021",),
        "label": "2021 10%",
        "mass_low_MeV": 50,
        "mass_high_MeV": 250,
        "n_rows": 201,
    },
    "pair_2015_2016": {
        "scope_type": "pairwise",
        "dataset_set": "2015+2016",
        "dataset_keys": ("2015", "2016"),
        "label": "2015 100% + 2016 100%",
        "mass_low_MeV": 39,
        "mass_high_MeV": 90,
        "n_rows": 52,
    },
    "pair_2015_2021": {
        "scope_type": "pairwise",
        "dataset_set": "2015+2021",
        "dataset_keys": ("2015", "2021"),
        "label": "2015 100% + 2021 10%",
        "mass_low_MeV": 50,
        "mass_high_MeV": 90,
        "n_rows": 41,
    },
    "pair_2016_2021": {
        "scope_type": "pairwise",
        "dataset_set": "2016+2021",
        "dataset_keys": ("2016", "2021"),
        "label": "2016 100% + 2021 10%",
        "mass_low_MeV": 50,
        "mass_high_MeV": 180,
        "n_rows": 131,
    },
}

STANDALONE_PANELS = (
    ("individual_2015", "(a) 2015 100%"),
    ("individual_2016", "(b) 2016 100%"),
    ("individual_2021", "(c) 2021 10%"),
)

PAIRWISE_UNION_SPECS: Dict[str, Dict[str, Any]] = {
    "pair_union_2015_2016": {
        "label": "2015 100% + 2016 100%",
        "panel_title": "(a) 2015 100% + 2016 100%",
        "pair_scope_key": "pair_2015_2016",
        "left_scope_key": "individual_2015",
        "right_scope_key": "individual_2016",
        "mass_low_MeV": 19,
        "mass_high_MeV": 180,
        "overlap_low_MeV": 39,
        "overlap_high_MeV": 90,
        "n_rows": 162,
    },
    "pair_union_2015_2021": {
        "label": "2015 100% + 2021 10%",
        "panel_title": "(b) 2015 100% + 2021 10%",
        "pair_scope_key": "pair_2015_2021",
        "left_scope_key": "individual_2015",
        "right_scope_key": "individual_2021",
        "mass_low_MeV": 19,
        "mass_high_MeV": 250,
        "overlap_low_MeV": 50,
        "overlap_high_MeV": 90,
        "n_rows": 232,
    },
    "pair_union_2016_2021": {
        "label": "2016 100% + 2021 10%",
        "panel_title": "(c) 2016 100% + 2021 10%",
        "pair_scope_key": "pair_2016_2021",
        "left_scope_key": "individual_2016",
        "right_scope_key": "individual_2021",
        "mass_low_MeV": 39,
        "mass_high_MeV": 250,
        "overlap_low_MeV": 50,
        "overlap_high_MeV": 180,
        "n_rows": 212,
    },
}

RAW_COUPLING_COLUMNS = (
    "eps2_obs",
    "eps2_obs_solved",
    "eps2_lo2",
    "eps2_lo1",
    "eps2_med",
    "eps2_hi1",
    "eps2_hi2",
    "eps2_mean",
)
RAW_QUANTILE_COLUMNS = (
    "eps2_lo2",
    "eps2_lo1",
    "eps2_med",
    "eps2_hi1",
    "eps2_hi2",
)
MINIMAL_VISIBLE_QUANTILE_COLUMNS = tuple(
    f"{column}_minimal_visible" for column in RAW_QUANTILE_COLUMNS
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def bool_values(series: pd.Series, column: str) -> np.ndarray:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0", "yes", "no"})
    if not bool(valid.all()):
        bad = sorted(set(normalized.loc[~valid].tolist()))
        raise RuntimeError(f"Invalid boolean values in {column}: {bad}")
    return normalized.isin({"true", "1", "yes"}).to_numpy(bool)


def require_all_boolean(
    frame: pd.DataFrame,
    column: str,
    expected: bool,
) -> None:
    values = bool_values(frame[column], column)
    if not bool(np.all(values == bool(expected))):
        raise RuntimeError(f"{column} is not uniformly {expected}")


def expected_scope_counts() -> Dict[str, int]:
    return {
        key: int(spec["n_rows"])
        for key, spec in SCOPE_SPECS.items()
    }


def validate_metadata(
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if validation.get("status") != "PASS":
        raise RuntimeError("Input validation status is not PASS")
    if provenance.get("status") != "PASS":
        raise RuntimeError("Input provenance status is not PASS")
    if int(validation.get("n_scope_mass_rows", -1)) != N_INPUT_ROWS:
        raise RuntimeError("Input validation row count is not 639")
    if int(provenance.get("n_scope_mass_rows", -1)) != N_INPUT_ROWS:
        raise RuntimeError("Input provenance row count is not 639")
    if int(validation.get("n_toys_per_scope_mass", -1)) != N_TOYS:
        raise RuntimeError("Input validation toy count is not 100")
    if int(provenance.get("n_toys_per_scope_mass", -1)) != N_TOYS:
        raise RuntimeError("Input provenance toy count is not 100")
    if int(provenance.get("accepted_parent_draw_count", -1)) != (
        ACCEPTED_PARENT_DRAW_COUNT
    ):
        raise RuntimeError("Accepted parent draw count is not 300")
    if list(validation.get("selected_parent_toy_indices", [])) != [0, 99]:
        raise RuntimeError("Validation does not select parent indices 0--99")
    if list(provenance.get("selected_parent_toy_indices", [])) != [0, 99]:
        raise RuntimeError("Provenance does not select parent indices 0--99")
    if validation.get("scope_row_counts") != expected_scope_counts():
        raise RuntimeError("Validation scope counts do not match the six scopes")
    if int(validation.get("n_finite_toy_limits", -1)) != (
        N_INPUT_ROWS * N_TOYS
    ):
        raise RuntimeError("Validation finite toy-limit count is incomplete")
    if int(provenance.get("n_finite_toy_limits", -1)) != (
        N_INPUT_ROWS * N_TOYS
    ):
        raise RuntimeError("Provenance finite toy-limit count is incomplete")
    if provenance.get("combined_mode") != "count_scale":
        raise RuntimeError("Input provenance is not the count_scale study")
    if bool(provenance.get("coverage_calibrated", True)):
        raise RuntimeError("Input bands were mislabeled coverage calibrated")
    if bool(provenance.get("scan_toy_calibrated", True)):
        raise RuntimeError("Input bands were mislabeled scan-toy calibrated")
    if bool(provenance.get("refit_gp_on_toy", True)):
        raise RuntimeError("Input bands were not generated at fixed GP states")
    if not str(
        provenance.get("standalone_limit_root_convention", "")
    ).startswith("reviewed native total-amplitude bisection"):
        raise RuntimeError(
            "Standalone native amplitude-root provenance is missing"
        )
    if not str(
        provenance.get("standalone_reported_eps2_conversion", "")
    ).startswith("authoritative reviewed A_up/eps2_up"):
        raise RuntimeError(
            "Standalone reviewed epsilon-squared conversion provenance is missing"
        )
    if provenance.get("pairwise_limit_root_convention") != (
        "accepted v4.2 combined epsilon2-coordinate bisection"
    ):
        raise RuntimeError("Pairwise epsilon-squared root provenance drifted")
    if not np.isclose(
        float(provenance.get("standalone_observed_closure_rtol", np.nan)),
        STANDALONE_OBSERVED_CLOSURE_RTOL,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("Standalone observed-closure tolerance drifted")

    scopes = provenance.get("scopes")
    if not isinstance(scopes, dict) or set(scopes) != set(SCOPE_SPECS):
        raise RuntimeError("Input provenance scope definitions are incomplete")
    for key, expected in SCOPE_SPECS.items():
        found = scopes[key]
        if str(found.get("scope_type")) != str(expected["scope_type"]):
            raise RuntimeError(f"Scope type drift for {key}")
        if tuple(str(item) for item in found.get("dataset_keys", [])) != tuple(
            expected["dataset_keys"]
        ):
            raise RuntimeError(f"Dataset membership drift for {key}")
        for field in ("mass_low_MeV", "mass_high_MeV"):
            if int(found.get(field, -1)) != int(expected[field]):
                raise RuntimeError(f"{field} drift for {key}")
        if int(found.get("n_masses", -1)) != int(expected["n_rows"]):
            raise RuntimeError(f"Mass-count drift for {key}")

    if provenance.get("output_csv_sha256") != sha256(INPUT_CSV):
        raise RuntimeError("Input CSV hash does not match provenance")
    if provenance.get("validation_sha256") != sha256(INPUT_VALIDATION):
        raise RuntimeError("Input validation hash does not match provenance")


def validate_input_rows(frame: pd.DataFrame) -> None:
    required = {
        "scope_key",
        "scope_type",
        "scope_label",
        "dataset_set",
        "n_datasets",
        "mass_GeV",
        "mass_MeV",
        "scope_mass_low_MeV",
        "scope_mass_high_MeV",
        "cls_alpha",
        "eps2_obs",
        "eps2_obs_solved",
        "A_obs_source",
        "A_obs_solved",
        "likelihood_amplitude_per_eps2",
        "reported_amplitude_per_eps2",
        "reported_conversion_source",
        "eps2_lo2",
        "eps2_lo1",
        "eps2_med",
        "eps2_hi1",
        "eps2_hi2",
        "eps2_mean",
        "n_toys_requested",
        "n_toys_finite",
        "parent_draw_count",
        "selected_parent_toy_low",
        "selected_parent_toy_high",
        "toy_index_shared_within_scope",
        "toy_dataset_stream_reused_across_scopes",
        "cls_statistic",
        "cls_calibration",
        "combined_mode",
        "bands_refit_gp_on_toy",
        "bands_refit_restarts",
        "bands_refit_optimize",
        "observed_gp_fit_mode",
        "observed_gp_optimizer_restarts",
        "limit_root_convention",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Input band table lacks columns: {missing}")
    if len(frame) != N_INPUT_ROWS:
        raise RuntimeError(f"Expected 639 input rows, found {len(frame)}")
    if frame.duplicated(["scope_key", "mass_MeV"]).any():
        raise RuntimeError("Duplicate scope-mass rows in input table")

    found_counts = {
        str(key): int(value)
        for key, value in frame.groupby("scope_key").size().to_dict().items()
    }
    if found_counts != expected_scope_counts():
        raise RuntimeError(
            f"Input scope counts differ: {found_counts}"
        )

    for key, spec in SCOPE_SPECS.items():
        rows = frame.loc[frame["scope_key"].astype(str) == key].copy()
        expected_masses = np.arange(
            int(spec["mass_low_MeV"]),
            int(spec["mass_high_MeV"]) + 1,
            dtype=int,
        )
        found_masses = np.sort(rows["mass_MeV"].to_numpy(int))
        if not np.array_equal(found_masses, expected_masses):
            raise RuntimeError(f"Mass grid is incomplete for {key}")
        if set(rows["scope_type"].astype(str)) != {str(spec["scope_type"])}:
            raise RuntimeError(f"Scope type mismatch for {key}")
        if set(rows["dataset_set"].astype(str)) != {str(spec["dataset_set"])}:
            raise RuntimeError(f"Dataset-set mismatch for {key}")
        if set(rows["scope_mass_low_MeV"].astype(int)) != {
            int(spec["mass_low_MeV"])
        }:
            raise RuntimeError(f"Lower mass metadata mismatch for {key}")
        if set(rows["scope_mass_high_MeV"].astype(int)) != {
            int(spec["mass_high_MeV"])
        }:
            raise RuntimeError(f"Upper mass metadata mismatch for {key}")
        if set(rows["n_datasets"].astype(int)) != {
            len(spec["dataset_keys"])
        }:
            raise RuntimeError(f"Dataset-count mismatch for {key}")

    if not np.allclose(
        frame["mass_GeV"].to_numpy(float),
        frame["mass_MeV"].to_numpy(float) / 1000.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Input masses are not on the exact 1 MeV grid")
    if not bool((frame["n_toys_requested"].astype(int) == N_TOYS).all()):
        raise RuntimeError("Not every input row requests 100 toys")
    if not bool((frame["n_toys_finite"].astype(int) == N_TOYS).all()):
        raise RuntimeError("Not every input row has 100 finite toy limits")
    if not bool(
        (
            frame["parent_draw_count"].astype(int)
            == ACCEPTED_PARENT_DRAW_COUNT
        ).all()
    ):
        raise RuntimeError("Input rows do not use the 300-draw parent stream")
    if not bool((frame["selected_parent_toy_low"].astype(int) == 0).all()):
        raise RuntimeError("Input rows do not start at parent toy index 0")
    if not bool((frame["selected_parent_toy_high"].astype(int) == 99).all()):
        raise RuntimeError("Input rows do not end at parent toy index 99")
    require_all_boolean(frame, "toy_index_shared_within_scope", True)
    require_all_boolean(
        frame,
        "toy_dataset_stream_reused_across_scopes",
        True,
    )
    require_all_boolean(frame, "bands_refit_gp_on_toy", False)
    require_all_boolean(frame, "bands_refit_optimize", False)
    if not bool((frame["bands_refit_restarts"].astype(int) == 0).all()):
        raise RuntimeError("Input rows unexpectedly refit GP hyperparameters")
    if not bool(
        (frame["observed_gp_optimizer_restarts"].astype(int) == 0).all()
    ):
        raise RuntimeError("Observed GP states were unexpectedly reoptimized")
    if set(frame["observed_gp_fit_mode"].astype(str)) != {
        "fixed_reviewed_max_lml"
    }:
        raise RuntimeError("Observed GP fit-mode metadata drifted")
    if set(frame["cls_statistic"].astype(str)) != {"tilde_q_mu"}:
        raise RuntimeError("Input rows are not tilde_q_mu limits")
    if set(frame["cls_calibration"].astype(str)) != {"asymptotic"}:
        raise RuntimeError("Input rows are not asymptotic limits")
    if set(frame["combined_mode"].astype(str)) != {"count_scale"}:
        raise RuntimeError("Input rows are not count_scale fits")
    if not np.allclose(
        frame["cls_alpha"].to_numpy(float),
        0.1,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("Input rows are not 90% CL limits")

    standalone = frame.loc[
        frame["scope_type"].astype(str) == "standalone"
    ]
    pairwise = frame.loc[
        frame["scope_type"].astype(str) == "pairwise"
    ]
    if set(standalone["limit_root_convention"].astype(str)) != {
        "standalone_native_amplitude_coordinate"
    }:
        raise RuntimeError("Standalone root-convention metadata drifted")
    if set(pairwise["limit_root_convention"].astype(str)) != {
        "combined_epsilon2_coordinate"
    }:
        raise RuntimeError("Pairwise root-convention metadata drifted")
    if set(standalone["reported_conversion_source"].astype(str)) != {
        "authoritative_individual_A_up_over_eps2_up"
    }:
        raise RuntimeError("Standalone reported-conversion source drifted")
    if set(pairwise["reported_conversion_source"].astype(str)) != {
        "accepted_v4p2_combined_config"
    }:
        raise RuntimeError("Pairwise reported-conversion source drifted")
    standalone_numeric = standalone.loc[
        :,
        [
            "A_obs_source",
            "A_obs_solved",
            "likelihood_amplitude_per_eps2",
            "reported_amplitude_per_eps2",
        ],
    ].to_numpy(float)
    if not np.isfinite(standalone_numeric).all() or np.any(
        standalone_numeric <= 0.0
    ):
        raise RuntimeError(
            "Standalone amplitude/conversion metadata are not finite positive"
        )
    if not np.allclose(
        standalone["A_obs_solved"].to_numpy(float),
        standalone["A_obs_source"].to_numpy(float),
        rtol=STANDALONE_OBSERVED_CLOSURE_RTOL,
        atol=0.0,
    ):
        raise RuntimeError("Standalone observed-amplitude closure failed")
    if not np.allclose(
        standalone["eps2_obs_solved"].to_numpy(float),
        standalone["eps2_obs"].to_numpy(float),
        rtol=STANDALONE_OBSERVED_CLOSURE_RTOL,
        atol=0.0,
    ):
        raise RuntimeError("Standalone observed epsilon-squared closure failed")

    numeric = frame.loc[
        :,
        ["eps2_obs", "eps2_obs_solved", *RAW_QUANTILE_COLUMNS],
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite observed limit or band quantile")
    if np.any(numeric <= 0.0):
        raise RuntimeError("Non-positive observed limit or band quantile")
    ordered = frame.loc[:, RAW_QUANTILE_COLUMNS].to_numpy(float)
    if np.any(np.diff(ordered, axis=1) < 0.0):
        raise RuntimeError("Input band quantiles are not ordered")


def dimuon_factor(masses_gev: np.ndarray) -> np.ndarray:
    masses = np.asarray(masses_gev, dtype=float)
    factor = np.ones_like(masses)
    above = masses > DIMUON_THRESHOLD_GEV
    if np.any(above):
        phase_space = np.sqrt(
            1.0 - 4.0 * M_MU_GEV**2 / masses[above] ** 2
        )
        phase_space *= (
            1.0 + 2.0 * M_MU_GEV**2 / masses[above] ** 2
        )
        factor[above] = 1.0 + phase_space
    return factor


def add_minimal_visible_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    factor = dimuon_factor(out["mass_GeV"].to_numpy(float))
    out["dimuon_threshold_GeV"] = DIMUON_THRESHOLD_GEV
    out["dimuon_threshold_MeV"] = DIMUON_THRESHOLD_MEV
    out["N_eff_BR"] = factor
    out["BR_ee_minimal"] = 1.0 / factor
    out["dimuon_correction_applied"] = (
        out["mass_GeV"].to_numpy(float) > DIMUON_THRESHOLD_GEV
    )
    for column in RAW_COUPLING_COLUMNS:
        if column not in out.columns:
            raise RuntimeError(f"Missing coupling column: {column}")
        out[f"{column}_ee_channel"] = out[column].to_numpy(float)
        out[f"{column}_minimal_visible"] = (
            out[column].to_numpy(float) * factor
        )
    ordered = out.loc[:, MINIMAL_VISIBLE_QUANTILE_COLUMNS].to_numpy(float)
    if not np.isfinite(ordered).all() or np.any(ordered <= 0.0):
        raise RuntimeError("Minimal-visible quantiles are not finite and positive")
    if np.any(np.diff(ordered, axis=1) < 0.0):
        raise RuntimeError("Minimal-visible quantiles are not ordered")
    if not np.allclose(
        out["eps2_obs_minimal_visible"].to_numpy(float),
        out["eps2_obs"].to_numpy(float) * factor,
        rtol=5.0e-15,
        atol=0.0,
    ):
        raise RuntimeError("Minimal-visible observed-limit closure failed")
    return out


def build_pairwise_unions(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index(
        ["scope_key", "mass_MeV"],
        drop=False,
        verify_integrity=True,
    )
    records = []
    for union_key, spec in PAIRWISE_UNION_SPECS.items():
        low = int(spec["mass_low_MeV"])
        high = int(spec["mass_high_MeV"])
        overlap_low = int(spec["overlap_low_MeV"])
        overlap_high = int(spec["overlap_high_MeV"])
        for mass_mev in range(low, high + 1):
            if mass_mev < overlap_low:
                source_key = str(spec["left_scope_key"])
                mass_region = "left_standalone"
            elif mass_mev <= overlap_high:
                source_key = str(spec["pair_scope_key"])
                mass_region = "pairwise_overlap"
            else:
                source_key = str(spec["right_scope_key"])
                mass_region = "right_standalone"
            try:
                source = indexed.loc[(source_key, mass_mev), :]
            except KeyError as exc:
                raise RuntimeError(
                    f"Missing {source_key} row at {mass_mev} MeV"
                ) from exc
            record = source.to_dict()
            record["source_scope_key"] = record.pop("scope_key")
            record["source_scope_type"] = record.pop("scope_type")
            record["source_scope_label"] = record.pop("scope_label")
            record["source_scope_mass_low_MeV"] = record.pop(
                "scope_mass_low_MeV"
            )
            record["source_scope_mass_high_MeV"] = record.pop(
                "scope_mass_high_MeV"
            )
            record["figure_scope_key"] = union_key
            record["figure_scope_label"] = str(spec["label"])
            record["active_dataset_set"] = str(record["dataset_set"])
            record["mass_region"] = mass_region
            record["pair_overlap"] = mass_region == "pairwise_overlap"
            record["union_mass_low_MeV"] = low
            record["union_mass_high_MeV"] = high
            record["pair_overlap_low_MeV"] = overlap_low
            record["pair_overlap_high_MeV"] = overlap_high
            records.append(record)

    out = pd.DataFrame.from_records(records)
    front = [
        "figure_scope_key",
        "figure_scope_label",
        "mass_GeV",
        "mass_MeV",
        "active_dataset_set",
        "mass_region",
        "pair_overlap",
        "union_mass_low_MeV",
        "union_mass_high_MeV",
        "pair_overlap_low_MeV",
        "pair_overlap_high_MeV",
        "source_scope_key",
        "source_scope_type",
        "source_scope_label",
        "source_scope_mass_low_MeV",
        "source_scope_mass_high_MeV",
    ]
    out = out.loc[:, [*front, *[c for c in out.columns if c not in front]]]
    out = out.sort_values(
        ["figure_scope_key", "mass_MeV"]
    ).reset_index(drop=True)
    validate_pairwise_unions(out)
    return out


def validate_pairwise_unions(frame: pd.DataFrame) -> None:
    if len(frame) != N_PAIRWISE_UNION_ROWS:
        raise RuntimeError(
            f"Expected 606 pairwise-union rows, found {len(frame)}"
        )
    if frame.duplicated(["figure_scope_key", "mass_MeV"]).any():
        raise RuntimeError("Duplicate pairwise-union mass rows")
    found_counts = {
        str(key): int(value)
        for key, value in frame.groupby("figure_scope_key").size().to_dict().items()
    }
    expected_counts = {
        key: int(spec["n_rows"])
        for key, spec in PAIRWISE_UNION_SPECS.items()
    }
    if found_counts != expected_counts:
        raise RuntimeError(
            f"Pairwise-union row counts differ: {found_counts}"
        )
    for key, spec in PAIRWISE_UNION_SPECS.items():
        rows = frame.loc[
            frame["figure_scope_key"].astype(str) == key
        ].sort_values("mass_MeV")
        expected_masses = np.arange(
            int(spec["mass_low_MeV"]),
            int(spec["mass_high_MeV"]) + 1,
            dtype=int,
        )
        if not np.array_equal(
            rows["mass_MeV"].to_numpy(int),
            expected_masses,
        ):
            raise RuntimeError(f"Union mass grid is incomplete for {key}")
        overlap = rows["mass_MeV"].between(
            int(spec["overlap_low_MeV"]),
            int(spec["overlap_high_MeV"]),
        )
        expected_pair_scope = str(spec["pair_scope_key"])
        if set(rows.loc[overlap, "source_scope_key"].astype(str)) != {
            expected_pair_scope
        }:
            raise RuntimeError(f"Overlap source drift for {key}")
        if not bool(rows.loc[overlap, "pair_overlap"].astype(bool).all()):
            raise RuntimeError(f"Overlap flag drift for {key}")
        if bool(rows.loc[~overlap, "pair_overlap"].astype(bool).any()):
            raise RuntimeError(f"Standalone tail flag drift for {key}")
        if set(
            rows.loc[rows["mass_MeV"] < int(spec["overlap_low_MeV"]),
                     "source_scope_key"].astype(str)
        ) != {str(spec["left_scope_key"])}:
            raise RuntimeError(f"Left standalone source drift for {key}")
        if set(
            rows.loc[rows["mass_MeV"] > int(spec["overlap_high_MeV"]),
                     "source_scope_key"].astype(str)
        ) != {str(spec["right_scope_key"])}:
            raise RuntimeError(f"Right standalone source drift for {key}")
    ordered = frame.loc[:, MINIMAL_VISIBLE_QUANTILE_COLUMNS].to_numpy(float)
    if np.any(np.diff(ordered, axis=1) < 0.0):
        raise RuntimeError("Pairwise-union quantiles are not ordered")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 10.8,
            "axes.titlesize": 13.2,
            "axes.labelsize": 11.8,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.55,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def set_mass_ticks(ax: plt.Axes, low: float, high: float) -> None:
    span = float(high) - float(low)
    major_step = 10.0 if span <= 85.0 else 20.0
    minor_step = 0.5 * major_step
    first = math.ceil(float(low) / major_step) * major_step
    majors = np.arange(first, float(high) + 0.1, major_step)
    ax.set_xlim(float(low), float(high))
    ax.xaxis.set_major_locator(FixedLocator(majors))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_step))


def set_log_y_axis(ax: plt.Axes, y_limits: Tuple[float, float]) -> None:
    ax.set_yscale("log")
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=70,
        )
    )
    ax.yaxis.set_minor_formatter(NullFormatter())


def shared_y_limits(frames: Iterable[pd.DataFrame]) -> Tuple[float, float]:
    arrays = []
    for frame in frames:
        for column in (
            "eps2_lo2_minimal_visible",
            "eps2_hi2_minimal_visible",
            "eps2_obs_minimal_visible",
        ):
            arrays.append(frame[column].to_numpy(float))
    values = np.concatenate(arrays)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise RuntimeError("No finite positive values for shared y limits")
    return (0.82 * float(np.min(values)), 1.22 * float(np.max(values)))


def plot_limit_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    title: str,
    y_limits: Tuple[float, float],
) -> None:
    work = rows.sort_values("mass_MeV")
    x = work["mass_MeV"].to_numpy(float)
    ax.fill_between(
        x,
        work["eps2_lo2_minimal_visible"].to_numpy(float),
        work["eps2_hi2_minimal_visible"].to_numpy(float),
        color=COLORS["band2"],
        alpha=0.76,
        linewidth=0.0,
        zorder=1,
    )
    ax.fill_between(
        x,
        work["eps2_lo1_minimal_visible"].to_numpy(float),
        work["eps2_hi1_minimal_visible"].to_numpy(float),
        color=COLORS["band1"],
        alpha=0.84,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(
        x,
        work["eps2_med_minimal_visible"].to_numpy(float),
        color=COLORS["expected"],
        linewidth=1.65,
        linestyle="--",
        zorder=3,
    )
    ax.plot(
        x,
        work["eps2_obs_minimal_visible"].to_numpy(float),
        color=COLORS["observed"],
        linewidth=2.0,
        zorder=4,
    )
    low = float(np.min(x))
    high = float(np.max(x))
    if low <= DIMUON_THRESHOLD_MEV <= high:
        ax.axvline(
            DIMUON_THRESHOLD_MEV,
            color=COLORS["threshold"],
            linewidth=1.0,
            linestyle=":",
            zorder=5,
        )
    set_log_y_axis(ax, y_limits)
    set_mass_ticks(ax, low, high)
    ax.text(
        0.012,
        0.925,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="semibold",
        color="#24272C",
    )


def legend_handles() -> Sequence[Any]:
    return [
        Patch(
            facecolor=COLORS["band2"],
            alpha=0.76,
            label="Central 95% fixed-GP toy-limit interval",
        ),
        Patch(
            facecolor=COLORS["band1"],
            alpha=0.84,
            label="Central 68% fixed-GP toy-limit interval",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["expected"],
            linewidth=1.65,
            linestyle="--",
            label="Fixed-GP toy-limit median",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["observed"],
            linewidth=2.0,
            label="Observed 90% CL",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["threshold"],
            linewidth=1.0,
            linestyle=":",
            label=rf"$2m_\mu={DIMUON_THRESHOLD_MEV:.3f}$ MeV",
        ),
    ]


def contiguous_activity_segments(
    frame: pd.DataFrame,
) -> Iterable[pd.DataFrame]:
    work = frame.sort_values("mass_MeV").reset_index(drop=True)
    values = work["active_dataset_set"].astype(str)
    groups = values.ne(values.shift()).cumsum()
    for _, segment in work.groupby(groups, sort=False):
        yield segment


def plot_activity_rail(ax: plt.Axes, frame: pd.DataFrame) -> None:
    for segment in contiguous_activity_segments(frame):
        key = str(segment["active_dataset_set"].iloc[0])
        if key not in ACTIVE_COLORS:
            raise RuntimeError(f"No activity-rail color for {key}")
        x0 = float(segment["mass_MeV"].min()) - 0.5
        x1 = float(segment["mass_MeV"].max()) + 0.5
        ax.axvspan(
            x0,
            x1,
            ymin=0.08,
            ymax=0.92,
            facecolor=ACTIVE_COLORS[key],
            edgecolor="white",
            linewidth=1.0,
        )
        ax.text(
            0.5 * (x0 + x1),
            0.50,
            ACTIVE_LABELS[key],
            ha="center",
            va="center",
            transform=ax.get_xaxis_transform(),
            fontsize=8.5,
            color="#30343B",
        )
    low = float(frame["mass_MeV"].min())
    high = float(frame["mass_MeV"].max())
    ax.set_xlim(low, high)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel(
        "Active",
        rotation=0,
        ha="right",
        va="center",
        labelpad=8,
        fontsize=8.8,
    )


def save_figure(
    fig: plt.Figure,
    stem: str,
    title: str,
    subject: str,
) -> Tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{stem}.pdf"
    png = FIGURE_DIR / f"{stem}.png"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Author": "HPS-GPR v4.2 analysis",
            "Subject": subject,
        },
    )
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return pdf, png


def make_standalone_figure(
    frame: pd.DataFrame,
    y_limits: Tuple[float, float],
) -> Tuple[Path, Path]:
    fig, axes = plt.subplots(3, 1, figsize=(12.4, 9.0))
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.845,
        bottom=0.075,
        hspace=0.22,
    )
    for ax, (scope_key, panel_title) in zip(axes, STANDALONE_PANELS):
        rows = frame.loc[frame["scope_key"].astype(str) == scope_key]
        plot_limit_panel(ax, rows, panel_title, y_limits)
    fig.supylabel(
        r"90% CL upper limit on minimal-visible $\epsilon^2$",
        x=0.018,
        fontsize=12.0,
    )
    fig.supxlabel(
        r"Mass hypothesis $m_{A'}$ (MeV)",
        y=0.018,
        fontsize=12.0,
    )
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.50, 0.925),
        frameon=False,
        ncol=3,
        columnspacing=1.8,
        handlelength=2.6,
    )
    fig.suptitle(
        "Standalone HPS observed limits and 100-toy quantiles",
        y=0.985,
        fontweight="semibold",
    )
    return save_figure(
        fig,
        STANDALONE_STEM,
        "Standalone HPS observed limits and 100-toy quantiles",
        "Three standalone fixed-GP conditional limit ensembles",
    )


def make_pairwise_figure(
    unions: pd.DataFrame,
    y_limits: Tuple[float, float],
) -> Tuple[Path, Path]:
    fig = plt.figure(figsize=(12.4, 10.8))
    grid = fig.add_gridspec(
        6,
        1,
        height_ratios=(0.10, 1.0, 0.10, 1.0, 0.10, 1.0),
        hspace=0.25,
        left=0.09,
        right=0.985,
        top=0.845,
        bottom=0.065,
    )
    for index, (union_key, spec) in enumerate(
        PAIRWISE_UNION_SPECS.items()
    ):
        rows = unions.loc[
            unions["figure_scope_key"].astype(str) == union_key
        ]
        rail = fig.add_subplot(grid[2 * index])
        ax = fig.add_subplot(grid[2 * index + 1], sharex=rail)
        plot_activity_rail(rail, rows)
        plot_limit_panel(
            ax,
            rows,
            str(spec["panel_title"]),
            y_limits,
        )
    fig.supylabel(
        r"90% CL upper limit on minimal-visible $\epsilon^2$",
        x=0.018,
        fontsize=12.0,
    )
    fig.supxlabel(
        r"Mass hypothesis $m_{A'}$ (MeV)",
        y=0.015,
        fontsize=12.0,
    )
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.50, 0.925),
        frameon=False,
        ncol=3,
        columnspacing=1.8,
        handlelength=2.6,
    )
    fig.suptitle(
        r"Pairwise shared-$\epsilon^2$ HPS limits and 100-toy quantiles",
        y=0.985,
        fontweight="semibold",
    )
    return save_figure(
        fig,
        PAIRWISE_STEM,
        "Pairwise shared-epsilon-squared HPS limits and 100-toy quantiles",
        "Three pairwise fixed-GP conditional limit ensembles over union ranges",
    )


def copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"Copied asset differs from source: {destination}")


def output_record(path: Path, role: str) -> Dict[str, Any]:
    return {
        "path": repo_path(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def publish_figures(
    figures: Sequence[Tuple[Path, Path]],
) -> Sequence[Dict[str, Any]]:
    outputs = []
    for pdf, png in figures:
        outputs.append(output_record(pdf, "campaign_pdf"))
        outputs.append(output_record(png, "campaign_png"))
        note_pdf = NOTE_FIGURE_DIR / pdf.name
        note_png = NOTE_FIGURE_DIR / png.name
        delivery_pdf = DELIVERY_DIR / pdf.name
        copy_exact(pdf, note_pdf)
        copy_exact(png, note_png)
        copy_exact(pdf, delivery_pdf)
        outputs.append(output_record(note_pdf, "note_pdf"))
        outputs.append(output_record(note_png, "note_png"))
        outputs.append(output_record(delivery_pdf, "delivery_pdf"))
    return outputs


def main() -> int:
    for path in (INPUT_CSV, INPUT_VALIDATION, INPUT_PROVENANCE):
        if not path.is_file():
            raise SystemExit(f"Required input does not exist: {path}")

    validation = load_json(INPUT_VALIDATION)
    provenance = load_json(INPUT_PROVENANCE)
    validate_metadata(validation, provenance)
    frame = pd.read_csv(
        INPUT_CSV,
        dtype={
            "scope_key": str,
            "scope_type": str,
            "scope_label": str,
            "dataset_set": str,
        },
    )
    validate_input_rows(frame)
    reviewed = add_minimal_visible_columns(frame)
    unions = build_pairwise_unions(reviewed)

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    unions.to_csv(
        PAIRWISE_UNION_CSV,
        index=False,
        float_format="%.17g",
    )

    setup_style()
    y_limits = shared_y_limits((reviewed, unions))
    standalone_figures = make_standalone_figure(reviewed, y_limits)
    pairwise_figures = make_pairwise_figure(unions, y_limits)
    output_records = publish_figures(
        (standalone_figures, pairwise_figures)
    )

    checks = {
        "status": "PASS",
        "input_status_pass": True,
        "input_rows": int(len(reviewed)),
        "expected_input_rows": N_INPUT_ROWS,
        "input_scope_row_counts": expected_scope_counts(),
        "toys_per_scope_mass": N_TOYS,
        "finite_toy_limits": N_INPUT_ROWS * N_TOYS,
        "accepted_parent_draw_count": ACCEPTED_PARENT_DRAW_COUNT,
        "selected_parent_toy_indices": list(SELECTED_PARENT_INDICES),
        "raw_quantiles_finite_positive_ordered": True,
        "minimal_visible_quantiles_finite_positive_ordered": True,
        "minimal_visible_formula_closed": True,
        "pairwise_union_rows": int(len(unions)),
        "expected_pairwise_union_rows": N_PAIRWISE_UNION_ROWS,
        "pairwise_union_row_counts": {
            key: int(spec["n_rows"])
            for key, spec in PAIRWISE_UNION_SPECS.items()
        },
        "pairwise_union_source_selection_closed": True,
        "common_y_limits": [float(y_limits[0]), float(y_limits[1])],
        "figure_count": 2,
        "campaign_pdf_count": 2,
        "campaign_png_count": 2,
        "note_copy_hashes_equal": True,
        "delivery_copy_hashes_equal": True,
    }
    atomic_write_json(checks, PLOT_VALIDATION)

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "purpose": (
            "Publication figures for three standalone and three pairwise "
            "shared-epsilon-squared fixed-GP 100-toy limit ensembles"
        ),
        "inputs": {
            "bands_csv": {
                "path": repo_path(INPUT_CSV),
                "sha256": sha256(INPUT_CSV),
                "rows": int(len(reviewed)),
            },
            "validation": {
                "path": repo_path(INPUT_VALIDATION),
                "sha256": sha256(INPUT_VALIDATION),
                "status": validation["status"],
            },
            "provenance": {
                "path": repo_path(INPUT_PROVENANCE),
                "sha256": sha256(INPUT_PROVENANCE),
                "status": provenance["status"],
            },
        },
        "ensemble": {
            "n_toys_per_scope_mass": N_TOYS,
            "accepted_parent_draw_count": ACCEPTED_PARENT_DRAW_COUNT,
            "selected_parent_toy_indices": list(SELECTED_PARENT_INDICES),
            "refit_gp_on_toy": False,
            "coverage_calibrated": False,
            "scan_toy_calibrated": False,
        },
        "minimal_visible_reinterpretation": {
            "muon_mass_GeV": M_MU_GEV,
            "threshold_GeV": DIMUON_THRESHOLD_GEV,
            "threshold_MeV": DIMUON_THRESHOLD_MEV,
            "formula": (
                "factor=1 below threshold; above threshold factor="
                "1+sqrt(1-4*m_mu^2/m^2)*(1+2*m_mu^2/m^2)"
            ),
            "applied_to": list(RAW_COUPLING_COLUMNS),
        },
        "pairwise_union": {
            "path": repo_path(PAIRWISE_UNION_CSV),
            "sha256": sha256(PAIRWISE_UNION_CSV),
            "rows": int(len(unions)),
            "policy": (
                "pairwise overlap rows inside the common search interval; "
                "authoritative standalone rows outside the overlap"
            ),
            "scopes": PAIRWISE_UNION_SPECS,
        },
        "plot_content": {
            "standalone_panels": [
                "2015 100%",
                "2016 100%",
                "2021 10%",
            ],
            "pairwise_panels": [
                "2015 100% + 2016 100%",
                "2015 100% + 2021 10%",
                "2016 100% + 2021 10%",
            ],
            "observed_limit": True,
            "toy_limit_median": True,
            "central_68pct_interval": True,
            "central_95pct_interval": True,
            "pairwise_active_set_rails": True,
            "dimuon_threshold": True,
            "observed_over_median_subpanel": False,
            "information_box": False,
            "footer": False,
            "shared_log_y_limits": [float(y_limits[0]), float(y_limits[1])],
        },
        "semantic_boundary": (
            "The bands are mass-local descriptive quantiles conditional on "
            "the accepted fixed GP states. They are not coverage-calibrated "
            "intervals and do not form a coherent scan-wide toy ensemble. "
            "Pairwise overlap rows profile one shared epsilon squared; their "
            "union-range tails are the corresponding standalone likelihood."
        ),
        "validation": {
            "path": repo_path(PLOT_VALIDATION),
            "sha256": sha256(PLOT_VALIDATION),
        },
        "generator": {
            "path": repo_path(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "outputs": output_records,
    }
    atomic_write_json(payload, PLOT_PROVENANCE)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
