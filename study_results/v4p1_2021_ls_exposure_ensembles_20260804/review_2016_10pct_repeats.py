#!/usr/bin/env python3
"""Review unchanged-card 2016 10% k=12 optimizer repeats.

The two complete scans are compared row by row.  Any materially different
maximum-LML branch must be reproduced by targeted unchanged-card repeats
before it can enter the reviewed table.  Rows are never interpolated.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
DERIVED = STUDY / "derived"
CAMPAIGN = STUDY / "observed_2016_10pct_k12_wide_support"
CONFIG = STUDY / "configs/config_2016_10pct_wide_support_lsupper12.yaml"
RECOVERY_MANIFEST = DERIVED / "2016_10pct_recovery_manifest.json"

COMPLETE_ATTEMPTS = {
    "attempt_01": CAMPAIGN / "attempt_01/results_single.csv",
    "attempt_02": CAMPAIGN / "attempt_02/results_single.csv",
}
TARGETED_REPEATS = {
    117: tuple(
        CAMPAIGN / f"repairs/m117_attempt_{index:02d}/results_single.csv"
        for index in range(1, 4)
    ),
    128: tuple(
        CAMPAIGN / f"repairs/m128_attempt_{index:02d}/results_single.csv"
        for index in range(1, 4)
    ),
}

EXPECTED_MASSES_MEV = np.arange(39, 181, dtype=int)
LML_REPRODUCTION_ATOL = 2.0e-5
STATIC_NUMERIC_ATOL = 1.0e-12
MATERIAL_BRANCH_MASSES_MEV = tuple(sorted(TARGETED_REPEATS))

COMPARISON_OUTPUT = DERIVED / "2016_10pct_attempt_01_vs_02.csv"
LEDGER_OUTPUT = DERIVED / "2016_10pct_optimizer_repeat_ledger.csv"
REVIEWED_OUTPUT = DERIVED / "2016_10pct_observed_k12_reviewed.csv"
MANIFEST_OUTPUT = DERIVED / "2016_10pct_optimizer_review_manifest.json"

REQUIRED_COLUMNS = {
    "dataset",
    "mass_GeV",
    "sigma_val",
    "integral_density",
    "density_window_fully_covered",
    "A_up",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
    "extract_success",
    "const_opt",
    "ls_opt",
    "sigma_x",
    "lml",
    "ls_hi_over_sigma_x",
    "ls_opt_over_sigma_x",
}

# Values fixed by the input, card, and training geometry.  Optimizer-dependent
# predictions, likelihood values, limits, and fitted coordinates are omitted.
STATIC_COLUMNS = (
    "sigma_val",
    "blind_lo",
    "blind_hi",
    "integral_density",
    "density_nsigma",
    "density_window_lo",
    "density_window_hi",
    "density_window_width",
    "density_source_lo",
    "density_source_hi",
    "density_source_n_bins",
    "density_source_bin_width_median",
    "density_window_fully_covered",
    "cls_statistic",
    "cls_calibration",
    "signal_model",
    "global_method",
    "visibility",
    "ls_lo",
    "ls_hi",
    "ls_init",
    "sigma_x",
    "n_train",
    "n_train_low",
    "n_train_high",
    "n_full",
    "n_blind",
    "train_domain_lo",
    "train_domain_hi",
    "bin_width_median",
    "const_init",
    "const_lo",
    "const_hi",
    "optimizer_restarts",
    "ls_lo_over_sigma_x",
    "ls_hi_over_sigma_x",
    "ls_lo_over_sigma",
    "ls_hi_over_sigma",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_exact_grid(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} is missing columns: {sorted(missing)}")
    if len(frame) != len(EXPECTED_MASSES_MEV):
        raise RuntimeError(
            f"{label} has {len(frame)} rows; expected {len(EXPECTED_MASSES_MEV)}"
        )
    if set(frame["dataset"].astype(str)) != {"2016"}:
        raise RuntimeError(f"{label} contains a dataset other than 2016")
    frame = frame.sort_values("mass_GeV").reset_index(drop=True)
    masses = np.rint(frame["mass_GeV"].to_numpy(float) * 1000.0).astype(int)
    if not np.array_equal(masses, EXPECTED_MASSES_MEV):
        raise RuntimeError(f"{label} does not have the exact 39--180 MeV grid")
    if not np.allclose(
        frame["mass_GeV"].to_numpy(float),
        EXPECTED_MASSES_MEV / 1000.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(f"{label} mass values are not exact to 1e-12 GeV")
    if not frame["extract_success"].astype(bool).all():
        raise RuntimeError(f"{label} contains a failed extraction")
    if not frame["density_window_fully_covered"].astype(bool).all():
        raise RuntimeError(f"{label} has an uncovered density window")
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError(f"{label} contains non-finite numeric values")
    if not np.allclose(
        frame["ls_hi_over_sigma_x"].to_numpy(float),
        12.0,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(f"{label} does not realize the k=12 upper bound")
    return frame


def require_static_match(
    reference: pd.DataFrame, candidate: pd.DataFrame, label: str
) -> None:
    for column in STATIC_COLUMNS:
        left = reference[column]
        right = candidate[column]
        if pd.api.types.is_numeric_dtype(left):
            if not np.allclose(
                left.to_numpy(float),
                right.to_numpy(float),
                rtol=0.0,
                atol=STATIC_NUMERIC_ATOL,
                equal_nan=True,
            ):
                raise RuntimeError(f"{label}: static column changed: {column}")
        elif not left.astype(str).equals(right.astype(str)):
            raise RuntimeError(f"{label}: static column changed: {column}")


def load_single_row(path: Path, mass_mev: int, columns: list[str]) -> pd.DataFrame:
    require_file(path)
    frame = pd.read_csv(path)
    if frame.columns.tolist() != columns:
        raise RuntimeError(f"targeted repeat schema changed: {path}")
    if len(frame) != 1:
        raise RuntimeError(f"targeted repeat must contain one row: {path}")
    actual = int(round(float(frame.iloc[0]["mass_GeV"]) * 1000.0))
    if actual != mass_mev or str(frame.iloc[0]["dataset"]) != "2016":
        raise RuntimeError(
            f"targeted repeat {path} is {actual} MeV, expected {mass_mev} MeV"
        )
    if not bool(frame.iloc[0]["extract_success"]):
        raise RuntimeError(f"targeted repeat extraction failed: {path}")
    if not bool(frame.iloc[0]["density_window_fully_covered"]):
        raise RuntimeError(f"targeted repeat density is uncovered: {path}")
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError(f"targeted repeat contains non-finite values: {path}")
    return frame


def source_record(label: str, path: Path) -> dict[str, Any]:
    validation = path.parent / "validation_report.json"
    require_file(validation)
    return {
        "label": label,
        "results": relative(path),
        "results_sha256": sha256_file(path),
        "validation_report": relative(validation),
        "validation_report_sha256": sha256_file(validation),
    }


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    require_file(CONFIG)
    require_file(RECOVERY_MANIFEST)

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config.get("make_ul_bands") or config.get("do_combined_bands"):
        raise RuntimeError("review card unexpectedly enables limit bands")
    if int(config.get("cls_num_toys", -1)) != 0:
        raise RuntimeError("review card unexpectedly enables CLs toys")
    if float(config["kernel_ls_res_upper_factor_by_dataset"]["2016"]) != 12.0:
        raise RuntimeError("review card is not the 2016 k=12 card")

    recovery = json.loads(RECOVERY_MANIFEST.read_text(encoding="utf-8"))
    recovered_input = Path(config["path_2016"])
    require_file(recovered_input)
    if sha256_file(recovered_input) != recovery["recovered_root_sha256"]:
        raise RuntimeError("recovered input hash does not match recovery manifest")
    if sha256_file(CONFIG) != recovery["output_config_sha256"]:
        raise RuntimeError("config hash does not match recovery manifest")

    complete: dict[str, pd.DataFrame] = {}
    raw_columns: list[str] | None = None
    for label, path in COMPLETE_ATTEMPTS.items():
        require_file(path)
        frame = require_exact_grid(pd.read_csv(path), label)
        if raw_columns is None:
            raw_columns = frame.columns.tolist()
        elif frame.columns.tolist() != raw_columns:
            raise RuntimeError("complete unchanged-card attempt schemas differ")
        complete[label] = frame
    assert raw_columns is not None

    attempt_01 = complete["attempt_01"]
    attempt_02 = complete["attempt_02"]
    require_static_match(attempt_01, attempt_02, "attempt_01 vs attempt_02")

    comparison = pd.DataFrame(
        {
            "dataset": "2016",
            "mass_GeV": attempt_01["mass_GeV"].to_numpy(float),
            "mass_MeV": EXPECTED_MASSES_MEV,
            "attempt_01_lml": attempt_01["lml"].to_numpy(float),
            "attempt_02_lml": attempt_02["lml"].to_numpy(float),
            "delta_lml_attempt_02_minus_attempt_01": (
                attempt_02["lml"].to_numpy(float)
                - attempt_01["lml"].to_numpy(float)
            ),
            "attempt_01_ls_opt_over_sigma_x": attempt_01[
                "ls_opt_over_sigma_x"
            ].to_numpy(float),
            "attempt_02_ls_opt_over_sigma_x": attempt_02[
                "ls_opt_over_sigma_x"
            ].to_numpy(float),
            "delta_ls_opt_over_sigma_x_attempt_02_minus_attempt_01": (
                attempt_02["ls_opt_over_sigma_x"].to_numpy(float)
                - attempt_01["ls_opt_over_sigma_x"].to_numpy(float)
            ),
            "attempt_01_const_opt": attempt_01["const_opt"].to_numpy(float),
            "attempt_02_const_opt": attempt_02["const_opt"].to_numpy(float),
            "attempt_01_A_up": attempt_01["A_up"].to_numpy(float),
            "attempt_02_A_up": attempt_02["A_up"].to_numpy(float),
            "attempt_01_eps2_up": attempt_01["eps2_up"].to_numpy(float),
            "attempt_02_eps2_up": attempt_02["eps2_up"].to_numpy(float),
            "attempt_01_p0_analytic": attempt_01["p0_analytic"].to_numpy(float),
            "attempt_02_p0_analytic": attempt_02["p0_analytic"].to_numpy(float),
        }
    )
    comparison["abs_delta_lml"] = comparison[
        "delta_lml_attempt_02_minus_attempt_01"
    ].abs()
    comparison["material_branch_difference"] = (
        comparison["abs_delta_lml"] > LML_REPRODUCTION_ATOL
    )
    material_masses = tuple(
        comparison.loc[
            comparison["material_branch_difference"], "mass_MeV"
        ].astype(int)
    )
    if material_masses != MATERIAL_BRANCH_MASSES_MEV:
        raise RuntimeError(
            "material branch masses changed: "
            f"found {material_masses}, expected {MATERIAL_BRANCH_MASSES_MEV}"
        )
    comparison.to_csv(COMPARISON_OUTPUT, index=False)

    candidates_by_mass: dict[int, list[tuple[str, Path, pd.Series]]] = {
        mass: [] for mass in EXPECTED_MASSES_MEV
    }
    for label, path in COMPLETE_ATTEMPTS.items():
        frame = complete[label]
        for _, row in frame.iterrows():
            mass_mev = int(round(float(row["mass_GeV"]) * 1000.0))
            candidates_by_mass[mass_mev].append((label, path, row))

    repair_records: list[dict[str, Any]] = []
    for mass_mev, paths in TARGETED_REPEATS.items():
        reference = attempt_01.loc[
            np.rint(attempt_01["mass_GeV"] * 1000.0).astype(int) == mass_mev
        ].reset_index(drop=True)
        for index, path in enumerate(paths, 1):
            label = f"m{mass_mev:03d}_targeted_{index:02d}"
            frame = load_single_row(path, mass_mev, raw_columns)
            require_static_match(reference, frame, label)
            candidates_by_mass[mass_mev].append((label, path, frame.iloc[0]))
            repair_records.append(source_record(label, path))

    ledger_rows: list[dict[str, Any]] = []
    reviewed_rows: list[dict[str, Any]] = []
    unresolved: list[int] = []
    for mass_mev in EXPECTED_MASSES_MEV:
        candidates = candidates_by_mass[int(mass_mev)]
        selected_label, selected_path, selected_row = max(
            candidates, key=lambda item: float(item[2]["lml"])
        )
        selected_lml = float(selected_row["lml"])
        reproducing = [
            item
            for item in candidates
            if abs(float(item[2]["lml"]) - selected_lml)
            <= LML_REPRODUCTION_ATOL
        ]
        resolved = len(reproducing) >= 2
        if not resolved:
            unresolved.append(int(mass_mev))

        for label, path, row in candidates:
            delta_lml = float(row["lml"]) - selected_lml
            ledger_rows.append(
                {
                    "dataset": "2016",
                    "mass_GeV": float(mass_mev / 1000.0),
                    "mass_MeV": int(mass_mev),
                    "attempt_label": label,
                    "attempt_source": relative(path),
                    "attempt_source_sha256": sha256_file(path),
                    "lml": float(row["lml"]),
                    "selected_max_lml": selected_lml,
                    "delta_lml_from_selected_max": delta_lml,
                    "abs_delta_lml_from_selected_max": abs(delta_lml),
                    "const_opt": float(row["const_opt"]),
                    "ls_opt": float(row["ls_opt"]),
                    "ls_opt_over_sigma_x": float(row["ls_opt_over_sigma_x"]),
                    "within_lml_reproduction_tolerance": bool(
                        abs(delta_lml) <= LML_REPRODUCTION_ATOL
                    ),
                    "is_selected_maximum": bool(
                        label == selected_label and path == selected_path
                    ),
                    "selected_attempt": selected_label,
                    "selected_source": relative(selected_path),
                    "branch_multiplicity": int(len(reproducing)),
                    "reproducing_attempts": "|".join(
                        item[0] for item in reproducing
                    ),
                    "review_status": (
                        "resolved_reproduced_max_lml"
                        if resolved
                        else "unresolved_max_lml_not_reproduced"
                    ),
                    "interpolated": False,
                }
            )

        reviewed = selected_row.to_dict()
        raw_lml = float(
            attempt_01.loc[
                np.rint(attempt_01["mass_GeV"] * 1000.0).astype(int)
                == mass_mev,
                "lml",
            ].iloc[0]
        )
        reviewed.update(
            {
                "mass_MeV": int(mass_mev),
                "selected_attempt": selected_label,
                "selected_source": relative(selected_path),
                "selected_source_sha256": sha256_file(selected_path),
                "row_source": (
                    "unchanged_card_reproduced_max_lml:"
                    + relative(selected_path)
                ),
                "selected_minus_attempt_01_lml": selected_lml - raw_lml,
                "optimizer_repair_applied": bool(
                    selected_lml - raw_lml > LML_REPRODUCTION_ATOL
                ),
                "review_status": (
                    "resolved_reproduced_max_lml"
                    if resolved
                    else "unresolved_max_lml_not_reproduced"
                ),
                "branch_multiplicity": int(len(reproducing)),
                "reproducing_attempts": "|".join(
                    item[0] for item in reproducing
                ),
                "reproducing_sources": "|".join(
                    relative(item[1]) for item in reproducing
                ),
                "candidate_count": int(len(candidates)),
                "interpolated": False,
            }
        )
        reviewed_rows.append(reviewed)

    ledger = pd.DataFrame(ledger_rows).sort_values(
        ["mass_MeV", "attempt_label"]
    )
    ledger.to_csv(LEDGER_OUTPUT, index=False)
    reviewed = pd.DataFrame(reviewed_rows).sort_values("mass_GeV")
    reviewed.to_csv(REVIEWED_OUTPUT, index=False)

    if unresolved:
        raise RuntimeError(
            f"unreproduced maximum-LML branches remain at {unresolved} MeV"
        )
    if len(reviewed) != len(EXPECTED_MASSES_MEV):
        raise RuntimeError("reviewed table does not have 142 rows")
    if reviewed["interpolated"].astype(bool).any():
        raise RuntimeError("reviewed table contains interpolation")
    repair_masses = reviewed.loc[
        reviewed["optimizer_repair_applied"].astype(bool), "mass_MeV"
    ].astype(int).tolist()
    if repair_masses != list(MATERIAL_BRANCH_MASSES_MEV):
        raise RuntimeError(
            f"unexpected material repair masses: {repair_masses}"
        )

    nonmaterial = comparison.loc[
        ~comparison["material_branch_difference"]
    ].copy()
    selected_source_counts = {
        str(key): int(value)
        for key, value in reviewed["selected_attempt"].value_counts().items()
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Review two complete unchanged-card 2016 10% k=12 scans and "
            "reproduce materially different maximum-LML branches with "
            "targeted unchanged-card repeats."
        ),
        "status": "pass",
        "policy": {
            "selection": "exact maximum LML among observed candidate rows",
            "reproduction_requirement": (
                "at least two candidate rows within absolute delta LML "
                f"<= {LML_REPRODUCTION_ATOL:g} of the selected maximum"
            ),
            "lml_reproduction_atol": LML_REPRODUCTION_ATOL,
            "interpolation_allowed": False,
        },
        "card": {
            "config": relative(CONFIG),
            "config_sha256": sha256_file(CONFIG),
            "input": relative(recovered_input),
            "input_sha256": sha256_file(recovered_input),
            "upper_factor_2016": 12.0,
            "scan_mass_min_MeV": 39,
            "scan_mass_max_MeV": 180,
            "rows": int(len(EXPECTED_MASSES_MEV)),
            "toys": 0,
            "limit_bands": False,
            "attempt_02_runtime_change": (
                "CLI output-dir override only; the same config and input "
                "were used unchanged"
            ),
        },
        "recovery_manifest": {
            "path": relative(RECOVERY_MANIFEST),
            "sha256": sha256_file(RECOVERY_MANIFEST),
            "histogram_sha256": recovery["histogram_sha256"],
            "independent_archives_checked": recovery[
                "independent_archives_checked"
            ],
        },
        "complete_attempts": [
            source_record(label, path)
            for label, path in COMPLETE_ATTEMPTS.items()
        ],
        "attempt_01_vs_attempt_02": {
            "material_branch_masses_MeV": list(material_masses),
            "material_branch_count": int(len(material_masses)),
            "maximum_abs_delta_lml_all_rows": float(
                comparison["abs_delta_lml"].max()
            ),
            "maximum_abs_delta_lml_excluding_material_branches": float(
                nonmaterial["abs_delta_lml"].max()
            ),
            "maximum_abs_delta_ls_opt_over_sigma_x_excluding_material_branches": float(
                nonmaterial[
                    "delta_ls_opt_over_sigma_x_attempt_02_minus_attempt_01"
                ].abs().max()
            ),
            "comparison_table": relative(COMPARISON_OUTPUT),
            "comparison_table_sha256": sha256_file(COMPARISON_OUTPUT),
        },
        "targeted_repeats": repair_records,
        "review": {
            "all_maximum_lml_branches_reproduced": True,
            "unresolved_masses_MeV": [],
            "material_optimizer_repairs_MeV": repair_masses,
            "material_optimizer_repair_count": int(len(repair_masses)),
            "selected_attempt_counts": selected_source_counts,
            "upper_boundary_rows": int(
                (
                    reviewed["ls_opt_over_sigma_x"].to_numpy(float)
                    >= 0.999 * reviewed["ls_hi_over_sigma_x"].to_numpy(float)
                ).sum()
            ),
            "ls_opt_over_sigma_x_min": float(
                reviewed["ls_opt_over_sigma_x"].min()
            ),
            "ls_opt_over_sigma_x_median": float(
                reviewed["ls_opt_over_sigma_x"].median()
            ),
            "ls_opt_over_sigma_x_max": float(
                reviewed["ls_opt_over_sigma_x"].max()
            ),
            "reviewed_table": relative(REVIEWED_OUTPUT),
            "reviewed_table_sha256": sha256_file(REVIEWED_OUTPUT),
            "repeat_ledger": relative(LEDGER_OUTPUT),
            "repeat_ledger_sha256": sha256_file(LEDGER_OUTPUT),
            "interpolated_rows": 0,
        },
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
