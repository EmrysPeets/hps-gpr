#!/usr/bin/env python3
"""Build the v4.9.7 signal-robustness audit.

This builder is deliberately scoped to ``signal_audit/``.  It snapshots the
archived v4.2 and v4.9.5 ledgers, reconstructs fixed-hyperparameter 65 MeV
counterfactuals, and writes machine-readable tables plus publication figures.

The products are diagnostics.  In particular, all p0/Z values are local and
asymptotic; the support x kernel swaps and the combined-scope swaps are
counterfactuals, not calibrated significances or replacement result scans.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
REPO = RELEASE.parent.parent
V4P95 = RELEASE.parent / "v4p9p5_2021_gp_support_edge_optimization_20260820"

DERIVED = HERE / "derived"
FIGURES = HERE / "figures"
SNAPSHOTS = HERE / "source_snapshots"
QA = HERE / "qa"

OLD_CURVE_REL = (
    "study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/"
    "derived/individual_observed_limits_reviewed_v4p2.csv"
)
OLD_M65_REL = (
    "study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/"
    "note_figures/extractions_m065/observed_extraction_m065_fit_summary.csv"
)
OLD_PAIR_REL = (
    "study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/"
    "standalone_pairwise_bands100_fixed/ul_bands_standalone_pairwise_100.csv"
)
OLD_ALL3_REL = (
    "study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/"
    "derived/combined_bands300_reviewed_v4p2.csv"
)
NEW_CURVE = V4P95 / "derived/analysis/observed_2021_10pct_support036_300.csv"
NEW_REPAIR = V4P95 / "observed_scan/final/optimizer_repair_ledger.csv"
OLD_CARD = V4P95 / "inputs/frozen_v4p2_analysis_card.yaml"
NEW_CARD = V4P95 / "inputs/v4p9p5_observed_2021_10pct_support036_300_card.yaml"
SOURCE_ROOT = V4P95 / "inputs/source_2021_10pct.root"
V4P95_PROVENANCE = V4P95 / "PROVENANCE.md"
RUNTIME_GPR = V4P95 / "runtime_overlay/hps_gpr/gpr.py"

RAW_FINALIST_PATH = (
    "/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/"
    "study_results/finalist_k15_2021_10pct_combined100toy_20260803/"
    "derived/observed_2021_reviewed.csv"
)
RAW_FINALIST_SHA256 = "4b5d8df6e4e5f3d0cdf4bb21b19fcd5dc9f92c3fdff28d5968662ba6fcabad93"
RAW_FINALIST_REPAIR_PATH = (
    "/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/"
    "study_results/finalist_k15_2021_10pct_combined100toy_20260803/"
    "derived/optimizer_repair_ledger.csv"
)
RAW_FINALIST_REPAIR_SHA256 = "6a28e060687663717d476e79d7d7dd01ac3d6006e917a1ecfedfb478e51d74ca"

CLAIM_BOUNDARY = (
    "Diagnostic only: p0 and Z are local/asymptotic. Support x kernel and "
    "combined-scope swaps are controlled counterfactuals, not a trials-corrected "
    "significance, coverage calibration, exclusion, or proof that the feature is "
    "signal or background."
)
CLAIM_BOUNDARY_FIG = (
    "Diagnostic only: p0 and Z are local/asymptotic. Support x kernel and combined-scope swaps are controlled "
    "counterfactuals, not trials-corrected significance or coverage calibration.\n"
    "These plots are not an exclusion and do not prove that the feature is signal or background."
)

NAVY = "#264653"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GOLD = "#E9C46A"
INK = "#24323D"
LIGHT = "#E9EEF2"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_show_bytes(relative_path: str) -> bytes:
    proc = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return proc.stdout


def git_text(command: List[str]) -> str:
    return subprocess.run(
        ["git", *command], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.write_bytes(csv_bytes(frame))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def reset_generated_directories() -> None:
    for directory in (DERIVED, FIGURES, SNAPSHOTS, QA):
        if directory.parent != HERE:
            raise RuntimeError(f"Refusing to reset unexpected path: {directory}")
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)


def source_record(
    path: str, payload: bytes, role: str, retrieval: str, snapshot: str = ""
) -> Dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "role": role,
        "retrieval": retrieval,
        "snapshot": snapshot,
    }


def load_and_snapshot_sources() -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    sources: List[Dict[str, Any]] = []

    old_curve_bytes = git_show_bytes(OLD_CURVE_REL)
    old_m65_bytes = git_show_bytes(OLD_M65_REL)
    old_pair_bytes = git_show_bytes(OLD_PAIR_REL)
    old_all3_bytes = git_show_bytes(OLD_ALL3_REL)
    new_curve_bytes = NEW_CURVE.read_bytes()
    new_repair_bytes = NEW_REPAIR.read_bytes()

    old_curve_all = pd.read_csv(io.BytesIO(old_curve_bytes))
    old_curve = old_curve_all.loc[old_curve_all["dataset"].astype(str) == "2021"].copy()
    old_m65 = pd.read_csv(io.BytesIO(old_m65_bytes))
    old_pair_all = pd.read_csv(io.BytesIO(old_pair_bytes))
    old_pair = old_pair_all.loc[np.isclose(old_pair_all["mass_MeV"], 65.0)].copy()
    old_all3_all = pd.read_csv(io.BytesIO(old_all3_bytes))
    old_all3 = old_all3_all.loc[np.isclose(old_all3_all["mass_MeV"], 65.0)].copy()
    new_curve = pd.read_csv(io.BytesIO(new_curve_bytes))
    new_repair = pd.read_csv(io.BytesIO(new_repair_bytes))

    snapshot_payloads = [
        ("old_v4p2_2021_curve.csv", old_curve, OLD_CURVE_REL, old_curve_bytes,
         "accepted v4.2 2021 observed local-Z/limit curve", "git show HEAD"),
        ("old_v4p2_m65_fit_summary.csv", old_m65, OLD_M65_REL, old_m65_bytes,
         "accepted v4.2 exact 65 MeV fit summary", "git show HEAD"),
        ("old_v4p2_pairwise_m65.csv", old_pair, OLD_PAIR_REL, old_pair_bytes,
         "accepted v4.2 standalone and pairwise rows at 65 MeV", "git show HEAD"),
        ("old_v4p2_all3_m65.csv", old_all3, OLD_ALL3_REL, old_all3_bytes,
         "accepted v4.2 all-three row at 65 MeV", "git show HEAD"),
        ("new_v4p9p5_2021_curve.csv", new_curve, str(NEW_CURVE.relative_to(REPO)), new_curve_bytes,
         "v4.9.5 2021 observed local-Z/limit curve", "filesystem in isolated worktree"),
        ("new_v4p9p5_optimizer_repair_ledger.csv", new_repair,
         str(NEW_REPAIR.relative_to(REPO)), new_repair_bytes,
         "v4.9.5 observed branch-repair ledger", "filesystem in isolated worktree"),
    ]
    for name, frame, original, original_bytes, role, retrieval in snapshot_payloads:
        snapshot_path = SNAPSHOTS / name
        write_frame(frame, snapshot_path)
        sources.append(
            source_record(
                original,
                original_bytes,
                role,
                retrieval,
                str(snapshot_path.relative_to(HERE)),
            )
        )

    for path, role in [
        (OLD_CARD, "frozen v4.2 analysis card"),
        (NEW_CARD, "v4.9.5 support36 observed card"),
        (SOURCE_ROOT, "common 2021 native-10% observed ROOT source"),
        (V4P95_PROVENANCE, "v4.9.5 support-selection provenance"),
        (RUNTIME_GPR, "archived GPR implementation used for reconstruction"),
    ]:
        payload = path.read_bytes()
        sources.append(
            source_record(
                str(path.relative_to(REPO)),
                payload,
                role,
                "filesystem in isolated worktree",
            )
        )

    sources.extend(
        [
            {
                "path": RAW_FINALIST_PATH,
                "sha256": RAW_FINALIST_SHA256,
                "bytes": None,
                "role": "read-only pre-v4.2-finalization 2021 curve used for the branch-lineage check",
                "retrieval": "auditor-recorded source; values embedded below, not reread by this builder",
                "snapshot": "derived/m65_branch_lineage.csv",
            },
            {
                "path": RAW_FINALIST_REPAIR_PATH,
                "sha256": RAW_FINALIST_REPAIR_SHA256,
                "bytes": None,
                "role": "read-only pre-v4.2-finalization targeted repair ledger",
                "retrieval": "auditor-recorded source; repair masses embedded below, not reread by this builder",
                "snapshot": "derived/m65_branch_lineage.csv",
            },
        ]
    )

    return {
        "old_curve": old_curve,
        "old_m65": old_m65,
        "old_pair": old_pair,
        "old_all3": old_all3,
        "new_curve": new_curve,
        "new_repair": new_repair,
    }, sources


def require_one(frame: pd.DataFrame, mask: Iterable[bool], description: str) -> pd.Series:
    out = frame.loc[np.asarray(mask, dtype=bool)]
    if len(out) != 1:
        raise RuntimeError(f"Expected one {description} row, found {len(out)}")
    return out.iloc[0]


def build_exact_tables(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_fit = require_one(
        data["old_m65"],
        data["old_m65"]["dataset"].astype(str) == "2021",
        "old 2021 m65 fit",
    )
    old_curve = require_one(
        data["old_curve"], np.isclose(data["old_curve"]["mass_MeV"], 65.0), "old curve m65"
    )
    new = require_one(
        data["new_curve"], np.isclose(data["new_curve"]["mass_MeV"], 65.0), "new curve m65"
    )

    rows = [
        {
            "state": "old_v4p2_support40_accepted",
            "support_low_MeV": 40.0,
            "support_high_MeV": 300.0,
            "mass_MeV": 65.0,
            "sigma_mass_MeV": float(old_fit["sigma_mass_MeV"]),
            "A_hat_events": float(old_fit["standalone_Ahat_events"]),
            "sigma_A_events": float(old_fit["standalone_sigmaA_events"]),
            "A_up_events": float(old_curve["A_up"]),
            "eps2_up": float(old_curve["eps2_up"]),
            "p0_local_asymptotic": float(old_fit["standalone_p0_asymptotic"]),
            "Z_local_asymptotic": float(old_fit["standalone_Z_asymptotic"]),
            "integral_density_counts_per_GeV": float(old_fit["integral_density_counts_per_GeV"]),
            "K_events_per_eps2": float(old_fit["K_events_per_eps2"]),
            "gp_const": float(old_fit["const_fixed"]),
            "gp_length_scale_log_mass": float(old_fit["ls_fixed"]),
            "gp_log_marginal_likelihood": float(old_fit["lml_fixed_refit"]),
            "n_train": int(old_fit["n_train"]),
            "source_role": "accepted_v4p2_fit_and_curve",
        },
        {
            "state": "new_v4p9p5_support36_accepted",
            "support_low_MeV": 36.0,
            "support_high_MeV": 300.0,
            "mass_MeV": 65.0,
            "sigma_mass_MeV": float(new["mass_GeV"] * 0.0 + old_fit["sigma_mass_MeV"]),
            "A_hat_events": float(new["A_hat"]),
            "sigma_A_events": float(new["sigma_A"]),
            "A_up_events": float(new["A_up"]),
            "eps2_up": float(new["eps2_up"]),
            "p0_local_asymptotic": float(new["p0_analytic"]),
            "Z_local_asymptotic": float(new["Z_analytic"]),
            "integral_density_counts_per_GeV": float(new["integral_density"]),
            "K_events_per_eps2": float(new["A_up"] / new["eps2_up"]),
            "gp_const": float(new["const_opt"]),
            "gp_length_scale_log_mass": float(new["ls_opt"]),
            "gp_log_marginal_likelihood": float(new["lml"]),
            "n_train": 407,
            "source_role": "accepted_v4p9p5_curve",
        },
    ]
    exact = pd.DataFrame(rows)
    old, new_row = exact.iloc[0], exact.iloc[1]
    change = pd.DataFrame(
        [
            {
                "mass_MeV": 65.0,
                "delta_A_hat_events_new_minus_old": new_row.A_hat_events - old.A_hat_events,
                "relative_A_hat_change": new_row.A_hat_events / old.A_hat_events - 1.0,
                "delta_sigma_A_events_new_minus_old": new_row.sigma_A_events - old.sigma_A_events,
                "relative_sigma_A_change": new_row.sigma_A_events / old.sigma_A_events - 1.0,
                "delta_Z_local_new_minus_old": new_row.Z_local_asymptotic - old.Z_local_asymptotic,
                "p0_ratio_new_over_old": new_row.p0_local_asymptotic / old.p0_local_asymptotic,
                "delta_eps2_up_new_minus_old": new_row.eps2_up - old.eps2_up,
                "relative_eps2_up_change": new_row.eps2_up / old.eps2_up - 1.0,
                "relative_density_change": (
                    new_row.integral_density_counts_per_GeV
                    / old.integral_density_counts_per_GeV
                    - 1.0
                ),
                "relative_K_change": new_row.K_events_per_eps2 / old.K_events_per_eps2 - 1.0,
                "delta_gp_lml": (
                    new_row.gp_log_marginal_likelihood - old.gp_log_marginal_likelihood
                ),
            }
        ]
    )

    raw = {
        "state": "old_finalist_raw_support40_before_v4p2_max_lml_selection",
        "support_low_MeV": 40.0,
        "mass_MeV": 65.0,
        "A_hat_events": 28038.83257296336,
        "sigma_A_events": 6609.535649191438,
        "p0_local_asymptotic": 1.0570983155669253e-05,
        "Z_local_asymptotic": 4.252476230276821,
        "gp_const": 66.34100812269503,
        "gp_length_scale_log_mass": 0.3194735488780418,
        "gp_log_marginal_likelihood": 1648.806088737719,
        "branch_status": "m65 absent from raw finalist targeted-repair ledger",
        "source_path": RAW_FINALIST_PATH,
        "source_sha256": RAW_FINALIST_SHA256,
    }
    accepted = {
        "state": "old_v4p2_accepted_support40",
        "support_low_MeV": 40.0,
        "mass_MeV": 65.0,
        "A_hat_events": float(old_fit["standalone_Ahat_events"]),
        "sigma_A_events": float(old_fit["standalone_sigmaA_events"]),
        "p0_local_asymptotic": float(old_fit["standalone_p0_asymptotic"]),
        "Z_local_asymptotic": float(old_fit["standalone_Z_asymptotic"]),
        "gp_const": float(old_fit["const_fixed"]),
        "gp_length_scale_log_mass": float(old_fit["ls_fixed"]),
        "gp_log_marginal_likelihood": float(old_fit["lml_fixed_refit"]),
        "branch_status": f"v4p2 accepted row selected {old_curve['selected_attempt']} by max LML",
        "source_path": OLD_M65_REL,
        "source_sha256": "see source_provenance.csv",
    }
    new_lineage = {
        "state": "new_v4p9p5_support36",
        "support_low_MeV": 36.0,
        "mass_MeV": 65.0,
        "A_hat_events": float(new["A_hat"]),
        "sigma_A_events": float(new["sigma_A"]),
        "p0_local_asymptotic": float(new["p0_analytic"]),
        "Z_local_asymptotic": float(new["Z_analytic"]),
        "gp_const": float(new["const_opt"]),
        "gp_length_scale_log_mass": float(new["ls_opt"]),
        "gp_log_marginal_likelihood": float(new["lml"]),
        "branch_status": "m65 absent from v4p9p5 repair ledger; primary branch retained",
        "source_path": str(NEW_CURVE.relative_to(REPO)),
        "source_sha256": sha256_file(NEW_CURVE),
    }
    lineage = pd.DataFrame([raw, accepted, new_lineage])
    lineage["delta_A_hat_from_raw_old_events"] = lineage["A_hat_events"] - raw["A_hat_events"]
    lineage["delta_Z_from_raw_old"] = lineage["Z_local_asymptotic"] - raw["Z_local_asymptotic"]

    write_frame(exact, DERIVED / "m65_2021_exact_comparison.csv")
    write_frame(change, DERIVED / "m65_2021_exact_changes.csv")
    write_frame(lineage, DERIVED / "m65_branch_lineage.csv")
    return exact, change, lineage


def build_curve_tables(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    old = data["old_curve"].copy()
    new = data["new_curve"].copy()
    old_columns = {
        "mass_MeV": "mass_MeV",
        "mass_GeV": "mass_GeV",
        "A_up": "old_A_up_events",
        "eps2_up": "old_eps2_up",
        "p0_analytic": "old_p0_local_asymptotic",
        "Z_analytic": "old_Z_local_asymptotic",
        "const_opt": "old_gp_const",
        "ls_opt": "old_gp_length_scale_log_mass",
        "lml": "old_gp_log_marginal_likelihood",
        "selected_attempt": "old_selected_attempt",
        "optimizer_repair_applied": "old_v4p2_attempt_selection_applied",
    }
    new_columns = {
        "mass_MeV": "mass_MeV",
        "A_up": "new_A_up_events",
        "eps2_up": "new_eps2_up",
        "p0_analytic": "new_p0_local_asymptotic",
        "Z_analytic": "new_Z_local_asymptotic",
        "A_hat": "new_A_hat_events",
        "sigma_A": "new_sigma_A_events",
        "const_opt": "new_gp_const",
        "ls_opt": "new_gp_length_scale_log_mass",
        "lml": "new_gp_log_marginal_likelihood",
    }
    old_s = old[list(old_columns)].rename(columns=old_columns)
    new_s = new[list(new_columns)].rename(columns=new_columns)
    curve = old_s.merge(new_s, on="mass_MeV", how="outer", validate="one_to_one")
    curve = curve.sort_values("mass_MeV").reset_index(drop=True)
    curve["delta_Z_new_minus_old"] = (
        curve["new_Z_local_asymptotic"] - curve["old_Z_local_asymptotic"]
    )
    curve["p0_ratio_new_over_old"] = (
        curve["new_p0_local_asymptotic"] / curve["old_p0_local_asymptotic"]
    )
    curve["relative_eps2_up_change"] = curve["new_eps2_up"] / curve["old_eps2_up"] - 1.0
    write_frame(curve, DERIVED / "old_new_2021_local_z_curves.csv")

    key_masses = [51, 64, 65, 66, 78, 79, 80, 93]
    key = curve.loc[curve["mass_MeV"].isin(key_masses)].copy()
    write_frame(key, DERIVED / "key_mass_curve_comparison.csv")

    delta = curve["delta_Z_new_minus_old"].to_numpy(float)
    old_max = curve.iloc[int(np.nanargmax(curve["old_Z_local_asymptotic"]))]
    new_max = curve.iloc[int(np.nanargmax(curve["new_Z_local_asymptotic"]))]
    curve_summary = {
        "n_mass_points": int(len(curve)),
        "mass_min_MeV": float(curve.mass_MeV.min()),
        "mass_max_MeV": float(curve.mass_MeV.max()),
        "old_max": {
            "mass_MeV": float(old_max.mass_MeV),
            "Z_local_asymptotic": float(old_max.old_Z_local_asymptotic),
            "p0_local_asymptotic": float(old_max.old_p0_local_asymptotic),
        },
        "new_max": {
            "mass_MeV": float(new_max.mass_MeV),
            "Z_local_asymptotic": float(new_max.new_Z_local_asymptotic),
            "p0_local_asymptotic": float(new_max.new_p0_local_asymptotic),
        },
        "n_Z_lower": int(np.count_nonzero(delta < -1.0e-12)),
        "n_Z_equal_within_1e-12": int(np.count_nonzero(np.abs(delta) <= 1.0e-12)),
        "n_Z_higher": int(np.count_nonzero(delta > 1.0e-12)),
        "mean_delta_Z": float(np.nanmean(delta)),
        "median_delta_Z": float(np.nanmedian(delta)),
        "minimum_delta_Z": float(np.nanmin(delta)),
        "minimum_delta_Z_mass_MeV": float(curve.iloc[int(np.nanargmin(delta))].mass_MeV),
        "maximum_delta_Z": float(np.nanmax(delta)),
        "maximum_delta_Z_mass_MeV": float(curve.iloc[int(np.nanargmax(delta))].mass_MeV),
    }
    return curve, key, curve_summary


def configure_archived_runtime() -> Dict[str, Any]:
    runtime = V4P95 / "runtime_overlay"
    for path in (REPO, runtime):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)

    from hps_gpr.config import load_config
    from hps_gpr.dataset import make_datasets
    from hps_gpr.gpr import fit_gpr, predict_counts_mean_var_from_log_gpr
    from hps_gpr.io import _build_model, estimate_background_for_dataset
    from hps_gpr.statistics import fit_A_profiled_gaussian, p0_profiled_gaussian_LRT
    from hps_gpr.template import build_window_template_from_full, cls_limit_for_template
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF

    return locals()


def reconstruct_counterfactuals(
    exact: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    api = configure_archived_runtime()
    load_config = api["load_config"]
    make_datasets = api["make_datasets"]
    estimate_background_for_dataset = api["estimate_background_for_dataset"]
    build_window_template_from_full = api["build_window_template_from_full"]
    fit_A_profiled_gaussian = api["fit_A_profiled_gaussian"]
    p0_profiled_gaussian_LRT = api["p0_profiled_gaussian_LRT"]
    cls_limit_for_template = api["cls_limit_for_template"]
    ConstantKernel = api["ConstantKernel"]
    RBF = api["RBF"]

    states = {
        "old_v4p2": {
            "const": float(exact.iloc[0].gp_const),
            "ls": float(exact.iloc[0].gp_length_scale_log_mass),
        },
        "new_v4p9p5": {
            "const": float(exact.iloc[1].gp_const),
            "ls": float(exact.iloc[1].gp_length_scale_log_mass),
        },
    }
    geometries = {
        "support40_old_rebin_phase": 0.040,
        "support36_shifted_rebin_phase": 0.036,
    }

    rows: List[Dict[str, Any]] = []
    predictions: Dict[Tuple[str, str], Any] = {}
    for geometry, support_low in geometries.items():
        cfg = load_config(str(NEW_CARD))
        cfg.path_2021 = str(SOURCE_ROOT)
        cfg.data_range_2021 = (support_low, 0.300)
        cfg.enable_2015 = False
        cfg.enable_2016 = False
        cfg.enable_2021 = True
        ds = make_datasets(cfg)["2021"]
        for state_name, state in states.items():
            kernel = ConstantKernel(
                state["const"], constant_value_bounds="fixed"
            ) * RBF(state["ls"], length_scale_bounds="fixed")
            pred = estimate_background_for_dataset(
                ds,
                0.065,
                cfg,
                kernel=kernel,
                optimize=False,
                restarts=0,
            )
            template, _ = build_window_template_from_full(
                pred.edges_full,
                pred.blind_mask,
                0.065,
                pred.sigma_val,
                config=cfg,
            )
            fit = fit_A_profiled_gaussian(
                pred.obs, pred.mu, pred.cov, template, allow_negative=True
            )
            p0, z_local, q0, _ = p0_profiled_gaussian_LRT(
                pred.obs, pred.mu, pred.cov, template
            )
            eps2_up, a_up = cls_limit_for_template(
                pred.obs,
                pred.mu,
                pred.cov,
                template,
                ds=ds,
                mass=0.065,
                integral_density=pred.integral_density,
                alpha=0.1,
                mode="asymptotic",
                use_eps2=True,
            )
            rows.append(
                {
                    "mass_MeV": 65.0,
                    "geometry": geometry,
                    "support_low_MeV": support_low * 1000.0,
                    "support_high_requested_MeV": 300.0,
                    "effective_domain_low_MeV": pred.train_domain_lo * 1000.0,
                    "effective_domain_high_MeV": pred.train_domain_hi * 1000.0,
                    "kernel_state": state_name,
                    "kernel_const_fixed": state["const"],
                    "kernel_length_scale_log_mass_fixed": state["ls"],
                    "A_hat_events": float(fit["A_hat"]),
                    "sigma_A_events": float(fit["sigma_A"]),
                    "A_up_events_reconstructed": float(a_up),
                    "eps2_up_reconstructed": float(eps2_up),
                    "p0_local_asymptotic": float(p0),
                    "Z_local_asymptotic": float(z_local),
                    "q0_local_asymptotic": float(q0),
                    "gp_log_marginal_likelihood_fixed_state": float(pred.lml),
                    "gp_sum_over_actual_fit_window_counts": float(np.sum(pred.mu)),
                    "observed_sum_over_actual_fit_window_counts": float(np.sum(pred.obs)),
                    "template_fraction_in_actual_fit_window": float(np.sum(template)),
                    "actual_fit_window_low_MeV": float(pred.edges[0] * 1000.0),
                    "actual_fit_window_high_MeV": float(pred.edges[-1] * 1000.0),
                    "n_train": int(pred.n_train),
                    "n_train_low": int(pred.n_train_low),
                    "n_train_high": int(pred.n_train_high),
                    "n_full": int(pred.n_full),
                    "n_blind": int(pred.n_blind),
                    "coarse_bin_width_MeV": float(pred.bin_width_median * 1000.0),
                    "integral_density_counts_per_GeV": float(pred.integral_density),
                    "method": "fixed archived GP hyperparameters; no optimization; profiled Poisson+Gaussian local LRT",
                    "claim_status": "controlled single-mass diagnostic",
                }
            )
            predictions[(geometry, state_name)] = (cfg, ds, pred)

    counter = pd.DataFrame(rows)
    write_frame(counter, DERIVED / "m65_support_kernel_counterfactual.csv")

    # Direct common-grid predictions isolate the GP continuum from the different
    # coarse-bin phase.  The GP target is counts per 0.625 MeV rebinned bin.
    _build_model = api["_build_model"]
    fit_gpr = api["fit_gpr"]
    predict_counts_mean_var_from_log_gpr = api["predict_counts_mean_var_from_log_gpr"]
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    sigma_mass = float(exact.iloc[0].sigma_mass_MeV) / 1000.0
    blind_low = 0.065 - 2.25 * sigma_mass
    blind_high = 0.065 + 2.25 * sigma_mass
    common_x = np.linspace(blind_low, blind_high, 10001)
    for label, geometry, state_name in [
        ("old_support40_old_kernel", "support40_old_rebin_phase", "old_v4p2"),
        ("new_support36_new_kernel", "support36_shifted_rebin_phase", "new_v4p9p5"),
    ]:
        cfg, ds, pred = predictions[(geometry, state_name)]
        state = states[state_name]
        kernel = ConstantKernel(
            state["const"], constant_value_bounds="fixed"
        ) * RBF(state["ls"], length_scale_bounds="fixed")
        model = _build_model(ds, pred.blind, cfg.neighborhood_rebin, cfg, mass=0.065)
        x_train_all = np.asarray(model.histogram.axes[0].centers, float)
        y_train_all = np.asarray(model.histogram.values(), float)
        mask = (x_train_all < blind_low) | (x_train_all > blind_high)
        fitted = fit_gpr(
            x_train_all[mask],
            y_train_all[mask],
            cfg,
            restarts=0,
            kernel=kernel,
            optimize=False,
        )
        mean, variance = predict_counts_mean_var_from_log_gpr(fitted, common_x, cfg)
        curves[label] = (mean, np.sqrt(np.clip(variance, 0.0, None)), x_train_all, pred.bin_width_median)

    old_mean, old_std, _, old_bin_width = curves["old_support40_old_kernel"]
    new_mean, new_std, _, new_bin_width = curves["new_support36_new_kernel"]
    common = pd.DataFrame(
        {
            "mass_MeV": common_x * 1000.0,
            "old_gp_mean_counts_per_0p625MeV": old_mean,
            "old_gp_pointwise_std_counts_per_0p625MeV": old_std,
            "new_gp_mean_counts_per_0p625MeV": new_mean,
            "new_gp_pointwise_std_counts_per_0p625MeV": new_std,
            "delta_gp_mean_new_minus_old_counts_per_0p625MeV": new_mean - old_mean,
        }
    )
    write_frame(common, DERIVED / "m65_gp_common_grid.csv")

    center_idx = int(np.argmin(np.abs(common_x - 0.065)))
    old_integral = float(np.trapezoid(old_mean, common_x) / old_bin_width)
    new_integral = float(np.trapezoid(new_mean, common_x) / new_bin_width)
    gp_summary = {
        "common_grid_method": (
            "direct fixed-state GP prediction on 10001 common points over the nominal "
            "65 MeV +/-2.25 sigma window; integral divides by the 0.625 MeV target-bin width"
        ),
        "nominal_window_low_MeV": blind_low * 1000.0,
        "nominal_window_high_MeV": blind_high * 1000.0,
        "old_gp_mean_at_65_counts_per_0p625MeV": float(old_mean[center_idx]),
        "old_gp_pointwise_std_at_65_counts_per_0p625MeV": float(old_std[center_idx]),
        "new_gp_mean_at_65_counts_per_0p625MeV": float(new_mean[center_idx]),
        "new_gp_pointwise_std_at_65_counts_per_0p625MeV": float(new_std[center_idx]),
        "delta_gp_mean_at_65_counts_per_0p625MeV": float(new_mean[center_idx] - old_mean[center_idx]),
        "old_gp_integral_common_window_counts": old_integral,
        "new_gp_integral_common_window_counts": new_integral,
        "delta_gp_integral_common_window_counts": new_integral - old_integral,
        "relative_gp_integral_change": new_integral / old_integral - 1.0,
    }
    return counter, common, gp_summary


def build_scope_hybrid_table(exact: pd.DataFrame) -> pd.DataFrame:
    old = exact.iloc[0]
    new = exact.iloc[1]
    k_old = float(old.K_events_per_eps2)
    k_new = float(new.K_events_per_eps2)
    rows = [
        {
            "scope": "2021 only",
            "dataset_set": "2021",
            "state": "old_v4p2",
            "eps2_hat": float(old.A_hat_events / k_old),
            "sigma_eps2": float(old.sigma_A_events / k_old),
            "eps2_up": float(old.eps2_up),
            "p0_local_asymptotic": float(old.p0_local_asymptotic),
            "Z_local_asymptotic": float(old.Z_local_asymptotic),
        },
        {
            "scope": "2021 only",
            "dataset_set": "2021",
            "state": "swap_new_2021",
            "eps2_hat": float(new.A_hat_events / k_new),
            "sigma_eps2": float(new.sigma_A_events / k_new),
            "eps2_up": float(new.eps2_up),
            "p0_local_asymptotic": float(new.p0_local_asymptotic),
            "Z_local_asymptotic": float(new.Z_local_asymptotic),
        },
        {
            "scope": "2015+2021",
            "dataset_set": "2015+2021",
            "state": "old_v4p2",
            "eps2_hat": 9.645755868e-6,
            "sigma_eps2": 2.072357557e-6,
            "eps2_up": 1.2303215629698661e-5,
            "p0_local_asymptotic": 1.6080354508547038e-6,
            "Z_local_asymptotic": 4.6565148515070209,
        },
        {
            "scope": "2015+2021",
            "dataset_set": "2015+2021",
            "state": "swap_new_2021",
            "eps2_hat": 6.422690742e-6,
            "sigma_eps2": 2.229235754e-6,
            "eps2_up": 9.265793796e-6,
            "p0_local_asymptotic": 0.0019835200,
            "Z_local_asymptotic": 2.880770800,
        },
        {
            "scope": "2016+2021",
            "dataset_set": "2016+2021",
            "state": "old_v4p2",
            "eps2_hat": 5.952058025e-6,
            "sigma_eps2": 1.632513367e-6,
            "eps2_up": 8.045013697208906e-6,
            "p0_local_asymptotic": 0.00013257514644955703,
            "Z_local_asymptotic": 3.647164394268538,
        },
        {
            "scope": "2016+2021",
            "dataset_set": "2016+2021",
            "state": "swap_new_2021",
            "eps2_hat": 3.721580260e-6,
            "sigma_eps2": 1.706005515e-6,
            "eps2_up": 5.925159583e-6,
            "p0_local_asymptotic": 0.0145408957,
            "Z_local_asymptotic": 2.182376038,
        },
        {
            "scope": "all three",
            "dataset_set": "2015+2016+2021",
            "state": "old_v4p2",
            "eps2_hat": 6.4270617905e-6,
            "sigma_eps2": 1.6100905505e-6,
            "eps2_up": 8.47828876910139e-6,
            "p0_local_asymptotic": 3.259182521304132e-5,
            "Z_local_asymptotic": 3.993214141165979,
        },
        {
            "scope": "all three",
            "dataset_set": "2015+2016+2021",
            "state": "swap_new_2021",
            "eps2_hat": 4.305764494e-6,
            "sigma_eps2": 1.680456046e-6,
            "eps2_up": 6.4715312393e-6,
            "p0_local_asymptotic": 0.0052230991,
            "Z_local_asymptotic": 2.560698457,
        },
    ]
    frame = pd.DataFrame(rows)
    frame["mass_MeV"] = 65.0
    frame["value_origin"] = frame.apply(
        lambda row: (
            "accepted_v4p2_archived_ledger"
            if row["state"] == "old_v4p2"
            else (
                "accepted_v4p9p5_individual_2021_ledger"
                if row["scope"] == "2021 only"
                else "audited_2026-09-02_fixed-block_profile-likelihood_reconstruction"
            )
        ),
        axis=1,
    )
    frame["builder_treatment"] = frame.apply(
        lambda row: (
            "parsed_or_derived_from_archived_ledger"
            if row["state"] == "old_v4p2" or row["scope"] == "2021 only"
            else "audited_constant_carried_into_self-contained_table"
        ),
        axis=1,
    )
    frame["swap_definition"] = (
        "replace only the 2021 m65 likelihood block with the support36 fixed-state block; retain old 2015/2016 blocks"
    )
    frame["official_status"] = (
        "single-mass controlled diagnostic; swap_new_2021 is not an official combined scan"
    )
    write_frame(frame, DERIVED / "m65_scope_hybrid_diagnostic.csv")
    return frame


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.edgecolor": "#59636B",
            "axes.linewidth": 0.8,
            "grid.color": "#CCD5DC",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str, subject: str) -> None:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    fig.savefig(
        png,
        dpi=240,
        bbox_inches="tight",
        metadata={"Software": "build_signal_robustness_audit.py", "Description": subject},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": stem.replace("_", " ").title(),
            "Author": "HPS GPR v4.9.7 signal audit",
            "Subject": subject,
            "Keywords": "diagnostic local asymptotic GP support robustness",
            "Creator": "build_signal_robustness_audit.py",
        },
    )
    plt.close(fig)


def plot_local_z(curve: pd.DataFrame, curve_summary: Dict[str, Any]) -> None:
    setup_plot_style()
    x = curve.mass_MeV.to_numpy(float)
    old = curve.old_Z_local_asymptotic.to_numpy(float)
    new = curve.new_Z_local_asymptotic.to_numpy(float)
    delta = new - old

    fig, (ax, axd) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.08},
    )
    ax.plot(x, old, color=NAVY, lw=1.8, label="v4.2: GP support 40–300 MeV")
    ax.plot(x, new, color=ORANGE, lw=1.8, label="v4.9.5: GP support 36–300 MeV")
    for level in (1, 2, 3, 4):
        ax.axhline(level, color="#AAB5BD", lw=0.55, ls=(0, (3, 4)), zorder=0)
    ax.set_ylabel("Local asymptotic Z")
    ax.set_ylim(-0.08, max(4.85, float(np.nanmax(old)) + 0.3))
    ax.grid(axis="x")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("2021 10% observed local-Z scan: support robustness", loc="left", color=INK, pad=14)
    ax.text(
        0.0,
        1.015,
        "Same observed ROOT input, resolution, signal normalization, likelihood, and 201-point mass grid",
        transform=ax.transAxes,
        fontsize=9.4,
        color="#52606A",
        va="bottom",
    )

    def annotate(mass: float, values: Tuple[float, float], text: str, xytext: Tuple[float, float]) -> None:
        ax.scatter([mass, mass], values, s=27, color=[NAVY, ORANGE], zorder=5, edgecolor="white", lw=0.5)
        ax.annotate(
            text,
            xy=(mass, max(values)),
            xytext=xytext,
            textcoords="data",
            arrowprops={"arrowstyle": "-", "color": "#6B747B", "lw": 0.8},
            fontsize=9,
            color=INK,
            ha="left",
        )

    r65 = curve.loc[np.isclose(curve.mass_MeV, 65.0)].iloc[0]
    r78 = curve.loc[np.isclose(curve.mass_MeV, 78.0)].iloc[0]
    annotate(
        65.0,
        (r65.old_Z_local_asymptotic, r65.new_Z_local_asymptotic),
        f"65 MeV: {r65.old_Z_local_asymptotic:.2f} → {r65.new_Z_local_asymptotic:.2f}",
        (84, 4.35),
    )
    annotate(
        78.0,
        (r78.old_Z_local_asymptotic, r78.new_Z_local_asymptotic),
        f"new scan maximum: Z={r78.new_Z_local_asymptotic:.2f}",
        (105, 3.25),
    )

    axd.axhline(0.0, color="#59636B", lw=0.8)
    axd.fill_between(x, delta, 0, where=delta <= 0, color=ORANGE, alpha=0.35, interpolate=True)
    axd.fill_between(x, delta, 0, where=delta > 0, color=TEAL, alpha=0.35, interpolate=True)
    axd.plot(x, delta, color=INK, lw=1.0)
    axd.axvline(65, color="#59636B", lw=0.7, ls=(0, (3, 3)))
    axd.set_xlabel("Mass hypothesis [MeV]")
    axd.set_ylabel("ΔZ (new − old)")
    axd.set_xlim(50, 250)
    axd.set_ylim(min(-3.0, float(np.nanmin(delta)) - 0.2), max(1.6, float(np.nanmax(delta)) + 0.2))
    axd.grid(axis="x")
    axd.text(
        0.99,
        0.06,
        (
            f"{curve_summary['n_Z_lower']} lower • {curve_summary['n_Z_equal_within_1e-12']} equal "
            f"• {curve_summary['n_Z_higher']} higher; median ΔZ={curve_summary['median_delta_Z']:.2f}"
        ),
        transform=axd.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        color="#52606A",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.14)
    fig.text(0.10, 0.022, CLAIM_BOUNDARY_FIG, fontsize=7.8, color="#5F6B73", ha="left", linespacing=1.25)
    save_figure(
        fig,
        "old_new_2021_local_z_curves",
        "Archived 2021 old/new local asymptotic Z curves with diagnostic claim boundary.",
    )


def plot_mechanism(
    exact: pd.DataFrame, counter: pd.DataFrame, common: pd.DataFrame, gp_summary: Dict[str, Any]
) -> None:
    setup_plot_style()
    fig = plt.figure(figsize=(10.0, 8.0))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05], hspace=0.38, wspace=0.40)
    axa = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[0, 1])
    axg = fig.add_subplot(gs[1, :])

    old, new = exact.iloc[0], exact.iloc[1]
    y = np.array([1.0, 0.0])
    x = np.array([old.A_hat_events, new.A_hat_events]) / 1000.0
    xe = np.array([old.sigma_A_events, new.sigma_A_events]) / 1000.0
    colors = [NAVY, ORANGE]
    for yi, xi, ei, color in zip(y, x, xe, colors):
        axa.errorbar(xi, yi, xerr=ei, fmt="o", ms=8, capsize=4, lw=1.8, color=color)
    axa.axvline(0, color="#727C84", lw=0.7)
    axa.set_yticks(y, ["v4.2 support40", "v4.9.5 support36"])
    axa.set_xlabel(r"Fitted signal amplitude $\hat{A}$ [10³ events]")
    axa.set_title("A  65 MeV fitted amplitude", loc="left", color=INK)
    axa.grid(axis="x")
    axa.set_ylim(-0.65, 1.65)
    axa.set_xlim(-2.0, 43.0)
    axa.text(x[0] + xe[0] + 0.8, y[0], f"Z={old.Z_local_asymptotic:.2f}", va="center", color=NAVY)
    axa.text(x[1] + xe[1] + 0.8, y[1], f"Z={new.Z_local_asymptotic:.2f}", va="center", color=ORANGE)

    geometry_order = ["support40_old_rebin_phase", "support36_shifted_rebin_phase"]
    kernel_order = ["old_v4p2", "new_v4p9p5"]
    matrix = np.zeros((2, 2))
    lml = np.zeros((2, 2))
    for i, geometry in enumerate(geometry_order):
        for j, state in enumerate(kernel_order):
            row = counter.loc[(counter.geometry == geometry) & (counter.kernel_state == state)].iloc[0]
            matrix[i, j] = row.Z_local_asymptotic
            lml[i, j] = row.gp_log_marginal_likelihood_fixed_state
    cmap = LinearSegmentedColormap.from_list("audit", ["#F3F6F8", GOLD, ORANGE, NAVY])
    im = axh.imshow(matrix, cmap=cmap, vmin=0, vmax=5, aspect="auto")
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > 3.6 else INK
            axh.text(j, i, f"Z={matrix[i, j]:.3f}\nLML={lml[i, j]:.1f}", ha="center", va="center", color=color, fontsize=9)
    axh.set_xticks([0, 1], ["old kernel state", "new kernel state"])
    axh.set_yticks([0, 1], ["40–300 MeV", "36–300 MeV"])
    axh.set_title("B  Fixed-state counterfactual", loc="left", color=INK)
    cbar = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)
    cbar.set_label("Local asymptotic Z", fontsize=9)

    gx = common.mass_MeV.to_numpy(float)
    gd = common.delta_gp_mean_new_minus_old_counts_per_0p625MeV.to_numpy(float)
    axg.axhline(0, color="#59636B", lw=0.8)
    axg.fill_between(gx, gd, 0, color=TEAL, alpha=0.28)
    axg.plot(gx, gd, color=TEAL, lw=2.0)
    axg.axvline(65, color="#59636B", lw=0.7, ls=(0, (3, 3)))
    axg.scatter([65], [gp_summary["delta_gp_mean_at_65_counts_per_0p625MeV"]], color=ORANGE, s=35, zorder=4)
    axg.set_xlabel("Mass [MeV] in the common nominal ±2.25σ window")
    axg.set_ylabel("GP mean shift [events / 0.625 MeV]\n(new diagonal − old diagonal)")
    axg.set_title("C  Common-grid GP continuum shift", loc="left", color=INK)
    axg.grid(axis="both")
    axg.text(
        0.015,
        0.95,
        (
            f"At 65 MeV: +{gp_summary['delta_gp_mean_at_65_counts_per_0p625MeV']:,.0f} events/bin\n"
            f"Integrated over common window: +{gp_summary['delta_gp_integral_common_window_counts']:,.0f} "
            f"({100*gp_summary['relative_gp_integral_change']:.3f}%)"
        ),
        transform=axg.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#C3CDD4", "alpha": 0.92},
    )
    axg.text(
        0.985,
        0.05,
        "support40 fit window: 60.000–70.000 MeV (16 bins)\n"
        "support36 fit window: 60.375–69.750 MeV (15 bins)\n"
        "The 4 MeV endpoint shift also moves the 5-bin rebin phase by 0.25 MeV.",
        transform=axg.transAxes,
        va="bottom",
        ha="right",
        fontsize=8.6,
        color="#52606A",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
    )
    fig.suptitle("Why the 65 MeV local excess changes under GP-support optimization", x=0.08, ha="left", y=0.97, fontsize=15, color=INK)
    fig.text(
        0.08,
        0.925,
        "The diagonal comparison mixes support geometry and the support-dependent maximum-LML kernel state; the 2×2 swap exposes both.",
        fontsize=9.5,
        color="#52606A",
    )
    fig.subplots_adjust(left=0.10, right=0.95, top=0.86, bottom=0.14)
    fig.text(0.10, 0.022, CLAIM_BOUNDARY_FIG, fontsize=7.8, color="#5F6B73", ha="left", linespacing=1.25)
    save_figure(
        fig,
        "m65_support_kernel_mechanism",
        "65 MeV exact amplitude comparison, fixed-state support-by-kernel diagnostic, and common-grid GP mean shift.",
    )


def plot_scope_hybrid(scope: pd.DataFrame) -> None:
    setup_plot_style()
    order = ["2021 only", "2015+2021", "2016+2021", "all three"]
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    y = np.arange(len(order))[::-1]
    for yi, scope_name in zip(y, order):
        part = scope.loc[scope.scope == scope_name]
        old = float(part.loc[part.state == "old_v4p2", "Z_local_asymptotic"].iloc[0])
        hybrid = float(part.loc[part.state == "swap_new_2021", "Z_local_asymptotic"].iloc[0])
        ax.plot([hybrid, old], [yi, yi], color="#AAB5BD", lw=2.0, zorder=1)
        ax.scatter(old, yi, color=NAVY, s=62, label="v4.2 blocks" if yi == y[0] else None, zorder=3)
        ax.scatter(hybrid, yi, color=ORANGE, s=62, marker="D", label="swap in v4.9.5 2021 block" if yi == y[0] else None, zorder=3)
        ax.text(old + 0.09, yi + 0.11, f"{old:.2f}", color=NAVY, fontsize=9)
        hybrid_label_y = yi + 0.11 if yi == 0 else yi - 0.23
        ax.text(hybrid + 0.09, hybrid_label_y, f"{hybrid:.2f}", color=ORANGE, fontsize=9)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 5.1)
    ax.set_xlabel("Local asymptotic Z at 65 MeV")
    ax.grid(axis="x")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.015, 0.50))
    ax.set_title("65 MeV scope check: the shift is carried by the 2021 likelihood block", loc="left", color=INK, pad=16)
    ax.text(
        0.0,
        1.015,
        "2015 and 2016 blocks held at their v4.2 states; only the 2021 block is replaced",
        transform=ax.transAxes,
        fontsize=9.4,
        color="#52606A",
        va="bottom",
    )
    fig.subplots_adjust(left=0.20, right=0.78, top=0.84, bottom=0.22)
    fig.text(
        0.20,
        0.07,
        "Orange points are controlled single-mass hybrids, not an official combined scan.\n" + CLAIM_BOUNDARY_FIG,
        fontsize=7.8,
        color="#5F6B73",
        ha="left",
        linespacing=1.25,
    )
    save_figure(
        fig,
        "m65_scope_hybrid_diagnostic",
        "Single-mass scope diagnostic holding 2015/2016 fixed while replacing the 2021 likelihood block.",
    )


def render_pdf_qa() -> Dict[str, Any]:
    pdftoppm = shutil.which("pdftoppm")
    pdfinfo = shutil.which("pdfinfo")
    records: List[Dict[str, Any]] = []
    for pdf in sorted(FIGURES.glob("*.pdf")):
        record: Dict[str, Any] = {
            "pdf": str(pdf.relative_to(HERE)),
            "pdf_sha256": sha256_file(pdf),
            "rendered": False,
            "render_path": "",
            "pdfinfo": "",
        }
        if pdftoppm:
            out_stem = QA / f"rendered_{pdf.stem}_page1"
            subprocess.run(
                [pdftoppm, "-f", "1", "-singlefile", "-r", "150", "-png", str(pdf), str(out_stem)],
                check=True,
                capture_output=True,
            )
            render_path = out_stem.with_suffix(".png")
            record["rendered"] = render_path.exists() and render_path.stat().st_size > 0
            record["render_path"] = str(render_path.relative_to(HERE))
            record["render_sha256"] = sha256_file(render_path)
        if pdfinfo:
            record["pdfinfo"] = subprocess.run(
                [pdfinfo, str(pdf)], check=True, capture_output=True, text=True
            ).stdout
        records.append(record)
    payload = {
        "pdftoppm": pdftoppm,
        "pdfinfo": pdfinfo,
        "records": records,
        "all_rendered": bool(records) and all(r["rendered"] for r in records),
    }
    write_json(payload, QA / "pdf_render_qa.json")
    return payload


def build_validation(
    data: Dict[str, pd.DataFrame],
    exact: pd.DataFrame,
    counter: pd.DataFrame,
    curve: pd.DataFrame,
    curve_summary: Dict[str, Any],
    render_qa: Dict[str, Any],
) -> Dict[str, Any]:
    root_hash = sha256_file(SOURCE_ROOT)
    checks = [
        ("old_curve_has_201_points", len(data["old_curve"]) == 201, len(data["old_curve"])),
        ("new_curve_has_201_points", len(data["new_curve"]) == 201, len(data["new_curve"])),
        ("mass_grids_identical", np.array_equal(data["old_curve"].mass_MeV.to_numpy(), data["new_curve"].mass_MeV.to_numpy()), "50--250 MeV"),
        ("old_max_is_65_MeV", np.isclose(curve_summary["old_max"]["mass_MeV"], 65.0), curve_summary["old_max"]),
        ("new_max_is_78_MeV", np.isclose(curve_summary["new_max"]["mass_MeV"], 78.0), curve_summary["new_max"]),
        ("same_expected_2021_root_hash", root_hash == "3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4", root_hash),
        ("normalization_density_unchanged_at_65", np.isclose(exact.iloc[0].integral_density_counts_per_GeV, exact.iloc[1].integral_density_counts_per_GeV, rtol=0, atol=1e-3), exact.integral_density_counts_per_GeV.tolist()),
        ("normalization_K_unchanged_at_65", np.isclose(exact.iloc[0].K_events_per_eps2, exact.iloc[1].K_events_per_eps2, rtol=1e-13), exact.K_events_per_eps2.tolist()),
        ("m65_Z_decreases", exact.iloc[1].Z_local_asymptotic < exact.iloc[0].Z_local_asymptotic, exact.Z_local_asymptotic.tolist()),
        ("four_counterfactual_cells", len(counter) == 4, len(counter)),
        ("new_repair_ledger_excludes_m65", not np.any(np.isclose(data["new_repair"].mass_MeV, 65.0)), sorted(set(data["new_repair"].mass_MeV.tolist()))),
        ("all_pdf_pages_rendered", bool(render_qa["all_rendered"]), len(render_qa["records"])),
        ("curve_table_complete", len(curve) == 201 and not curve[["old_Z_local_asymptotic", "new_Z_local_asymptotic"]].isna().any().any(), len(curve)),
    ]
    records = [{"name": name, "pass": bool(ok), "detail": detail} for name, ok, detail in checks]
    payload = {
        "overall_pass": all(r["pass"] for r in records),
        "n_checks": len(records),
        "n_pass": sum(r["pass"] for r in records),
        "checks": records,
    }
    write_json(payload, QA / "semantic_validation.json")
    return payload


def build_summary_json(
    exact: pd.DataFrame,
    change: pd.DataFrame,
    lineage: pd.DataFrame,
    curve_summary: Dict[str, Any],
    counter: pd.DataFrame,
    gp_summary: Dict[str, Any],
    scope: pd.DataFrame,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    old = exact.iloc[0]
    new = exact.iloc[1]
    c = change.iloc[0]
    support40_old = counter.loc[
        (counter.geometry == "support40_old_rebin_phase") & (counter.kernel_state == "old_v4p2")
    ].iloc[0]
    support40_new = counter.loc[
        (counter.geometry == "support40_old_rebin_phase") & (counter.kernel_state == "new_v4p9p5")
    ].iloc[0]
    support36_old = counter.loc[
        (counter.geometry == "support36_shifted_rebin_phase") & (counter.kernel_state == "old_v4p2")
    ].iloc[0]
    support36_new = counter.loc[
        (counter.geometry == "support36_shifted_rebin_phase") & (counter.kernel_state == "new_v4p9p5")
    ].iloc[0]
    summary = {
        "schema_version": "signal_robustness_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release": RELEASE.name,
        "git_head": git_text(["rev-parse", "HEAD"]),
        "claim_boundary": {
            "local_p0_Z": "local and asymptotic; no look-elsewhere calibration",
            "support_kernel_cells": "fixed-state single-mass diagnostics; not a model-selection calibration",
            "scope_hybrids": "single-mass controlled swaps; not official combined scans",
            "toys": "support-selection toys are conditional diagnostics, not direct coverage",
            "physics": "the audit cannot classify the feature as signal or background",
        },
        "m65_exact": {
            "old": old.to_dict(),
            "new": new.to_dict(),
            "change": c.to_dict(),
        },
        "curve_summary": curve_summary,
        "mechanism": {
            "support40_old_kernel_Z": float(support40_old.Z_local_asymptotic),
            "support40_new_kernel_Z": float(support40_new.Z_local_asymptotic),
            "support36_old_kernel_Z": float(support36_old.Z_local_asymptotic),
            "support36_new_kernel_Z": float(support36_new.Z_local_asymptotic),
            "support40_preferred_kernel_by_LML": "old_v4p2",
            "support36_preferred_kernel_by_LML": "new_v4p9p5",
            "common_grid_gp": gp_summary,
            "defensible_statement": (
                "At 65 MeV the local excess is not robust to the GP-support prescription. "
                "The changed support geometry moves the maximum-LML GP hyperparameters; "
                "the new state raises the common-grid continuum through the masked region "
                "and reduces the fitted signal coefficient."
            ),
            "not_established": (
                "The raw data feature did not disappear, and this controlled audit does not "
                "prove that it is background or that signal is absent."
            ),
        },
        "rebin_geometry": {
            "native_bin_width_MeV": 0.125,
            "neighborhood_rebin": 5,
            "coarse_bin_width_MeV": 0.625,
            "support_endpoint_shift_MeV": 4.0,
            "support_shift_modulo_coarse_bin_MeV": 0.25,
            "old_actual_fit_window_MeV": [60.0, 70.0],
            "new_actual_fit_window_MeV": [60.375, 69.75],
            "limitation": (
                "The v4.9.5 endpoint change does not isolate support extent from coarse-rebin "
                "phase. A fixed-anchor rebin rerun is needed to separate those effects."
            ),
        },
        "branch_lineage": lineage.to_dict(orient="records"),
        "scope_hybrid": scope.to_dict(orient="records"),
        "source_ledger_microdifference": {
            "accepted_m65_extraction_Z": float(old.Z_local_asymptotic),
            "accepted_full_curve_m65_Z": float(
                curve_summary["old_max"]["Z_local_asymptotic"]
            ),
            "absolute_delta_Z": abs(
                float(old.Z_local_asymptotic)
                - float(curve_summary["old_max"]["Z_local_asymptotic"])
            ),
            "interpretation": (
                "The exact table uses the dedicated accepted m65 extraction summary, "
                "while the full curve is copied without alteration from the reviewed "
                "individual scan ledger. The sub-micro-Z difference is immaterial."
            ),
        },
        "sources": sources,
    }
    write_json(summary, DERIVED / "signal_robustness_summary.json")
    return summary


def build_readme(
    exact: pd.DataFrame,
    change: pd.DataFrame,
    curve_summary: Dict[str, Any],
    gp_summary: Dict[str, Any],
    sources: List[Dict[str, Any]],
    validation: Dict[str, Any],
) -> None:
    old, new = exact.iloc[0], exact.iloc[1]
    c = change.iloc[0]
    source_lines = []
    for src in sources:
        source_lines.append(
            f"| `{src['path']}` | `{src['sha256']}` | {src['role']} |"
        )
    source_table = "\n".join(source_lines)
    readme = f"""# Signal-robustness audit for v4.9.7

This self-contained audit product explains the change in the earlier 65 MeV
local excess when the 2021 native-10% GP support moved from 40--300 MeV (v4.2)
to 36--300 MeV (v4.9.5). The builder snapshots the archived ledgers, reconstructs
the four fixed-hyperparameter support x kernel cells, carries the full 201-point
old/new local-Z curves, and makes PDF/PNG figures.

## Result at 65 MeV

| quantity | v4.2 support40 | v4.9.5 support36 | change |
|---|---:|---:|---:|
| fitted amplitude Ahat [events] | {old.A_hat_events:.6f} | {new.A_hat_events:.6f} | {100*c.relative_A_hat_change:+.1f}% |
| amplitude uncertainty [events] | {old.sigma_A_events:.6f} | {new.sigma_A_events:.6f} | {100*c.relative_sigma_A_change:+.1f}% |
| local asymptotic Z | {old.Z_local_asymptotic:.6f} | {new.Z_local_asymptotic:.6f} | {c.delta_Z_local_new_minus_old:+.6f} |
| local asymptotic p0 | {old.p0_local_asymptotic:.8g} | {new.p0_local_asymptotic:.8g} | x{c.p0_ratio_new_over_old:.1f} |
| 90% asymptotic epsilon^2 UL | {old.eps2_up:.8g} | {new.eps2_up:.8g} | {100*c.relative_eps2_up_change:+.1f}% |
| GP length scale in log(m) | {old.gp_length_scale_log_mass:.6f} | {new.gp_length_scale_log_mass:.6f} | -- |
| GP constant | {old.gp_const:.6f} | {new.gp_const:.6f} | -- |
| GP log marginal likelihood | {old.gp_log_marginal_likelihood:.6f} | {new.gp_log_marginal_likelihood:.6f} | {c.delta_gp_lml:+.3f} |

The observed ROOT file, mass resolution, density normalization, and conversion
K=A/epsilon^2 are unchanged at 65 MeV. The raw data feature therefore did not
"disappear." Rather, its fitted signal interpretation is not robust to this GP
support prescription.

## Mechanism supported by the audit

The fixed-state 2x2 table shows that this is not the mechanical effect of adding
four MeV of low-side support while holding the old GP state fixed. On the old
support geometry, swapping to the new kernel state already lowers Z from about
4.25 to 2.85. On the support36 geometry, holding the old state instead gives Z
about 4.74, whereas the support36 maximum-LML state gives 2.40. Each support
geometry prefers its corresponding diagonal kernel state in GP marginal
likelihood. This supports a mechanistic statement: the support change migrated
the preferred GP correlation structure, raising the inferred continuum through
the masked region and reducing Ahat.

On an identical 10,001-point grid over the nominal 65 MeV +/-2.25 sigma window,
the new diagonal GP mean is {gp_summary['delta_gp_mean_at_65_counts_per_0p625MeV']:+,.1f}
events per 0.625 MeV target bin at 65 MeV and
{gp_summary['delta_gp_integral_common_window_counts']:+,.1f} events
({100*gp_summary['relative_gp_integral_change']:+.4f}%) when integrated across the
common window. This is a fixed-state diagnostic; the profiled amplitude change
is not a one-bin subtraction.

## Scope of the earlier excess

The pre-optimization 2021-only maximum was at
{curve_summary['old_max']['mass_MeV']:.0f} MeV with local Z={curve_summary['old_max']['Z_local_asymptotic']:.3f}.
The v4.9.5 maximum is at {curve_summary['new_max']['mass_MeV']:.0f} MeV with local
Z={curve_summary['new_max']['Z_local_asymptotic']:.3f}. The often-quoted pre-optimization
Z=4.657 at 65 MeV is the 2015+2021 pair, not the all-three combination; the old
all-three local value is Z=3.993. Controlled hybrids that replace only the 2021
block lower the pair and all-three values as recorded in
`derived/m65_scope_hybrid_diagnostic.csv`. Those hybrid points are not an official
combined scan.

## Branch and geometry checks

- The raw finalist support40 state and the accepted v4.2 max-LML state differ by
  only 0.091 fitted event and 0.000017 in Z at 65 MeV. The v4.9.5 repair ledger
  repairs 94, 152, and 212 MeV, not 65 MeV. Branch repair is therefore not a
  material explanation for the 1.85-Z shift.
- The exact table uses the dedicated accepted 65 MeV extraction summary
  (Z={old.Z_local_asymptotic:.9f}); the unmodified full-curve ledger gives
  Z={curve_summary['old_max']['Z_local_asymptotic']:.9f} at that point. Their
  {abs(old.Z_local_asymptotic-curve_summary['old_max']['Z_local_asymptotic']):.2g}
  difference is a harmless ledger/refit-state microdifference, not a physics effect.
- The 2021 native bin width is 0.125 MeV and five-bin rebinning gives 0.625 MeV.
  Moving the lower support edge by 4 MeV shifts the coarse-bin phase by 0.25 MeV.
  The old 65 MeV fit window contains 16 bins spanning 60.000--70.000 MeV; the new
  window contains 15 bins spanning 60.375--69.750 MeV. The accepted v4.9.5 scan
  therefore does not isolate support extent from rebin phase. A fixed-anchor rebin
  audit is the appropriate follow-up if that separation is required.

## Claim boundary

{CLAIM_BOUNDARY} The v4.9.5 support choice itself did not minimize an observed
amplitude, p-value, or upper limit, but its 0.75 practical criterion was a
documented post-phase-1 amendment. The conditional support toys are source-recovery
diagnostics, not direct coverage.

## Products

- `derived/m65_2021_exact_comparison.csv` and `m65_2021_exact_changes.csv` -- exact accepted old/new values.
- `derived/m65_support_kernel_counterfactual.csv` -- four fixed-state support x kernel cells.
- `derived/m65_gp_common_grid.csv` -- direct old/new GP mean and pointwise standard deviation on a common grid.
- `derived/old_new_2021_local_z_curves.csv` -- complete 201-point archived curve comparison.
- `derived/key_mass_curve_comparison.csv` -- compact key-mass extract.
- `derived/m65_scope_hybrid_diagnostic.csv` -- 2021/pair/all-three single-mass scope check.
- `derived/m65_branch_lineage.csv` -- raw-finalist, accepted-v4.2, and v4.9.5 branch lineage.
- `derived/signal_robustness_summary.json` -- machine-readable interpretation and claim boundaries.
- `figures/*.pdf` and `figures/*.png` -- publication figures.
- `source_snapshots/` -- compact snapshots of every archived ledger used directly.
- `qa/pdf_render_qa.json`, `qa/semantic_validation.json`, and rendered PDF page PNGs -- QA evidence.
- `qa/artifact_manifest_sha256.csv` -- SHA-256 for every artifact except the manifest itself.

## Source provenance

| source | SHA-256 | role |
|---|---|---|
{source_table}

The two absolute finalist paths above are read-only lineage inputs recorded by the
audit; the builder does not reread the primary dirty checkout. All substantive
curve and accepted-result inputs are either snapshotted here or available by
`git show HEAD:<path>` from the isolated release revision.

## Rebuild

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  python3 {HERE.relative_to(REPO)}/build_signal_robustness_audit.py
```

Semantic QA: `{validation['n_pass']}/{validation['n_checks']}` checks passed.
"""
    (HERE / "README.md").write_text(readme, encoding="utf-8")


def build_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == QA / "artifact_manifest_sha256.csv":
            continue
        rows.append(
            {
                "path": str(path.relative_to(HERE)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = pd.DataFrame(rows)
    write_frame(manifest, QA / "artifact_manifest_sha256.csv")
    return manifest


def main() -> None:
    reset_generated_directories()
    data, sources = load_and_snapshot_sources()
    write_frame(pd.DataFrame(sources), DERIVED / "source_provenance.csv")
    exact, change, lineage = build_exact_tables(data)
    curve, _, curve_summary = build_curve_tables(data)
    counter, common, gp_summary = reconstruct_counterfactuals(exact)
    scope = build_scope_hybrid_table(exact)

    plot_local_z(curve, curve_summary)
    plot_mechanism(exact, counter, common, gp_summary)
    plot_scope_hybrid(scope)
    render_qa = render_pdf_qa()
    validation = build_validation(data, exact, counter, curve, curve_summary, render_qa)
    build_summary_json(exact, change, lineage, curve_summary, counter, gp_summary, scope, sources)
    build_readme(exact, change, curve_summary, gp_summary, sources, validation)
    manifest = build_manifest()

    print(f"signal audit: {HERE}")
    print(f"semantic QA: {validation['n_pass']}/{validation['n_checks']} pass")
    print(f"manifest entries: {len(manifest)}")
    for path in sorted(FIGURES.glob("*.pdf")):
        print(f"{path.relative_to(HERE)}  {sha256_file(path)}")


if __name__ == "__main__":
    main()
