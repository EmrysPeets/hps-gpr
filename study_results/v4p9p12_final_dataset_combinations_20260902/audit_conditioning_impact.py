#!/usr/bin/env python3
"""Run the frozen raw-vs-explicit covariance-conditioning impact audit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAMPAIGN = REPO / "study_results/v4p9p7_2016_support_combined_100toy_20260902"
sys.path.insert(0, str(CAMPAIGN))
from runtime_guard import activate_and_verify  # noqa: E402


activate_and_verify()
sys.path.insert(0, str(HERE / "runtime"))
sys.path.insert(0, str(HERE))

import run_final_combinations as workflow  # noqa: E402
from hps_gpr.config import load_config  # noqa: E402
from hps_gpr.dataset import make_datasets  # noqa: E402
from hps_gpr.evaluation import build_combined_components  # noqa: E402
from hps_gpr.statistics import p0_profiled_gaussian_LRT  # noqa: E402
from piecewise_cached_solver import CachedPiecewiseBoundedLimit  # noqa: E402


CARD = HERE / "inputs" / "analysis_card.yaml"
STATES = HERE / "inputs" / "reviewed_gp_states.csv"
PROVENANCE = HERE / "inputs" / "analysis_input_provenance.json"
PREDICTION_LEDGER = HERE / "derived" / "prediction_state_ledger.csv"
QA = HERE / "qa"
LIMIT_RTOL_MAX = 5.0e-4
Z_ATOL_MAX = 5.0e-3

AUDIT_COORDINATES: Tuple[Tuple[str, Tuple[str, ...], int], ...] = (
    ("individual_2015_full", ("2015",), 19),
    ("individual_2015_full", ("2015",), 50),
    ("individual_2015_full", ("2015",), 90),
    ("individual_2016_full", ("2016",), 39),
    ("individual_2016_full", ("2016",), 65),
    ("individual_2016_full", ("2016",), 102),
    ("individual_2016_full", ("2016",), 180),
    ("individual_2021_10pct", ("2021",), 50),
    ("individual_2021_10pct", ("2021",), 78),
    ("individual_2021_10pct", ("2021",), 150),
    ("individual_2021_10pct", ("2021",), 250),
    ("pair_2015_2016", ("2015", "2016"), 50),
    ("pair_2015_2021", ("2015", "2021"), 50),
    ("pair_2016_2021", ("2016", "2021"), 50),
    ("all_2015_2016_2021", ("2015", "2016", "2021"), 50),
    ("pair_2015_2016", ("2015", "2016"), 65),
    ("pair_2015_2021", ("2015", "2021"), 65),
    ("pair_2016_2021", ("2016", "2021"), 65),
    ("all_2015_2016_2021", ("2015", "2016", "2021"), 65),
    ("pair_2015_2016", ("2015", "2016"), 90),
    ("pair_2015_2021", ("2015", "2021"), 90),
    ("pair_2016_2021", ("2016", "2021"), 90),
    ("all_2015_2016_2021", ("2015", "2016", "2021"), 90),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p0_result(obs, bkg, cov, s_unit) -> Tuple[float, float, Dict[str, object]]:
    scale = float(np.sum(s_unit))
    p0, z_value, _q0, info = p0_profiled_gaussian_LRT(
        obs, bkg, cov, s_unit / scale
    )
    nll_alt = float(info.get("nll_alt", float("nan")))
    nll0 = float(info.get("nll0", float("nan")))
    tolerance = 1.0e-6 + 1.0e-8 * max(1.0, abs(nll0 - nll_alt))
    if not (
        bool(info.get("ok", False))
        and np.isfinite(nll_alt)
        and np.isfinite(nll0)
        and nll_alt <= nll0 + tolerance
    ):
        raise RuntimeError("conditioning-audit p0 profile failed")
    return float(p0), float(z_value), {
        "nll_alt": nll_alt,
        "nll0": nll0,
        "tolerance": tolerance,
    }


def main() -> None:
    config = load_config(CARD)
    workflow.result_config = config
    workflow.validate_card(config)
    workflow.validate_input_provenance(PROVENANCE, CARD, STATES, config)
    workflow.validate_histogram_inputs(config)
    states_frame = workflow.load_states(STATES, config)
    states = workflow.state_map(states_frame)
    datasets = make_datasets(config)

    cache: Dict[int, tuple] = {}
    rows: List[Dict[str, object]] = []
    for scope, keys, mass_mev in AUDIT_COORDINATES:
        if mass_mev not in cache:
            cache[mass_mev] = workflow.reconstruct_predictions(
                mass_mev / 1000.0, datasets, config, states
            )
        predictions, conditioned, conditioning, _records = cache[mass_mev]
        ds_here = [datasets[key] for key in keys]
        pred_here = [predictions[key] for key in keys]
        obs, bkg, raw_cov, s_unit = build_combined_components(
            mass_mev / 1000.0, ds_here, pred_here, config=config
        )
        conditioned_cov = workflow.block_diagonal(
            [conditioned[key] for key in keys]
        )
        raw_limit = CachedPiecewiseBoundedLimit(
            bkg,
            raw_cov,
            s_unit,
            alpha=float(config.cls_alpha),
            combined_mode="count_scale",
        ).limit(obs)
        conditioned_limit = CachedPiecewiseBoundedLimit(
            bkg,
            conditioned_cov,
            s_unit,
            alpha=float(config.cls_alpha),
            combined_mode="count_scale",
        ).limit(obs)
        raw_p0, raw_z, raw_p0_meta = p0_result(obs, bkg, raw_cov, s_unit)
        conditioned_p0, conditioned_z, conditioned_p0_meta = p0_result(
            obs, bkg, conditioned_cov, s_unit
        )
        relative_limit_difference = abs(
            conditioned_limit.eps2_90 / raw_limit.eps2_90 - 1.0
        )
        z_difference = abs(conditioned_z - raw_z)
        passed = (
            relative_limit_difference <= LIMIT_RTOL_MAX
            and z_difference <= Z_ATOL_MAX
        )
        rows.append(
            {
                "scope_key": scope,
                "dataset_set": "+".join(keys),
                "mass_MeV": mass_mev,
                "raw_eps2_90": raw_limit.eps2_90,
                "conditioned_eps2_90": conditioned_limit.eps2_90,
                "relative_limit_difference": relative_limit_difference,
                "raw_p0": raw_p0,
                "conditioned_p0": conditioned_p0,
                "raw_Z": raw_z,
                "conditioned_Z": conditioned_z,
                "absolute_Z_difference": z_difference,
                "limit_relative_tolerance": LIMIT_RTOL_MAX,
                "Z_absolute_tolerance": Z_ATOL_MAX,
                "passed": passed,
                "conditioning_by_dataset": json.dumps(
                    {key: conditioning[key] for key in keys},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "raw_p0_profile": json.dumps(raw_p0_meta, sort_keys=True),
                "conditioned_p0_profile": json.dumps(
                    conditioned_p0_meta, sort_keys=True
                ),
            }
        )

    frame = pd.DataFrame(rows)
    prediction_ledger = pd.read_csv(PREDICTION_LEDGER)
    full_grid_max_load = float(
        prediction_ledger.selected_diagonal_load_relative.max()
    )
    passed = bool(frame.passed.all() and full_grid_max_load < 1.0e-4)
    QA.mkdir(parents=True, exist_ok=True)
    csv_path = QA / "numerical_conditioning_impact.csv"
    frame.to_csv(csv_path, index=False)
    report = {
        "schema_version": 1,
        "status": "audit_passed" if passed else "audit_failed",
        "passed": passed,
        "audit_rows": int(len(frame)),
        "expected_audit_rows": len(AUDIT_COORDINATES),
        "maximum_relative_limit_difference": float(
            frame.relative_limit_difference.max()
        ),
        "maximum_absolute_Z_difference": float(
            frame.absolute_Z_difference.max()
        ),
        "full_grid_maximum_selected_diagonal_load_relative": full_grid_max_load,
        "full_grid_requires_1e-4_cap": bool(full_grid_max_load >= 1.0e-4),
        "tolerances": {
            "relative_limit_difference": LIMIT_RTOL_MAX,
            "absolute_Z_difference": Z_ATOL_MAX,
            "full_grid_load_must_be_strictly_below": 1.0e-4,
        },
        "inputs": {
            "audit_script_sha256": sha256(Path(__file__).resolve()),
            "audit_protocol_sha256": sha256(
                HERE / "NUMERICAL_CONDITIONING_AUDIT_PROTOCOL.md"
            ),
            "card_sha256": sha256(CARD),
            "states_sha256": sha256(STATES),
            "prediction_ledger_sha256": sha256(PREDICTION_LEDGER),
            "audit_csv_sha256": sha256(csv_path),
        },
    }
    (QA / "numerical_conditioning_impact.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        raise SystemExit(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
