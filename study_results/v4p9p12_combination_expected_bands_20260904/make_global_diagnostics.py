#!/usr/bin/env python3
"""Report resolution-Sidak references and finite-grid trials adjustments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARENT = REPO / "study_results/v4p9p12_final_dataset_combinations_20260902"
DERIVED = HERE / "derived"
CARD = PARENT / "inputs/analysis_card.yaml"
OBSERVED = PARENT / "derived/final_dataset_result_curves.csv"
METHOD = HERE / "GLOBAL_PVALUE_METHOD.md"
WIDTH_SIGMA = 2.25
TOTAL_KEY = "final_total_search_window"
TOTAL_SEGMENTS = (
    (19, 38, "individual_2015_full"),
    (39, 49, "pair_2015_2016"),
    (50, 90, "all_2015_2016_2021"),
    (91, 180, "pair_2016_2021"),
    (181, 250, "individual_2021_10pct"),
)
UNAVAILABLE_REASON = (
    "Band pseudoexperiments are generated independently at different masses. "
    "Their toy IDs do not define coherent whole-spectrum scans, so they cannot "
    "supply the null distribution of scan maxima or a scan-calibrated p-value."
)
REFERENCES = [
    {
        "title": "SAS/STAT: p-Value Adjustments",
        "url": "https://support.sas.com/documentation/cdl/en/statug/66859/HTML/default/statug_multtest_details11.htm",
        "use": "Sidak formula and independence conditions; Bonferroni grid adjustment",
    },
    {
        "title": "NIST/SEMATECH: Bonferroni's method",
        "url": "https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm",
        "use": "Union bound for arbitrary dependence, conditional on valid local tests",
    },
    {
        "title": "Gross and Vitells, Trial factors for the look elsewhere effect in high energy physics",
        "url": "https://arxiv.org/abs/1005.1891",
        "use": "Scan-level look-elsewhere calibration and correlation-aware trial factors",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(path: Path) -> dict[str, str]:
    return {"path": path.resolve().relative_to(REPO).as_posix(), "sha256": sha256(path)}


def sigma_from_card(card: dict, dataset: str, mass_gev: float) -> float:
    """Independent cross-check of the attested runtime's resolution function."""
    coefficients = card[f"sigma_coeffs_{dataset}"]

    def polynomial(mass: float) -> float:
        return float(sum(c * mass**i for i, c in enumerate(coefficients)))

    tail_start = card.get(f"sigma_tail_m0_{dataset}")
    if tail_start is None or mass_gev <= float(tail_start):
        return polynomial(mass_gev)
    slope = card.get(f"sigma_tail_slope_override_{dataset}")
    if slope is None:
        slope = sum(i * c * tail_start ** (i - 1) for i, c in enumerate(coefficients) if i)
    slope = max(float(slope), float(card.get(f"sigma_tail_slope_floor_{dataset}", 0.0)))
    return polynomial(float(tail_start)) + slope * (mass_gev - float(tail_start))


def require_grid(frame: pd.DataFrame, low: int, high: int, scope: str) -> pd.DataFrame:
    frame = frame.sort_values("mass_MeV").reset_index(drop=True)
    if not np.array_equal(frame.mass_MeV.to_numpy(float), np.arange(low, high + 1)):
        raise RuntimeError(f"Incomplete or duplicate mass grid for {scope}")
    return frame


def summarize_window(
    frame: pd.DataFrame,
    *,
    scope_key: str,
    scope_label: str,
    datasets: dict,
    card: dict,
    target_toys: int,
) -> tuple[dict, list[dict], float]:
    masses = frame.mass_MeV.to_numpy(float) / 1000.0
    resolutions = []
    sigma_mappings = []
    maximum_sigma_difference = 0.0
    for row, mass in zip(frame.itertuples(), masses):
        mapping = {}
        for key in str(row.dataset_set).split("+"):
            sigma = float(datasets[key].sigma(float(mass)))
            crosscheck = sigma_from_card(card, key, float(mass))
            maximum_sigma_difference = max(maximum_sigma_difference, abs(sigma - crosscheck))
            if not np.isfinite(sigma) or sigma <= 0.0 or not np.isclose(
                sigma, crosscheck, rtol=1e-13, atol=1e-15
            ):
                raise RuntimeError(f"Resolution check failed: {scope_key}/{key}/{mass}")
            mapping[key] = sigma
        sigma_mappings.append(mapping)
        resolutions.append(min(mapping.values()))
    sigma = np.asarray(resolutions, dtype=float)
    steps = np.diff(masses)
    sigma_mid = 0.5 * (sigma[:-1] + sigma[1:])
    contributions = steps / (WIDTH_SIGMA * sigma_mid)
    raw_neff = float(np.sum(contributions))
    neff = float(np.clip(raw_neff, 1.0, len(frame)))
    pvalues = frame.p0_local_asymptotic.to_numpy(float)
    minimum_index = int(np.argmin(pvalues))
    minimum = frame.iloc[minimum_index]
    local_p = float(pvalues[minimum_index])
    sidak = float(np.clip(-np.expm1(neff * np.log1p(-local_p)), 0.0, 1.0))
    bonferroni = float(min(1.0, len(frame) * local_p))
    if not (1.0 <= neff <= len(frame) and local_p <= sidak <= bonferroni <= 1.0):
        raise RuntimeError(f"Trials-adjustment ordering failed: {scope_key}")
    summary = {
        "scope_key": scope_key,
        "scope_label": scope_label,
        "mass_min_MeV": int(frame.mass_MeV.min()),
        "mass_max_MeV": int(frame.mass_MeV.max()),
        "n_mass_points": len(frame),
        "N_eff_resolution_spacing": neff,
        "N_eff_before_grid_cap": raw_neff,
        "independence_width_sigma": WIDTH_SIGMA,
        "mass_at_min_p0_MeV": int(minimum.mass_MeV),
        "selected_scope_key_at_min": str(minimum.scope_key),
        "p0_local_asymptotic_min": local_p,
        "Z_local_asymptotic_at_min": float(minimum.Z_local_asymptotic),
        "p_sidak_resolution_spacing_analytic": sidak,
        "Z_sidak_resolution_spacing_analytic": float(norm.isf(sidak)),
        "p_bonferroni_grid": bonferroni,
        "scan_toy_calibrated": False,
        "uses_limit_tail_pvalues": False,
        "report_target_toys": target_toys,
    }
    ledger = []
    for index, row in enumerate(frame.itertuples()):
        ledger.append(
            {
                "scope_key": scope_key,
                "selected_scope_key": str(row.scope_key),
                "mass_MeV": int(row.mass_MeV),
                "mass_GeV": float(masses[index]),
                "dataset_set": str(row.dataset_set),
                "sigma_by_dataset_GeV": json.dumps(sigma_mappings[index], sort_keys=True, separators=(",", ":")),
                "sigma_effective_GeV": float(sigma[index]),
                "next_mass_step_GeV": float(steps[index]) if index < len(steps) else 0.0,
                "next_interval_sigma_mid_GeV": float(sigma_mid[index]) if index < len(steps) else 0.0,
                "N_eff_next_interval_contribution": float(contributions[index]) if index < len(steps) else 0.0,
                "independence_width_sigma": WIDTH_SIGMA,
                "p0_local_asymptotic": float(row.p0_local_asymptotic),
                "is_scope_scan_minimum": index == minimum_index,
                "report_target_toys": target_toys,
            }
        )
    return summary, ledger, maximum_sigma_difference


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-toys", type=int, choices=(100, 300), required=True)
    args = parser.parse_args(argv)

    # Activates and verifies the same frozen runtime as the observed analysis;
    # constructing DatasetConfig objects does not open data or run fits/toys.
    sys.path.insert(0, str(PARENT))
    import run_final_combinations as final

    config = final.load_config(CARD)
    final.validate_card(config)
    datasets = final.make_datasets(config)
    card = yaml.safe_load(CARD.read_text(encoding="utf-8"))
    if float(config.blind_nsigma) != WIDTH_SIGMA:
        raise RuntimeError("Frozen card no longer matches the documented spacing convention")
    observed = pd.read_csv(OBSERVED, dtype={"dataset_set": str})
    if observed.duplicated(["scope_key", "mass_MeV"]).any():
        raise RuntimeError("Duplicate observed scope-mass rows")
    if set(observed.scope_key) != {item[0] for item in final.SCOPES}:
        raise RuntimeError("Observed scope set differs from the frozen seven-scope definition")
    pvalues = observed.p0_local_asymptotic.to_numpy(float)
    zvalues = observed.Z_local_asymptotic.to_numpy(float)
    if not (
        np.isfinite(pvalues).all()
        and np.isfinite(zvalues).all()
        and np.all((pvalues > 0.0) & (pvalues <= 0.5))
        and np.allclose(norm.sf(zvalues), pvalues, rtol=1e-10, atol=1e-15)
    ):
        raise RuntimeError("Observed analytic p0/Z values fail input validation")

    windows = []
    for scope, label, keys, low, high in final.SCOPES:
        frame = require_grid(observed[observed.scope_key == scope], low, high, scope)
        if set(frame.dataset_set) != {"+".join(keys)}:
            raise RuntimeError(f"Dataset composition mismatch: {scope}")
        windows.append((scope, label, frame))
    total_pieces = [
        require_grid(
            observed[(observed.scope_key == scope) & observed.mass_MeV.between(low, high)],
            low, high, scope,
        )
        for low, high, scope in TOTAL_SEGMENTS
    ]
    total = require_grid(pd.concat(total_pieces, ignore_index=True), 19, 250, TOTAL_KEY)
    windows.append((TOTAL_KEY, "Total search window (maximal available datasets)", total))
    summaries = []
    resolution_rows = []
    max_sigma_difference = 0.0
    for scope, label, frame in windows:
        summary, ledger, difference = summarize_window(
            frame, scope_key=scope, scope_label=label, datasets=datasets,
            card=card, target_toys=args.target_toys,
        )
        summaries.append(summary)
        resolution_rows.extend(ledger)
        max_sigma_difference = max(max_sigma_difference, difference)
    if len(observed) != 680 or len(summaries) != 8 or len(resolution_rows) != 912:
        raise RuntimeError("Unexpected current-release window or grid counts")

    DERIVED.mkdir(parents=True, exist_ok=True)
    summary_path = DERIVED / f"global_pvalue_summary_{args.target_toys}toys.csv"
    ledger_path = DERIVED / f"global_resolution_ledger_{args.target_toys}toys.csv"
    manifest_path = DERIVED / f"global_pvalue_manifest_{args.target_toys}toys.json"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(resolution_rows).to_csv(ledger_path, index=False)
    family_minimum = observed.sort_values(["p0_local_asymptotic", "scope_key", "mass_MeV"]).iloc[0]
    runtime_manifest = Path(final.RUNTIME_PROVENANCE["runtime_manifest"])
    manifest = {
        "schema_version": 1,
        "report_target_toys": args.target_toys,
        "target_toys_role": "Report-stage identifier only; all quantities depend on frozen observed p0 and resolution, not band-toy count or completion.",
        "requires_completed_band_stage": False,
        "sources": {
            "producer": source_record(Path(__file__).resolve()),
            "method_document": source_record(METHOD),
            "observed_result_curves": source_record(OBSERVED),
            "analysis_card": source_record(CARD),
            "parent_runner": source_record(PARENT / "run_final_combinations.py"),
            "frozen_dataset_module": source_record(Path(sys.modules["hps_gpr.dataset"].__file__)),
            "frozen_config_module": source_record(Path(sys.modules["hps_gpr.config"].__file__)),
            "frozen_runtime_manifest": source_record(runtime_manifest),
        },
        "outputs": {
            "summary": source_record(summary_path),
            "resolution_ledger": source_record(ledger_path),
        },
        "window_summary_rows": len(summaries),
        "resolution_ledger_rows": len(resolution_rows),
        "independence_width_sigma": WIDTH_SIGMA,
        "sigma_effective_rule": "Minimum detector mass resolution among the datasets selected at each mass; no observed p0 or limit enters selection.",
        "resolution_integration_rule": "Sum adjacent mass steps divided by W times the arithmetic mean of the two endpoint effective resolutions; cap at [1, number of tested masses].",
        "formulas": {
            "N_eff": "clip(sum_i((m[i+1]-m[i])/(W*(sigma_eff[i]+sigma_eff[i+1])/2)),1,M)",
            "p_sidak": "-expm1(N_eff*log1p(-p0_min))",
            "p_bonferroni_grid": "min(1,M*p0_min)",
            "Z_sidak": "norm.isf(p_sidak)",
        },
        "total_window_segments": [
            {"mass_min_MeV": low, "mass_max_MeV": high, "scope_key": scope}
            for low, high, scope in TOTAL_SEGMENTS
        ],
        "all_scope_family": {
            "n_tests": len(observed),
            "scope_key_at_min": str(family_minimum.scope_key),
            "mass_at_min_p0_MeV": int(family_minimum.mass_MeV),
            "p0_min": float(family_minimum.p0_local_asymptotic),
            "p_bonferroni_grid": min(1.0, len(observed) * float(family_minimum.p0_local_asymptotic)),
            "N_eff_resolution_spacing": None,
            "p_sidak_resolution_spacing_analytic": None,
            "reason_no_sidak": "Overlapping dataset scopes do not define an additional calibrated effective trials count.",
            "interpretation": "Adjustment for choosing the minimum among all 680 scope-mass tests; distinct from the fixed total-window selection.",
        },
        "scan_calibrated_empirical_pvalue": None,
        "scan_calibrated_empirical_pvalue_status": "unavailable_from_mass_independent_band_toys",
        "scan_calibrated_empirical_pvalue_unavailable_reason": UNAVAILABLE_REASON,
        "uses_limit_tail_pvalues": False,
        "claim_boundary": "Resolution-Sidak is an approximation; grid Bonferroni is a union-bound adjustment conditional on valid fixed-model asymptotic local p0. Neither calibrates the partially unblinded model history or a continuum between grid points.",
        "checks": {
            "complete_unique_grids": True,
            "positive_finite_resolutions": True,
            "runtime_sigma_agrees_with_independent_card_polynomial": True,
            "runtime_sigma_max_absolute_difference_GeV": max_sigma_difference,
            "local_p0_Z_consistent": True,
            "pvalue_ordering_and_bounds": True,
        },
        "references": REFERENCES,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries)[[
        "scope_key", "n_mass_points", "N_eff_resolution_spacing",
        "p_sidak_resolution_spacing_analytic", "p_bonferroni_grid",
    ]].to_string(index=False))
    print(f"Wrote {manifest_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
