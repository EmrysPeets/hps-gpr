#!/usr/bin/env python3
"""Fail-closed validation for a completed v4.9.12 expected-band stage."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_expected_bands as band  # noqa: E402

TOTAL_RULE = (
    (19, 38, "individual_2015_full"),
    (39, 49, "pair_2015_2016"),
    (50, 90, "all_2015_2016_2021"),
    (91, 180, "pair_2016_2021"),
    (181, 250, "individual_2021_10pct"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-toys", type=int, required=True)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str, checks: Dict[str, object], key: str) -> None:
    checks[key] = bool(condition)
    if not condition:
        raise RuntimeError(message)


def validate_prefix(
    limits: pd.DataFrame,
    draws: pd.DataFrame,
    target_toys: int,
    checks: Dict[str, object],
) -> None:
    earlier = [value for value in band.ALLOWED_RELEASE_STAGES if value < target_toys]
    if not earlier:
        checks["earlier_stage_prefix_preserved"] = True
        return
    previous = max(earlier)
    previous_limits_path = HERE / "derived" / f"toy_limits_{previous}toys.csv"
    previous_draws_path = HERE / "derived" / f"toy_observations_{previous}toys.csv"
    if not previous_limits_path.is_file() or not previous_draws_path.is_file():
        raise RuntimeError(f"missing preserved {previous}-toy stage ledgers")
    previous_limits = pd.read_csv(
        previous_limits_path, dtype={"dataset_set": str}, low_memory=False
    )
    previous_draws = pd.read_csv(previous_draws_path)
    current_limit_prefix = limits[limits.toy_id < previous].reset_index(drop=True)
    current_draw_prefix = draws[draws.toy_id < previous].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        previous_draws.reset_index(drop=True),
        current_draw_prefix,
        check_exact=True,
    )
    if previous == 50:
        metadata_columns = {"limit_solver", "solver_counter_delta"}
        comparison_columns = [
            column
            for column in previous_limits.columns
            if column not in metadata_columns
        ]
        previous_numeric_prefix = previous_limits[comparison_columns].reset_index(
            drop=True
        )
        current_numeric_prefix = current_limit_prefix[comparison_columns].reset_index(
            drop=True
        )
        # ``dataset_set`` contains both digit-only standalone labels and ``+``
        # combination labels. Chunked CSV inference can therefore represent the
        # former as integers in one frame and strings in the other. Canonicalize
        # only this categorical field; every numerical column remains exact.
        previous_numeric_prefix["dataset_set"] = previous_numeric_prefix[
            "dataset_set"
        ].astype(str)
        current_numeric_prefix["dataset_set"] = current_numeric_prefix[
            "dataset_set"
        ].astype(str)
        pd.testing.assert_frame_equal(
            previous_numeric_prefix,
            current_numeric_prefix,
            check_exact=True,
        )
        old_solver = (
            "v4p9p12_cached_piecewise_bounded_tildeq_v3_"
            "centered_fixed_profile_v2"
        )
        if set(previous_limits.limit_solver.astype(str)) != {old_solver}:
            raise RuntimeError("preserved 50-toy ledger has an unexpected solver")
        retry_keys = {
            "bounded_free_centered_retries",
            "unbounded_free_centered_retries",
            "null_centered_retries",
        }
        for old_text, new_text in zip(
            previous_limits.solver_counter_delta,
            current_limit_prefix.solver_counter_delta,
        ):
            old_counters = json.loads(str(old_text))
            new_counters = json.loads(str(new_text))
            if any(int(new_counters.pop(key, 0)) != 0 for key in retry_keys):
                raise RuntimeError(
                    "centered-profile retry activated inside the accepted "
                    "50-toy numeric prefix"
                )
            if old_counters != new_counters:
                raise RuntimeError(
                    "non-amendment solver counters changed inside the 50-toy prefix"
                )
        checks["accepted_50toy_numeric_prefix_preserved"] = True
        checks["accepted_50toy_draw_prefix_preserved"] = True
    else:
        pd.testing.assert_frame_equal(
            previous_limits.reset_index(drop=True),
            current_limit_prefix,
            check_exact=True,
        )
        checks["earlier_stage_prefix_preserved"] = True


def validate_global_diagnostics(target_toys: int, checks: Dict[str, object]) -> None:
    derived = HERE / "derived"
    metadata_path = derived / f"global_pvalue_manifest_{target_toys}toys.json"
    metadata = json.loads(metadata_path.read_text())
    for entry in (*metadata["sources"].values(), *metadata["outputs"].values()):
        if sha256(REPO / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"global-diagnostic hash mismatch: {entry['path']}")
    require(
        metadata["report_target_toys"] == target_toys
        and metadata["requires_completed_band_stage"] is False
        and metadata["uses_limit_tail_pvalues"] is False
        and metadata["scan_calibrated_empirical_pvalue"] is None
        and metadata["scan_calibrated_empirical_pvalue_status"]
        == "unavailable_from_mass_independent_band_toys"
        and metadata["independence_width_sigma"] == 2.25,
        "global-diagnostic interpretation changed", checks, "global_semantics_and_hashes",
    )
    summary = pd.read_csv(derived / f"global_pvalue_summary_{target_toys}toys.csv")
    ledger = pd.read_csv(derived / f"global_resolution_ledger_{target_toys}toys.csv",
                         dtype={"dataset_set": str})
    observed = pd.read_csv(band.DEFAULT_OBSERVED, dtype={"dataset_set": str})
    pieces = [observed[(observed.scope_key == scope) & observed.mass_MeV.between(low, high)]
              for low, high, scope in TOTAL_RULE]
    expected = {scope: observed[observed.scope_key == scope]
                for scope, *_ in band.final.SCOPES}
    expected["final_total_search_window"] = pd.concat(pieces).sort_values("mass_MeV")
    if (len(summary) != 8 or len(ledger) != 912 or set(summary.scope_key) != set(expected)
            or summary.scope_key.duplicated().any()
            or ledger.duplicated(["scope_key", "mass_MeV"]).any()):
        raise RuntimeError("global-diagnostic grids are incomplete or duplicated")
    datasets = band.make_datasets(band.load_config(band.DEFAULT_CARD))
    for key, frame in expected.items():
        frame = frame.sort_values("mass_MeV").reset_index(drop=True)
        trace = ledger[ledger.scope_key == key].sort_values("mass_MeV").reset_index(drop=True)
        row = summary.loc[summary.scope_key == key].iloc[0]
        pd.testing.assert_series_equal(frame.mass_MeV, trace.mass_MeV, check_names=False)
        pd.testing.assert_series_equal(frame.scope_key, trace.selected_scope_key, check_names=False)
        pd.testing.assert_series_equal(frame.dataset_set, trace.dataset_set, check_names=False)
        np.testing.assert_allclose(frame.p0_local_asymptotic, trace.p0_local_asymptotic,
                                   rtol=1e-13, atol=0)
        sigma = []
        for source, recorded in zip(frame.itertuples(), trace.itertuples()):
            mapping = {dataset: float(datasets[dataset].sigma(source.mass_MeV / 1000.0))
                       for dataset in str(source.dataset_set).split("+")}
            saved_mapping = json.loads(recorded.sigma_by_dataset_GeV)
            if saved_mapping != mapping:
                raise RuntimeError("global detector resolutions do not match the frozen runtime")
            sigma.append(min(mapping.values()))
        sigma = np.asarray(sigma)
        np.testing.assert_allclose(sigma, trace.sigma_effective_GeV, rtol=1e-12, atol=0)
        contributions = np.diff(frame.mass_MeV.to_numpy(float) / 1000.0) / (
            2.25 * (sigma[1:] + sigma[:-1]) / 2.0
        )
        np.testing.assert_allclose(np.r_[contributions, 0.0],
                                   trace.N_eff_next_interval_contribution, rtol=1e-12, atol=1e-15)
        neff = float(np.clip(np.sum(contributions), 1, len(frame)))
        minimum = frame.loc[frame.p0_local_asymptotic.idxmin()]
        pmin = float(minimum.p0_local_asymptotic)
        sidak = float(-np.expm1(neff * np.log1p(-pmin)))
        bonferroni = min(1.0, len(frame) * pmin)
        np.testing.assert_allclose(
            [row.N_eff_resolution_spacing, row.p0_local_asymptotic_min,
             row.p_sidak_resolution_spacing_analytic, row.Z_sidak_resolution_spacing_analytic,
             row.p_bonferroni_grid],
            [neff, pmin, sidak, norm.isf(sidak), bonferroni], rtol=1e-12, atol=1e-15,
        )
        if (int(row.n_mass_points) != len(frame)
                or int(row.mass_at_min_p0_MeV) != int(minimum.mass_MeV)
                or bool(row.scan_toy_calibrated) or bool(row.uses_limit_tail_pvalues)):
            raise RuntimeError("global-diagnostic scope metadata mismatch")
    family = metadata["all_scope_family"]
    minimum = observed.loc[observed.p0_local_asymptotic.idxmin()]
    require(
        family["n_tests"] == 680
        and family["scope_key_at_min"] == minimum.scope_key
        and family["mass_at_min_p0_MeV"] == int(minimum.mass_MeV)
        and family["p0_min"] == float(minimum.p0_local_asymptotic)
        and family["p_bonferroni_grid"] == min(1.0, 680 * float(minimum.p0_local_asymptotic))
        and family["p_sidak_resolution_spacing_analytic"] is None,
        "all-scope Bonferroni family calculation differs", checks, "all_scope_trials_family_exact",
    )
    checks["global_resolution_ledger_and_formulas_recompute"] = True


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target_toys = int(args.target_toys)
    if target_toys not in band.ALLOWED_RELEASE_STAGES:
        raise SystemExit(f"release target must be one of {band.ALLOWED_RELEASE_STAGES}")
    derived = HERE / "derived"
    paths = band.stage_paths(derived, target_toys)
    pvalue_path = derived / f"pvalue_diagnostics_{target_toys}toys.csv"
    total_window_path = (
        derived / f"final_total_search_window_summary_{target_toys}toys.csv"
    )
    required = list(paths.values()) + [
        derived / "contract.json",
        pvalue_path,
        total_window_path,
        HERE / "figures" / f"figure_manifest_{target_toys}toys.json",
        HERE / "note" / f"HPS_GPR_Analysis_Note_v4p9p12_expected_bands_{target_toys}toys.pdf",
        REPO
        / "output"
        / "pdf"
        / "v4p9p12_expected_bands_20260904"
        / f"HPS_GPR_Analysis_Note_v4p9p12_expected_bands_{target_toys}toys.pdf",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing release artifacts: {missing}")

    checks: Dict[str, object] = {}
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    current_manifest = json.loads(
        paths["manifest_current"].read_text(encoding="utf-8")
    )
    require(
        manifest == current_manifest,
        "stage and current run manifests differ",
        checks,
        "current_manifest_matches_stage",
    )
    require(
        manifest.get("status") == "pointwise_conditional_expected_bands_complete"
        and int(manifest.get("stage_toys_per_mass", -1)) == target_toys
        and manifest.get("toy_id_range") == [0, target_toys - 1]
        and manifest.get("conditional_on_frozen_gp") is True
        and manifest.get("gp_refit_per_toy") is False
        and manifest.get("pointwise_not_scanwide") is True,
        "run-manifest semantics or stage metadata drifted",
        checks,
        "run_manifest_semantics",
    )

    contract_path = derived / "contract.json"
    saved_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    recomputed_digest, recomputed_payload = band.contract(
        card=band.DEFAULT_CARD,
        states=band.DEFAULT_STATES,
        provenance=band.DEFAULT_INPUT_PROVENANCE,
        observed=band.DEFAULT_OBSERVED,
        seed=band.MASTER_SEED,
    )
    recomputed_json_contract = json.loads(
        json.dumps(
            {"contract_sha256": recomputed_digest, **recomputed_payload},
            sort_keys=True,
        )
    )
    require(
        saved_contract == recomputed_json_contract
        and manifest.get("contract_sha256") == recomputed_digest,
        "frozen expected-band contract hash does not close",
        checks,
        "contract_hash_closes",
    )
    source_states = pd.read_csv(band.DEFAULT_STATES)
    states_2021 = source_states[source_states.dataset.astype(str) == "2021"]
    optimized_source = Path(str(states_2021.source_ledger_path.iloc[0]))
    config = band.load_config(band.DEFAULT_CARD)
    require(
        len(states_2021) == 201
        and np.array_equal(
            np.rint(1000.0 * states_2021.mass_GeV.to_numpy(float)).astype(int),
            np.arange(50, 251),
        )
        and set(states_2021.gp_support_low_MeV.astype(int)) == {36}
        and set(states_2021.gp_support_high_MeV.astype(int)) == {300}
        and set(states_2021.source_state.astype(str))
        == {"v4p9p5_frozen_support_repaired_observed_state"}
        and set(states_2021.source_ledger_sha256.astype(str))
        == {"28e6a10b8633fc69c1bab62d32fe39417c42ac886ef27f74ca0c9aeb7cc620e9"}
        and optimized_source.is_file()
        and sha256(optimized_source)
        == "28e6a10b8633fc69c1bab62d32fe39417c42ac886ef27f74ca0c9aeb7cc620e9"
        and tuple(float(value) for value in config.range_2021) == (0.05, 0.25)
        and tuple(float(value) for value in config.data_range_2021) == (0.036, 0.3)
        and float(config.kernel_ls_res_upper_factor_by_dataset["2021"]) == 15.0,
        "2021 input is not the latest optimized-support 10% configuration",
        checks,
        "optimized_2021_configuration_exact",
    )

    limits = pd.read_csv(paths["limits"], dtype={"dataset_set": str}, low_memory=False)
    draws = pd.read_csv(paths["draws"])
    predictions = pd.read_csv(paths["predictions"])
    summary = pd.read_csv(paths["summary"])
    current_summary = pd.read_csv(paths["summary_current"])
    pd.testing.assert_frame_equal(summary, current_summary, check_exact=True)
    checks["current_summary_matches_stage"] = True
    band.validate_raw_frames(limits, draws, predictions, target_toys)
    checks["raw_grid_and_numerical_gates"] = True
    if target_toys == 300:
        archive_record = json.loads((HERE / "qa/ledger_archive_300toys.json").read_text())
        archive = HERE / archive_record["archive_path"]
        raw_digest = hashlib.sha256()
        with gzip.open(archive, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                raw_digest.update(block)
        require(
            sha256(archive) == archive_record["archive_sha256"]
            and raw_digest.hexdigest() == archive_record["raw_sha256"]
            == manifest["artifacts_sha256"]["limits"]
            and archive_record["round_trip_exact"] is True,
            "compressed toy ledger fails its exact round-trip check", checks,
            "compressed_limit_ledger_round_trip_exact",
        )
    expected_limit_rows = target_toys * 680
    expected_draw_rows = target_toys * 415
    require(
        len(limits) == expected_limit_rows
        and len(draws) == expected_draw_rows
        and len(predictions) == 415
        and len(summary) == 680,
        "release row counts are not exact",
        checks,
        "exact_row_counts",
    )
    require(
        set(limits.limit_solver.astype(str)) == {band.SOLVER_VERSION}
        and manifest.get("solver_version") == band.SOLVER_VERSION,
        "toy ledger or manifest does not use the contracted band solver",
        checks,
        "band_solver_version_exact",
    )
    solver_counter_totals: Dict[str, int] = {}
    for encoded in limits.solver_counter_delta.astype(str):
        for key, value in json.loads(encoded).items():
            solver_counter_totals[key] = (
                int(solver_counter_totals.get(key, 0)) + int(value)
            )
    retry_keys = {
        "bounded_free_centered_retries",
        "unbounded_free_centered_retries",
        "null_centered_retries",
    }
    require(
        manifest.get("solver_counter_totals")
        == dict(sorted(solver_counter_totals.items()))
        and retry_keys.issubset(solver_counter_totals)
        and (
            target_toys < 100
            or int(solver_counter_totals["unbounded_free_centered_retries"]) >= 1
        ),
        "solver counter totals or centered free-profile retry evidence do not close",
        checks,
        "solver_counter_totals_and_free_retry_evidence",
    )
    require(
        not draws.latent_clip_fallback.astype(bool).any(),
        "a released latent draw used clipping",
        checks,
        "zero_latent_clip_fallbacks",
    )

    for key, expected_hash in dict(manifest["artifacts_sha256"]).items():
        artifact = paths[key]
        if sha256(artifact) != expected_hash:
            raise RuntimeError(f"run-manifest artifact hash failed: {key}")
    checks["run_manifest_artifact_hashes"] = True

    observed = pd.read_csv(band.DEFAULT_OBSERVED)
    recomputed_summary = band.summarize(limits, observed, target_toys)
    pd.testing.assert_frame_equal(
        summary,
        recomputed_summary,
        check_exact=False,
        rtol=5.0e-13,
        atol=0.0,
    )
    checks["quantiles_recompute_within_csv_roundoff"] = True
    quantiles = summary[list(band.QUANTILE_COLUMNS)].to_numpy(float)
    require(
        np.isfinite(quantiles).all()
        and np.all(quantiles > 0.0)
        and np.all(np.diff(quantiles, axis=1) >= 0.0),
        "band quantiles are non-finite, non-positive, or unordered",
        checks,
        "quantiles_finite_positive_ordered",
    )

    diagnostics = pd.read_csv(pvalue_path)
    require(
        len(diagnostics) == len(summary)
        and set(diagnostics.n_toys.astype(int)) == {target_toys}
        and set(diagnostics.toy_id_min.astype(int)) == {0}
        and set(diagnostics.toy_id_max.astype(int)) == {target_toys - 1},
        "p-value diagnostic grid or toy range is not exact",
        checks,
        "pvalue_grid_and_toy_range",
    )
    obs_for_tail = observed[["scope_key", "mass_MeV", "eps2_90"]]
    tail_rows = limits.merge(
        obs_for_tail,
        on=["scope_key", "mass_MeV"],
        how="left",
        validate="many_to_one",
        suffixes=("_toy", "_observed"),
    )
    tail_rows["strong"] = tail_rows.eps2_90_toy <= tail_rows.eps2_90_observed
    tail_rows["weak"] = tail_rows.eps2_90_toy >= tail_rows.eps2_90_observed
    tail_counts = (
        tail_rows.groupby(["scope_key", "mass_MeV"], as_index=False, sort=True)
        .agg(
            n_toys=("toy_id", "count"),
            n_strong=("strong", "sum"),
            n_weak=("weak", "sum"),
        )
        .sort_values(["scope_key", "mass_MeV"])
        .reset_index(drop=True)
    )
    diagnostic_sorted = diagnostics.sort_values(
        ["scope_key", "mass_MeV"]
    ).reset_index(drop=True)
    require(
        diagnostic_sorted[["scope_key", "mass_MeV"]].astype(str).to_numpy().tolist()
        == tail_counts[["scope_key", "mass_MeV"]].astype(str).to_numpy().tolist()
        and np.array_equal(
            diagnostic_sorted[["n_toys", "n_strong", "n_weak"]].to_numpy(int),
            tail_counts[["n_toys", "n_strong", "n_weak"]].to_numpy(int),
        ),
        "empirical tail counts do not recompute from the toy ledger",
        checks,
        "empirical_tail_counts_recompute",
    )
    expected_strong = tail_counts.n_strong.to_numpy(float) / target_toys
    expected_weak = tail_counts.n_weak.to_numpy(float) / target_toys
    expected_two = np.clip(
        2.0 * np.minimum(expected_strong, expected_weak), 0.0, 1.0
    )
    require(
        np.allclose(
            diagnostic_sorted.p_strong,
            expected_strong,
            rtol=0.0,
            atol=1.0e-15,
        )
        and np.allclose(
            diagnostic_sorted.p_weak,
            expected_weak,
            rtol=0.0,
            atol=1.0e-15,
        )
        and np.allclose(
            diagnostic_sorted.p_two,
            expected_two,
            rtol=0.0,
            atol=1.0e-15,
        )
        and np.allclose(
            diagnostic_sorted.empirical_p_resolution,
            1.0 / target_toys,
            rtol=0.0,
            atol=1.0e-15,
        ),
        "empirical p-value fractions or resolution do not recompute",
        checks,
        "empirical_pvalue_fractions_recompute",
    )
    observed_diagnostics = observed[
        [
            "scope_key",
            "mass_MeV",
            "eps2_90",
            "p0_local_asymptotic",
            "Z_local_asymptotic",
            "pvalue_method",
        ]
    ].sort_values(["scope_key", "mass_MeV"]).reset_index(drop=True)
    require(
        diagnostic_sorted[["scope_key", "mass_MeV"]].astype(str).to_numpy().tolist()
        == observed_diagnostics[["scope_key", "mass_MeV"]]
        .astype(str)
        .to_numpy()
        .tolist()
        and np.allclose(
            diagnostic_sorted.eps2_observed,
            observed_diagnostics.eps2_90,
            rtol=0.0,
            atol=1.0e-15,
        )
        and np.allclose(
            diagnostic_sorted.p0_local_asymptotic,
            observed_diagnostics.p0_local_asymptotic,
            rtol=0.0,
            atol=1.0e-15,
        )
        and np.allclose(
            diagnostic_sorted.Z_local_asymptotic,
            observed_diagnostics.Z_local_asymptotic,
            rtol=0.0,
            atol=1.0e-15,
        )
        and diagnostic_sorted.pvalue_method.astype(str).tolist()
        == observed_diagnostics.pvalue_method.astype(str).tolist()
        and set(diagnostic_sorted.pvalue_method.astype(str))
        == {"fixed-mass local asymptotic one-sided profile LRT"},
        "analytic local p0/Z provenance does not match the frozen observed ledger",
        checks,
        "analytic_local_pvalue_provenance_exact",
    )

    total_window = pd.read_csv(total_window_path)
    require(
        len(total_window) == 232
        and np.array_equal(total_window.mass_MeV.to_numpy(int), np.arange(19, 251))
        and set(total_window.construction.astype(str))
        == {"maximal_available_final_dataset_scope_at_each_mass"},
        "total-search-window ledger does not have the exact 19--250 MeV grid",
        checks,
        "total_search_window_grid_exact",
    )
    for low, high, scope in TOTAL_RULE:
        selected = total_window[
            total_window.mass_MeV.between(low, high)
        ].sort_values("mass_MeV")
        if set(selected.selected_scope_key.astype(str)) != {scope}:
            raise RuntimeError("total-search-window scope stitching rule drifted")
        source_rows = summary[
            (summary.scope_key == scope) & summary.mass_MeV.between(low, high)
        ].sort_values("mass_MeV")
        for column in (
            "eps2_observed",
            "expected_q025",
            "expected_q16",
            "expected_median",
            "expected_q84",
            "expected_q975",
        ):
            if not np.allclose(
                selected[column],
                source_rows[column],
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise RuntimeError(
                    f"total-search-window values do not match selected source: {column}"
                )
    checks["total_search_window_stitching_and_values_exact"] = True

    draw_hash = {
        (str(row.dataset), int(row.mass_MeV), int(row.toy_id)): str(
            row.observation_sha256
        )
        for row in draws.itertuples(index=False)
    }
    for row in limits.itertuples(index=False):
        recorded = json.loads(str(row.observation_sha256_by_dataset))
        expected_keys = str(row.dataset_set).split("+")
        if set(recorded) != set(expected_keys):
            raise RuntimeError("scope observation-hash set is not exact")
        for dataset in expected_keys:
            key = (dataset, int(row.mass_MeV), int(row.toy_id))
            if draw_hash.get(key) != recorded[dataset]:
                raise RuntimeError("a scope did not reuse its paired dataset draw")
    checks["paired_constituent_draw_hashes"] = True
    validate_prefix(limits, draws, target_toys, checks)
    validate_global_diagnostics(target_toys, checks)

    figure_manifest_path = HERE / "figures" / f"figure_manifest_{target_toys}toys.json"
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    expected_figure_stems = {
        f"all_three_expected_bands_{target_toys}toys",
        f"individual_expected_band_panels_{target_toys}toys",
        f"combination_expected_band_panels_{target_toys}toys",
        f"final_total_search_window_expected_bands_{target_toys}toys",
        f"combination_pvalue_panels_{target_toys}toys",
        f"individual_pvalue_panels_{target_toys}toys",
    }
    require(
        int(figure_manifest.get("stage_toys_per_mass", -1)) == target_toys
        and set(figure_manifest.get("figures", [])) == expected_figure_stems
        and figure_manifest.get("source_summary_sha256") == sha256(paths["summary"])
        and figure_manifest.get("source_toy_limits_sha256") == sha256(paths["limits"])
        and figure_manifest.get("source_observed_sha256")
        == sha256(band.DEFAULT_OBSERVED)
        and figure_manifest.get("pvalue_diagnostics_sha256") == sha256(pvalue_path)
        and figure_manifest.get("total_search_window_summary_sha256")
        == sha256(total_window_path)
        and figure_manifest.get("layout")
        == "one curve family per axis; figure-level legends outside data regions",
        "figure manifest does not bind the stage summary and non-overlap layout",
        checks,
        "figure_manifest_closes",
    )
    for stem in figure_manifest["figures"]:
        pdf = HERE / "figures" / f"{stem}.pdf"
        png = HERE / "figures" / f"{stem}.png"
        if not pdf.is_file() or not png.is_file() or len(PdfReader(str(pdf)).pages) != 1:
            raise RuntimeError(f"figure PDF/PNG pair is missing or malformed: {stem}")
        with Image.open(png) as image:
            if image.width < 1800 or image.height < 900:
                raise RuntimeError(f"figure raster is unexpectedly small: {stem}")
        figure_text = PdfReader(str(pdf)).pages[0].extract_text() or ""
        if any(fragment in figure_text for fragment in
               ("toys per mass", "Outer quantiles", "conditional on frozen", "One-sided empirical")):
            raise RuntimeError(f"explanatory footer remains in figure: {stem}")
    checks["figure_files_and_resolution"] = True
    checks["plot_explanations_moved_to_captions"] = True
    figure_blocks = re.findall(
        r"\\begin\{figure\}.*?\\end\{figure\}",
        (HERE / "note" / "results_section.tex").read_text(), re.DOTALL,
    )
    require(
        len(figure_blocks) == 6
        and all("\\StageToys" in block.split("\\caption{", 1)[1] for block in figure_blocks),
        "every figure caption must identify the toy count", checks, "all_figure_captions_state_toy_count",
    )

    note_pdf = HERE / "note" / f"HPS_GPR_Analysis_Note_v4p9p12_expected_bands_{target_toys}toys.pdf"
    output_pdf = (
        REPO
        / "output"
        / "pdf"
        / "v4p9p12_expected_bands_20260904"
        / note_pdf.name
    )
    require(
        sha256(note_pdf) == sha256(output_pdf),
        "study and output PDF copies differ",
        checks,
        "pdf_mirror_hash_matches",
    )
    reader = PdfReader(str(note_pdf))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    normalized_text = " ".join(
        extracted.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .split()
    )
    # pypdf can retain a space after an explicit compound-word hyphen when the
    # word wraps across a PDF line (for example, ``profile- likelihood``).
    normalized_text = re.sub(r"(?<=[A-Za-z])- (?=[A-Za-z])", "-", normalized_text)
    semantic_fragments = (
        "Analysis Note v4.9.12",
        f"{target_toys} background-only",
        "optimized-support 2021 10%",
        "36-300 MeV GP support",
        "pointwise expected-limit bands",
        "do not establish unconditional coverage",
        "total search window",
        "upper-limit ensemble positions",
        "one-sided, fixed-mass, asymptotic profile-likelihood",
        "Bonferroni",
        "resolution-spacing approximation",
        "scan-toy trials p-value is unavailable",
        "likelihood-equivalent centering",
        "no toy is dropped or reseeded",
        "No latent GP draw used the clipping fallback",
    )
    missing_fragments = [
        item for item in semantic_fragments if item not in normalized_text
    ]
    require(
        not missing_fragments,
        f"results-only PDF lacks semantic fragments: {missing_fragments}",
        checks,
        "pdf_semantic_text",
    )
    require(
        7 <= len(reader.pages) <= 12,
        f"unexpected results-only note page count: {len(reader.pages)}",
        checks,
        "compact_note_page_count",
    )

    qa = {
        "status": "release_validation_passed",
        "passed": True,
        "stage_toys_per_mass": target_toys,
        "checks": checks,
        "counts": {
            "scope_mass_rows": len(summary),
            "toy_limit_rows": len(limits),
            "toy_observation_rows": len(draws),
            "prediction_rows": len(predictions),
            "pvalue_rows": len(diagnostics),
            "total_search_window_rows": len(total_window),
            "note_pages": len(reader.pages),
        },
        "contract_sha256": recomputed_digest,
        "solver_counter_totals": dict(sorted(solver_counter_totals.items())),
        "note_pdf_sha256": sha256(note_pdf),
        "claim_boundary": manifest["claim_boundary"],
    }
    qa_dir = HERE / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_path = qa_dir / f"final_validation_{target_toys}toys.json"
    band.final.atomic_json(qa_path, qa)

    manifest_files = [
        HERE / "README.md",
        HERE / "STATISTICAL_PROTOCOL.md",
        HERE / "NUMERICAL_AMENDMENT_PRE_PRODUCTION.md",
        HERE / "NUMERICAL_AMENDMENT_100TOY_CONTINUATION.md",
        HERE / "band_solver.py",
        HERE / "run_expected_bands.py",
        HERE / "make_figures.py",
        HERE / "make_note_assets.py",
        HERE / "continue_balanced.py",
        HERE / "pack_stage.py",
        HERE / ".gitignore",
        HERE / "make_global_diagnostics.py",
        HERE / "GLOBAL_PVALUE_METHOD.md",
        HERE / "validate_release.py",
        contract_path,
        *paths.values(),
        pvalue_path,
        total_window_path,
        derived / f"global_pvalue_summary_{target_toys}toys.csv",
        derived / f"global_resolution_ledger_{target_toys}toys.csv",
        derived / f"global_pvalue_manifest_{target_toys}toys.json",
        HERE / "qa" / f"execution_{target_toys}toys.json",
        derived / "prediction_state_ledger.csv",
        figure_manifest_path,
        HERE / "note" / "results_section.tex",
        HERE / "note" / "generated_values.tex",
        HERE / "note" / "generated_summary.tex",
        HERE / "note" / "generated_pvalue_summary.tex",
        HERE / "note" / "generated_global_summary.tex",
        note_pdf,
        output_pdf,
        qa_path,
        *sorted((derived / "checkpoints").glob("m*.json")),
    ]
    for stem in figure_manifest["figures"]:
        manifest_files.extend(
            [HERE / "figures" / f"{stem}.pdf", HERE / "figures" / f"{stem}.png"]
        )
    if target_toys == 300:
        manifest_files.extend([
            HERE / "derived/toy_limits_300toys.csv.gz",
            HERE / "qa/ledger_archive_300toys.json",
        ])
    unique = sorted({path.resolve() for path in manifest_files})
    lines = []
    for path in unique:
        if not path.is_file():
            raise RuntimeError(f"manifest target is missing: {path}")
        lines.append(f"{sha256(path)}  {path.relative_to(REPO.resolve()).as_posix()}")
    (HERE / f"MANIFEST_{target_toys}toys.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
