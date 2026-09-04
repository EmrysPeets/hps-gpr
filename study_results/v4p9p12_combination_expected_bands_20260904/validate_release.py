#!/usr/bin/env python3
"""Fail-closed validation for a completed v4.9.12 expected-band stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_expected_bands as band  # noqa: E402


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
    previous_limits = pd.read_csv(previous_limits_path)
    previous_draws = pd.read_csv(previous_draws_path)
    current_limit_prefix = limits[limits.toy_id < previous].reset_index(drop=True)
    current_draw_prefix = draws[draws.toy_id < previous].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        previous_limits.reset_index(drop=True),
        current_limit_prefix,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        previous_draws.reset_index(drop=True),
        current_draw_prefix,
        check_exact=True,
    )
    checks["earlier_stage_prefix_preserved"] = True


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target_toys = int(args.target_toys)
    if target_toys not in band.ALLOWED_RELEASE_STAGES:
        raise SystemExit(f"release target must be one of {band.ALLOWED_RELEASE_STAGES}")
    derived = HERE / "derived"
    paths = band.stage_paths(derived, target_toys)
    required = list(paths.values()) + [
        derived / "contract.json",
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

    limits = pd.read_csv(paths["limits"])
    draws = pd.read_csv(paths["draws"])
    predictions = pd.read_csv(paths["predictions"])
    summary = pd.read_csv(paths["summary"])
    current_summary = pd.read_csv(paths["summary_current"])
    pd.testing.assert_frame_equal(summary, current_summary, check_exact=True)
    checks["current_summary_matches_stage"] = True
    band.validate_raw_frames(limits, draws, predictions, target_toys)
    checks["raw_grid_and_numerical_gates"] = True
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

    figure_manifest_path = HERE / "figures" / f"figure_manifest_{target_toys}toys.json"
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    require(
        int(figure_manifest.get("stage_toys_per_mass", -1)) == target_toys
        and figure_manifest.get("source_summary_sha256") == sha256(paths["summary"])
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
    checks["figure_files_and_resolution"] = True

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
    semantic_fragments = (
        "Analysis Note v4.9.12",
        f"{target_toys} background-only",
        "optimized-support 2021 10%",
        "36-300 MeV GP support",
        "pointwise expected-limit bands",
        "do not establish unconditional coverage",
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
        4 <= len(reader.pages) <= 8,
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
            "note_pages": len(reader.pages),
        },
        "contract_sha256": recomputed_digest,
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
        HERE / "band_solver.py",
        HERE / "run_expected_bands.py",
        HERE / "make_figures.py",
        HERE / "make_note_assets.py",
        HERE / "validate_release.py",
        contract_path,
        *paths.values(),
        derived / "prediction_state_ledger.csv",
        figure_manifest_path,
        HERE / "note" / "results_section.tex",
        HERE / "note" / "generated_values.tex",
        HERE / "note" / "generated_summary.tex",
        note_pdf,
        output_pdf,
        qa_path,
        *sorted((derived / "checkpoints").glob("m*.json")),
    ]
    for stem in figure_manifest["figures"]:
        manifest_files.extend(
            [HERE / "figures" / f"{stem}.pdf", HERE / "figures" / f"{stem}.png"]
        )
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
