#!/usr/bin/env python3
"""Apply the frozen v4.9.10 factor and lower-support decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC_PATH = REPO / "study_configs/v4p9p10_2016_full_low_control_confirmation_20260902/study_spec.json"
EXPECTED_SPEC_SHA = "680444ef63267cabd88830c0cd5e54ee40b495e8caa3a1b30b0c0ed1a016e33e"
EXPECTED_PROTOCOL_SHA = "3bc17d683faf50195b632416a7cbb96fb5463a93d714fcb3bff45ef5f2ec8d84"
QUALIFICATION_SCRIPT_PATH = HERE / "run_qualification.py"


class AnalysisError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cells(path: Path, *, expected_mode: str, expected_factor: float) -> pd.DataFrame:
    manifest = load_json(path.parent / "run_manifest.json")
    if manifest.get("status") != "complete":
        raise AnalysisError(f"run did not complete: {path.parent}")
    if manifest.get("mode") != expected_mode:
        raise AnalysisError(f"run mode mismatch: {path.parent}")
    if float(manifest.get("upper_factor")) != float(expected_factor):
        raise AnalysisError(f"run factor mismatch: {path.parent}")
    if manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA:
        raise AnalysisError(f"run protocol hash mismatch: {path.parent}")
    if manifest.get("spec_sha256") != EXPECTED_SPEC_SHA:
        raise AnalysisError(f"run spec hash mismatch: {path.parent}")
    if manifest.get("script_sha256") != sha256_file(QUALIFICATION_SCRIPT_PATH):
        raise AnalysisError(f"qualification script drift: {path.parent}")
    if manifest["outputs"]["selected_cells.csv"]["sha256"] != sha256_file(path):
        raise AnalysisError(f"selected-cell hash mismatch: {path}")
    attempts = path.parent / "optimizer_attempts.csv"
    if manifest["outputs"]["optimizer_attempts.csv"]["sha256"] != sha256_file(attempts):
        raise AnalysisError(f"optimizer-attempt hash mismatch: {attempts}")
    frame = pd.read_csv(path)
    if len(frame) != int(manifest["outputs"]["selected_cells.csv"]["rows"]):
        raise AnalysisError(f"selected-cell row mismatch: {path}")
    return frame


def analyze_factor(factor: float) -> None:
    spec = load_json(SPEC_PATH)
    if sha256_file(SPEC_PATH) != EXPECTED_SPEC_SHA:
        raise AnalysisError("spec hash mismatch")
    path = HERE / "derived" / f"length_factor_{factor:g}" / "selected_cells.csv"
    frame = load_cells(path, expected_mode="length", expected_factor=factor)
    expected = len(spec["shortlist_lower_edges_MeV"]) * len(
        spec["development_production_geometry_anchors_GeV"]
    )
    if len(frame) != expected:
        raise AnalysisError(f"factor {factor:g}: expected {expected} cells")
    if frame.duplicated(["support_lower_MeV", "anchor_GeV"]).any():
        raise AnalysisError("duplicate factor cells")
    all_without_upper = bool(frame["technical_without_length_upper_pass"].astype(bool).all())
    upper_contacts = int(frame["length_at_upper"].astype(bool).sum())
    all_pass = bool(frame["technical_pass"].astype(bool).all())
    warning_free_min = int(frame["warning_free_repeat_count"].min())
    reproduced_min = int(frame["reproduced_warning_free_count"].min())
    factors = [float(item) for item in spec["candidate_upper_factors"]]
    index = factors.index(float(factor))
    if float(factor) == float(spec["upper_factor_default"]) and all_pass:
        status = "factor_frozen"
        selected = factor
        next_factor = None
        reason = "default factor 12 passed every warning-free reproduction, covariance, and nonbinding gate"
    elif not all_without_upper:
        status = "stopped_non_upper_bound_technical_failure"
        selected = None
        next_factor = None
        reason = "at least one cell failed for a reason not uniquely attributable to an upper-length contact"
    elif upper_contacts == 0:
        # A nondefault factor cannot freeze without the next-factor plateau.
        if index + 1 >= len(factors):
            status = "stopped_no_next_factor_plateau"
            selected = None
            next_factor = None
            reason = "first nonbinding factor has no declared next-factor plateau check"
        else:
            next_path = HERE / "derived" / f"length_factor_{factors[index + 1]:g}" / "selected_cells.csv"
            if not next_path.is_file():
                status = "next_factor_required_for_plateau"
                selected = None
                next_factor = factors[index + 1]
                reason = "nondefault nonbinding factor requires its next declared factor"
            else:
                next_frame = load_cells(
                    next_path, expected_mode="length", expected_factor=factors[index + 1]
                )
                keys = ["support_lower_MeV", "anchor_GeV"]
                merged = frame.merge(next_frame, on=keys, suffixes=("_low", "_next"), validate="one_to_one")
                plateau = (
                    np.abs(np.log(merged["selected_length_next"] / merged["selected_length_low"]))
                    < float(spec["plateau_abs_log_length_max"])
                ) & (
                    np.abs(merged["selected_lml_next"] - merged["selected_lml_low"])
                    / merged["n_train_low"]
                    < float(spec["plateau_abs_delta_lml_per_train_max"])
                )
                next_clean = bool(next_frame["technical_pass"].astype(bool).all())
                if bool(plateau.all()) and next_clean:
                    status = "factor_frozen"
                    selected = factor
                    next_factor = None
                    reason = "first nonbinding expanded factor passed the next-factor plateau in every cell"
                else:
                    status = "stopped_plateau_failure"
                    selected = None
                    next_factor = None
                    reason = "expanded factor failed the required next-factor plateau"
    else:
        if index + 1 >= len(factors):
            status = "stopped_upper_bound_at_largest_factor"
            selected = None
            next_factor = None
            reason = "largest declared factor remains upper-bound constrained"
        else:
            status = "next_factor_authorized"
            selected = None
            next_factor = factors[index + 1]
            reason = "every other gate passed and at least one reproduced branch occupied the upper length bound"
    decision = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluated_upper_factor": factor,
        "selected_upper_factor": selected,
        "next_factor_authorized": next_factor,
        "reason": reason,
        "cells": len(frame),
        "all_technical_without_upper_pass": all_without_upper,
        "all_technical_pass": all_pass,
        "upper_length_contacts": upper_contacts,
        "minimum_warning_free_repeats": warning_free_min,
        "minimum_reproduced_warning_free_repeats": reproduced_min,
        "selected_cells_sha256": sha256_file(path),
        "protocol_sha256": "3bc17d683faf50195b632416a7cbb96fb5463a93d714fcb3bff45ef5f2ec8d84",
        "spec_sha256": EXPECTED_SPEC_SHA,
        "script_sha256": sha256_file(Path(__file__)),
        "selection_metrics_used": [
            "optimizer_warning_count", "LML_reproduction", "length_reproduction",
            "kernel_bound_contacts", "predictive_covariance_validity"
        ],
        "selection_metrics_forbidden": spec["forbidden_selection_metrics"],
    }
    output = HERE / "derived/length_factor_decision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


def block_se(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def summarize_support(frame: pd.DataFrame, support: int, spec: dict[str, Any]) -> dict[str, Any]:
    cells = frame.loc[frame["support_lower_MeV"] == support].copy()
    if len(cells) != 20:
        raise AnalysisError(f"support {support}: expected 20 confirmation cells")
    block = cells.groupby("block", as_index=False).agg(
        nlpd_per_bin=("nlpd_per_bin", "mean"),
        poisson_deviance_per_bin=("poisson_deviance_per_bin", "mean"),
        mahalanobis_per_bin=("mahalanobis_per_bin", "mean"),
        anchors=("anchor_GeV", "count"),
    )
    if len(block) != 4 or not (block["anchors"] == 5).all():
        raise AnalysisError(f"support {support}: incomplete blocks")
    technical = bool(cells["technical_pass"].astype(bool).all())
    mean_mahal = float(cells["mahalanobis_per_bin"].mean())
    max_mahal = float(cells["mahalanobis_per_bin"].max())
    max_marginal = float(cells["max_abs_marginal_standardized_residual"].max())
    absolute = bool(
        mean_mahal < float(spec["mean_mahalanobis_per_bin_strict_max"])
        and max_mahal < float(spec["individual_anchor_block_mahalanobis_per_bin_exclusive_max"])
        and max_marginal < float(spec["max_abs_marginal_standardized_residual_exclusive_max"])
    )
    return {
        "support_lower_MeV": support,
        "technical_pass": technical,
        "absolute_predictive_guard_pass": absolute,
        "mean_mahalanobis_per_bin": mean_mahal,
        "max_anchor_block_mahalanobis_per_bin": max_mahal,
        "max_abs_marginal_standardized_residual": max_marginal,
        "mean_nlpd_per_bin": float(cells["nlpd_per_bin"].mean()),
        "mean_poisson_deviance_per_bin": float(cells["poisson_deviance_per_bin"].mean()),
        "minimum_warning_free_repeats": int(cells["warning_free_repeat_count"].min()),
        "minimum_reproduced_warning_free_repeats": int(cells["reproduced_warning_free_count"].min()),
        "block_scores": block.sort_values("block").to_dict(orient="records"),
    }


def analyze_confirmation() -> None:
    spec = load_json(SPEC_PATH)
    factor_decision_path = HERE / "derived/length_factor_decision.json"
    factor_decision = load_json(factor_decision_path)
    if factor_decision["status"] != "factor_frozen":
        raise AnalysisError("length factor did not freeze")
    factor = float(factor_decision["selected_upper_factor"])
    cells_path = HERE / "derived/full_low_control_confirmation/selected_cells.csv"
    frame = load_cells(cells_path, expected_mode="confirmation", expected_factor=factor)
    if len(frame) != 40 or set(frame["support_lower_MeV"].astype(int)) != {29, 30}:
        raise AnalysisError("confirmation shortlist or cell count drift")
    if int(frame["n_search_train_centers"].sum()) != 0 or int(frame["n_search_score_centers"].sum()) != 0:
        raise AnalysisError("search center entered full confirmation")
    summaries = {support: summarize_support(frame, support, spec) for support in (29, 30)}
    blocks29 = pd.DataFrame(summaries[29]["block_scores"]).set_index("block")
    blocks30 = pd.DataFrame(summaries[30]["block_scores"]).set_index("block")
    nlpd_delta = (blocks30["nlpd_per_bin"] - blocks29["nlpd_per_bin"]).to_numpy(float)
    poisson_delta = (
        blocks30["poisson_deviance_per_bin"] - blocks29["poisson_deviance_per_bin"]
    ).to_numpy(float)
    mean_delta = float(np.mean(nlpd_delta))
    se_delta = block_se(nlpd_delta)
    deletion = [float(np.mean(np.delete(nlpd_delta, index))) for index in range(4)]
    relative_pass = bool(
        mean_delta > float(spec["nlpd_improvement_se_required"]) * se_delta
        and float(np.mean(poisson_delta)) >= 0.0
        and all(item > 0.0 for item in deletion)
    )
    both_pass = bool(
        summaries[29]["technical_pass"]
        and summaries[29]["absolute_predictive_guard_pass"]
        and summaries[30]["technical_pass"]
        and summaries[30]["absolute_predictive_guard_pass"]
    )
    if both_pass and relative_pass:
        status = "support_frozen"
        selected = 29
        reason = "29 MeV passed every absolute/technical gate and the frozen paired displacement rule"
    elif summaries[30]["technical_pass"] and summaries[30]["absolute_predictive_guard_pass"]:
        status = "support_frozen_default_30"
        selected = 30
        reason = "29 MeV did not pass every displacement rule; the reviewed 30 MeV default passed"
    else:
        status = "stopped_no_support"
        selected = None
        reason = "the 30 MeV fallback failed a required technical or absolute gate"
    summary_frame = pd.DataFrame(
        [{key: value for key, value in record.items() if key != "block_scores"} for record in summaries.values()]
    )
    summary_path = HERE / "derived/full_low_control_confirmation/support_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    decision = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_support_lower_MeV": selected,
        "selected_support_upper_MeV": 210 if selected is not None else None,
        "selected_upper_factor": factor if selected is not None else None,
        "reason": reason,
        "support_summaries": summaries,
        "paired_nlpd_improvement_30_minus_29": mean_delta,
        "paired_nlpd_improvement_se": se_delta,
        "paired_nlpd_improvement_z": mean_delta / se_delta if se_delta > 0 else math.inf,
        "paired_poisson_deviance_improvement_30_minus_29": float(np.mean(poisson_delta)),
        "leave_one_block_out_nlpd_improvements": deletion,
        "relative_displacement_pass": relative_pass,
        "both_absolute_and_technical_pass": both_pass,
        "search_train_centers_used": 0,
        "search_score_centers_used": 0,
        "phase_c_authorized": selected is not None,
        "claim_boundary": "same-experiment out-of-search control confirmation; final inference remains conditional on partially unblinded related-sample model selection",
        "factor_decision_sha256": sha256_file(factor_decision_path),
        "selected_cells_sha256": sha256_file(cells_path),
        "support_summary_sha256": sha256_file(summary_path),
        "protocol_sha256": "3bc17d683faf50195b632416a7cbb96fb5463a93d714fcb3bff45ef5f2ec8d84",
        "spec_sha256": EXPECTED_SPEC_SHA,
        "script_sha256": sha256_file(Path(__file__)),
    }
    output = HERE / "derived/support_freeze_decision.json"
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("factor", "confirmation"))
    parser.add_argument("--factor", type=float, default=12.0)
    args = parser.parse_args()
    if args.mode == "factor":
        analyze_factor(args.factor)
    else:
        analyze_confirmation()


if __name__ == "__main__":
    main()
