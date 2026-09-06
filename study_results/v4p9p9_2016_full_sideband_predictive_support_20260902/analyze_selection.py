#!/usr/bin/env python3
"""Apply the frozen sequential support-selection rules to stored CV scores."""

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
CONFIG = REPO / "study_configs/v4p9p9_2016_full_sideband_predictive_support_20260902"
SPEC_PATH = CONFIG / "study_spec.json"
AMENDMENT2_PATH = CONFIG / "preexecution_amendment2.json"


class SelectionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_se(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return math.nan
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def load_stage(stage: str) -> pd.DataFrame:
    root = HERE / "derived" / stage
    manifest = load_json(root / "run_manifest.json")
    path = root / "selected_predictive_scores.csv"
    declared = manifest["outputs"]["selected_predictive_scores.csv"]
    if int(declared["rows"]) != sum(1 for _ in path.open(encoding="utf-8")) - 1:
        raise SelectionError(f"row-count mismatch for {stage}")
    if declared["sha256"] != sha256_file(path):
        raise SelectionError(f"score-ledger hash mismatch for {stage}")
    frame = pd.read_csv(path)
    if (frame["n_forbidden_search_train_centers"] != 0).any():
        raise SelectionError(f"forbidden training center in {stage}")
    if (frame["n_forbidden_search_score_centers"] != 0).any():
        raise SelectionError(f"forbidden score center in {stage}")
    return frame


def summarize_stage(
    frame: pd.DataFrame,
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = load_json(SPEC_PATH)
    adequacy = load_json(AMENDMENT2_PATH)["absolute_predictive_adequacy"]
    mean_mahal_max = float(adequacy["mean_mahalanobis_per_bin_strict_max"])
    cell_mahal_max = float(
        adequacy["individual_anchor_block_mahalanobis_per_bin_exclusive_max"]
    )
    marginal_max = float(
        adequacy["max_abs_marginal_standardized_residual_exclusive_max"]
    )
    low = frame.loc[frame["region"] == "low"].copy()
    high = frame.loc[frame["region"] == "high"].copy()
    if low.empty or high.empty:
        raise SelectionError(f"missing low/high cells in {stage}")
    if low.duplicated(["support_lower_MeV", "anchor_GeV", "block"]).any():
        raise SelectionError(f"duplicate low cells in {stage}")
    if high.duplicated(["anchor_GeV", "block"]).any():
        raise SelectionError(f"duplicate high cells in {stage}")

    block = (
        low.groupby(["support_lower_MeV", "block"], as_index=False)
        .agg(
            nlpd_per_bin=("nlpd_per_bin", "mean"),
            mahalanobis_per_bin=("mahalanobis_per_bin", "mean"),
            poisson_deviance_per_bin=("poisson_deviance_per_bin", "mean"),
            max_abs_marginal_standardized_residual=(
                "max_abs_marginal_standardized_residual",
                "max",
            ),
            anchors=("anchor_GeV", "count"),
        )
        .sort_values(["support_lower_MeV", "block"])
        .reset_index(drop=True)
    )
    reference = block.loc[block["support_lower_MeV"] == 30].set_index("block")
    if len(reference) != 4:
        raise SelectionError(f"reference support is incomplete in {stage}")

    records: list[dict[str, Any]] = []
    for support, cells in low.groupby("support_lower_MeV", sort=True):
        support = int(support)
        blocks = block.loc[block["support_lower_MeV"] == support].set_index("block")
        if len(cells) != 20 or len(blocks) != 4 or not (blocks["anchors"] == 5).all():
            raise SelectionError(f"incomplete support {support} in {stage}")
        joined = reference[["nlpd_per_bin", "poisson_deviance_per_bin"]].join(
            blocks[["nlpd_per_bin", "poisson_deviance_per_bin"]],
            lsuffix="_ref",
            rsuffix="_candidate",
            how="inner",
            validate="one_to_one",
        )
        nlpd_deltas = (
            joined["nlpd_per_bin_ref"] - joined["nlpd_per_bin_candidate"]
        ).to_numpy(float)
        poisson_deltas = (
            joined["poisson_deviance_per_bin_ref"]
            - joined["poisson_deviance_per_bin_candidate"]
        ).to_numpy(float)
        delta = float(np.mean(nlpd_deltas))
        se = paired_se(nlpd_deltas)
        if support == 30:
            improvement_gt_one_se = True
            leave_one_out_positive = True
            poisson_nonnegative = True
            z_score = 0.0
            loo_min = 0.0
        else:
            improvement_gt_one_se = bool(np.isfinite(se) and delta > se)
            loo = np.asarray(
                [np.mean(np.delete(nlpd_deltas, i)) for i in range(len(nlpd_deltas))],
                dtype=float,
            )
            loo_min = float(np.min(loo))
            leave_one_out_positive = bool(np.all(loo > 0.0))
            poisson_nonnegative = bool(float(np.mean(poisson_deltas)) >= 0.0)
            z_score = float(delta / se) if np.isfinite(se) and se > 0 else math.inf

        technical_pass = bool(cells["technical_pass"].astype(bool).all())
        mean_mahal = float(cells["mahalanobis_per_bin"].mean())
        max_mahal = float(cells["mahalanobis_per_bin"].max())
        max_marginal = float(cells["max_abs_marginal_standardized_residual"].max())
        absolute_pass = bool(
            np.isfinite(mean_mahal)
            and mean_mahal < mean_mahal_max
            and np.isfinite(max_mahal)
            and max_mahal < cell_mahal_max
            and np.isfinite(max_marginal)
            and max_marginal < marginal_max
        )
        relative_pass = bool(
            improvement_gt_one_se
            and delta >= 0.0
            and poisson_nonnegative
            and leave_one_out_positive
        )
        records.append(
            {
                "stage": stage,
                "support_lower_MeV": support,
                "eligible": support in [int(x) for x in spec["eligible_lower_edges_MeV"]],
                "geometry_control": support == int(spec["geometry_control_lower_edge_MeV"]),
                "technical_pass": technical_pass,
                "absolute_predictive_guard_pass": absolute_pass,
                "mean_mahalanobis_per_bin": mean_mahal,
                "max_anchor_block_mahalanobis_per_bin": max_mahal,
                "max_abs_marginal_standardized_residual": max_marginal,
                "mean_nlpd_per_bin": float(cells["nlpd_per_bin"].mean()),
                "mean_poisson_deviance_per_bin": float(
                    cells["poisson_deviance_per_bin"].mean()
                ),
                "paired_nlpd_improvement_vs_30": delta,
                "paired_nlpd_improvement_se": se,
                "paired_nlpd_improvement_z": z_score,
                "improvement_gt_one_se": improvement_gt_one_se,
                "mean_poisson_deviance_improvement_vs_30": float(
                    np.mean(poisson_deltas)
                ),
                "poisson_deviance_direction_pass": poisson_nonnegative,
                "leave_one_block_out_min_nlpd_improvement": loo_min,
                "leave_one_block_out_positive": leave_one_out_positive,
                "relative_displacement_pass": relative_pass,
                "all_candidate_rules_pass": bool(
                    technical_pass and absolute_pass and relative_pass
                ),
            }
        )
    summary = pd.DataFrame(records).sort_values("support_lower_MeV").reset_index(drop=True)

    high_summary = {
        "stage": stage,
        "n_cells": int(len(high)),
        "technical_pass": bool(high["technical_pass"].astype(bool).all()),
        "mean_mahalanobis_per_bin": float(high["mahalanobis_per_bin"].mean()),
        "max_anchor_block_mahalanobis_per_bin": float(
            high["mahalanobis_per_bin"].max()
        ),
        "max_abs_marginal_standardized_residual": float(
            high["max_abs_marginal_standardized_residual"].max()
        ),
    }
    high_summary["absolute_predictive_guard_pass"] = bool(
        np.isfinite(high_summary["mean_mahalanobis_per_bin"])
        and high_summary["mean_mahalanobis_per_bin"] < mean_mahal_max
        and np.isfinite(high_summary["max_anchor_block_mahalanobis_per_bin"])
        and high_summary["max_anchor_block_mahalanobis_per_bin"] < cell_mahal_max
        and np.isfinite(high_summary["max_abs_marginal_standardized_residual"])
        and high_summary["max_abs_marginal_standardized_residual"] < marginal_max
    )
    high_summary["all_rules_pass"] = bool(
        high_summary["technical_pass"]
        and high_summary["absolute_predictive_guard_pass"]
    )
    return summary, block, high_summary


def write_stage_summary(
    stage: str,
    summary: pd.DataFrame,
    block: pd.DataFrame,
    high: dict[str, Any],
) -> dict[str, Any]:
    out = HERE / "derived" / stage
    summary_path = out / "support_summary.csv"
    block_path = out / "block_summary.csv"
    high_path = out / "high_control_summary.json"
    summary.to_csv(summary_path, index=False)
    block.to_csv(block_path, index=False)
    high_path.write_text(json.dumps(high, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "support_summary.csv": sha256_file(summary_path),
        "block_summary.csv": sha256_file(block_path),
        "high_control_summary.json": sha256_file(high_path),
    }


def phase1() -> None:
    frame = load_stage("development")
    summary, block, high = summarize_stage(frame, "development")
    products = write_stage_summary("development", summary, block, high)
    reference = summary.loc[summary["support_lower_MeV"] == 30].iloc[0]
    reference_pass = bool(
        reference["technical_pass"]
        and reference["absolute_predictive_guard_pass"]
    )
    if not reference_pass or not high["all_rules_pass"]:
        status = "stopped_development_absolute_or_technical_failure"
        qualifiers: list[int] = []
        supports_to_confirm: list[int] = []
    else:
        eligible = summary.loc[
            summary["eligible"]
            & (summary["support_lower_MeV"] != 30)
            & summary["all_candidate_rules_pass"]
        ]
        qualifiers = [int(item) for item in eligible["support_lower_MeV"].tolist()]
        eligible_edges = set(int(item) for item in load_json(SPEC_PATH)["eligible_lower_edges_MeV"])
        supports = {30, *qualifiers}
        for item in qualifiers:
            supports.update({item - 1, item + 1})
        supports_to_confirm = sorted(supports & eligible_edges)
        status = "phase1_complete"
    decision = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "development",
        "reference_support_lower_MeV": 30,
        "reference_pass": reference_pass,
        "high_control_pass": bool(high["all_rules_pass"]),
        "phase1_qualifiers": qualifiers,
        "supports_to_confirm": supports_to_confirm,
        "search_bins_used_for_training": 0,
        "search_bins_used_for_scoring": 0,
        "selection_metrics_excluded": [
            "A_hat", "signal_pull", "p0", "Z", "upper_limit", "epsilon2", "toy_limit"
        ],
        "products_sha256": products,
        "script_sha256": sha256_file(Path(__file__)),
    }
    path = HERE / "derived/development/phase1_decision.json"
    path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


def final() -> None:
    phase1_decision = load_json(HERE / "derived/development/phase1_decision.json")
    if phase1_decision["status"] != "phase1_complete":
        raise SelectionError("phase1 did not authorize confirmation")
    development_frame = load_stage("development")
    confirmation_frame = load_stage("confirmation")
    development, dev_block, dev_high = summarize_stage(development_frame, "development")
    confirmation, conf_block, conf_high = summarize_stage(
        confirmation_frame, "confirmation"
    )
    dev_products = write_stage_summary("development", development, dev_block, dev_high)
    conf_products = write_stage_summary("confirmation", confirmation, conf_block, conf_high)
    dev_ref = development.loc[development["support_lower_MeV"] == 30].iloc[0]
    conf_ref = confirmation.loc[confirmation["support_lower_MeV"] == 30].iloc[0]
    reference_pass = bool(
        dev_ref["technical_pass"]
        and dev_ref["absolute_predictive_guard_pass"]
        and conf_ref["technical_pass"]
        and conf_ref["absolute_predictive_guard_pass"]
        and dev_high["all_rules_pass"]
        and conf_high["all_rules_pass"]
    )
    qualifiers = set(int(item) for item in phase1_decision["phase1_qualifiers"])
    confirmed_rows = confirmation.loc[
        confirmation["support_lower_MeV"].isin(qualifiers)
        & confirmation["all_candidate_rules_pass"]
    ].copy()
    confirmed = [int(item) for item in confirmed_rows["support_lower_MeV"].tolist()]

    if not reference_pass:
        status = "stopped_reference_or_high_control_failure"
        selected: int | None = None
        reason = "support 30 or a candidate-independent high control failed a required technical/absolute gate"
        ranked: list[dict[str, Any]] = []
    elif not confirmed:
        status = "reference_retained_no_common_clear_improvement"
        selected = 30
        reason = "no Phase-1 qualifier passed every full-control confirmation rule"
        ranked = []
    else:
        ranked = []
        for support in confirmed:
            dev = development.loc[development["support_lower_MeV"] == support].iloc[0]
            conf = confirmation.loc[confirmation["support_lower_MeV"] == support].iloc[0]
            score = min(
                float(dev["paired_nlpd_improvement_z"]),
                float(conf["paired_nlpd_improvement_z"]),
            )
            ranked.append(
                {
                    "support_lower_MeV": support,
                    "minimum_phase_z": score,
                    "development_z": float(dev["paired_nlpd_improvement_z"]),
                    "confirmation_z": float(conf["paired_nlpd_improvement_z"]),
                }
            )
        best = max(item["minimum_phase_z"] for item in ranked)
        tie_margin = float(load_json(SPEC_PATH)["practical_z_tie_margin"])
        tied = [item for item in ranked if item["minimum_phase_z"] >= best - tie_margin]
        tied.sort(key=lambda item: (abs(item["support_lower_MeV"] - 30), item["support_lower_MeV"]))
        selected = int(tied[0]["support_lower_MeV"])
        status = "nonreference_support_selected"
        reason = "candidate passed development and full-control predictive, technical, absolute, and deletion-stability gates"

    decision = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_support_lower_MeV": selected,
        "selected_support_upper_MeV": 210 if selected is not None else None,
        "reference_support_lower_MeV": 30,
        "reference_pass": reference_pass,
        "phase1_qualifiers": sorted(qualifiers),
        "full_control_confirmed_qualifiers": confirmed,
        "ranked_confirmed_candidates": ranked,
        "reason": reason,
        "observed_2016_production_authorized": bool(selected is not None),
        "authorization_conditions": [
            "build a card changing only data_range_2016 lower edge",
            "run every 39--180 MeV production mass with frozen support",
            "review unchanged-card optimizer repeats by maximum reproducible LML only",
            "require no kernel-bound contacts and finite predictive covariance",
            "do not select or repair using A_hat, p0, Z, or limit strength"
        ],
        "search_bins_used_for_support_training": 0,
        "search_bins_used_for_support_scoring": 0,
        "claim_boundary": "out-of-search control-region support selection for a partially unblinded analysis; not coverage, expected sensitivity, exclusion, or global-significance calibration",
        "products_sha256": {
            "development": dev_products,
            "confirmation": conf_products,
        },
        "protocol_sha256": "38e82537d04330ce66d5e39007df03bb1c7269fb62a16557e3ee96d6d2f380b2",
        "preexecution_amendment_sha256": "f45dbad8ee99f22d8500e4a8effd74b35854e2f694a8bd44ec4704e6b500d14c",
        "preexecution_amendment2_sha256": "d37a934e91d595123fe9ef543a3bb2ae7dce23aa78859d4747aa536360eb4b9e",
        "script_sha256": sha256_file(Path(__file__)),
    }
    path = HERE / "derived/final_support_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("phase1", "final"))
    args = parser.parse_args()
    if args.stage == "phase1":
        phase1()
    else:
        final()


if __name__ == "__main__":
    main()
