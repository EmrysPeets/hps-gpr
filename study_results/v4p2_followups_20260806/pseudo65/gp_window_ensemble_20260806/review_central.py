#!/usr/bin/env python3
"""Review 55--75 MeV repeated scans by maximum finite GP LML."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CONFIG_MANIFEST = HERE / "derived" / "config_manifest.json"

import numpy as np
import pandas as pd


CENTRAL_MASSES = np.round(np.arange(0.055, 0.075 + 0.0005, 0.001), 3)
FULL_MASSES = np.round(np.arange(0.050, 0.250 + 0.0005, 0.001), 3)
LML_MATCH_ATOL = 3.0e-5
PARAM_MATCH_RTOL = 5.0e-4
PARAM_MATCH_ATOL = 1.0e-10
FINITE_COLUMNS = (
    "lml",
    "const_opt",
    "ls_opt",
    "A_hat",
    "sigma_A",
    "A_up",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
)
BOUND_COLUMNS = (
    "ls_at_lower",
    "ls_at_upper",
    "const_at_lower",
    "const_at_upper",
)


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def state_match(a: pd.Series, b: pd.Series) -> bool:
    return bool(
        abs(float(a["lml"]) - float(b["lml"])) <= LML_MATCH_ATOL
        and np.isclose(
            float(a["const_opt"]),
            float(b["const_opt"]),
            rtol=PARAM_MATCH_RTOL,
            atol=PARAM_MATCH_ATOL,
        )
        and np.isclose(
            float(a["ls_opt"]),
            float(b["ls_opt"]),
            rtol=PARAM_MATCH_RTOL,
            atol=PARAM_MATCH_ATOL,
        )
    )


def load_frame(path: Path, expected_masses: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["dataset"].astype(str) == "2021"].copy()
    frame["mass_GeV"] = np.round(frame["mass_GeV"].to_numpy(float), 3)
    frame = frame.sort_values("mass_GeV").reset_index(drop=True)
    if (
        len(frame) != len(expected_masses)
        or not np.array_equal(frame["mass_GeV"].to_numpy(float), expected_masses)
    ):
        raise RuntimeError(f"Unexpected mass grid in {path}")
    return frame


def finite_extract_row(row: pd.Series) -> bool:
    return bool(
        np.isfinite(np.asarray([float(row[key]) for key in FINITE_COLUMNS])).all()
        and bool_value(row["extract_success"])
        and str(row["cls_calibration"]) == "asymptotic"
    )


def row_at_bound(row: pd.Series) -> bool:
    return any(bool_value(row[column]) for column in BOUND_COLUMNS)


def state_clusters(
    candidates: list[tuple[str, pd.Series, Path]],
) -> list[list[int]]:
    """Connected components under the symmetric state-match relation."""
    clusters: list[list[int]] = []
    remaining = set(range(len(candidates)))
    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            matched = [
                index
                for index in remaining
                if state_match(
                    candidates[current][1], candidates[index][1]
                )
            ]
            for index in matched:
                remaining.remove(index)
                cluster.append(index)
                frontier.append(index)
        clusters.append(sorted(cluster))
    return clusters


def main() -> None:
    (HERE / "derived" / "reviewed").mkdir(parents=True, exist_ok=True)
    manifest = json.loads(CONFIG_MANIFEST.read_text())
    all_reviewed = []
    central_records = []
    draw_summaries = []
    for item in manifest["records"]:
        window = item["window"]
        draw_index = int(item["draw_index"])
        full_path = REPO / item["output_dir"] / "results_single.csv"
        repeat_path = (
            HERE
            / "central_repeat"
            / window
            / f"draw_{draw_index:02d}"
            / "attempt_02"
            / "results_single.csv"
        )
        full = load_frame(full_path, FULL_MASSES)
        repeat = load_frame(repeat_path, CENTRAL_MASSES)
        reviewed_rows = []
        selected_from_repeat = 0
        branch_multiplicity_two = 0
        reproduced_count = 0
        unreproduced_count = 0
        selected_bound_count = 0
        for _, full_row in full.iterrows():
            mass = float(full_row["mass_GeV"])
            if mass not in set(CENTRAL_MASSES):
                output = full_row.to_dict()
                output.update(
                    {
                        "review_selected_source": repo_relative(full_path),
                        "review_attempt_count": 1,
                        "review_finite_attempt_count": int(
                            finite_extract_row(full_row)
                        ),
                        "review_branch_multiplicity": 1,
                        "review_selected_state_reproducing_attempt_count": 1,
                        "review_status": "single_attempt_outside_55_75MeV",
                        "review_interpolated": False,
                    }
                )
                reviewed_rows.append(output)
                continue

            repeat_match = repeat[
                np.isclose(repeat["mass_GeV"].to_numpy(float), mass)
            ]
            if len(repeat_match) != 1:
                raise RuntimeError(
                    f"{window} draw {draw_index} mass {mass}: missing repeat row"
                )
            repeat_row = repeat_match.iloc[0]
            candidates = [
                ("attempt_01_full", full_row, full_path),
                ("attempt_02_55_75MeV", repeat_row, repeat_path),
            ]
            mass_mev = int(round(1000.0 * mass))
            repair_paths = sorted(
                (
                    HERE
                    / "central_repairs"
                    / window
                    / f"draw_{draw_index:02d}"
                    / f"m{mass_mev:03d}"
                ).glob("attempt_*/results_single.csv")
            )
            for repair_path in repair_paths:
                repair = pd.read_csv(repair_path)
                repair = repair[
                    (repair["dataset"].astype(str) == "2021")
                    & np.isclose(
                        repair["mass_GeV"].to_numpy(float), mass
                    )
                ]
                if len(repair) != 1:
                    raise RuntimeError(
                        f"{repair_path}: expected one 2021 row at {mass}"
                    )
                candidates.append(
                    (
                        f"{repair_path.parent.name}_targeted",
                        repair.iloc[0],
                        repair_path,
                    )
                )
            finite_candidates = [
                candidate
                for candidate in candidates
                if finite_extract_row(candidate[1])
            ]
            if not finite_candidates:
                raise RuntimeError(
                    f"{window} draw {draw_index} mass {mass}: no finite attempt"
                )
            selected_index = max(
                range(len(finite_candidates)),
                key=lambda index: float(finite_candidates[index][1]["lml"]),
            )
            selected_label, selected, selected_path = finite_candidates[
                selected_index
            ]
            clusters = state_clusters(finite_candidates)
            selected_cluster = [
                cluster
                for cluster in clusters
                if selected_index in cluster
            ]
            if len(selected_cluster) != 1:
                raise RuntimeError(
                    f"{window} draw {draw_index} mass {mass}: "
                    "selected-state cluster ambiguity"
                )
            branch_multiplicity = len(clusters)
            reproducing_count = sum(
                state_match(selected, candidate[1])
                for candidate in finite_candidates
            )
            if branch_multiplicity > 1:
                branch_multiplicity_two += 1
            if reproducing_count >= 2:
                reproduced_count += 1
                status = (
                    "resolved_reproduced_max_lml"
                    if branch_multiplicity > 1
                    else "stable_reproduced"
                )
            else:
                unreproduced_count += 1
                status = (
                    "selected_max_finite_lml_unreproduced_"
                    f"{len(finite_candidates)}_attempts"
                )
            if selected_label != "attempt_01_full":
                selected_from_repeat += 1
            selected_bound = row_at_bound(selected)
            selected_bound_count += int(selected_bound)
            output = selected.to_dict()
            output.update(
                {
                    "review_selected_source": repo_relative(selected_path),
                    "review_selected_attempt": selected_label,
                    "review_attempt_count": len(candidates),
                    "review_finite_attempt_count": len(finite_candidates),
                    "review_branch_multiplicity": branch_multiplicity,
                    "review_selected_state_reproducing_attempt_count": reproducing_count,
                    "review_status": status,
                    "review_selected_at_kernel_bound": selected_bound,
                    "review_interpolated": False,
                    "attempt_01_lml": (
                        float(full_row["lml"])
                        if finite_extract_row(full_row)
                        else None
                    ),
                    "attempt_02_lml": (
                        float(repeat_row["lml"])
                        if finite_extract_row(repeat_row)
                        else None
                    ),
                    "review_all_attempt_sources": "|".join(
                        repo_relative(candidate[2])
                        for candidate in candidates
                    ),
                    "review_all_finite_lml": "|".join(
                        f"{float(candidate[1]['lml']):.12g}"
                        for candidate in finite_candidates
                    ),
                }
            )
            reviewed_rows.append(output)
            central_records.append(
                {
                    "window": window,
                    "draw_index": draw_index,
                    "mass_GeV": mass,
                    "attempt_01_source": repo_relative(full_path),
                    "attempt_02_source": repo_relative(repeat_path),
                    "attempt_01_finite": finite_extract_row(full_row),
                    "attempt_02_finite": finite_extract_row(repeat_row),
                    "attempt_01_lml": (
                        float(full_row["lml"])
                        if finite_extract_row(full_row)
                        else None
                    ),
                    "attempt_02_lml": (
                        float(repeat_row["lml"])
                        if finite_extract_row(repeat_row)
                        else None
                    ),
                    "targeted_repair_attempt_count": len(repair_paths),
                    "all_attempt_sources": "|".join(
                        repo_relative(candidate[2])
                        for candidate in candidates
                    ),
                    "all_finite_lml": "|".join(
                        f"{float(candidate[1]['lml']):.12g}"
                        for candidate in finite_candidates
                    ),
                    "selected_source": repo_relative(selected_path),
                    "selected_attempt": selected_label,
                    "selected_lml": float(selected["lml"]),
                    "branch_multiplicity": branch_multiplicity,
                    "selected_state_reproducing_attempt_count": reproducing_count,
                    "selected_at_kernel_bound": selected_bound,
                    "review_status": status,
                    "interpolated": False,
                }
            )

        reviewed = pd.DataFrame(reviewed_rows).sort_values("mass_GeV")
        reviewed["window"] = window
        reviewed["draw_index"] = draw_index
        output_path = (
            HERE
            / "derived"
            / "reviewed"
            / f"{window}_draw_{draw_index:02d}.csv"
        )
        reviewed.to_csv(output_path, index=False)
        all_reviewed.append(reviewed)
        draw_summaries.append(
            {
                "window": window,
                "draw_index": draw_index,
                "reviewed_mass_count": int(len(reviewed)),
                "central_mass_count": int(len(CENTRAL_MASSES)),
                "central_selected_from_repeat_count": selected_from_repeat,
                "central_branch_multiplicity_gt1_count": branch_multiplicity_two,
                "central_reproduced_state_count": reproduced_count,
                "central_unreproduced_selected_state_count": unreproduced_count,
                "central_selected_kernel_bound_count": selected_bound_count,
                "reviewed_csv": repo_relative(output_path),
            }
        )

    reviewed_all = pd.concat(all_reviewed, ignore_index=True, sort=False)
    reviewed_all = reviewed_all.sort_values(
        ["window", "draw_index", "mass_GeV"]
    )
    reviewed_path = HERE / "derived" / "reviewed_curves.csv"
    reviewed_all.to_csv(reviewed_path, index=False)
    review_frame = pd.DataFrame(central_records).sort_values(
        ["window", "draw_index", "mass_GeV"]
    )
    review_path = HERE / "derived" / "central_optimizer_review.csv"
    review_frame.to_csv(review_path, index=False)
    summary = {
        "schema_version": 1,
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "review_window_MeV": [55.0, 75.0],
        "selection_rule": "maximum finite GP log-marginal likelihood; no interpolation",
        "state_match_tolerances": {
            "lml_atol": LML_MATCH_ATOL,
            "const_ls_rtol": PARAM_MATCH_RTOL,
            "const_ls_atol": PARAM_MATCH_ATOL,
        },
        "full_grid_repeat_stability_established": False,
        "draws": draw_summaries,
        "central_review_row_count": int(len(review_frame)),
        "central_nonfinite_attempt_row_count": int(
            np.count_nonzero(
                ~review_frame["attempt_01_finite"].to_numpy(bool)
                | ~review_frame["attempt_02_finite"].to_numpy(bool)
            )
        ),
        "central_branch_multiplicity_gt1_count": int(
            np.count_nonzero(
                review_frame["branch_multiplicity"].to_numpy(int) > 1
            )
        ),
        "central_unreproduced_selected_state_count": int(
            np.count_nonzero(
                review_frame[
                    "selected_state_reproducing_attempt_count"
                ].to_numpy(int)
                < 2
            )
        ),
        "central_selected_kernel_bound_count": int(
            np.count_nonzero(
                review_frame["selected_at_kernel_bound"].to_numpy(bool)
            )
        ),
        "interpolated_row_count": int(
            np.count_nonzero(review_frame["interpolated"].to_numpy(bool))
        ),
        "reviewed_curves_csv": repo_relative(reviewed_path),
        "central_optimizer_review_csv": repo_relative(review_path),
    }
    summary["pass_finite_complete_no_bounds"] = bool(
        len(review_frame) == 20 * len(CENTRAL_MASSES)
        and summary["central_nonfinite_attempt_row_count"] == 0
        and summary["central_unreproduced_selected_state_count"] == 0
        and summary["central_selected_kernel_bound_count"] == 0
        and summary["interpolated_row_count"] == 0
    )
    summary_path = HERE / "derived" / "central_optimizer_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {reviewed_path}")
    print(f"Wrote {review_path}")
    print(f"Wrote {summary_path}")
    print(
        "central review: "
        f"rows={summary['central_review_row_count']}, "
        f"multi_branch={summary['central_branch_multiplicity_gt1_count']}, "
        f"unreproduced={summary['central_unreproduced_selected_state_count']}, "
        f"bounds={summary['central_selected_kernel_bound_count']}"
    )
    if not summary["pass_finite_complete_no_bounds"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
