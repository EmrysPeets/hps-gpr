#!/usr/bin/env python3
"""Review repeated optimizer attempts and select reproduced maximum-LML states.

No interpolation is permitted.  A selected state is closed only when an
unchanged-card attempt reproduces its LML, constant, and length scale.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

import numpy as np
import pandas as pd


LANES = ("gp_mean", "functional_form")
EXPECTED_MASSES = np.round(np.arange(0.050, 0.250 + 0.0005, 0.001), 3)
LML_MATCH_ATOL = 3.0e-5
PARAM_MATCH_RTOL = 5.0e-4
PARAM_MATCH_ATOL = 1.0e-10


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def relativize_in_repo_strings(value: Any) -> Any:
    """Convert absolute paths under this checkout to repo-relative strings."""
    if isinstance(value, dict):
        return {
            str(key): relativize_in_repo_strings(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [relativize_in_repo_strings(item) for item in value]
    if isinstance(value, str):
        repo_prefix = str(REPO.resolve())
        if value == repo_prefix or value.startswith(repo_prefix + "/"):
            return str(Path(value).resolve().relative_to(REPO.resolve()))
    return value


def normalize_scan_validation_reports() -> int:
    """Relocate in-repo paths in existing scan validation reports."""
    count = 0
    for path in sorted((HERE / "runs").glob("**/validation_report.json")):
        payload = json.loads(path.read_text())
        normalized = relativize_in_repo_strings(payload)
        path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
        count += 1
    return count


def state_match(a: pd.Series, b: pd.Series) -> bool:
    required = ("lml", "const_opt", "ls_opt")
    if not all(np.isfinite(float(a[key])) and np.isfinite(float(b[key])) for key in required):
        return False
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


def discover_sources(lane: str) -> list[Path]:
    base = HERE / "runs" / lane
    sources = sorted(base.glob("attempt_*/results_single.csv"))
    sources.extend(sorted(base.glob("repairs/**/results_single.csv")))
    unique = []
    seen = set()
    for path in sources:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def load_ledger(lane: str) -> pd.DataFrame:
    sources = discover_sources(lane)
    if len(sources) < 2:
        raise RuntimeError(f"{lane}: need at least two unchanged-card attempts")
    frames = []
    for path in sources:
        frame = pd.read_csv(path)
        if "dataset" not in frame or "mass_GeV" not in frame:
            raise RuntimeError(f"Malformed scan CSV: {path}")
        frame = frame[frame["dataset"].astype(str) == "2021"].copy()
        frame["source_csv"] = str(path.relative_to(HERE))
        frame["source_sha256_pending"] = False
        frames.append(frame)
    ledger = pd.concat(frames, ignore_index=True, sort=False)
    ledger["mass_GeV"] = np.round(ledger["mass_GeV"].to_numpy(float), 3)
    return ledger.sort_values(["mass_GeV", "source_csv"]).reset_index(drop=True)


def cluster_rows(rows: pd.DataFrame) -> list[list[int]]:
    clusters: list[list[int]] = []
    for index, row in rows.iterrows():
        assigned = False
        for cluster in clusters:
            representative = rows.loc[cluster[0]]
            if state_match(row, representative):
                cluster.append(index)
                assigned = True
                break
        if not assigned:
            clusters.append([index])
    return clusters


def review_lane(lane: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ledger = load_ledger(lane)
    reviewed_rows = []
    review_records = []
    pending_masses = []

    for mass in EXPECTED_MASSES:
        rows = ledger[np.isclose(ledger["mass_GeV"], mass, atol=5.0e-10)].copy()
        finite = rows[
            np.isfinite(rows["lml"].to_numpy(float))
            & np.isfinite(rows["const_opt"].to_numpy(float))
            & np.isfinite(rows["ls_opt"].to_numpy(float))
            & rows["extract_success"].astype(bool)
        ]
        if finite.empty:
            pending_masses.append(float(mass))
            review_records.append(
                {
                    "lane": lane,
                    "mass_GeV": float(mass),
                    "n_attempt_rows": int(len(rows)),
                    "branch_multiplicity": 0,
                    "selected_state_reproducing_attempt_count": 0,
                    "review_status": "pending_no_finite_state",
                }
            )
            continue

        finite = finite.sort_values(
            ["lml", "source_csv"], ascending=[False, True]
        )
        selected = finite.iloc[0]
        clusters = cluster_rows(finite)
        selected_cluster = [
            cluster
            for cluster in clusters
            if state_match(finite.loc[cluster[0]], selected)
        ]
        if len(selected_cluster) != 1:
            raise RuntimeError(f"{lane} {mass:.3f}: selected cluster ambiguity")
        reproducing_indices = selected_cluster[0]
        reproducing = finite.loc[reproducing_indices]
        reproducing_count = int(len(reproducing))
        branch_multiplicity = int(len(clusters))
        lml_spread = float(
            np.nanmax(finite["lml"].to_numpy(float))
            - np.nanmin(finite["lml"].to_numpy(float))
        )
        selected_at_bound = bool(
            selected.get("ls_at_lower", False)
            or selected.get("ls_at_upper", False)
            or selected.get("const_at_lower", False)
            or selected.get("const_at_upper", False)
        )
        if reproducing_count < 2:
            status = "pending_unreproduced_max_lml"
            pending_masses.append(float(mass))
        elif branch_multiplicity > 1:
            status = "resolved_reproduced_max_lml"
        elif selected_at_bound:
            status = "stable_reproduced_at_bound"
        else:
            status = "stable_reproduced"

        output = selected.to_dict()
        output.update(
            {
                "selected_source": str(selected["source_csv"]),
                "selected_state_reproducing_attempt_count": reproducing_count,
                "reproducing_sources": "|".join(
                    reproducing["source_csv"].astype(str).tolist()
                ),
                "all_attempt_sources": "|".join(
                    rows["source_csv"].astype(str).tolist()
                ),
                "attempt_row_count": int(len(rows)),
                "finite_attempt_row_count": int(len(finite)),
                "branch_multiplicity": branch_multiplicity,
                "attempt_lml_spread": lml_spread,
                "optimizer_repair_applied": bool(
                    reproducing["source_csv"].astype(str).str.contains("/repairs/").any()
                ),
                "review_status": status,
                "interpolated": False,
            }
        )
        reviewed_rows.append(output)
        review_records.append(
            {
                "lane": lane,
                "mass_GeV": float(mass),
                "n_attempt_rows": int(len(rows)),
                "n_finite_attempt_rows": int(len(finite)),
                "branch_multiplicity": branch_multiplicity,
                "lml_spread": lml_spread,
                "selected_lml": float(selected["lml"]),
                "selected_const_opt": float(selected["const_opt"]),
                "selected_ls_opt": float(selected["ls_opt"]),
                "selected_source": str(selected["source_csv"]),
                "selected_state_reproducing_attempt_count": reproducing_count,
                "selected_at_kernel_bound": selected_at_bound,
                "review_status": status,
            }
        )

    reviewed = pd.DataFrame(reviewed_rows)
    review_frame = pd.DataFrame(review_records)
    if len(reviewed):
        reviewed = reviewed.sort_values("mass_GeV").reset_index(drop=True)
    out_reviewed = HERE / "derived" / f"{lane}_results_reviewed.csv"
    out_ledger = HERE / "derived" / f"{lane}_optimizer_attempt_ledger.csv"
    out_review = HERE / "derived" / f"{lane}_optimizer_review.csv"
    out_pending = HERE / "derived" / f"{lane}_repair_masses.txt"
    reviewed.to_csv(out_reviewed, index=False)
    ledger.to_csv(out_ledger, index=False)
    review_frame.to_csv(out_review, index=False)
    out_pending.write_text(
        "".join(f"{mass:.3f}\n" for mass in pending_masses)
    )

    summary = {
        "lane": lane,
        "source_csv_count": len(discover_sources(lane)),
        "source_csvs": [repo_relative(path) for path in discover_sources(lane)],
        "expected_mass_count": int(len(EXPECTED_MASSES)),
        "reviewed_mass_count": int(len(reviewed)),
        "pending_mass_count": int(len(pending_masses)),
        "pending_masses_GeV": pending_masses,
        "branch_multiplicity_gt1_count": int(
            np.count_nonzero(
                review_frame.get("branch_multiplicity", pd.Series(dtype=int)).to_numpy(int)
                > 1
            )
        ),
        "selected_at_kernel_bound_count": int(
            np.count_nonzero(
                review_frame.get(
                    "selected_at_kernel_bound", pd.Series(dtype=bool)
                ).to_numpy(bool)
            )
        ),
        "reviewed_csv": repo_relative(out_reviewed),
        "optimizer_ledger_csv": repo_relative(out_ledger),
        "optimizer_review_csv": repo_relative(out_review),
        "repair_mass_list": repo_relative(out_pending),
    }
    return reviewed, review_frame, summary


def main() -> None:
    (HERE / "derived").mkdir(parents=True, exist_ok=True)
    normalized_reports = normalize_scan_validation_reports()
    summaries = {}
    pending_total = 0
    for lane in LANES:
        _, _, summary = review_lane(lane)
        summaries[lane] = summary
        pending_total += int(summary["pending_mass_count"])
        print(
            f"{lane}: reviewed={summary['reviewed_mass_count']}/"
            f"{summary['expected_mass_count']}, "
            f"pending={summary['pending_mass_count']}, "
            f"multi-branch={summary['branch_multiplicity_gt1_count']}"
        )
    audit = {
        "schema_version": 1,
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "maximum finite GP log-marginal likelihood; no interpolation"
        ),
        "reproduction_rule": {
            "minimum_unchanged_card_rows": 2,
            "lml_absolute_tolerance": LML_MATCH_ATOL,
            "const_and_ls_relative_tolerance": PARAM_MATCH_RTOL,
            "const_and_ls_absolute_tolerance": PARAM_MATCH_ATOL,
        },
        "lanes": summaries,
        "normalized_scan_validation_report_count": normalized_reports,
        "pending_mass_count_total": pending_total,
        "pass": bool(pending_total == 0),
    }
    path = HERE / "derived" / "optimizer_audit.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {path}")
    if pending_total:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
