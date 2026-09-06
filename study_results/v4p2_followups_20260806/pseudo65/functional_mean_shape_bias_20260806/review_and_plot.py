#!/usr/bin/env python3
"""Review deterministic scans and compare them with the pseudo65 draws."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PSEUDO65 = HERE.parent
REPO = HERE.parents[3]
DERIVED = HERE / "derived"
PLOTS = HERE / "plots"

EXPECTED_MASSES = np.round(np.arange(0.055, 0.075 + 0.0005, 0.001), 3)
LANES = ("functional_mean", "gp_mean")
LML_MATCH_ATOL = 3.0e-5
PARAM_MATCH_RTOL = 5.0e-4
PARAM_MATCH_ATOL = 1.0e-10

REFERENCE_DRAWS = {
    "functional_poisson_draw": (
        PSEUDO65 / "derived" / "functional_form_results_reviewed.csv",
        "4ef284c894d8ad6be65fefc0b063cf6100add4d6fe735935e6a90363f7ad7ca1",
    ),
    "gp_poisson_draw": (
        PSEUDO65 / "derived" / "gp_mean_results_reviewed.csv",
        "7ff22bb70d7ee9c0387d20c66b6c20fd359d80af3ff7e51303db607cd88efb77",
    ),
}

SERIES = {
    "functional_mean_fractional": {
        "label": r"Functional mean (fractional diagnostic)",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "o",
    },
    "functional_poisson_draw": {
        "label": r"Functional Poisson draw",
        "color": "#D55E00",
        "linestyle": "--",
        "marker": "s",
    },
    "gp_mean_fractional": {
        "label": r"GP mean (fractional diagnostic)",
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
    },
    "gp_poisson_draw": {
        "label": r"GP-mean Poisson draw",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "s",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def state_match(a: pd.Series, b: pd.Series) -> bool:
    required = ("lml", "const_opt", "ls_opt")
    if not all(
        np.isfinite(float(a[key])) and np.isfinite(float(b[key]))
        for key in required
    ):
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
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in sources:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def load_attempts(lane: str) -> pd.DataFrame:
    sources = discover_sources(lane)
    if len(sources) < 2:
        raise RuntimeError(f"{lane}: need at least two unchanged-card attempts")
    frames = []
    for path in sources:
        frame = pd.read_csv(path)
        frame = frame[frame["dataset"].astype(str) == "2021"].copy()
        frame["source_csv"] = path.relative_to(HERE).as_posix()
        frames.append(frame)
    ledger = pd.concat(frames, ignore_index=True, sort=False)
    ledger["mass_GeV"] = np.round(ledger["mass_GeV"].to_numpy(float), 3)
    return ledger.sort_values(["mass_GeV", "source_csv"]).reset_index(drop=True)


def cluster_rows(rows: pd.DataFrame) -> list[list[int]]:
    clusters: list[list[int]] = []
    for index, row in rows.iterrows():
        for cluster in clusters:
            if state_match(row, rows.loc[cluster[0]]):
                cluster.append(index)
                break
        else:
            clusters.append([index])
    return clusters


def review_lane(lane: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = load_attempts(lane)
    reviewed_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    pending: list[float] = []
    for mass in EXPECTED_MASSES:
        rows = ledger[np.isclose(ledger["mass_GeV"], mass, atol=5.0e-10)].copy()
        finite = rows[
            np.isfinite(rows["lml"].to_numpy(float))
            & np.isfinite(rows["const_opt"].to_numpy(float))
            & np.isfinite(rows["ls_opt"].to_numpy(float))
            & rows["extract_success"].astype(bool)
        ].copy()
        if finite.empty:
            pending.append(float(mass))
            review_rows.append(
                {
                    "lane": lane,
                    "mass_GeV": mass,
                    "review_status": "pending_no_finite_state",
                }
            )
            continue
        finite = finite.sort_values(
            ["lml", "source_csv"],
            ascending=[False, True],
        )
        selected = finite.iloc[0]
        clusters = cluster_rows(finite)
        selected_cluster = [
            cluster
            for cluster in clusters
            if state_match(finite.loc[cluster[0]], selected)
        ]
        if len(selected_cluster) != 1:
            raise RuntimeError(f"{lane} {mass:.3f}: cluster ambiguity")
        reproducing = finite.loc[selected_cluster[0]]
        reproducing_count = int(len(reproducing))
        selected_at_bound = bool(
            selected.get("ls_at_lower", False)
            or selected.get("ls_at_upper", False)
            or selected.get("const_at_lower", False)
            or selected.get("const_at_upper", False)
        )
        if reproducing_count < 2:
            status = "pending_unreproduced_max_lml"
            pending.append(float(mass))
        elif len(clusters) > 1:
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
                    reproducing["source_csv"].astype(str)
                ),
                "attempt_row_count": int(len(rows)),
                "finite_attempt_row_count": int(len(finite)),
                "branch_multiplicity": int(len(clusters)),
                "selected_at_kernel_bound": selected_at_bound,
                "review_status": status,
                "interpolated": False,
            }
        )
        reviewed_rows.append(output)
        review_rows.append(
            {
                "lane": lane,
                "mass_GeV": mass,
                "attempt_row_count": int(len(rows)),
                "finite_attempt_row_count": int(len(finite)),
                "branch_multiplicity": int(len(clusters)),
                "selected_lml": float(selected["lml"]),
                "selected_const_opt": float(selected["const_opt"]),
                "selected_ls_opt": float(selected["ls_opt"]),
                "selected_source": str(selected["source_csv"]),
                "selected_state_reproducing_attempt_count": reproducing_count,
                "selected_at_kernel_bound": selected_at_bound,
                "review_status": status,
            }
        )

    reviewed = pd.DataFrame(reviewed_rows).sort_values("mass_GeV")
    review = pd.DataFrame(review_rows).sort_values("mass_GeV")
    reviewed_path = DERIVED / f"{lane}_results_reviewed.csv"
    ledger_path = DERIVED / f"{lane}_optimizer_attempt_ledger.csv"
    review_path = DERIVED / f"{lane}_optimizer_review.csv"
    pending_path = DERIVED / f"{lane}_repair_masses.txt"
    reviewed.to_csv(reviewed_path, index=False)
    ledger.to_csv(ledger_path, index=False)
    review.to_csv(review_path, index=False)
    pending_path.write_text(
        "".join(f"{mass:.3f}\n" for mass in pending),
        encoding="utf-8",
    )
    summary = {
        "lane": lane,
        "source_csvs": [repo_relative(path) for path in discover_sources(lane)],
        "expected_mass_count": int(len(EXPECTED_MASSES)),
        "reviewed_mass_count": int(len(reviewed)),
        "pending_mass_count": int(len(pending)),
        "pending_masses_GeV": pending,
        "branch_multiplicity_gt1_count": int(
            np.count_nonzero(review["branch_multiplicity"].to_numpy(int) > 1)
        ),
        "selected_at_kernel_bound_count": int(
            np.count_nonzero(
                review["selected_at_kernel_bound"].to_numpy(bool)
            )
        ),
        "reviewed_csv": repo_relative(reviewed_path),
        "reviewed_csv_sha256": sha256_file(reviewed_path),
    }
    return reviewed, summary


def load_reference_draw(path: Path, expected_hash: str) -> pd.DataFrame:
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"Reviewed comparator checksum changed: {path}")
    frame = pd.read_csv(path)
    frame = frame[
        (frame["dataset"].astype(str) == "2021")
        & frame["mass_GeV"].between(0.055, 0.075)
    ].copy()
    frame["mass_GeV"] = np.round(frame["mass_GeV"].to_numpy(float), 3)
    if not np.array_equal(frame["mass_GeV"].to_numpy(), EXPECTED_MASSES):
        raise RuntimeError(f"Comparator mass grid changed: {path}")
    return frame.sort_values("mass_GeV").reset_index(drop=True)


def build_comparison(
    deterministic: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames = {
        "functional_mean_fractional": deterministic["functional_mean"],
        "gp_mean_fractional": deterministic["gp_mean"],
    }
    for key, (path, expected_hash) in REFERENCE_DRAWS.items():
        frames[key] = load_reference_draw(path, expected_hash)

    table = pd.DataFrame({"mass_MeV": 1000.0 * EXPECTED_MASSES})
    selected_columns = (
        "A_hat",
        "sigma_A",
        "p0_analytic",
        "Z_analytic",
        "eps2_up",
        "lml",
        "const_opt",
        "ls_opt",
    )
    for key, frame in frames.items():
        if not np.array_equal(
            np.round(frame["mass_GeV"].to_numpy(float), 3),
            EXPECTED_MASSES,
        ):
            raise RuntimeError(f"{key}: mass grid mismatch")
        for column in selected_columns:
            table[f"{key}__{column}"] = frame[column].to_numpy(float)
        table[f"{key}__signed_Ahat_over_sigma"] = (
            frame["A_hat"].to_numpy(float)
            / frame["sigma_A"].to_numpy(float)
        )

    table["functional_mean_minus_gp_mean__A_hat"] = (
        table["functional_mean_fractional__A_hat"]
        - table["gp_mean_fractional__A_hat"]
    )
    table["functional_draw_minus_functional_mean__A_hat"] = (
        table["functional_poisson_draw__A_hat"]
        - table["functional_mean_fractional__A_hat"]
    )
    table["gp_draw_minus_gp_mean__A_hat"] = (
        table["gp_poisson_draw__A_hat"]
        - table["gp_mean_fractional__A_hat"]
    )
    path = DERIVED / "comparison_55_75MeV.csv"
    table.to_csv(path, index=False, float_format="%.17g")
    return table, frames


def anchor_record(table: pd.DataFrame, mass_mev: int) -> dict[str, Any]:
    row = table[np.isclose(table["mass_MeV"], mass_mev)].iloc[0]
    record: dict[str, Any] = {"mass_MeV": mass_mev}
    for key in SERIES:
        record[key] = {
            "A_hat": float(row[f"{key}__A_hat"]),
            "sigma_A": float(row[f"{key}__sigma_A"]),
            "p0": float(row[f"{key}__p0_analytic"]),
            "Z": float(row[f"{key}__Z_analytic"]),
        }
    record["functional_mean_minus_gp_mean_A_hat"] = float(
        row["functional_mean_minus_gp_mean__A_hat"]
    )
    record["functional_draw_minus_functional_mean_A_hat"] = float(
        row["functional_draw_minus_functional_mean__A_hat"]
    )
    return record


def peak_record(table: pd.DataFrame, key: str) -> dict[str, Any]:
    shoulder = table[table["mass_MeV"].between(61.0, 63.0)]
    index = shoulder[f"{key}__Z_analytic"].idxmax()
    row = table.loc[index]
    return {
        "mass_MeV": int(row["mass_MeV"]),
        "A_hat": float(row[f"{key}__A_hat"]),
        "sigma_A": float(row[f"{key}__sigma_A"]),
        "p0": float(row[f"{key}__p0_analytic"]),
        "Z": float(row[f"{key}__Z_analytic"]),
    }


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.0,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D3D7DC",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.65,
            "legend.fontsize": 8.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def draw_plot(table: pd.DataFrame) -> list[Path]:
    set_style()
    fig, (ax_a, ax_p) = plt.subplots(
        2,
        1,
        figsize=(9.4, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.08},
    )
    x = table["mass_MeV"].to_numpy(float)
    for key, style in SERIES.items():
        ax_a.plot(
            x,
            table[f"{key}__A_hat"].to_numpy(float) / 1000.0,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.6,
            markevery=1,
            linewidth=2.0 if "fractional" in key else 1.6,
            label=style["label"],
        )
        ax_p.plot(
            x,
            np.clip(
                table[f"{key}__p0_analytic"].to_numpy(float),
                1.0e-6,
                0.5,
            ),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.6,
            markevery=1,
            linewidth=2.0 if "fractional" in key else 1.6,
            label=style["label"],
        )
    for ax in (ax_a, ax_p):
        ax.axvspan(61.0, 63.0, color="#B8A1CF", alpha=0.13, linewidth=0)
        ax.axvline(65.0, color="#7F858C", linestyle=":", linewidth=1.0)
        ax.set_xlim(55.0, 75.0)
        ax.set_xticks(np.arange(55, 76, 2))
        ax.minorticks_on()
    ax_a.axhline(0.0, color="#555B62", linewidth=1.0)
    ax_a.set_ylabel(r"Fitted $\hat A$ ($10^3$ events)")
    handles, labels = ax_a.get_legend_handles_labels()
    fig.suptitle(
        "Conditional central-mean shape-bias diagnostic (2021 10%)",
        y=0.988,
        fontweight="semibold",
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        frameon=False,
        ncol=2,
    )
    fig.subplots_adjust(top=0.84)

    ax_p.set_yscale("log")
    ax_p.set_ylim(1.0e-4, 0.55)
    for z, label in ((1.0, r"$1\sigma$"), (2.0, r"$2\sigma$"), (3.0, r"$3\sigma$")):
        pvalue = 0.5 * math.erfc(z / math.sqrt(2.0))
        ax_p.axhline(
            pvalue,
            color="#969CA3",
            linestyle=":",
            linewidth=0.85,
        )
        ax_p.text(
            74.8,
            pvalue * 1.08,
            label,
            ha="right",
            va="bottom",
            fontsize=8.0,
            color="#767C83",
        )
    ax_p.set_ylabel(r"Local asymptotic $p_0$")
    ax_p.set_xlabel(r"Mass hypothesis $m_{A'}$ (MeV)")
    ax_p.text(
        0.012,
        0.04,
        (
            "Fractional mean curves are deterministic shape diagnostics; "
            "dashed curves are single Poisson draws."
        ),
        transform=ax_p.transAxes,
        fontsize=8.4,
        color="#51575E",
        ha="left",
        va="bottom",
    )
    outputs = []
    stem = PLOTS / "functional_mean_shape_bias_Ahat_p0"
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.08,
            "facecolor": "white",
        }
        if suffix == "png":
            kwargs["dpi"] = 300
        if suffix == "pdf":
            kwargs["metadata"] = {
                "Title": "pseudo65 functional-mean shape-bias diagnostic",
                "Subject": (
                    "Deterministic central means compared with single "
                    "Poisson replacement draws"
                ),
            }
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_memo(
    table: pd.DataFrame,
    anchors: list[dict[str, Any]],
    peaks: dict[str, dict[str, Any]],
) -> Path:
    def cell(record: dict[str, Any], key: str) -> str:
        lane = record[key]
        return (
            f"{lane['A_hat']:.1f}; {lane['sigma_A']:.1f}; "
            f"{lane['p0']:.4g}; {lane['Z']:.3f}"
        )

    lines = [
        "# Functional-mean shape-bias diagnostic",
        "",
        "## Question and construction",
        "",
        (
            "This conditional diagnostic asks whether replacing [60,70) MeV "
            "with the stored smooth `fGenGammaThresh` expectation can itself "
            "produce a 61--63 MeV positive response when the result is analyzed "
            "by the unchanged v4.2 GP card."
        ),
        "",
        (
            "The central inputs are fractional deterministic means (Asimov-like "
            "only within the replacement window), while the original observed "
            "2021 10% counts are retained bitwise outside. They are not observed "
            "datasets, complete Asimov datasets, pseudoexperiments, expected "
            "results, or coverage tests."
        ),
        "",
        "Each table entry is `Ahat; sigma_A; local p0; local Z`.",
        "",
        (
            "| mass | functional mean | functional Poisson draw | "
            "GP mean | GP-mean Poisson draw |"
        ),
        "|---:|---:|---:|---:|---:|",
    ]
    for record in anchors:
        lines.append(
            f"| {record['mass_MeV']} MeV | "
            f"{cell(record, 'functional_mean_fractional')} | "
            f"{cell(record, 'functional_poisson_draw')} | "
            f"{cell(record, 'gp_mean_fractional')} | "
            f"{cell(record, 'gp_poisson_draw')} |"
        )

    row62 = next(item for item in anchors if item["mass_MeV"] == 62)
    lines.extend(
        [
            "",
            "## Quantitative answer",
            "",
            (
                "Yes, conditionally: the deterministic functional mean produces "
                "a positive GP-extraction shoulder in the 61--63 MeV region even "
                "without a Poisson draw. Its largest local response in that "
                f"window is Z={peaks['functional_mean_fractional']['Z']:.3f} at "
                f"{peaks['functional_mean_fractional']['mass_MeV']} MeV. The "
                "functional Poisson draw is larger, reaching "
                f"Z={peaks['functional_poisson_draw']['Z']:.3f} at "
                f"{peaks['functional_poisson_draw']['mass_MeV']} MeV."
            ),
            "",
            (
                "At 62 MeV the functional deterministic mean exceeds the GP "
                "deterministic mean by "
                f"{row62['functional_mean_minus_gp_mean_A_hat']:.1f} fitted "
                "events. The particular functional Poisson draw adds another "
                f"{row62['functional_draw_minus_functional_mean_A_hat']:.1f} "
                "events relative to its deterministic mean response. Thus the "
                "reviewed 61--63 MeV shoulder contains both a deterministic "
                "truth-model/GP mismatch component and an additional fluctuation "
                "component in that one draw."
            ),
            "",
            (
                "The GP-mean deterministic lane also has a smaller positive "
                "response near the low side of the window, so this construction "
                "does not identify every positive fitted event exclusively with "
                "the functional interpolation. The relevant functional-specific "
                "diagnostic is its excess over the otherwise identical GP-mean "
                "lane."
            ),
            "",
            "## Statistical boundary",
            "",
            (
                "All p-values are local asymptotic responses of one conditional "
                "hybrid spectrum. No ensemble was generated. The comparison is "
                "not a coverage statement, expected sensitivity, global p-value, "
                "or probability that the interpolation will create a shoulder."
            ),
        ]
    )
    path = HERE / "MEMO.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    reviewed: dict[str, pd.DataFrame] = {}
    optimizer_summaries = {}
    pending_total = 0
    for lane in LANES:
        frame, summary = review_lane(lane)
        reviewed[lane] = frame
        optimizer_summaries[lane] = summary
        pending_total += int(summary["pending_mass_count"])
        print(
            f"{lane}: {summary['reviewed_mass_count']}/"
            f"{summary['expected_mass_count']} reviewed; "
            f"{summary['pending_mass_count']} pending"
        )
    audit = {
        "schema_version": 1,
        "selection_rule": "maximum finite GP log-marginal likelihood",
        "interpolation_permitted": False,
        "reproduction_rule": {
            "minimum_unchanged_card_rows": 2,
            "lml_absolute_tolerance": LML_MATCH_ATOL,
            "const_and_ls_relative_tolerance": PARAM_MATCH_RTOL,
            "const_and_ls_absolute_tolerance": PARAM_MATCH_ATOL,
        },
        "lanes": optimizer_summaries,
        "pending_mass_count_total": pending_total,
        "pass": pending_total == 0,
    }
    audit_path = DERIVED / "optimizer_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if pending_total:
        raise SystemExit(2)

    table, _ = build_comparison(reviewed)
    anchors = [anchor_record(table, mass) for mass in (61, 62, 63)]
    peaks = {key: peak_record(table, key) for key in SERIES}
    plot_paths = draw_plot(table)
    memo_path = write_memo(table, anchors, peaks)
    summary = {
        "schema_version": 1,
        "status": "GENERATED",
        "question": (
            "Does the stored fGenGammaThresh central interpolation create a "
            "61--63 MeV GP-extraction shoulder without Poisson fluctuation?"
        ),
        "answer": (
            "The deterministic functional mean has a positive 61--63 MeV "
            "response larger than the deterministic GP mean; the reviewed "
            "functional Poisson draw amplifies it. This is a conditional "
            "truth-model mismatch diagnostic, not an ensemble probability."
        ),
        "interpretation": (
            "Fractional deterministic central means with observed data outside; "
            "not an observed dataset, expected result, coverage study, or "
            "global-null pseudoexperiment."
        ),
        "mass_range_GeV": [0.055, 0.075],
        "anchors": anchors,
        "peaks_61_63MeV": peaks,
        "optimizer_audit": {
            "path": repo_relative(audit_path),
            "sha256": sha256_file(audit_path),
            "pass": True,
        },
        "comparison_csv": {
            "path": repo_relative(DERIVED / "comparison_55_75MeV.csv"),
            "sha256": sha256_file(DERIVED / "comparison_55_75MeV.csv"),
            "rows": int(len(table)),
        },
        "memo": {
            "path": repo_relative(memo_path),
            "sha256": sha256_file(memo_path),
        },
        "plots": [
            {
                "path": repo_relative(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in plot_paths
        ],
        "reference_draws": {
            key: {
                "path": repo_relative(path),
                "sha256": expected_hash,
            }
            for key, (path, expected_hash) in REFERENCE_DRAWS.items()
        },
    }
    path = DERIVED / "summary.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
