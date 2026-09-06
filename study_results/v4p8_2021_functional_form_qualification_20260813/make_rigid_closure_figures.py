#!/usr/bin/env python3
"""Make the v4.8.2 Figure-48-style zero-signal closure diagnostic.

Run this only after both

    python3 run_rigid_study.py collect
    python3 run_rigid_study.py analytic-mean

have completed successfully.  The figure is a conditional spurious-signal
diagnostic for the frozen rigid source-conditioned mean.  It is not a coverage
plot, an observed-data bias measurement, or a CLs result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived/rigid_closure_v4p8p2_20toy_frozen"
OUTPUT = Path(
    "/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/"
    "output/pdf/v4p8_2021_rigid_threshold_truth_20260813"
)
STEM = "v4p8_rigid_zero_signal_conditional_closure_20toy"

N_TOYS = 20
ZERO_INJECTION = 0.0
MASS_GRID_MEV = (65, 90, 120, 180, 210)
SCENARIOS = (
    "2021_1pct_x10",
    "2021_10pct",
    "2021_1pct_x100",
    "2021_10pct_x10",
)
LABELS = {
    "2021_1pct_x10": r"1% source $\times 10$",
    "2021_10pct": "native 10% (1% shape frozen)",
    "2021_1pct_x100": r"1% source $\times 100$",
    "2021_10pct_x10": r"native 10% $\times 10$ (1% shape frozen)",
}
COLORS = {
    "accepted": "#0072B2",
    "raw": "#E69F00",
    "analytic": "#009E73",
    "excluded": "#D55E00",
    "median": "#6A3D9A",
}
KEY_COLUMNS = (
    "scenario",
    "background_toy_index",
    "mass_GeV",
    "inj_nsigma",
)
REQUIRED_FILES = (
    "collection_summary.json",
    "accepted_extraction_rows.csv",
    "raw_primary_extraction_rows.csv",
    "exclusion_ledger.csv",
    "closure_summary.csv",
    "analytic_mean_zero_signal_closure.csv",
    "analytic_mean_closure_summary.json",
)


class FigureInputError(RuntimeError):
    """Raised when the frozen result products are missing or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FigureInputError(f"{label} is missing columns: {missing}")


def finite_numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise FigureInputError(f"{label}.{column} contains nonfinite values")


def canonical_keys(frame: pd.DataFrame) -> set[tuple[str, int, int, int]]:
    return {
        (
            str(row.scenario),
            int(row.background_toy_index),
            int(round(1000.0 * float(row.mass_GeV))),
            int(round(1000.0 * float(row.inj_nsigma))),
        )
        for row in frame.itertuples(index=False)
    }


def zero_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[np.isclose(pd.to_numeric(frame["inj_nsigma"]), ZERO_INJECTION)].copy()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [str(DERIVED / name) for name in REQUIRED_FILES if not (DERIVED / name).is_file()]
    if missing:
        raise FigureInputError(
            "final v4.8.2 closure products are incomplete; run collect and "
            f"analytic-mean first. Missing: {missing}"
        )

    collection = load_json(DERIVED / "collection_summary.json")
    analytic_summary = load_json(DERIVED / "analytic_mean_closure_summary.json")
    if collection.get("status") != "pass":
        raise FigureInputError("collection_summary.json does not report status=pass")
    if analytic_summary.get("status") != "pass":
        raise FigureInputError("analytic_mean_closure_summary.json does not report status=pass")

    declared_hashes = collection.get("derived_sha256", {})
    for name in (
        "accepted_extraction_rows.csv",
        "raw_primary_extraction_rows.csv",
        "exclusion_ledger.csv",
        "closure_summary.csv",
    ):
        expected = str(declared_hashes.get(name, ""))
        if not expected or sha256_file(DERIVED / name) != expected:
            raise FigureInputError(f"collection hash mismatch or missing declaration: {name}")
    if sha256_file(DERIVED / "analytic_mean_zero_signal_closure.csv") != str(
        analytic_summary.get("selected_sha256", "")
    ):
        raise FigureInputError("analytic-mean selected-row hash mismatch")

    accepted = pd.read_csv(DERIVED / "accepted_extraction_rows.csv")
    raw = pd.read_csv(DERIVED / "raw_primary_extraction_rows.csv")
    exclusions = pd.read_csv(DERIVED / "exclusion_ledger.csv")
    summary = pd.read_csv(DERIVED / "closure_summary.csv")
    analytic = pd.read_csv(DERIVED / "analytic_mean_zero_signal_closure.csv")

    require_columns(accepted, (*KEY_COLUMNS, "pull"), "accepted ledger")
    require_columns(raw, (*KEY_COLUMNS, "pull"), "raw-first ledger")
    require_columns(exclusions, KEY_COLUMNS, "exclusion ledger")
    require_columns(
        summary,
        (
            "scenario",
            "mass_GeV",
            "mass_MeV",
            "inj_nsigma",
            "raw_n",
            "raw_pull_mean",
            "raw_pull_mean_ci90_low",
            "raw_pull_mean_ci90_high",
            "accepted_n",
            "accepted_pull_mean",
            "accepted_pull_median",
            "accepted_pull_mean_ci90_low",
            "accepted_pull_mean_ci90_high",
            "accepted_pull_width",
            "accepted_pull_width_ci90_low",
            "accepted_pull_width_ci90_high",
            "n_excluded",
        ),
        "closure summary",
    )
    require_columns(analytic, ("scenario", "mass_GeV", "mass_MeV", "pull"), "analytic mean")

    accepted_zero = zero_rows(accepted)
    raw_zero = zero_rows(raw)
    excluded_zero_ledger = zero_rows(exclusions)
    summary_zero = zero_rows(summary)

    finite_numeric(accepted_zero, ("mass_GeV", "pull"), "accepted zero-signal ledger")
    finite_numeric(raw_zero, ("mass_GeV", "pull"), "raw-first zero-signal ledger")
    finite_numeric(
        summary_zero,
        (
            "mass_GeV",
            "mass_MeV",
            "raw_pull_mean",
            "raw_pull_mean_ci90_low",
            "raw_pull_mean_ci90_high",
            "accepted_pull_mean",
            "accepted_pull_median",
            "accepted_pull_mean_ci90_low",
            "accepted_pull_mean_ci90_high",
            "accepted_pull_width",
            "accepted_pull_width_ci90_low",
            "accepted_pull_width_ci90_high",
        ),
        "zero-signal closure summary",
    )
    finite_numeric(analytic, ("mass_GeV", "mass_MeV", "pull"), "analytic mean")

    expected_scenarios = set(SCENARIOS)
    expected_masses = set(MASS_GRID_MEV)
    for frame, label in (
        (accepted_zero, "accepted zero-signal ledger"),
        (raw_zero, "raw-first zero-signal ledger"),
        (summary_zero, "zero-signal closure summary"),
        (analytic, "analytic mean"),
    ):
        found_scenarios = set(frame["scenario"].astype(str))
        found_masses = set(np.rint(pd.to_numeric(frame["mass_GeV"]) * 1000.0).astype(int))
        if found_scenarios != expected_scenarios or found_masses != expected_masses:
            raise FigureInputError(
                f"{label} lane/mass inventory mismatch: "
                f"scenarios={sorted(found_scenarios)}, masses={sorted(found_masses)}"
            )

    if len(raw_zero) != len(SCENARIOS) * len(MASS_GRID_MEV) * N_TOYS:
        raise FigureInputError("raw-first zero-signal ledger is not the complete 4x5x20 grid")
    if raw_zero.duplicated(list(KEY_COLUMNS)).any() or accepted_zero.duplicated(list(KEY_COLUMNS)).any():
        raise FigureInputError("duplicate zero-signal extraction keys")
    if len(summary_zero) != len(SCENARIOS) * len(MASS_GRID_MEV):
        raise FigureInputError("zero-signal closure summary must contain exactly 20 cells")
    if len(analytic) != len(SCENARIOS) * len(MASS_GRID_MEV):
        raise FigureInputError("analytic-mean ledger must contain exactly 20 rows")
    if summary_zero.duplicated(["scenario", "mass_GeV"]).any() or analytic.duplicated(
        ["scenario", "mass_GeV"]
    ).any():
        raise FigureInputError("duplicate summary or analytic-mean cells")

    raw_keys = canonical_keys(raw_zero)
    accepted_keys = canonical_keys(accepted_zero)
    exclusion_keys = canonical_keys(excluded_zero_ledger)
    if not accepted_keys.issubset(raw_keys):
        raise FigureInputError("accepted zero-signal keys are not a subset of raw-first keys")
    if raw_keys - accepted_keys != exclusion_keys:
        raise FigureInputError("zero-signal exclusion ledger does not match raw-minus-accepted keys")

    for scenario in SCENARIOS:
        for mass_mev in MASS_GRID_MEV:
            raw_cell = raw_zero[
                (raw_zero.scenario == scenario)
                & np.isclose(raw_zero.mass_GeV, mass_mev / 1000.0)
            ]
            accepted_cell = accepted_zero[
                (accepted_zero.scenario == scenario)
                & np.isclose(accepted_zero.mass_GeV, mass_mev / 1000.0)
            ]
            summary_cell = summary_zero[
                (summary_zero.scenario == scenario)
                & np.isclose(summary_zero.mass_GeV, mass_mev / 1000.0)
            ]
            if len(raw_cell) != N_TOYS or set(raw_cell.background_toy_index.astype(int)) != set(
                range(N_TOYS)
            ):
                raise FigureInputError(f"raw inventory mismatch for {scenario}, {mass_mev} MeV")
            if len(summary_cell) != 1:
                raise FigureInputError(f"summary cell mismatch for {scenario}, {mass_mev} MeV")
            cell = summary_cell.iloc[0]
            if int(cell.raw_n) != N_TOYS or int(cell.accepted_n) != len(accepted_cell):
                raise FigureInputError(f"summary count mismatch for {scenario}, {mass_mev} MeV")
            if int(cell.n_excluded) != N_TOYS - len(accepted_cell):
                raise FigureInputError(f"summary exclusion mismatch for {scenario}, {mass_mev} MeV")
            median = float(np.median(pd.to_numeric(accepted_cell.pull)))
            if not math.isclose(median, float(cell.accepted_pull_median), rel_tol=1e-10, abs_tol=1e-12):
                raise FigureInputError(f"accepted median mismatch for {scenario}, {mass_mev} MeV")

    return accepted_zero, raw_zero, summary_zero, analytic, excluded_zero_ledger


def asymmetric_errors(center: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if np.any(low > center) or np.any(high < center):
        raise FigureInputError("an interval does not bracket its central value")
    return np.vstack((center - low, high - center))


def save_atomic(fig: plt.Figure, path: Path, *, dpi: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        fig.savefig(
            temporary,
            format=path.suffix.lstrip("."),
            dpi=dpi,
            bbox_inches="tight",
            metadata={"Title": "v4.8 rigid conditional zero-signal closure"}
            if path.suffix == ".pdf"
            else None,
        )
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def make_figure(
    accepted: pd.DataFrame,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    analytic: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(
        len(SCENARIOS),
        3,
        figsize=(14.8, 12.2),
        sharex="col",
        constrained_layout=False,
    )
    mass_values = np.asarray(MASS_GRID_MEV, dtype=float)

    for row_index, scenario in enumerate(SCENARIOS):
        accepted_lane = accepted[accepted.scenario == scenario]
        raw_lane = raw[raw.scenario == scenario]
        summary_lane = summary[summary.scenario == scenario].sort_values("mass_MeV")
        analytic_lane = analytic[analytic.scenario == scenario].sort_values("mass_MeV")
        excluded_keys = canonical_keys(exclusions[exclusions.scenario == scenario])
        excluded_lane = raw_lane[
            [
                key in excluded_keys
                for key in canonical_key_sequence(raw_lane)
            ]
        ]

        # Every accepted z=0 pull is retained.  Toy-index jitter is deterministic
        # and only separates coincident points; it does not encode another result.
        axis = axes[row_index, 0]
        for mass_mev in MASS_GRID_MEV:
            cell = accepted_lane[np.isclose(accepted_lane.mass_GeV, mass_mev / 1000.0)]
            toy_index = pd.to_numeric(cell.background_toy_index).to_numpy(float)
            jitter = (toy_index - 0.5 * (N_TOYS - 1)) / (N_TOYS - 1) * 1.6
            axis.scatter(
                mass_mev + jitter,
                pd.to_numeric(cell.pull),
                s=17,
                marker="o",
                facecolors="none",
                edgecolors=COLORS["accepted"],
                linewidths=0.75,
                alpha=0.78,
                zorder=2,
            )
        if len(excluded_lane):
            toy_index = pd.to_numeric(excluded_lane.background_toy_index).to_numpy(float)
            jitter = (toy_index - 0.5 * (N_TOYS - 1)) / (N_TOYS - 1) * 1.6
            axis.scatter(
                1000.0 * pd.to_numeric(excluded_lane.mass_GeV) + jitter,
                pd.to_numeric(excluded_lane.pull),
                s=48,
                marker="x",
                color=COLORS["excluded"],
                linewidths=1.6,
                zorder=5,
            )
        axis.plot(
            summary_lane.mass_MeV,
            summary_lane.accepted_pull_median,
            marker="D",
            markersize=4.8,
            color=COLORS["median"],
            linewidth=1.15,
            zorder=4,
        )
        axis.axhline(0.0, color="black", linewidth=0.8, zorder=0)
        axis.text(
            0.985,
            0.94,
            f"excluded raw states: {len(excluded_lane)}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.7,
        )

        # Raw means use the first optimizer attempt.  Accepted means use the
        # pull-blind reproducible-branch selection.  Both intervals are the
        # finite-sample 90% Student-t intervals stored by collect().
        axis = axes[row_index, 1]
        for center, low, high, offset, color, marker, linestyle in (
            (
                summary_lane.raw_pull_mean.to_numpy(float),
                summary_lane.raw_pull_mean_ci90_low.to_numpy(float),
                summary_lane.raw_pull_mean_ci90_high.to_numpy(float),
                -0.58,
                COLORS["raw"],
                "s",
                "--",
            ),
            (
                summary_lane.accepted_pull_mean.to_numpy(float),
                summary_lane.accepted_pull_mean_ci90_low.to_numpy(float),
                summary_lane.accepted_pull_mean_ci90_high.to_numpy(float),
                0.0,
                COLORS["accepted"],
                "o",
                "-",
            ),
        ):
            axis.errorbar(
                mass_values + offset,
                center,
                yerr=asymmetric_errors(center, low, high),
                marker=marker,
                markersize=4.6,
                color=color,
                markerfacecolor="white" if marker == "s" else color,
                linestyle=linestyle,
                linewidth=1.1,
                capsize=2.5,
                zorder=3,
            )
        axis.plot(
            analytic_lane.mass_MeV.to_numpy(float) + 0.58,
            analytic_lane.pull,
            marker="D",
            markersize=4.5,
            color=COLORS["analytic"],
            linestyle=":",
            linewidth=1.1,
            zorder=4,
        )
        axis.axhline(0.0, color="black", linewidth=0.8, zorder=0)

        # Width intervals are the two-sided 90% chi-square intervals produced
        # by collect().  No manual y range is set, so unfavorable intervals are
        # retained in the plot rather than clipped.
        axis = axes[row_index, 2]
        width = summary_lane.accepted_pull_width.to_numpy(float)
        width_low = summary_lane.accepted_pull_width_ci90_low.to_numpy(float)
        width_high = summary_lane.accepted_pull_width_ci90_high.to_numpy(float)
        axis.errorbar(
            mass_values,
            width,
            yerr=asymmetric_errors(width, width_low, width_high),
            marker="o",
            markersize=4.8,
            color=COLORS["accepted"],
            linestyle="-",
            linewidth=1.1,
            capsize=2.5,
            zorder=3,
        )
        axis.axhline(1.0, color="black", linewidth=0.8, zorder=0)
        for mass_mev, n_accepted in zip(mass_values, summary_lane.accepted_n.astype(int)):
            axis.annotate(
                f"{n_accepted}/{N_TOYS}",
                (mass_mev, width[np.where(mass_values == mass_mev)[0][0]]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color="0.3",
            )

        for axis in axes[row_index, :]:
            axis.grid(alpha=0.20, linewidth=0.6)
            axis.set_xticks(mass_values)
            axis.margins(y=0.10)
        axes[row_index, 0].set_ylabel(f"{LABELS[scenario]}\nzero-signal pull")

    axes[0, 0].set_title("Accepted z=0 pulls and median")
    axes[0, 1].set_title("Raw-first and accepted mean pull")
    axes[0, 2].set_title("Accepted pull width")
    axes[-1, 0].set_xlabel("mass hypothesis [MeV]")
    axes[-1, 1].set_xlabel("mass hypothesis [MeV]")
    axes[-1, 2].set_xlabel("mass hypothesis [MeV]")
    axes[0, 1].set_ylabel("mean pull (90% Student-t CI)")
    axes[0, 2].set_ylabel("sample pull width (90% chi-square CI)")

    accepted_handle = Line2D(
        [], [], marker="o", markerfacecolor="none", markeredgecolor=COLORS["accepted"],
        linestyle="none", label="accepted pull"
    )
    median_handle = Line2D(
        [], [], marker="D", color=COLORS["median"], linewidth=1.1, label="accepted median"
    )
    excluded_handle = Line2D(
        [], [], marker="x", color=COLORS["excluded"], linestyle="none", label="excluded raw-first pull"
    )
    axes[0, 0].legend(
        handles=(accepted_handle, median_handle, excluded_handle),
        loc="best",
        frameon=False,
        fontsize=7.3,
    )
    axes[0, 1].legend(
        handles=(
            Line2D([], [], marker="s", markerfacecolor="white", color=COLORS["raw"], linestyle="--", label="raw first: mean +/- 90% t CI"),
            Line2D([], [], marker="o", color=COLORS["accepted"], linestyle="-", label="accepted: mean +/- 90% t CI"),
            Line2D([], [], marker="D", color=COLORS["analytic"], linestyle=":", label="deterministic analytic mean"),
        ),
        loc="best",
        frameon=False,
        fontsize=7.1,
    )
    axes[0, 2].legend(
        handles=(
            Line2D([], [], marker="o", color=COLORS["accepted"], linestyle="-", label="width +/- 90% chi-square CI"),
            Line2D([], [], color="0.3", linestyle="none", marker="", label="labels show accepted/20"),
        ),
        loc="best",
        frameon=False,
        fontsize=7.2,
    )

    fig.suptitle(
        "v4.8 rigid threshold stress mean: conditional zero-signal spurious-signal diagnostic",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.018,
        "Each lane uses 20 analyzed backgrounds; masses share backgrounds and exposure lanes are nested within source family. "
        "All finite pulls set the axes; no clipping is applied.",
        ha="center",
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.004,
        "The 90% t/chi-square intervals are finite-ensemble diagnostics, not 90% CLs. "
        "This figure is not coverage, observed-data bias, an expected band, or a physical-background claim.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.015, 0.045, 1.0, 0.972), h_pad=1.35, w_pad=1.10)
    return fig


def canonical_key_sequence(frame: pd.DataFrame) -> list[tuple[str, int, int, int]]:
    return [
        (
            str(row.scenario),
            int(row.background_toy_index),
            int(round(1000.0 * float(row.mass_GeV))),
            int(round(1000.0 * float(row.inj_nsigma))),
        )
        for row in frame.itertuples(index=False)
    ]


def main() -> int:
    accepted, raw, summary, analytic, exclusions = load_inputs()
    figure = make_figure(accepted, raw, summary, analytic, exclusions)
    pdf_path = OUTPUT / f"{STEM}.pdf"
    png_path = OUTPUT / f"{STEM}.png"
    save_atomic(figure, pdf_path)
    save_atomic(figure, png_path, dpi=220)
    plt.close(figure)
    print(
        json.dumps(
            {
                "status": "pass",
                "interpretation": "conditional spurious-signal diagnostic; not coverage or observed-data bias",
                "pdf": str(pdf_path),
                "pdf_sha256": sha256_file(pdf_path),
                "png": str(png_path),
                "png_sha256": sha256_file(png_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
