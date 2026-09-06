#!/usr/bin/env python3
"""Render the validated v4.9.7 combined epsilon-squared result and bands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BANDS = HERE / "combined" / "bands_100toy_cached" / "ul_bands_combined_all.csv"
VALIDATION = HERE / "qa" / "combined_release_validation.json"
FIGURES = HERE / "figures"
DERIVED = HERE / "combined" / "derived"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not BANDS.is_file() or not VALIDATION.is_file():
        raise RuntimeError("validated combined production products are missing")
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "pass":
        raise RuntimeError("combined production validation has not passed")
    frame = pd.read_csv(BANDS).sort_values("mass_GeV").reset_index(drop=True)
    expected_mass = np.arange(19, 251, dtype=float)
    mass = 1000.0 * frame["mass_GeV"].to_numpy(float)
    if len(frame) != 232 or not np.array_equal(mass, expected_mass):
        raise RuntimeError("combined mass grid is not exactly 19--250 MeV")
    if not np.all(frame["n_toys_finite"].to_numpy(int) == 100):
        raise RuntimeError("not every combined mass has exactly 100 finite toys")
    columns = ("eps2_obs", "eps2_lo2", "eps2_lo1", "eps2_med", "eps2_hi1", "eps2_hi2")
    values = frame[list(columns)].to_numpy(float)
    if not np.isfinite(values).all() or not np.all(values > 0.0):
        raise RuntimeError("nonfinite or nonpositive combined limit value")
    if not np.all(
        (frame.eps2_lo2 <= frame.eps2_lo1)
        & (frame.eps2_lo1 <= frame.eps2_med)
        & (frame.eps2_med <= frame.eps2_hi1)
        & (frame.eps2_hi1 <= frame.eps2_hi2)
    ):
        raise RuntimeError("combined quantile ordering failure")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 240,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06},
    )
    ax, ratio = axes
    ax.fill_between(
        mass,
        frame["eps2_lo2"],
        frame["eps2_hi2"],
        color="#F0E442",
        alpha=0.72,
        linewidth=0,
        label="95% conditional band",
    )
    ax.fill_between(
        mass,
        frame["eps2_lo1"],
        frame["eps2_hi1"],
        color="#009E73",
        alpha=0.68,
        linewidth=0,
        label="68% conditional band",
    )
    ax.plot(
        mass,
        frame["eps2_med"],
        color="#111111",
        lw=1.25,
        ls="--",
        label="conditional median",
    )
    ax.plot(
        mass,
        frame["eps2_obs"],
        color="#0072B2",
        lw=1.65,
        label="observed 90% asymptotic CL$_s$",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"upper limit on $\epsilon^2$")
    ax.set_title(
        "2015 full + 2016 full + 2021 10%: shared-$\\epsilon^2$ combination",
        loc="left",
    )
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(alpha=0.18, which="both")

    observed_ratio = frame["eps2_obs"].to_numpy(float) / frame["eps2_med"].to_numpy(float)
    ratio.plot(mass, observed_ratio, color="#0072B2", lw=1.3)
    ratio.axhline(1.0, color="#111111", lw=0.8, ls="--")
    ratio.set_yscale("log")
    ratio.set_ylabel("obs. / median")
    ratio.set_xlabel("mass hypothesis [MeV]")
    ratio.grid(alpha=0.18, which="both")
    ratio.set_xlim(19, 250)
    for boundary in (39, 50, 91, 181):
        for panel in axes:
            panel.axvline(boundary - 0.5, color="#6b7280", lw=0.55, ls=":", alpha=0.8)
    labels = (
        (19, 38, "2015"),
        (39, 49, "15+16"),
        (50, 90, "all three"),
        (91, 180, "16+21"),
        (181, 250, "2021"),
    )
    for low, high, text in labels:
        ratio.text(
            0.5 * (low + high),
            0.94,
            text,
            transform=ratio.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.7,
            color="#4b5563",
        )
    fig.text(
        0.01,
        0.004,
        (
            "Bands: quantiles of 100 mass-local, fixed-GP background-only limit pseudoexperiments; "
            "each inner limit remains asymptotic."
        ),
        fontsize=7.7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.95, bottom=0.105)

    FIGURES.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / "combined_eps2_observed_and_100toy_bands.pdf"
    png = FIGURES / "combined_eps2_observed_and_100toy_bands.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    min_p0_row = frame.loc[frame["p0_analytic"].astype(float).idxmin()]
    key_masses = frame.loc[frame["mass_GeV"].isin([0.039, 0.050, 0.065, 0.078, 0.090, 0.180, 0.250])].copy()
    key_columns = [
        "mass_GeV",
        "dataset_set",
        "eps2_obs",
        "eps2_lo2",
        "eps2_lo1",
        "eps2_med",
        "eps2_hi1",
        "eps2_hi2",
        "p0_analytic",
        "Z_analytic",
        "n_toys_finite",
        "tail_count_strong_le_observed",
        "tail_count_weak_ge_observed",
    ]
    key_path = DERIVED / "combined_key_mass_summary.csv"
    key_masses[key_columns].to_csv(key_path, index=False)
    summary = {
        "status": "pass",
        "mass_points": 232,
        "mass_range_MeV": [19, 250],
        "finite_mass_local_toy_limits": 23200,
        "n_toys_per_mass": 100,
        "observed_eps2_min": float(frame["eps2_obs"].min()),
        "observed_eps2_max": float(frame["eps2_obs"].max()),
        "minimum_local_p0": float(min_p0_row["p0_analytic"]),
        "minimum_local_p0_mass_MeV": float(1000.0 * min_p0_row["mass_GeV"]),
        "minimum_local_Z": float(min_p0_row["Z_analytic"]),
        "bands_csv_sha256": sha256(BANDS),
        "combined_validation_sha256": sha256(VALIDATION),
        "figure_pdf": str(pdf.relative_to(HERE)),
        "figure_pdf_sha256": sha256(pdf),
        "figure_png": str(png.relative_to(HERE)),
        "figure_png_sha256": sha256(png),
        "key_mass_csv": str(key_path.relative_to(HERE)),
        "key_mass_csv_sha256": sha256(key_path),
        "claim_boundary": (
            "Observed curve uses 90% asymptotic CLs. Bands are quantiles of "
            "100 conditional fixed-GP background-only limit pseudoexperiments; "
            "not direct coverage, toy-calibrated inner CLs, global significance, "
            "or a calibrated sensitivity."
        ),
    }
    output = DERIVED / "combined_figure_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
