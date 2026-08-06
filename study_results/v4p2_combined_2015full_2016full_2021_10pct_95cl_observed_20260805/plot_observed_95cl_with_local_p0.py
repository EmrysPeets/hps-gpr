#!/usr/bin/env python3
"""Plot the v4.2 observed 95% CLs limit above its local asymptotic p0."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-gpr-v4p2-observed95-plot-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "derived" / "combined_observed_95cl_reviewed_v4p2.csv"
VALIDATION_JSON = HERE / "derived" / "validation_observed_95cl_v4p2.json"
PROVENANCE_JSON = HERE / "derived" / "provenance_observed_95cl_v4p2.json"
FIGURE_DIR = HERE / "figures"
STEM = "combined_observed_95cl_with_local_asymptotic_p0_v4p2"
FIGURE_PROVENANCE = FIGURE_DIR / f"{STEM}_provenance.json"

DIMUON_THRESHOLD_MEV = 211.316749
REGIONS = (
    (18.5, 38.5, "#d9e6f2"),
    (38.5, 49.5, "#e9ddcb"),
    (49.5, 90.5, "#d9ead3"),
    (90.5, 180.5, "#e6d9ed"),
    (180.5, 250.5, "#e1e1e1"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style_log_axis(axis: plt.Axes) -> None:
    axis.grid(True, which="major", color="#d8d8d8", linewidth=0.75, alpha=0.85)
    axis.grid(True, which="minor", color="#eeeeee", linewidth=0.45, alpha=0.72)
    axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    axis.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10, dtype=float) * 0.1,
            numticks=80,
        )
    )
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(axis="both", which="major", direction="in", length=6)
    axis.tick_params(axis="both", which="minor", direction="in", length=3)
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)


def main() -> None:
    for path in (INPUT_CSV, VALIDATION_JSON, PROVENANCE_JSON):
        if not path.is_file():
            raise SystemExit(f"Required input does not exist: {path}")

    validation = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise SystemExit("The 95% observed-limit validation did not pass.")
    if validation.get("n_toys") != 0 or not validation.get("observed_only"):
        raise SystemExit("The requested plot must remain observed-only.")

    frame = pd.read_csv(INPUT_CSV).sort_values("mass_MeV").reset_index(drop=True)
    required = {
        "mass_MeV",
        "eps2_obs_95_minimal_visible",
        "p0_analytic",
        "Z_analytic",
        "dataset_set",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Missing plotting columns: {sorted(missing)!r}")
    for column in ("eps2_obs_95_minimal_visible", "p0_analytic", "Z_analytic"):
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise SystemExit(f"Non-finite plotting values in {column}.")

    mass = frame["mass_MeV"].to_numpy(float)
    limit = frame["eps2_obs_95_minimal_visible"].to_numpy(float)
    p0 = frame["p0_analytic"].to_numpy(float)
    z = frame["Z_analytic"].to_numpy(float)
    minimum_index = int(np.argmin(p0))
    minimum_mass = float(mass[minimum_index])
    minimum_p0 = float(p0[minimum_index])
    minimum_z = float(z[minimum_index])
    minimum_p0_exponent = int(np.floor(np.log10(minimum_p0)))
    minimum_p0_mantissa = minimum_p0 / (10.0**minimum_p0_exponent)

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 280,
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.labelsize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "mathtext.fontset": "dejavusans",
        }
    )

    figure = plt.figure(figsize=(11.3, 8.4))
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=(0.20, 3.05, 1.55),
        hspace=0.075,
        left=0.105,
        right=0.975,
        bottom=0.105,
        top=0.865,
    )
    activity_axis = figure.add_subplot(grid[0])
    limit_axis = figure.add_subplot(grid[1])
    p0_axis = figure.add_subplot(grid[2], sharex=limit_axis)

    activity_axis.set_xlim(18.5, 250.5)
    activity_axis.set_ylim(0.0, 1.0)
    for lo, hi, color in REGIONS:
        activity_axis.add_patch(
            Rectangle(
                (lo, 0.0),
                hi - lo,
                1.0,
                facecolor=color,
                edgecolor="#888888",
                linewidth=0.55,
            )
        )
    activity_axis.set_xticks([])
    activity_axis.set_yticks([])
    for spine in activity_axis.spines.values():
        spine.set_visible(False)

    figure.text(
        0.105,
        0.958,
        "Active datasets — 19–38 MeV: 2015 full   |   "
        "39–49: 2015 full + 2016 full   |   "
        "50–90: 2015 full + 2016 full + 2021 (10% dev.)",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#303030",
    )
    figure.text(
        0.105,
        0.933,
        "91–180 MeV: 2016 full + 2021 (10% dev.)   |   "
        "181–250: 2021 (10% dev.)",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#303030",
    )

    limit_axis.plot(
        mass,
        limit,
        color="#101010",
        linewidth=2.25,
        solid_capstyle="round",
        label=r"Observed 95% $\mathrm{CL_s}$",
        zorder=4,
    )
    limit_axis.set_yscale("log")
    limit_axis.set_ylim(float(np.min(limit) * 0.62), float(np.max(limit) * 1.65))
    limit_axis.set_xlim(18.5, 250.5)
    limit_axis.set_ylabel(
        "Observed 95% $\\mathrm{CL_s}$ upper limit\n"
        "on $\\epsilon^2$"
    )
    limit_axis.text(
        0.012,
        0.965,
        "(a) Simultaneous observed limit",
        transform=limit_axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
            "pad": 1.8,
        },
    )
    limit_axis.text(
        0.988,
        0.965,
        r"minimal-visible $A^\prime$ model",
        transform=limit_axis.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#444444",
    )
    limit_axis.axvline(
        DIMUON_THRESHOLD_MEV,
        color="#777777",
        linestyle="--",
        linewidth=1.05,
        zorder=1,
    )
    limit_axis.text(
        DIMUON_THRESHOLD_MEV + 2.0,
        float(np.max(limit) * 1.20),
        r"$2m_\mu$",
        color="#606060",
        fontsize=10,
        ha="left",
        va="center",
    )
    limit_axis.legend(
        loc="lower left",
        frameon=False,
        handlelength=2.5,
        borderaxespad=0.75,
    )
    style_log_axis(limit_axis)
    limit_axis.tick_params(axis="x", which="both", labelbottom=False)

    p0_axis.plot(
        mass,
        p0,
        color="#8b1e1e",
        linewidth=1.95,
        solid_capstyle="round",
        zorder=4,
    )
    p0_axis.set_yscale("log")
    p0_axis.set_ylim(min(1.15e-5, minimum_p0 * 0.42), 0.78)
    p0_axis.set_ylabel(r"Local asymptotic $p_0$")
    p0_axis.set_xlabel(r"Mass hypothesis $m_{A^\prime}$ [MeV]")
    p0_axis.text(
        0.012,
        0.955,
        r"(b) Shared-$\epsilon^2$ local significance",
        transform=p0_axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
            "pad": 1.8,
        },
    )

    sigma_levels = tuple(range(1, 5))
    for sigma in sigma_levels:
        probability = float(norm.sf(float(sigma)))
        p0_axis.axhline(
            probability,
            color="#858585",
            linestyle=":",
            linewidth=0.85,
            zorder=1,
        )
        p0_axis.text(
            248.2,
            probability * 1.07,
            rf"{sigma}$\sigma$",
            color="#606060",
            fontsize=9.2,
            ha="right",
            va="bottom",
        )

    p0_axis.plot(
        [minimum_mass],
        [minimum_p0],
        marker="o",
        markersize=6.5,
        markerfacecolor="#ffffff",
        markeredgecolor="#8b1e1e",
        markeredgewidth=1.7,
        zorder=6,
    )
    p0_axis.annotate(
        f"{minimum_mass:.0f} MeV: "
        rf"$p_0={minimum_p0_mantissa:.2f}"
        rf"\times10^{{{minimum_p0_exponent}}}$"
        "\n"
        rf"$Z_{{\mathrm{{local}}}}={minimum_z:.2f}$",
        xy=(minimum_mass, minimum_p0),
        xycoords="data",
        xytext=(85.0, 1.30e-4),
        textcoords="data",
        arrowprops={
            "arrowstyle": "-",
            "color": "#8b1e1e",
            "linewidth": 1.0,
            "shrinkA": 2,
            "shrinkB": 5,
        },
        fontsize=10.2,
        color="#6f1818",
        ha="left",
        va="bottom",
    )
    style_log_axis(p0_axis)
    p0_axis.xaxis.set_minor_locator(AutoMinorLocator(5))

    figure.text(
        0.975,
        0.028,
        "Fixed reviewed v4.2 GP states; asymptotic "
        "$\\mathrm{CL_s}$ and fixed-mass local $p_0$; "
        "no look-elsewhere correction or expected bands",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{STEM}.png"
    svg_path = FIGURE_DIR / f"{STEM}.svg"
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    provenance = {
        "schema_version": 1,
        "input_csv": str(INPUT_CSV),
        "input_csv_sha256": sha256(INPUT_CSV),
        "validation_json": str(VALIDATION_JSON),
        "validation_json_sha256": sha256(VALIDATION_JSON),
        "source_provenance_json": str(PROVENANCE_JSON),
        "source_provenance_json_sha256": sha256(PROVENANCE_JSON),
        "png": str(png_path),
        "png_sha256": sha256(png_path),
        "svg": str(svg_path),
        "svg_sha256": sha256(svg_path),
        "n_rows": int(len(frame)),
        "mass_range_MeV": [float(np.min(mass)), float(np.max(mass))],
        "limit_field": "eps2_obs_95_minimal_visible",
        "p0_field": "p0_analytic",
        "p0_family": "local asymptotic shared-epsilon2",
        "minimum_p0_mass_MeV": minimum_mass,
        "minimum_p0": minimum_p0,
        "minimum_Z": minimum_z,
        "bands_plotted": False,
        "global_pvalue_plotted": False,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    FIGURE_PROVENANCE.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "png": str(png_path),
                "svg": str(svg_path),
                "minimum_p0_mass_MeV": minimum_mass,
                "minimum_p0": minimum_p0,
                "minimum_Z": minimum_z,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
