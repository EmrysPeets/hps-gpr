#!/usr/bin/env python3
"""Audit ensemble scans and make GP-only descriptive comparison figures."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from matplotlib.ticker import LogLocator, MultipleLocator


ROOT_FILE = HERE / "inputs" / "gp_window_ensemble.root"
CONFIG_MANIFEST = HERE / "derived" / "config_manifest.json"
PROVENANCE = HERE / "derived" / "input_provenance.json"
BASELINE_CSV = (
    REPO
    / "study_results"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "derived"
    / "observed_gp_states_v4p2_enriched.csv"
)
SOURCE_KEY = "source/preselection/h_invM_8000"
WINDOWS = {
    "window_2p25eq2p5": {
        "title": r"$\pm2.25\sigma_m$ and $\pm2.5\sigma_m$",
        "subtitle": r"same 16-bin grid, $[60,70)$ MeV",
        "color": "#0072B2",
    },
    "window_3p0": {
        "title": r"$\pm3\sigma_m$",
        "subtitle": r"20 bins, $[58.75,71.25)$ MeV",
        "color": "#D55E00",
    },
}
COLORS = {
    "original": "#6C737C",
    "ink": "#23262B",
    "grid": "#C7CBD1",
    "mean": "#20242A",
    "gp": "#009E73",
}
REQUIRED_FINITE = (
    "mass_GeV",
    "A_hat",
    "sigma_A",
    "A_up",
    "eps2_up",
    "p0_analytic",
    "Z_analytic",
    "lml",
    "const_opt",
    "ls_opt",
    "integral_density",
)
BOUND_COLUMNS = (
    "ls_at_lower",
    "ls_at_upper",
    "const_at_lower",
    "const_at_upper",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def bool_values(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    return series.astype(str).str.strip().str.lower().eq("true").to_numpy(bool)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.8,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.3,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.65,
            "legend.fontsize": 7.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def rebin_five(values: np.ndarray, edges: np.ndarray):
    values = np.asarray(values, float)
    edges = np.asarray(edges, float)
    if len(values) % 5:
        raise RuntimeError("Native histogram is not divisible by five")
    rebinned = values.reshape(-1, 5).sum(axis=1)
    rebinned_edges = edges[::5]
    if len(rebinned_edges) != len(rebinned) + 1:
        rebinned_edges = np.r_[rebinned_edges, edges[-1]]
    centers = 0.5 * (rebinned_edges[:-1] + rebinned_edges[1:])
    return rebinned, rebinned_edges, centers


def load_and_audit_scans():
    manifest = json.loads(CONFIG_MANIFEST.read_text())
    qc_records = []
    expected_mass = np.round(np.arange(0.050, 0.250 + 0.0005, 0.001), 3)
    for record in manifest["records"]:
        window = record["window"]
        draw_index = int(record["draw_index"])
        result_path = REPO / record["output_dir"] / "results_single.csv"
        log_path = result_path.parent / "scan.log"
        if not result_path.exists():
            raise RuntimeError(f"Missing full-scan result: {result_path}")
        if not log_path.exists():
            raise RuntimeError(f"Missing full-scan log: {log_path}")
        log_text = log_path.read_text(errors="replace")
        scan_complete_marker = "Scan complete!" in log_text
        traceback_count = log_text.count("Traceback (most recent call last)")
        convergence_warning_count = log_text.count("ConvergenceWarning")
        frame = pd.read_csv(result_path)
        frame = frame[frame["dataset"].astype(str) == "2021"].copy()
        frame = frame.sort_values("mass_GeV").reset_index(drop=True)
        mass = np.round(frame["mass_GeV"].to_numpy(float), 3)
        finite = np.isfinite(frame[list(REQUIRED_FINITE)].to_numpy(float)).all(axis=1)
        extract_success = bool_values(frame["extract_success"])
        bound_mask = np.zeros(len(frame), dtype=bool)
        for column in BOUND_COLUMNS:
            bound_mask |= bool_values(frame[column])
        calibration_ok = frame["cls_calibration"].astype(str).eq("asymptotic")
        grid_ok = bool(
            len(frame) == len(expected_mass)
            and len(np.unique(mass)) == len(expected_mass)
            and np.array_equal(mass, expected_mass)
        )
        qc_records.append(
            {
                "window": window,
                "draw_index": draw_index,
                "result_csv": repo_relative(result_path),
                "result_sha256": sha256_file(result_path),
                "scan_log": repo_relative(log_path),
                "scan_log_sha256": sha256_file(log_path),
                "mass_count": int(len(frame)),
                "grid_complete": grid_ok,
                "scan_complete_marker": scan_complete_marker,
                "traceback_count": traceback_count,
                "convergence_warning_count": convergence_warning_count,
                "nonfinite_row_count": int(np.count_nonzero(~finite)),
                "extract_failure_count": int(np.count_nonzero(~extract_success)),
                "non_asymptotic_count": int(np.count_nonzero(~calibration_ok)),
                "selected_kernel_bound_count": int(np.count_nonzero(bound_mask)),
                "selected_kernel_bound_masses_MeV": [
                    float(1000.0 * item)
                    for item in frame.loc[bound_mask, "mass_GeV"].to_numpy(float)
                ],
                "minimum_local_p0": float(np.min(frame["p0_analytic"])),
                "minimum_local_p0_mass_MeV": float(
                    1000.0
                    * frame.iloc[int(np.argmin(frame["p0_analytic"]))]["mass_GeV"]
                ),
                "optimizer_attempt_count_per_mass": 1,
                "within_fit_optimizer_restarts": 12,
                "unchanged_card_max_lml_reproduced": False,
                "optimizer_review_status": (
                    "single scan attempt; finite/bound audited; selected maximum-LML "
                    "branch not independently reproduced"
                ),
                "pass_finite_grid_bound_gates": bool(
                    grid_ok
                    and scan_complete_marker
                    and traceback_count == 0
                    and np.all(finite)
                    and np.all(extract_success)
                    and np.all(calibration_ok)
                    and not np.any(bound_mask)
                ),
            }
        )
    qc = pd.DataFrame(qc_records).sort_values(["window", "draw_index"])
    qc.to_csv(HERE / "derived" / "scan_qc.csv", index=False)
    if not qc["pass_finite_grid_bound_gates"].all():
        failed = qc.loc[~qc["pass_finite_grid_bound_gates"]]
        raise RuntimeError(
            "One or more ensemble scans failed finite/grid/bound gates:\n"
            + failed.to_string(index=False)
        )

    reviewed_path = HERE / "derived" / "reviewed_curves.csv"
    if not reviewed_path.exists():
        raise RuntimeError("Run review_central.py before postprocess.py")
    individual = pd.read_csv(reviewed_path)
    all_frames: dict[str, list[pd.DataFrame]] = {window: [] for window in WINDOWS}
    for window in WINDOWS:
        for draw_index in range(10):
            frame = individual[
                (individual["window"].astype(str) == window)
                & (individual["draw_index"].astype(int) == draw_index)
            ].copy()
            frame = frame.sort_values("mass_GeV").reset_index(drop=True)
            if len(frame) != 201:
                raise RuntimeError(
                    f"{window} draw {draw_index}: reviewed curve has {len(frame)} rows"
                )
            all_frames[window].append(frame)
    individual.to_csv(HERE / "derived" / "individual_curves.csv", index=False)
    baseline = pd.read_csv(BASELINE_CSV)
    baseline = baseline[baseline["dataset"].astype(str) == "2021"].copy()
    baseline = baseline.sort_values("mass_GeV").reset_index(drop=True)
    if len(baseline) != 201:
        raise RuntimeError("Unexpected original v4.2 2021 baseline grid")
    return all_frames, baseline, qc


def summarize_curves(all_frames: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    records = []
    metrics = ("eps2_up", "p0_analytic", "Z_analytic", "A_hat", "sigma_A")
    for window, frames in all_frames.items():
        for mass_index, mass in enumerate(frames[0]["mass_GeV"].to_numpy(float)):
            record: dict[str, Any] = {
                "window": window,
                "mass_GeV": float(mass),
                "draw_count": len(frames),
                "summary_scope": (
                    "pointwise descriptive statistics across ten conditional draws"
                ),
                "quantile_method": "numpy linear",
            }
            for metric in metrics:
                values = np.array(
                    [float(frame.iloc[mass_index][metric]) for frame in frames]
                )
                record[f"{metric}_mean"] = float(np.mean(values))
                record[f"{metric}_median"] = float(np.median(values))
                record[f"{metric}_q16"] = float(np.quantile(values, 0.16))
                record[f"{metric}_q84"] = float(np.quantile(values, 0.84))
                record[f"{metric}_min"] = float(np.min(values))
                record[f"{metric}_max"] = float(np.max(values))
            records.append(record)
    summary = pd.DataFrame(records)
    summary.to_csv(HERE / "derived" / "ensemble_pointwise_summary.csv", index=False)
    return summary


def audit_pilot_reproduction(
    all_frames: dict[str, list[pd.DataFrame]],
) -> pd.DataFrame:
    records = []
    config_records = json.loads(CONFIG_MANIFEST.read_text())["records"]
    for window in WINDOWS:
        pilot_path = HERE / "pilot" / window / "draw_00" / "results_single.csv"
        if not pilot_path.exists():
            raise RuntimeError(f"Missing pilot result: {pilot_path}")
        pilot = pd.read_csv(pilot_path)
        pilot = pilot[
            (pilot["dataset"].astype(str) == "2021")
            & np.isclose(pilot["mass_GeV"].to_numpy(float), 0.065)
        ]
        config_record = next(
            item
            for item in config_records
            if item["window"] == window and int(item["draw_index"]) == 0
        )
        full_path = REPO / config_record["output_dir"] / "results_single.csv"
        full = pd.read_csv(full_path)
        full = full[
            (full["dataset"].astype(str) == "2021")
            & np.isclose(full["mass_GeV"].to_numpy(float), 0.065)
        ]
        reviewed = all_frames[window][0]
        reviewed = reviewed[
            np.isclose(reviewed["mass_GeV"].to_numpy(float), 0.065)
        ]
        if len(pilot) != 1 or len(full) != 1 or len(reviewed) != 1:
            raise RuntimeError(f"{window}: expected one pilot/full row at 65 MeV")
        pilot_row = pilot.iloc[0]
        full_row = full.iloc[0]
        reviewed_row = reviewed.iloc[0]
        lml_match = abs(float(pilot_row["lml"]) - float(full_row["lml"])) <= 3.0e-5
        const_match = np.isclose(
            float(pilot_row["const_opt"]),
            float(full_row["const_opt"]),
            rtol=5.0e-4,
            atol=1.0e-10,
        )
        ls_match = np.isclose(
            float(pilot_row["ls_opt"]),
            float(full_row["ls_opt"]),
            rtol=5.0e-4,
            atol=1.0e-10,
        )
        records.append(
            {
                "window": window,
                "draw_index": 0,
                "mass_MeV": 65.0,
                "pilot_csv": repo_relative(pilot_path),
                "full_csv": repo_relative(full_path),
                "review_selected_source": str(
                    reviewed_row["review_selected_source"]
                ),
                "review_selected_lml": float(reviewed_row["lml"]),
                "pilot_lml": float(pilot_row["lml"]),
                "full_lml": float(full_row["lml"]),
                "delta_lml_full_minus_pilot": float(
                    full_row["lml"] - pilot_row["lml"]
                ),
                "pilot_const_opt": float(pilot_row["const_opt"]),
                "full_const_opt": float(full_row["const_opt"]),
                "pilot_ls_opt": float(pilot_row["ls_opt"]),
                "full_ls_opt": float(full_row["ls_opt"]),
                "lml_match_atol_3e-5": bool(lml_match),
                "const_match_rtol_5e-4": bool(const_match),
                "ls_match_rtol_5e-4": bool(ls_match),
                "state_reproduced": bool(lml_match and const_match and ls_match),
                "scope": (
                    "sparse unchanged-card reproduction check for draw 00 at "
                    "65 MeV only; not a full-grid optimizer review"
                ),
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(HERE / "derived" / "pilot_m065_reproduction.csv", index=False)
    if not frame["state_reproduced"].all():
        raise RuntimeError(
            "The draw-00 65 MeV pilot did not reproduce the full-scan state"
        )
    return frame


def summarize_m065(
    all_frames: dict[str, list[pd.DataFrame]],
    summary: pd.DataFrame,
) -> None:
    rows = []
    for window, frames in all_frames.items():
        for draw_index, frame in enumerate(frames):
            match = frame[np.isclose(frame["mass_GeV"].to_numpy(float), 0.065)]
            if len(match) != 1:
                raise RuntimeError(f"{window} draw {draw_index}: missing 65 MeV row")
            row = match.iloc[0]
            rows.append(
                {
                    "window": window,
                    "draw_index": draw_index,
                    "A_hat": float(row["A_hat"]),
                    "sigma_A": float(row["sigma_A"]),
                    "eps2_up_90cl": float(row["eps2_up"]),
                    "p0_local_asymptotic": float(row["p0_analytic"]),
                    "Z_local_asymptotic": float(row["Z_analytic"]),
                }
            )
    pd.DataFrame(rows).to_csv(
        HERE / "derived" / "m065_individual_results.csv", index=False
    )
    summary[np.isclose(summary["mass_GeV"].to_numpy(float), 0.065)].to_csv(
        HERE / "derived" / "m065_ensemble_summary.csv", index=False
    )


def summarize_paired_window_differences(
    all_frames: dict[str, list[pd.DataFrame]],
) -> None:
    individual = []
    for draw_index in range(10):
        narrow = all_frames["window_2p25eq2p5"][draw_index]
        wide = all_frames["window_3p0"][draw_index]
        if not np.array_equal(
            narrow["mass_GeV"].to_numpy(float),
            wide["mass_GeV"].to_numpy(float),
        ):
            raise RuntimeError(f"Paired draw {draw_index}: mass grids differ")
        for row_index, mass in enumerate(narrow["mass_GeV"].to_numpy(float)):
            eps_narrow = float(narrow.iloc[row_index]["eps2_up"])
            eps_wide = float(wide.iloc[row_index]["eps2_up"])
            individual.append(
                {
                    "draw_index": draw_index,
                    "mass_GeV": float(mass),
                    "eps2_up_ratio_3p0_over_2p25eq2p5": eps_wide / eps_narrow,
                    "delta_log10_eps2_up_3p0_minus_2p25eq2p5": float(
                        np.log10(eps_wide) - np.log10(eps_narrow)
                    ),
                    "delta_p0_3p0_minus_2p25eq2p5": float(
                        wide.iloc[row_index]["p0_analytic"]
                        - narrow.iloc[row_index]["p0_analytic"]
                    ),
                    "delta_Z_3p0_minus_2p25eq2p5": float(
                        wide.iloc[row_index]["Z_analytic"]
                        - narrow.iloc[row_index]["Z_analytic"]
                    ),
                    "delta_A_hat_3p0_minus_2p25eq2p5": float(
                        wide.iloc[row_index]["A_hat"]
                        - narrow.iloc[row_index]["A_hat"]
                    ),
                    "scope": (
                        "paired conditional difference with common Poisson counts "
                        "in the shared [60,70) MeV bins"
                    ),
                }
            )
    individual_frame = pd.DataFrame(individual)
    individual_frame.to_csv(
        HERE / "derived" / "paired_window_differences.csv", index=False
    )
    metrics = [
        column
        for column in individual_frame.columns
        if column.startswith(("eps2_", "delta_"))
    ]
    summary_records = []
    for mass, group in individual_frame.groupby("mass_GeV", sort=True):
        record: dict[str, Any] = {
            "mass_GeV": float(mass),
            "paired_draw_count": int(len(group)),
            "summary_scope": (
                "pointwise descriptive paired differences across ten conditional draws"
            ),
        }
        for metric in metrics:
            values = group[metric].to_numpy(float)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_q16"] = float(np.quantile(values, 0.16))
            record[f"{metric}_q84"] = float(np.quantile(values, 0.84))
        summary_records.append(record)
    pd.DataFrame(summary_records).to_csv(
        HERE / "derived" / "paired_window_difference_summary.csv", index=False
    )


def load_spectra():
    root = uproot.open(ROOT_FILE)
    source_values, source_edges = root[SOURCE_KEY].to_numpy(flow=False)
    source_rebin, rebin_edges, centers = rebin_five(source_values, source_edges)
    spectra: dict[str, list[np.ndarray]] = {window: [] for window in WINDOWS}
    expectations = {}
    for window in WINDOWS:
        for draw_index in range(10):
            key = (
                f"gp/{window}/draw_{draw_index:02d}/"
                "preselection/h_invM_8000"
            )
            values, edges = root[key].to_numpy(flow=False)
            if not np.array_equal(edges, source_edges):
                raise RuntimeError(f"{key}: inconsistent histogram edges")
            spectra[window].append(rebin_five(values, edges)[0])
        expectation_key = f"expectations/{window}/gp_mean_m065"
        expectation, edges = root[expectation_key].to_numpy(flow=False)
        expectations[window] = rebin_five(expectation, edges)[0]
    return source_rebin, rebin_edges, centers, spectra, expectations


def draw_main_figure(
    all_frames: dict[str, list[pd.DataFrame]],
    baseline: pd.DataFrame,
) -> list[Path]:
    source, _, centers, spectra, expectations = load_spectra()
    mass_mask = (centers >= 0.050) & (centers <= 0.250)
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.4, 10.2),
        sharex="col",
        gridspec_kw={
            "height_ratios": [1.05, 1.0, 1.0],
            "hspace": 0.08,
            "wspace": 0.16,
        },
    )
    all_limits = []
    all_p0 = []
    for frames in all_frames.values():
        for frame in frames:
            all_limits.extend(frame["eps2_up"].to_numpy(float))
            all_p0.extend(frame["p0_analytic"].to_numpy(float))
    all_limits.extend(baseline["eps2_up"].to_numpy(float))
    all_p0.extend(baseline["p0_analytic"].to_numpy(float))
    positive_limits = np.array(all_limits)[np.array(all_limits) > 0.0]
    positive_p0 = np.array(all_p0)[np.array(all_p0) > 0.0]
    limit_lo = 10 ** np.floor(np.log10(np.min(positive_limits) * 0.75))
    limit_hi = 10 ** np.ceil(np.log10(np.max(positive_limits) * 1.15))
    p0_floor = max(min(np.min(positive_p0) * 0.5, 1.0e-7), 1.0e-12)
    base_mass = 1000.0 * baseline["mass_GeV"].to_numpy(float)

    for column, (window, info) in enumerate(WINDOWS.items()):
        color = info["color"]
        frames = all_frames[window]
        arrays_limit = np.vstack(
            [frame["eps2_up"].to_numpy(float) for frame in frames]
        )
        arrays_p0 = np.vstack(
            [frame["p0_analytic"].to_numpy(float) for frame in frames]
        )
        mass = 1000.0 * frames[0]["mass_GeV"].to_numpy(float)
        spectrum_array = np.vstack(spectra[window])
        top, middle, bottom = axes[:, column]
        top.set_title(
            f"{info['title']}  |  {info['subtitle']}",
            color=color,
            fontweight="semibold",
        )
        top.step(
            1000.0 * centers[mass_mask],
            source[mass_mask],
            where="mid",
            color=COLORS["original"],
            linestyle=(0, (3, 2)),
            linewidth=1.5,
            label="Original 2021 10%",
            zorder=1,
        )
        for draw_index, draw in enumerate(spectrum_array):
            top.step(
                1000.0 * centers[mass_mask],
                draw[mass_mask],
                where="mid",
                color=color,
                linewidth=0.75,
                alpha=0.24,
                label="10 individual draws" if draw_index == 0 else None,
                zorder=2,
            )
        top.step(
            1000.0 * centers[mass_mask],
            np.mean(spectrum_array, axis=0)[mass_mask],
            where="mid",
            color=color,
            linewidth=1.7,
            label="Arithmetic mean spectrum",
            zorder=3,
        )
        replacement = expectations[window] > 0.0
        top.plot(
            1000.0 * centers[replacement],
            expectations[window][replacement],
            color=COLORS["gp"],
            linestyle="--",
            linewidth=1.5,
            label="Fixed GP generating mean",
            zorder=4,
        )
        top.set_yscale("log")
        top.set_ylim(1.4e4, 1.6e6)
        top.legend(loc="lower left", frameon=False)

        for draw_index, values in enumerate(arrays_limit):
            middle.plot(
                mass,
                values,
                color=color,
                linewidth=0.75,
                alpha=0.25,
                label="10 individual curves" if draw_index == 0 else None,
            )
        q16_limit, q84_limit = np.quantile(arrays_limit, [0.16, 0.84], axis=0)
        middle.fill_between(
            mass,
            q16_limit,
            q84_limit,
            color=color,
            alpha=0.18,
            linewidth=0,
            label="Empirical 16-84% spread",
        )
        middle.plot(
            mass,
            np.median(arrays_limit, axis=0),
            color=color,
            linewidth=2.2,
            label="Median",
        )
        middle.plot(
            mass,
            np.mean(arrays_limit, axis=0),
            color=COLORS["mean"],
            linestyle=(0, (5, 2)),
            linewidth=1.6,
            label="Arithmetic mean",
        )
        middle.plot(
            base_mass,
            baseline["eps2_up"].to_numpy(float),
            color=COLORS["original"],
            linestyle=(0, (2, 2)),
            linewidth=1.35,
            label="Original v4.2 observed",
        )
        middle.set_yscale("log")
        middle.set_ylim(limit_lo, limit_hi)
        middle.yaxis.set_major_locator(LogLocator(base=10))
        middle.legend(loc="upper right", frameon=False, ncol=2)

        clipped = np.clip(arrays_p0, p0_floor, 0.5)
        for draw_index, values in enumerate(clipped):
            bottom.plot(
                mass,
                values,
                color=color,
                linewidth=0.72,
                alpha=0.25,
                label="10 individual curves" if draw_index == 0 else None,
            )
        q16_p0, q84_p0 = np.quantile(arrays_p0, [0.16, 0.84], axis=0)
        bottom.fill_between(
            mass,
            np.clip(q16_p0, p0_floor, 0.5),
            np.clip(q84_p0, p0_floor, 0.5),
            color=color,
            alpha=0.18,
            linewidth=0,
            label="Empirical 16-84% spread",
        )
        bottom.plot(
            mass,
            np.clip(np.median(arrays_p0, axis=0), p0_floor, 0.5),
            color=color,
            linewidth=2.2,
            label="Median",
        )
        bottom.plot(
            mass,
            np.clip(np.mean(arrays_p0, axis=0), p0_floor, 0.5),
            color=COLORS["mean"],
            linestyle=(0, (5, 2)),
            linewidth=1.6,
            label="Arithmetic mean",
        )
        bottom.plot(
            base_mass,
            np.clip(
                baseline["p0_analytic"].to_numpy(float), p0_floor, 0.5
            ),
            color=COLORS["original"],
            linestyle=(0, (2, 2)),
            linewidth=1.35,
            label="Original v4.2 observed",
        )
        bottom.set_yscale("log")
        bottom.set_ylim(p0_floor, 0.72)
        bottom.yaxis.set_major_locator(LogLocator(base=10))
        for z_value, label in ((3.0, r"$3\sigma$"), (5.0, r"$5\sigma$")):
            p_value = 0.5 * math.erfc(z_value / np.sqrt(2.0))
            bottom.axhline(
                p_value,
                color=COLORS["grid"],
                linestyle=":",
                linewidth=0.9,
            )
            if p_value >= p0_floor:
                bottom.text(
                    248.0,
                    p_value * 1.10,
                    label,
                    ha="right",
                    va="bottom",
                    fontsize=7.8,
                    color=COLORS["original"],
                )
        bottom.legend(loc="lower right", frameon=False, ncol=2)

        lo, hi = (
            (60.0, 70.0)
            if window == "window_2p25eq2p5"
            else (58.75, 71.25)
        )
        for axis in (top, middle, bottom):
            axis.axvspan(lo, hi, color=color, alpha=0.075, linewidth=0)
            axis.axvline(65.0, color=color, linestyle=":", linewidth=0.85)
            axis.set_xlim(50.0, 250.0)
            axis.xaxis.set_major_locator(MultipleLocator(25.0))
            axis.xaxis.set_minor_locator(MultipleLocator(5.0))
        bottom.set_xlabel(r"Mass hypothesis $m_{A'}$ [MeV]")

    for column in range(2):
        axes[0, column].set_ylabel("Events / 0.625 MeV")
        axes[1, column].set_ylabel(
            r"Observed 90% CL$_s$ limit on $\epsilon^2$"
        )
        axes[2, column].set_ylabel(r"Local asymptotic $p_0$")
    fig.suptitle(
        "2021 10% conditional GP-mean replacement-window ensembles",
        y=0.995,
        fontsize=14.0,
        fontweight="semibold",
        color=COLORS["ink"],
    )
    fig.text(
        0.5,
        0.012,
        "55-75 MeV uses reproduced max-LML review with targeted unchanged-card repeats; outside is single-attempt. Ribbons are empirical spreads.",
        ha="center",
        fontsize=9.2,
        color=COLORS["ink"],
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.955, bottom=0.072)
    outputs = [
        HERE / "plots" / "gp_window_ensemble_observed_limit_p0.pdf",
        HERE / "plots" / "gp_window_ensemble_observed_limit_p0.png",
    ]
    for output in outputs:
        fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return outputs


def draw_spectrum_zoom() -> list[Path]:
    source, _, centers, spectra, expectations = load_spectra()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.2, 6.8),
        sharex="col",
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.08, "wspace": 0.15},
    )
    zoom = (centers >= 0.055) & (centers <= 0.075)
    for column, (window, info) in enumerate(WINDOWS.items()):
        color = info["color"]
        array = np.vstack(spectra[window])
        top, bottom = axes[:, column]
        top.set_title(
            f"{info['title']}\n{info['subtitle']}",
            color=color,
            fontweight="semibold",
            pad=7.0,
        )
        top.step(
            1000.0 * centers[zoom],
            source[zoom],
            where="mid",
            color=COLORS["original"],
            linestyle=(0, (3, 2)),
            linewidth=1.6,
            label="Original 2021 10%",
        )
        for index, values in enumerate(array):
            top.step(
                1000.0 * centers[zoom],
                values[zoom],
                where="mid",
                color=color,
                linewidth=0.8,
                alpha=0.28,
                label="10 individual draws" if index == 0 else None,
            )
        top.step(
            1000.0 * centers[zoom],
            np.mean(array, axis=0)[zoom],
            where="mid",
            color=color,
            linewidth=1.9,
            label="Arithmetic mean spectrum",
        )
        replacement = expectations[window] > 0.0
        top.plot(
            1000.0 * centers[replacement],
            expectations[window][replacement],
            color=COLORS["gp"],
            linestyle="--",
            linewidth=1.6,
            label="Fixed GP generating mean",
        )
        top.legend(loc="best", frameon=False, ncol=2)
        top.set_ylabel("Events / 0.625 MeV")

        expected = expectations[window]
        for index, values in enumerate(array):
            residual = np.full_like(values, np.nan, dtype=float)
            replacement = expected > 0.0
            residual[replacement] = (
                values[replacement] - expected[replacement]
            ) / np.sqrt(expected[replacement])
            bottom.step(
                1000.0 * centers[replacement],
                residual[replacement],
                where="mid",
                color=color,
                linewidth=0.8,
                alpha=0.35,
                label="Individual Poisson residuals" if index == 0 else None,
            )
        bottom.axhline(0.0, color=COLORS["ink"], linewidth=0.9)
        bottom.axhline(2.0, color=COLORS["grid"], linestyle=":", linewidth=0.8)
        bottom.axhline(-2.0, color=COLORS["grid"], linestyle=":", linewidth=0.8)
        bottom.set_ylabel(r"$(n-\mu_{\rm GP})/\sqrt{\mu_{\rm GP}}$")
        bottom.set_xlabel("Invariant mass [MeV]")
        bottom.legend(loc="lower right", frameon=False)
        lo, hi = (
            (60.0, 70.0)
            if window == "window_2p25eq2p5"
            else (58.75, 71.25)
        )
        for axis in (top, bottom):
            axis.axvspan(lo, hi, color=color, alpha=0.075, linewidth=0)
            axis.axvline(65.0, color=color, linestyle=":", linewidth=0.85)
            axis.set_xlim(55.0, 75.0)
            axis.xaxis.set_major_locator(MultipleLocator(2.5))
            axis.xaxis.set_minor_locator(MultipleLocator(0.625))
    fig.suptitle(
        "Paired GP-mean pseudo-observations near 65 MeV",
        y=0.985,
        fontsize=13.5,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.012,
        "Matching draw indices use identical counts in the shared [60,70) MeV bins.",
        ha="center",
        fontsize=9.2,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.095)
    outputs = [
        HERE / "plots" / "gp_window_ensemble_central_spectra.pdf",
        HERE / "plots" / "gp_window_ensemble_central_spectra.png",
    ]
    for output in outputs:
        fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    (HERE / "derived").mkdir(parents=True, exist_ok=True)
    (HERE / "plots").mkdir(parents=True, exist_ok=True)
    set_style()
    all_frames, baseline, qc = load_and_audit_scans()
    pilot_reproduction = audit_pilot_reproduction(all_frames)
    summary = summarize_curves(all_frames)
    summarize_m065(all_frames, summary)
    summarize_paired_window_differences(all_frames)
    outputs = draw_main_figure(all_frames, baseline)
    outputs.extend(draw_spectrum_zoom())
    caption = (
        "Conditional GP-mean replacement-window ensembles for the 2021 10% "
        "spectrum. The +/-2.25 sigma and +/-2.5 sigma selections coincide on "
        "the production grid, while +/-3 sigma adds four 0.625 MeV edge bins. "
        "Ten fixed paired draws are shown. Solid and black-dashed curves give "
        "the pointwise median and arithmetic mean; the transparent 16-84% "
        "region is an empirical descriptive spread across these ten draws. "
        "Rows from 55-75 MeV use a reproduced maximum-finite-GP-marginal-"
        "likelihood review, with targeted unchanged-card repeats where the "
        "first two attempts differed; rows outside that interval use one "
        "attempt and can include optimizer-branch noise. "
        "The original v4.2 observed curve is dashed context only. These "
        "conditional replacements retain the outside-window observation and "
        "do not establish expected sensitivity, coverage, or a global p-value."
    )
    (HERE / "plots" / "CAPTION.txt").write_text(caption + "\n")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "draw_count_per_unique_geometry": 10,
        "pointwise_summary": (
            "arithmetic mean, median, and linear 16-84% empirical descriptive spread"
        ),
        "no_expected_limit_bands": True,
        "no_cls_calibration_or_limit_band_toys": True,
        "optimizer_limitation": (
            "All 420 rows in the 55-75 MeV region have a reproduced selected "
            "maximum-finite-LML state after two baseline attempts and targeted "
            "unchanged-card repeats; 17 rows retain documented multi-branch "
            "histories. Outside that region each draw has one attempt with 12 "
            "within-fit restarts, so full-grid optimizer-repeat stability is "
            "not established."
        ),
        "pilot_m065_draw00_state_reproduction_pass": bool(
            pilot_reproduction["state_reproduced"].all()
        ),
        "scan_qc_pass": bool(qc["pass_finite_grid_bound_gates"].all()),
        "files": [
            {
                "path": repo_relative(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
        "caption": caption,
    }
    path = HERE / "derived" / "plot_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {path}")
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
