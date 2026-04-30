#!/usr/bin/env python3
"""Compare 2015 functional-form signal-kernel closure studies."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDIES = (
    ("lslb0p5_sigkl1p0", 0.5, 1.0, 1.55, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_0pt5_sigkernel_l1pt0_w1pt55"),
    ("lslb0p5_sigkl1p5", 0.5, 1.5, 1.24, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_0pt5_sigkernel_l1pt5_w1pt24"),
    ("lslb0p5_sigkl2p0", 0.5, 2.0, 1.13, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_0pt5_sigkernel_l2pt0_w1pt13"),
    ("lslb1p0_sigkl1p0", 1.0, 1.0, 1.55, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt0_sigkernel_l1pt0_w1pt55"),
    ("lslb1p0_sigkl1p5", 1.0, 1.5, 1.24, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt0_sigkernel_l1pt5_w1pt24"),
    ("lslb1p0_sigkl2p0", 1.0, 2.0, 1.13, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt0_sigkernel_l2pt0_w1pt13"),
    ("lslb1p5_sigkl1p0", 1.5, 1.0, 1.55, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt5_sigkernel_l1pt0_w1pt55"),
    ("lslb1p5_sigkl1p5", 1.5, 1.5, 1.24, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt5_sigkernel_l1pt5_w1pt24"),
    ("lslb1p5_sigkl2p0", 1.5, 2.0, 1.13, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt5_sigkernel_l2pt0_w1pt13"),
)


def _read_toy_table(label: str, gp_lslb: float, sig_ell: float, sig_width: float, outdir: str) -> pd.DataFrame | None:
    candidates = [
        Path(outdir) / "injection_summary" / "inj_extract_toys_2015.csv",
        Path(outdir) / "inj_extract_toys_2015.csv",
        Path(outdir) / "injection_extraction" / "inj_extract_toys_2015.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["study"] = label
            df["gp_lslb"] = float(gp_lslb)
            df["sigk_ell"] = float(sig_ell)
            df["sigk_width"] = float(sig_width)
            df["source_csv"] = str(path)
            return df
    print(f"[compare_signal_kernel_studies] warning: no merged toy CSV found for {label} under {outdir}")
    return None


def _numeric(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mass_GeV"] = _numeric(out, "mass_GeV")
    out["inj_nsigma"] = _numeric(out, "inj_nsigma")
    out["A_hat"] = _numeric(out, "A_hat")
    out["A_inj"] = _numeric(out, "A_inj")
    out["sigma_A"] = _numeric(out, "sigma_A")
    out["sigmaA_ref"] = _numeric(out, "sigmaA_ref")
    out["Zhat"] = _numeric(out, "Zhat")
    if "pull_param" not in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["pull_param"] = (out["A_hat"] - out["A_inj"]) / out["sigma_A"]
    else:
        out["pull_param"] = _numeric(out, "pull_param")

    with np.errstate(divide="ignore", invalid="ignore"):
        z_from_fit = out["A_hat"] / out["sigma_A"]
        out["Zhat_eff"] = np.where(np.isfinite(out["Zhat"]), out["Zhat"], z_from_fit)
        out["Ahat_over_Ainj"] = out["A_hat"] / out["A_inj"]
        out["Ahat_over_sigmaAref_minus_Zinj"] = out["A_hat"] / out["sigmaA_ref"] - out["inj_nsigma"]
        out["sigmaA_over_sigmaAref"] = out["sigma_A"] / out["sigmaA_ref"]
        out["Nsig_train_over_Nsig_win"] = _numeric(out, "Nsig_train") / _numeric(out, "Nsig_win")
    out["Zhat_minus_Zinj"] = out["Zhat_eff"] - out["inj_nsigma"]
    return out


def _finite_mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _finite_std(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["study", "gp_lslb", "sigk_ell", "sigk_width", "mass_GeV", "inj_nsigma"]
    rows: list[dict] = []
    for keys, sub in df.groupby(group_cols, dropna=False, sort=True):
        study, gp_lslb, sigk_ell, sigk_width, mass, inj_nsigma = keys
        row = {
            "study": study,
            "gp_lslb": float(gp_lslb),
            "sigk_ell": float(sigk_ell),
            "sigk_width": float(sigk_width),
            "mass_GeV": float(mass),
            "inj_nsigma": float(inj_nsigma),
            "n_toys": int(len(sub)),
            "pull_mean": _finite_mean(sub["pull_param"]),
            "pull_width": _finite_std(sub["pull_param"]),
            "Ahat_over_Ainj_mean": _finite_mean(sub["Ahat_over_Ainj"]),
            "Zhat_minus_Zinj_mean": _finite_mean(sub["Zhat_minus_Zinj"]),
            "Ahat_over_sigmaAref_minus_Zinj_mean": _finite_mean(sub["Ahat_over_sigmaAref_minus_Zinj"]),
            "sigmaA_over_sigmaAref_mean": _finite_mean(sub["sigmaA_over_sigmaAref"]),
            "Nsig_train_over_Nsig_win_mean": _finite_mean(sub["Nsig_train_over_Nsig_win"]),
        }
        for col in ("ls_lo", "ls_hi", "ls_opt", "initial_ls_opt", "refit_ls_opt"):
            if col in sub.columns:
                row[f"{col}_mean"] = _finite_mean(sub[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _metric_panels(
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    filename: Path,
    include_zero: bool,
    hline: float | None = None,
) -> None:
    sub = summary[np.isfinite(summary[metric].to_numpy(float))].copy()
    if not include_zero:
        sub = sub[np.abs(sub["inj_nsigma"].to_numpy(float)) > 1.0e-9]
    if sub.empty:
        print(f"[compare_signal_kernel_studies] skipped {filename.name}: no finite {metric}")
        return

    strengths = sorted(float(x) for x in sub["inj_nsigma"].dropna().unique())
    sig_ells = sorted(float(x) for x in sub["sigk_ell"].dropna().unique())
    nrows = len(strengths)
    ncols = len(sig_ells)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * max(1, ncols), 2.9 * max(1, nrows)),
        sharex=True,
        squeeze=False,
    )

    for row_idx, strength in enumerate(strengths):
        for col_idx, sig_ell in enumerate(sig_ells):
            ax = axes[row_idx][col_idx]
            panel = sub[
                np.isclose(sub["inj_nsigma"].to_numpy(float), strength)
                & np.isclose(sub["sigk_ell"].to_numpy(float), sig_ell)
            ]
            for gp_lslb, grp in panel.groupby("gp_lslb", sort=True):
                grp = grp.sort_values("mass_GeV")
                ax.plot(
                    grp["mass_GeV"].to_numpy(float),
                    grp[metric].to_numpy(float),
                    marker="o",
                    linewidth=1.6,
                    label=f"GP LSLB {gp_lslb:g}",
                )
            if hline is not None:
                ax.axhline(float(hline), color="0.25", linestyle="--", linewidth=1.0)
            if row_idx == 0:
                width = panel["sigk_width"].dropna().unique()
                width_label = f", w={float(width[0]):g}" if len(width) else ""
                ax.set_title(f"Signal ell {sig_ell:g}{width_label}")
            if col_idx == 0:
                ax.set_ylabel(f"Injected {strength:g} sigma\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            ax.set_xlabel("Mass [GeV]")
            ax.grid(True, alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(handles)))
        fig.subplots_adjust(top=0.94)
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    print(f"Wrote {filename}")


def _metric_heatmap(summary: pd.DataFrame, *, metric: str, filename: Path, include_zero: bool) -> None:
    sub = summary[np.isfinite(summary[metric].to_numpy(float))].copy()
    if not include_zero:
        sub = sub[np.abs(sub["inj_nsigma"].to_numpy(float)) > 1.0e-9]
    if sub.empty:
        return
    pivot = (
        sub.groupby(["sigk_ell", "gp_lslb"], as_index=False)[metric]
        .mean(numeric_only=True)
        .pivot(index="sigk_ell", columns="gp_lslb", values=metric)
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:g}" for x in pivot.index])
    ax.set_xlabel("Background GP lower bound / sigma_m")
    ax.set_ylabel("Signal kernel length / sigma_m")
    ax.set_title(metric.replace("_", " "))
    for yi, sig_ell in enumerate(pivot.index):
        for xi, gp_lslb in enumerate(pivot.columns):
            val = pivot.loc[sig_ell, gp_lslb]
            if np.isfinite(val):
                ax.text(xi, yi, f"{val:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    print(f"Wrote {filename}")


def _plot_kernel_diagnostics(summary: pd.DataFrame, outdir: Path) -> None:
    diag_cols = [c for c in ("ls_lo_mean", "ls_hi_mean", "ls_opt_mean", "initial_ls_opt_mean", "refit_ls_opt_mean") if c in summary.columns]
    if not diag_cols:
        print("[compare_signal_kernel_studies] skipped kernel diagnostics: no length-scale columns")
        return

    sub = summary.copy()
    sub = sub[np.isclose(sub["inj_nsigma"].to_numpy(float), 0.0)]
    if sub.empty:
        sub = summary.groupby(["gp_lslb", "sigk_ell", "sigk_width", "mass_GeV"], as_index=False)[diag_cols].mean(numeric_only=True)

    sig_ells = sorted(float(x) for x in sub["sigk_ell"].dropna().unique())
    for col in diag_cols:
        fig, axes = plt.subplots(1, len(sig_ells), figsize=(4.8 * max(1, len(sig_ells)), 3.6), sharex=True, squeeze=False)
        for ax, sig_ell in zip(axes.ravel(), sig_ells):
            panel = sub[np.isclose(sub["sigk_ell"].to_numpy(float), sig_ell)]
            for gp_lslb, grp in panel.groupby("gp_lslb", sort=True):
                grp = grp.sort_values("mass_GeV")
                ax.plot(
                    grp["mass_GeV"].to_numpy(float),
                    grp[col].to_numpy(float),
                    marker="o",
                    linewidth=1.6,
                    label=f"GP LSLB {gp_lslb:g}",
                )
            ax.set_title(f"Signal ell {sig_ell:g}")
            ax.set_xlabel("Mass [GeV]")
            ax.set_ylabel(col.replace("_mean", ""))
            ax.grid(True, alpha=0.25)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(3, len(handles)))
            fig.subplots_adjust(top=0.86)
        fig.tight_layout()
        outfile = outdir / f"{col}_by_signal_kernel_and_gplslb.png"
        fig.savefig(outfile, dpi=180)
        plt.close(fig)
        print(f"Wrote {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_signal_kernel_comparison",
        help="Directory for combined CSVs and cross-study comparison plots.",
    )
    args = parser.parse_args()

    frames = []
    for label, gp_lslb, sig_ell, sig_width, outdir in STUDIES:
        frame = _read_toy_table(label, gp_lslb, sig_ell, sig_width, outdir)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise SystemExit("No study toy tables were available. Run compile_all.sh after the SLURM jobs finish.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    toys = _prepare(pd.concat(frames, ignore_index=True))
    toys_out = outdir / "combined_toys_by_signal_kernel_and_gplslb.csv"
    toys.to_csv(toys_out, index=False)
    print(f"Wrote {toys_out}")

    summary = _summarize(toys)
    summary_out = outdir / "summary_by_signal_kernel_gplslb_mass_injsigma.csv"
    summary.to_csv(summary_out, index=False)
    print(f"Wrote {summary_out}")

    metrics = [
        ("pull_mean", "Pull mean", True, 0.0),
        ("pull_width", "Pull width", True, 1.0),
        ("Ahat_over_Ainj_mean", r"$\hat{A}/A_{inj}$", False, 1.0),
        ("Zhat_minus_Zinj_mean", r"$\hat{Z}-Z_{inj}$", True, 0.0),
        ("Ahat_over_sigmaAref_minus_Zinj_mean", r"$\hat{A}/\sigma_{A,ref}-Z_{inj}$", True, 0.0),
        ("sigmaA_over_sigmaAref_mean", r"$\sigma_A/\sigma_{A,ref}$", True, 1.0),
        ("Nsig_train_over_Nsig_win_mean", r"$N_{sig,train}/N_{sig,win}$", True, None),
    ]
    for metric, ylabel, include_zero, hline in metrics:
        _metric_panels(
            summary,
            metric=metric,
            ylabel=ylabel,
            filename=outdir / f"{metric}_by_signal_kernel_and_gplslb.png",
            include_zero=include_zero,
            hline=hline,
        )
        _metric_heatmap(
            summary,
            metric=metric,
            filename=outdir / f"{metric}_average_heatmap.png",
            include_zero=include_zero,
        )
    _plot_kernel_diagnostics(summary, outdir)


if __name__ == "__main__":
    main()
