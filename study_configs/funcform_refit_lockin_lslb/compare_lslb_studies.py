#!/usr/bin/env python3
"""Compare 2015 functional-form lock-in closure studies across LS lower bounds."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDIES = (
    ("lslb0p5", 0.5, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_0pt5"),
    ("lslb1p0", 1.0, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt0"),
    ("lslb1p5", 1.5, "outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_1pt5"),
)


def _read_toy_table(label: str, lslb: float, outdir: str) -> pd.DataFrame | None:
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
            df["lslb"] = float(lslb)
            df["source_csv"] = str(path)
            return df
    print(f"[compare_lslb_studies] warning: no merged toy CSV found for {label} under {outdir}")
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
    group_cols = ["study", "lslb", "mass_GeV", "inj_nsigma"]
    rows: list[dict] = []
    for keys, sub in df.groupby(group_cols, dropna=False, sort=True):
        study, lslb, mass, inj_nsigma = keys
        row = {
            "study": study,
            "lslb": float(lslb),
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


def _subplot_grid(n_panels: int) -> tuple[int, int]:
    ncols = min(3, max(1, n_panels))
    nrows = int(math.ceil(n_panels / ncols))
    return nrows, ncols


def _plot_metric_by_strength(
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
        print(f"[compare_lslb_studies] skipped {filename.name}: no finite {metric}")
        return

    strengths = sorted(float(x) for x in sub["inj_nsigma"].dropna().unique())
    nrows, ncols = _subplot_grid(len(strengths))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, strength in zip(axes_arr, strengths):
        panel = sub[np.isclose(sub["inj_nsigma"].to_numpy(float), strength)]
        for lslb, grp in panel.groupby("lslb", sort=True):
            grp = grp.sort_values("mass_GeV")
            ax.plot(
                grp["mass_GeV"].to_numpy(float),
                grp[metric].to_numpy(float),
                marker="o",
                linewidth=1.8,
                label=f"LSLB {lslb:g}",
            )
        if hline is not None:
            ax.axhline(float(hline), color="0.25", linestyle="--", linewidth=1.0)
        ax.set_title(f"Injected {strength:g} sigma")
        ax.set_xlabel("Mass [GeV]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    for ax in axes_arr[len(strengths) :]:
        ax.axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(handles)))
        fig.subplots_adjust(top=0.86)
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    print(f"Wrote {filename}")


def _plot_kernel_diagnostics(summary: pd.DataFrame, outdir: Path) -> None:
    diag_cols = [c for c in ("ls_lo_mean", "ls_hi_mean", "ls_opt_mean", "initial_ls_opt_mean", "refit_ls_opt_mean") if c in summary.columns]
    if not diag_cols:
        print("[compare_lslb_studies] skipped kernel diagnostics: no length-scale columns")
        return

    sub = summary.copy()
    sub = sub[np.isclose(sub["inj_nsigma"].to_numpy(float), 0.0)]
    if sub.empty:
        sub = summary.groupby(["study", "lslb", "mass_GeV"], as_index=False)[diag_cols].mean(numeric_only=True)

    nrows, ncols = _subplot_grid(len(diag_cols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes_arr, diag_cols):
        finite = sub[np.isfinite(sub[col].to_numpy(float))]
        for lslb, grp in finite.groupby("lslb", sort=True):
            grp = grp.sort_values("mass_GeV")
            ax.plot(
                grp["mass_GeV"].to_numpy(float),
                grp[col].to_numpy(float),
                marker="o",
                linewidth=1.8,
                label=f"LSLB {lslb:g}",
            )
        ax.set_title(col.replace("_mean", ""))
        ax.set_xlabel("Mass [GeV]")
        ax.set_ylabel("Length scale")
        ax.grid(True, alpha=0.25)

    for ax in axes_arr[len(diag_cols) :]:
        ax.axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(handles)))
        fig.subplots_adjust(top=0.86)
    fig.tight_layout()
    outfile = outdir / "length_scale_diagnostics_by_lslb.png"
    fig.savefig(outfile, dpi=180)
    plt.close(fig)
    print(f"Wrote {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/study_2015_w1p64_train1p96_95CL_funcform100_refit_lslb_comparison",
        help="Directory for combined CSVs and cross-study comparison plots.",
    )
    args = parser.parse_args()

    frames = []
    for label, lslb, outdir in STUDIES:
        frame = _read_toy_table(label, lslb, outdir)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise SystemExit("No study toy tables were available. Run compile_all.sh after the SLURM jobs finish.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    toys = _prepare(pd.concat(frames, ignore_index=True))
    toys_out = outdir / "combined_toys_by_lslb.csv"
    toys.to_csv(toys_out, index=False)
    print(f"Wrote {toys_out}")

    summary = _summarize(toys)
    summary_out = outdir / "summary_by_lslb_mass_injsigma.csv"
    summary.to_csv(summary_out, index=False)
    print(f"Wrote {summary_out}")

    _plot_metric_by_strength(
        summary,
        metric="pull_mean",
        ylabel="Pull mean",
        filename=outdir / "pull_mean_by_lslb.png",
        include_zero=True,
        hline=0.0,
    )
    _plot_metric_by_strength(
        summary,
        metric="pull_width",
        ylabel="Pull width",
        filename=outdir / "pull_width_by_lslb.png",
        include_zero=True,
        hline=1.0,
    )
    _plot_metric_by_strength(
        summary,
        metric="Ahat_over_Ainj_mean",
        ylabel=r"$\hat{A}/A_{inj}$",
        filename=outdir / "Ahat_over_Ainj_by_lslb.png",
        include_zero=False,
        hline=1.0,
    )
    _plot_metric_by_strength(
        summary,
        metric="Zhat_minus_Zinj_mean",
        ylabel=r"$\hat{Z}-Z_{inj}$",
        filename=outdir / "Zhat_minus_Zinj_by_lslb.png",
        include_zero=True,
        hline=0.0,
    )
    _plot_metric_by_strength(
        summary,
        metric="Ahat_over_sigmaAref_minus_Zinj_mean",
        ylabel=r"$\hat{A}/\sigma_{A,ref}-Z_{inj}$",
        filename=outdir / "Ahat_over_sigmaAref_minus_Zinj_by_lslb.png",
        include_zero=True,
        hline=0.0,
    )
    _plot_metric_by_strength(
        summary,
        metric="sigmaA_over_sigmaAref_mean",
        ylabel=r"$\sigma_A/\sigma_{A,ref}$",
        filename=outdir / "sigmaA_over_sigmaAref_by_lslb.png",
        include_zero=True,
        hline=1.0,
    )
    _plot_metric_by_strength(
        summary,
        metric="Nsig_train_over_Nsig_win_mean",
        ylabel=r"$N_{sig,train}/N_{sig,win}$",
        filename=outdir / "Nsig_train_over_Nsig_win_by_lslb.png",
        include_zero=True,
        hline=None,
    )
    _plot_kernel_diagnostics(summary, outdir)


if __name__ == "__main__":
    main()
