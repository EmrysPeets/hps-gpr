#!/usr/bin/env python3
"""Compare truth-specific calibrated endpoints from saved ledgers; no fits."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import sys
import tempfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hps-v4p9p13-truth-mpl")
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCOPES = (("individual_2015_full", "2015, full sample", 19, 90),
          ("individual_2016_full", "2016, full sample", 39, 180),
          ("individual_2021_10pct", "2021, 10% sample", 50, 250),
          ("all_2015_2016_2021", "All three, shared coupling", 50, 90))
COLORS = {"profiled": "#0072B2", "fixed": "#D55E00"}
LABELS = {"profiled": "Gaussian profile", "fixed": "Fixed GP mean"}
STATUSES = {"resolved", "limited_mc", "right_censored"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 11.5,
    "axes.titlesize": 12.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
    "grid.linewidth": .5, "pdf.fonttype": 3, "ps.fonttype": 3,
    "savefig.facecolor": "white"})


def endpoint_ratio(frame, scope, method, lower, upper):
    """A finite ratio requires both positive endpoints and neither censor flag."""
    grid = np.arange(lower, upper+1)
    selected = frame[(frame.scope_key == scope) & (frame.method == method)]
    by_truth = {truth: selected[selected.truth == truth].set_index("mass_MeV").reindex(grid)
                for truth in ("gp", "stress")}
    gp, stress = by_truth["gp"], by_truth["stress"]
    values_gp = gp.eps2_display.to_numpy(float)
    values_stress = stress.eps2_display.to_numpy(float)
    missing = gp.status.isna().to_numpy() | stress.status.isna().to_numpy()
    censored = gp.status.eq("right_censored").to_numpy() | stress.status.eq("right_censored").to_numpy()
    valid = (~missing & ~censored & np.isfinite(values_gp) & np.isfinite(values_stress)
             & (values_gp > 0) & (values_stress > 0))
    ratio = np.full(len(grid), np.nan)
    np.divide(values_stress, values_gp, out=ratio, where=valid)
    valid &= np.isfinite(ratio) & (ratio > 0)
    ratio[~valid] = np.nan
    limited = valid & ~(gp.status.eq("resolved").to_numpy() & stress.status.eq("resolved").to_numpy())
    audit = dict(finite_pairs=int(valid.sum()), limited_mc_pairs=int(limited.sum()),
                 missing_pairs=int(missing.sum()), censored_pairs=int(censored.sum()),
                 other_nonfinite_pairs=int((~valid & ~missing & ~censored).sum()))
    return grid, ratio, limited, audit


def save(fig, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "png"):
        target = output/("truth_dependence."+extension)
        descriptor, temporary = tempfile.mkstemp(prefix=".truth_dependence", suffix="."+extension, dir=output)
        os.close(descriptor)
        try:
            fig.savefig(temporary, format=extension, dpi=220)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        paths.append(str(target))
    plt.close(fig)
    return paths


def draw(frame, output):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.8))
    fig.subplots_adjust(left=.095, right=.972, bottom=.17, top=.795,
                        wspace=.255, hspace=.53)
    fig.suptitle("Dependence on the generating truth", y=.978, fontsize=18)
    fig.text(.5, .937, r"Conditional observed 90% CL$_s$ endpoint: archived stress truth / mass-local GP truth",
             ha="center", fontsize=11.5)
    handles = [Line2D([], [], color=COLORS[method], lw=2, label=LABELS[method]) for method in COLORS]
    handles.append(Line2D([], [], color=".4", marker="o", markerfacecolor="white",
                          ls="none", markersize=5, label="Limited MC precision"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .909),
               ncol=3, frameon=False, fontsize=10.5, handlelength=2.3, columnspacing=2.)
    count = len(frame[["scope_key", "mass_MeV"]].drop_duplicates())
    pilot = "checkpoint_status" in frame and frame.checkpoint_status.eq("pilot").any()
    progress = "Pilot preview" if pilot else "Partial production" if count < 456 else "Complete mass grid"
    fig.text(.5, .849, f"{progress}: {count} of 456 mass hypotheses; each ratio uses two truth-specific endpoints",
             ha="center", fontsize=10.5, color=".28")
    audit = {}
    for ax, (scope, label, lower, upper) in zip(axes.flat, SCOPES):
        scope_audit, all_values = {}, []
        for method, color in COLORS.items():
            grid, ratio, limited, row = endpoint_ratio(frame, scope, method, lower, upper)
            scope_audit[method] = row
            all_values.extend(ratio[np.isfinite(ratio)])
            # Reindexing the full integer grid preserves every missing/censored gap.
            ax.plot(grid, ratio, color=color, lw=1.55, marker=".", markersize=2.8, zorder=3)
            ax.scatter(grid[limited], ratio[limited], s=25, facecolors="white", edgecolors=color,
                       linewidths=.9, zorder=4)
        audit[scope] = scope_audit
        minimum, maximum = min([.5, *all_values]), max([2., *all_values])
        ax.set_yscale("log")
        ax.set(xlim=(lower-.5, upper+.5), ylim=(minimum/1.16, maximum*1.16),
               xlabel=r"Mass hypothesis $m_{A'}$ (MeV)", ylabel="Stress / GP endpoint")
        ax.set_title(label, pad=10)
        ax.axhline(1., color=".4", lw=.9, ls=(0, (3, 3)), zorder=2)
        ax.set_axisbelow(True)
        ax.grid(color=".88", alpha=.7)
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.), numticks=8))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
        ax.tick_params(direction="out", width=.7)
        if not all_values:
            ax.text(.5, .7, "No finite endpoint pairs", transform=ax.transAxes,
                    ha="center", va="center", color=".45", fontsize=11)
    fig.text(.5, .10, "The dashed line at 1 denotes equal observed endpoints; this ratio is not a coverage test.",
             ha="center", fontsize=10.4)
    fig.text(.5, .074, "Open circles: at least one endpoint has limited MC precision. Missing or censored endpoints leave gaps.",
             ha="center", fontsize=10.2, color=".25")
    fig.text(.5, .048, "The all-three comparison covers two joint truth scenarios only; it does not cover every mixed constituent truth.",
             ha="center", fontsize=10.2, color=".25")
    return save(fig, output), audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE/"summary/truth_specific_limits.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE/"figures")
    args = parser.parse_args()
    payload = args.input.read_bytes()
    frame = pd.read_csv(io.BytesIO(payload))
    required = {"scope_key", "mass_MeV", "method", "truth", "status", "eps2_display"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing truth-specific fields: {required-set(frame.columns)}")
    if frame.duplicated(["scope_key", "mass_MeV", "method", "truth"]).any():
        raise ValueError("Duplicate truth/method coordinates; do not pool endpoints")
    allowed = {(scope, mass) for scope, _, lower, upper in SCOPES for mass in range(lower, upper+1)}
    if not set(zip(frame.scope_key, frame.mass_MeV)).issubset(allowed):
        raise ValueError("Truth-specific rows leave the declared mass grids")
    if not set(frame.method).issubset(COLORS) or not set(frame.truth).issubset({"gp", "stress"}):
        raise ValueError("Unexpected method or truth family")
    if not set(frame.status).issubset(STATUSES):
        raise ValueError(f"Unexpected endpoint statuses: {set(frame.status)-STATUSES}")
    outputs, audit = draw(frame, args.output_dir)
    provenance = dict(input=str(args.input), source_sha256=hashlib.sha256(payload).hexdigest(),
                      source_rows=len(frame), numerator="stress.eps2_display",
                      denominator="gp.eps2_display", ratio_audit=audit, outputs=outputs,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      output_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in outputs})
    (args.output_dir/"truth_plot_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
