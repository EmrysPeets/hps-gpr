#!/usr/bin/env python3
"""Render separate-cell validation diagnostics; never refit or pool toy counts."""
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hps-v4p9p13-validation-mpl")
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCOPES = (("individual_2015_full", "2015, full sample", 19, 90),
          ("individual_2016_full", "2016, full sample", 39, 180),
          ("individual_2021_10pct", "2021, 10% sample", 50, 250),
          ("all_2015_2016_2021", "All three, shared coupling", 50, 90))
COLORS = {"profiled": "#0072B2", "fixed": "#D55E00"}
MARKERS = {"gp": "o", "stress": "^"}
STYLES = {"gp": "-", "stress": (0, (5, 2.5))}
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 11.5,
    "axes.titlesize": 12.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
    "grid.linewidth": .5, "pdf.fonttype": 3, "ps.fonttype": 3,
    "savefig.facecolor": "white"})


def description(frame):
    points = frame[["scope_key", "mass_MeV"]].drop_duplicates()
    pilot = "checkpoint_status" in frame and (frame.checkpoint_status == "pilot").any()
    progress = "Pilot preview" if pilot else "Partial production" if len(points) < 456 else "Complete mass grid"
    sizes = sorted(frame.n.astype(int).unique())
    counts = (f"{sizes[0]:,} independent spectra per cell" if len(sizes) == 1 else
              f"{min(sizes):,}–{max(sizes):,} independent spectra per cell" if sizes else "No completed validation cells")
    return f"{progress}: {len(points)} of 456 mass hypotheses; {counts}"


def canvas(title, subtitle, progress, truth_style):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.8))
    fig.subplots_adjust(left=.095, right=.972, bottom=.17, top=.795,
                        wspace=.255, hspace=.53)
    fig.suptitle(title, y=.978, fontsize=18)
    fig.text(.5, .937, subtitle, ha="center", fontsize=11.5)
    handles = [Line2D([], [], color=COLORS[method], lw=2,
                      label=label) for method, label in
               (("profiled", "Gaussian profile"), ("fixed", "Fixed GP mean"))]
    for truth, label in (("gp", "GP mean truth"), ("stress", "Archived stress truth")):
        handles.append(Line2D([], [], color=".3", marker=MARKERS[truth] if truth_style == "markers" else None,
                              ls="none" if truth_style == "markers" else STYLES[truth],
                              markersize=6, lw=1.6, label=label))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .909),
               ncol=4, frameon=False, fontsize=10.5, handlelength=2.3,
               columnspacing=1.6)
    fig.text(.5, .849, progress, ha="center", fontsize=10.8, color=".28")
    for ax, (_, label, _, _) in zip(axes.flat, SCOPES):
        ax.set_title(label, pad=10)
        ax.set_axisbelow(True)
        ax.grid(color=".88", alpha=.7)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(direction="out", width=.7)
    return fig, axes


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        descriptor, temporary = tempfile.mkstemp(prefix="."+name, suffix="."+extension, dir=output)
        os.close(descriptor)
        try:
            fig.savefig(temporary, format=extension, dpi=220)
            os.replace(temporary, output/(name+"."+extension))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    plt.close(fig)


def exclusion_figure(frame, output):
    data = frame[frame.strength == 5]
    fig, axes = canvas("Independent validation of 90% CL$_s$ exclusion",
                       r"Signal injection $A_{\rm true}=5\sigma_{\rm ref}$; paired background treatments",
                       description(frame), "markers")
    maximum_x = float(data.raw_exclusion_fraction.max()) if len(data) else .1
    maximum_y = float(data.exclusion_fraction.max()) if len(data) else .1
    # Every outlier remains inside the common axes, including fractions of1.
    xmax = max(.16, maximum_x*1.06+.015)
    ymax = max(.16, maximum_y*1.08+.015)
    for ax, (scope, _, _, _) in zip(axes.flat, SCOPES):
        selected = data[data.scope_key == scope]
        ax.axhline(.1, color=".42", lw=.85, ls=(0, (3, 3)), zorder=1)
        ax.axvline(.1, color=".42", lw=.85, ls=(0, (3, 3)), zorder=1)
        for method, color in COLORS.items():
            for truth, marker in MARKERS.items():
                cell = selected[(selected.method == method) & (selected.truth == truth)]
                ax.scatter(cell.raw_exclusion_fraction, cell.exclusion_fraction,
                           c=color, marker=marker, s=23 if truth == "gp" else 29,
                           alpha=.67, linewidths=.3, edgecolors="white", zorder=3)
        ax.set(xlim=(-.007*xmax, xmax), ylim=(-.007*ymax, ymax),
               xlabel="Raw asymptotic exclusion fraction",
               ylabel="Calibrated exclusion fraction")
        if selected.empty:
            ax.text(.5, .72, "No completed checkpoints", transform=ax.transAxes,
                    ha="center", va="center", color=".45", fontsize=11)
    fig.text(.5, .075, r"One point per mass, truth and method; exclusion means CL$_s(A_{\rm true})<0.10$.",
             ha="center", fontsize=10.5)
    fig.text(.5, .048, "Dashed guides mark 0.10. Cells retain their own counts; the calibrated test uses the two-truth envelope.",
             ha="center", fontsize=10.2, color=".25")
    save(fig, output, "validation_exclusion")


def bias_figure(frame, output):
    data = frame[frame.strength == 0]
    fig, axes = canvas("Background-only signal extraction",
                       r"Ensemble mean amplitude relative to the fixed reference uncertainty $\sigma_{\rm ref}$",
                       description(frame), "lines")
    low, high = ((min(0., float(data.signal_bias_sigma_mc95_low.min())),
                  max(0., float(data.signal_bias_sigma_mc95_high.max()))) if len(data) else (-.2, .2))
    padding = max(.08, (high-low)*.08)
    for ax, (scope, _, lower, upper) in zip(axes.flat, SCOPES):
        selected = data[data.scope_key == scope]
        grid = np.arange(lower, upper+1)
        for method, color in COLORS.items():
            for truth, style in STYLES.items():
                cell = selected[(selected.method == method) & (selected.truth == truth)]
                # Missing masses stay NaN: a partial scan cannot bridge a gap.
                curve = cell.set_index("mass_MeV").reindex(grid)
                ax.fill_between(grid, curve.signal_bias_sigma_mc95_low.to_numpy(float),
                                curve.signal_bias_sigma_mc95_high.to_numpy(float),
                                color=color, alpha=.12, linewidth=0, zorder=1)
                ax.plot(grid, curve.signal_bias_sigma, color=color, ls=style, lw=1.55,
                        marker=".", markersize=3, alpha=.95)
        ax.axhline(0., color=".42", lw=.85, ls=(0, (3, 3)))
        ax.set(xlim=(lower-.5, upper+.5), ylim=(low-padding, high+padding),
               xlabel=r"Mass hypothesis $m_{A'}$ (MeV)",
               ylabel=r"$\langle\widehat A\rangle/\sigma_{\rm ref}$")
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        if selected.empty:
            ax.text(.5, .55, "No completed checkpoints", transform=ax.transAxes,
                    ha="center", va="center", color=".45", fontsize=11)
    fig.text(.5, .10, "Shading: approximate pointwise 95% Monte Carlo intervals for the ensemble mean; no simultaneous band.",
             ha="center", fontsize=10.2, color=".25")
    fig.text(.5, .075, r"$\sigma_{\rm ref}$ is the frozen Gaussian-profile Fisher uncertainty at each mass, common to both methods.",
             ha="center", fontsize=10.5)
    fig.text(.5, .048, "This is a mean amplitude in fixed reference units, not the historical mean of per-toy pulls. No counts are pooled.",
             ha="center", fontsize=10.2, color=".25")
    save(fig, output, "bias_by_truth")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE/"summary/validation_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE/"figures")
    args = parser.parse_args()
    payload = args.input.read_bytes()
    frame = pd.read_csv(io.BytesIO(payload))
    required = {"scope_key", "mass_MeV", "truth", "strength", "method", "n",
                "exclusion_fraction", "raw_exclusion_fraction", "signal_bias_sigma",
                "signal_bias_sigma_mc_se", "signal_bias_sigma_mc95_low", "signal_bias_sigma_mc95_high"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing validation columns: {required-set(frame.columns)}")
    if frame.duplicated(["scope_key", "mass_MeV", "truth", "strength", "method"]).any():
        raise ValueError("Duplicate validation cells; do not pool them in figures")
    allowed = {(scope, mass) for scope, _, lower, upper in SCOPES for mass in range(lower, upper+1)}
    if not set(zip(frame.scope_key, frame.mass_MeV)).issubset(allowed):
        raise ValueError("Validation rows leave the declared mass grids")
    if not set(frame.method).issubset(COLORS) or not set(frame.truth).issubset(MARKERS):
        raise ValueError("Unexpected method or truth family")
    for column in ("exclusion_fraction", "raw_exclusion_fraction"):
        if not np.isfinite(frame[column]).all() or not frame[column].between(0, 1).all():
            raise ValueError(f"Invalid rejection fractions in {column}")
    if not np.isfinite(frame.signal_bias_sigma).all() or (frame.n <= 0).any():
        raise ValueError("Invalid bias or toy-count entries")
    for column in ("signal_bias_sigma_mc_se", "signal_bias_sigma_mc95_low", "signal_bias_sigma_mc95_high"):
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"Nonfinite bias MC interval entries in {column}")
    if ((frame.signal_bias_sigma_mc_se < 0).any() or
            (frame.signal_bias_sigma_mc95_low > frame.signal_bias_sigma).any() or
            (frame.signal_bias_sigma_mc95_high < frame.signal_bias_sigma).any()):
        raise ValueError("Invalid bias MC uncertainty intervals")
    exclusion_figure(frame, args.output_dir)
    bias_figure(frame, args.output_dir)
    provenance=dict(input=str(args.input), source_sha256=hashlib.sha256(payload).hexdigest(),
                          status=description(frame), validation_cells=len(frame),
                          outputs=[str(args.output_dir/(name+extension)) for name in
                                   ("validation_exclusion", "bias_by_truth") for extension in (".pdf", ".png")])
    provenance['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance['output_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in provenance['outputs']}
    (args.output_dir/'validation_plot_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))


if __name__ == "__main__":
    main()
