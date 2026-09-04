#!/usr/bin/env python3
"""Generate the compact v4.9.12 results table and build its standalone PDF."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOTE = HERE / "note"
LABELS = {
    "individual_2015_full": "2015 full",
    "individual_2016_full": "2016 full",
    "individual_2021_10pct": "2021 10\\%",
    "pair_2015_2016": "2015 + 2016",
    "pair_2015_2021": "2015 + 2021",
    "pair_2016_2021": "2016 + 2021",
    "all_2015_2016_2021": "All three",
}
ORDER = tuple(LABELS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-toys", type=int, required=True)
    return parser.parse_args(argv)


def fmt(value: float) -> str:
    return rf"\num{{{value:.3e}}}"


def fmt_p(value: float) -> str:
    return rf"\num{{{value:.3g}}}"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target_toys = int(args.target_toys)
    summary_path = HERE / "derived" / f"expected_band_summary_{target_toys}toys.csv"
    manifest_path = HERE / "derived" / f"run_manifest_{target_toys}toys.json"
    pvalue_path = HERE / "derived" / f"pvalue_diagnostics_{target_toys}toys.csv"
    total_path = (
        HERE / "derived" / f"final_total_search_window_summary_{target_toys}toys.csv"
    )
    if not all(
        path.is_file()
        for path in (summary_path, manifest_path, pvalue_path, total_path)
    ):
        raise SystemExit(
            "completed band, p-value, total-window, and run-manifest artifacts are required"
        )
    summary = pd.read_csv(summary_path)
    pvalues = pd.read_csv(pvalue_path)
    total = pd.read_csv(total_path)
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(summary.n_toys.astype(int)) != {target_toys}:
        raise RuntimeError("note stage does not match the summary toy count")
    if set(pvalues.n_toys.astype(int)) != {target_toys}:
        raise RuntimeError("note stage does not match the p-value toy count")
    if int(run_manifest.get("stage_toys_per_mass", -1)) != target_toys:
        raise RuntimeError("note stage does not match the run manifest")
    if not total.mass_MeV.astype(int).tolist() == list(range(19, 251)):
        raise RuntimeError("total-search-window table is not the exact 19--250 MeV grid")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Scope & $m$ [MeV] & Observed & $-2\sigma$ & Median & $+2\sigma$ \\",
        r"\midrule",
    ]
    for scope in ORDER:
        frame = summary[summary.scope_key == scope]
        row = frame.loc[frame.expected_median.idxmin()]
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                LABELS[scope],
                int(row.mass_MeV),
                fmt(float(row.eps2_observed)),
                fmt(float(row.expected_q025)),
                fmt(float(row.expected_median)),
                fmt(float(row.expected_q975)),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Observed and conditional expected $\epsilon^2$ limits at "
                r"each scope's minimum expected-median mass. The outer columns are "
                rf"empirical central-95\% endpoints from {target_toys} toys per mass.}}"
            ),
            r"\label{tab:band-minima}",
            r"\end{table}",
        ]
    )
    (NOTE / "generated_summary.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    pvalue_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5.1pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Scope & $m$ [MeV] & $p_{\rm strong}$ & $p_{\rm weak}$ & "
            r"$p_{\rm two}$ & analytic $p_0$ & $Z_{\rm local}$ \\"
        ),
        r"\midrule",
    ]
    for scope in ORDER:
        frame = pvalues[pvalues.scope_key == scope]
        row = frame.loc[frame.p0_local_asymptotic.idxmin()]
        pvalue_lines.append(
            "{} & {} & {} & {} & {} & {} & {:.3f} \\\\".format(
                LABELS[scope],
                int(row.mass_MeV),
                fmt_p(float(row.p_strong)),
                fmt_p(float(row.p_weak)),
                fmt_p(float(row.p_two)),
                fmt_p(float(row.p0_local_asymptotic)),
                float(row.Z_local_asymptotic),
            )
        )
    pvalue_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Fixed-mass diagnostics at each scope's smallest analytic "
                r"local $p_0$. The one-sided empirical fractions have granularity "
                rf"$1/{target_toys}={1.0 / target_toys:.3f}$ and $p_{{\rm two}}$ is "
                r"constructed from them; the selected minimum "
                r"analytic $p_0$ is a compact reporting choice, not a trials-corrected "
                r"global test.}"
            ),
            r"\label{tab:pvalue-summary}",
            r"\end{table}",
        ]
    )
    (NOTE / "generated_pvalue_summary.tex").write_text(
        "\n".join(pvalue_lines) + "\n", encoding="utf-8"
    )

    toy_rows = target_toys * len(summary)
    counter_totals = dict(run_manifest.get("solver_counter_totals", {}))
    bounded_free_retries = int(
        counter_totals.get("bounded_free_centered_retries", 0)
    )
    unbounded_free_retries = int(
        counter_totals.get("unbounded_free_centered_retries", 0)
    )
    null_retries = int(counter_totals.get("null_centered_retries", 0))
    free_retries = bounded_free_retries + unbounded_free_retries
    centered_profile_retries = free_retries + null_retries
    if target_toys == 50:
        stage_title = "Initial 50-toy stage"
        stage_text = (
            "The outer expected band is provisional and will be sharpened by the "
            "planned 100- and 300-toy stages."
        )
        continuation_text = (
            "Re-running the driver with \\texttt{--target-toys 100} appends IDs "
            "50--99; using \\texttt{--target-toys 300} then appends IDs 100--299."
        )
    elif target_toys == 100:
        stage_title = "Cumulative 100-toy stage"
        stage_text = (
            "Toy IDs 0--49 are the bitwise-preserved initial ensemble and IDs "
            "50--99 are the continuation. The planned 300-toy stage will sharpen "
            "the finite-ensemble tails further."
        )
        continuation_text = (
            "Re-running the driver with \\texttt{--target-toys 300} appends only "
            "IDs 100--299 while retaining IDs 0--99 exactly."
        )
    else:
        stage_title = "Cumulative 300-toy stage"
        stage_text = (
            "This is the planned full cumulative ensemble; all earlier toy IDs are "
            "retained exactly."
        )
        continuation_text = (
            "The contracted cumulative schedule is complete at 300 toys per mass."
        )
    values = [
        rf"\newcommand{{\StageToys}}{{{target_toys}}}",
        r"\newcommand{\StageToyMin}{0}",
        rf"\newcommand{{\StageToyMax}}{{{target_toys - 1}}}",
        rf"\newcommand{{\ScopeMassRows}}{{{len(summary)}}}",
        rf"\newcommand{{\ToyLimitRows}}{{{toy_rows:,}}}",
        rf"\newcommand{{\TotalWindowRows}}{{{len(total)}}}",
        rf"\newcommand{{\EmpiricalPResolution}}{{{1.0 / target_toys:.3f}}}",
        rf"\newcommand{{\CenteredProfileRetries}}{{{centered_profile_retries}}}",
        rf"\newcommand{{\FreeProfileRetries}}{{{free_retries}}}",
        rf"\newcommand{{\NullProfileRetries}}{{{null_retries}}}",
        rf"\newcommand{{\StageStatusTitle}}{{{stage_title}}}",
        rf"\newcommand{{\StageStatusText}}{{{stage_text}}}",
        rf"\newcommand{{\ContinuationText}}{{{continuation_text}}}",
        r"\newcommand{\StageDate}{September 4, 2026}",
    ]
    (NOTE / "generated_values.tex").write_text(
        "\n".join(values) + "\n", encoding="utf-8"
    )

    tectonic = shutil.which("tectonic") or "/opt/homebrew/bin/tectonic"
    if not Path(tectonic).is_file():
        raise SystemExit("tectonic is required to build the results-only note")
    build = NOTE / "build" / f"{target_toys}toys"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            tectonic,
            "-C",
            "--keep-logs",
            "--outdir",
            str(build),
            "results_section.tex",
        ],
        cwd=NOTE,
        check=True,
    )
    built = build / "results_section.pdf"
    if not built.is_file():
        raise RuntimeError("tectonic did not produce the expected PDF")
    note_pdf = NOTE / f"HPS_GPR_Analysis_Note_v4p9p12_expected_bands_{target_toys}toys.pdf"
    output_dir = REPO / "output" / "pdf" / "v4p9p12_expected_bands_20260904"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / note_pdf.name
    shutil.copy2(built, note_pdf)
    shutil.copy2(built, output_pdf)
    print(note_pdf)
    print(output_pdf)


if __name__ == "__main__":
    main()
