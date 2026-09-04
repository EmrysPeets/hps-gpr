#!/usr/bin/env python3
"""Generate the compact v4.9.12 results table and build its standalone PDF."""

from __future__ import annotations

import argparse
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target_toys = int(args.target_toys)
    summary_path = HERE / "derived" / f"expected_band_summary_{target_toys}toys.csv"
    manifest_path = HERE / "derived" / f"run_manifest_{target_toys}toys.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise SystemExit("completed band summary and run manifest are required")
    summary = pd.read_csv(summary_path)
    if set(summary.n_toys.astype(int)) != {target_toys}:
        raise RuntimeError("note stage does not match the summary toy count")

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

    toy_rows = target_toys * len(summary)
    values = [
        rf"\newcommand{{\StageToys}}{{{target_toys}}}",
        r"\newcommand{\StageToyMin}{0}",
        rf"\newcommand{{\StageToyMax}}{{{target_toys - 1}}}",
        rf"\newcommand{{\ScopeMassRows}}{{{len(summary)}}}",
        rf"\newcommand{{\ToyLimitRows}}{{{toy_rows:,}}}",
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
