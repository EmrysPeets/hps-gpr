#!/usr/bin/env python3
"""Export reviewed numerical macros, extraction table, and figures to Harvard."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HARVARD = REPO / "study_results/harvard_writing_sample_final_combinations_20260902"
CURVES = HERE / "derived/final_dataset_result_curves.csv"
EXTRACTION = HERE / "derived/all_three_peak_extraction_table.csv"
EXTRACTION_SUMMARY = HERE / "derived/all_three_peak_extraction_summary.json"
PROVENANCE = HERE / "inputs/analysis_input_provenance.json"
RELEASE_ATTESTATION = HERE / "qa/release_attestation.json"
FIGURE_STEMS = (
    "individual_final_results",
    "combined_final_results",
    "final_asymptotic_pvalues",
    "all_three_peak_extraction",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latex_scientific(value: float, digits: int = 3) -> str:
    rendered = f"{float(value):.{digits}e}"
    mantissa, exponent = rendered.split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def main() -> None:
    release = json.loads(RELEASE_ATTESTATION.read_text(encoding="utf-8"))
    if (
        release.get("status")
        != "conditional_release_complete_with_numerical_exception"
        or not release.get("passed")
    ):
        raise RuntimeError("the fail-closed v4.9.12 release attestation is absent")
    released_hashes = dict(release.get("artifact_sha256", {}))
    required_released = {
        "derived/final_dataset_result_curves.csv": CURVES,
        "derived/all_three_peak_extraction_table.csv": EXTRACTION,
        "derived/all_three_peak_extraction_summary.json": EXTRACTION_SUMMARY,
        "inputs/analysis_input_provenance.json": PROVENANCE,
    }
    required_released.update(
        {
            f"figures/{stem}{suffix}": HERE / "figures" / f"{stem}{suffix}"
            for stem in FIGURE_STEMS
            for suffix in (".pdf", ".png")
        }
    )
    for relative, path in required_released.items():
        if released_hashes.get(relative) != sha256(path):
            raise RuntimeError(f"release-attested artifact drift: {relative}")
    curves = pd.read_csv(CURVES)
    extraction = pd.read_csv(EXTRACTION)
    extraction_summary = json.loads(EXTRACTION_SUMMARY.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    triple = curves[curves.scope_key == "all_2015_2016_2021"]
    peak = triple.loc[triple.p0_local_asymptotic.idxmin()]
    if int(peak.mass_MeV) != int(extraction_summary["selection"]["mass_MeV"]):
        raise RuntimeError("extraction mass does not match the all-three p0 minimum")
    if len(extraction) != 3 or set(extraction.dataset.astype(str)) != {"2015", "2016", "2021"}:
        raise RuntimeError("extraction table does not contain exactly three campaigns")

    support = provenance["selected_support_2016_MeV"]
    shared = extraction_summary["shared_fit"]
    macros = "\n".join(
        [
            "% Generated from the frozen v4.9.12 result ledger; do not edit by hand.",
            rf"\newcommand{{\FinalSixteenSupportLowMeV}}{{{int(support[0])}}}",
            rf"\newcommand{{\FinalSixteenSupportHighMeV}}{{{int(support[1])}}}",
            rf"\newcommand{{\FinalSixteenLengthScaleUpperFactor}}{{{float(provenance['selected_ls_upper_factor_2016']):g}}}",
            rf"\newcommand{{\SelectedAllThreeMinMass}}{{{int(peak.mass_MeV)}}}",
            rf"\newcommand{{\SelectedAllThreeMinP}}{{{latex_scientific(float(peak.p0_local_asymptotic))}}}",
            rf"\newcommand{{\SelectedAllThreeMinZ}}{{{float(peak.Z_local_asymptotic):.3f}}}",
            rf"\newcommand{{\SelectedSharedEpsTwoHat}}{{{latex_scientific(float(shared['eps2_hat']))}}}",
            rf"\newcommand{{\SelectedSharedEpsTwoSigma}}{{{latex_scientific(float(shared['sigma_eps2']))}}}",
            "",
        ]
    )

    table_lines = [
        "% Generated from all_three_peak_extraction_table.csv; do not edit by hand.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        "  \\begin{tabular}{@{}lrrr@{}}",
        "  \\toprule",
        "  Data set & Shared full yield & Shared window yield & Independent signed window yield \\\\",
        "  \\midrule",
    ]
    for row in extraction.sort_values("dataset").itertuples(index=False):
        table_lines.append(
            "  "
            + str(row.dataset)
            + " & "
            + rf"${row.shared_full_template_yield:.2f} \pm {row.shared_full_template_sigma:.2f}$"
            + " & "
            + rf"${row.shared_fitted_window_yield:.2f} \pm {row.shared_fitted_window_sigma:.2f}$"
            + " & "
            + rf"${row.independent_signed_fitted_window_yield:.2f} \pm {row.independent_signed_fitted_window_sigma:.2f}$ \\\\" 
        )
    table_lines.extend(
        [
            "  \\bottomrule",
            "  \\end{tabular}",
            "  \\caption{Signal-yield decomposition at the all-three local-$p_0$ minimum. "
            "The shared columns use one common $\\epsilon^2$. The signed campaign-level "
            "fits are post-selection concordance diagnostics, not independent measurements.}",
            "  \\label{tab:all-three-peak-extraction}",
            "\\end{table}",
            "",
        ]
    )
    table_tex = "\n".join(table_lines)

    derived = HARVARD / "derived"
    figures = HARVARD / "figures"
    derived.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    macro_path = derived / "generated_selected_results.tex"
    table_path = derived / "all_three_peak_extraction_table.tex"
    macro_path.write_text(macros, encoding="utf-8")
    table_path.write_text(table_tex, encoding="utf-8")
    copied = []
    for stem in FIGURE_STEMS:
        for suffix in (".pdf", ".png"):
            source = HERE / "figures" / f"{stem}{suffix}"
            if not source.is_file():
                raise RuntimeError(f"missing reviewed figure: {source}")
            destination = figures / source.name
            shutil.copy2(source, destination)
            copied.append(
                {
                    "path": str(destination.relative_to(HARVARD)),
                    "sha256": sha256(destination),
                    "source": str(source),
                    "source_sha256": sha256(source),
                }
            )
    manifest = {
        "status": "exported_from_conditional_release_with_numerical_exception",
        "release_attestation_sha256": sha256(RELEASE_ATTESTATION),
        "source_release": str(HERE),
        "source_curves_sha256": sha256(CURVES),
        "source_extraction_sha256": sha256(EXTRACTION),
        "support_provenance_sha256": sha256(PROVENANCE),
        "generated_macros_sha256": sha256(macro_path),
        "generated_extraction_table_sha256": sha256(table_path),
        "copied_figures": copied,
    }
    (derived / "selected_results_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
