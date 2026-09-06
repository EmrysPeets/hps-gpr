#!/usr/bin/env python3
"""Validate numerical, provenance, and rendered BaBar projection artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUTS = HERE / "inputs"
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"
QA = HERE / "qa"
RENDERS = QA / "pdf_renders"

RAW_BABAR = INPUTS / "BaBar_Lees2014xha.txt"
REVIEWED = DERIVED / "v4p2_babar_projection_reviewed.csv"
INTERVALS = DERIVED / "crossing_intervals.csv"
PROVENANCE = DERIVED / "provenance.json"

EXPECTED_HPS_SHA256 = (
    "8f4b37ff6a998e236c1ea959db56a76f21ce509c05f24c17675cef676fcbeadd"
)
EXPECTED_BABAR_SHA256 = (
    "5b03037c27f248126830114229300f938d89c1509b47eae0088c55bb0b0a2778"
)
EXPECTED_STEMS = (
    "v4p2_babar_observed_equivalent_projection_eps2",
    "v4p2_babar_observed_equivalent_projection_ratio",
    (
        "v4p2_babar_observed_equivalent_projection_eps2_"
        "with_projected_over_babar_ratio"
    ),
)
EXPECTED_PROJECTED_INTERVALS = [
    (56, 62),
    (69, 80),
    (86, 94),
    (96, 101),
    (108, 109),
    (113, 114),
    (124, 125),
    (132, 140),
    (171, 176),
]
EXPECTED_NUMERICAL = {
    "observed_minimum": (72, 1.1339702931939487e-6),
    "projected_minimum": (73, 4.5126619351315103e-7),
    "current_minimum_ratio": (98, 1.0433990169096585),
    "projected_minimum_ratio": (98, 0.37466110560989985),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def minimum(table: pd.DataFrame, column: str) -> tuple[int, float]:
    finite = table[np.isfinite(table[column])]
    row = finite.loc[finite[column].idxmin()]
    return int(row["mass_MeV"]), float(row[column])


def verify_numeric(table: pd.DataFrame, intervals: pd.DataFrame) -> dict[str, Any]:
    require(len(table) == 232, "Reviewed table must have 232 rows")
    require(
        np.array_equal(table["mass_MeV"].to_numpy(int), np.arange(19, 251)),
        "Reviewed grid is not 19--250 MeV in 1 MeV steps",
    )
    forbidden_tokens = ("p0", "eps2_lo", "eps2_med", "eps2_hi", "toy_")
    forbidden = [
        column
        for column in table.columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    require(
        not forbidden,
        f"Projected p-values or limit-band columns are present: {forbidden}",
    )

    density_columns = [
        "density_2015_counts_per_GeV",
        "density_2016_counts_per_GeV",
        "density_2021_10pct_counts_per_GeV",
    ]
    current_density = table[density_columns].fillna(0.0).sum(axis=1)
    projected_density = (
        table["density_2015_counts_per_GeV"].fillna(0.0)
        + table["density_2016_counts_per_GeV"].fillna(0.0)
        + 10.0
        * table["density_2021_10pct_counts_per_GeV"].fillna(0.0)
    )
    scale = np.sqrt(current_density / projected_density)
    require(
        np.allclose(
            scale,
            table["full2021_projection_scale_eps2"],
            rtol=2.0e-13,
            atol=0.0,
        ),
        "Projection scale does not reproduce the declared density formula",
    )
    require(
        np.allclose(
            table[
                "hps_v4p2_projected_full2021_eps2_minimal_visible"
            ],
            table["hps_v4p2_eps2_obs_minimal_visible"] * scale,
            rtol=2.0e-13,
            atol=0.0,
        ),
        "Projected epsilon-squared curve does not close",
    )
    require(
        np.allclose(
            table["hps_v4p2_eps2_obs_minimal_visible"],
            table["hps_v4p2_eps2_obs_ee_channel"] * table["N_eff_BR"],
            rtol=2.0e-13,
            atol=0.0,
        ),
        "Minimal-visible conversion does not close",
    )
    require(
        np.allclose(
            table.loc[
                table["mass_MeV"] < 50,
                "full2021_projection_scale_eps2",
            ],
            1.0,
            rtol=0.0,
            atol=1.0e-14,
        ),
        "Projection changed masses below 50 MeV",
    )
    require(
        np.allclose(
            table.loc[
                table["mass_MeV"] > 180,
                "full2021_projection_scale_eps2",
            ],
            1.0 / math.sqrt(10.0),
            rtol=0.0,
            atol=1.0e-14,
        ),
        "2021-only projection is not 1/sqrt(10)",
    )

    observed_minimum = minimum(
        table, "hps_v4p2_eps2_obs_minimal_visible"
    )
    projected_minimum = minimum(
        table, "hps_v4p2_projected_full2021_eps2_minimal_visible"
    )
    current_ratio = minimum(table, "hps_v4p2_observed_over_babar")
    projected_ratio = minimum(
        table, "hps_v4p2_projected_full2021_over_babar"
    )
    observed = {
        "observed_minimum": observed_minimum,
        "projected_minimum": projected_minimum,
        "current_minimum_ratio": current_ratio,
        "projected_minimum_ratio": projected_ratio,
    }
    for key, expected in EXPECTED_NUMERICAL.items():
        actual = observed[key]
        require(actual[0] == expected[0], f"{key} mass changed: {actual}")
        require(
            np.isclose(actual[1], expected[1], rtol=2.0e-12, atol=0.0),
            f"{key} value changed: {actual}",
        )

    require(
        int(table["hps_v4p2_observed_below_babar_on_grid"].sum()) == 0,
        "Current v4.2 curve unexpectedly crosses below BaBar",
    )
    require(
        int(
            table[
                "hps_v4p2_projected_full2021_below_babar_on_grid"
            ].sum()
        )
        == 55,
        "Unexpected number of projected grid points below BaBar",
    )
    projected_intervals = intervals[
        intervals["hps_curve"]
        == "v4p2_projected_full2021_observed_equivalent"
    ]
    observed_intervals = list(
        zip(
            projected_intervals[
                "first_below_grid_mass_MeV"
            ].astype(int),
            projected_intervals[
                "last_below_grid_mass_MeV"
            ].astype(int),
        )
    )
    require(
        observed_intervals == EXPECTED_PROJECTED_INTERVALS,
        f"Projected crossing intervals changed: {observed_intervals}",
    )
    require(
        not (
            intervals["hps_curve"] == "v4p2_observed"
        ).any(),
        "Current observed curve should have no crossing interval",
    )
    return {
        key: {"mass_MeV": value[0], "value": value[1]}
        for key, value in observed.items()
    }


def verify_provenance(payload: dict[str, Any]) -> None:
    require(payload["schema_version"] == 1, "Unexpected provenance schema")
    require(payload["status"] == "GENERATED", "Generation did not finish")
    require(
        payload["sources"]["hps_v4p2_reviewed_table"]["sha256"]
        == EXPECTED_HPS_SHA256,
        "Provenance has the wrong v4.2 table hash",
    )
    require(
        payload["sources"]["babar_visible2014_raw"]["sha256"]
        == EXPECTED_BABAR_SHA256,
        "Provenance has the wrong BaBar hash",
    )
    require(
        payload["hps_result"]["column"] == "eps2_obs_minimal_visible",
        "Provenance does not identify the minimal-visible result",
    )
    require(
        payload["hps_result"]["combined_mode"] == "count_scale",
        "Provenance does not identify count_scale",
    )
    require(
        payload["projection"]["bands_projected"] is False,
        "Provenance claims projected bands",
    )
    require(
        payload["projection"]["pvalues_projected"] is False,
        "Provenance claims projected p-values",
    )
    ratio_panel = payload["babar_comparison"]["companion_ratio_panel"]
    require(
        ratio_panel["quantity"]
        == "projected HPS observed-equivalent proxy / BaBar 2014",
        "Companion ratio provenance has the wrong direction",
    )
    require(
        ratio_panel["stronger_projected_limit_region"] == "ratio below unity",
        "Companion ratio provenance has the wrong improvement region",
    )
    require(
        ratio_panel["in_axes_explanatory_text"] is False,
        "Companion ratio provenance permits text over data",
    )
    for record in payload["outputs"]:
        path = REPO / record["path"]
        require(path.is_file(), f"Missing provenance output: {path}")
        require(
            sha256(path) == record["sha256"],
            f"Output checksum changed: {path}",
        )


def inspect_image(path: Path, min_width: int, min_height: int) -> dict[str, Any]:
    with Image.open(path) as image:
        require(
            image.width >= min_width and image.height >= min_height,
            f"Image resolution is too small: {path}",
        )
        grayscale = image.convert("L")
        extrema = grayscale.getextrema()
        stdev = float(ImageStat.Stat(grayscale).stddev[0])
        require(
            extrema[0] < 90 and extrema[1] > 240 and stdev > 18,
            f"Image appears blank or low contrast: {path}",
        )
        return {
            "path": str(path.relative_to(HERE)),
            "pixels": [image.width, image.height],
            "grayscale_extrema": list(extrema),
            "grayscale_stdev": stdev,
            "sha256": sha256(path),
        }


def verify_figures() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    RENDERS.mkdir(parents=True, exist_ok=True)
    pdftoppm = shutil.which("pdftoppm")
    require(pdftoppm is not None, "pdftoppm is required for rendered PDF QA")
    figure_records = []
    render_records = []
    for stem in EXPECTED_STEMS:
        pdf = FIGURES / f"{stem}.pdf"
        png = FIGURES / f"{stem}.png"
        svg = FIGURES / f"{stem}.svg"
        for path in (pdf, png, svg):
            require(path.is_file(), f"Missing figure: {path}")

        reader = PdfReader(str(pdf))
        require(len(reader.pages) == 1, f"PDF is not one page: {pdf}")
        page = reader.pages[0]
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        require(width > 550 and height > 300, f"Small PDF canvas: {pdf}")
        text = page.extract_text() or ""
        require("BaBar" in text, f"BaBar label missing from PDF: {pdf}")
        require("HPS" in text, f"HPS label missing from PDF: {pdf}")
        if stem == "v4p2_babar_observed_equivalent_projection_eps2":
            require(
                "v4.2" not in text,
                "The Figure 63 legend still contains the literal 'v4.2'",
            )
            require(
                "HPS combined observed" in text,
                "The version-neutral Figure 63 HPS label is missing",
            )
            require(
                "HPS observed-equivalent proxy" in text,
                "The Figure 63 projection is not explicitly labeled as a proxy",
            )
        if stem.endswith("with_projected_over_babar_ratio"):
            require(
                "Projected HPS proxy" in text and "BaBar 2014" in text,
                "Companion ratio direction is not stated as projected HPS / BaBar",
            )
            require(
                "Ratio > 1" not in text
                and "Ratio < 1" not in text
                and "numerically lower" not in text,
                "Companion ratio panel contains explanatory text over data",
            )

        png_record = inspect_image(png, min_width=2400, min_height=1250)
        root = ET.parse(svg).getroot()
        require(root.tag.endswith("svg"), f"Invalid SVG root: {svg}")
        require(svg.stat().st_size > 20_000, f"SVG is unexpectedly small: {svg}")

        render = RENDERS / f"{stem}.png"
        subprocess.run(
            [
                pdftoppm,
                "-png",
                "-singlefile",
                "-r",
                "180",
                str(pdf),
                str(render.with_suffix("")),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        render_records.append(
            inspect_image(render, min_width=1400, min_height=750)
        )
        figure_records.append(
            {
                "stem": stem,
                "pdf": {
                    "path": str(pdf.relative_to(HERE)),
                    "pages": 1,
                    "media_box_points": [width, height],
                    "sha256": sha256(pdf),
                },
                "png": png_record,
                "svg": {
                    "path": str(svg.relative_to(HERE)),
                    "bytes": svg.stat().st_size,
                    "sha256": sha256(svg),
                },
            }
        )
    return figure_records, render_records


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    report_path = QA / "validation_report.json"
    try:
        require(RAW_BABAR.is_file(), "Frozen raw BaBar input is missing")
        require(
            sha256(RAW_BABAR) == EXPECTED_BABAR_SHA256,
            "Frozen raw BaBar input checksum changed",
        )
        require(REVIEWED.is_file(), "Reviewed projection table is missing")
        require(INTERVALS.is_file(), "Crossing-interval table is missing")
        require(PROVENANCE.is_file(), "Provenance JSON is missing")

        table = pd.read_csv(REVIEWED)
        intervals = pd.read_csv(INTERVALS)
        provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        numerical = verify_numeric(table, intervals)
        verify_provenance(provenance)
        figures, renders = verify_figures()

        report = {
            "schema_version": 1,
            "status": "PASS",
            "checks": {
                "frozen_source_hashes": True,
                "reviewed_grid_and_semantics": True,
                "density_projection_formula": True,
                "minimal_visible_conversion": True,
                "no_projected_bands_or_pvalues": True,
                "babar_crossing_anchors": True,
                "provenance_output_hashes": True,
                "pdf_png_svg_artifacts": True,
                "rendered_pdf_pages": True,
                "figure63_version_neutral_legend": True,
                "projected_over_babar_ratio_direction": True,
                "ratio_panel_has_no_in_axes_explanatory_text": True,
            },
            "reviewed_rows": int(len(table)),
            "projected_grid_points_below_babar": int(
                table[
                    "hps_v4p2_projected_full2021_below_babar_on_grid"
                ].sum()
            ),
            "projected_crossing_intervals": EXPECTED_PROJECTED_INTERVALS,
            "numerical_anchors": numerical,
            "figures": figures,
            "pdf_renders": renders,
            "source_sha256": {
                "hps_v4p2": EXPECTED_HPS_SHA256,
                "babar_visible2014": EXPECTED_BABAR_SHA256,
            },
        }
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        report_path.write_text(
            json.dumps(failure, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
