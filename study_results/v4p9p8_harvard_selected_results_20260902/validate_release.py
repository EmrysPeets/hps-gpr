#!/usr/bin/env python3
"""Fail-closed validation for the v4.9.8 selected-results release."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
SOURCE = HERE / "source_tables"
FIGURES = HERE / "figures"
QA = HERE / "qa"
PDF = HERE / "pdf" / "HPS_GPR_Harvard_Writing_Sample_Selected_Results.pdf"
LOG = QA / "build_selected_results" / "writing_sample_selected_results.log"
REVIEW = QA / "visual_review.json"
REPORT = QA / "release_validation.json"

EXPECTED_COLUMNS = {
    "group",
    "scope_key",
    "scope_label",
    "dataset_set",
    "source_state",
    "source_file",
    "mass_GeV",
    "mass_MeV",
    "A90_events",
    "eps2_90",
    "p0_local_asymptotic",
    "Z_local_asymptotic",
    "gp_support",
    "edge_diagnostic",
    "yield_coordinate",
    "limit_method",
    "pvalue_method",
    "limit_coordinate",
}

EXPECTED_SCOPES = {
    "individual_2015_full": (72, 19, 90, "historical_v4p2", "2015"),
    "individual_2016_10pct": (142, 39, 180, "reviewed_v4p1", "2016"),
    "individual_2016_full": (142, 39, 180, "historical_v4p2", "2016"),
    "individual_2021_1pct": (201, 50, 250, "reviewed_v4_support040", "2021"),
    "individual_2021_10pct": (201, 50, 250, "v4p9p5_support036", "2021"),
    "pair_2015_2021": (41, 50, 90, "historical_v4p2", "2015+2021"),
    "pair_2016_2021": (131, 50, 180, "historical_v4p2", "2016+2021"),
    "pair_2015_2016": (52, 39, 90, "historical_v4p2", "2015+2016"),
    "all_2015_2016_2021": (41, 50, 90, "historical_v4p2", "2015+2016+2021"),
}

EXPECTED_MINIMA = {
    "individual_2015_full": (51, 8.462825907417903e-4, 3.1394651973521204),
    "individual_2016_10pct": (65, 1.86286434733427e-2, 2.0829334844892258),
    "individual_2016_full": (90, 3.087814073843e-4, 3.4237813777746164),
    "individual_2021_10pct": (78, 2.4748453213005e-3, 2.810289817861899),
    "pair_2015_2016": (90, 1.265672327023e-4, 3.6590677024823215),
    "pair_2015_2021": (65, 1.6080354508547038e-6, 4.656514851507021),
    "pair_2016_2021": (66, 1.226171212908e-4, 3.6671861782648594),
    "all_2015_2016_2021": (65, 3.259182521304132e-5, 3.9932141411659794),
}

EXPECTED_HASHES = {
    "source_tables/v4_2021_1pct_observed.csv": "a3b3c4feac9a0ce8be07329514b3696bd7e8505a1cd874bf660f6afeebe7475f",
    "source_tables/v4p1_2016_10pct_observed.csv": "d3fa2848a868299d303c34d1eb241a119ed93328601496a3eac3534711c60057",
    "source_tables/v4p2_all_period_source.csv": "8f4b37ff6a998e236c1ea959db56a76f21ce509c05f24c17675cef676fcbeadd",
    "source_tables/v4p2_individual_observed.csv": "1e3e99fb7c0a171d6d496de87ac6664b485928042b2cede242dffab55e0cc410",
    "source_tables/v4p2_m065_fit_summary.csv": "dc06707637511644e6bad06638451351a9995b9e363b4cfe0aeddcae18bf3c4f",
    "source_tables/v4p2_m065_plot_data.csv": "fbb4c6a00435799d8868326067a4fb5b3187cf16644a6e690af4a44520b1840f",
    "source_tables/v4p2_standalone_pairwise_source.csv": "efa73576adae356d4805b7548a0bc14da4d4a2572fd6a19cc6404e7cd5386e47",
    "source_tables/v4p9p5_2021_10pct_observed.csv": "28e6a10b8633fc69c1bab62d32fe39417c42ac886ef27f74ca0c9aeb7cc620e9",
    "figures/historical_all_three_m065_extraction.pdf": "60860b129ce5f1ee2190d911945cf50e360e179c39fb68cca0640621eab7ae09",
    "figures/historical_all_three_m065_extraction.png": "e239689fc1a177547a7caa178764e454112a8ba854659865805219df8f6312e2",
}


checks = []


def check(name, condition, detail=""):
    checks.append({"name": name, "passed": bool(condition), "detail": str(detail)})


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_vectors(name, actual, expected, rtol=1e-12, atol=1e-15):
    ok = len(actual) == len(expected) and np.allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=rtol, atol=atol
    )
    check(name, ok, f"actual={len(actual)} expected={len(expected)}")


def reconstruct_combined_a90(curves):
    individual = pd.read_csv(SOURCE / "v4p2_individual_observed.csv")
    k_map = {
        (str(row.dataset), int(round(float(row.mass_MeV)))): float(row.A_up) / float(row.eps2_up)
        for row in individual.itertuples(index=False)
    }
    selected = curves[curves["group"] == "combination"]
    expected = []
    for row in selected.itertuples(index=False):
        total_k = sum(k_map[(key, int(round(float(row.mass_MeV))))] for key in str(row.dataset_set).split("+"))
        expected.append(float(row.eps2_90) * total_k)
    compare_vectors("combination total-yield reconstruction", selected["A90_events"], expected)


def validate_sources(curves):
    hist = pd.read_csv(SOURCE / "v4p2_individual_observed.csv")
    specs = [
        ("individual_2015_full", hist[hist["dataset"].astype(str) == "2015"], "A_up", "eps2_up"),
        ("individual_2016_full", hist[hist["dataset"].astype(str) == "2016"], "A_up", "eps2_up"),
        ("individual_2016_10pct", pd.read_csv(SOURCE / "v4p1_2016_10pct_observed.csv"), "A_up", "eps2_up"),
        ("individual_2021_1pct", pd.read_csv(SOURCE / "v4_2021_1pct_observed.csv"), "A_up", "eps2_up"),
        ("individual_2021_10pct", pd.read_csv(SOURCE / "v4p9p5_2021_10pct_observed.csv"), "A_up", "eps2_up"),
    ]
    for scope, source, acol, ecol in specs:
        actual = curves[curves["scope_key"] == scope].sort_values("mass_MeV")
        source = source.copy()
        source["mass_MeV_check"] = source["mass_MeV"] if "mass_MeV" in source else 1000 * source["mass_GeV"]
        source = source.sort_values("mass_MeV_check")
        compare_vectors(scope + " source A90", actual["A90_events"], source[acol])
        compare_vectors(scope + " source eps2", actual["eps2_90"], source[ecol])
        compare_vectors(scope + " source p0", actual["p0_local_asymptotic"], source["p0_analytic"])
        compare_vectors(scope + " source Z", actual["Z_local_asymptotic"], source["Z_analytic"])

    pair = pd.read_csv(SOURCE / "v4p2_standalone_pairwise_source.csv")
    for scope in ("pair_2015_2016", "pair_2015_2021", "pair_2016_2021"):
        actual = curves[curves["scope_key"] == scope].sort_values("mass_MeV")
        source = pair[pair["scope_key"] == scope].sort_values("mass_MeV")
        compare_vectors(scope + " source eps2", actual["eps2_90"], source["eps2_obs"])
        compare_vectors(scope + " source p0", actual["p0_local_asymptotic"], source["p0_analytic"])
        compare_vectors(scope + " source Z", actual["Z_local_asymptotic"], source["Z_analytic"])

    all_source = pd.read_csv(SOURCE / "v4p2_all_period_source.csv")
    all_source = all_source[all_source["dataset_set"] == "2015+2016+2021"].sort_values("mass_MeV")
    actual = curves[curves["scope_key"] == "all_2015_2016_2021"].sort_values("mass_MeV")
    compare_vectors("all-three source eps2", actual["eps2_90"], all_source["eps2_obs"])
    compare_vectors("all-three source p0", actual["p0_local_asymptotic"], all_source["p0_analytic"])
    compare_vectors("all-three source Z", actual["Z_local_asymptotic"], all_source["Z_analytic"])
    reconstruct_combined_a90(curves)


def main():
    QA.mkdir(parents=True, exist_ok=True)
    curve_path = DERIVED / "selected_result_curves.csv"
    check("curve ledger exists", curve_path.is_file(), curve_path)
    if not curve_path.is_file():
        REPORT.write_text(json.dumps({"passed": False, "checks": checks}, indent=2) + "\n")
        return 1

    curves = pd.read_csv(curve_path)
    check("exact released columns", set(curves.columns) == EXPECTED_COLUMNS, sorted(curves.columns))
    forbidden = ("band", "expected", "quantile", "q02", "q16", "q50", "q84", "q97", "lo1", "lo2", "hi1", "hi2", "global_p")
    bad_columns = [column for column in curves.columns if any(token in column.lower() for token in forbidden)]
    check("no band or global-p columns", not bad_columns, bad_columns)
    check("total curve rows", len(curves) == 1023, len(curves))
    check("individual rows", int((curves["group"] == "individual").sum()) == 758)
    check("combination rows", int((curves["group"] == "combination").sum()) == 265)
    check("unique mass keys", not curves.duplicated(["scope_key", "mass_MeV"]).any())

    for scope, (count, low, high, state, dataset_set) in EXPECTED_SCOPES.items():
        frame = curves[curves["scope_key"] == scope].sort_values("mass_MeV")
        check(scope + " row count", len(frame) == count, len(frame))
        check(scope + " exact grid", np.array_equal(frame["mass_MeV"].to_numpy(), np.arange(low, high + 1)), [frame["mass_MeV"].min(), frame["mass_MeV"].max()])
        check(scope + " state", set(frame["source_state"]) == {state}, sorted(set(frame["source_state"])))
        check(scope + " dataset set", set(frame["dataset_set"].astype(str)) == {dataset_set}, sorted(set(frame["dataset_set"].astype(str))))

    all_three = curves[curves["scope_key"] == "all_2015_2016_2021"]
    check("all-three explicit label", set(all_three["scope_label"]) == {"2015 full + 2016 full + 2021 10%"})
    combo_states = set(curves[curves["group"] == "combination"]["source_state"])
    check("all combinations remain historical", combo_states == {"historical_v4p2"}, combo_states)

    numeric = ["mass_GeV", "mass_MeV", "A90_events", "eps2_90", "p0_local_asymptotic", "Z_local_asymptotic"]
    check("all numeric values finite", bool(np.isfinite(curves[numeric].to_numpy(dtype=float)).all()))
    check("positive limits", bool((curves["A90_events"] > 0).all() and (curves["eps2_90"] > 0).all()))
    check("one-sided local p0 range", bool((curves["p0_local_asymptotic"] > 0).all() and (curves["p0_local_asymptotic"] <= 0.5).all()))
    check("nonnegative local Z", bool((curves["Z_local_asymptotic"] >= 0).all()))
    check("mass-unit identity", bool(np.allclose(1000 * curves["mass_GeV"], curves["mass_MeV"], rtol=0, atol=1e-9)))
    check("p0-Z identity", bool(np.allclose(curves["p0_local_asymptotic"], norm.sf(curves["Z_local_asymptotic"]), rtol=2e-11, atol=1e-300)))
    check("limit-method label", set(curves["limit_method"]) == {"observed asymptotic 90% CLs"})
    check("pvalue-method label", set(curves["pvalue_method"]) == {"fixed-mass local asymptotic profile LRT"})

    edge = curves[curves["edge_diagnostic"].astype(str).str.lower() == "true"]
    check("edge flag exact rows", set(zip(edge["scope_key"], edge["mass_MeV"])) == {("individual_2021_1pct", 50.0), ("individual_2021_1pct", 51.0), ("individual_2021_1pct", 52.0)})
    one = curves[curves["scope_key"] == "individual_2021_1pct"]
    formal = one.loc[one["p0_local_asymptotic"].idxmin()]
    interior = one[one["mass_MeV"] >= 53].loc[lambda x: x["p0_local_asymptotic"].idxmin()]
    check("2021 1% formal edge minimum", float(formal["mass_MeV"]) == 50 and np.isclose(formal["p0_local_asymptotic"], 1.4586133496083017e-18))
    check("2021 1% interior minimum", float(interior["mass_MeV"]) == 244 and np.isclose(interior["p0_local_asymptotic"], 0.0026164046820882))

    for scope, (mass, p0, z_value) in EXPECTED_MINIMA.items():
        frame = curves[curves["scope_key"] == scope]
        row = frame.loc[frame["p0_local_asymptotic"].idxmin()]
        check(scope + " frozen minimum", float(row["mass_MeV"]) == mass and np.isclose(row["p0_local_asymptotic"], p0, rtol=1e-12) and np.isclose(row["Z_local_asymptotic"], z_value, rtol=1e-12))

    validate_sources(curves)

    for relative, expected in EXPECTED_HASHES.items():
        path = HERE / relative
        check(relative + " immutable hash", path.is_file() and sha256(path) == expected, sha256(path) if path.is_file() else "missing")

    figure_stems = ["individual_results_triptych", "combined_results_triptych", "asymptotic_pvalue_series", "historical_all_three_m065_extraction"]
    for stem in figure_stems:
        pdf = FIGURES / (stem + ".pdf")
        png = FIGURES / (stem + ".png")
        check(stem + " figure pair", pdf.stat().st_size > 1000 and png.stat().st_size > 1000 if pdf.exists() and png.exists() else False)
        if pdf.exists():
            check(stem + " one-page PDF", len(PdfReader(str(pdf)).pages) == 1)

    check("writing-sample PDF exists", PDF.is_file(), PDF)
    page_count = 0
    pdf_sha = ""
    pdf_text = ""
    if PDF.is_file():
        reader = PdfReader(str(PDF))
        page_count = len(reader.pages)
        pdf_sha = sha256(PDF)
        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        required_text = [
            "Selected observed results",
            "2016 10%",
            "2021 1%",
            "2021 10%",
            "historical v4.2",
            "v4.9.5",
            "local asymptotic",
            "No expected-limit bands",
            "65 MeV",
            "look-elsewhere",
        ]
        for phrase in required_text:
            check("PDF contains " + phrase, phrase.lower() in pdf_text.lower())
        for token in ("??", "TODO", "PLACEHOLDER", "\ufffd", "provenance"):
            check("PDF excludes " + repr(token), token.lower() not in pdf_text.lower())

    check("Tectonic log exists", LOG.is_file(), LOG)
    if LOG.is_file():
        log_text = LOG.read_text(errors="replace")
        fatal_patterns = [
            r"undefined references",
            r"undefined citations",
            r"LaTeX Error: File .* not found",
            r"fatal error",
            r"overfull \\hbox",
            r"overfull \\vbox",
        ]
        for pattern in fatal_patterns:
            check("log excludes " + pattern, re.search(pattern, log_text, flags=re.IGNORECASE) is None)

    check("visual review record exists", REVIEW.is_file(), REVIEW)
    if REVIEW.is_file():
        review = json.loads(REVIEW.read_text())
        check("visual review passed", review.get("status") == "pass", review.get("status"))
        check("visual review matches PDF", review.get("pdf_sha256") == pdf_sha and review.get("page_count") == page_count, review)
        check("visual review covers result pages", bool(review.get("reviewed_result_pages")), review.get("reviewed_result_pages"))

    passed = all(item["passed"] for item in checks)
    payload = {
        "passed": passed,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "pdf_sha256": pdf_sha,
        "pdf_page_count": page_count,
        "checks": checks,
    }
    REPORT.write_text(json.dumps(payload, indent=2) + "\n")
    for item in checks:
        print(("PASS" if item["passed"] else "FAIL") + "  " + item["name"] + ("  " + item["detail"] if item["detail"] else ""))
    print(f"SUMMARY {payload['checks_passed']}/{payload['checks_total']} checks passed")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
