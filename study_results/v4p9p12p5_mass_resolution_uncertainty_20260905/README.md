# Version 4.9.12.5 — 2021 mass-resolution uncertainty study

Reference snapshot, 5 September 2026. This section archives the mass-resolution
**template-width sensitivity scan** and the supporting peak–dip diagnostics.
The 2021 10% observed sample uses the reviewed optimized configuration:
36–300 MeV GP support, length-scale upper factor 15, and nominal ±2.25-sigma
training-exclusion / fitted bins.

## Resolution scan and upper limits

Five widths (0.8, 0.9, 1.0, 1.1, 1.2 times nominal) were evaluated at 201 masses
from 50 to 250 MeV. All **1,005 observed 90% CLs upper limits** pass the numerical
checks. This scan generated **zero new toys**.

- A ±20% width change typically shifts the yield and epsilon-squared limits
  by about ±20%; at 78 MeV the shifts are −31.3% and +32.9%.
- At +20% width, the 78 MeV excess's signed local score changes from +2.810
  to +3.216; the 71 MeV deficit changes from −4.019 to −4.239.
- The largest gains among other separated excess regions are modest:
  182 MeV, +1.570 to +1.894; 93 MeV, +1.515 to +1.762, both at −20% width.
- The nominal limits reproduce the saved production values with maximum
  relative difference 1.28 × 10⁻⁶. Four varied-width fits required documented
  tighter-optimizer retries; no statistical acceptance gate was relaxed.

![Signal-yield upper limits](../v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/resolution_width_scan/figures/upper_limits_signal_yield.png)

Figure 1. Observed 90% CLs upper limits in full-template signal yield for the
five widths; ratios to nominal below. Background predictions and fitted bins
remain fixed at each mass. Zero additional toys.

![Coupling-squared upper limits](../v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/resolution_width_scan/figures/upper_limits_coupling.png)

Figure 2. Corresponding limits in epsilon squared, using the fixed nominal
yield-to-coupling conversion and the recomputed signal fraction in the fitted
bins. These are individual fixed-width curves, not expected bands or a
resolution-nuisance-profiled limit. Zero additional toys.

The [full resolution-study section](../v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/resolution_width_scan/README.md)
contains the signed-score plots, regional comparisons, methods, numerical
retry record, and reproduction commands. Direct data links:
[upper-limit table](../v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/resolution_width_scan/derived/width_scan_upper_limits.csv),
[all fits](../v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/resolution_width_scan/derived/width_scan_all_points.csv),
and [summary / provenance](../v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/resolution_width_scan/derived/summary.json).

## Supporting peak–dip studies

The [20-toy pilot and deterministic reverse-injection section](../v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/README.md)
archives the earlier checks:

- Ten paired background-only / background-plus-66-MeV-signal comparisons:
  exactly 20 cached random spectra, testing whether a positive injection can
  generate neighboring deficits through the moving sideband fit.
- Deterministic injections at 66, 78, and 80 MeV, with zero additional toys,
  testing the reverse response on a common smooth generating background.

These establish conditional analysis responses, not the physical origin of
the observed features. The mass, width, and generating-background choices
are exploratory; the signed scores are not calibrated global significances.
The resolution scan does not refit the training mask, GP kernel bounds, or
density normalization, and it does not constrain a resolution nuisance.

## Reuse and provenance

This publication adds a standalone reference section; it does not rebuild or
replace the full analysis note or the main 300-toy combination study. The
original study filenames are retained for traceability. Numerical tables,
plots, protocols, cached toys, and fitting scripts are copied byte-for-byte.
Publication-only changes add navigation and make the width-study validator
resolve historical absolute paths against the current checkout. Its QA
report is regenerated; the historical numerical provenance is not rewritten.

Only the frozen dependencies required by these studies are included from
the v4.9.12 final-combination archive and v4.9.7 attested runtime. Those
dependency folders are not complete republications of their parent studies.
The included binned CSV input is sufficient for these diagnostics; the old
ROOT paths in the frozen card are not read by the diagnostic runners.

Validate the release from any checkout location, without fitting or generating
toys, using standard Python:

```bash
python3 -B study_results/v4p9p12p5_mass_resolution_uncertainty_20260905/validate_release.py
```

To recompute only the deterministic resolution scan after installing the
repository's scientific Python dependencies:

```bash
nice -n 10 python3 -B study_results/v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/resolution_width_scan/run_width_scan.py
```

That command uses one numerical thread and no toys. Recalculation replaces
derived outputs in that checkout; use a separate checkout to preserve this
reference snapshot. Historical pilot commands generate their original toys
and are not needed to inspect or validate this release.

The [release manifest](release_manifest.json) records version, base commit,
file hashes, and exact-source-copy verification. Use the GitHub permalink to
this section at its publication commit for a stable reference.
