# v4.9.16 with an illustrative deficit scan

- [Updated nine-page analysis note](../../output/pdf/v4p9p16_deficit_extension_20260906/HPS_GPR_Analysis_Note_v4p9p16_with_Deficit_Scan.pdf)
- [Standalone deficit figure](figures/combined_deficit_scan.pdf)
- [PNG figure](figures/combined_deficit_scan.png)
- [Numerical curves](analysis/deficit_curves.csv)
- [Independent HEP review](review/HEP_DEFICIT_REVIEW.md)

The new deficit figure is page 3, immediately after the complete observed
upper-limit and excess-probability figure. It shows signed fitted roots,
stress-background offsets, local deficit tails and the two distinct
union-global deficit orderings over 19–250 MeV.

The deepest raw profiled deficit is at 72 MeV
with root -3.490339. The strongest stress-centered
deficit is at 83 MeV: the observed root is -0.676096
and the stress Asimov offset is +7.706823. Its extreme conditional
tail is unresolved by both ensembles. Every simulated raw-depth maximum
exceeds the observed raw-depth maximum. These are background diagnostics,
not particle claims or tests of overall goodness of fit.

This extension reuses 1,000 coherent joint Poisson scans and exactly replays
the same 200,000 Gaussian fields per method. Both positive-maxima vectors
were reproduced bitwise before accepting the negative maxima. No new fits,
Poisson spectra or independent Gaussian realizations were added.

The original v4.9.16 directory and PDF remain unchanged. No upper-limit
endpoint or positive-scan result was modified. The deficit direction was
requested after the excess study; these conditional probabilities do not
adjust for selecting a direction, method or ordering. All original 2016
qualifications remain.

## Reproduction

Run from the repository root. Preserve this final manifest before rebuilding.

    python3 -B study_results/v4p9p16_deficit_extension_20260906/analyze_deficits.py
    python3 -B study_results/v4p9p16_deficit_extension_20260906/make_report.py
    python3 -B study_results/v4p9p16_deficit_extension_20260906/review/independent_audit.py

The final independent audit passed 939 conditions.
Product validation passed 21 checks; all nine rendered
pages were inspected. Rebuilds require fresh rendering, product validation,
visual review and a new manifest. No fitting runner is needed.

PROTOCOL.md defines the mirrored negative-root gate and both orderings.
The parent manifest is bound in provenance/parent.json and rechecked in full.
MANIFEST.csv covers this derivative and its final PDF; it excludes itself
and its own verification record.
