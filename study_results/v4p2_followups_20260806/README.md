# HPS-GPR v4.2 follow-up studies (2026-08-06)

Status: **scientific and artifact validators PASS**

This bundle is grounded in the accepted v4.2 analysis tree at commit
`fb1295680bacdd5edbabff9546ee200e3c68b78a`. It adds targeted diagnostics and
presentation products without replacing the accepted v4.2 card or principal
combined result.

## Contents

- `babar_projection/`: the reviewed v4.2 combined observed 90% limit, the
  published BaBar 2014 visible contour, and a 2021-only
  observed-equivalent 100%-statistics density-response proxy. The bundle
  includes frozen inputs, a 232-row ledger, crossing intervals, provenance,
  a version-neutral Figure 63 overlay, a companion with
  `projected HPS proxy / BaBar` underneath, slide/results PDFs and PNGs, and a
  fail-closed validator.  Values below unity in that companion mark
  numerically stronger projected limits.
- `m065_extraction/`: exact fixed-mass common-binning and profiled-background
  studies at 65 MeV. It documents why exact 0.625 MeV aggregation is
  unavailable from the histogram-only 2015/2016 inputs, supplies exact 0.5 MeV
  and 1.25 MeV refits, and repairs the Figure 62 coefficient display with
  physical-domain nominal profile sets.
- `pseudo65/`: one ROOT file containing the source histogram, GP-mean and
  functional-form conditional central-window replacements, and their
  expectations. Both 201-point observed/asymptotic scans have reviewed
  maximum-LML optimizer ledgers, aligned spectrum/limit/local-p0 figures, and
  no expected bands. Additive substudies diagnose the 61--63 MeV
  functional-form shoulder with deterministic central means and compare ten
  fixed-GP-mean draws for the exactly equivalent 2.25/2.5-sigma replacement
  geometry with ten paired 3-sigma replacements.

## Interpretation boundaries

- The BaBar projection preserves the fluctuations of the current 2021 10%
  observation. It is not expected sensitivity, a future observed result, or a
  full-2021 refit. No bands or p-values are projected.
- The common-binning result is a 65 MeV fixed-mass robustness check, not a full
  rebinned scan or a new global-significance calculation.
- The physical `Delta(-2 ln L)=1` coefficient bars are nominal asymptotic
  profile sets and are not coverage calibrated.
- Each pseudo65 lane is one fixed-seed, background-only conditional
  replacement. The observed spectrum outside `[60,70)` MeV is retained, so
  these are not independent global-null pseudoexperiments, an expected
  sensitivity ensemble, or a coverage study.
- The functional deterministic-mean comparison is a conditional
  truth-model/analysis-model shape diagnostic, not a calibrated bias.
- The ten-draw GP summaries and their 16--84% ribbons are descriptive
  conditional spreads, not expected-limit bands or a scan-wide significance
  calibration. Full optimizer-repeat closure is established over 55--75 MeV;
  the displayed full-range curves remain single-attempt outside that interval.

## Validation

From the repository root:

```bash
python3 study_results/v4p2_followups_20260806/babar_projection/validate_babar_projection.py

python3 - <<'PY'
import json
from pathlib import Path
p = Path("study_results/v4p2_followups_20260806/m065_extraction/validation.json")
assert json.loads(p.read_text())["status"] == "PASS"
print("m065 extraction: PASS")
PY

python3 study_results/v4p2_followups_20260806/pseudo65/validate_study.py --stage final
python3 -m pytest -q \
  study_results/v4p2_followups_20260806/pseudo65/tests/test_pseudo65.py

python3 study_results/v4p2_followups_20260806/pseudo65/functional_mean_shape_bias_20260806/validate.py

python3 study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/validate_ensemble.py \
  --stage final
python3 -m pytest -q \
  study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/tests
```

The individual subdirectory READMEs give exact reproduction commands, source
hashes, numerical summaries, and plot captions.
