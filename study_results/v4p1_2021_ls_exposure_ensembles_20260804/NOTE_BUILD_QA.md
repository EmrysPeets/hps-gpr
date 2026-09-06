# v4.1 analysis-note build and QA

Date: 2026-08-04

## Source integration

The following patches were applied in order to the exact dirty sibling note
checkout at
`/Users/emryspeets/Desktop/gp_mods/hps-gpr-analysis-note-v4-20260803`:

1. `v4p1_analysis_note_source_exposure_pilot.patch`
   (`95d53fe411bd0f8242a2e99f40ccd5deff94c155588e58bc5e8d5a211d52f78b`)
2. `v4p1_analysis_note_source_exposure_pilot_review_improvements.patch`
   (`f3a0c55ecd81c04853095e570ea1b0443fbdf6e0286fe44ffee1adcbe5c65c94`)
3. `v4p1_analysis_note_background_scan_results.patch`
   (`22a2c4fbd27a2002362a231bfcc9e0dc7509f8930940510e204a5c60e388800e`)
4. `v4p1_analysis_note_completed_injection_results.patch`
   (`d981f023d7397e1447374a80cb2de2d03a1f002a61e8e7905480f888ace30e5e`)

Every patch passed `git apply --check --whitespace=error-all` immediately
before it was applied. The pre-existing dirty checkout was preserved.

## Build

- Engine: Tectonic 0.15.0
- Output: `HPS_GPR_Analysis_Note_v4p1_20260804.pdf`
- Pages: 126, US letter
- PDF SHA-256:
  `6a8db676acc361d083768c2bce8e95ffc547c66465276babd36fcf0fbf5027f4`
- No undefined citation or reference warning was found.
- No overfull box was found.
- The retained diagnostics are existing underfull-box and `!h`-to-`!ht`
  float-placement warnings; neither clips or overlaps the new material.

## Rendered inspection

Pages 37--38, 71--76, 83, and 86 were rendered to PNG and inspected. These cover:

- the length-scale definition and physics-interpretation boundary;
- the observed 2016 10%-versus-100% and 2021 1%-versus-10% comparison;
- all ten primary-truth factor-20 toy trajectories;
- the native-10% versus 1%-source-times-ten comparison;
- boundary occupancy and adjacent-factor log-marginal-likelihood diagnostics;
- fixed-amplitude response and sensitivity;
- the exact combined observed/asymptotic result framing; and
- the conclusions and promotion boundary.

The figures, captions, line numbers, and surrounding prose are legible, with
no clipping or overlap. Alternate-truth plots are present in the figure
manifest and copied into the note artifact directory; the main narrative
uses the primary lane and states the alternate numerical checks in prose.

## Statistical scope

No expected-limit band was made. The ten-toy study is a screening pilot, not
a coverage qualification. The combined update remains observed/asymptotic;
the analytic Sidak reference is not a scan-toy-calibrated global
significance. `q_mu` outputs and the pinned non-promotable epsilon-squared
conversion are not used for a physics claim.
