# HPS-GPR v4.2 observed-equivalent projection compared with BaBar

This self-contained bundle compares the exact reviewed HPS-GPR v4.2 combined
observed 90% asymptotic CLs limit with the published BaBar 2014 visible-dark-
photon 90% limit. It also shows a deliberately limited observed-equivalent
projection in which the reviewed 2021 10% normalization-window density is
scaled to 100% statistics while the already-full 2015 and 2016 samples remain
fixed.

## Primary artifacts

- `figures/v4p2_babar_observed_equivalent_projection_eps2.pdf`: clean
  single-panel slide/results overlay.
- `figures/v4p2_babar_observed_equivalent_projection_eps2_with_projected_over_babar_ratio.pdf`:
  companion overlay with a lower `projected HPS proxy / BaBar` ratio panel.
  Values below unity identify masses where the proxy is numerically stronger;
  the pale improvement region contains no explanatory text over the curve.
- `figures/v4p2_babar_observed_equivalent_projection_ratio.pdf`: aligned
  HPS/BaBar ratio diagnostic.
- The same figures are supplied as 300-dpi PNG and editable SVG.
- `derived/v4p2_babar_projection_reviewed.csv`: complete 232-row numerical
  ledger.
- `derived/provenance.json`: source hashes, formula, numerical summary,
  statistical boundaries, and output hashes.
- `qa/validation_report.json`: numerical, provenance, PDF, raster, SVG, and
  rendered-page gate.
- `CAPTION.md`: results-section caption text.

## Authoritative HPS input

The plotted current HPS curve is
`eps2_obs_minimal_visible` from:

`study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/derived/combined_bands300_reviewed_v4p2.csv`

Its SHA-256 is
`8f4b37ff6a998e236c1ea959db56a76f21ce509c05f24c17675cef676fcbeadd`.
It contains 232 integer-MeV hypotheses from 19 through 250 MeV. The shared
nonnegative-epsilon-squared likelihood uses full 2015, full 2016, and the
reviewed 2021 10% sample, `combined_mode: count_scale`, and asymptotic
`tilde_q_mu` CLs with `alpha=0.1`.

The plotted minimal-visible conversion is identical to the electron-channel
result below `2m_mu = 211.316749 MeV` and applies the reviewed visible-width
factor once above threshold.

## Projection definition

At every mass, the bundle evaluates

```text
S(m) = sqrt[sum_d rho_d(m) / sum_d f_d rho_d(m)]
eps2_projected(m) = eps2_observed_minimal_visible(m) * S(m)
f = (f_2015, f_2016, f_2021) = (1, 1, 10).
```

Each `rho_d` is the source row's observed counts-per-GeV density in the
physical `m +/- 1.64 sigma_m` normalization window. It is not luminosity or a
GP background expectation. Consequently the scale is exactly one below
50 MeV, density weighted where 2021 overlaps the earlier campaigns, and
exactly `1/sqrt(10)` above 180 MeV where only 2021 is active.

This is an observed-equivalent response proxy. It preserves fluctuations in
the current 10% observation and is not an expected median sensitivity, a
future observed result, or a refit to full-2021 data. No limit bands or
discovery p-values are projected.

## BaBar input

`inputs/BaBar_Lees2014xha.txt` is the frozen raw visible-dark-photon contour
for Lees et al., *Phys. Rev. Lett.* **113**, 201801 (2014),
arXiv:1406.2980. Its SHA-256 is
`5b03037c27f248126830114229300f938d89c1509b47eae0088c55bb0b0a2778`.
The source stores mass in GeV and the observed 90% upper limit on epsilon; the
plotted epsilon-squared contour is its square.

The raw BaBar contour is plotted directly. Only the comparison ledger uses
linear interpolation in log epsilon squared, with no extrapolation. This is
not the 2017 BaBar invisible-dark-photon result, which tests a different
decay hypothesis.

## Numerical results

- Current v4.2 observed minimum: `1.13397029319e-6` at 72 MeV.
- Observed-equivalent projected minimum: `4.51266193513e-7` at 73 MeV.
- Minimum current HPS/BaBar ratio: `1.04339901691` at 98 MeV; the current
  curve is not numerically below BaBar on the HPS grid.
- Minimum projected HPS/BaBar ratio: `0.374661105610` at 98 MeV.
- The projected curve is numerically below BaBar at 55 grid points in nine
  disconnected intervals: 56--62, 69--80, 86--94, 96--101, 108--109,
  113--114, 124--125, 132--140, and 171--176 MeV.

These disconnected intervals inherit the observed HPS fluctuation pattern.
They are numerical crossings of the response proxy, not probabilities of
future reach.

## Reproduction and validation

From the repository root:

```bash
MPLCONFIGDIR=/private/tmp/hps-gpr-v4p2-babar-mpl \
XDG_CACHE_HOME=/private/tmp/hps-gpr-v4p2-babar-cache \
python3 study_results/v4p2_followups_20260806/babar_projection/make_babar_projection.py

python3 study_results/v4p2_followups_20260806/babar_projection/validate_babar_projection.py
```

The generator fails closed if the reviewed v4.2 table, configuration, note,
or frozen BaBar input differs from its declared checksum. The validator
recomputes the density scaling, minimal-visible conversion, numerical
anchors, and crossing intervals; verifies every provenance hash; inspects the
PDF, PNG, and SVG products; and renders all three PDFs back to PNG for layout
QA. The plotted Figure 63 legend uses the version-neutral label `HPS combined
observed`; the v4.2 source identity remains fixed here, in `CAPTION.md`, and
in the provenance ledger.

## Statistical boundary

The v4.2 conditional 300-pseudoexperiment limit bands are intentionally absent
from this comparison. They are fixed-GP, mass-local descriptive quantiles and
are not a 100%-statistics projection or a direct-coverage calibration. The
accepted 2016 factor-12 length-scale ceiling also followed an observed
boundary diagnostic, so the v4.2 post-selection qualification remains.
