# v5.9.5 study changelog

## Added

- Investigated the apparent null offsets on presentation slides 14 and 50.
- Separated fixed-mass signed roots, positive-part roots, and scan maxima.
- Audited means and widths of the saved 256 Poisson refit scans.
- Generated 200,000 paired Gaussian controls for mean, scale and mass correlations.
- Decomposed residual Q into repeated-sampling variance and prediction-bias terms.
- Replayed 201 masses on all 256 saved spectra: 51,456 exact GP predictions, with frozen, refitted, bias-subtracted and adaptive-covariance controls.
- Reproduced the original slide-14 curve exactly and connected its source residual to the slide-50 signed-root offset through a signal-template projection.
- Added an explicitly exploratory residual-scan maximum diagnostic.
- Recorded the unresolved 2016 width discrepancy near 43.5–44 MeV.
- Produced a standalone PDF, vector figures, CSV/JSON results, portable scripts, independent audit and source hashes.

## Scope retained

Production ±2.25σ exclusion window, original observed roots, original generating source and parent analysis releases are unchanged. No Slides presentation was changed during this study. Suggested wording improvements are in the report, separate from the earlier presentation-edit changelog.
