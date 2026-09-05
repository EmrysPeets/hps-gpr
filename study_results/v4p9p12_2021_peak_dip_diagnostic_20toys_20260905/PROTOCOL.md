# 2021 peak–dip mechanism pilot

This separate, post-selection diagnostic does not modify the analysis note,
production studies, or their inputs. It uses the exact v4.9.12 2021 rebinned
observed spectrum, 36–300 MeV support, and reviewed factor-15 kernel states.

Freeze before execution:

- Scan integer mass hypotheses 60–80 MeV with the production ±2.25σ training
  exclusion and signal-fit window.
- Hold the reviewed kernel hyperparameters fixed at each mass, but refit the
  GP posterior mean and covariance from each entire pseudo-spectrum. This
  isolates fixed-kernel sideband response, not hyperparameter migration.
- Define one smooth background truth by fitting the observed 2021 spectrum
  outside 60–78 MeV using the reviewed 66 MeV kernel. This data-selected truth
  excludes the focal peak and dip together. It is a conditional mechanism
  screen, not independent background validation.
- Inject the saved independent 2021 best-fit 66 MeV Gaussian signal, using its
  full-template yield. Its size is data-selected, not a sensitivity benchmark.
- Generate exactly 10 background-only Poisson spectra and 10 corresponding
  signal-plus-background spectra: **20 toy spectra total**. Each pair shares
  its background counts, with an independent Poisson signal added. A spectrum
  is reused coherently across all tested masses. Seed: 49126672.
- Also evaluate deterministic mean spectra with and without the injection.
  These are not additional random pseudoexperiments.
- Use the existing profiled Poisson-plus-Gaussian-background likelihood and
  covariance-conditioning rule. Report signed square-root likelihood ratios,
  allowing negative signal-template amplitudes only as deficit diagnostics.
- Check reconstruction against all saved observed 2021 fits from 60–80 MeV
  before generating any toys. Fail on nonfinite values or material fit failure.
- Use one process, one BLAS thread, and low scheduling priority. No optimizer
  study, extra toys, global significance, coverage claim, or production edits.

Plots compare observed background-mask response, paired injection-induced
signed-fit changes, and the joint 66/71 MeV response. Toy bands are descriptive
16th–84th percentiles of only ten spectra per hypothesis. No tail probability
is estimated from this pilot. The negative template coefficient is not a
physical negative squared coupling.

Numerical implementation note, added after observed reconstruction and before
any toy fits: a deterministic near-null fit stopped slightly above the fitted
null likelihood. Apply the production-style known-feasible-null safeguard,
recording its use and the raw likelihood difference for every evaluated fit.
The twenty already-generated spectra are cached and reused; no replacement or
additional toys are generated.
