# Independent implementation and numerical review: v6.2

Status: passed, 23 September 2026. No unresolved scientific implementation
blocker was found in the reviewed extraction and aggregation code. This is a
review of the conditional study and its saved numerical outputs, not a
detector-response or frequentist-coverage certification.

## Scope and implementation

The review read `scripts/injection_core.py`, `scripts/run_study.py`,
`scripts/aggregate.py`, the pinned v6.1 core and likelihood routines, and the
v5.8.2 background-truth preparation. The two extraction methods retain an
identical direct-MC template with full-selected normalization, change both the
fit and training masks together, and recompute count-dependent GP targets,
noise, prediction, and covariance. MC-only centers remain fixed. The 260 MeV
point uses its own resolution with the explicitly declared 250 MeV kernel
anchor. Clean-sideband and known-background controls use the same fitted toy
counts as the primary fits.

The initially permissive checkpoint reuse was identified during review. The
final code includes a dependency signature and hashes for all three mass
outputs; the final production signatures and output hashes were verified.

## Independent numerical checks

`scripts/validate_independent.py` performs the checks below in one numerical
thread. The complete production validation took about three seconds and is recorded in
`qa/independent_validation.json`.

- All eleven masses, all 1760 nonzero exact-N draws, all 440 background draws,
  and all 12320 extraction rows have the expected dimensions and identities.
  Regenerating every draw from its saved seed exactly reproduces the saved
  integer arrays. All strengths and methods share the intended background,
  and all methods/controls share the intended injected spectrum.
- A separate overlap-integral calculation rebins each original MC histogram
  to the analysis bins. The largest probability difference from the saved CDF
  rebinning is 2.084e-16. Analysis, below-support, above-support, and recorded
  overflow probabilities retain full-selected normalization. Source underflow
  is zero and source overflow bookkeeping closes for all eleven masses.
- Saved actual support/window/training/outside counts and expected window
  probabilities close against the saved draws and independently reconstructed
  masks. Signed pull and profile-ratio identities close for every row.
- The cached GP prediction at 60 and 260 MeV, for both masks and
  30000-event toys 0 and 20, exactly matches the parent GP implementation for both mean
  and factored covariance.
- An independently invoked SciPy BFGS optimizer checks eight free fits and
  eight profiles at the true injection count. Free-yield differences are below
  3.5e-8 fitted standard deviations; free/fixed negative-log-likelihood
  differences are below 7.4e-13. Independently reconstructed Hessian errors
  agree to better than 1e-10 in ratio.
- A separate scalar score-root solution checks eight known-background fits.
  Fitted yields agree within 1.2e-8 standard deviations and their curvature
  errors within 5e-12 in ratio.
- All 308 summary cells and all 44 paired-difference cells were recomputed
  from the checkpoint rows. All 308 deterministic extraction rows exist.
  Deterministic known-background fits recover N with a maximum relative error
  of 2.49e-8 across all nonzero strengths and both methods.
- The maximum saved optimizer score is 1.9991e-7; the minimum fitted Poisson
  mean is 8604.55. Every checked yield uncertainty is finite and positive.

## Mechanism check

The deterministic 30000-event experiments separate baseline offsets from
injection response. Subtracting each method's own N=0 GP fitted yield gives:

| Training used for extraction | Pole-centered recovery | Core-shifted recovery |
| --- | ---: | ---: |
| Background-only sidebands | 0.999965 to 1.000067 | 0.999959 to 1.000067 |
| Injected sidebands | 0.717063 to 0.893818 | 0.781631 to 0.939578 |

These ranges cover the eleven native masses. Counts in the likelihood and the
signal templates are identical between the two control rows at each mass and
method; their GP training inputs differ. The contrast therefore identifies
training feedback from injected tails as the cause of the incomplete
incremental response in this conditional experiment. Separately, nonzero
fitted yields at N=0 explain why raw recovery and incremental response differ.
The subtraction remains a diagnostic; the raw yields and pulls are preserved.

## Extension from 20 to 40 toys

The extra 20 toys use indices 20--39 in the original mass- and strength-keyed
seed scheme. The four-worker local run took 5.6 seconds. The original study is
retained in `history/20_toy_release`; no S3DF computation was used.

The production run was repeated deterministically with 40 toys. For every
mass, the fit rows for the first 20 toys reproduce the original CSV bytes exactly, and the
saved first-20 background and signal arrays are identical. All fixed arrays,
input-file hashes, and deterministic control CSVs are unchanged. The abstract
syntax trees of the source probabilities, extraction class, and mass runner
are also identical after normalizing the runner's default toy-count argument.
The original checkpoint output hashes were verified before comparison. Added
signal and background draws reproduce their new seeds and have no repeated
hashes within or across the two halves. These checks are saved in
`qa/toy_extension.json`.

`results/toy_extension_comparison.csv` records all 308 cells for the original
20, the additional 20, and the combined 40. At 30000 injected candidates, the
combined paired-null-subtracted recovery is 0.715236--0.894518 for mass-centered
windows and 0.780108--0.938222 for shifted windows. The largest change from the
original 20-toy mean is 0.002891 in recovery, or 0.289 percentage points, across
those masses and methods. The additional toys preserve the conclusion about
signal entering the GP training bins. They improve the statistical precision
of this conditional diagnostic without establishing calibrated coverage.

## Interpretation limits

The selected-candidate MC has unvalidated signal-daughter association and
production-selection equivalence. Centers and templates are fixed; their
finite-MC uncertainty is not sampled. Exact-N signal bins are multinomial,
whereas the inherited fit uses independent Poisson counts. Forty toys per
cell give imprecise width and containment estimates. The saved binomial
intervals and conditional labels must remain attached to those results; the
study establishes neither detector-calibrated coverage nor an exclusion or
discovery claim. The report's rendering and presentation require their own QA.
