# v4.9.16 reproduction and continuation

The completed study combines full 2015 and 2016 with native 2021 10% on the
232-point 19–250 MeV grid. Membership follows PROTOCOL.md. The observed
upper limit is pointwise asymptotic CLs; the GP method supplies a separate
conditional global discovery-score calculation. No expected-limit bands
were made.

## Check or reproduce the completed work

Run from the repository root, sequentially with one numerical worker.
Preserve the final directory and its manifest before rebuilding any products.

    python3 -B study_results/v4p9p16_combined_global_20260906/verify_combined.py
    python3 -B study_results/v4p9p16_combined_global_20260906/review/independent_audit.py --require-complete

The runner resumes complete, contracted mass checkpoints. It does not
silently regenerate incompatible spectra or omit failed fits.

    nice -n 10 python3 -B study_results/v4p9p16_combined_global_20260906/run_combined.py
    python3 -B study_results/v4p9p16_combined_global_20260906/analyze_combined.py
    python3 -B study_results/v4p9p16_combined_global_20260906/make_figures.py
    python3 -B study_results/v4p9p16_combined_global_20260906/build_report.py

After rebuilding, render every page, inspect labels and layout, run
validate_products.py, and regenerate the manifest only after independent,
numerical, semantic and visual acceptance. Re-running the analyzer changes
its recorded timing and requires a new report/manifest even with identical
random streams and probabilities.

## Additional direct global-tail precision

The current 1,000 combined experiments pair equal row IDs from three distinct
year-specific source streams. They reuse the earlier 1,000 spectra per year
and are not 3,000 independent joint experiments. The pilot is separate.
Changing output filenames or replaying the backend does not create new toys.

Create a new derivative with declared new cohort IDs and seeds for each
year. Draw a complete spectrum for each dataset, then fit the shared
amplitude at every applicable mass using those same counts. Keep one global
response basis; never splice independently generated local roots or
independently sampled segment maxima. Preserve source, state and generating
mean hashes and the declared dataset membership map.

For probability p, n experiments yield about n*p exceedances. Approximately
100 exceedances give roughly 10% relative counting precision for a small
tail. With zero events the one-sided 95% upper bound is
1 - 0.05**(1/n). In particular, zero of 1,000 supports only a bound near
0.00299 under the specified joint background. It cannot establish a much
smaller Gaussian extrapolation.

Archived full-spectrum importance-sampling toys may be reusable only when
their actual joint proposals and spectrum-level likelihood weights can be
reconstructed. Refit the shared-amplitude likelihood on every complete
retained experiment and assess effective sample size for the joint maximum.
Do not reuse pointwise tail weights or pointwise effective sample sizes as
global quantities.

## More GP draws and a finer scan

More GP fields are inexpensive, but improve only Monte Carlo precision
within the assumed Gaussian field. The analyzer seed fixes a nested stream:
increasing --gp-samples extends rather than independently replicates it.
Record the new derivative and sample count explicitly.

A 0.5 or 0.25 MeV grid needs a declared policy for intermediate GP states,
fresh observed fits and consistent whole-spectrum toy fits. The two 2 MeV
subgrids do not certify convergence to a continuous search. Membership
boundaries are real discontinuities of the defined procedure; do not use a
smooth-kernel upcrossing approximation across them.

## A fully calibrated observed upper-limit curve

The current background-only covariance and tail sample do not calibrate
signal-plus-background CLs tests. v4.9.13 has calibrated standalone and
all-three endpoints, but no pairwise calibrated endpoints. Do not stitch
those to asymptotic pairwise points under one calibrated label.

For a uniform calibrated curve, declare the generating family, common signal
coupling and joint nuisance treatment first. At each mass evaluate the same
bounded q_mu on observed and toy spectra under background and
signal-plus-background hypotheses, retraining with the same fixed-kernel
policy. Start a small paired pilot at each requested strength, then extend
only after numerical/weight closure and measured timing.

If using importance sampling, require a full-spectrum target/proposal
likelihood ratio for every constituent and a correctly normalized joint
mixture. Keep effective sample size and endpoint uncertainty explicit.
Preserve any two-truth envelope policy, including its mixed joint-scenario
scope, rather than silently assuming that all independent background choices
are covered by one all-stress experiment.

The established dimuon correction is a single display/conversion factor
above 2*m_mu; it is not a look-elsewhere correction and must not be applied
twice. A simultaneous confidence construction is a separate objective from
the usual pointwise upper-limit scan.

## Physical qualification

Retain the 2016 development overlap, 75–85 MeV source join, restricted
source-fit waiver and inherited numerical exception. Qualify joint background
alternatives using predeclared predictive controls, not favorable observed
limits or p-values. Include any justified cross-dataset systematics in a
declared model. Compare frozen and reestimated kernel policies on matched
experiments. Expected sensitivity needs a separate signal-injection or
expected-limit study; background-only tail calibration alone does not
establish it.
