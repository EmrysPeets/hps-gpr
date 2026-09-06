# Auxiliary validation of the known 71 MeV reverse-truth failure

This single diagnostic asks whether the final two-truth calibration rejects
the original 71 MeV injections too frequently when the generating background
is the archived reverse-injection truth. That truth remains outside the
calibration envelope. This retrospective check does not replace the main
validation, add a calibration truth, establish global/unconditional coverage,
or erase the previously reported failure. The coordinate, seeds, counts,
thresholds and numerical gates below are fixed before execution.

## Execution and identity

Do not run numerical work until the current first-pass process has exited and
the coordinator has reviewed this source. Run one process with one BLAS thread.
The driver requires all 456 first-pass coordinates to be complete, their exact
47-entry source contract, the companion GP-runtime hashes, and a passed
numerical audit of an explicitly selected final 2021 10%, 71 MeV checkpoint.
It accepts the original checkpoint or the declared sampling-refinement
derivative, checking its original hash map, extra hashes, plan identities,
per-truth counts and original-checkpoint link. No coordinate or attempt is
chosen from this diagnostic's results.

Example, from the repository root after first-pass completion:

```sh
python3 -B study_results/v4p9p13_calibration_20260905/validate_reverse_truth_71.py \
  --checkpoint study_results/v4p9p13_calibration_20260905/derived/individual_2021_10pct/m071/result.json
```

`--preflight-only` checks files and freezes the auxiliary contract without
importing the fitting runtime or generating toys. Outputs are isolated under
`reverse_truth_71/checkpoint_<selected SHA>/`; a different contract requires a
new output directory. Any recorded failure also requires a new `--output`
directory; rerunning cannot overwrite its evidence. The 4 GiB conservative array guard is explicit and may
be set with `--max-memory-gib`; a failure never reduces counts or proposals.
The coordinator continues to enforce the user's weekly availability floor.

## Calibration bank reconstruction

Keep exactly the selected checkpoint's local-GP and archived fSigPowExpQ-anchored stress
calibration truths. Reproduce the first-pass numerical backend, reconstruct
the selected proposal means and labels, and require their hashes to agree.
For a sampling derivative these proposal means were frozen before any later
numerical candidate; the derivative's final inference backend is not used to
redesign them. Reuse the recorded original/refinement seed namespace and
256/512/1024 draws per proposal, checking the complete count-array hash.

After fixing the generating arrays, compute both methods' statistics with the
archived `fit_gpr(..., optimize=False)`, count-dependent alpha, full count-space
GP covariance, unchanged covariance conditioning and full Cholesky factor.
There is no eigenfeature or nuisance-mode truncation in either calibration or
validation statistics. The unchanged Context/Bank likelihood and density
weights are reused. This is a dense numerical reconstruction of the selected
calibration, not a new observed-limit scan or another proposal optimization.

At background-only and both actual positive injected yields, require finite
importance weights, normalization SE at most 0.05, and normalization mean
within max(0.05, 5 SE) of one. Failure preserves bank artifacts and stops before
validation. Save calibration r, q(Atrue), density weights, strata, array hashes,
and fit/normalization diagnostics so later rescoring does not require GP refits.

## Original validation spectra and statistic

Regenerate exactly the three original 500-spectrum `retrained_sidebands`
ensembles using NumPy SeedSequence `[491305,2,71,strength]` for strength 0, 2,
5. Use the released reverse-injection smooth truth from the 66 MeV kernel
fitted outside 60–86 MeV, the original full signal template, and the saved
reference profiled error. Generate each spectrum with the original per-toy
Poisson call. Atrue is the saved physical signal count inside the target
window; target weights use Atrue divided by the current reference error rather
than rounding that ratio to 2 or 5. The two methods remain paired.

Check all 3,000 saved method rows against the original released checkpoints:
absolute Atrue error <=1e-7 counts, Ahat error <=0.05 counts, signed-r error
<=2e-5, and asymptotic CLs error <=2e-5. Every positive-yield asymptotic exclusion
classification must match. Original data/source hashes, bin/mask/window
fraction, full-template normalization and all toy IDs must close. A mismatch
halts and is retained; neither tolerances nor seeds adapt to a failure.

Compute bounded q(Atrue) directly with the Poisson/Gaussian fit, with the null
denominator for a negative unconstrained estimate and q=0 when Ahat>Atrue.
Check the first two toys per positive strength and method against the scalar
reference at the actual Atrue, with absolute q error <=1e-4. Save q and its
Asimov counterpart. No Wald reconstruction is permitted. Use the maximum of the two truth-specific
CLs values and exclude at CLs<0.10. At B0 compare the maximum calibrated local
p0 with 0.05, assigning the physical atom p0=1 when signed r<=0. The 500 spectra
per cell are the original reused ensembles, not additional independent toys.

## Reporting and finite-bank limits

Write the paired toy ledger with exact q, raw/calibrated statistics and
truth-specific tail errors/ESS; original spectra; legacy closure checks;
source/count/statistic hashes; numerical QA; and a small `results_table.csv`.
Report raw/calibrated positive-injection exclusion and B0 local rejection
frequencies with exact 95% binomial intervals conditional on this calibration
bank. These intervals do not include uncertainty from constructing the bank.

Report finite-bank qualifications separately: require both truth-specific
tail ESS values >=100 and finite errors for a ready toy. A decision is MC
resolved only if the envelope of pointwise +/-1.96-SE tail estimates lies
wholly on one side of the decision threshold. These MC bands are diagnostics,
not simultaneous confidence bands or coverage intervals. Limited decisions
remain in the reported point estimates and are explicitly counted; no failed
toy or inconvenient truth is dropped. Preserve raw importance estimates,
including zero or >1 estimates, alongside any p0 display truncation. No
zero tail is displayed as an exact zero p0; overshoots above one are displayed
at one only within three MC SE, otherwise marked unresolved. No validation
counts are pooled across old and reconstructed tables.

The note includes this result only when its builder receives an explicit
`--reverse-truth-dir`; no latest-directory search is permitted. A completed
result must match the 71 MeV checkpoint selected by the stable collected CSV,
pass the source/array/legacy-fit/numerical gates and retain all MC qualifications.
Without that argument the note preserves the old 5-sigma failure and states
that this follow-up has not been evaluated in the note. The note records the
exact auxiliary directory and all consumed input hashes. Publication includes
only that auxiliary result's JSON, CSV/gzip and logs; regenerable NPZ arrays
remain excluded. Native histogram identities remain explicit external inputs.
Logs are hashed after completion by note/release provenance, not by the running
numerical process before its final output has been written.
