# Frozen protocol: v4.9.12 pointwise expected-limit bands

This protocol was written before the v4.9.12 expected-band ensemble was
executed.  It supplements, but does not replace, the observed-result protocol
in `study_results/v4p9p12_final_dataset_combinations_20260902/`.

## Inputs and scopes

- Reuse the exact v4.9.12 analysis card, 415 reviewed GP coordinates, numerical
  covariance-conditioning rule, signal normalization, and bounded piecewise
  asymptotic 90% CLs solver.
- Reuse the exact seven observed scopes and their mass grids: the three final
  samples, all three pairs, and the all-three shared-coupling combination.
- Preserve the disclosed 2016 cross-process state-replay exception.  The band
  study does not repair or supersede that exception.

## Pointwise pseudoexperiment construction

At each mass, dataset, and integer toy ID, draw one latent background vector
from the mass-local, frozen-GP multivariate normal.  Use the same effective GP
covariance that enters the likelihood after its attested Cholesky
regularization.  Reject a latent draw containing a negative component up to 80
times; a clip fallback is implemented and recorded, but a released stage must
have zero clip fallbacks.  Poisson-sample the accepted latent vector to obtain
the pseudo-observation.

For a fixed mass and toy ID, reuse each dataset's pseudo-observation in every
scope containing that dataset.  Thus standalone, pairwise, and all-three
comparisons are paired at a mass.  Different masses are generated independently;
the ensemble is pointwise and is not a scan-wide pseudoexperiment ensemble.

The GP hyperparameters and conditional mean/covariance are held fixed.  There
is no toy-by-toy sideband refit in this first band construction.

## Deterministic cumulative stages

The master seed is 491204.  Each pseudo-observation has an independent seed
descriptor `(master seed, mass in MeV, toy ID, dataset index)`.  The stages are
cumulative:

- 50 toys: IDs 0--49;
- 100 toys: add IDs 50--99;
- 300 toys: add IDs 100--299.

Running a later stage must preserve every earlier per-toy limit byte-for-byte.
Per-mass atomic checkpoints allow an interrupted stage to resume without
discarding completed masses.

## Summaries and claim boundary

For each mass and scope, report the empirical 2.5%, 16%, 50%, 84%, and 97.5%
quantiles using NumPy's linear quantile interpolation.  At 50 toys the outer
quantiles are supported by only the extreme few order statistics and are
therefore visibly provisional; 100 and finally 300 toys are the planned
precision stages.

These are conditional pointwise expected-limit bands.  They are not an
unconditional coverage result, a toy-calibrated CLs construction, a global
significance calculation, or a look-elsewhere correction.  The limits inside
each pseudoexperiment use the same piecewise asymptotic CLs mapping as the
observed v4.9.12 curves.
