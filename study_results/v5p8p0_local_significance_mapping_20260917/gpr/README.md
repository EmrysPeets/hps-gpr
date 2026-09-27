# v5.8.0 GPR response and scan-grid study

This is a frozen-model, conditional diagnostic. It supplies no replacement calibrated discovery significance. It tests the spacing of the tested signal masses; histogram binning, signal resolution, sideband support, and signal windows are unchanged. The 2015 and 2016 histogram bins are 0.25 MeV; the 2021 bins are 0.625 MeV.

## Results

The complete 2016 domain is 39–180 MeV. The primary comparison has 142 integer masses and 283 half-integer masses. Ten additional quarter-step masses within 74–79 MeV are a separate local diagnostic, giving 293 stored coordinates. Of these, 153 were inherited from v5.1.1 and 140 were newly evaluated. Existing coordinates are unchanged to floating-point precision.

Uniform 1 MeV versus 0.5 MeV sampling changes the largest observed signed likelihood root from 3.424751 at 90 MeV to 3.452500 at 90.5 MeV. The RMS deterministic stress offset is 4.967417 versus 4.965644. This grid average is descriptive and is not an exposure-weighted physical norm. Finer sampling resolves the field more closely; it does not repair the stress offset.

For the **ungated centered field**, the conditional Gaussian probability of a maximum exceeding the fixed threshold 3 changes from 7179/100000 = 0.07179 to 7724/100000 = 0.07724. The paired increase is 545/100000 = 0.00545, with a two-sided 95% Clopper–Pearson interval [0.005003, 0.005926]. The same-spectrum direct Poisson comparison is 23/256 versus 25/256. These are discretization diagnostics, not the inherited gated global-discovery ordering.

Within 74–79 MeV, the 0.5 MeV versus 0.25 MeV ungated fixed-threshold-3 tails are 0.00482 and 0.00499. Every observed signed root in this interval is negative; the large stress-centered contrasts therefore do not represent positive fitted signals.

On the uniform inherited integer grids, the RMS stress offsets for 2015, 2016 and 2021 are 0.654741, 4.967417 and 0.807208. Their average centered toy standard deviations are 0.987203, 0.995442 and 0.998673. The much larger 2016 offset is distinct from conditional covariance/bulk-normality performance.

## Definitions and finite-Monte-Carlo reporting

Let r(m) be the signed root of the profiled Poisson/Gaussian likelihood ratio, a(m) its deterministic stress-spectrum value, D the full-spectrum response matrix, C=D^T D, s(m)=sqrt(C_mm), and K=C/(s s^T). The stress-reference contrast is z(m)=[r(m)-a(m)]/s(m). It tests a specified stress reference and must not be relabeled a calibrated physical local significance.

The 100000 paired Gaussian draws W have mean zero and covariance K. A corresponding raw field is a+sW. `ordering_observed_maxima.csv` and `ordering_fixed_thresholds.csv` separately report:

- `ungated_centered`: max W, compared with max observed z;
- `gated_centered`: max W only where a+sW>0, compared with the observed maximum z only over masses with positive r;
- `raw_positive_root`: max(0,max(a+sW)), compared with max(0,max observed r).

A gated draw with no positive raw root has maximum minus infinity. If the observed search has no positive root, its discovery-tail result is assigned p=1 and its gated threshold is left empty. This is explicit in `observed_positive_fit_exists`. The same orderings are applied to the 256 coherent direct Poisson spectra.

For the full scan, the gated stress-reference maximum changes from 9.568109 at 42 MeV to 9.763556 at 42.5 MeV; both have zero Gaussian and direct exceedances. Conversely, all Gaussian and direct stress-null scans exceed the observed raw maximum. These are different orderings under a strongly displaced reference surface; neither result establishes a new calibrated resonance claim.

**Every `gaussian_p` and `direct_p` column is the Monte Carlo fraction k/N, not a resolved p value when k=0.** Status columns explicitly mark zero exceedances as `zero_exceedances_upper_bound_only`. The `*_low` and `*_high` columns are two-sided 95% Clopper–Pearson intervals. The separate `*_one_sided_95_upper` columns are one-sided 95% bounds. With zero exceedances the upper bounds are approximately 2.996e-5 for N=100000 and 0.011635 for N=256. Their two-sided-interval upper endpoints are instead 3.689e-5 and 0.014306. Do not interchange these conventions or invert a zero k/N into infinite significance.

`local_mapping.csv` additionally records the add-one local empirical estimate (k+1)/257 and both gated and ungated Gaussian conditional contrasts. Its raw Gaussian-tail column uses sf(max(0,r)), whose value at a nonpositive root is 0.5; the inherited explicit discovery gate instead returns p=1. These conventions are kept separate.

## Computation and validation

The same 256 archived complete Poisson spectra are reused at every mass within each dataset. The first 128 and last 128 also furnish the declared empirical-standardization/holdout diagnostic. They are not 256 independent toys per mass, and additional mass coordinates do not increase tail resolution. No new Poisson spectra were generated by this sub-study.

Each new coordinate uses the complete deterministic stress spectrum and 720 individual one-bin perturbations by sqrt(expected count). The first 36 new coordinates used the original exact cached-Cholesky implementation. The remaining 104 used an algebraically exact rank-one GP matrix/target update for the single-bin Asimov perturbations, with exact-Cholesky checks at each coordinate and an exact fallback on failure. Poisson-spectrum retraining retains exact cached Cholesky. A batched matrix-product implementation of the unchanged profile Hessian accelerates the same likelihood calculation; scalar reference checks and fallback are retained.

All 721 response roots at 100 and 150 MeV were checked against the frozen reference. Maximum absolute discrepancies were 1.32e-6 and 4.71e-7; response-SD differences were 3.14e-7 and 3.23e-8. The targeted and full passes contain 1116 and 3224 scalar comparison checks, respectively. One worker and one BLAS thread were used. Their recorded runtimes are 421.8 and 390.1 seconds. Validation confirms full-grid completeness, inherited-node invariance, finite response arrays, positive widths, covariance positive semidefiniteness, nested-grid maximum-tail monotonicity, and input hashes.

The archived 2015 extension above 90 MeV retains the 90 MeV kernel coordinates, as in v5.1.1/v5.0.4. Fractional 2016 masses use log-linear interpolation of frozen adjacent integer kernel coordinates. No hyperparameter selection uses favorable observed results.

## Files and reproduction

- `uniform_grid_summary.csv`: primary campaign comparison with uniform grids; excludes the extra quarter-step coordinates.
- `grid_tail_summary.csv`, `paired_grid_changes.csv`: ungated centered-field sampling diagnostics.
- `ordering_observed_maxima.csv`, `ordering_fixed_thresholds.csv`: explicitly distinct gated, ungated and raw orderings.
- `local_mapping.csv`, `anchor_metadata.csv`: pointwise quantities and signal/coupling conversion metadata.
- `field_*/field.npz`: complete response, covariance, observed-root and coherent-validation arrays.
- `gaussian_*_maxima.npz`: the three orderings of the same seeded 100000 Gaussian fields.
- `input_manifest.json`, `validation.json`, `same_node_invariance.json`, `rank_one_validation.json`: provenance and checks.

`python3 summarize_grid.py` rebuilds the summaries and figures using saved arrays plus the pinned `inputs/v511_2016_field.npz`. This saved-array rebuild is portable with NumPy, SciPy, pandas and Matplotlib; it regenerates the same seed-defined Gaussian sample, not a new statistical campaign. Exact fit replay through `run_grid.py` additionally requires the frozen runtime and source inputs listed by absolute path in `input_manifest.json`. `run_grid_compute.py` preserves the exact source executed for the full fit pass. The published `run_grid.py` differs only by empty-check defaults that permit replay of an already-complete checkpoint bank; this does not change any fit or statistic.
