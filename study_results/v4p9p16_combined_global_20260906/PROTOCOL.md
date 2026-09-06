# v4.9.16 combined search protocol

Declared 6 September 2026 before new combined toy results. This derivative
preserves v4.9.12 through v4.9.15. The requested note version is v4.9.16.

## Search and likelihood

Use the 232 integer masses from 19 through 250 MeV. Membership is fixed by
the existing dataset search ranges, never by the observed result:

| Mass [MeV] | Active datasets |
|---|---|
| 19–38 | 2015 full |
| 39–49 | 2015 full, 2016 full |
| 50–90 | 2015 full, 2016 full, 2021 native 10% |
| 91–180 | 2016 full, 2021 native 10% |
| 181–250 | 2021 native 10% |

At each mass use the actual count-scale likelihood with one shared signal
coupling and independent dataset background nuisance blocks. Retain the
unconstrained auxiliary common signal fit for the signed likelihood root.
The discovery alternative and upper-limit denominator enforce nonnegative
signal strength. Do not add individual Z values or multiply p-values.
Kernel states, masks, templates, support and conversion factors remain fixed;
each full-spectrum toy retrains log-count predictions and count-dependent
errors. Cross-dataset systematic correlations are not newly introduced.

## Coherent experiments and Gaussian field

Reuse the ten pilot and 1,000 separate validation full spectra per dataset
from the frozen v4.9.14/v4.9.15 collections. Pair equal indices from their
distinct dataset-specific RNG streams. These are 1,000 joint experiments,
not 3,000 experiments. Each dataset supplies the same complete spectrum
at every hypothesis. Pilot spectra do not enter validation.

The generating mean is the same archived common stress spectrum used in
those studies. This specifies one joint scenario, with independent Poisson
bins and datasets. It does not certify the physical background or calibrate
the v4.9.13 envelope over different truths.

Embed every response in a common 1,626-bin basis: 484 bins for 2015, 720 for
2016 and 422 for 2021. The Asimov ensemble consists of the common baseline
and one positive square-root-of-mean bin perturbation per bin. Inactive
dataset response rows are exactly zero. Their unchanged baseline evaluations
need not be refitted. Reuse individual likelihood-root and response columns
only in the two single-dataset intervals. Refit all 142 multi-dataset masses.

Use a=r(B), D_i=r(B+sqrt(B_i)e_i)-a, C=D.T D, s=sqrt(diag C),
K=C/(s s.T). Sample 200,000 fields independently per method, preserving
the nonzero offset and nonunit width. Apply the same existing principal
ordering: p=sf((r-a)/s) for raw r>0, otherwise p=1; the scan score is the
largest standardized root among positive raw roots. Keep the separate raw
nonnegative-root maximum as a different test. Generate each full 232-point
field at once. Do not join independently sampled segment maxima or use
smooth-kernel upcrossing formulae across membership boundaries.

Validate means, widths, correlations and both maximum distributions with the
1,000 joint Poisson experiments. Report all normality and KS diagnostics,
with their scope/multiplicity. Do not fit the covariance or centering to
validation outcomes. Zero tails receive one-sided 95% Monte Carlo bounds;
pointwise central 95% binomial intervals are also retained. Neither interval
includes background-model or Gaussian-approximation uncertainty.

## Observed upper limit and requested figure

The upper-limit curve is the pointwise 90% bounded, piecewise-asymptotic CLs
result from the same Gaussian-profiled count-scale likelihood. Fresh dense
observed fits accompany multi-dataset toy evaluation; single-dataset
intervals reuse v4.9.13 dense asymptotic endpoints. The GP calculation
calibrates search probabilities and does not alter or provide simultaneous
coverage for these confidence limits.

Apply the established visible-to-electron branching correction once above
the dimuon threshold, keeping raw and displayed epsilon-squared columns.
Compare against the corresponding v4.9.12 active-scope curve in raw units.
Do not splice calibrated all-three limits into asymptotic pairwise intervals.
Legacy expected bands use a different generating ensemble and are omitted.

The main figure places the complete limit above representative local and
scan-global p-value curves on the same mass axis, with membership indicated.
Representative numerical rows are the two profiled extrema (principal and
raw orderings) plus 30, 65, 120 and 220 MeV. They are descriptive selections;
all global probabilities include the full declared grid.

## Numerical gates and resource bounds

Run one low-priority fitting process with one BLAS thread. Preserve per-mass
checkpoints, exact references, failures and source hashes. Repeated identical
dataset predictions may be memoized by predictor identity, numerical mode
and exact training-count bytes; this changes no numerical operation.

At every new mass retain the parent's low-rank proposal checks and exact
fallback policy. Compare all ten joint pilot roots to the exact backend.
Require root differences below 1e-3 and zero changes in the raw-positive
classification. Compare an exact Asimov baseline and a fixed response
stencil (16 uniform full-support bin indices per active dataset, plus each
blind-window first, middle and last bin).

Use six complete exact joint-response sentinels: 39, 49, 50, 90, 91 and
180 MeV. Reused exact/previously audited single-member columns at 38 and
181 provide the outer boundary partners. Require baseline error/width
below 1e-3, response-entry error below 1e-4 both absolutely and divided
by width, full response relative L2 and width errors below 1e-3, and the
embedded sentinel correlation difference below 1e-3. Failed approximation
gates select the exact backend for the entire coordinate and all ensembles.
Failed exact reference fits halt; no toy or hypothesis is removed.

Retain scalar/batch root tolerance 2e-5, bounded-q tolerance 1e-4 and
optimizer score below 2e-7. Reproduce v4.9.13 dense observed roots within
2e-5 and relative endpoints within 1e-8 where matching scopes exist.
The older v4.9.12 optimizer is a diagnostic comparator, not an accuracy
oracle: relative upper-limit differences above 3% or bounded-root
differences above 0.15 trigger a dense-fit investigation, without choosing
the more favorable endpoint.

All 2016 source-fit, development-overlap, transition and inherited numerical
qualifications remain attached to the combined result. This study does not
establish discovery calibration, physical background closure, expected
sensitivity, unconditional coverage, or a continuous-mass search.
