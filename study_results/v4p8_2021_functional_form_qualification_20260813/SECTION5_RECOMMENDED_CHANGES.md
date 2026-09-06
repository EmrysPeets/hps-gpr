# Recommended restructuring of Section 5 after the v4.8 generator audit and conditional branch

This is the joint recommendation of the GPR implementation, statistical, and
physics reviews.  It is a change list, not an edit to the user-modified master
note source.

## Central scientific correction

Do not call an analytic source fit the physical 2021 background truth.  It is a
declared generating mean for a conditional pseudoexperiment.  The source
spectrum combines production with trigger, reconstruction, acceptance, and
selection effects.

The requested `fSigPowExpQ` family is rejected as a common source generator.
The archived 1% and native-10% Pearson chi-square/ndf values are 1.572 and 6.167;
both fits place `c1` at its +50 bound and both records have `fit_ok=false`.
Literal five-parameter `fSigPow` and the other archived simple families also fail
the common-source implementation audit.  The existing aggregate
`primary_validation_pass` checks normalized totals and sideband fractions; it
does not override failed minimization or source goodness-of-fit.

An exploratory positive log-Chebyshev order scan found degrees 18, 20, and 24
inside the deliberately broad in-sample engineering window for both sources.
Degree 18 is the lowest such order, but no degree is qualified.  For degree 18,
the current reconnaissance screen gives blocked-validation deviance per held-out bin
18.107 (1%) and 24.045 (native 10%), against a maximum 1.25.  The largest
full-fit versus fake-gap diagonal-Poisson matched-filter shifts are 2.679 and
9.889 standard deviations, against a 0.2 budget.  The reconnaissance fitter also
lacks the required multistart/stationarity, independent-fold initialization,
bin-integration, and full-support smoothness certificate.  Therefore
`selected_degree=null`, no common generator is promoted from that nominal-
qualification branch, and no support or kernel-ceiling result is accepted from
it.  These rejected-candidate findings remain part of the record and must not be
rewritten as though the later conditional-stress choice had qualified them.

Suggested note text:

> In the rejected common-family scan, a common family denotes a common positive
> formula and qualification protocol, with parameters fitted independently to
> the native 1% and native 10% spectra. The requested fSigPowExpQ family is
> rejected: the native-10% fit has
> Pearson chi-square/ndf 6.167 and contacts the declared c1=+50 boundary, while
> the 1% fit has Pearson chi-square/ndf 1.572 and contacts the same boundary.
> Both archived fits report fit_ok=false. No tested low-dimensional common
> analytic family qualifies in both sources.

> Degree 18 is the lowest tested positive log-Chebyshev order satisfying the
> broad in-sample source screen in both spectra, but it fails both predictive
> qualification gates. Its blocked-validation Poisson deviance per held-out bin
> is 18.107 and 24.045 for the 1% and native-10% sources, and its maximum
> resolution-window full-fit versus gap-fit model shifts are 2.679 and 9.889
> diagonal-Poisson standard deviations. No tested degree is fully qualified.
> We therefore promote no common analytic generator from this candidate scan.
> These findings concern reconstructed-spectrum parameterizations, not the underlying continuum
> physics, and they cannot choose the kernel ceiling or production card.

## Final user-directed conditional branch

After the rejected `fSigPowExpQ` and Chebyshev-18 investigations, a separate
user-directed branch was frozen for the narrower purpose of generating smooth,
signal-rigid stress pseudoexperiments.  Its mean is

\[
 \mu(m)=A\,\operatorname{expit}\!\left(\frac{m-m_t}{w}\right)
 (m-m_0)^a
 \exp\!\left[-\left(\frac{m-m_0}{\lambda}\right)^p
 +d_2T_2(u)+d_6T_6(u)\right],
 \qquad u=\frac{2(m-0.040)}{0.260}-1 .
\]

This is a thresholded generalized-gamma core with only the sparse `T2` and
`T6` broad correction modes; it is not a sixth-order polynomial.  The threshold
constants `m0`, `mt`, and `w` and the mode pair were fixed during development.
The final native-1% fit has six continuous coordinates: one normalization
(`A`) and five non-normalization shape coordinates (`a`, `lambda`, `p`, `d2`,
and `d6`).  That wording must retain the effective-complexity caveat: six final
free coordinates does not mean that only six data-informed choices were made
during development.

Functional fidelity is assessed primarily over the 50--250 MeV search region.
The 40--50 and 250--300 MeV intervals are positive, stable GP-training
shoulders and must be displayed, but their residuals do not override the
search-region decision.  On native 0.125-MeV bins in 50--250 MeV, the frozen
1% shape has Pearson/deviance ratios 1.088/1.088.  Applying that shape to native
10% and refitting only normalization gives 2.676/2.676.  These are engineering
averages, not formal Poisson-model acceptance; wider aggregation exposes
coherent residuals that must remain visible.

The shape is learned once from native 1% and then frozen.  Native 10% changes
normalization only; it does not refit the five non-normalization shape
coordinates.  A six-coordinate native-10% comparator gives a slightly smaller
deviance but absorbs
17.3--32.2% of injected signal in the recorded Poisson projection, whereas the
normalization-only application gives 1.3--10.4%.  The comparator is therefore
rejected for generation.  The same sparse family fails the 30-MeV support test,
so this branch is authorized only on 40--300 MeV.

Suggested note text for the conditional branch:

> Following the rejected nominal-candidate scan, we froze a distinct sparse
> threshold stress mean containing only broad T2 and T6 correction modes.  Its
> native-1% fit has one normalization and five non-normalization shape
> coordinates.  The five shape coordinates are held fixed when the mean is
> applied to native 10%; only normalization changes.  In
> the 50--250 MeV search region the native-bin Pearson/deviance ratios are
> 1.088/1.088 for the 1% fit and 2.676/2.676 for the shape-frozen native-10%
> application.  These values are engineering fidelity scores rather than
> evidence of an exact Poisson model.  This declared mean supports only a
> conditional stress test and has no production-card or kernel-ceiling impact.

## Proposed Section 5 structure

### 5.1 Scope, estimands, and vocabulary

Define `declared generating mean`, `conditional stress ensemble`, `background
cluster`, `pull`, `recovery`, `Delta Z`, `coverage`, and `CLs` before any result.
State that source-conditioned diagnostics are not observed-data bias, coverage,
global significance, expected bands, or physical truth.

### 5.2 Frozen estimator and 90% CLs contract

Put the v4.2 production settings in one table: search 50--250 MeV, support
40--300 MeV, `pre_log=true`, `alpha=1/y`, rebin 5, masks 2.25 sigma, 12 restarts,
and 2021 length factors 1.1--15.  State `tilde_q_mu`, asymptotic CLs, and
`cls_alpha=0.10`; make v4.8 fail closed if any field drifts.

Keep `eps2_density_nsigma=1.64`: it is the signal-integration normalization
window, not a confidence level.  Do not mechanically replace 1.64485 by 1.28155.
The legacy per-toy `max(0,Ahat)+1.64485 sigma` expression is not actual generic
90% CLs.  Either run the actual CLs construction or label a Gaussian proxy as a
proxy.  At the deterministic background-only Asimov point, 1.64485 sigma can be
rederived as an approximate median 90% CLs scale because CLb=0.5; that special
case does not promote the per-toy proxy.  Use 90% Student-t and chi-square
intervals for finite-ensemble pull summaries.  They are diagnostic confidence
intervals, not CLs and not a coverage statement.

### 5.3 Source-only generator qualification

First preserve the nominal-candidate audit: formulas, parameter counts,
edge/support scans, source hashes, fit status, Pearson and Poisson deviance,
bound contacts, blocked prediction, and fake-gap projections for
`fSigPowExpQ`, the archived simple families, and the Chebyshev scan.  That audit
documents rejection; it is not superseded by the conditional branch.

Then document the sparse `T2+T6` branch separately.  Show the fixed threshold
constants, the 21-pair mode-development scan, the six native-1% fit coordinates
(one normalization plus five shape coordinates), 24-start reproducibility
audit, search-region and support-shoulder residuals, the shape-frozen native-10%
normalization fit, and the signal-absorption
comparison against the rejected six-coordinate native-10% refit.  The primary
fidelity interval is 50--250 MeV and the only authorized support is 40--300
MeV.  The failed 30-MeV result must remain visible rather than being repaired by
another edge choice.

Native-bin Pearson and deviance are correlated views.  With roughly 1,600 bins
in 50--250 MeV, a ratio near 2.7 is not formal compatibility with an exact
prespecified independent-Poisson model.  It is the user-declared `<3`
engineering tolerance for a smooth stress mean when bin-by-bin agreement is not
required.  Rebin-5 and wider residuals expose coherent discrepancies and must
be plotted, not used to claim a better fit.  Smoothing may aid visualization or
initialization, but a diagonal chi-square on smoothed points is invalid unless
the induced covariance is included.

No GPR pull, recovery, p0, CLs, limit, or length-factor result may enter
source-shape selection or factor/card selection.  Optimizer-branch reproducibility is
instead checked with its declared pull-blind coordinates (LML per training bin,
length scale, kernel constant, fitted uncertainty where applicable, and
covariance validity).  Those coordinates validate numerical reproducibility;
they do not promote a source generator or production card.  The six-coordinate
native-10% comparator is retained only to demonstrate why shape refitting was
rejected.

### 5.4 Frozen toy construction and dependence

Fit one normalization and five non-normalization shape coordinates to native
1% once, then freeze the five shape coordinates.  Set only the normalization
when applying that shape to the observed 40--300 MeV total of native 10%; do
not refit native-10% shape.  The 1%x10 and 1%x100 means are exact exposure
scalings of the frozen 1% mean, and 10%x10 is an exact scaling of the native-10%
mean with the same frozen 1% shape.
Store the shape, normalization targets, source hashes, and bin-integrated means
in the toy manifest.

Within the 1% source, use independent increments for 1x, +9x, and +90x.  Within
native 10%, use 1x and +9x.  The two source families are unpaired.  Reuse toy
index across masses and strengths.  The cache contains indices 0--24, but only
0--19 enter the authoritative v4p8p2 closure and locked length scan.  Indices
20--22 had already been inspected in a superseded one-lane development run and
are not an unopened statistical reserve; indices 23--24 were not consumed by
the authoritative products.  None of 20--24 enters the reported plots,
summaries, tuning, or optimizer decisions.  Thus there are 40 analyzed
independent background clusters, 20 per source family, rather than the number
of extraction rows.

Twenty backgrounds per source are a conditional screening ensemble, not a
coverage sample.  A pull-width estimate has about 16% relative sampling error,
the empirical probability resolution is 5%, and even apparently favorable
cells cannot establish 90% coverage.  No cached index beyond 19 may be pooled
into this result; any continuation requires a separately frozen disclosure and
must acknowledge the prior development inspection of indices 20--22.

### 5.5 Generator-sampling QA (Figure 46 analogue)

Separate source qualification from Poisson sampling.  The source figure should
show the sparse mean against both sources, identify 50--250 MeV as the primary
fidelity region, shade or label the 40--50 and 250--300 MeV training shoulders,
and show native and aggregated residuals without hiding the coherent shoulder
or wide-bin discrepancies.  Place the rejected `fSigPowExpQ` and Chebyshev-18
results in an archival comparison panel or adjacent failure figure, not in the
legend as coequal accepted truths.

Each row should show analytic expectation plus mean, median, and central 68% of
exactly the 20 analyzed backgrounds, followed by
`(toy mean - expectation)/sqrt(expectation/20)`.  The four rows are 1%x10,
native 10%, 1%x100, and 10%x10.  The ribbon is a pointwise count interval, not
an expected-limit band, and it must not include indices 20--24.  Sampling
agreement validates only cached Poisson construction, normalization, and nested
exposure scaling.

### 5.6 Conditional extraction closure (Figure 48 analogue)

Put zero-injection spurious-signal diagnostics first, then nonzero recovery and
leakage.  Plot every accepted pull plus median; raw-first, accepted, and
deterministic analytic-mean results; declared Student-t mean and chi-square width
intervals; and excluded raw states without clipping.  State that strengths and
masses share backgrounds.  Use the 20 analyzed backgrounds only.

For each cell, use its actual accepted count `n` in the two-sided 90% Student-t
interval `mean +/- t(0.95,n-1) s/sqrt(n)` and in the two-sided 90% chi-square
interval for the pull width.  For a complete 20-row cell with observed width
one, the latter is approximately `[0.794, 1.370]`, illustrating the limited
precision.  These are finite-sample
diagnostic intervals, not 90% CLs, coverage, or observed-data bias.  Preserve
negative amplitudes and all raw optimizer attempts; do not clip a pull or add
cached toys because a cell looks unfavorable.

#### Completed 20-background result

Collection accepted 1,599 of 1,600 extraction rows.  The sole exclusion is
native-10%x10 background index 12 at 65 MeV and `z=5`, where five attempts did
not reproduce the injected-refit top branch under the declared pull-blind LML,
length-scale, kernel-constant, fitted-uncertainty, and covariance-validity
reproducibility gate.  The affected cell retains 19 accepted rows and
passes the declared sample-size gate.  This is an isolated optimizer exclusion,
not evidence that the scientific closure pattern is satisfactory.

The zero-injection result is nonuniform.  The two low-exposure lanes have the
largest 65-MeV offsets: native 10% gives mean pull `-1.327` with 90% interval
`[-1.800,-0.854]` and width `1.223`, while 1%x10 gives `-1.305`
`[-1.763,-0.847]` and width `1.184`.  The offsets shrink but do not disappear
at the higher exposures: 1%x100 gives `-0.501` `[-0.966,-0.035]` and
native-10%x10 gives `-0.441` `[-0.909,0.026]`.  Additional zero-signal
intervals excluding zero occur at 120 MeV in native 10% (`-0.362`,
`[-0.628,-0.096]`) and 1%x10 (`-0.714`, `[-1.019,-0.409]`), at 90 MeV in
1%x100 (`+0.462`, `[+0.066,+0.858]`), and at 180 MeV in 1%x100 (`-0.524`,
`[-0.958,-0.090]`).  These intervals are cellwise and descriptive; the grid
shares backgrounds and has not been promoted to a scan-wise test.

The same two width patterns recur across all four correlated injection-strength
strata and have 90% width intervals excluding one: native 10% at 120 MeV is narrow
(`0.681--0.691` across `z=0,1,3,5`; at `z=0`, `0.688` with interval
`[0.546,0.943]`), while 1%x10 at 90 MeV is broad (`1.392--1.401`; at `z=0`,
`1.398` with interval `[1.110,1.916]`).  Across all cells, accepted widths
range from `0.681` to `1.401`.  This is evidence of conditional uncertainty
miscalibration under the stress mean, not a coverage measurement.

Nonzero injections largely preserve the zero-signal offset.  At 65 MeV,
native-10% mean pulls progress from `-1.347` to `-1.402` to `-1.434` for
`z=1,3,5`, with median recoveries `-0.214`, `0.570`, and `0.744`; the 1%x10
values are `-1.324`, `-1.377`, and `-1.402`, with recoveries `-0.321`, `0.533`,
and `0.714`.  At `z=5`, median recovery over the complete grid ranges from
`0.714` to `1.062`, so strong-injection recovery is much closer to unity away
from the persistent low-mass offsets.  Weak `z=1` recovery remains noisy and
background-offset dominated, ranging from `-0.321` to `1.421`; it should not be
summarized as a signal-efficiency measurement.

The deterministic analytic-mean rows support a structural rather than purely
Poisson interpretation of the 65-MeV effect: their zero-signal pulls are
`-1.174` for native 10% and `-1.148` for 1%x10.  They also show that a single
analytic row is not a substitute for an ensemble: other mass/lane differences
between analytic pull and the 20-background mean reach about `0.68` pull units.

A separate major concern is length-scale saturation.  Seventy percent of the
1,599 accepted rows are at the factor-15 upper bound; occupancy is 85--100% in
most lanes at 120--210 MeV and zero or small at 65 MeV.  Simple upper-ceiling
saturation therefore cannot by itself explain the 65-MeV offset, and these
pulls cannot select a larger ceiling.  The pattern instead motivates the
separately pull-blind Figure-136 likelihood, reproducibility, and factor-20-to-25
plateau check.

Recommended decision language:

> The sparse threshold ensemble is operationally complete but does not show
> uniform conditional extraction closure.  Persistent negative 65-MeV offsets,
> additional mass-dependent mean shifts, narrow/broad pull-width cells, and
> widespread factor-15 length-scale saturation remain.  The result is a
> source-conditioned stress diagnosis only.  It neither establishes coverage
> nor supports a production-card or kernel-ceiling change.  No cached index
> beyond 19 is pooled into the reported result.

### 5.7 Hyperparameter and support adequacy (Figure 136 analogue)

Keep this independent of extraction pulls.  Plot all 20 analyzed raw ell/sigma_x
trajectories for the four requested lanes over 50,70,...,250 MeV at each
predeclared factor 15/20/25.  Add bound/near-bound occupancy, same-input nested
LML, branch reproducibility, and a 20-to-25 coordinate/LML plateau check as a
mandatory companion table/figure.  Figure 48 cannot identify the ceiling;
Figure 136 alone cannot
select it either.  Pulls, recovery, p0, CLs, and limits are forbidden inputs to
the optimizer or ceiling comparison, and indices 20--24 remain excluded from
the locked scan.

The length scan remains on 40--300 MeV support at factors 15/20/25.  The failed
30-MeV sparse-family fit is not a support-control candidate for this branch.
Even a visually stable 20-to-25 plateau is conditional evidence under one
source-fitted stress mean; it may motivate a separately predeclared follow-up
but cannot promote the production ceiling or card.

#### Completed optimizer-only length result

The externally locked background-only scan is complete for toys 0--19 at 11
masses and factors 15, 20, and 25.  It contains 2,637 selected trajectories out
of 2,640 intended factor points and 8,004 optimizer-attempt rows.  Every selected
trajectory has a valid fit and covariance, common input geometry and optimizer
seeds across factors, and a reproduced maximum-LML branch.  No lower
length-scale or constant-kernel boundary is occupied.  Indices 20--24 were not
consumed by this locked scan, and the scan produced no amplitude, pull, CLs, p0, limit, or
factor-selection quantity.

The three exclusions all occur in `2021_1pct_x100`, background index 8: factor
15 at 90 and 190 MeV, and factor 25 at 150 MeV.  In each case the maximum-LML
candidate was seen only once after five attempts, so the pull-blind
reproducibility gate rejected it.  Consequently, 878 same-input pairs are
available for 15-to-20 and 879 for 20-to-25.  The concentration in one
background is an optimizer-reproducibility warning, not three independent
physics failures.

The accepted-state gate is reproducibility based, not warning free.  Of the
2,637 selected rows, 2,083 carry at least one optimizer warning, including
1,827 L-BFGS abnormal-line-search warnings and 588 length-upper-bound warnings
(categories overlap).  Every selected row nevertheless reproduces its
maximum-LML branch and passes the declared fit and covariance-validity gates.
The relative minimum covariance eigenvalues are slightly negative,
`-6.04e-4` to `-4.85e-8`, but remain inside the declared `-0.01` tolerance.
These fits should therefore be called tolerance-valid and reproducible under
the frozen gate, not warning-free or strictly positive-semidefinite.

Upper-bound occupancy falls sharply when the ceiling is released:

| Lane | Factor 15 exact / near | Factor 20 exact / near | Factor 25 exact / near |
| --- | ---: | ---: | ---: |
| Native 10% | 168/220 (76.4%) / 177/220 (80.5%) | 2/220 (0.9%) / 3/220 (1.4%) | 0/220 / 0/220 |
| Native 10%x10 | 147/220 (66.8%) / 153/220 (69.5%) | 0/220 / 1/220 (0.5%) | 0/220 / 0/220 |
| 1%x10 | 150/220 (68.2%) / 156/220 (70.9%) | 0/220 / 0/220 | 0/220 / 0/220 |
| 1%x100 | 134/218 (61.5%) / 143/218 (65.6%) | 0/220 / 1/220 (0.5%) | 0/219 / 0/219 |
| All lanes | 599/878 (68.2%) / 629/878 (71.6%) | 2/880 (0.2%) / 5/880 (0.6%) | 0/879 / 0/879 |

These are descriptive optimizer-row fractions, not binomial estimates: each of
the 20 background realizations is reused at 11 correlated mass coordinates.
The factor comparisons are paired on identical inputs wherever both rows pass,
but the rows are not 878 or 879 independent pseudoexperiments.

Here `exact` means `ell/ell_hi >= 0.999`, while `near` is the declared 2%
window.  At factor 15, exact occupancy is zero at 50 MeV, 2/80 at 70 MeV,
48/79 at 90 MeV, 93.8--97.5% at 110--210 MeV, 56/80 at 230 MeV, and 33/80 at
250 MeV.  The only factor-20 exact contacts are native-10% background index 2
at 170 and 210 MeV.  Factor 25 has no exact or near-upper contacts.

This mass dependence is not confined to the turn-on edge.  The frozen sigmoid
factor is approximately 0.507, 0.960, and 0.998 at 50, 70, and 90 MeV.  Factor
15 is nearly inactive at the first two points but strongly binding through the
interior 110--210 MeV region; pressure then relaxes again toward 250 MeV.
Within each source family, the higher-exposure lane has a roughly 4.4--4.6%
shorter median optimized relative scale.  Matched-exposure lane medians differ
by less than about 1%.  Because all lanes use the same frozen native-1% shape,
these are conditional exposure/normalization responses, not source-shape
evidence or a luminosity law.

The paired optimizer coordinate `ell/sigma_x` confirms that factor 15 is an
active constraint.  From 15 to 20, the median change over 878 pairs is `+0.864`,
the 95th percentile of the absolute change is `2.986`, and the largest change
is `+5.969`; only 276/878 pairs change by less than 0.01.  Among the 599 exact
factor-15 contacts, the median change is `+1.442`.  Median 15-to-20 changes by
lane are `+1.365` (native 10%), `+0.712` (native-10%x10), `+0.815` (1%x10),
and `+0.651` (1%x100).

The 20-to-25 comparison is a strong but not exact plateau.  Its median signed
change is `-6.7e-6`, median absolute change is `2.33e-4`, and the 95th
percentile absolute change is `0.00345`; 869/879 pairs change by less than
0.01.  Excluding the two exact factor-20 contacts, the largest absolute change
is `0.0222`.  The two contacts move from 20 to `21.128` and `20.249` at 170 and
210 MeV, with LML gains `0.2326` and `0.0160`, respectively.  Thus factor 20
contains nearly all selected optima but is not a universal non-contact ceiling;
factor 25 is the clean non-contact stress control in this ensemble.

Nested-LML behavior is consistent with that picture.  The 15-to-20 median LML
gains are `1.068`, `0.389`, `0.482`, and `0.338` in the four lanes listed above,
with maxima `4.841`, `5.234`, `5.049`, and `4.706`; there are no strict or
material nested-order violations.  For 20-to-25 the lane medians are all within
`8.1e-7` of zero.  Ten of 879 comparisons cross the deliberately strict
`Delta LML < -1e-4` numerical threshold: six in native-10%x10 and four in
1%x100.  Their changes range from `-1.89e-4` to `-1.11e-4`, or only
`-4.89e-7` to `-2.78e-7` per training bin.  None crosses the material
`Delta LML/n_train < -0.001` gate.  These ten rows prevent a claim of exact
optimizer monotonicity, but they are numerical-scale discrepancies rather than
material evidence favoring the smaller nested domain.

Recommended decision language:

> In the locked background-only stress scan, factor 15 is demonstrably active:
> 68.2% of selected points lie exactly at its upper boundary and same-input
> factor-20 fits yield substantial coordinate and likelihood changes.  Factor 20
> removes all but two exact contacts, while factor 25 removes all contacts and
> changes 98.9% of paired coordinates by less than 0.01 relative to factor 20.
> Ten tiny strict nested-LML reversals occur, but none reaches the pull-blind
> material threshold.  This supports describing factor 25 as a non-contact
> conditional stress control and 20-to-25 as a near-plateau.  Because the scan
> uses one source-fitted stress mean and deliberately performs no factor
> selection, it does not by itself authorize a production ceiling or card
> change.

### 5.8 Direct inference and calibration

Reserve `limit`, `coverage`, and `90% CLs` for actual `tilde_q_mu` CLs at
`alpha=0.10`.  Store alpha, CLs mode, bracketing/crossing checks, and calculation
status in every result row and caption.  Injection pulls do not calibrate CLs
coverage.  Archive or mathematically relabel legacy `eps2_95` and per-toy proxy
products.  The 90% Student-t/chi-square intervals in Figure 48 summarize 20
conditional pulls; they do not substitute for the CLs construction.

### 5.9 Decision ledger

End with columns: question, frozen inputs, independent unit, diagnostic, result,
allowed claim, promotion gate, and production impact.  For v4.8 the entries are:

| Question | Result | Allowed claim | Production impact |
| --- | --- | --- | --- |
| `fSigPowExpQ` common source family | rejected | poor/invalid reconstructed-source fit | none |
| Literal `fSigPow` and archived simple pool | rejected | no low-dimensional common family qualifies | none |
| Positive log-Chebyshev scan | in-sample candidates only; predictive qualification fails | useful failure diagnosis | none |
| Sparse threshold `T2+T6` mean | frozen 1% shape; conditional-stress use only | declared smooth generator for this screen | none |
| Native-10% shape refit | rejected by 17.3--32.2% absorption diagnostic | normalization-only transfer is more signal-rigid | none |
| Primary fidelity interval | 50--250 MeV; native-bin engineering ratios below 3 | adequate for declared conditional stress screen, with visible broad residuals | none |
| 40--300 MeV support | authorized; outer intervals are training shoulders | positive stable support for the conditional GP runs | none |
| 30-MeV support | rejected for the sparse family | no claim below 40 MeV | none |
| Four exposure lanes | 1,599/1,600 accepted; one 65-MeV `z=5` injected row excluded; authoritative products use indices 0--19 only | operationally complete finite conditional screen; 20--22 were inspected only in a superseded development run | none |
| Zero-signal closure | nonuniform: 65-MeV means down to -1.327/-1.305; additional 90/120/180-MeV offsets | conditional model/estimator mismatch under this stress mean | no coverage or card claim |
| Pull width and recovery | widths 0.681--1.401; strong-injection median recovery 0.714--1.062 | finite-sample extraction diagnostic with mass/exposure dependence | none |
| Factor 15/20/25 ceiling scan | complete: factor-15 exact occupancy 599/878; factor 20 has 2/880 exact contacts; factor 25 has 0/879; 10 strict and 0 material nesting reversals | factor 15 is active and 20-to-25 is a near-plateau under this stress mean | no production-ceiling recommendation |
| Confidence level | frozen card already alpha=0.10 | actual products must assert 90% CLs | card unchanged |

## Wording fixes in the current Section 5

- Replace `true toy spectrum` with `declared generating mean`.
- Replace `most realistic` with `procedurally faithful conditional refit`.
- Replace claims that a generator change *caused* a response with `the
  conditional response changes when the generating mean is replaced` unless
  same-background paired causal controls exist.
- Replace `identifies the source of underprediction` with `shows that the offset
  is stable under this stress generator; it does not uniquely assign the cause`.
- Replace `answers` with `constrains` or `informs` for diagnostic studies.
- Move long v3/v4.5/v4.6 provenance and ten-toy development history to an
  appendix so the current estimator, generator gate, dependence, and claim
  boundary are visible before results.

## Figure status for this v4.8 branch

The original source-qualification and blocked-placeholder figures remain valid
records of the rejected `fSigPowExpQ`/Chebyshev nominal-qualification branch.
Their blocked status must not be generalized to the later conditional branch.

For the user-directed sparse branch:

- the source-fit figure documents the 50--250 MeV fidelity decision and the
  40--300 MeV support shoulders;
- the Figure-46 analogue is Poisson-construction QA for analyzed indices 0--19,
  not an expected-limit band;
- the Figure-48 analogue is a 20-background conditional pull/recovery screen
  with 90% finite-sample diagnostic intervals, not coverage or observed bias;
  and
- the Figure-136 analogue is a pull-blind optimizer/bound diagnostic under the
  fixed stress mean, not authority to change the production length-scale
  ceiling.

Indices 20--24 do not appear in any of those active-branch figures or summaries;
20--22 are not an unopened reserve because of the superseded development run.
No Figure-46/48/136 outcome from this branch promotes a
production card, observed result, expected band, coverage statement, or kernel
ceiling.
