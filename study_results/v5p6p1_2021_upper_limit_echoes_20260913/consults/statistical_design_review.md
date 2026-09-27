# v5.6.1 upper-limit echo review

This bounded review used local source and equations; no numerical processes or new toys were launched. The extension is defensible as a conditional catalogue of how the declared GP procedure responds to injected peaks. Reuse all 15 scenarios and their exact 300 saved spectra. Freeze the extension protocol before inspecting full 2021 data, while stating that the source peak choices already depend on observed partial data. This freeze does not grant unblinding approval.

Implementation update: `scan_limits.py` saves both a per-spectrum pseudo-data-density coupling display and the fixed-continuum-density alternative described below. Its echo statistic uses A90, so the principal echo comparison is unaffected by denominator fluctuations. Only the fixed-density coupling ratio equals the yield ratio. For the per-spectrum display the exact relationship is R_epsilon(m)=R_A(m) rho_background(m)/rho_spectrum(m); injected signal tails and ordinary count fluctuations can therefore alter its visual dip contrast. The script recreates the inherited dimuon factor analytically; validation compares it with the pinned released table.

## Limits and reference curves

Use the existing `OneSignalProfile.limit(n, alpha=0.1)` with the same bin-integrated full-yield Gaussian template, fixed source-specific kernels, fully scaled widths and per-spectrum sideband reconditioning. Keep the 1% 53–250 MeV and 10% 50–250 MeV integer grids explicit. The historical exclusion of 50–52 MeV is not independent qualification of every retained mass.

For tested nonnegative yield A, the bounded likelihood ratio uses the free optimum when its yield is nonnegative and the null optimum otherwise; its statistic is zero when the free fitted yield exceeds A. The existing solver profiles both the observed statistic and its internal fixed-model background-Asimov reference, then solves CLs(A90)=0.10. Its internal Asimov reference uses the conditioned GP mean and covariance at that mass. Distinguish this from the separate full-support background-only generator spectrum, which is passed through fresh GP conditioning to produce the reference curve.

For each scenario retain matched-injection Asimov, yield-scaled-injection Asimov, background-only Asimov and 20 existing toy curves. Call the toy curves "upper limits obtained from injected pseudo-data" or "projected pseudo-data limits"; they are neither observed full-data limits nor calibrated expected bands. The matched and yield-scaled deterministic curves answer different injection questions, as documented in v5.6.0.

## Exact, defensible coupling display

A90 is the primary result and denotes the full Gaussian electron-pair yield, not the yield inside the fit mask. Define a scenario's fixed expected continuum bin counts beta_i from its saved full-exposure `background` array. For density-bin edges e_i, widths Delta_i, and h(m)=1.64 sigma(m), use

\[
w_i(m)=\max\{0,\min[e_{i+1},m+h]-\max[e_i,m-h]\},\qquad
\rho_{\rm fixed}(m)=\frac{1}{2h}\sum_i\beta_i\frac{w_i(m)}{\Delta_i}.
\]

Masses and widths in this equation must be in GeV, so rho has units of counts/GeV. Require complete density-window support; never clip the requested interval. Use this same fixed function for every injection convention, every toy and the background-only reference within the scenario. Do not include injected counts or a toy-dependent GP fit in this denominator.

The saved continuum has 0.625 MeV bins. The cleanest implementation integrates those bins directly with the exact fractional-overlap equation. If a 0.125 MeV native-bin representation is needed, conservatively split each saved count equally among its five native sub-bins; this is exactly the same piecewise-constant continuum and supplies no additional resolution. Smooth interpolation would introduce a new density model and is unnecessary. Label the result "fixed expected-continuum density proxy," not native observed prompt density: the production conversion uses an uncropped observed density histogram, as documented by `hps_gpr/io.py::_compute_integral_density`.

The corresponding electron-normalized display is

\[
\varepsilon^2_{90,ee}(m)=\frac{2\alpha_{\rm EM}A_{90}(m)}
 {3\pi m f_{\rm rad}^{\rm eff}(m)\rho_{\rm fixed}(m)},
\qquad
\varepsilon^2_{90,\rm visible}=F_{\rm vis}(m)\,\varepsilon^2_{90,ee}(m).
\]

Use the pinned effective radiative fraction and alpha_EM=1/137 from the inherited implementation. For the integer mass grid, reuse the already-pinned `released_2021_asymptotic.csv::dimuon_factor` as F_vis. Apply it exactly once; it changes coupling interpretation above the dimuon threshold while leaving electron-pair yields unchanged. Preserve the electron-only and corrected columns separately. This is a conditional count-density display with inherited response/branching assumptions, especially for the selection-distinct 1% source.

Because the denominator and branching factor are common at a fixed mass within a scenario,

\[
R(m)=\frac{A_{90,\rm injected}(m)}{A_{90,\rm background}(m)}
 =\frac{\varepsilon^2_{90,\rm injected}(m)}{\varepsilon^2_{90,\rm background}(m)}.
\]

Thus the echo result is independent of this display conversion. A90 and this ratio should carry the scientific comparison.

## Echo selection and interpretation

Define u=(m-m0)/sigma(m0), with the injected mass and its width held fixed. Search the two disjoint flanks -8<=u<=-2 and 2<=u<=8 separately, intersecting them with the lane's scan grid. Find minima of R, not raw A90 or epsilon-squared curves, whose mass dependence can otherwise create apparent dips. Predeclare the neighbor/plateau tie rule. Treat the two flanks as disconnected; never compare neighbors across the excluded central interval. An absent flank or a boundary minimum must be identified explicitly. The 244 MeV scenario, for example, has no accessible right flank starting at +2 sigma within the 250 MeV endpoint.

List connected grid runs with R<0.9, their sampled endpoints and boundary flags. These are descriptive threshold crossings with 1 MeV sampling, not confidence regions; do not imply sub-grid location precision. A low ratio alone does not uniquely establish a GP echo: it can reflect a statistical deficit or a change in fitted uncertainty. At deterministic candidate minima retain the fitted signed yield, uncertainty, GP-predicted background and corresponding background-only quantities. A GP background uplift accompanied by a negative fitted amplitude supports the proposed mechanism.

Under the implemented "prefer an interior local minimum" rule, an even lower flank-endpoint value can coexist with the selected interior minimum. Name the selected quantity accordingly and retain the absolute sampled flank minimum/boundary flag separately; do not call the interior choice the global flank minimum. Plateau ties can be resolved by the lowest mass. For a threshold run, report both the sampled endpoint span and number of grid points so that a one-point run is not misread as a precisely measured zero-width structure.

For toy stability, report the ratio at the selected deterministic minimum as well as each toy's deepest flank minimum. To track a specific dip, predefine a nearest-minimum matching radius around that deterministic location and record unmatched toys; do not force a match. The alternative of choosing every toy's lowest point selects downward fluctuations and therefore overstates stable echo depth. All 20-toy spreads and match fractions remain descriptive, with no coverage or rare-tail interpretation.

## Minimal validation

1. Verify input NPZ and numerical-source hashes, exact scenario/seed identity, and unchanged saved counts. Check every prescribed mass for each of 23 spectra per scenario: 9x198x23 + 6x201x23 = 68,724 limit rows. Keep failure ledgers and unchanged-input retries; never replace a failed spectrum or interpolate a failed mass.
2. Record positive finite A90, fitted means/backgrounds and uncertainties; signed Ahat may be negative. Retain solver method, score, likelihood nesting, CLs(root)-0.1, monotonicity and covariance/feature diagnostics. Preserve the solver's existing tolerances and branch behavior. Reuse its closed one-bin, deficit-branch and amplitude-unit invariance checks rather than creating another limit implementation.
3. At the original injection masses, compare each new limit call's fitted Ahat and signed root against the existing v5.6.0 extraction of the identical spectrum. Require numerical agreement within declared tolerances. The limit call's internal background Asimov does not imply recalibration of the entire GP procedure.
4. Verify density-window coverage, positive rho, conservative native rebinning if used, GeV/MeV unit invariance, round-trip yield/coupling conversion, exactly-once dimuon correction, and the equality of yield and coupling ratios. Persist the frozen density/branching arrays and their hashes.
5. Verify flank membership, disconnected-side local minima, deterministic tie rules, absent/boundary flags and R<0.9 runs from saved rows. Check representative deterministic deepest dips under the already-declared stable-GP cutoff comparison using identical counts. A material numerical change is a limitation to resolve or report, not a reason to select a more attractive minimum.

Local references inspected: `hps_gpr/conversion.py`, `hps_gpr/io.py`, v5.6.0 `scripts/limit_solver.py`, `scripts/parent_core.py`, and the pinned v5.0.4 dimuon interpretation. Memory was used only to locate the earlier physical-window density audit, not as unverified numerical evidence.
