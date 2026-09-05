# 2021 signal-template width sensitivity

Predeclared exploratory scan, with zero additional toys:

- Use the v4.9.12 2021 10% observed data and reviewed optimized-support
  configuration (36–300 MeV GP support, length-scale factor 15).
- Evaluate all 201 integer mass hypotheses from 50 through 250 MeV.
- At every mass, reconstruct the nominal GP background mean and covariance
  using its reviewed kernel coordinates. Hold that background, covariance,
  and nominal ±2.25σ fitted/training-exclusion bins fixed for all five widths.
- Change only the Gaussian signal-template width: 0.8, 0.9, 1.0, 1.1, and 1.2
  times the nominal resolution at that mass. Integrate each shape over the
  original bins and refit its amplitude. Record its fraction in the fixed
  fit window. This is not a full detector-resolution systematic rerun: masks,
  GP length-scale constraints, density normalization, and data are unchanged.
- Compute signed profile-likelihood r with the existing covariance rule and
  production best-feasible-profile reconciliation. Retain optimizer and
  fallback diagnostics; do not discard unfavorable points.
- Cache the fixed background-only nuisance fit at each mass. Check nominal
  r against all 201 saved production signed fits before accepting the scan.
- Following the additional request for yield and coupling limits, evaluate
  observed 90% CLs limits with the production bounded piecewise asymptotic
  solver at each mass and width. Keep the saved full-template conversion
  K(m) = signal yield per epsilon squared fixed, but recompute the signal
  fraction f(m, width) inside the original fitted bins. The solver receives
  K times the unnormalized in-window template; report epsilon squared,
  epsilon, full-template yield K*epsilon squared, and fitted-window yield
  K*f*epsilon squared. Require nominal limits to reproduce the saved values
  within 0.05% relative error and retain full solver diagnostics. Plot every
  width and its ratio to nominal, without expected bands or an optimized
  exclusion envelope. These fixed-width curves do not profile or marginalize
  a resolution nuisance and are not a resolution-systematic-inclusive limit.
- Numerical retry policy after the first complete limit pass: for the four
  rejected points (50 MeV, 1.2x; 64 MeV, 0.9x; 99 MeV, 1.2x; 169 MeV, 0.9x),
  retry the same likelihood and CLs construction with L-BFGS-B ftol=1e-15,
  gtol=1e-8, maxiter=2000, and maxls=50. Apply this rule automatically to any
  rejected limit on reproduction, independent of its magnitude. Retain the
  first error and retry method in the ledger; do not relax nesting, root,
  or monotonicity gates. Leave a gap if the retry also fails. The override is
  process-local and temporary; no production runtime files are changed.
- Show all five widths and their envelope. The envelope is a variation range,
  not a confidence band or trials-corrected significance.
- Tabulate positive and negative excursions separately. For compact region
  summaries, find local maxima of each sign's best-width envelope (including
  scan endpoints where applicable), retain candidates with |r| >= 1, and
  greedily separate maxima by 2.25 times the larger nominal mass resolution.
  This grouping is descriptive, not an independent-trials definition.
- Report the gain at the same mass and, separately, over the best nominal
  score within ±2.25σ of the selected point. Distinguish the prior 60–88 MeV
  focal region from other regions. Identify search-grid endpoints explicitly.
- Run one low-priority process with one numerical-library thread. Preserve
  original pilot and reverse-injection artifacts; write only in this folder.

The mass and width choices are selected on observed data. All scores remain
conditional local diagnostics. A larger optimized score is not independent
evidence, a mass-resolution measurement, or a calibrated global significance.
