# Candidate-region replacement: independent protocol review

2026-09-06. This is a prospective review; no intervention result or traditional fit has been evaluated by this reviewer. Frozen parents remain unchanged.

**Disposition:** the proposed experiment is useful as a conditional diagnostic of the extraction procedure. It cannot identify the removed counts as particles, establish a background model, or provide discovery probabilities from the modified data.

## Intervention controls

Freeze the two leading separated positive observed-profiled peaks: 2015 at 51/21 MeV, 2016 at 90/117 MeV, and released 2021 10% at 78/65 MeV. Keep the original mass grid, native bins, resolution, kernel states and likelihood. Replace each hole separately and their union. Use precisely specified whole-bin masks derived from the existing +/-2.25-resolution convention; record actual bin indices/edges and the integrated Gaussian fraction removed. A nominal +/-2.25-sigma hole leaves approximately 2.44% of an ideal Gaussian outside, so the experiment removes candidate regions, not every possible signal contribution. The declared +/-3-sigma deterministic variant tests this limitation without selecting the more favorable result.

The filler must exclude **both** candidate holes while being constructed, even for a single-hole intervention. The updated choice of the frozen kernel at the year's leading peak is a reproducible anchor. The observed spectrum and saved stress-generating spectrum must have **separate** fillers trained from their respective exterior bins. Pasting an observed-data mean into the stress spectrum would mix the two questions. Freeze and save the filler before scanning; do not interpolate a new modified spectrum at each tested mass.

For the updated ten joint latent-GP-plus-Poisson replicates, draw the latent field jointly over both holes, preserving its off-diagonal covariance. Convert through the production log/count convention exactly once, then draw Poisson counts. Ensure the latent covariance does not already include a second observation-noise contribution. Use one hole realization in both the corresponding individual replacement and the joint replacement for each replicate ID. Keep every exterior bin fixed, and reuse that entire modified spectrum at all masses. Distinct year/source namespaces prevent accidental pooling. These ten realizations explore **conditional interpolation and counting variation**, not uncertainty in kernel choice, source adequacy, or a new independent background-null ensemble. The reference-spectrum lane needs deterministic replacement only for its stated question.

The no-change replay must reproduce the frozen raw scan, and the stress no-change replay must reproduce the archived Asimov offset. Verify that the exterior counts are byte-identical, masks are disjoint or their overlap has a declared ownership rule, replacement means are positive, and every mass is retained or has an explicit numerical failure. Refit the background with the modified training counts and their count-dependent errors; merely subtracting a Gaussian from a saved root does not perform this intervention.

## Readouts and interpretation

Keep `r(m)` for observed-data interventions separate from `a(m)=r_m(B)` for stress-spectrum interventions. A deterministic modified reference gives `a_new(m)`; it does not provide a newly calibrated response width or covariance. Do not reuse the original global-tail bank to assign probabilities to the modified data.

Report complete scans and changes at the fixed candidate/echo masses, plus metrics on a predeclared common exterior set. The principal exterior set should contain masses whose **entire signal-fit window is disjoint from both original holes**. Otherwise a reduction partly measures direct replacement of the fitted data. It is useful to show a second, less restrictive set with centers outside holes, clearly labeled. Keep these sets fixed across all interventions. Do not count a sign crossing across a gap in the metric's domain.

For each set, save the mean root, RMS about zero, standard deviation about its own mean, RMS change from the unmodified scan, correlation with that scan, sign crossings, and fixed-location roots. RMS and standard deviation answer different questions when offsets are large. Use the actual pointwise change of each paired replicate, then summarize the ten changes; do not subtract separately computed ensemble medians. Show the deterministic mean result alongside all ten or descriptive ranges. These are descriptive quantities, not confidence bands on a physical effect.

If an administrative persistence flag is needed to release the traditional-fit stage, freeze it before inspecting the replacement results, name it a diagnostic trigger, and report the underlying metrics even when they disagree. A simple bounded rule may use retained exterior RMS and nontrivial positive/negative structure; its threshold is a workflow decision, not a hypothesis-test level. Never change candidate locations, filler or mask width to achieve a desired trigger.

Disappearance means the chosen regions are influential under the chosen imputation. Persistence means those regions are not sufficient to explain the remaining pattern under that experiment. Neither conclusion alone establishes a particle or invalidates all GP methods. A GP-based filler can partially impose the same smoothness assumptions as the extraction, so disappearance is especially vulnerable to self-consistency rather than independent confirmation. An exterior-only alternative filler is a useful bounded robustness check if interpretation depends on the primary filler; its results must be retained without selecting the most favorable version.

## Traditional comparison, prepared before the trigger

The traditional search uses the **original released observed spectra**, at the six frozen GP-selected masses. Fits to imputed data may be additional mechanism checks but are not fresh search evidence. GP selection and the decision to run this comparison are data-dependent; local reference p-values remain descriptive, not an independent confirmation or a newly adjusted global significance.

The 2015 HPS paper uses an exponential Chebyshev background with a fixed Gaussian signal, degree 5 below 39 MeV and degree 3 above, and **total** window widths of 14 and 13 resolutions respectively. The 2016 paper uses an exponential Legendre polynomial, degree 5 at lower masses and 3 above 66 MeV, with total widths varying from 6 to 10 resolutions. These facts are verified in the primary publications: [2015 resonance search](https://arxiv.org/abs/1807.11530), [2016 prompt/displaced search](https://arxiv.org/abs/2212.10629). Neither publication by itself validates the current 2021 sample or the following adapted fixed-width comparison.

Use these predeclared baselines:

| Dataset and fixed mass | Polynomial basis and degree | Total window width |
|---|---|---:|
| 2015, 21 MeV | Chebyshev, 5 | 14 sigma |
| 2015, 51 MeV | Chebyshev, 3 | 13 sigma |
| 2016, 90/117 MeV | Legendre, 3 | 8 sigma |
| 2021 10%, 78/65 MeV | Legendre, 3 | 8 sigma |

At each coordinate retain five variants: baseline, degree minus/plus one at fixed width, and total width minus/plus two resolutions at fixed degree. They form a fixed robustness display; do not choose the lowest p-value. The 2016/2021 8-sigma baseline is an HPS-inspired adaptation, not an exact replication of the published mass-dependent window map. Exponentials with base e and base 10 span the same polynomial-background family after coefficient rescaling.

Fit a positive bin-integrated exponential-polynomial background and a bin-integrated Gaussian jointly in a Poisson likelihood. Fix mass and resolution; allow a signed auxiliary signal amplitude while requiring every total expectation positive. The positive discovery reference is `sf(max(r,0))`, with the same nonpositive-root convention as the current note. Use the full Gaussian-yield normalization and record the fraction inside each fit window, because the GP and traditional windows differ. Include full fitted counts, background, signal, free/null NLL, covariance, optimization diagnostics, and deviance/degrees of freedom. Nominal goodness-of-fit probabilities are diagnostics at selected locations, not a selector for a preferred p-value.

Use whole original bins only. If a window extends outside the available spectrum, shift the fixed-width window to the available edge and record the resulting asymmetry, following the published edge principle. No invented or fractionally split counts are allowed. At 21 MeV the nominal 14-sigma window reaches slightly below the available 2015 support, making that rule relevant. Nearby candidate tails may enter another candidate's traditional fit window; width variation and eventual joint multi-signal modeling address different aspects of this limitation.

Before using a result, require finite positive expectations, consistent free/null nesting, positive local curvature, agreement of fixed multistarts, and adequate stationarity. Independently check integrated predictions and signed-root arithmetic from stored components. The traditional stage has no authorization to generate a new global correction or new limits. Matched toys, spurious-signal studies and independent predictive qualification remain necessary before formal significance or coverage claims.
