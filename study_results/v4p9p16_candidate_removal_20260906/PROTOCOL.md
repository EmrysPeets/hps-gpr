# Frozen candidate-removal and conventional-search protocol

Frozen before calculating intervention outcomes, 6 September 2026.

## Scope and selection

Use the released full 2015/full 2016/native 2021 10% spectra and the current dense likelihood. Preserve all sealed studies, production inputs, kernels, scan grids, signal resolutions and conversions. The two selected observed positive GP peaks are frozen from the presentation catalogue: 2015 51/21 MeV; 2016 90/117 MeV; 2021 78/65 MeV. Do not select masses from the extreme stress-centered probability ordering. No additional real data will be opened.

## Counterfactual replacement

Primary holes contain whole analysis bins whose centers lie within 2.25 times the archived resolution of the selected mass. Each hole and their union are replaced. All bins outside the chosen hole(s) remain byte-identical. Fit the primary replacement GP using only bins outside BOTH holes, with the year's frozen kernel at its leading selected peak. Keep this kernel fixed. Predict a joint latent log-intensity for both holes. Use its mean count expectation for deterministic fills and ten joint latent-GP draws followed by independent Poisson counts for conditional replacement replicas. The query latent covariance excludes added Poisson observation noise. Each replicate supplies one fixed full modified spectrum for every scan mass; the three removal lanes share its draws for paired comparisons.

The observed spectrum and the archived global-study reference spectrum are separate source lanes. Learn each replacement only from that source's retained bins. Never insert an observed-data fit into the stress spectrum. Reference-spectrum interventions are deterministic; no observed p-values are inferred from modified spectra.

Two fixed robustness checks replace both holes deterministically: (i) a local positive exponential polynomial from retained sidebands within +/-7 resolutions at each candidate (degree5 for 2015 below39MeV, degree3 otherwise), with both primary holes excluded; (ii) the same primary GP construction with both holes widened to3 resolutions. Neither alternative is chosen by the resulting oscillations. Report both even if they disagree.

Retrain the original moving-mask background for every modified spectrum, preserving all per-hypothesis kernel states. Use the exact cached Cholesky backend, no new kernel optimization or low-rank approximation. Compute signed profile roots and retain native likelihood components for deterministic fits and representative replica fits. The original dense observed scan and reference offsets must close before interpreting modifications.

## Oscillation comparisons and conditional follow-up trigger

Report full-grid and remote-grid variation. A mass is remote only if its entire native signal-fit window is disjoint from BOTH primary holes; for the widened-hole robustness use the same primary remote grid and mark the distinction. Report standard deviation of the root about its own mean, RMS, peak-to-peak span, correlation with the original field, maximum absolute change and sign transitions within contiguous retained intervals. Deleting a candidate's own bins trivially weakens that local feature; it cannot establish physical causation.

A descriptive substantial-persistence flag requires at least half the original remote standard deviation and at least two remaining sign transitions within contiguous remote intervals. This is a study-routing threshold, not a hypothesis test. If any dataset retains oscillations in either the observed or reference lane, perform the conventional cross-checks at all six originally selected masses. Report continuous metrics regardless of that flag.

## Traditional fits, if triggered

Fit the ORIGINAL released spectra, never the fake-filled spectra as new evidence. Use a positive exponential-polynomial background plus a bin-integrated Gaussian of the archived mass and resolution, profiling all polynomial coefficients jointly with the signal amplitude. Baselines: 2015 m21 degree5 with total14sigma window; 2015 m51 degree3 with total13sigma window; 2016 and2021 degree3 with total8sigma window. These are HPS-inspired diagnostics, not an exact replay of the historical publication. Retain five variants per mass: baseline; degree-1/+1 at fixed width; total width-2/+2 resolutions at fixed degree. Use full original bins and report actual edges. Record signed root, nominal asymptotic one-sided local p0 (0.5 convention for nonpositive roots), fitted yield/error, convergence, covariance and descriptive deviance/ndof. Do not pick the smallest probability or claim independence from the GP search that selected these masses.

## Interpretation and release

Modified spectra are conditional influence experiments. They do not establish that a removed feature was a particle, calibrate discovery or coverage, supply a global significance, or quantify expected sensitivity. Ten replacements are not ten independent background experiments. Incorporate source-specific reference-offset findings, residual oscillations, fill-method dependence and conventional-fit robustness into a new section of the current v4.9.16 note. Preserve prior text and plots except explicit cross-references/context updates. Require independent HEP review, numerical checks, rendered PDF QA and a SHA-256 manifest.
