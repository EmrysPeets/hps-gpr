> **Latest follow-up:** see the [updated review](../unblind_meeting_RCmeet_20260923b/RECOMMENDED_CHANGES.md), [D_side and residual explanation](../unblind_meeting_RCmeet_20260923b/READING_SLIDES_13_14.md), and [revised speaker notes](../unblind_meeting_RCmeet_20260923b/SPEAKER_NOTES.md).

# Spoken explanations for the instructed revisions

Slide 9 is unchanged in numbering. For the other walkthroughs, both original and final numbers are shown: original 22 is now 23, original 26 is now 28, and original 27 is now 29. The two inserted validation slides are final slides 15 and 25. All three slide-9 options are also saved in its native speaker notes.

## Slide 9 — option 1: concise, about 30 seconds

“These three panels separate the roles of the GP parameters. ℓ sets how far background variations stay correlated in log mass. C sets their variance in the latent log rate. α describes counting noise: approximately one over each bin count, so noisier bins constrain the fit less. We optimize C and ℓ on the sidebands; α follows the count rule. The curves illustrate these roles, rather than showing an HPS fit.”

## Slide 9 — option 2: intuitive, about 60 seconds

“Think of the background as a smooth trend inferred from noisy histogram bins. The length scale ℓ tells us how far one fluctuation carries information: a larger ℓ links more widely separated masses and produces broader variations. Because the coordinate is log mass, that distance is a fractional change in mass, not a fixed number of MeV.

“C controls the size of the allowed fluctuations in the log rate. Increasing C changes their amplitude, while keeping ℓ fixed leaves the correlation distance unchanged. Finally, α describes how noisy each measured bin is. High-count bins have smaller relative noise and constrain the trend more strongly; low-count bins get less weight. C and ℓ are fitted from sidebands. α comes from the count rule, rather than being another adjustable kernel parameter.”

## Slide 9 — option 3: technical, about 90 seconds

“We fit log counts as a function of log mass with this squared-exponential covariance. The length scale ℓ controls the falloff with separation: at a log-mass distance of one ℓ, the correlation is exp minus one-half, about 0.61. Thus ℓ describes the fractional mass scale of background variation; it is distinct from the detector’s resonance resolution.

“At zero separation the kernel equals C. C is therefore the prior variance of the latent log rate, and its square root is the corresponding standard deviation. It controls covariance amplitude, not the normalization of the event histogram. Scaling C alone does not change the normalized correlation curve.

“For the observed bins, we add α to the covariance diagonal. Propagating Poisson count variance through the logarithm gives α approximately equal to one over the count for positive bins. This is why a noisier bin pulls the prediction less strongly. The zero-count case has an explicit finite floor. We optimize C and ℓ using the allowed sidebands, while recomputing α from the observed counts. The three panels isolate these effects schematically; they are not three alternative fits to the data.”

## Optional parameter prompts — ℓ: how far a fluctuation stays correlated

“The horizontal coordinate is log mass, so a fixed separation represents a fixed fractional change in mass. The length scale ℓ sets how far a background fluctuation remains correlated: larger ℓ gives broader, smoother variations. At a separation of one ℓ, the correlation is about 0.61. We fit this scale using the sidebands; it is different from the detector’s resonance mass resolution.”

## Optional parameter prompts — C: how much the background can vary

“C sets the vertical scale of the covariance. Specifically, it is the prior variance of the latent log rate at one point, so its square root is the corresponding standard deviation. Increasing C allows larger fluctuations while keeping the same correlation distance if ℓ is fixed. C is not the event-count normalization; the sideband fit determines the predicted background.”

## Optional parameter prompts — α: how strongly each noisy bin informs the fit

“The histogram bins also have counting noise. In log space, Poisson counting gives a variance of approximately one over the bin count, which we call α. A high-count bin has smaller α and constrains the fit more strongly. A low-count bin has larger α and gets less weight. This term is added only on the covariance diagonal and follows the count rule; it is not a third fitted kernel parameter.”

The α illustration is schematic. For zero-count bins the implementation uses its stated finite floor, α = 1; the approximation 1/y applies to positive counts. These figures illustrate parameter roles, not fits to the HPS spectrum.

## New slide 15 — why the validation is needed

“A flexible background fit can look smooth and still create or absorb a narrow signal. We therefore generate background toys from separately fitted analytic spectra, with the GP seeing only the permitted sidebands. First we ask whether a background-only toy produces a spurious fitted yield and whether its reported error describes the spread. Then we add a known signal to the same background realization and test its recovery. The residual-size plot is one diagnostic; it is not automatically a chi-squared test with a mean of one.”

For the revised sideband-only slide 14, add: “This new diagnostic measures the bins used to train the model. At the selected 78-MeV exclusion, 237 of 256 conditional refitted toys have a sideband deviance at least as large as the data. That comparison does not test the withheld bins, include uncertainty in the generating source, or certify the background model.” The sideband D/Nside values at the 65, 78 and 120 MeV exclusions are 0.889, 0.852 and 0.915. Nside is a bin count, not fitted degrees of freedom. The tail interval is 0.887–0.955 at 95% confidence.

If comparing with the old held-out residual plot: “That Q/Nbin statistic answers a different question. Its conditional expected mean is 0.998–1.185 across the v5.9.5 scan; neither statistic has an automatic unit expectation.”

## Slide 22 → final slide 23 — exactly what is injected

“We start with a particular background toy at a particular mass. Before adding a signal, we fit that toy and record the error on its signed fitted yield. That reference error sets the injected count: one, three or five times the error, as well as the zero-signal case. We add the full Gaussian template to that same background toy, rerun the GP sideband fit, and extract the signal again. The injected yield stays fixed during this refit. Its label z is an input strength, not a promise that the fitted significance will equal z.”

If explaining the pull, add: “The pull uses the error from the fit after injection. That is distinct from the reference error used to choose the injected yield.” Do not add a fresh Poisson signal-count draw to this description unless the archived injection mode is established; the source record supports the shared background realization and full-template injection.

## New slide 25 — what the tests establish

“These checks support the selected training and extraction geometry, but the numerical limits matter. All twenty displayed background-only mean pulls lie inside our practical half-sigma tolerance, which was chosen after seeing the results. The refined 65-MeV study recovers about 97 percent of the added amplitude, leaving a small measured loss. Some pull widths differ from one. In the separate frozen-source test at 78 MeV, the mean signed likelihood root is about 0.548, rather than zero. These are checks of specific sources and procedures; background-source uncertainty and confidence-limit coverage remain open.”

## Slide 26 → final slide 28 — reading the four upper-limit equations

“At a fixed mass, choose a trial signal yield A. The first equation measures how incompatible that yield is with the data after profiling the correlated background. If the data prefer a yield larger than A, the statistic is zero: that upward fluctuation does not exclude A. If the unrestricted best fit is negative, the physical zero-yield fit sets the denominator.

“The next two equations take the tail of that same statistic under signal plus background and under background alone. Their ratio is CLs. Dividing by the background tail avoids an overly strong exclusion when a downward fluctuation gives little sensitivity to the tested signal. We scan A and take the first crossing of 0.10 as the 90 percent upper limit. This is a pointwise limit at the chosen mass; the discovery look-elsewhere correction is a separate calculation.”

Profile the likelihood nuisance coordinates at each A; do not describe this as rerunning the GP kernel optimization at every A. The existing equations are the definitions; a Gaussian illustrative curve is not an observed HPS limit or an independent coverage check.

## Slide 27 → final slide 29 — from yield to coupling

“The upper limit so far is a count of signal events. To express it as a coupling, divide by K: the predicted electron-pair signal yield at unit electron-channel coupling. K uses the local prompt count density, the radiative fraction, the mass and the fine-structure constant. The fitted coordinate is epsilon-squared for the electron channel. For the minimal visible dark-photon interpretation, multiply by the inverse electron branching fraction. That factor is one below the dimuon threshold and rises above it. The plot already includes this branching correction.”

If asked about windows: “The prompt density uses the established ±1.64σ normalization interval. The GP training exclusion and signal extraction use ±2.25σ; those intervals serve different purposes.” The symbol α_EM here is the fine-structure constant, not the GP bin-noise α on slide 9.

## Source anchors

- `study_results/v5p0p5_analysis_note_20260916/source/sections/04_methodology.tex`: log-count kernel/noise; matched-reference injection; bounded statistic and CLs; yield/coupling and dimuon conversion.
- `study_results/v5p0p5_analysis_note_20260916/source/sections/05_toys_validation.tex`: analytic toy sources, paired response, practical mean tolerance, width qualifications.
- `study_results/v5p9p5_null_bias_20260922/residual_diagnostic/findings.md`: conditional expectation of Q/Nbin.
- `study_results/v5p0p4_analysis_note_20260911/scripts/make_v504_figures.py`, lines 29–32 and 67–68: the preserved observed-coupling overlay plots branching-corrected `eps2_observed`. The v5.0.4 and v5.0.5 PNGs have identical SHA-256 `65ce9b8b1926f8f33ee30bbdf265a6c824a5a315ada3e3b22b36f224d6c56afe` and visually match the existing slide-27 plot; the Slides image bytes were not independently hashed.
