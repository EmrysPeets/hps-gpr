# Interpretation and novelty review

Reviewed 17 September 2026 against the linked paper, v5.8.2 report and implementation, and the saved v5.8.1 source-injection results. No new fits or toys were generated for this review.

## Published method and defensible contribution

The central look-elsewhere method is already published. Ananiev and Read construct a signed local-likelihood-root field, estimate its covariance through controlled background-bin perturbations, and sample Gaussian fields to obtain scan-maximum tails. Their treatment includes covariance-based upcrossings and finite-grid effects; their examples already allow mass-dependent resolution. The significance field assumes centered, unit-normal local marginals. These ideas should be credited directly to [Ananiev and Read, arXiv:2206.12328, Sections 2–3](https://arxiv.org/pdf/2206.12328). The current HPS result is an application and extension of that framework, not a new invention of GP trials factors. No broad priority claim has been established by this bounded review.

Useful contributions specific to this implementation are:

* Propagating count fluctuations through the moving-mask, log-count GP interpolation, its count-dependent training errors, its predictive covariance, and the profiled Poisson/Gaussian signal likelihood.
* Differentiating that response efficiently, then checking it against direct recomputed fits and complete one-standard-deviation response banks.
* Building the exact common-coupling field across changing dataset participation over 19–250 MeV, preserving the contributions of independent dataset noise directions.
* Making reference offsets, marginal widths, positive-fit ordering, source construction, grid effects, and direct-Poisson validation explicit and separately testable.

A defensible description is: “We implement and validate a response-GP look-elsewhere calculation for the moving-mask HPS background analysis, including source-dependent offsets and a common-coupling search with changing dataset participation.” Its publication value would depend on the demonstrated performance, reproducibility and background qualification. Analytic differentiation and covariance propagation themselves are established ideas; first-use or general-method novelty needs a separate literature comparison.

## Two different GPs

The **background GP** models expected counts as a function of reconstructed invariant mass. Its kernel is part of the interpolation model; it supplies both a mean and a covariance for the masked signal-region fit.

The **response/significance GP** approximates the distribution of repeated signed fit results as a function of the *tested signal mass*. Its covariance is derived from the full fitting procedure:

\[
D_{jm}=\sqrt{B_j}\,\partial r_m/\partial n_j,
\qquad \Gamma=D^TD.
\]

It is not obtained by reusing the background GP's length scale. The v5.8.2 implementation normalizes the derived covariance and retains its deterministic offset and width. Its positive-fit-gated, reference-centered ordering is an explicit HPS extension whose particle-discovery interpretation requires additional null qualification.

## Resolution and the number of search opportunities

Nearby signal hypotheses share events because their signal templates overlap. In the illustrative limit of Gaussian templates of common width \(\sigma\), constant background variance, and no background fitting, normalized template overlap gives the derived correlation

\[
R(m,m+\Delta m)=\exp[-\Delta m^2/(4\sigma^2)].
\]

Thus detector resolution sets a characteristic correlation scale in this simple limit. It does not make hypotheses separated by exactly one resolution independent, nor does it establish an exact trials count equal to search width divided by resolution. In HPS, changing resolution, moving masks, sideband reuse, nuisance profiling, nonuniform counts, and changing campaign weights all alter the response correlation. Signed response directions can also produce negative correlations.

An equivalent independent-trial count fitted at one tail threshold need not work at another. The appropriate result here is the maximum-tail distribution over the declared correlated search. A denser numerical grid resolves that search more closely; it does not create a proportionate number of independent experiments. The finite-grid discussion in [Ananiev and Read, Section 2.2](https://arxiv.org/pdf/2206.12328) supports explicit convergence checks rather than a universal grid-to-resolution rule. In the saved HPS calculation, 1→0.5 MeV changes the fixed-threshold-three Gaussian tail from 0.059655 to 0.070060 for 2015, and 0.134105 to 0.148175 for the combination. The half-MeV result therefore remains a finite-grid calculation, not a demonstrated continuous-search limit.

## What the implementation validation establishes

`scripts/engine.py` differentiates the raw GP predictive mean and covariance, including their count dependence. The numerical covariance-conditioning derivative is omitted; `qa/response_derivative_validation.json` states this explicitly. Direct recomputation checks the actual conditioned fitter. The saved maximum derivative discrepancy is 3.67×10⁻⁶. Four complete perturbation banks contain 3,252 directions and give a largest relative response-width discrepancy of 0.1371% and a smallest direction cosine of 0.999989. This supports a validated differential approximation rather than an exact derivative of every numerical operation.

`scripts/analyze.py` uses the same positive-fit gate for observed scores, Gaussian maxima and complete-Poisson maxima. It keeps integer and half-integer evaluations within the same Gaussian draw when comparing grids. The 256 direct scans test bulk distributions and accessible tails under the specified fixed source means. Their interval agreement with the Gaussian peak probabilities is useful conditional evidence; it does not establish arbitrary rare-tail accuracy or physical background adequacy.

## Conditioning and source-signal absorption

`scripts/prepare.py` fits one all-data GP mean per dataset at its reviewed 76 MeV kernel, draws complete Poisson spectra from that mean, and then holds the source fixed. The main mass-local extractor is still retrained on each toy's sidebands, but the coherent source-estimation step is not repeated. These are therefore fixed-source conditional probabilities, with no propagated uncertainty in how the source was estimated or selected. A final calibration must specify whether source estimation is part of the tested procedure and, if so, repeat that operation in validation or supply an independent reference.

The source-injection ledger `v5p8p1_background_truths_20260917/truths/source_absorption.csv` shows why this matters. After rebuilding the nominal 2016 GP source with a five-reference-error injection, the source mean gains 88.67% and 88.41% of the injected **window-summed yield** at 76 and 90 MeV. The signal-template Poisson projections are instead 71.83% and 71.73%. These are distinct deterministic contamination diagnostics, not measured signal efficiencies, discovery power, or a direct estimate of bias in the observed excess. Strong later recovery from injections into a frozen background cannot undo a signal already incorporated during source estimation.

Accordingly, the v5.8.2 curves provide a useful coherent-reference comparison and a tested implementation of correlated scan probabilities. They do not yet replace a background-qualified particle-discovery significance. The source family, its construction uncertainty, any model or scope selection, and complete-search calibration remain part of that eventual inference.
