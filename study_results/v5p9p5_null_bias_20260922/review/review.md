# Independent statistical review: slides 14 and 50

The audited formulas and pinned-input identities do not reveal an implementation error. The apparent positive null maximum is primarily a selection property of a scan, with additional source-dependent mean response. The residual-Q diagnostic addresses a different quantity and does not, by definition, have expectation one per bin. These findings do not establish unconditional calibration or exclude every possible numerical problem.

## What “bias” means here

For a fixed mass, use the **signed** root r or the signed yield error to diagnose centering. If r follows N(0,1), the upward-only display Z=max(r,0) already has E[Z]=1/√(2π)=0.398942, an atom of probability 1/2 at zero, and E[Z²]=1/2. Taking the maximum across mass increases the typical positive value. Neither a positive mean of Z nor a positive mode of the scan maximum proves a biased signed estimator.

The saved response field is r*(m)≈a(m)+s(m)W(m), where a=r(B) is the deterministic response to one frozen generating spectrum. The exact toy mean E[r(Y)] can differ from r(E[Y]) through nonlinear fitting. Report that remainder separately; do not identify the Asimov response with the exact ensemble mean by definition.

At the observed 2021 peak, 78 MeV, the independently reproduced values are a=0.576784, s=0.980025, toy signed mean=0.547975, and toy SD=1.011688. The mean-minus-a remainder is −0.028809 with Monte Carlo SE 0.063230. Thus there is a nonzero fixed-source mean response, but no resolved additional nonlinear remainder at this coordinate. The mean positive part is 0.747868 in the saved toys, consistent with the Gaussian response expectation 0.745189. The 78 MeV coordinate was selected by the observed maximum; these local numbers are explanatory diagnostics, not a newly preselected test.

## Slide 50: reproduction and interpretation

The raw observed root is 2.808645. Its conventional standard-normal local tail is 0.00248752. Under the frozen response source, the corresponding marginal Gaussian tail is 0.0113826. The saved raw-global tail is 50,183/200,000, or 0.250918745 with the stated add-one convention. The complete Poisson-rescan check has 57/256 exceedances, with Clopper–Pearson 95% interval [0.173218,0.278639]. These quantities were independently recomputed directly from the pinned NPZ, rather than copied from the study summary.

The ratio global/marginal≈22 compares probabilities from the same response source. The ratio global/conventional-local≈101 mixes a source-conditioned global calibration with a nominal standard-normal marginal interpretation; it is not a pure look-elsewhere factor. Neither ratio is a universal effective number of trials: the field is nonstationary and the result depends on the threshold and mass coordinate.

The paired Gaussian controls separate mechanisms without changing the observed threshold. Removing a changes the global tail from approximately 0.2511 to 0.1622; setting s=1 gives 0.2826; centering and setting unit scale gives 0.1878. Even the last control retains a typical positive scan maximum (median 2.338). These comparisons are conditional counterfactuals, with interactions between mean, scale and correlation; they are not additive contributions or replacements for production probabilities. Independent-node and perfectly correlated controls illustrate correlation sensitivity. In particular, independence is not a general upper bound when the actual field contains negative correlations.

For 2021 the simultaneous response-model mean check gives max|z|=2.433 against a 95% critical value 3.435 (conditional Monte Carlo p≈0.557). This supports the adequacy of a as a mean-response approximation within the precision of 256 complete scans. It does not prove exact centering, validate the generating source, or establish a rare discovery tail. Marginal Student-t and chi-square width intervals also rely on approximate normality.

There is a **localized 2016 width tension at 43.5–44 MeV**. The direct SDs are 1.12989 and 1.11759, compared with response scales 0.95489 and 0.95380 (+18.3% and +17.2%). The per-scope Bonferroni normal-theory intervals [0.96648,1.34906] and [0.95596,1.33438] exclude the response scale. Adjacent nodes share toys and fitting windows, so this is one correlated region rather than two independent failures. It prevents a blanket all-dataset width-closure claim. It alone does not identify a code error or invalidate the separately checked 2021 tail.

## Slide 14: the correct repeated-sampling comparison

Let S denote held-out window bins, T the disjoint training bins, and B the fixed Poisson mean. Write e=Y_S−b̂(Y_T), δ=B_S−b̂(B_T), and J=∂b̂/∂Y_T at B. With independent Poisson bins, the first-order repeated-sampling covariance is

\[
W=\operatorname{diag}(B_S)+J\operatorname{diag}(B_T)J^T.
\]

For fixed V=diag(b̂(B_T))+C_GP and Q=eᵀV⁻¹e,

\[
E[Q]\simeq\operatorname{tr}(V^{-1}W)+\delta^TV^{-1}\delta.
\]

This identity is exact when δ is the actual mean residual and W its actual covariance, with fixed V. Using b̂(B_T) and a first-order J makes it an approximation to the nonlinear estimator. When V is recomputed for each toy, E[eᵀV(Y_T)⁻¹e] requires the joint distribution; replacing V by an average is not an exact calculation.

The GP posterior covariance C_GP is not generally the repeated-Poisson sampling covariance of the sideband predictor at a fixed truth. It includes the model's conditional function uncertainty. Consequently Q/Nbin=1 requires matching residual covariance and centering; it is not guaranteed by the GP posterior. Nbin is a count of held-out bins, not an inferred effective number of degrees of freedom. The archived slide's “unit reference only” qualification is appropriate.

The audited residual implementation uses disjoint masks, the right count-space covariance, and a Jacobian including both alpha=1/y and the lognormal mean correction. Four directional finite-difference checks have relative L2 error at most 9.51×10⁻⁷. The analytic mean Q/Nbin ranges from 0.9976 to 1.1848 across the 201 integer masses. The repeated-sampling variance term is close to unity (0.9879–0.9986); the largest deterministic bias contribution is 0.1957 at 50 MeV. At 78 MeV the expected mean is 1.0262, while the observed single-spectrum value is 2.3300. The latter is a fluctuation at a selected coordinate, not an estimate of the null mean. A high or low segment of one observed curve cannot establish a systematic estimator bias.

The paired frozen-prediction, linear-refit, exact fixed-kernel refit, fixed-V, adaptive-V, and deterministic-bias-subtracted controls are the appropriate way to separate these mechanisms. All 201 masses and eight controls now have 256 finite entries. Independently checked CSV means exactly match the arrays; the fixed-V sample covariance-plus-mean identity agrees to 6.7×10⁻¹⁶. The maximum paired exact-minus-linear mean change is 0.000214, and the maximum adaptive-V-minus-fixed-V change is 0.000951 in Q/Nbin. These effects are small under the tested source.

A positive empirical squared mean shift is not by itself evidence of Jensen bias: its finite-Monte-Carlo expectation contains a covariance/N contribution. The final specialist output explicitly subtracts that estimated positive floor in a separate descriptive ledger and allows the corrected estimates to be negative; the uncorrected sample-mean terms remain labeled as such.

The connection to the likelihood root is only a projection. In a Gaussian approximation with signal vector w,

\[
r\simeq\frac{w^TV^{-1}e}{\sqrt{w^TV^{-1}w}},\quad
E[r]\simeq\frac{w^TV^{-1}\delta}{\sqrt{w^TV^{-1}w}},\quad
\operatorname{Var}(r)\simeq\frac{w^TV^{-1}WV^{-1}w}{w^TV^{-1}w}.
\]

Q includes all residual directions, whereas r measures the direction aligned with a signal. This is why a nearly unit average Q/Nbin does not imply zero signal-like response, and a displaced scan-maximum distribution does not imply Q is incorrectly normalized. At 78 MeV the GLS projection is 0.576797 versus the saved profiled Asimov root 0.576784. The total deterministic bias power is 0.487138, with 68.30% in that signal direction. The full posterior covariance versus the likelihood's numerical covariance conditioning changes the GLS root by no more than 4.44×10⁻⁸ over the tested grid.

An independently reproduced exploratory residual-scan maximum has 20/256 exceedances of the observed 2.329997, giving add-one tail 0.081712 and Clopper–Pearson interval [0.048372,0.118080]. This refers to max Q/Nbin over 201 mass hypotheses. It is not a resonance significance, an independent background validation, or an unconditional goodness-of-fit probability.

## Source, units and scope checks

- The null source's observed counts and bin edges exactly equal the pinned v5.0.5 spectrum; the response field's D columns have norm s and its stored K is the correlation DᵀD/(s sᵀ), not raw covariance.
- The nominal fit support is 36–300 MeV; the retained whole-bin edges are exactly 36.0–299.75 MeV with 0.625 MeV bins. The blind half-width remains ±2.25 sigma. The code converts tested mass from MeV to GeV before comparing to the GeV-valued bin centers and resolution.
- Slide14 uses a 1 MeV mass grid and full posterior count covariance. Slide50 uses a 0.5 MeV scan and the likelihood's numerically loaded/truncated covariance factor. Do not conflate their mass grids or claim exact equality of Q and the profiled statistic.
- “Refit” in this audit means recomputing the conditioned GP, including its data-dependent noise, and local likelihood under the frozen archived kernel policy. It does not mean rerunning hyperparameter optimization.
- The generating mean was learned from the observed spectrum and then frozen. All toy checks condition on that source and those analysis choices. Their Monte Carlo intervals omit source-estimation and model-family uncertainty; passing them does not establish unconditional discovery probabilities or confidence-limit coverage.

Reproducible independent input checks: `review/independent_checks.py` and `review/independent_input_checks.json`. Audited scientific implementations: `residual_diagnostic/run.py`, `scan_maximum/scripts/audit.py`, the pinned v5.0.5 `scripts/common.py`, and v5.8.2 `scripts/engine.py` / `scripts/analyze.py`.

Final numerical cross-check of the completed residual outputs is recorded in `review_checks.json`; all deterministic identities pass. As additional independent controls, the exact Gaussian independent-node tail is 0.675160 and the exact perfectly correlated tail is 0.0680665. The simulations differ by −1.16 and −0.62 Monte Carlo standard errors, respectively.

The final `source/report.tex` and its counterfactual, residual, residual-findings and cross-scope table fragments were checked against the scientific outputs. No scientific blocker was found. The distinction between fixed-source statements and physical-source qualification is maintained throughout. The report's PDF rendering and package checks are owned by the root agent.
