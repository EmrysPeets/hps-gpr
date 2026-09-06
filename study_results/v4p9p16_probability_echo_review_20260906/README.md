# v4.9.16: probability audit and signal echoes

The extraction fits did not change the probabilities. All 232 masses were evaluated, and all 15 displayed likelihood roots match the saved scans to below 1e-15. The earlier Figure 1 mixed a nominal asymptotic local reference with a conditional calculation against a particular background spectrum. The sign gate, large background offsets and unresolved Monte Carlo tails explain its abrupt changes and apparent gaps.

The revised Figure 1 retains the complete observed upper limit and shows the signed root plus nominal local probability from those same fits. The v4.9.12 legend entry and the archived-stress subtitle are removed. Separate full-range and zoom figures explain the background-reference probability calculation and show all direct-toy checks with intervals or zero-count upper bounds. No probability rule, fitted value or limit has been altered. The Gaussian local approximation describes fluctuations of the fit statistic, not a Gaussian-shaped mass spectrum.

At 66 MeV the largest positive combined root is 2.760, with nominal local p0=0.002889. At 76 MeV the root is only 0.166; subtracting the reference offset -8.700 produces a formal Gaussian tail of 6.98e-20. That number is not a validated particle significance. Zero of 1,000 direct scans supports only a one-sided 95% sampling bound of 0.002991 under the specified background; the 200,000 GP fields describe an additional approximation.

A replay with the current dense 2021 solver confirms that positive signals can create neighboring positive and negative fitted echoes as the background-training mask moves. The 116 deterministic fits reuse the saved smooth reference and injected templates. No new random toys or unblinded events were used. A selected two-peak injection produces a deep dip but overshoots the observed peaks; it is not evidence for two particles. The note explains why both a parent peak and its echoes may grow with data, and why the independent additional 20% should be tested before cumulative 30%.

## Deliverables

- Full LaTeX report: `note/analysis_note.tex`.
- PDF: `output/pdf/v4p9p16_probability_echo_review_20260906/HPS_GPR_Analysis_Note_v4p9p16_Probability_Audit_and_Echoes.pdf` (repository-relative).
- Revised Figure 1: `figures/combined_observed_limit_and_pvalues.pdf` and PNG.
- Conditional probability diagnostics: `figures/probability_reference_full.pdf` and `probability_reference_zoom.pdf`.
- Current-solver echo figure: `figures/signal_echo_dense_replay.pdf`.
- All-mass probability ledger: `derived/probability_grid.csv`.
- Native deterministic fit components: `derived/echo_likelihood_components.npz`.
- Independent HEP review: `review/HEP_PROBABILITY_ECHO_REVIEW.md` and associated audit outputs.

The full note retains the earlier extraction displays, deficit study, staged-exposure discussion and completed 2015 low-mass side study. Their source products remain frozen. There is no final calibrated global particle probability, expected-sensitivity result, or newly calibrated full-union limit in this revision.

## Reproduction

Run from the repository root:

```bash
python3 -B study_results/v4p9p16_probability_echo_review_20260906/analyze.py
python3 -B study_results/v4p9p16_probability_echo_review_20260906/make_figures.py
python3 -B study_results/v4p9p16_probability_echo_review_20260906/make_report.py
python3 -B study_results/v4p9p16_probability_echo_review_20260906/validate_products.py
```

`analyze.py` validates frozen inputs, recounts stored Monte Carlo samples, and performs only deterministic echo fits. It does not regenerate the global ensemble. The report build requires cached Tectonic resources. A rebuild changes PDF bytes and requires refreshed rendered-page QA before resealing. Audit scripts under `review/` independently check saved products and print JSON; they do not regenerate simulations. `MANIFEST.csv` binds the delivered source, numerical products, figures, PDF and QA.
