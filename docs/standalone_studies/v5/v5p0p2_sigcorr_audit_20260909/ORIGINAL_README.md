# Analysis note v5.0.2: sigcorr audit revision

This revised copy preserves the earlier 9 September v5.0.2 delivery and all frozen observed results. Section 6.11 adds the executable sigcorr comparison, resolves the remaining conditional Gaussian tails, and tests the 2016 stress-source construction. This is a review draft, not unblinding approval.

The official sigcorr commit is a66c08771a81230c67755f2ca6fb902cb7a8ef66 (software version 5.0.1). Its covariance normalization and Gaussian factors agree with the saved HPS matrices to floating-point precision. The HPS nonzero stress offset, width standardization and positive-fit gate are explicit extensions of the paper's centered null field. Their probabilities have not been qualified as particle-discovery significances.

All eight previously empty Gaussian tails now have estimates. Two independent sets of 20,000 conditional-mixture fields per selected threshold resolve 23 tails, with eight ordinary-Monte-Carlo overlap checks. Relative MC standard errors are below 0.33%. The run took 3.1 seconds and 207 MiB on one linear-algebra thread. The 1,000-scan direct Poisson tails remain unresolved. The new method integrates the frozen finite-grid Gaussian model; it is separate from sigcorr's upcrossing routines.

Three deterministic 2016 source controls were scanned at all 142 masses using exact frozen-kernel fits. Removing the low-mass component reduces some offsets but worsens the full-grid RMS from 4.967 to 9.430. Integrating the blend inside each bin changes the signed response by at most 0.001135. Neither control warrants replacing the background. Two bounded diagnostic passes were made, about 39 and 36 seconds; the final timed pass used 363.1 MiB, one worker, and no new Poisson toys.

## Files and reproduction

- `pdf/HPS_GPR_Analysis_Note_v5p0p2_Unblinding_Review_Draft.pdf`: final reviewed document.
- `HPS_GPR_v5p0p2_Review_Draft_Source.zip`: portable document, figures, numerical ledgers, source code and provenance.
- `derived/v502_refined_gaussian_tails.csv`: targeted estimates, pointwise approximate 95% MC intervals and analytic union bounds.
- `derived/v502_global_display_*.csv`: plotted probabilities; ordinary counts are explicitly named separately from targeted estimates.
- `provenance/sigcorr`: pinned unmodified relevant official modules, license, source manifest and original crosschecks.
- `provenance/stress_2016_control`: full paired-control results, scripts, source hashes and numerical gates.

Use `bash scripts/build_note.sh` to build with cached Tectonic, or omit its `-C` option if the TeX bundle has not been cached. Python dependencies are NumPy, SciPy, pandas, matplotlib, pypdf, PyMuPDF and Pillow. No sigcorr installation is needed for `python3 scripts/check_sigcorr.py`.

`python3 scripts/refine_gaussian_union.py` validates and reuses the targeted checkpoints. `python3 scripts/make_v502_figures.py` followed by `python3 scripts/make_audit_displays.py` regenerates the probability and diagnostic figures from bundled inputs. `python3 scripts/validate_v502.py` followed by `python3 scripts/validate_sigcorr_revision.py` runs document/numerical checks in the HPS authoring checkout. The validation scripts intentionally verify preserved parent files and checkout state. Re-running the exact 2016 fits requires that original checkout and its input ROOT files; the portable document build and figure regeneration do not.

The parent document is bound in `provenance/v502_parent.json`. `qa/final_validation.json`, `qa/visual_review.json`, `qa/portable_build.json`, and `MANIFEST.json` bind checks to the delivered PDF. The inherited Tectonic/BibTeX rerun warning is tracked; resolved references and portable page text are checked directly.
