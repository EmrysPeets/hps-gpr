# v4.9.13 fixed-GP observed extension: pre-execution protocol

Frozen before numerical execution on 5 September 2026. This isolated extension
does not modify or replace the v4.9.12 release or certify its 2016 state.

1. Read the native ROOT histograms and exactly reconstruct the reviewed GP
   states through `v4p9p12_final_dataset_combinations_20260902/run_final_combinations.py`.
   Match every prediction SHA-256 to the released ledger, preserving native
   mass coordinates, signal templates, moving masks, kernel coordinates,
   covariance conditioning, and yield conversion. No optimization of GP states.
2. Evaluate full 2015 (19--90 MeV), full 2016 (39--180 MeV), 2021 10%
   (50--250 MeV), and all three jointly (50--90 MeV), on 1 MeV grids: 456 rows.
   The 2016 numerical exception is inherited by 2016 and the combination.
3. The fixed model is independent Poisson counts with expectation
   `lambda_i = b_GP,i + epsilon^2 s_unit,i`. The GP mean is treated as known;
   this removes its uncertainty and is a conditional diagnostic. Fit a shared
   epsilon squared in the combination by concatenating bins and their exact
   counts-per-epsilon-squared signal templates. Do not combine limits or p-values.
4. Use the previously validated, centered Poisson `Profile` solver from
   `background_profile_comparison_20260905/run_comparison.py`, with zero
   background-nuisance columns. The reported limit is the 90% bounded,
   piecewise-asymptotic CLs upper limit. The unbounded signed fit is used only
   as a diagnostic; discovery uses `Z = max(r, 0)` and `p0 = Normal.sf(Z)`.
   Retain log p0. No trials correction or ad hoc significance scaling is used.
5. Compare to the exact released observed limit and local-asymptotic p0 columns.
   Recompute all 201 existing 2021 fixed results and require closure.
6. Also evaluate deterministic background-only Asimov limits under the fixed
   and correlated-Gaussian-profile models, using the same stable solver and
   identical GP mean. These isolate the effect of removing GP uncertainty
   conditional on this background. They are not new expected-toy bands or
   calibrated sensitivity estimates.
7. Validate fixed fits independently using the scalar Poisson score root,
   independent scalar likelihood ratios and piecewise normal-tail evaluation.
   Check positivity, likelihood nesting, roots, exact grid, shared normalization,
   source immutability, and once-only displayed dimuon conversion.
8. Plot four-scope observed limits with fixed/released ratios, corresponding
   local-asymptotic p-values, and paired conditional Asimov limits. All displayed
   epsilon-squared curves receive the expanded note's dimuon branching factor;
   the electron-channel raw values remain in the CSV. p-values are unchanged.
9. Run one single-threaded process at nice +10. Store code, data, figures,
   numerical checks, provenance, and conclusions only in this directory.

The interpretation must distinguish observed tightening from sensitivity or
coverage. A smaller local p0 after discarding uncertainty is not evidence for
discovery. The parent note contains separate injection and extraction controls.
