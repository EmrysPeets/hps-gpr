# 2015 Closure Diagnostics Handoff

## Purpose

This document summarizes the 2015 functional-form closure studies discussed in this thread, the current interpretation of the pull-vs-mass and `\Delta Z` plots, the reporting bugs that were confirmed and fixed, and the remaining physics concern.

The short version is:

- The no-refit functional-form closure study looks healthy.
- The refit studies do not close.
- Widening the excluded training region from `1.64σ` to `1.98σ` helps, but does not fix the problem.
- Turning GP hyperparameter optimization off during toy refits does **not** materially change the result.
- Switching multinomial to poisson injection does **not** materially change the nonzero-injection refit result.
- A `0σ` refit-poisson study behaves near zero bias, so the failure is not obviously a generic extraction or plotting catastrophe.
- Static code audit did **not** find evidence that the blind region is being used in the refit GP training.
- Static code audit did **not** find a template-normalization mismatch between injection and extraction.
- The refit failure is therefore still physically alarming. It may be leakage plus uncertainty inflation, but a deeper diagnosis is still warranted before treating the refit closure study as understood.

## Definitions

- `pull = (\hat{A} - A_{inj}) / \sigma_A`
- `\hat{Z} = \hat{A} / \sigma_A`
- `\Delta Z = \hat{Z} - Z_{inj}`
- `\sigma_{A,\mathrm{ref}}` is the background-only reference uncertainty used to define sigma-scaled injections.
- `Z_{inj}` or `inj_nsigma` means `A_{inj} / \sigma_{A,\mathrm{ref}}`, not `A_{inj} / \sigma_A`.
- `no refit` means the GP blind-window background prediction is held fixed toy-by-toy.
- `refit` means a full toy is generated and the GP is retrained on the toy sidebands before extraction in the blind window.
- `optimize=true` means the GP hyperparameters are re-optimized for each toy sideband fit.
- `optimize=false` still refits on each toy sideband sample, but disables hyperparameter optimization and uses the configured kernel seed directly.
- `restarts` only matters when `optimize=true`; it controls additional optimizer starting points.
- `multinomial` injection fixes the total injected signal yield and fluctuates only the bin allocation.
- `poisson` injection fluctuates the total injected signal yield as well.
- The `0σ` study is a background-only refit closure check. It is useful for spurious-signal style sanity checking, but it is not a substitute for signal-injection closure.

## Study Inventory

Main 2015 folders examined:

- `/Users/emryspeets/Desktop/gp_mods/2015_gpr/closure`
- `/Users/emryspeets/Desktop/gp_mods/2015_gpr/closure_refit`
- `/Users/emryspeets/Desktop/gp_mods/2015_gpr/closure_164_noopt_refit`
- `/Users/emryspeets/Desktop/gp_mods/2015_gpr/closure_198_refit`
- `/Users/emryspeets/Desktop/gp_mods/2015_gpr/closure_refit_poisson`

Important data-shape note:

- `closure`, `closure_refit`, `closure_164_noopt_refit`, and `closure_198_refit` each contain `2800` rows in `inj_extract_summary_2015.csv`.
- Those files are **not** compact summary tables in the usual sense. They are fragmented one-toy rows written with summary-style columns, with repeated `(dataset, mass_GeV, inj_nsigma)` groups and `n_toys = 1`.
- `closure_refit_poisson` is mixed-format:
  - `0σ` appears as a proper aggregated `100`-toy summary row per mass.
  - nonzero injections are again fragmented one-toy summary rows.
  - `2σ` is absent from this folder.

Reference outputs refreshed during this implementation:

- `pull_vs_mass_2015.png`
- `z_calibration_residual_2015.png`
- `delta_z_minus_pull_vs_inj_sigma_all.png`

These now exist in all five 2015 closure folders above.

## Quantitative Comparison

Mass-averaged summaries by injected sigma level:

| Study | Key result |
| --- | --- |
| `closure` | Healthy closure. `pull_mean ~ -0.02`, `\Delta Z ~ -0.03`, `\sigma_A / \sigma_{A,ref} ~ 1.00`, `Ahat/Ainj ~ 0.98-1.00`. |
| `closure_refit` | Severe failure. `pull_mean ~ -0.54, -1.13, -1.71, -2.86` for `Zinj = 1,2,3,5`. `\Delta Z ~ -0.89, -1.82, -2.78, -4.69`. `\sigma_A / \sigma_{A,ref} ~ 1.56-1.62`. `Ahat/Ainj ~ 0.10-0.18`. |
| `closure_164_noopt_refit` | Essentially identical to `closure_refit`. Turning optimization off did not materially improve closure. |
| `closure_198_refit` | Improved signal recovery but still not closed. `pull_mean ~ -0.28, -0.72, -1.13, -1.99`. `\Delta Z ~ -0.73, -1.63, -2.53, -4.40`. `\sigma_A / \sigma_{A,ref} ~ 1.89-2.00`. `Ahat/Ainj ~ 0.25-0.50`. |
| `closure_refit_poisson` | Matches the regular refit for nonzero injections. `0σ` is near unbiased: `pull ~ 0.045`, `Z ~ 0.045`, but still with inflated `\sigma_A / \sigma_{A,ref} ~ 1.52`. |

Leakage proxy from the stored `f_train_frac` column:

- `1.64σ` training exclusion studies: `f_train_frac ~ 0.0955-0.1069`
- `1.98σ` training exclusion study: `f_train_frac ~ 0.0438-0.0507`

Interpretation:

- Moving from `1.64σ` to `1.98σ` roughly halves the leaked signal fraction in the training sidebands.
- That substantially improves `Ahat/Ainj`.
- But `\sigma_A` inflates even more in the `1.98σ` refit, so `\Delta Z` remains strongly negative.
- The failure is therefore not just “signal disappears”; it is “signal recovery is poor and the fitted uncertainty also blows up.”

## What The Pull And `\Delta Z` Plots Are Showing

How to read the two diagnostics:

- The pull-vs-mass plot is sensitive to bias relative to the fitted toy uncertainty.
- The `\Delta Z` plot is sensitive to recovered significance relative to the injected sigma level defined with `\sigma_{A,ref}`.

In the bad refit studies, both plots move in the wrong direction, but not by exactly the same amount. That is expected because they use different denominators:

- pull uses toy-level `\sigma_A`
- `Z_{inj}` uses `\sigma_{A,ref}`

The important consistency check is that the measured `pull`, `Ahat/Ainj`, `\sigma_A/\sigma_{A,ref}`, and `\Delta Z` satisfy the expected algebra approximately:

- `pull ~ ((Ahat/Ainj - 1) * Zinj) / (\sigma_A/\sigma_{A,ref})`
- `\Delta Z ~ (Ahat/Ainj * Zinj) / (\sigma_A/\sigma_{A,ref}) - Zinj`

That consistency holds well enough across the refit studies that the main pull-vs-mass and `\Delta Z` plots are unlikely to be grossly misplotted. In other words, the refit failure looks internally self-consistent, not like a random plotting accident.

## Static Code Audit Conclusions

### Things that look correct

- The blind region is **not** used in the refit GP training mask.
  - The training mask is built in `hps_gpr/injection.py` in `_build_injection_mass_context`.
  - The actual GP refit uses `X_tr = ctx.x_full[ctx.msk_train]` and `y_tr = y_toy[ctx.msk_train]` in `_simulate_toy_rows_chunk`.
- `inj_refit_gp_optimize=false` still performs a toy-by-toy sideband refit.
  - It only changes `fit_gpr(..., optimize=False)`, which disables the sklearn optimizer by setting `optimizer=None`.
- The template normalization convention is internally consistent.
  - `build_window_template_from_full` returns a full-range normalized template plus an unrenormalized blind-window slice.
  - The extraction fit uses the same blind-window slice convention.
- The nonzero refit-poisson results matching the regular refit, and the no-opt refit matching the optimized refit, both argue against the optimizer or the injection mode being the dominant explanation.

### Things that are still physically concerning

- The background-only profiled fit appears to absorb far more signal than expected in the refit studies.
- The amount of suppression is largest at low mass and high injected signal.
- The `1.98σ` study helps but does not restore closure.
- A pure static audit did not expose a definitive extraction bug, but it also did not explain why the effect is this large.

Current best statement:

- No definitive core extraction bug was proven by static audit.
- The observed refit behavior remains physically alarming and should be treated as unresolved.

## Confirmed Reporting / Plotting Bugs

These issues were confirmed and fixed in the code:

1. Fragmented summary rows were being treated as already-aggregated summaries for `\Delta Z` plotting.
   - Fixed by regrouping repeated `(dataset, mass_GeV, inj_nsigma)` rows before building `z_calibration_residual`.

2. `delta_z_minus_pull` was computed incorrectly.
   - Old behavior: `Zhat_mean - pull_mean`
   - Correct behavior: `(Zhat_mean - inj_nsigma) - pull_mean`

3. The pull-vs-mass legend title was misleading.
   - Old label implied `Ainj / sigma_A`
   - Correct label is `Ainj / sigma_Aref`

4. `pull_hist_2015` grouping was wrong for sigma-scaled injections.
   - Old behavior grouped by fluctuated raw `strength`, creating one PNG per nonzero amplitude.
   - Correct behavior groups by `(mass, inj_nsigma)` when sigma-scaled injections are present.

5. The `inject-plot` summary-only flow was discarding fragmented one-toy summary rows.
   - Fixed by not deduplicating those rows away before regrouping.

Important limitation:

- The current closure folders do **not** contain toy-level `inj_extract_toys_2015.csv` files.
- That means the legacy `pull_hist_2015` directories cannot be cleanly regenerated from the data now on disk.
- Those histogram directories should therefore be treated as legacy outputs, not as freshly validated diagnostics.

## Recommended Next Studies

1. Keep `blind_nsigma = 1.64` and widen only the GP training exclusion.
   - This isolates leakage control from changes in the extraction definition.

2. Compare `optimize=true` vs `optimize=false` at fixed masks.
   - This is still useful as a cross-check, even though the existing `closure_164_noopt_refit` result suggests optimizer motion is not the main culprit.

3. Tighten the GP smoothness floor.
   - The current lower bound allows relatively short correlation lengths and may make absorption easier.

4. Standardize study outputs.
   - Every closure folder should contain one compact summary CSV.
   - When available, also keep one toy-level CSV.
   - Avoid mixed-format files where `0σ` is aggregated but nonzero injections are fragmented one-toy rows.

5. Recover or regenerate toy-level tables if pull histograms are needed for final documentation.

## Code Changes Made In This Implementation

- `hps_gpr/injection.py`
  - corrected `delta_z_minus_pull` to use true `\Delta Z`

- `hps_gpr/plotting.py`
  - regroup duplicated one-toy summary rows for `z_calibration_residual`
  - recompute `delta_z_minus_pull` from `Zhat_mean`, `inj_nsigma`, and `pull_mean` when possible
  - regroup fragmented summary rows for `pull_vs_mass`
  - relabel injected sigma level as `Ainj / sigma_Aref`
  - group pull histograms by `inj_nsigma` instead of raw `strength` when sigma-scaled injections are present

- `hps_gpr/cli.py`
  - preserve fragmented one-toy summary rows in summary-only `inject-plot` mode instead of deduplicating them away

- tests added or updated
  - fragmented summary-row regrouping
  - corrected `delta_z_minus_pull`
  - sigma-level pull-hist grouping
  - summary-only `inject-plot` smoke coverage
