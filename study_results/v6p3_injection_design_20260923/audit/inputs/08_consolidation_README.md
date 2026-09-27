# v4.9.1 2021 background-validation consolidation

This directory is a provenance-complete continuation of the v4.9 threshold work. It extends two 65 MeV injection--extraction ensembles to 100 independently seeded background spectra, recomputes all displayed intervals at 90% confidence, archives the historical v4.6 full-100 inputs used elsewhere in the consolidated figures, and provides revised Figure 64, the requested two-by-four mean/width summary, and a 1% source times ten Figure 65 diagnostic.

The requested production is complete. Under the practical centering criterion adopted after the full-100 results, `abs(mean pull) < 0.5`, both replacement ensembles pass and both zero-signal 90% mean intervals lie wholly inside the band. The Table-17 1% source times ten result is also compatible with zero, while the native-10% `fSigPowExpQ`-anchored 30--300 MeV result retains a statistically resolved negative offset. These are reported as different statements rather than allowing either criterion to erase the other.

## Result at the zero-signal endpoint

All intervals below are two-sided 90% intervals. Mean intervals use Student's t distribution. Width intervals use the normal-theory chi-square construction for a sample standard deviation; they are not an exact calibration or coverage statement.

| lane | cohort | n | mean pull (90% t interval) | sample width (90% normal-theory interval) | two-sided t p-value | recorded mean-bias screen |
|---|---:|---:|---:|---:|---:|---:|
| 1% source times ten, Table-17 truth, 40--300 MeV | full | 100 | +0.139 [-0.050, +0.329] | 1.140 [1.022, 1.293] | 0.224 | pass |
| 1% source times ten, Table-17 truth, 40--300 MeV | independent continuation | 80 | +0.099 [-0.115, +0.314] | 1.151 [1.020, 1.326] | 0.443 | pass |
| native 10%, `fSigPowExpQ`-anchored truth, 30--300 MeV | full | 100 | -0.246 [-0.417, -0.076] | 1.025 [0.919, 1.162] | 0.0182 | fail |
| native 10%, `fSigPowExpQ`-anchored truth, 30--300 MeV | independent continuation | 75 | -0.284 [-0.494, -0.074] | 1.093 [0.965, 1.266] | 0.0275 | fail |

The recorded mean-bias screen flags a cell only when both `abs(mean pull) >= 0.2` and the 90% mean interval excludes zero. The user subsequently adopted the gray +/-0.5 band as the practical background-model-bias criterion. Because that margin was chosen after the full-100 values were available, it is descriptive rather than a predeclared equivalence test. It establishes practical centering for both replacements without erasing the resolved native-10% offset under the more sensitive zero-null screen.

For injected strengths z = 1, 3, and 5, the Table-17 full-100 mean intervals also include zero. The native-10% full-100 mean intervals exclude zero at every displayed strength. The complete full, initial, and independent-continuation results are in `derived/new_65mev_full_and_reserve_moments_90cl.csv` and summarized in `STATUS.md`.

## Scientific interpretation

The Table-17 study does meaningfully test whether the GP can model a data-like, low-chi-squared-per-degree-of-freedom threshold distribution under the specified card and recover injected signals. It is therefore more than a toy-form comparison. Its restricted functional truth is appropriate for a threshold-focused validation because the fitted log-mass length scales imply very small direct RBF correlations between 65 and 150 MeV.

That statement should not be strengthened to say that the high-mass tail has literally no estimator influence. At fixed selected hyperparameters, removing training bins above 150 MeV shifts the zero-signal pull by a mean of -0.1949 for native 10% and -0.1304 for 1% source times ten. The tail diagnostic's machine-readable `status: pass` means only that execution and full-fit reconstruction QA passed. It is not a tail-stability pass.

The Table-17 means support threshold-region centering for that conditional truth. Its full-100 sample widths are about 1.14 and their normal-theory intervals lie above one, so this study should not be described as exact pull-width calibration or frequentist coverage. The native-10% result remains a conditional mean offset under the recorded screen but lies inside the adopted practical band. Across the substituted composite, all 20 zero-signal point estimates and their 90% intervals are inside +/-0.5. The bundle therefore supports an operational statement that practical background-model-mean-bias validation is complete for the explicitly labelled composite suite. A scale-up or card-freeze decision should preserve the truth, support, optimizer-gate, post-result tolerance, width, response, and tail qualifications.

## Production integrity

- Native 10%: 25 archived pilot toys plus an independently seeded continuation of 75, with 400 of 400 strength-by-background extraction rows accepted, 610 optimizer attempts, zero technical exclusions, and no accepted kernel-bound contacts.
- 1% source times ten: 20 archived Table-17 pilot toys plus an independently seeded continuation of 80, with 400 of 400 extraction rows accepted, 614 optimizer attempts, zero technical exclusions, and no accepted kernel-bound contacts.
- The first 25 native toys and first 20 Table-17 toys were regenerated bit-identically from their original analytic means, base seeds, and seed namespaces.
- Their accepted extraction rows are bit-identical to the archived rows in pull, fitted amplitude, amplitude uncertainty, GP log marginal likelihood, length scale, constant, selected attempt, optimizer seed, and attempt count.
- Optimizer selection used pull-blind log-marginal-likelihood, reproducibility, covariance, and boundary gates. No rows were excluded or repaired using pull information.
- The minimum accepted count is 100 in each mass/strength cell, exceeding the declared minimum of 95.

## Main artifacts

- `FIGURE_CAPTIONS.md`: exact, paste-ready captions and scope notes.
- `STATUS.md`: complete numerical result inventory and release decision.
- `COMPLETED.md`: checklist of finished work.
- `REMAINING_STEPS.md`: bounded work needed before a stronger validation/freeze claim.
- `PROVENANCE.md`: source, seed, hash, runtime, and ledger chain.
- `METADATA_ERRATA.md`: interpretation of inherited hash-bound wording.
- `derived/new_65mev_full_and_reserve_moments_90cl.csv`: new-lane moments for initial, independent-continuation, and full samples.
- `derived/historical_v4p6_moments_recomputed_90cl.csv`: historical full-100 moments recomputed at 90% confidence.
- `derived/consolidated_pull_moments_90cl.csv`: exact source table for the consolidated plots.
- `derived/analysis_figure_manifest.json`: hash manifest for all figures and plotted tables.
- `qa/release_validation.json`: fail-closed release validation report.
- `qa/toy_extension_identity.json`: initial-toy identity audit.
- `qa/tail_above150_fixed_hyperparameter_influence_summary.json`: fixed-selected-hyperparameter tail-influence diagnostic.
- `reference/v4p6_full100/`: archived historical specification, runner, accepted rows, closure summary, exclusion ledger, and collection summary.

The `note/` subtree contains the integrated v4.9.1 analysis-note source, final PDF, and render QA. It is included in the final top-level release manifest produced after note validation.

## Reproduction and validation

Run commands from this directory.

```bash
python3 build_continuation_toys.py
python3 run_native10_fsig_100.py preflight
python3 run_1pctx10_table17_100.py preflight
python3 run_native10_fsig_100.py collect
python3 run_1pctx10_table17_100.py collect
python3 analyze_and_plot.py
python3 tail_influence_diagnostic.py
python3 validate_release.py
python3 build_release_manifest.py
```

The per-toy `_SUCCESS.json` products under `runs/` bind task outputs to the study specification, input ROOT/manifest hashes, and the raw/accepted/attempt ledgers. Production should not be rerun after changing either hash-bound study specification; a semantic wording correction belongs in a new declared study revision.
