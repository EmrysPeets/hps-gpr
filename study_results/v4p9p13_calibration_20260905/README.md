# v4.9.13: conditional calibration of Gaussian-profiled and fixed GP limits

**Status: all 456 calibration coordinates complete.** The combined scan has 41/41 Monte Carlo-resolved endpoints for both methods. The machine-readable summary and note report the final individual precision counts separately; finite Monte Carlo qualifications remain visible.

The target is 90% CLs (CLs=0.10), with 2015 full, 2016 full, native 2021 10%, and their exact shared-epsilon-squared combination. The grids contain 72, 142, 201 and 41 points, respectively. The previous background comparison, fixed observed scans, injection tests and v4.9.13 note were already published on `codex/v4p9p13-background-calibration`, through commit 4f52e0cee2fdca645ae4b9c680df16840d31900b. This directory extends that study while preserving its outputs.

## What is calibrated

Full-support Poisson pseudoexperiments rerun sideband log-count GP training with count-dependent errors and full signal tails. Both background likelihoods use identical spectra and the bounded one-sided likelihood-ratio statistic. Empirical CLs replaces asymptotic tails; exact Poisson mixture weights reuse fits across tested signal yields. Independent direct-Poisson validation uses 500 spectra per generating truth at each of 0, 2 and 5 fixed-reference-sigma signal yields. No expected-limit bands or global significance are inferred.

This is conditional on reviewed observed kernels, support, masks, resolution and yield conversion. Hyperparameter optimization and historical support/model selection are not repeated. The archived stress truths retain their source-fit limitations. The combination envelopes two joint scenarios (all local-GP truths or all archived stress shapes), not all mixed assignments. The 2016 numerical exception and common-systematic qualifications remain. The original historical closure fits reoptimized kernels and used per-toy reference yields, so their mean pulls are not interchangeable with these fixed-yield ensemble offsets. See `history_review.md` and the note.

A fixed-background likelihood can serve as a calibrated test statistic. Its uncorrected small curvature error is not evidence that the estimated background is known. Compare the calibrated observed limits and independent injection decisions before interpreting the raw improvement. A mass-search trials correction does not repair a wrong local sampling distribution.

## Final results

| Scope | Grid points | Profile resolved | Fixed resolved | Median calibrated fixed/profiled |
|---|---:|---:|---:|---:|
| 2015, 100% | 72 | 70 | 72 | 1.446 |
| 2016, 100% | 142 | 133 | 141 | 1.333 |
| 2021, 10% | 201 | 201 | 200 | 1.199 |
| Combined | 41 | 41 | 41 | 1.346 |

All 456 pairs have finite endpoints. Thirteen individual method/mass endpoints retain a limited-MC marker; the entire combined scan meets the stated endpoint precision gates. The independent validation suite has zero Holm-adjusted excess-exclusion flags among 3,648 tests and zero excess-local-rejection flags among 1,824 tests. This finite-power screen does not certify percent-level or unconditional coverage. See `NEXT_STEPS.md` for the exact remaining qualifications.

## Reproduction

Run from the canonical checkout `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow` with the native data paths identified in the frozen parent input-provenance file. Hashed analysis cards contain absolute paths; silently relocating or rewriting those cards changes identity. Saved checkpoint paths also resolve against this canonical checkout. Three archived model/toy ROOT containers are part of the calibration input contract. The 2015 container also retains its binned reference input histogram and fitted functions; the inventory is recorded in provenance/synthetic_input_inventory.json. The three original native observed-data ROOT files remain external.

```bash
python3 -B study_results/v4p9p13_calibration_20260905/run_calibration.py --ntoy 256 --nvalid 500
```

The production log records the actual command. Production used one numerical worker. Authorized refinements used at most two workers, each with one BLAS thread. Ordinary chunked workers are bounded at 4 GiB each; combined-only resource policies allow one 6 or 8 GiB worker alongside at most one 4 GiB worker, following a fresh memory-pressure check. These are conservative array-budget estimates, not measured resident memory. Existing checkpoints resume only if the source/count contract is unchanged. Changed source or toy counts require a new `--output` directory; never overwrite a frozen run.

After production has stopped, collect a stable snapshot and rebuild:

```bash
python3 -B study_results/v4p9p13_calibration_20260905/rebuild_products.py
```

This sequential command reads the exact ordered `collection_inputs.json`, waits for collection to finish, reruns the supplemental execution audit, makes all eight figure pairs, and builds the note with the independent reverse-truth diagnostic at 71 MeV. It does not generate or fit toys. The builder requires 456 completed points. Later refinement directories replace a whole matching coordinate, never pool toys. Only the protocol's failed MC-precision gates motivate refinement. Each frozen `selection.json` and `point_plan.json` records the replay configuration. Retained failed and memory-deferred attempts are part of the provenance, not selected successful results.

At combined 74 MeV, an independent scalar reference fit required a deterministic root-based initial value before the unchanged solver passed its original convergence gate. Both full calibration banks were reproduced exactly. The full run's final ledger assembly then encountered a mixed-schema metadata error after all numerical results were saved. A separate `finalize_reference_metadata.py` reconstructs that ledger from the saved successful checks; it verifies exact equality of every numerical result field and copied companion. The original failure files remain intact. See `SCALAR_REFERENCE_RECOVERY_PROTOCOL.md` and `REFERENCE_METADATA_FINALIZATION.md`.

The output note is `output/pdf/v4p9p13_calibration_20260905/HPS_GPR_Analysis_Note_v4p9p13_calibrated_backgrounds.pdf`. The frozen parent 15-page note remains intact.

## Artifacts and interpretation

- `derived/contract.json`: 47 frozen source/input hashes and ensemble counts. `provenance/frozen_runtime_v1/` preserves the numerical source snapshot; companion runtime hashes and environment versions are recorded separately.
- `derived/<scope>/mNNN/`: truth-specific endpoint traces, numerical audits, exact proposal/array hashes, independent validation summary and saved validation toy statistics.
- `summary/observed_calibrated_limits.csv`: all 456 observed coordinates, missing/censored/limited/resolved status, both raw and calibrated curves, MC bounds, local p-values and fixed/profiled ratios.
- `summary/truth_specific_limits.csv`: separate generating-truth endpoints. The larger endpoint defines the finite-family envelope.
- `summary/validation_summary.csv`: per-cell counts and exact binomial intervals, raw/calibrated decisions, Holm-adjusted excess-rate screens and mean-response diagnostics. No cells are discarded or pooled across mass.
- `summary/calibration_summary.json` and `numerical_qa.json`: completion, all numerical/source audits, validation families, selected checkpoints and output hashes.
- `figures/`: four observed-limit plots, local p-values, generating-truth dependence, conditional mean response and independent injection exclusion comparison. Shading denotes Monte Carlo uncertainty, not expected experimental bands. The envelope shading takes componentwise maxima of two approximate intervals and is not simultaneous.
- `validate_sampler.py` and `sampler_validation.json`: independent 120,000-draw one-bin Poisson weighting check; largest discrepancy 2.01 Monte Carlo standard errors.

The local-p0 envelope uses the larger tail from the two truths. The bounded-statistic atom has p0=1 when the observed signed root is nonpositive; the retained asymptotic convention has p0=.5 at Z=0. MC-limited estimates stay marked; small above-one importance estimates are displayed at one with an explicit MC-boundary status, preserving the raw estimate.

Five hundred validation spectra per cell can reveal large mismatches; an absence of family-adjusted flags does not certify percent-level coverage at every mass. Completeness, Monte Carlo precision, validation success and physical background qualification are separate statements.

The full numerical QA JSON exceeds GitHub's single-file size limit. The release stores it losslessly as `summary/numerical_qa.json.gz`, with compressed and original SHA-256 identities in `summary/numerical_qa_compression.json`. Run `gzip -dk study_results/v4p9p13_calibration_20260905/summary/numerical_qa.json.gz` to restore the exact JSON used by the release checks. The collection script also regenerates the uncompressed report. No audit rows are pruned.

## Publication controls

Publication uses an explicit allowlist and isolated Git index on `codex/v4p9p13-background-calibration`, preserving the dirty shared checkout and its index. Do not publish large development toy-bank NPZ arrays, draft PDFs/pages, bytecode, unrelated edits or external native data. See `release_review.md` for exact dependency and release rules.
