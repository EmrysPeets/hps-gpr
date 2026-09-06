# Calibration release dependency review

Read-only audit on 5 September 2026, while production was running. No fits,
runtime changes, staging, commits, or pushes were performed. This is a release
recommendation, not a declaration that the unfinished production is complete.

## Verified baseline

Publication branch `codex/v4p9p13-background-calibration` currently points to
`4f52e0cee2fdca645ae4b9c680df16840d31900b`.

- All 47 files in `derived/contract.json` match their current SHA-256 values.
  The branch already contains identical bytes for all 38 prior non-ROOT
  dependencies; the six new-study source/protocol files and three synthetic
  truth ROOT containers are absent, as expected before this release.
- All seven files in `provenance/frozen_runtime_v1/`, including its contract,
  match their live source counterparts. Preserve these bytes.
- All seven companion `gp` modules listed in
  `provenance/additional_runtime_hashes.json` match both their recorded hashes
  and the publication branch. The separately required v4.9.7
  `runtime_guard.py` and `runtime_combined/runtime_manifest.json` also match the
  branch. Do not amend the running contract to add these: retain their separate
  final provenance attestation.
- All 390 entries in the original publication file ledger match the branch.
  Both study manifests independently verify there: 225 entries for the initial
  comparison and 161 for the expanded v4.9.13 study. All 41 in-repository
  dependencies in the observed study's final `derived/summary.json` match both
  current files and the branch. Its three native data ROOT files remain external.
- The previous dependency fix is present: expanded snapshot `make_figures.py`
  and the v4.9.5 observed ledger and support decision. The parent observed CSV,
  note TeX, figures, and repository requirements are already published.

## Minimal new-study inclusion rules

Use an explicit file allowlist rooted at this study, plus the single final PDF
under `output/pdf/v4p9p13_calibration_20260905/`. Extend the existing publication
branch; do not import unrelated changes from the shared working tree.

1. Include the five numerical Python files named in `derived/contract.json`,
   `PROTOCOL.md`, and all seven `provenance/frozen_runtime_v1/*` files. Include
   `provenance/additional_runtime_hashes.json`.
2. Include `collect_results.py`, `make_figures.py`,
   `make_validation_figures.py`, `build_note.py`, `calibration_sections.tex`,
   `history_review.md`, `release_review.md`, and a concise final README with the
   actual production, collection, plotting and note-build commands.
3. Include `derived/contract.json` and each selected production coordinate's
   `result.json`, `validation_summary.csv`, and `validation_toys.csv.gz`.
   Include any actual refinement directory's contract and corresponding files,
   preserving the collector's ordered whole-coordinate replacement list.
   Preserve failure markers and unresolved/censored statuses if any remain;
   do not select only favorable points. The saved validation toy ledger is
   needed to reproduce the requested mean-amplitude Monte Carlo uncertainties.
4. Include all five `summary/` products, all seven final figure pairs in
   `figures/`, generated `note/*.tex`, and
   `HPS_GPR_Analysis_Note_v4p9p13_calibrated_backgrounds.pdf` in the output folder.
   Retain a final rendered-page/semantic QA record and its selected final pages.
5. Include `gp_refit_pilot.json`, `gp_lowrank_pilot.json`,
   `validate_sampler.py`, and `sampler_validation.json` as bounded numerical
   validation evidence. The four-coordinate `pilot_frozen` results are optional;
   if cited, publish their contract and their result/validation files together,
   explicitly separated from production.
6. Add a final SHA-256 manifest over the exact release allowlist, excluding the
   manifest itself, and a compact environment record for the actual production
   Python, NumPy, SciPy, scikit-learn, pandas, uproot, BLAS/threadpool and plotting
   versions plus Tectonic. The existing minimum-version requirements and pilot
   NumPy/scikit-learn fields alone do not pin the full runtime.

Exclude `__pycache__`, bytecode, temporary files, large `*_bank.npz` development
arrays, old `pilot*` directories other than an explicitly selected frozen pilot,
draft pages, TeX build intermediates, and redundant work-copy PDFs. Keep useful
run commands/status in the final README rather than publishing contradictory
intermediate progress statements. The deterministic seed rules, proposal
metadata, and whole-bank hashes remain in the frozen runner/checkpoints even
though the full calibration toy banks are intentionally not stored.

## Missing practical inputs and supporting evidence

The three synthetic-truth containers total only **6,146,345 bytes**. Publish
these exact files at their existing paths, or provide immutable downloads with
their frozen contract hashes and path placement instructions:

| Repository path | Bytes |
|---|---:|
| `outputs/funcform_toys/funcform_2015_dataset_mod_toys.root` | 4,303,314 |
| `study_results/v4p9p7_2016_support_combined_100toy_20260902/inputs/2016_threshold_qualified_background_toys_100.root` | 1,102,629 |
| `study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/inputs/native10_fsig_background_toys_100.root` | 740,402 |

The runner hashes the entire container. Extracting only its mean histogram
would change the contract, so a trimmed substitute cannot reproduce this frozen
run without a separately declared input/contract version. These are distinct
from the three native observed-data ROOT files intentionally kept external.

Twelve unique small historical evidence files linked from `history_review.md`
are also absent from the branch (236,307 bytes total). Publishing these makes
the historical closure and truth-selection claims inspectable without importing
the historical campaigns. Paths below are relative to `study_results/`:

- `v4p9_2021_threshold_support_qualification_20260817/`:
  `README.md`, `study_spec.json`, `build_fsig_anchor_truth.py`.
- `v4p9p1_2021_background_validation_consolidation_20260817/`:
  `README.md`, `derived/consolidated_pull_moments_90cl.csv`,
  `reference/v4p6_full100/study_spec.json`, `build_continuation_toys.py`.
- `v4p9p5_2021_gp_support_edge_optimization_20260820/`:
  `run_support_scan.py`, `STUDY_PROTOCOL.md`,
  `STEERING_AMENDMENT_20260820.md`,
  `reference/v4p9_fsig_anchor_fit_summary.json`.
- `v4p9p12_2021_peak_dip_diagnostic_20toys_20260905/reverse_injection/`:
  `run_reverse_injection.py`.

These are supporting evidence, not a promise to rebuild every historical toy
campaign. The frozen calibration consumes the three archived mean inputs.

## Final release gates

After production stops, collect a stable snapshot, verify all frozen hashes and
new postprocessor/output hashes, then rebuild figures and the note from that
snapshot. Require 456 distinct coordinates and 5,472 validation cells for a
complete-grid release, or label the exact partial count prominently. Completeness
is separate from MC resolution and validation success: retain all pointwise
precision statuses and adjusted excess-rate flags. Do not require failures to
disappear by changing the procedure.

The current input card and historical provenance include absolute paths.
Document the canonical checkout/input path mapping needed for strict replay;
silently rewriting hashed input files in a relocated checkout breaks identity.
Postprocessing and TeX use repository-relative dependencies, while some saved
checkpoint paths are absolute and must also resolve for the note builder.
Verify every selected release file against the publication tree before pushing,
and preserve the shared checkout's original branch, index and unrelated edits.

## Addendum after the review

The final figure set now also includes `truth_dependence.pdf/png`, for eight
figure pairs, plus three plot-provenance JSON files. Include the separate
sampling-refinement driver, frozen protocol/selection artifacts and all actual
attempts if that second pass is executed; preserve the original inference
source hashes and verify derivative hashes separately.

ROOT-key inspection is recorded in `provenance/synthetic_input_inventory.json`.
The archived 2015 model/toy container also includes its binned input reference
histogram and fitted functions; it is not a purely synthetic container. The
three original external native ROOT files remain external.
