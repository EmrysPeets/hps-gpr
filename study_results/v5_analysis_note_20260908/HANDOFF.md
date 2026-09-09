# v5 analysis-note handoff — 8 September 2026

The 30-minute session began at 14:40 PDT and is capped at 15:10 PDT. This is a substantial review draft, not a declaration that the statistical qualification or unblinding approval is complete.

The saved draft has 138 pages and passed all 13 integration checks. The full-range connected limit is Figure 37 on page 65; the observed-only overlay is Figure 38 on page 66. All pages were rendered and inspected, with detailed checks of the main curve and wide extraction panels. The source package includes a file manifest and hashes.

## Where to continue

Authoritative isolated worktree:

`/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5-analysis-note-20260908`

Branch: `codex/v5-analysis-note-20260908`.
Base commit: `0c1d4692b` (`Refine calibrated background validation workflow`).
Project directory: `study_results/v5_analysis_note_20260908/`.
Canonical source: `source/main.tex` and its input sections.
Final PDF: `pdf/HPS_GPR_Analysis_Note_v5p0p0_Unblinding_Review_Draft.pdf`.

The original shared checkout switched to `main` during the initial reads at 14:41:15 PDT. None of the agents made that change. Work proceeded in the isolated checkout; no frozen parent file, shared index, branch selection or remote was changed by the v5 work. No commit or push was made. Keep using this worktree rather than switching the shared checkout back.

## What was completed

- Consolidated the detector, selection, resolution, normalization, statistical methodology and historical validation prose with current observed results and the recent studies.
- Corrected the obsolete 2021 combined-support description from 40–300 to 36–300 MeV, including the dataset overview figure. The overview's raw ROOT hashes and histogram keys match the frozen input record.
- Put the connected full-range observed limit and local asymptotic p-values at the beginning of the results. Both that figure and the observed-only four-curve overlay connect every adjacent mass over 19–250 MeV, including all dataset-membership transitions. Their 232 values exactly match the saved dimuon-corrected source array.
- Included the original 300-toy bands and saved targeted tail refinement. Zero-count cells use open downward triangles at the exact one-sided 95% upper bound; analytic plotting-floor markers remain distinct.
- Added the profiling rationale, injection strengths, alternate background comparison, and the complete curated calibration explanation with flowchart, generating-truth descriptions, observed effects, offsets and independent validation.
- Added profiled-only individual and combined GP global-probability displays, explicit definitions, finite-toy bounds and remaining qualification statements.
- Rebuilt extraction displays with native-bin model lines and sideband context. Their residuals subtract the same GP mean everywhere; the background displacement, total displacement and fitted signal are separate. Wide multi-campaign displays use landscape pages.
- Added individual and combined correlation matrices, actual 2021 injection echoes, and carefully labelled Gaussian conditional-response slices for the three campaigns. Those slices are not simulated signals or calibrated echo significances.
- Explained the shared-rate likelihood loss, moved traditional fits and the joint deficit scan to dedicated appendices, and placed the exploratory 15–20 MeV study last.
- Made the change log the first appendix section and placed a separate appendix contents page after it.
- Created source/claim and figure provenance ledgers, a request checklist, a build script, and a read-only validation script. The new figure family contains 51 PDF/PNG pairs; some are supporting alternatives rather than included panels.

## Statistical work still outstanding

1. **Unresolved limit tails.** This session did not generate additional toys. The saved refinement still has nine zero-count cells, shown as bounds. For continuation, make a new derivative from `v4p9p12_targeted_tail_refinement_20260905`; do not run its generators in place or overwrite its manifest. Inspect the runner and its plan mode, predeclare the selected coordinates and precision objective, allocate fresh nonoverlapping toy IDs, keep fixed seeds and constituent pairing, and record all outcomes. More tail draws improve conditional precision; they do not supply coverage or discovery calibration.
2. **Final background qualification.** The later calibration studies reveal raw asymptotic rejection-frequency failures that the older mean-pull summaries do not test. Preserve the exact truth, kernel and retraining policies when comparing studies. Qualifying a physical background family requires predictive controls and suitable alternative spectra; taking more Gaussian-field draws does not resolve a deficient stress model.
3. **2016 status.** The earlier support study returned no provisional edge. The current result inherits a separate numerical-state exception (87/142 independent state replays at the strict tolerance). Downstream prediction checks do not erase that history. Any final decision must explicitly resolve or accept that exception without representing it as clean independent certification.
4. **Final global interpretation.** The existing GP studies are conditional method-development results. In particular, the extreme combined score at 76 MeV is driven by a large stress offset while the observed raw local coordinate is only about 0.166. Final 2016/2021/combined particle-global values require a qualified generating model, adequate independent validation, and the complete declared mass/model/direction/sequential-look procedure. No final value has been manufactured.
5. **Additional data.** This work opens no new 2021 events. Freeze the inference choices and any staged-data checks before analyzing an additional fraction. The note is an input to that review, not an authorization.

The unresolved-tail coordinates are now listed in `derived/unresolved_tail_continuation.csv`, with a source-hash record alongside it. They are 2015 at 19 MeV; 2016 at 100–104 and 117 MeV; and 2021 plus the 2015+2021 combination at 71 MeV. The 117 MeV cell is the weaker-limit tail; the others are stronger-limit tails. The table records the existing 3,000/10,000-toy bounds and next unused IDs relative to this archive. It is a continuation inventory, not a new allocation: check all later campaigns for ID collisions before drawing. The historical runner lives at the study root (`run_tail_refinement.py`), and even its `--plan` mode writes into that study; adapt it only in a fresh derivative.

## Editorial work to consider next

- Perform a collaboration-level content review of the 100% unblinding proposal, including the primary asymptotic/calibrated policy. The draft deliberately avoids carrying over the unsupported Harvard readiness sentence.
- Decide whether any detailed historic figure archives should be restored to the note. They are summarized by study purpose in the appendix and remain intact in the repository; v5 avoids automatically appending every older plot.
- Continue stylistic cleanup of the inherited detector, legacy validation and low-mass artwork if desired. The newly curated figures have clean significance labels and caption-based explanations; some historical graphics retain original labels and annotations.
- Confirm the final title, author list and document status with the collaborators before circulation as an official request.

## Rebuild and check

From the authoritative worktree:

```bash
bash study_results/v5_analysis_note_20260908/scripts/build_note.sh
```

Then, with a Python environment containing `pypdf`:

```bash
python3 study_results/v5_analysis_note_20260908/scripts/validate_v5.py
```

For the bundled desktop runtime, the verified interpreter is:

`/Users/emryspeets/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`

Compile from the current LaTeX source. The bootstrap assembly scripts describe the initial assembly and must not be rerun over reviewed edits. Figure regeneration requires the archived study files and the scientific Python environment; the dataset overview additionally checks the released ROOT inputs. **Finish all figure writing before starting TeX.** A concurrent write caused one discarded intermediate build to encounter a partial PDF; the final build is from frozen figure files.

Tectonic reports a cached BibTeX rerun notice. This is distinct from unresolved references: the validator separately checks actual citation keys, labels, final PDF text, page ordering, overfull boxes and source hashes. After any source or figure change, rebuild, render the affected pages, inspect their layout, rerun validation and refresh the output manifest. Do not reuse the current visual-review status for changed bytes.

`editorial/REQUEST_CHECKLIST.md` maps the attachment and the live full-range-line clarification to the draft. `qa/final_validation.json`, `qa/visual_review.json` and the manifest record the actual final checks and artifact identities.
