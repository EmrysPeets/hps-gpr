# HPS GPR analysis note v5.0.0 — unblinding review draft

This isolated derivative consolidates the main analysis description and the v4.9.12–v4.9.16 studies into a single review draft. It uses full 2015, full 2016 and the released 2021 10% sample. It does not open more data, generate toys, change a limit, or certify unblinding readiness.

The user's clarified main result is implemented as one connected observed curve across all 232 integer mass hypotheses from 19 to 250 MeV, including the changes in active datasets. The standalone figure and observed-only overlay use exactly the frozen, dimuon-corrected source values.

- Authoritative source: `source/main.tex` and its active section files.
- Built PDF: `pdf/HPS_GPR_Analysis_Note_v5p0p0_Unblinding_Review_Draft.pdf`.
- Review and continuation: `HANDOFF.md` and `editorial/REQUEST_CHECKLIST.md`.
- Figure source/array identities and numerical display checks: `editorial/figure_provenance.json`.
- Scientific claim provenance: `editorial/source_claims.json`.
- Final semantic and file checks: `qa/final_validation.json`; rendered review: `qa/visual_review.json`.
- Portable source rebuild: `qa/portable_build.json`. The extracted source archive rebuilt all 138 pages with identical page text and dimensions.

## Rebuild the existing note

From this directory, with Tectonic on PATH:

```bash
bash scripts/build_note.sh
```

This reads the current curated LaTeX source and saved figures. It uses cached TeX resources. A cached BibTeX rerun notice may occur; unresolved citations, references and overfull boxes are checked independently.

From the repository root, run the read-only integration checks with a Python environment providing `pypdf`:

```bash
python3 study_results/v5_analysis_note_20260908/scripts/validate_v5.py
```

`make_v5_figures.py` regenerates plot derivatives from the frozen study arrays. Run figure generation to completion **before** compiling; the figure writer must not run concurrently with TeX. The bootstrap assembly scripts describe the initial curation only and are not the canonical rebuild route: later reviewed source edits are authoritative.

## Isolation and provenance

The worktree is `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5-analysis-note-20260908`, branch `codex/v5-analysis-note-20260908`, based on commit `0c1d4692b`. The shared original checkout changed branches during the initial read; this derivative was isolated to preserve that activity and the frozen source files. There is no push, merge or new commit from this session.

The main displayed limits remain bounded, pointwise asymptotic 90% CLs limits. The calibration and global studies retain their generating-background assumptions, finite-Monte-Carlo bounds, and 2016 qualifications. The expected-limit bands do not establish coverage or supply scan-wide significance. See the note and handoff for the remaining review items.
