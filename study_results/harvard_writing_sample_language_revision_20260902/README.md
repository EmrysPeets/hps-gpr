# Harvard writing-sample reader-facing revision

This directory is a derivative editorial workspace for the Harvard fellowship writing
sample. It is based on the immutable accepted Sections 2--5 source and incorporates the
separately frozen selected-results Section 6. The source releases are not overwritten.

## Deliverables

- `pdf/HPS_GPR_Harvard_Writing_Sample_Reader_Facing.pdf`: final reader-facing PDF.
- `source/writing_sample.tex`: self-contained LaTeX driver.
- `LANGUAGE_REVIEW_CATALOGUE.md`: prioritized audit, vocabulary map, completed changes,
  and retained recommendations.
- `qa/`: build logs, text checks, rendered pages, contact sheets, figure checks, and
  file hashes.
- `scripts/`: deterministic reader-facing figure rebuilds.

## Editorial boundary

The revision changes prose, section order, and visible labels. It preserves the numbers,
statistical definitions, uncertainty qualifications, conditional-validation scope, and
distinction between local diagnostics and calibrated physics claims.

## Build

From `source/`:

```sh
/opt/homebrew/bin/tectonic -C --keep-logs --outdir ../qa/build writing_sample.tex
```

The build is strict: a missing figure causes an error rather than a placeholder.
