# Recent-study publication snapshot

This snapshot preserves the study folders dated **13-27 September 2026**, all three saved `unblind_meeting_RCmeet` presentation revisions, the APEX comparison, the finalized 2021 injection handoff, and the corresponding delivery artifacts. The previously pushed v5.0.5 note branch (`eb393ba86906a5738cc1cd4574fc5df7e2a43e0c`) is retained in the merge history. The original working checkout was not rebased, reset or cleaned.

The [study logbook](../../docs/study_logbook_20260927/README.md) explains what the 40 catalogued entries accomplished. [snapshot.json](snapshot.json) lists the exact roots, 28,984 source files, 125 additional unique archive payloads, all original hashes, storage hashes, ZIP identities, and the 10 excluded OS metadata/bytecode/process-lock files. These counts describe archive preservation, not independent statistical experiments.

## Storage and completeness

Most artifacts retain their original repository-relative paths and bytes. The large field maps, field caches, two CSV ledgers and two already-compressed remote delivery payloads are stored losslessly compressed; the two remote payloads additionally use bounded parts. Their original hashes and storage representation are in `snapshot.json`. Twelve originals need restoration before running workflows that require those paths.

The **53 original delivery ZIPs are represented by their complete member inventories**, with each of the **49,901 member payloads** mapped to an identical stored file. The additional unique historical payloads live in [archive_objects](archive_objects). This avoids repeatedly committing large ZIP containers containing the same scientific files, and preserves historical release contents. Each original ZIP has an adjacent `.zip.README.md` with its reconstruction command. ZIP container bytes are not claimed to be identical after reconstruction; member names and bytes are checked individually. Original container hashes remain recorded for identity.

Read-only source paths in old provenance records describe the originating machine. They are not required for reading this release. Existing historical manifests retain their original meaning and are not rewritten to hide later revisions; restoration reestablishes the large original paths before package-level checks. The new snapshot hashes define this publication's exact payload.

## Verify and restore

From the repository root, using Python 3.9 or newer and only its standard library:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py --verify
python3 publication/recent_studies_20260927/scripts/restore.py --restore-large
```

Verification checks storage hashes, decompresses large payloads for their original hashes, and confirms every ZIP member mapping. Restoration never overwrites a differing local file. The restored originals are specifically ignored by Git; their compressed sources remain tracked.

Reconstruct a delivery ZIP to a new path, for example:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3p6_readability_2021_20260924/HPS_GPR_v6p3p6_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p3p6_Reproducible_Study.zip
```

The reconstruction checks every output member against its source hash. Full study reruns are separate commands documented by each package. This publication does not rerun HPS fits, create new ensembles or use S3DF compute.

## Publication QA

The new logbook checks its quoted values against saved numerical tables, verifies source links and figure hashes, and undergoes text plus rendered-page review. Source/archive integrity and a reconstruction of the large v6.3.6 delivery are checked independently of scientific-model adequacy. Historical numerical, portable-rebuild and visual evidence remains inside the study packages.

The manifest pins `origin/main` at the start and the inherited v5.0.5 commit. The final GitHub pull request and merge commit, plus the annotated tag `studies-2026-09-27`, identify the published release. The branch contains the source/data snapshot and logbook; the tag provides stable links from the PDF.
