# v4.9.6 analysis-note source

This self-contained source tree is an editorial continuation of the frozen v4.9.5
note in
`study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/note/build_source/`.
Version 4.9.6 changes the writing and document structure only. It does not change any
data input, analysis card, fit, figure, or numerical result.

## Build both documents

Run Tectonic from this directory so that all section, figure, and bibliography paths
resolve relative to the source tree:

```bash
mkdir -p ../qa/build_full ../qa/build_writing_sample ../pdf
tectonic -C --keep-logs -o ../qa/build_full main.tex
tectonic -C --keep-logs -o ../qa/build_writing_sample writing_sample.tex
cp ../qa/build_full/main.pdf ../pdf/HPS_GPR_Analysis_Note_v4p9p6.pdf
cp ../qa/build_writing_sample/writing_sample.pdf \
  ../pdf/HPS_GPR_Harvard_Writing_Sample_Sections_2_to_5.pdf
```

The build outputs are `../qa/build_full/main.pdf` and
`../qa/build_writing_sample/writing_sample.pdf`; the final two commands place stable
release names under `../pdf/`. The `-C` option enforces an offline build from the local
Tectonic cache. On a new machine, omit `-C` for one initial build to populate that
cache, then repeat the cached build.

`main.tex` builds the complete v4.9.6 analysis note. `writing_sample.tex` builds a
standalone excerpt containing Sections 2--5, including the 2021 support-selection
subsection. The excerpt has its own cover and bibliography, suppresses review line
numbers, and fails if a referenced figure is absent.

## Source layout and continuation points

The current main-text sequence is declared explicitly in both drivers:

- `sections/05_toys_validation.tex` contains the core background-model validation.
- `sections/05a_2021_support_selection.tex` is its final subsection and records the
  2021 support selection.
- `sections/06_results.tex` contains the accepted and supporting results.
- `sections/06a_2021_observed_result.tex` is its final subsection and records the
  observed native-10% scan made with the selected support.

A future support or validation addition belongs after `05a_2021_support_selection`
and before `06_results`. A future observed-result addition belongs after
`06a_2021_observed_result` and before `07_conclusions`. Add new files at those points
rather than overwriting the frozen v4.9.5 subsections. Inclusion in the writing sample
must remain an explicit editorial choice.

## Interpretation boundary

The v4.9.5 support study remains a source-conditioned injection--extraction
diagnostic. The observed native-10% scan remains an asymptotic result with no expected
bands, CLs toys, direct-coverage claim, or scan-global significance calibration. The
accepted v4.2 three-campaign result is unchanged.
