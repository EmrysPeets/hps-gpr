# Harvard writing sample: current final-data combinations

This derivative replaces the historical selected-results section with the
v4.9.12 observed result contract for full 2015, full 2016, and the currently
available 2021 10% sample. It contains the three standalone scans, all three
pairs, and the all-three shared-coupling combination. The 2021 1% and 2016 10%
samples are not result curves.

The opening sentence states that the analysis is partially unblinded and that
full unblinding is coming shortly. Section 6 includes local fixed-mass
asymptotic p-values, the all-three signal extraction at the smallest local p0,
and no expected bands, toys, look-elsewhere correction, or global significance.

## Final product

- PDF: `pdf/hps_gpr_writing_sample_junior_fellows.pdf`
- portable copy: `output/pdf/hps_gpr_writing_sample_junior_fellows.pdf`
- source: `source/writing_sample.tex`
- selected results: `source/sections/06_selected_results.tex`
- numerical release: `../v4p9p12_final_dataset_combinations_20260902/`

The numerical release attestation is a conditional pass. Full 2016 uses the
30--210 MeV support and a resolution-scaled length-scale upper factor of 12
under a disclosed cross-process state-replay exception. The downstream
optimizer-off prediction replay, statistical release checks, and numerical
conditioning audit pass, but 2016-inclusive curves remain preliminary and
conditional on that exception.

## Section 5--6 reader-facing revision

The current PDF includes a time-boxed language and figure-display pass over
Sections 5 and 6. The rejected support-scan table was removed and its useful
numerical range retained in prose. Technical decision language was made more
direct, while the conditional-validation and local-significance boundaries
were preserved. Figure legends and the 66 MeV callout were moved outside the
data regions so they no longer cover plotted curves. The Table 9 now present
in the PDF is the later signal-yield decomposition, not the removed support
scan.

Build from `source/` with:

```sh
/opt/homebrew/bin/tectonic -C --keep-logs --outdir ../qa/build_final writing_sample.tex
```
