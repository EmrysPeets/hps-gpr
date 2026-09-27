# HPS-GPR analysis note v5.0.4

The full analysis note includes 2015 through 100 MeV in every applicable observed and combined search. Most of the parent note is retained: 31 existing section files are unchanged, with focused changes to 13 files and three new appendices. The parent is the v5.0.2 note with v5.1.1 diagnostic support. Earlier study products remain unchanged.

Read `pdf/HPS_GPR_Analysis_Note_v5p0p4_Unblinding_Review_Draft.pdf`; edit `source/main.tex`. `editorial/figure_map.csv` maps the figure numbers in the request to the new PDF. The detailed global-method checks are in Appendix C, and the earlier displays remain in the historical appendix. Sections 6.11 and 6.12 retain their numbering. The fixed-mass recovery discussion, signal-response equation, and injection-intensity comparison remain in the main text.

## Extension and interpretation

- The 2015 spectrum, binning and 14–135 MeV training support are retained. Kernel parameters above 90 MeV are held at their archived 90 MeV values, while the mask, template, density and likelihood are evaluated at each actual mass. The existing resolution rises from 4.6975 to 5.2297 MeV between 90 and 100 MeV.
- Saved integer-mass observed fits from v5.2.1 supply 40 added scope–mass cells. New conditional ensembles contain 300 toys per cell: 9,000 constituent draws shared across scopes, yielding 12,000 fitted limits. They provide pointwise bands and upper-limit tail diagnostics, without establishing coverage. Seven new empty tail cells retain Monte Carlo upper bounds.
- The combined field recomputes the ten changed coordinates. The 2015 and combined direct checks use 256 coherent archived whole-spectrum experiments, 200,000 Gaussian fields and 40,000 conditional-mixture samples at each declared high threshold. The unchanged 2016 and 2021 displays retain their original benchmarks. Conditional Gaussian tails remain distinct from calibrated discovery probabilities.
- The 2015 Sidak trial estimate rises from 13.624 to 14.521. The complete-search value remains 35.381 because the prescription uses the narrowest active resolution. The strongest nominal local complete-search excess remains at 66 MeV, with Z = 2.765 and Sidak reference p = 0.0959.
- The new three-campaign 92 MeV fit gives local Z = 2.520, fitted epsilon squared = 3.096e-6 and a 90% CLs limit of 4.675e-6. Its separate-rate likelihood loss is 11.552 for two freed amplitudes. This is a descriptive comparison at a selected mass.
- The statistical review identifies the narrow 76 MeV probability spike as a combined stress-reference and positive-fit-gate effect. The 2016 raw fit there is negative, and the combined raw root is only 0.166. A probability-curve width is not a measured resonance width. See `statistics_review/review.md` and Appendix C.

## Rebuild

With Tectonic available and its LaTeX resources cached:

```sh
bash scripts/build_note.sh
```

The PDF is written to `qa/build/main.pdf`. LaTeX compilation uses only the bundled `source`, `figures` and `derived/*.tex` assets. `scripts/portable_v504.py` verifies a separate build from those assets and compares every rendered page and its extracted text.

The saved-array figure build is:

```sh
python3 scripts/make_v504_figures.py
```

It needs numpy, scipy, pandas and matplotlib. `scripts/validate_v504.py` additionally uses PyMuPDF and Pillow. The extension runners are `scripts/extend_results.py` and `scripts/extend_global.py`; they are checkpointed, use one numerical thread, and read the bundled spectra and ledgers. To rerun their computations, work in a separate copy and move the relevant `derived/extension*` checkpoints aside. `edit_v504_note.py` records the original editorial transformation and requires the original parent source; it is not needed to compile or edit this complete note.

## Evidence and verification

`inputs/extension_sources.json` pins the inputs and solver sources. The primary numerical products are `derived/v504_result_curves.csv`, `derived/v504_union.csv`, `derived/v504_limit_tails.csv`, `derived/v504_rate_consistency.csv`, and `derived/extension_global/*_curves.csv`. Per-toy limits and the generated constituent counts are retained under `derived/extension`. Numerical and document checks, the portable-build result, and the visual-review record are under `qa`. `MANIFEST.sha256` identifies the package contents.

The extension follows examination of the original data. Its conditional checks do not create a selection-adjusted discovery claim, independently qualify the stress background, or authorize unblinding additional events.
