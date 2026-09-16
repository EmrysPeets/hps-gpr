# HPS GPR Analysis Note v5.0.5

16 September 2026. This is the complete revised analysis note, based on the supplied v5.0.4 Figure 2 revision. It preserves the original delivery and all 242 inherited files in `derived/` byte for byte. The new joint binning diagnostic is saved separately; it does not replace the nominal observed result.

## Changes

- Section 1.1 now describes the two published engineering-run searches directly. Figure 1 reproduces their published limits as vector crops, retaining the original 95% confidence constructions.
- Figure 2 uses the updated poster's world-contour and displaced-reach sources, with matched lower panels for the three- and four-campaign prompt projections. The latter are teal and dark red. HPS engineering-run contours retain their published 95% level; the prompt projections remain conditional 90% observed-equivalent curves.
- Section 3 adds campaign geometry and magnetic transport, explaining how the curves and generated event displays were made. A detailed appendix preserves the decay equations, sensor intersection tests, field maps, coordinates, hit conventions, integration method and numerical validation. The geometry-study 2021 strip-hit convention is explicitly distinct from the selection provenance of the observed prompt-TC histogram.
- A new section after the observed results integrates the v5.6.0/v5.6.1 10%-based projections. It retains the exposure and known-background Asimov scaling equations, distinguishes target-matched from direct yield-scaled injections, tabulates actual toy medians, explains finite-toy maxima and shows predicted upper-limit echoes. The 1% alternatives and fit catalogues are in an appendix.
- The 76 MeV discussion now includes a bounded joint binning check. The combined signed root at 76 MeV changes from +0.166 to -0.216/-0.137 for doubled/quadrupled widths at the original origins. Half-coarse-bin origin shifts give +0.530/-0.695. The first positive integer test mass in 72–80 MeV is 76 or 77 MeV. The changed bin sampling also changes the far support edges slightly. No stress-field or global-tail recalibration was performed.
- The final appendix summarizes the beam-energy mass-scaling tests and the v5.5.4 rate, uncertainty and extracted-signal comparisons, with key plots. A fitted energy factor remains an empirical response model, not calibrated acceptance or a production law.

The September meeting PDF guides the selection of material. Slide labels are not treated as numerical authority: the 120 MeV region's saved injection is at 123 MeV, and target significances differ from empirical medians. The supplied PDF, meeting slides, study arrays and poster inputs are identified in `provenance/` and the manifest.

## Read and rebuild

The delivered PDF is `pdf/HPS_GPR_Analysis_Note_v5p0p5.pdf`. The entry point is `source/main.tex`; its relative figure/table dependencies are bundled. With Tectonic and its LaTeX resources installed, run:

```bash
bash scripts/build_note.sh
```

The script uses cached resources (`-C`). On a machine without the required TeX cache, remove that flag for the first build so Tectonic can obtain its normal resources. The output is `qa/build/main.pdf`.

The standalone figure and bounded-diagnostic scripts use Python with numpy, scipy, pandas, matplotlib, PyMuPDF and shapely. `scripts/check_76_binning.py` and `scripts/plot_76_binning.py` reproduce the new check from bundled spectra without accessing ROOT files or generating toys. `scripts/make_echo_overview.py` uses bundled v5.6.1 scan arrays. `scripts/make_updated_figure2.py` uses the copied poster contour sources; it does not perform new sensitivity fits. The two published-paper figure crops preserve the original vector content.

The editorial scripts in `provenance/` record construction from the local parent checkout; they are not needed to rebuild the delivered note and should not be rerun over edited source. Numerical provenance is separate from document-build reproducibility: reproducing every historical study requires that study's original source/data package.

## Checks and scope

The QA record accompanies the final delivery. It includes resolved cross-references, data-table comparisons, a nominal 76 MeV replay, convergence and positive expectations for all 180 new pointwise coordinates, inherited-ledger identity, rendered-page inspection, and a separate source build. The historical studies retain their original validation limits. The note remains a review draft; these additions do not resolve the documented full-2016 state-replay exception or the incomplete combined global extension.

The GitHub study archive preserves 35 distinct delivered PDFs, including earlier note versions and separately delivered figure PDFs. Its version index explains the changes within the 5.1, 5.2, 5.3, 5.5, 5.6 and 5.7 study families. Identical PDF copies are stored once with source aliases and SHA-256 identities.
