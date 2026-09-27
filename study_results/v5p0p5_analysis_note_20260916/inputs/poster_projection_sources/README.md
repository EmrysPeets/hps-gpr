# Figure 2 publication revision, 13 September 2026

This is a derivative of the complete v5.0.4 analysis note. It replaces Figure 2, adds the two requested full-exposure-equivalent projections, and adds a short construction appendix. Of the 47 existing section files, 46 are byte-for-byte unchanged. The original v5.0.4 delivery remains intact.

## Figures

- `figures/figure2_clean_overview.pdf`: clean standalone world-contour overview.
- `figures/figure2_overview_and_projections.pdf`: overview with two matched zoom panels, used as Figure 2 in the note.
- `figures/figure2_projection_panels.pdf`: the two zoom panels as a separate publication asset.
- Every plot has an editable SVG and a 320 dpi PNG preview. The PDFs contain vector paths and embedded fonts, with no raster plot objects.
- `figures/figure2_captioned_proof.pdf`: the typeset Figure 2 page and caption.

The gray background is the union of the archived exclusion polygons. Their original vertices, contour ordering and missing intervals are retained; no smoothing alters a limit. Individual internal boundaries are faint, the HPS curves are thinner, and the two zoom panels use identical axes. The archived APEX 2019 projection is excluded from the published-exclusion shading, as identified in its source notebook. This is a historical world-data context, not a complete 2026 exclusion survey. The caption retains the legacy approximate 95%-to-90% display conversion for HPS, BaBar and NA48/2.

## Projection calculation

The current-sample inputs are 2015 full, 2016 full, 2021 10%, and, for the added-2019 curve, the measured nominal 2019 1% spectrum selected with psum > 3.64 GeV. The 2015 extension through 100 MeV is retained. The 2019 search is 75–250 MeV with 50–300 MeV GP support. Its archived factor-15 kernel scan supplies fixed kernel coordinates; its resolution, radiative fraction and conversion are borrowed from 2021. The 2019 length-scale ceiling remains active at every mass.

Each displayed combination first uses an actual shared nonnegative-epsilon-squared profiled Poisson likelihood with block-diagonal GP constraints. There are 232 three-campaign and 176 four-campaign fits. Both curves use the same current numerical implementation. Three-campaign endpoints differ by at most 2.502% from the frozen note ledger; this figure-specific recomputation does not replace the observed Section 6 results. The saved 2019 upper limits are not combined or substituted for the new joint profile.

For each active set, the full-equivalent transformation is

`u_full_eq = u_current * sqrt(sum(d_y) / sum(f_y * d_y))`,

where `d_y` is the observed native-bin density in the ±1.64-sigma window and `f_2015=f_2016=1`, `f_2019=100`, `f_2021=10`. The two curves agree exactly below 75 MeV. The minimal-visible dimuon correction is applied once. This is a conditional, statistics-only observed-equivalent projection: it preserves fluctuations and background structure in the current samples and is not an expected sensitivity, future observed exclusion, or ensemble median. No new toys or discovery probabilities are computed.

The numerical products are `derived/projected_contours.csv`, the per-mass checkpoints in `derived/projection`, and `derived/projection_protocol.json`. The ledger explicitly retains the source-ledger comparison and the separately computed 2019 response.

## Lepton central lines

The muon line uses WP25 with the final 2025 experimental average, Delta a_mu = 38e-11. Its quoted uncertainty, 63e-11, includes zero; the line is a central reference, not a nonzero favored band. The electron line uses the rounded positive Rb20 + Fan23 residual, approximately 0.34e-12. The published rounded inverse-alpha values reproduce that residual through the QED derivative. A negative Cs-based residual is not mapped into a positive-vector favored line. These are central-value loci, not posterior medians.

The one-loop vector integral is evaluated numerically without a heavy-mass approximation. Curves are tabulated in `derived/g2_central_curves.csv`. Source details are in `inputs/literature_sources.json`, with citations in the caption and Appendix L:

- [Muon Theory Initiative, WP25](https://arxiv.org/abs/2505.21476v3).
- [Fan et al., electron magnetic moment](https://arxiv.org/abs/2209.13084).
- [Morel et al., rubidium fine-structure constant](https://doi.org/10.1038/s41586-020-2964-7).

## Rebuild and verification

With Tectonic and its LaTeX resources available, run `bash scripts/build_note.sh`. The complete source is `source/main.tex`; its build needs only the bundled source, figures, and derived LaTeX tables. It does not need the original checkout.

For calculations and figures, Python requires numpy, scipy, pandas, matplotlib, uproot, PyMuPDF, and shapely. The plotting script first checks `scripts/vendor` for the locally installed shapely; a regular shapely installation works when that directory is absent. `provenance/environment.json` records the versions used.

1. `python3 scripts/build_projections.py` reuses saved checkpoints and reconstructs the full-equivalent contour ledger.
2. `python3 scripts/make_figure2.py` regenerates the three plot variants.
3. `bash scripts/build_note.sh` compiles the full note.
4. `python3 scripts/validate_figure2.py` checks the projections, fit roots, covariance representation, lepton loop, vector figures and document references.
5. `python3 scripts/portable_build.py` independently builds from the bundled assets and compares every page's text and rendering.

`update_note.py` records the original editorial transformation from the parent note; it is not needed to compile or edit this completed derivative. The archived numerical results in the rest of the note are unchanged and are not recalibrated by the checks for this figure revision. Manifest hashes identify all packaged artifacts.
