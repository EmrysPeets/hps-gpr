# v5.0.5 sources for requested slides

All source paths below are relative to the repository root. Prepared PNGs are in `output/slides/unblind_meeting_RCmeet_20260922/science/assets/`. They were visually inspected. No release file, event selection, observed result, or live deck was edited by this subtask.

## Slide 3: updated 2021 selection

Source: `study_results/v5p0p5_analysis_note_20260916/source/tables/event_selection_table.tex`, and `source/sections/03_event_selection.tex` in the same package. The note explicitly says the archived field did not establish the eight-hit two-dimensional electron requirement. Therefore the user's new bracket instructions are the authority for this update; do not imply the old kinematic plots validate the updated hit cut.

Suggested compact selection table:

| Requirement | 2021 prompt-TC selection |
|---|---|
| Trigger | Single-2 or Single-3 |
| Track momenta | Each track > 0.4 GeV; electron < 2.9 GeV |
| Pair momentum | p_sum > 2.8 GeV; p_vtx < 4.0 GeV |
| Track quality | chi²_track/ndf < 20 |
| Electron hits | 8 hits (updated selection, per user) |
| Vertex | Exactly one selected good vertex; vertex chi² cut removed (updated) |
| Positron cluster | E_cluster > 0.2 GeV |

The user did not specify strip/2D versus 3D or the comparison operator for the new electron hit requirement. Do not silently infer those. Exactly one selected good vertex is explicit at table source line 113 (Nvtx > 0 and Nvtx < 2). The full-note preselection has ≥10 two-dimensional hits for both tracks, but the new electron wording updates that context and its counting convention needs eventual reconciliation.

If space permits, retain timing: electron-track / positron-cluster < 6.0 ns; positron-track / positron-cluster < 5.1 ns; track-track < 7.8 ns, using the source note's timing-difference convention.

Assets: `2021_preselection_electron_momentum.png`, `2021_preselection_vertex_chi2.png`, rendered from the same-named `signal_overlay_loose_loose_ele_p.pdf` and `signal_overlay_loose_loose_vtx_chi2.pdf` under the note's `source/preselection_plots/`.

Caption: “Archived loose-preselection examples from v5.0.5; signal MC scaled for display. Updated hit and vertex requirements are listed at left.” Keep each complete plot including axes and MC scaling legend.

## Slides 6–7: support and search intervals

Asset: `datasets_current_support_and_search.png`, an unmodified copy of the note's `figures/v5_dataset_mass_distributions_log.png`. It was visually checked: the 2015 upper dashed line is at 100 MeV and the 2021 shaded support begins at 36 MeV. Source code corroboration: `study_results/v5p0p4_analysis_note_20260911/scripts/make_v504_figures.py`, function `contextplots()`.

For the deck's wide existing frame, use `datasets_current_horizontal.png` (2520×680), a fresh three-column rendering of the identical pinned native spectra and the same supports/search limits. Created by `build_v505_assets.py`; no rebinned data or scientific inputs changed.

| Dataset | GP support [MeV] | Released search [MeV] |
|---|---:|---:|
| Full 2015 | 14–135 | 19–100 |
| Full 2016 | 30–210 | 39–180 |
| 2021 10% | 36–300 | 50–250 |

Caption: “Shading: GP training support. Dashed lines: tested search interval.” The deck separately proposes a 56 MeV lower search edge for 2021 100%; do not relabel the released 10% interval as 56 MeV. Companion `v5_dataset_mass_distributions_provenance.json` has a stale 90-MeV entry for 2015, so it must not be treated as final interval authority. The current figure and source code show 100 MeV.

## Slide 11: background profiling example

Asset: `2021_profile_78MeV.png` (1600 px high), cropped directly from the left column of vector `figures/v5_extraction_2021_peaks.pdf`, retaining both axes, both panels, and the 78-MeV title. The shared top legend was removed by this column crop; add compact native labels: black points = data; gold dotted = GP mean; blue dashed = profiled background; red = background + signal; purple dotted = fitted signal; blue shading = GP constraint width.

Optional full figure: `2021_profile_65_and_78MeV.png`. Optional second column `2021_profile_65MeV.png` and deficit example `2021_profile_71MeV_deficit.png` are also prepared.

Caption: “2021 10% at 78 MeV: the GP constrains the background; the local fit profiles its allowed displacement. Residuals subtract the same GP mean.” Small qualifier: “Bars: counting error. Shading: GP constraint width, not post-fit covariance.” Source: note `source/sections/v5_extraction_results.tex`, lines 39–42 and 85–90. Nominal local Z = 2.81 is optional; it is not a goodness-of-fit probability.

## Slides 13–14: fit examples and meaningful residual summary

No archived GP chi²/dof or KS scan with a defensible effective-dof/null calibration was found in the pinned v5.0.5 package, v5.0.4 provenance ledgers, or code. Traditional-polynomial deviances in the appendix belong to a different model and must not be relabeled as GPR GOF.

At root's request, `build_v505_assets.py` replays GP predictions using the saved v5.0.5 2021 spectrum, archived per-mass constant/length-scale states, log preprocessing, and ±2.25σ exclusion. It performs no hyperparameter optimization, signal fit, toy generation, or new-data access. BLAS threads are bounded to one.

Slide 13 asset: `2021_fixed_state_examples.png` (2600×1220), with windows at 60, 120, and 220 MeV; data, GP prediction and GP constraint width above, standardized residuals below. Plot annotations are Q/Nbin = 0.87, 0.78, 0.70. The residual displayed per bin is `(n_i-b_i)/sqrt(b_i+C_ii)`; residual bins remain correlated.

Suggested copy: “Same archived GP policy at 60, 120 and 220 MeV. Each displayed window is withheld from its background training.” Footnote: “Fixed-state replay of released 2021 10%; bars show counting error; shaded width is the GP constraint.”

Slide 14 asset: `2021_conditional_residual_scan.png` (2080×730), 201 test masses from 50 to 250 MeV. Definition:

`Q(m) = (n-b_GP)^T [diag(b_GP)+C_GP]^{-1} (n-b_GP)`

Plot `Q/Nbin`, where Nbin counts held-out bins; do not rename it chi²/dof. Range is 0.355–2.330. All covariance matrices are positive definite after adding the Poisson term (minimum eigenvalue 12028.99 counts²).

Suggested copy: “A covariance-aware check of withheld data against the GP prediction. Includes Poisson counting variance and correlated GP uncertainty.” Qualification: “Nbin is a bin count, not fitted effective dof. Conditional diagnostic; null distribution uncalibrated. No chi² or KS p-value is claimed.” Adjacent test windows overlap and are correlated.

CSV, vector PDF and `2021_conditional_residual_protocol.json` accompany each new plot. Pinned input SHA256: `fd4867a7e2df69d88d62cdaaed133096d015371e2fe5bd700596f411c8437294` for `study_results/v5p0p5_analysis_note_20260916/inputs/spectrum_2021.npz`.

## Feedback to preserve outside unrequested slides

- v5.0.5's event-selection narrative predates the user's new eight-hit electron wording; reconcile the note and histogram provenance before presenting the new selection as already validated.
- Keep 2021 released 10% search 50–250 MeV distinct from the proposed full-sample start at 56 MeV.
- The stale companion dataset-figure JSON and a residual 90-MeV sentence in the note's mass-resolution section should be repaired separately; they do not justify changing non-bracket slides.

Memory used only for routing: `MEMORY.md:147–182`, especially the v5.0.5 note pointer and search intervals; all displayed source facts were checked live. Corresponding rollout id: `01a0ac48-5e83-7e23-983c-a61dc7db6cec`.
