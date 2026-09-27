# Completed remote audit, 22 September 2026

The user-authenticated s3dfdtn SSH connection and second hop to iana were used. Outputs are under `/sdf/home/e/epeets/src/hps_gpr_v6p1_mc_signal_templates_20260922`. This completed audit supersedes the pre-access status in `selection_audit.md`.

## Input and extraction

All 26 ROOT files in the supplied v13 `ap_signal_prompt_smeared` directory were inspected: 12 generated masses, 40–260 MeV in 20 MeV steps. The latest `preselection` tree cycle is read once per file. All 17,867,674 selected candidate rows have positive unit weights, finite reconstructed masses, and `TargetConstrained` vertex type. Generated `true_ap.mass_` agrees with each directory's nominal pole within the recorded numerical spread. This identifies the generated event mass, not the ancestry of the reconstructed pair.

`vertex./vertex.invM_` in GeV is histogrammed at 0.1 MeV spacing over 0–400 MeV. There are 2–206 overflow entries per mass and no underflows. Overflow counts remain in metadata and the denominator for the all-selected analysis-support fraction; conditional histogram quantiles and display densities use the retained 0–400 MeV range. Source-conservation checks explicitly include overflow. These bookkeeping details do not change the fit shape conditional on the unchanged 36–299.75 MeV analysis support. Per-file SHA-256, sizes, modification times, ROOT cycles, entries and cutflows are stored in each mass JSON.

## Selection and smearing

The MC and observed productions share target-constraint and upstream `psum > 2.8 GeV` labels. Inspected MC timing labels use 5.7/9.2 ns where observed v16 uses 5.1/7.8 ns. The exact upstream production definitions and smearing chain were not recovered by bounded searches. Saved historical code is evidence of the search, not an authoritative v13 configuration. All electron and positron smear-ratio branches are one throughout the full extracted samples; this does not prove that the reconstructed masses are unsmeared. No additional smearing, recentering, hit requirement, timing selection or truth matching is imposed.

The scalar sum calculated from stored particle px/py/pz falls below 2.8 GeV for some candidates. It is not the authoritative upstream variable. A separate bounded read of the first 100,000 rows per source file finds zero failures in either stored `psum` or `psum_scalar`. Thus the particle-vector diagnostic alone does not demonstrate a selection violation. The full diagnostic and its bounded sampling scope are saved in `qa/truth_flag_diagnostic.json`.

## Signal association and interpretation

Both `ele_has_truth_link` and `pos_has_truth_link` are false in every row of that bounded read. Their unpopulated or unavailable associations cannot establish that the selected pairs are unrelated, and cannot identify A-prime daughters. The local historical `Track.h` distinguishes a truth-track reference from an MC-particle reference; even nonzero truth-track flags would require a documented daughter/parent mapping before proving A-prime ancestry.

The 40 MeV selected distribution is especially displaced: its conditional reconstructed median is about 102 MeV and only about 0.53% falls within the nominal primary window. Pulser overlay, combinatorial candidates, wrong-pair reconstruction and upstream response effects are possible explanations, not separately demonstrated causes. Their continuum cannot automatically be assumed to scale with the signal coupling. The v6.1 limits therefore report conditional substitutions of the supplied selected-candidate distributions in the inherited coupling coordinate, not calibrated A-prime exclusions or certified true-signal shapes.

Broad components outside the original blind window can contaminate background training under signal injection. Keeping the original observed-data GP prescription permits a paired observed comparison, but does not test that feedback or establish limit coverage. The independent-bin resampling checks do not include correlations between multiple candidates from the same event.
