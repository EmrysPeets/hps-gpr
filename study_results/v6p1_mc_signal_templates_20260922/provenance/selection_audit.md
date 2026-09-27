# Local selection and smearing audit for v6.1

22 September 2026. Historical pre-access snapshot: this paragraph describes the initial local audit only; the completed remote audit is documented in `remote_selection_audit.md`. This is a bounded, read-only local audit. The requested remote MC directory `/sdf/data/hps/physics2021/preselection/v13/ap_signal_prompt_smeared` has not been inspected, and its selection, mass branch, weights and smearing are **not verified**. No MC-template fit or Gaussian/MC numerical agreement result follows from this audit.

## Observed-sample source

`/Users/emryspeets/Desktop/recoil_project/source/hpstr/make_final_10pct_invM_hists_uproot_only.py` identifies the observed input directory at lines 35–38 as `/sdf/data/hps/physics2021/preselection/v16/data_10pc_prompt_TC_psum2p8`. Its opening documentation states that the sample already carries the momentum-sum cut and that the parser reads only the invariant mass; branch aliases beginning at line 80 include `vertex.invM_` and `vertex./vertex.invM_`. It adds no event-level selection.

The v5.0.5 source `/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow/study_results/v5p0p5_analysis_note_20260916/source/sections/03_event_selection.tex`, lines 18–22, specifies prompt target-constrained vertices, `psum > 2.8 GeV`, no additional vertex-fit chi-squared requirement, and no inferred extra two-dimensional-hit requirement. The parser documentation `/Users/emryspeets/Desktop/recoil_project/source/hpstr/HPS_GPR_V4P2_10PCT_DIAGNOSTIC_PARSER.md`, lines 10–17, distinguishes algorithm reproduction from verification of the complete input file set; lines 37–41 explain that the stored track hit counts are three-dimensional.

## Locally inspected specimen

The available file `/Users/emryspeets/Desktop/gp_mods/10pct_2021/merged_hps_014745_job621_merge-batch-2.root` contains a latest tree `preselection;2` with 30,353 entries and an older `preselection;1` with 18,732 entries. **Read the latest cycle only; do not add the cycles.** This file is a schema specimen, not proof of the complete frozen observed input set.

Verified branch names include `vertex./vertex.invM_`, `vertex./vertex.type_`, `weight`, `ele_p_smear_ratio`, `pos_p_smear_ratio`, `ele./ele.track_/ele.track_.n_hits_` and the corresponding positron branch. The first ten inspected rows have vertex type `TargetConstrained`, unit weights and unit electron/positron smear ratios. Particle corrected and uncorrected momentum branches also exist. These observed-data checks establish no MC smearing convention.

The specimen's `vertex_cutflow_h` labels describe electron momentum above 0.2 GeV and below 2.9 GeV, positron momentum above 0.4 GeV, track chi-squared/ndf below 20, electron/positron two-dimensional-hit thresholds of 8/10, timing windows of 6/5.1/7.8 ns, `psum > 2.8 GeV`, and vertex momentum below 4.0 GeV. Its vertex chi-squared label says below 30, but the cumulative count remains 118,725 before and after that step. Thus the label alone does not establish an active cut. Likewise, the two-dimensional-hit labels do not justify applying those numbers directly to the stored three-dimensional `n_hits_` fields. Actual production configurations are required to reconcile the semantics.

## MC requirements and smearing

No authoritative v13 production configuration or matching local A-prime ROOT sample was found in the bounded search. Generic local hpstr configurations include unconstrained-vertex analyses and must not substitute for the v13 production definition.

The v5.0.5 resolution source, lines 325–335, describes nominal target-constrained data/MC broadening using `2.186/1.745`, following curvature and hit smearing. The pinned Gaussian resolution coefficients are `[0.00184825, -0.001375, 0.085875]` for mass and sigma in GeV. Preserve the original primary mask and prompt-density conversion for this template comparison. Before constructing an MC template, establish whether the stored reconstructed vertex mass already includes each smearing step or whether a documented transformation is needed. Do not automatically multiply an already smeared MC template by another factor of 1.25, or apply stored momentum-smear ratios twice.

Minimum MC metadata: generated pole mass; reconstructed target-constrained mass branch and units; production/config identity; exact upstream selection and trigger treatment; smearing definition and application order; event weights; selected counts and sums of weights and squared weights; candidate multiplicity; file identities; generated mass grid; and declared template normalization, interpolation and any recentering. Preserve a reconstructed mass bias unless a separate correction is explicitly justified. A directory name is not sufficient selection or calibration evidence.

## Appropriate curve comparison

At common fitted masses, compare `R = UL_MC / UL_Gaussian`, using the same amplitude/coupling convention. Report the median, descriptive 16th–84th percentile range, extrema and their masses, median absolute fractional change, and root-mean-square log ratio. Compare signed likelihood roots and local Z through paired differences and report peak-location changes. These are descriptive summaries of correlated observed curves, not uncertainties or agreement p-values: both curves use the same counts and overlapping sidebands. Do not form an independent-point chi-squared or an agreement probability without a justified joint sampling covariance. Actual numerical metrics require the MC-template scan.

The companion JSON records exact local source identities. No remote data selection or smearing has been certified by this audit.
