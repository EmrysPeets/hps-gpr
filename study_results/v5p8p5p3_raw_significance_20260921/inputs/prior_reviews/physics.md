# Particle-physics analysis review and bounded new controls

Completed 2026-09-21, using the current checkout and immutable copies of prior source/data. All new extraction masks are ±2.25 sigma. No width scan, new observed release, calibrated discovery probability, coverage assertion or reach replacement is supplied.

## Main findings

1. v5.8.2 Figure 3 is easy to misread because it overlays an unstandardized deterministic root and two standardized toy moments. The colored line is r(B), the black line mean((r_toy−a)/s), and gray line SD((r_toy−a)/s). The source GP fits every observed bin with the 76-MeV kernel; the extraction masks ±2.25 sigma and uses mass-dependent reviewed kernels. The deterministic offset measures their imperfect mutual reproduction. It does not identify a signal or an amount of counts to subtract. `physics_fig3_explained` separates these quantities using the actual saved 2016 arrays.
2. The source-signal absorption concern is real and separately demonstrated: v5.8.1 source rebuilding absorbs ~88% window yield while extraction from a fixed nominal source recovers ~96%. Source learning is not resolved by widening only the extraction mask. Existing toy agreement is conditional on one fixed, data-derived source.
3. The high-psum candidate is 2021 `final_1pct_invM_v7.root:preselection/h_invM_psumgt2p8_8000`, SHA256 `412026fc37b65066d906c1d4acab524d274ee9d215960fb65a0b728d70c157a8`. It is not the historical unqualified `final_1pct_invM.root` input. The source key indicates high psum but neither selection equivalence to current prompt-TC native10, exact exposure nor overlap is established. Source is 4.4755% of all native10 visible counts, 4.4827% of current rebinned support counts and 5.1485% at 87–97 MeV. None is a luminosity ratio. It cannot directly set a 2016 offset.
4. New common-method conditional comparison: 603 deterministic states = 201 integer masses×three means; 768 Poisson fits = two target-scale means×64 coherent spectra×six anchors. RMS a is 0.055418 native high1, 0.028348 its fixed mean×10 and 0.257595 native10. The low source RMS demonstrates self-consistency, not a physical null. Mean and covariance of source estimation remain unpropagated.
5. Nominal×10 high1 GP/native10 GP ratio varies from 0.132335 to 0.516075 over 50–250 MeV. Thus shape as well as normalization differs. No normalization adjustment is silently applied.
6. New source-injection transfer: fixed-source extraction recovery is .966710/.966782 at 76/90 MeV. Injecting one tenth of that target-scale signal into the smaller source, rebuilding and scaling absorbs .840383/.778294 of window yield and .690233/.638411 of the Poisson-weighted full-template component. Statistical independence would address reused fluctuations but cannot by itself stop a genuine common signal entering the source.
7. A defensible independent-source procedure needs audited selection/exposure and overlap, fixed choices, source-contamination tests, and an outer repeated-source ensemble or equivalent source uncertainty propagation. For a specified underlying truth, jointly generate the control and target with correct exposures/overlap, fit the control source, and analyze the corresponding target; generating the target from that noisy source estimate would conceal source error. Simple illustrative variance of Ntarget−kNsource for independent Poisson inputs is (k+k²)lambda, versus k lambda if the source is treated fixed; this is not a direct formula for the GP scan variance.

## Delivered material

- `source/physics.tex`: report-ready explanation and prospective validation protocol.
- `figures/physics_fig3_explained.{pdf,png}`: old Figure 3 unpacked, actual stored 2016 roots and 256 scans.
- `figures/physics_source_transfer.{pdf,png}`: high-psum transfer, deterministic source response, and source-injection absorption.
- `scripts/physics_source_diagnostic.py`: full conditional source comparison, 480-second relative watchdog and STOP check.
- `scripts/physics_signal_transfer.py`: separate extraction/source-building injection tests, 120-second relative watchdog and STOP check.
- `scripts/physics_figures.py`: reproducible figures from saved arrays.
- `results/physics_source_manifest.json`, `physics_source_summary.json`, `physics_validation.json`: provenance, completed fit ledger and QA.
- `results/physics_sources.npz`, `physics_source_scan.csv`, `physics_source_toys.{csv,npz}`, `physics_signal_transfer.csv`: numerical artifacts.
- `inputs/physics_high_psum_1pct.root`: exact original source copied by root; original selection audit/manifest also copied under `physics_prior_selection_*`.
- `inputs/v5p8p2_nominal_gp_significance_20260917/`: root's archived engine and fields, with seven required engine inputs copied byte-for-byte by this agent.

## Reproduction and QA

Run the three Python scripts in the order diagnostic, signal_transfer, figures. NumPy, SciPy, pandas, uproot and Matplotlib are required. Scripts prefer the bundled engine snapshot. They never optimize new kernels. Current validated runtime: `/Applications/Xcode.app/Contents/Developer/usr/bin/python3`.

After switching from the live parent to the bundled engine, all three physics CSVs reproduced byte-for-byte. Native10 a and s match prior v5.8.2 integer-grid entries to 1.11e−16 and 2.22e−16. All 603 deterministic and 768 toy fits are finite with score below 2e−7; the additional injection fits also satisfy that criterion. Both figures were visually inspected at full rendered PNG size; labels, line meanings, ratios and legends are legible. Parent owns final PDF pagination/render QA.

An initial development attempt fed integer Poisson arrays to a legacy engine whose preprocessing preserves array dtype; it failed before yielding toy output. The final implementation explicitly converts generated counts to float, matching the original engine's saved-count convention. Final results contain 64 completed coherent scans per source, not accumulated attempts or failed samples.

## Memory use

A quick memory search identified prior source and significance caveats; all scientific values above were then verified from source code/current artifacts. Relevant memory route: MEMORY.md:36–39 and 351. The parent should append the usual single memory-citation block if the final response relies on this routing. No memory files were edited.
