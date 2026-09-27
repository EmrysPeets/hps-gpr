# Available subset inputs and conditional projections

Read-only source audit followed by bounded fixed-92-MeV fits. Exact source files are copied and SHA-256 bound in `inputs/subset_input_manifest.json`; numerical outputs and implementation are `derived/subsets_*.csv` and `scripts/subset_checks.py`.

## Real files and selection limits

| Sample | Source / histogram | Visible selected rows |
|---|---|---:|
| 2016 nominal 10% | `study_results/v4p9p7_2016_support_combined_100toy_20260902/inputs/source_2016_10pct.root` / `h_Minv_General_Final_1` | 7,483,101 |
| 2021 historical nominal 1% | `/Users/emryspeets/Desktop/gp_mods/data_input_21/final_1pct_invM.root` / `preselection/h_invM_8000` | 12,534,434 |
| 2021 nominal 1%, v7 | same directory, `final_1pct_invM_v7.root` / `preselection/h_invM_psumgt2p8_8000` | 6,336,937 |

The 2016 histogram name and 0.05 MeV native bins match the full-sample input; its counts are 10.220% of the full histogram and 10.233% in 87--97 MeV. This supports a nominal fraction, not an exact luminosity or event-overlap measurement. `config_2016_10pct_10k.yaml` identifies the historical remote `EventSelection_Data_10Percent.root` with this histogram. That historical configuration uses different analysis settings; the new checks instead inherit the parent 92 MeV kernel, support and resolution.

Neither 2021 1% ROOT file contains a selection manifest, TC marker, run list or event identities. The historical `config_2021_1pct_10k.yaml` refers to a remote file whose basename says `psumlt2p8`; that path does not prove the cuts of the local `final_1pct_invM.root`. The v7 histogram explicitly labels `psumgt2p8`, but its TC membership and exact processing equivalence remain unverified. Therefore **neither smaller file is identified here as a verified prompt-TC subset**. Both are shown separately.

By contrast, the native 10% parser `/Users/emryspeets/Desktop/recoil_project/source/hpstr/make_final_10pct_invM_hists_uproot_only.py` explicitly reads `/sdf/data/hps/physics2021/preselection/v16/data_10pc_prompt_TC_psum2p8`, with no further cuts. Its diagnostic-parser memo records 1,189 expected files and no established run/event identity branch. The current note, `v5p0p4_analysis_note_20260911/source/sections/02_datasets.tex`, explicitly states that the ROOT input does not determine luminosity; 10% denotes sample construction, from an approximately 160 pb^-1 parent run.

The historical and v7 1% inputs contain 8.853% and 4.476% of native-10% visible rows, respectively; 87--97 MeV ratios are 8.854% and 5.148%. Counts are not luminosity or signal-efficiency ratios. The v7 input even exceeds native10 in one 682.000--682.125 MeV bin (1 versus 0), so exact global histogram nesting fails. The other smaller histograms satisfy binwise non-exceedance, which does not prove event nesting. All smaller-versus-parent comparisons remain potentially overlapping controls; no joint independent likelihood or subtraction is performed.

## Implemented checks and projections

Each smaller histogram is exactly rebinned to its year's parent edges. The fixed parent kernel is conditioned on the smaller sample's own sidebands, and its native local density supplies the conversion. The inherited effective radiative fractions (2016: 0.0465; 2021: 0.0477) and resolution are explicit proxies until selection matching is established. Report event yields together with equivalent amplitudes. The three signed fits are weak: approximately (10.98 +/- 8.72), (0.762 +/- 4.38), and (1.99 +/- 5.73) times 10^-6. Frozen-response penalties are 0.0872, 0.0864, and 0.000116; small penalties here are weak constraint, not confirmation.

Same-selection 2021 projections use the released native10 count model only. Freeze either (a) the v5.5.0 response truth, beta=-2.90358457 and reference amplitude 8.402656456e-6, or (b) the positive native-2021 standalone fit. Center the continuum on the corresponding profiled parent background. At exposure multiplier t, construct Asimov counts n_A=t(b_fit+A S), response S_t=t S, and continuum b_t=t b_fit. Profile the background under both the free signal and zero-signal hypotheses. Use two separately labeled assumptions: C_t=t C for GP uncertainty that scales statistically, and C_t=t^2 C for fixed fractional GP uncertainty. These scenarios are not calibrated uncertainty bounds, and neither scales the parent fluctuations into future observed data.

At nominal 100% (t=10), the frozen-response truth predicts 96,800 full-template signal rows and sqrt(Q0)=4.782 or 3.266 under the two covariance assumptions. The native-fit truth predicts 78,206 rows and 3.863 or 2.639. These are conditional count-model diagnostics, not future measured significance or a global discovery forecast. The nominal 1% same-selection Asimov point is distinct from both real 1% histogram variants.

All fitted backgrounds remain positive; fixed-line and Asimov convergence scores are below 2e-7. Input snapshots, native-bin alignment, signal recovery, and both rendered figures were checked. No new mass scan, toys, observed full-2021 data, or unmeasured acceptance factor enters this calculation.
