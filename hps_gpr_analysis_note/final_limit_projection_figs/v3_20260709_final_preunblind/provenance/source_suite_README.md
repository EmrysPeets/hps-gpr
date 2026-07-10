# Corrected Final Pre-Unblind Plot Suite

Source directory: `/Users/emryspeets/Desktop/gp_mods/combined_15_16_10pct_21_1pct/final_pre_unblind`
Output directory: `/Users/emryspeets/Desktop/gp_mods/combined_15_16_10pct_21_1pct/final_pre_unblind/summary_combined_all_note_corrected`

## Corrections Applied
- Observed scan/table disagreements recorded: 11 rows.
- Expected-band repaired mass points: 3.
- Dimuon correction applied to primary eps2 plotting columns for 39 rows above 211.3167 MeV.
- The optional 2021-only companion plots have their own observed and expected-band repair audit.
- Plot titles avoid dimuon-threshold wording; use `caption_suggestions.md` for the note caption.
- Sigma reference lines are added only when they fall inside the visible axis range.

## Main Outputs
- `ul_bands_eps2_obsexp.png/pdf`
- `combined_90cls_dimuon_corrected_eps2.png/pdf`
- `ul_observed_only_eps2.png/pdf`
- `2021_UL_sig_yield_bands.png/pdf`
- `2021_UL_eps2_bands.png/pdf`
- `p0_analytic_local_global.png/pdf` and `Z_local_global.png/pdf`
- `2015_*`, `2016_*`, and `2021_*` local/global p0/Z figures
- `ul_pvalues.png/pdf` and `ul_pvalues_components_local_global_refs.png/pdf`
- `expected_band_repair_diagnostics.png/pdf`

## Corrected Tables
- `combined_ul_bands_combined_all_corrected_ee_channel.csv`
- `combined_ul_bands_combined_all_with_dimuon_columns.csv`
- `combined_ul_bands_combined_all_dimuon_for_plotting.csv`
- `plot_repair_audit.csv`
- `combined_ul_bands_2021_repair_audit.csv`
- `dimuon_correction_factor_table.csv`
