# Version 4 wide-support figure provenance

These five figure pairs belong to the reviewed internal-analysis state that
combines the full 2015 and 2016 samples with the 2021 10% sample. The expected
bands are conditional fixed-GP diagnostics, not a full-procedure coverage
calibration. The analytic Šidák curve is separate from the 300 upper-limit
pseudoexperiments.

## Frozen analysis state

| Dataset | Search interval | GP fit support | \(k_{\min}\) | \(k_{\max}\) |
| --- | ---: | ---: | ---: | ---: |
| 2015 full | 19–90 MeV | 14–135 MeV | 1.0 | 8 |
| 2016 full | 39–180 MeV | 30–210 MeV | 0.9 | 8 |
| 2021 10% | 50–250 MeV | 40–300 MeV | 1.1 | 15 |

The frozen configuration is
`study_configs/v4_wide_support_2015full_2016full_2021_10pct_20260803/config_obsUL90_combined_wide_support_v4_observed_only.yaml`
with SHA-256
`16f686602514c5e156a8da83ed4f5facc1027788e184ef32ff72313f3fadd2a3`.

The run-host input hashes are:

- 2015: `58ce717cde753d8566c754a73cb056560ed19e781fe9a43e8634111cc746531f`
- 2016: `c2119a4ac9b91df9ae619857877b91cedba7fa6a58c10ece76b7d3a673a4e301`
- 2021: `3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4`

## Observed-state review

The three unchanged-card scans are represented by
`observed_attempt_01/`, `observed_attempt_02/`, and
`observed_attempt_03/` in the campaign directory. The authoritative outputs
are:

- `derived/observed_gp_states_reviewed.csv`
- `derived/observed_attempt_ledger.csv`
- `derived/observed_review_summary.json`
- `derived/unresolved_observed_states.csv`

All 415 dataset–mass states were resolved without interpolation. The 2016
length scale occupies its frozen upper bound at all 142 search hypotheses;
that fact is retained as a validation diagnostic.

## Conditional ensemble

`run_combined_bands_cached_fixed_reviewed.py` generated 300 finite
background-only pseudoexperiment limits at each of 232 mass hypotheses. The
GP state is reconstructed at the reviewed coordinates and held fixed; each
toy draws a conditional GP-posterior background intensity and then
Poisson-fluctuates the active spectra. The GP is not retrained toy by toy.

The cache was checked bit for bit against the ordinary solver in
`derived/cached_profile_closure_v4.json`. The production outputs are:

- `combined_bands_300toy_cached/ul_bands_combined_all.csv`
- `combined_bands_300toy_cached/ul_bands_combined_all_provenance.json`

The production CSV SHA-256 is
`33f576e09d0e603978b2e0b71eb608663b95806606ab056d1ba8f32c8f5b2cdb`.

## Figure mapping

All paths below are relative to
`study_results/v4_wide_support_2015full_2016full_2021_10pct_20260803/`.

| Note figure stem | Principal source | Interpretation |
| --- | --- | --- |
| `combined_observed_bands300_minimal_visible` | `derived/combined_bands300_reviewed.csv` | Observed 90% CL limit and conditional central 68%/95% toy-limit intervals. |
| `combined_limit_tail_pvalues300` | Same reviewed table and raw tail counts | Fixed-mass strong, weak, and bounded two-sided upper-limit diagnostics. A raw 0/300 is below one-count resolution, not exact \(p=0\). |
| `combined_local_p0_sidak_reference` | `derived/combined_bands300_sidak_reference.csv` | Local asymptotic \(p_0\) and separate analytic resolution-spacing Šidák reference. |
| `wide_support_search_kernel_audit` | `derived/observed_gp_states_reviewed.csv` | Search/support geometry and reviewed kernel-bound occupancy. |
| `wide_vs_narrow_observed_limit_ratio` | `derived/wide_vs_narrow_observed_limit.csv` | Matched observed support-dependence diagnostic; not a support optimization. |

The generator and machine-readable interpretation are
`postprocess_combined_bands300.py` and
`derived/combined_bands300_summary.json`.

## Outstanding gates

- support-matched functional-form closure for the widened domains;
- production-faithful pseudoexperiments that retrain the GP for direct
  coverage;
- a scan-wise maximum-\(q_0\) ensemble for a calibrated global discovery
  probability;
- the final full-2021 unblinding.
