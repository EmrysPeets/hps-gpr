# HPS-GPR v4.2 postprocessing

This directory contains the publication postprocessing for the accepted v4.2 combined 2015 100% + 2016 100% + 2021 10% search.

## Scope

- The combined search uses exactly 300 finite shared conditional fixed-GP background-only limit pseudoexperiments at each of 232 masses.
- Auxiliary standalone and pairwise expected-limit bands select parent toy indices 0--99 at each mass, reusing the same campaign pseudo-spectrum whenever that campaign appears at a common toy index.
- The auxiliary production contains 415 standalone and 224 genuine pair-overlap mass rows, each with exactly 100 finite limits. Pairwise display curves reuse the corresponding standalone rows outside the overlap, producing three union-range panels with 606 display rows.
- Inner 90% CL limits use asymptotic tilde-q_mu CLs with count_scale combination.

## Statistical interpretation

The central 68% and 95% bands are descriptive quantiles of a conditional fixed-GP limit ensemble. They are not coverage-calibrated confidence intervals. A shared toy index combines independently drawn active-dataset spectra at one mass; it is not a coherent correlated full-mass scan.

For the standalone scopes, the cached likelihood reproduces the reviewed native
total-amplitude root convention and reports every toy amplitude through the
same frozen rowwise `A_up/eps2_up` conversion as the authoritative observed
limit. Pairwise overlap rows retain the accepted combined epsilon-squared root
convention. Five representative masses reconstruct the complete accepted
300-parent observed limit, quantiles, mean, and tail counts.

The p_strong, p_weak, and p_two curves are fixed-mass observed-limit diagnostics. A raw 0/300 count is below one-count resolution and is not exact p=0. The local p0 curves are asymptotic discovery diagnostics. The Sidak curves are analytic resolution-spacing references and are not scan-toy calibrations.

The 2016 length-scale upper factor of 12 was accepted after the v4 observed saturation diagnostic. Consequently these p-values remain conditional, post-selection diagnostics rather than a pre-specified discovery claim.

The minimal-visible reinterpretation multiplies the observed limit and every combined toy-limit quantile by the same visible-width factor above the dimuon threshold. It does not alter yields, p-values, or the observed/median ratio.

## Numerical minima

- Combined local minimum: p0=3.2591825213e-05, Z=3.99321, mass=[65] MeV.
- 2015 standalone local minimum: p0=0.000846282590742, Z=3.13947, mass=[51] MeV.
- 2016 standalone local minimum: p0=0.000308781407384, Z=3.42378, mass=[90] MeV.
- 2021 10% standalone local minimum: p0=1.05702045272e-05, Z=4.25249, mass=[65] MeV.

Machine-readable validation, summaries, provenance hashes, reviewed tables, and the figure manifest are under `derived/`. Publication PDFs and 300 dpi PNGs are under `note_figures/`.

## 65 MeV observed extraction

The `note_figures/extractions_m065/` directory contains the exact fixed-state
observed extraction at the combined local minimum. The wide figure shows each
dataset over plus or minus five mass resolutions, including observed sideband
data on both sides of the former blind region, the fixed-GP background, and the
standalone and simultaneous signal shapes. The companion figure shows
background-only, standalone, and shared-fit conditional Pearson residuals plus
the standalone-versus-shared coupling comparison.

The exact shared fit gives
`eps2_hat = (6.42706 +/- 1.61009)e-6`. The feature is predominantly carried by
the 2021 10% sample, is modestly supported by 2015, and is opposed by 2016 at
the common-coupling yield. The residuals are correlated fit diagnostics, not
independent per-bin significances. The extraction remains a fixed-card local
asymptotic result and does not acquire a toy-calibrated global significance
from the 300 mass-local limit pseudoexperiments.

The extraction reconstructs from the compact production ledger
`study_results/v4p1_2016_ls_upper_optimization_20260804/derived/observed_gp_states_k12_reviewed.csv`.
This preserves the exact 2016 length-scale value
`0.4037156638852792`; the enriched presentation table rounds that coordinate by
one floating-point unit and is used only for validation metadata.

## Reproduction

Run these commands from the repository root. The production runner requires the explicit confirmation flag and refuses cards, reviewed ledgers, toy counts, or closure reports outside the declared v4.2 state.

```bash
python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/build_v4p2_individual_ledger.py

python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/benchmark_cached_profile_closure.py \
  --json-out study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/derived/cached_profile_closure_v4p2.json

python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/run_combined_bands_cached_fixed_reviewed.py \
  --closure-report study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/derived/cached_profile_closure_v4p2.json \
  --confirm-production

python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/postprocess_v4p2.py

python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/make_presentation_limit_figure.py

python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/make_m065_extraction_figures.py

python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/run_standalone_pairwise_bands100_fixed_reviewed.py \
  --workers 6 \
  --confirm-production

python3 study_results/v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805/make_standalone_pairwise_band_figures.py
```

The reviewed production CSV has SHA-256 `b90768ab361928c63f57b3981d424fd36506893da2447e40824acdf3d20081c2`; the v4.2 configuration has SHA-256 `5a52abb41896b161bd6dd8f66859737b6d98ea7d40d2eb8d8c677c3161ed6055`.
The authoritative pass/fail gate is `derived/postprocessing_validation_v4p2.json`.
