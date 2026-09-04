# HPS-GPR v4.9.8 Harvard selected results

This directory contains a reproducible, presentation-only selected-results
assembly.  It combines reviewed observed curves from existing releases without
changing any fit, limit, or local-p-value result.

## Principal outputs

- `pdf/HPS_GPR_Harvard_Writing_Sample_Selected_Results.pdf`: writing sample with
  the new selected-results section.
- `figures/individual_results_triptych.pdf`: five individual observed scans.
- `figures/combined_results_triptych.pdf`: three pairwise and one all-three
  historical v4.2 shared-coupling scans.
- `figures/asymptotic_pvalue_series.pdf`: the requested local asymptotic p0 series.
- `figures/historical_all_three_m065_extraction.pdf`: the 65 MeV historical
  all-three extraction.
- `derived/selected_result_curves.csv`: no-band curve ledger used by the plots.
- `derived/minima_summary.csv`: formal and interpretation-eligible local minima.
- `derived/result_state_ledger.csv`: explicit source-state map.
- `derived/source_manifest_sha256.csv`: immutable copied-input hashes.

The all-three label always means full 2015 + full 2016 + 2021 10%.  The
all-three curve is restricted to its common 50--90 MeV overlap.  Combination
signal-yield limits in the upper panel are a derived display coordinate,
`eps2_90 * sum(A_up / eps2_up)`, evaluated from the reviewed historical individual
normalizations.  The shared-coupling epsilon-squared limits and p0 values are copied
unchanged from their reviewed combination tables.

## Result-state separation

Full 2016 and all combinations use the accepted historical v4.2 state.  The current
2021 10% individual curve uses v4.9.5 support 36--300 MeV; it is not mixed into the
historical combinations, which retain the v4.2 2021 support of 40--300 MeV.  The
v4.9.7 scaled-to-full 2016 support study did not pass its frozen selection rule and
contributes no observed result.  See `STUDY_SCOPE.md` for the boundary.

## Rebuild

From the repository root:

```bash
python3 study_results/v4p9p8_harvard_selected_results_20260902/build_selected_results.py
cd study_results/v4p9p8_harvard_selected_results_20260902/source
mkdir -p ../qa/build_selected_results ../pdf
tectonic -C --keep-logs -o ../qa/build_selected_results writing_sample_selected_results.tex
cp ../qa/build_selected_results/writing_sample_selected_results.pdf \
  ../pdf/HPS_GPR_Harvard_Writing_Sample_Selected_Results.pdf
cd ../../../
python3 study_results/v4p9p8_harvard_selected_results_20260902/validate_release.py
```

The build is offline after the local Tectonic cache has been populated.  The
validator checks exact grids, source hashes, numerical identities, result-state
separation, PDF semantics, and the recorded rendered-page review.

