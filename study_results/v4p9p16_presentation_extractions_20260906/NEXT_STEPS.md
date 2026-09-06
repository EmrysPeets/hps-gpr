# Continue at 30%, then 100%

The only 2021 input used here is the released 10%. These are observed-fit displays and conditional future means; no further events have been opened.

1. Identify a deterministic, disjoint additional 20% and retain event-membership/input hashes. Check exposure normalization, selection, acceptance and mass resolution against the original 10%. Do not count the original 10% twice.
2. Freeze the primary coordinates 66 and 92 MeV and the signal-width/background protocol before opening the increment. Declare whether 65/78 MeV individual excesses and 71/72 MeV deficits belong to a secondary validation family. No final significance threshold, sequential testing budget or model-family choice is prescribed by this exploratory note; these must be set before making formal claims.
3. Fit the additional 20% alone first. Compare its amplitude estimates with both the individual-2021 and common-fit estimates in `derived/dataset_consistency.csv`. The 92 MeV common rate is already less consistent across years than 66 MeV. Retain the full likelihood, covariance and residual diagnostics. Do not include the selected old 10%, 2015 or 2016 in this independent confirmation statistic.
4. Inspect cumulative 30% only after preserving the independent-increment result. Keep 2015 and 2016 fixed in the combined fit. Use the same physical shared-coupling convention, with exposure-appropriate signal and background uncertainties. Treat the 30% and later 100% cumulative looks as correlated.
5. Validate background interpolation and signal response with declared predictive controls and detector/selection studies. Look at deficits as well as excesses. Resolve or carry forward the 2016 source-fit waiver, development overlap, 75--85 MeV transition and numerical exception. Do not select a background family using the observed peak height or smallest p-value.
6. For new sampling calculations, start with ten complete spectra per declared generating hypothesis in fresh seed namespaces. Reuse each complete spectrum throughout its scan; separate random streams by dataset and hypothesis. Preserve fixed toy-ID ranges and restartable chunks. Record per-spectrum wall time and exact-solver/accelerator gates before scaling to 100 or 1,000 trials. Additional GP draws only improve precision within a validated approximation.
7. Rebuild the response basis and joint covariance for the new exposure; the current GP maximum bank does not calibrate 30% or 100%. Background-only toys can validate discovery-score tails. Calibrating upper-limit coverage/CLs additionally needs the chosen signal-plus-background hypotheses. Account for the declared mass, direction, method and sequential-look family before formal claims.

## Present study reproduction

Use a new derivative directory if changing the protocol. For a faithful replay, preserve the sealed manifest, use the exact upstream inputs and run from the repository root:

```bash
python3 -B study_results/v4p9p16_presentation_extractions_20260906/extract.py
python3 -B study_results/v4p9p16_presentation_extractions_20260906/make_figures.py
python3 -B study_results/v4p9p16_presentation_extractions_20260906/make_report.py
```

Rebuilding requires fresh product validation, rendered-page review, independent audit verification and a new manifest. The current means and precision tables are deterministic and require no new toys. `derived/exposure_contract.json` states all extrapolation assumptions. The counting envelope is the variation of the added sample, not a calibrated background-model or fitted-amplitude error band.
