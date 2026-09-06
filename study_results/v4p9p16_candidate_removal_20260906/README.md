# v4.9.16: candidate removal and traditional signal searches

Replacing the two selected GP peak regions changes the local peak–dip pattern but does not eliminate the broader oscillatory response. With both regions filled by the primary conditional GP expectation, remote observed variation retains 67%, 78% and 92% of its original standard deviation in full 2015, full 2016 and released 2021 10%. The corresponding reference-spectrum responses retain 110%, 93% and 86%. The pattern is not invariant: the 2021 root at 71 MeV moves from −4.02 to −0.72, and wider exclusions reduce that year's reference variation substantially.

Traditional fits on the original counts are strongly background-dependent. Baseline roots at the selected masses are:

| Dataset | Mass [MeV] | GP root | Traditional root | Traditional D/dof |
|---|---:|---:|---:|---:|
| 2015 full | 51 | +3.139 | +0.131 | 141.7/131 |
| 2015 full | 21 | +2.516 | +5.580 | 92.5/50 |
| 2016 full | 90 | +3.425 | +1.998 | 125.3/113 |
| 2016 full | 117 | +3.279 | +1.525 | 165.6/155 |
| 2021 10% | 78 | +2.809 | −1.350 | 29.2/24 |
| 2021 10% | 65 | +2.396 | +1.960 | 17.4/22 |

The large 2015/21 root has poor overall fit quality and changes substantially with degree and window. The 2021/65 degree-two variant produces root 8.46 with D/dof 28.10, illustrating model failure. No variant was selected by its probability. These are nominal local references at masses selected with the same data, not independent confirmations or calibrated particle significances.

## Products

- The full updated note is `output/pdf/v4p9p16_candidate_removal_20260906/HPS_GPR_Analysis_Note_v4p9p16_Candidate_Removal_and_Traditional_Searches.pdf` from the repository root. The new section is Section 5, “Candidate removal and traditional signal searches,” pages 11–20. Earlier Figure 1, limits, extractions, probability/echo discussion and low-mass study are retained.
- `PROTOCOL.md` records the numerical choices fixed before the interventions.
- `derived/{year}/inputs.npz` contains every complete modified spectrum and its source-specific joint latent replacement moments.
- `derived/{year}/scans.csv` contains all 42 spectra at every mass: 72/142/201 coordinates, 17,430 profile tests in total.
- `derived/{year}/components.npz` stores all deterministic likelihood components and every conditional replica at the two selected masses.
- `derived/oscillation_metrics.csv` retains every full and remote metric. The remote criterion requires the entire native fit window to avoid both primary holes.
- `derived/selected_root_changes.csv` and `remote_summary.csv` are compact interpretation tables.
- `traditional/derived/fit_summary.csv` and `points/` contain all thirty traditional fits to original observed counts, with parameters, covariance, native arrays, convergence and actual fit-window edges.
- `figures/` contains the removal scans; `traditional/figures/` contains count/residual displays and all-variant presentations.
- `review/` contains independent HEP interpretation and intervention checks. `qa/numerical_validation.json` independently reconstructs 5,160 GP likelihoods and all thirty traditional fits.

The fake counts are conditional replacements, not an independent null ensemble. Each replica has fixed untouched data and frozen kernels. Polynomial interpolation and ±3σ holes are retained as model checks, including unsuccessful fills. No additional 2021 events were opened, and no probability ensemble or upper limit was recalibrated.

## Reproduction and validation

From the repository root, verify the saved numerical products without generating new toys:

```bash
python3 -B study_results/v4p9p16_candidate_removal_20260906/verify_numerics.py
python3 -B study_results/v4p9p16_candidate_removal_20260906/verify_display.py
python3 -B study_results/v4p9p16_candidate_removal_20260906/review/independent_intervention_audit.py
```

To reproduce in a new derivative directory, use `run_removal.py inputs --dataset YEAR`, then `pilot`, then `scan` for 2015, 2016 and 2021. Only run `metrics` after every complete scan QA exists and its row counts are verified. The deterministic streams use seed namespace 4916160906 and fixed replica IDs 00–09. Run the traditional script only with the recorded persistence trigger, as documented in `traditional/README.md`. Preserve the sealed release before rerunning.

The report build uses cached Tectonic resources. `make_figures.py`, `traditional/make_figures.py`, `traditional/make_paper_figures.py` and `make_report.py` regenerate figures and LaTeX/PDF from completed fits. Run `validate_products.py` to render and check the assembled report; then inspect those renders before updating the visual QA and resealing. A new PDF has new bytes and requires a refreshed manifest. `seal_release.py` creates the final SHA-256 inventory and immediately verifies it.

## Interpretation and continuation

The interventions establish influence under a specified fill, not the physical identity of the removed events. Traditional fits demonstrate model dependence, not that polynomial backgrounds are automatically correct. Formal inference still requires a qualified background family, spurious-signal and injected-signal checks, and accounting for the full mass/model search and sequential data looks. More conditional replacements could improve description of interpolation variability, but would not supply discovery calibration.

Before a 30% 2021 checkpoint, freeze masses, widths, model checks and the complete predicted response. Examine the disjoint additional 20% separately as well as the cumulative 30%. The growth of a peak or its fitted echoes alone cannot distinguish a particle from persistent detector/background effects. This study supplies no new sensitivity projection or permission to change the released data fraction.
