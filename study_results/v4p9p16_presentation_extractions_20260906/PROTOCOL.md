# v4.9.16 presentation extraction extension

This derivative preserves the frozen v4.9.13--v4.9.16 studies and the shared checkout. It uses only the released 2015 full, 2016 full and native 2021 10% spectra. The previous global and deficit studies were merged through PR 67 before this extension was started.

## Selection fixed before new extraction

Select the two largest positive local maxima of the dense observed profiled signed root in each individual dataset and in the full combined union. Include endpoints, explicitly flagged. Order ties by increasing mass and require separation greater than 2.25 times the larger mass resolution at the two coordinates. For the combined rule use the largest active dataset resolution. This chooses 2015: 51, 21 MeV; 2016: 90, 117 MeV; 2021: 78, 65 MeV; combined: 66, 21 MeV. The 21 MeV point is covered only by 2015. Also display 92 MeV, the second leading separated combined peak among coordinates with at least two active datasets, for a second 2021 exposure comparison.

Display the deepest negative root in the combined scan (72 MeV) and each individual scan (2015: 19, 2016: 102, 2021: 71 MeV). Reconstruct 76 and 83 MeV as explicitly separate stress-centering diagnostics: these are not selected as leading observed signal/deficit amplitudes. No plot choice changes a fitted mass, signal width, binning used in inference, kernel or background support.

## Fit and display contract

Reconstruct the archived predictions and conditioned covariance through the same frozen runtime and dense Poisson/Gaussian solver as v4.9.16. Require prediction-state hashes to agree exactly and observed roots to agree within 2e-5 with the frozen dense scan. Require the signal and background components to sum to the fitted expectations and the per-dataset signal yields to sum to the common-amplitude fit. Expected Poisson counts must remain positive. Keep negative auxiliary amplitudes signed; they are not physical negative couplings.

For individual display bins, choose the nearest positive integer number of native bins to half the dataset mass resolution; anchor grouping at the original first histogram edge. Retain only whole groups contained in the actual likelihood window. No phase, width or grouping is chosen from residuals. For summed multi-dataset displays use whole native bins on the common 1.25 MeV lattice anchored at 36 MeV and restrict to bins contained in all participating likelihood windows. This display sum omits boundary pieces and is not the statistic used for the combined fit. Export all mapping matrices and retained-bin fractions. The likelihood always uses every original fit bin.

Show observed counts, the original GP mean, the background profiled with signal, and their signal-plus-background prediction. Show data minus that profiled background with its fitted signal below. Curves are confined to the fitted window. Counting errors are sqrt(N) (these counts are large); they omit fitted-background uncertainty and inter-bin correlations, and residuals are not independent pulls. Save the original covariance and the exact likelihood components for auditing. At each joint coordinate, fit each dataset separately as a diagnostic of amplitude consistency and keep those estimates distinct from the common-coupling contributions.

## Exposure illustrations and the 30% step

Only 2021 changes exposure: the reference is 10%, then an independent additional 20%, cumulative 30%, and cumulative 100%. Leave 2015 and 2016 unchanged. Use the common-coupling fitted rate at 66 and 92 MeV as explicitly selected, illustrative persistent-signal hypotheses; also document the individual 2021 peak locations. At 21 MeV there is no 2021 coverage or gain.

For the new fraction f-1 in units of the original 10%, means are (f-1)B10 or (f-1)(B10+S10). Conditional cumulative means retain the actual first sample: N10+(f-1)B10 or N10+(f-1)(B10+S10). Future spectra and uncertainty envelopes are expectations, not observed events, generated toys or significance estimates. Counting uncertainty of the added sample does not include background-model uncertainty. Selected fitted amplitudes can be upward biased.

A separate statistics-dominated precision illustration uses I_d = S_d^T[diag(B_d)+C_d]^-1 S_d and I(f)=I_2015+I_2016+f I_2021, explicitly assuming the entire count covariance scales as f and unchanged resolution/acceptance. It is not a sensitivity, global-p-value or systematic-floor forecast. No stored global tail is reused at a new exposure.

## Validation and release

Require numerical reconstruction, bin-integral and covariance checks, an independent HEP/statistical review, semantic PDF checks and rendered-page inspection. Retain reusable PDF/PNG figures, LaTeX source, numeric tables, scripts, a provenance ledger and a separately sealed artifact manifest. Do not modify the parent releases or open further 2021 data. A continuation plan freezes the locations and tests on the additional 20% before examining 30% and 100% cumulative samples.
