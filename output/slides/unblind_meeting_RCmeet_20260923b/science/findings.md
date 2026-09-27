# Expanded sideband-only diagnostic for slide 14

The display compares the observed sideband fit with a **pointwise conditional reference across the 2021 10% search region**. It uses 41 centers from 50 to 250 MeV at 5 MeV intervals, plus the previous 78 MeV example. The supplemental anchor preserves continuity; this is a 42-center display, not a full 1 MeV scan.

For each center, GP training excludes its ±2.25 sigma window and uses every other available support bin. The Poisson deviance evaluates only search bins between 50 and 250 MeV outside that same window:

`D_side = 2 sum_side [n_i log(n_i / b_i) − n_i + b_i]`.

`N_side` counts those bins. It is **not an effective number of fit degrees of freedom**. The available support has edges 36–299.75 MeV. The mask is applied identically when fitting observed data and toys. Boundary centers retain the same rule, so part of an excluded window can lie outside the search interval while still being excluded from training.

The same 256 complete Poisson spectra are reused at every center. Their source is the fixed, observed-data-derived nominal GP source used in v5.9.5. Each center uses its own archived kernel state, held fixed across toys; targets and count-dependent training noise are recomputed in every replay. The 78 MeV toy column is reused from the previous slide revision, with its first and last entries verified by direct replay. The three earlier observed values at 65, 78 and 120 MeV reproduce to 1e-12. There are no new random draws, signal fits or hyperparameter optimizations.

## Results

All 42 observed values fall inside their pointwise central 90% conditional bands. This is a descriptive result, **not a calibrated global acceptance statement**. The observed D/N ratios range from 0.8399 at 190 MeV to 0.9588 at 235 MeV. The conditional medians range from 0.9436 to 0.9572, which illustrates why one is not an exact target here. The number of evaluated sideband bins ranges from 275 to 312.

| Center | Observed D/N | Toy median | Pointwise central 90% band | Lower-tail count | Upper-tail count |
|---|---:|---:|---|---:|---:|
| 78 MeV | 0.8522 | 0.9552 | [0.8447, 1.1041] | 19/256 | 237/256 |
| 190 MeV | 0.8399 | 0.9436 | [0.8245, 1.1027] | 18/256 | 238/256 |
| 235 MeV | 0.9588 | 0.9456 | [0.8286, 1.1144] | 143/256 | 113/256 |

The last two centers illustrate the smallest and largest observed ratios on this sampled grid; their selection is descriptive. Across the grid, lower-tail fractions range from 18/256 = 0.0703 to 143/256 = 0.5586, and upper-tail fractions range from 113/256 = 0.4414 to 238/256 = 0.9297. At 190 MeV the lower-tail add-one estimate is 19/257 = 0.07393, with exact 95% interval [0.04220, 0.10885]. The finite toy sample leaves uncertainty near a 5% pointwise threshold; there is no evidence here sufficient to label the low ratios overfitting.

The 41 new centers required 10,496 toy GP replays, plus two verification replays at 78 MeV. Total measured replay time was 55.84 seconds on one BLAS thread. The saved 78 MeV toy metrics were reused. No wider 1 MeV scan was needed for this display.

Suggested slide sentence: **Observed sideband deviance stays within the pointwise toy reference across the sampled search region. Compare with the fitted-toy reference, not an assumed target of one.**

## Reading the plot

`assets/slide14_sideband_scan.png` shows the observed curve in red, the conditional toy median in dashed blue, and the central 90% toy band (5th–95th percentiles) in light blue. Lines connect the sampled centers. The bands are **pointwise**, not simultaneous bounds on the entire curve. Neighboring centers reuse almost all the same data and exactly the same toy spectra; their diagnostics are strongly dependent.

A useful value is one compatible with the reference for that center, interpreted alongside its scientific purpose. A value near the conditional median is typical under the frozen source and fitting procedure. A value above the upper band indicates unusually large sideband residuals at that center under this reference; a value below the lower band is unusually small. There is no automatic requirement that the ratio equal one. Fitted sidebands can have reduced deviance because fitting adapts to their noise, and the reference repeats that adaptation.

An unusually small value could motivate checks of fit flexibility, training/evaluation dependence, or source mismatch. It does not by itself prove overfitting. Demonstrating predictive overfitting requires a held-out diagnostic and appropriate calibration. This display evaluates the same sidebands used to train the GP; it does not validate predictions inside the excluded signal window. The archived kernel states were originally derived from observed data, whereas these toys hold those states fixed. Repeating kernel selection in toys would test another source of fitting adaptivity; this bounded reference does not include that step.

`data/sideband_center_summary.csv` records both upper- and lower-tail counts, raw fractions, add-one estimates and exact 95% binomial intervals at every center. The upper tail answers how often the toy deviance is at least as large as observed; the lower tail answers how often it is at most as large. The intervals cover finite toy-count uncertainty only. They do not include source or model uncertainty. `assets/sideband_lower_tail_reference.png` is an optional supplementary plot of the lower-tail fractions and their pointwise intervals.

Counting centers outside a pointwise band does not create a global goodness-of-fit test. No simultaneous band, multiple-comparison correction, unconditional model calibration or resonance significance is supplied. The 78 MeV example was selected in earlier observed scans; it remains descriptive here.

## Reproduction and provenance

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 /Applications/Xcode.app/Contents/Developer/usr/bin/python3 output/slides/unblind_meeting_RCmeet_20260923b/science/expand_sideband.py
```

The script limits BLAS to one thread and checkpoints after every center. A repeated run verifies source hashes and resumes completed centers without repeating their calculations. `data/checkpoint.npz` retains the paired 42×256 toy array; `data/paired_toy_deviance.csv` stores the same vectors in readable form. JSON summaries, exact per-center CSVs, PNG/vector-PDF plots and source/code hashes are retained under this directory. All input and parent-study hashes are unchanged.
