# v5.9.5: what is shifted in the slide-50 null distribution?

The positive location of the scan maximum is expected from one-sided clipping and searching many correlated masses. It is not, by itself, a bias diagnostic. There is also a genuine **mass-dependent signed-root response under the pinned generating source**. The saved full Poisson refits reproduce that response; it should not be dismissed as only the look-elsewhere effect. Neither observation proves that this fitted source represents the physical null.

This audit keeps every observed raw root unchanged and preserves the ±2.25σ procedure. It uses the saved v5.8.5.3 fields and complete refit scans, with no likelihood refits, source changes, kernel optimization or new data access.

## Three random variables, three different centers

Let `r(m)` be the signed likelihood root, `Z(m)=max(r(m),0)`, and `T=max_m Z(m)`. In the saved response approximation,

`r* ≃ a + diag(s) W`, with `W ~ N(0,R)` and `a = r(B)`.

The reference response `a=r(B)` is deterministic. It need not equal the exact ensemble expectation `E_B[r(Y)]` because the complete fit is nonlinear; their difference is measured separately below.

Even a perfectly centered signed root `r~N(0,1)` gives `E[max(r,0)] = 1/sqrt(2π) = 0.39894`. A maximum over the correlated mass domain is larger still. The saved 2021 maximum distribution has median **2.44584**. Subtracting the model response only in a diagnostic null ensemble leaves a median of **2.29124**: a large positive scan maximum remains even after removing `a`.

For 2021, `a(m)` is not one constant upward offset: its mass-average is −0.00719, RMS 0.25076, and range −1.09594 at 71 MeV to +1.40020 at 50 MeV. The average can cancel local positive and negative offsets; it does not establish local centering.

## Slide-50 reproduction, unchanged raw observation

2021 native 10%, 50–250 MeV, 401 half-MeV hypotheses:

| Quantity at the observed 78 MeV peak | Result |
|---|---:|
| Raw observed signed root / local Z | 2.808645 |
| Conventional raw asymptotic local p | 0.00248752 |
| Deterministic response a | 0.576784 |
| Model response width s | 0.980025 |
| Mean of 256 signed-root refits | 0.547975 |
| 95% t interval on that mean | [0.423454, 0.672495] |
| Direct standard deviation | 1.011688 |
| 95% normal-theory SD interval | [0.930986, 1.107828] |
| Fixed-source Gaussian marginal tail at the raw threshold | 0.01138261 |
| Direct marginal exceedances | 2/256; 95% interval [0.000948, 0.027935] |
| Saved Gaussian scan exceedances | 50,183/200,000; add-one p = 0.25091875 |
| Gaussian tail 95% Monte Carlo interval | [0.249016, 0.252821] |
| Direct full-scan exceedances | 57/256 = 0.222656; 95% interval [0.173218, 0.278639] |

The positive-part model at 78 MeV has zero-atom probability 0.27808 and mean 0.74519, consistent with the independent review's direct clipped mean 0.74787. These are properties of the clipped variable, not estimates of the signed-root offset.

The marginal tail is `P_B(r*(78)>=r_obs(78))` under the same fixed source, not a replacement observed scan. The Gaussian global tail divided by this marginal tail is about **22.04**. The ratio using the raw asymptotic p is **100.87** and mixes marginal calibration with the mass-search penalty. Both calculations refer to the same raw threshold; no observed centering or scaling was performed.

**78 MeV was selected from the observed scan.** Its marginal numbers diagnose the selected feature and do not provide an independently specified fixed-mass claim. The global result is the appropriate mass-scan tail for the declared raw ordering, conditional on the source and scan policy.

## Does the direct ensemble show extra bias beyond a?

Across 401 2021 masses, `direct mean − a` has RMS 0.04685 and range [−0.12267, +0.15000]. At 78 MeV the difference is −0.02881 with Monte Carlo standard error 0.06323. The reference mean is within its pointwise interval there.

For a conditional simultaneous check, approximate the mean of 256 independent scan vectors by a Gaussian vector with covariance `diag(s) R diag(s)/256`. The observed maximum absolute standardized mean difference is **2.43285**, versus a simulated 95% critical value **3.43470**; the corresponding model tail is **0.55664**. This check resolves no additional scan-wide mean discrepancy beyond `a` in the tested 2021 ensemble. It does not show that `a=0`, nor does it qualify the physical background source.

Nine pointwise mean intervals and 34 pointwise SD intervals exclude the model values. No mean or SD interval excludes its model value after Bonferroni correction over the 401 masses. Neighboring masses are highly correlated, so the number of pointwise exclusions is not a binomial goodness-of-fit test. The SD intervals assume normal signed roots, and absence of a simultaneous rejection is not a proof of the field approximation, tail accuracy, or source adequacy.

## Paired null controls at the unchanged threshold Tobs = 2.808645

200,000 newly drawn Gaussian fields, one BLAS thread, one fixed seed. Each control shares base random coordinates. These are controlled changes to assumed null distributions, not candidate production probabilities.

| Null control | Median maximum | Tail at raw observed threshold |
|---|---:|---:|
| Nominal a, s, R | 2.44471 | 0.25113 |
| Remove a only; retain s and R | 2.29124 | 0.16225 |
| Set s=1 only; retain a and R | 2.49159 | 0.28256 |
| Set a=0 and s=1; retain R | 2.33761 | 0.18779 |
| Independent masses; same a and s | 2.96595 | 0.67394 |
| Perfectly correlated masses; same a and s | 1.39987 | 0.06772 |

The new nominal tail agrees with the saved Gaussian tail to 0.16 combined Monte Carlo standard errors. Removing `a` changes the tail by −0.08889, with paired 95% Monte Carlo interval [−0.09045, −0.08732]. Setting `s=1` changes it by +0.03143 [0.03066, 0.03219]. Correlation strongly affects the maximum distribution. Mean, scale and correlation effects interact; these shifts are not additive causal percentages. The independence and perfect-correlation controls preserve fixed-mass marginals but are deliberately different scan models.

The four mean/scale transformations were also applied to the same 256 saved direct scan vectors as a separate diagnostic ledger. Those transformed outputs are not fresh full refits under alternative generating models.

## Cross-campaign caution

Cheap moment summaries for 2015, 2016 and the shared-coupling union are retained in the CSV/JSON. They do not support a blanket validation statement. Two 2016 SD discrepancies survive the 283-mass Bonferroni normal-theory interval:

| Mass | Model s | Direct SD | Bonferroni 95% SD interval |
|---|---:|---:|---:|
| 43.5 MeV | 0.954889 | 1.129886 | [0.966476, 1.349062] |
| 44.0 MeV | 0.953804 | 1.117591 | [0.955960, 1.334382] |

These adjacent low-mass widths exceed the response approximation and remain unresolved by this audit. The interval assumptions and common source conditioning still apply. Sparse fixed-mass tails, including zero exceedances for the 2015 and 2016 selected peaks, are bounds rather than zero probabilities.

## What the current calculation does not decide

The source was estimated from observed data and frozen. Its estimation uncertainty, background-family error, potential learned signal, analysis-policy selection, and method/domain choices are not propagated by the quoted Monte Carlo intervals. A source can reproduce its own conditional null response while being inadequate as a physical background model. Pull/moment agreement is not unconditional p-value calibration, exclusion coverage, or a justification for subtracting `a` from the observed scan. Qualifying those questions needs a declared source-building procedure and independent or outer-ensemble validation.

## Reproducible deliverables

- `scripts/audit.py`: archived moment audit, paired null controls and exact source hashes; rebuilds from bundled NPZ inputs without the parent checkout.
- `scripts/make_figures.py`: figures from saved numeric outputs.
- `results/summary.json`: all definitions and scope summaries.
- `results/local_moments_all_scopes.csv`: pointwise and Bonferroni intervals.
- `results/paired_gaussian_counterfactuals_2021.csv` and `paired_maxima_2021.npz`.
- `results/transformed_direct_scan_diagnostics_2021.csv`.
- `provenance/input_hashes.json`, `provenance/runtime.json`.

Final figure basenames (PNG and vector PDF): `2021_local_response_moments`, `2021_signed_clipped_and_maximum`, `2021_paired_maximum_controls`.
