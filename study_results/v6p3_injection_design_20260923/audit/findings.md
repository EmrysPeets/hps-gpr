# Historical injection-design audit

**The audited historical matched-reference ensemble really uses a different expected signal yield for each background toy: A_inj,t = z sigma_ref,t. Signal counts are then Poisson-fluctuated, not deterministically added.** The reference is the selected zero-signal fit uncertainty from that same background toy. It is fixed before the injected refit, but is not independent of the background realization.

Scope: the v4.6 smooth-threshold full-100 baseline carried into the v5.0.5 consolidated validation figures, plus code confirmation in its v4.9.5 descendant. The 20-cell numerical audit covers four 2021 exposure/source scenarios and five masses. It does not pool the later targeted 65 MeV replacements or claim to audit every historical injection branch. Evidence and exact line references are in `evidence.md`.

## Actual implementation

1. Fit the background toy with GP refitting and signed signal extraction. Save the actual returned `sigma_A` and a separate Asimov diagnostic called `sigmaA_reference`.
2. Set `sigma_ref = reference_row['sigma_A']`, then `A_injected = z*sigma_ref`. In accepted rows, **`sigmaA_ref` is the correct column for the injected scale**. The similarly named `sigmaA_reference` has role-dependent meaning: Asimov diagnostic on background-only rows, actual selected uncertainty on injected rows.
3. Keep the same background histogram for all strengths of that toy. Draw independent Poisson signal-bin counts with means `A_injected*w_i`, using a seed depending on scenario, toy, mass and strength. Add those integer counts to the existing background, then refit. The background realization is shared across strengths; the signal seeds differ across strengths.
4. The ordinary pull is `(Ahat(z)−A_injected)/sigma_A(z)`, with the post-injection error. It is centered on the expected injected yield, not the realized fluctuated signal total.
5. v5.0.5 defines paired response `(Ahat(z)−Ahat(0))/A_injected`. This removes a background-specific intercept in a linear response model. However, older v4.6 recovery figures explicitly plotted **unpaired** `Ahat/A_injected`. The audit computes both from saved rows and does not relabel those old figures.

## Saved-row checks

There are 7,994 accepted rows: 1,999 background-only references and 5,995 injected rows with a matched baseline. All injected rows satisfy `strength = z*sigmaA_ref` to maximum floating-point discrepancy 2.91e-11. Their `sigmaA_ref/sigma_A(0)` is exactly one. Every source/mass cell has nonzero reference variation. The reference CV, sample SD divided by mean, ranges from **1.166% to 2.999%** across the 20 cells.

| Historical native 2021 10% mass | Mean sigma_ref | SD sigma_ref | CV | Corr(sigma_ref, Ahat0) |
|---|---:|---:|---:|---:|
| 65 MeV | 7327.46 | 206.23 | 2.814% | +0.162 |
| 90 MeV | 6494.33 | 154.38 | 2.377% | −0.344 |
| 120 MeV | 5055.33 | 90.69 | 1.794% | −0.065 |
| 180 MeV | 2853.62 | 53.92 | 1.890% | +0.079 |
| 210 MeV | 2144.01 | 43.46 | 2.027% | +0.180 |

All five native cells have 100 reference rows. One other source/mass cell has 99. These are descriptive, unadjusted sample correlations; they are not formal tests or estimates of response-slope covariance. The strongest observed correlation in the full table is −0.584 for 1%×10 at 90 MeV. No independence of sigma_ref and baseline amplitude should be presumed.

The realized signal count differs from the expected yield in every injected row. Across the accepted rows, `(Nsig_full−A_injected)/sqrt(A_injected)` has mean −0.0137 and sample SD 1.0052, consistent descriptively with the explicit Poisson-generation code. This pooled summary is not a separate calibration. See `sigma_reference_by_scenario_mass.csv`, `native2021_10pct_summary.csv`, and `paired_recovery_by_cell.csv` for complete results.

## Statistical meaning

Per-toy matching is a coherent **adaptive injection diagnostic**: it asks how the procedure responds to a signal of z times that toy's background-only reference uncertainty. It does not sample a conventional fixed-signal-yield experiment. The signal expectation is a function of the same realized background being analyzed. The reference is pre-injection, but not an external, noise-independent design quantity.

Using one fixed `A=z*s0` instead asks a different question: performance at one common physical expected yield. An independently determined s0 (for example a frozen Asimov or independent-pilot reference) gives the clearest fixed-yield interpretation. Replacing it with the mean reference from the same toy batch makes the common yield random and weakly dependent on every member of that batch.

For a linear conditional-mean response `Ahat_t(A)=u_t+g_t*A`, with `s0=E[s]`, adaptive minus fixed mean bias is `z Cov(g,s)`. The per-toy paired conditional response is g for either design, whereas adaptive ratio-of-means recovery is `E[g*s]/E[s]`. With finite signal draws, individual paired recovery also contains signal noise. A paired response near one does not establish zero absolute bias, unit pull width, correct fixed-yield coverage, or discovery power. Raw recovery additionally contains the background offset `u/A`.

For iid background toys, a shared same-batch reference has `Cov(z*mean(s),u_t)=z*Cov(s,u)/T`. This does not make it equivalent to an independent fixed reference at finite T. The archived sigma CV is modest, but it does not determine `Cov(g,s)` and cannot prove that the two designs give the same fitted results. **No fixed-A counterfactual fits have been performed or inferred from these rows.**

## v6.2 contrast

The checked v6.2 code uses fixed selected-candidate totals N={1000,5000,10000,30000}, draws exact-N multinomial signal categories, and adds them to an independent Poisson background reused across levels. GP kernel states are archived and fixed; conditioning is repeated on each spectrum. This is neither deterministic signal addition nor the earlier Poisson-signal matched-reference design. The different signal-count covariance must be retained when interpreting pull widths or containment.

## Reproduction

Run `PYTHONDONTWRITEBYTECODE=1 python3 audit_saved_rows.py` from any directory with NumPy and pandas installed. All numerical calculations read `inputs/minimal_accepted_rows.csv`; no external checkout or HPS runtime is required. `_source_line` records each row's original CSV line. Original source hashes, copied code evidence, local input hashes and reproducible summaries are included. Original files, if present, are hash-checked read-only. No fits, new random draws or parent edits were performed.
