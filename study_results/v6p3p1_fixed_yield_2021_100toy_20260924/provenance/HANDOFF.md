# Handoff: 2021 10% fixed-yield injection study with 100 pilot and 100 evaluation toys

Prepared 24 September 2026 with the HPS-GPR and statistics agents. This is an execution instruction document; the study has **not** been run.

## Task for the receiving chat

Implement and run an independent-pilot, common-expected-yield injection–extraction study for **2021 10%**, using both the native reconstructed-MC signal shapes and the regular Gaussian signal shapes. Use **100 pilot background spectra and a separate 100 evaluation background spectra**. Produce reproducible numerical results, clear comparison plots, and a short LaTeX/PDF report. Keep the existing published study packages intact.

The question is: **at the same fixed expected signal yield, how accurately does the GP-based procedure recover MC-shaped and Gaussian-shaped signals, and how well do its returned errors and nominal intervals perform?** This follows the design explained in v6.3. It is not another per-toy reference-matched injection study.

Repository root, denoted `ROOT` below:

```text
/Users/emryspeets/Desktop/gp_mods/hps-gpr-emrys-validation_workflow
```

Paths below are relative to `ROOT`. Create a new study directory, for example `study_results/v6p3p1_fixed_yield_2021_100toy_YYYYMMDD/`, using the execution date. Treat these scientific defaults as settled for the primary run; record any necessary implementation correction before proceeding.

## 1. Fixed scope

| Item | Primary specification |
|---|---|
| Dataset | 2021 10%; do not scale to full exposure |
| Masses | 60, 80, 100, 120, 140, 160, 180, 200, 220, 240 MeV |
| Why this grid | Native MC masses within the inherited 50–250 MeV search range |
| Background truth | Pinned nominal 2021 GP arithmetic mean already bundled with v6.2 |
| Spectrum support | Preserve the input bin edges; approximately 36–299.75 MeV |
| Pilot cohort | 100 independent full-support background spectra |
| Evaluation cohort | 100 further independent full-support background spectra, independent of the pilot |
| Reuse | Reuse each cohort's spectra across all masses; reuse evaluation backgrounds across shapes and injection levels |
| Primary shape cases | Gaussian generated/Gaussian fitted; native MC generated/native MC fitted |
| Injection levels | `z = 0, 1, 3, 5`, with common expected yield `A(m,z) = z*s0(m)` |
| Common reference | Mean **Gaussian** pilot yield uncertainty at that mass, frozen before evaluation |
| Window | Pole mass `m ± 2.25*sigma_m(m)` for both shapes; identical fit and training-exclusion masks |
| GP procedure | v6.2 archived kernel parameters fixed; recompute count-dependent preprocessing, conditional mean and covariance for every spectrum |
| Signal fluctuations | Independent Poisson signal counts, including a fluctuating full signal total |
| Fitted amplitude | Signed full-template signal yield; keep negative fitted values |

Thus there are **200 unique background spectra total**, each used at every mass. Each mass/shape/level cell has 100 evaluation toys, but cells are correlated through shared backgrounds. This is intentional pairing, not extra independent statistics.

The primary comparison uses one common Gaussian reference scale so both shapes receive the same expected number of candidates. Save MC pilot uncertainties too, but do not silently use a different injection yield for MC. A shape-specific frozen scale would be a separate equal-nominal-difficulty study.

Do not include 40 MeV, the 260 MeV extension, shifted MC-core windows, new mask widths, shape morphing, hyperparameter reoptimization, extra datasets, or global-significance scans in this first run. MC injected/Gaussian fitted is a useful **separate signal-model-mismatch diagnostic**, not a substitute for either primary case. These extensions can be proposed after the primary result is complete.

## 2. Inputs and implementation references

Use the following verified local sources. Read their code, not just their names:

| Path | Use |
|---|---|
| `study_results/v6p2_mc_injection_20260923/inputs/null_2021.npz` | `truth` background means and `edges_GeV`; also contains observed counts for input-identity checks |
| `study_results/v6p2_mc_injection_20260923/inputs/v6p1/inputs/spectrum_2021.npz` | Binning, nominal resolution coefficients, archived kernel states and dataset identity |
| `study_results/v6p2_mc_injection_20260923/inputs/v6p1/histograms/m060.npz` through the corresponding `m240` files | Native selected-MC histograms and metadata; retain recorded overflow |
| `study_results/v6p2_mc_injection_20260923/scripts/injection_core.py` | Count-yield likelihood, GP prediction, fit diagnostics and checkpoint patterns to adapt |
| `study_results/v6p2_mc_injection_20260923/inputs/v6p1/scripts/core_centering.py` | Native MC loading and `cumulative(m, edges)`; primary window remains pole centered |
| `study_results/v6p2_mc_injection_20260923/inputs/v6p1/scripts/common.py` | Resolution, archived kernel states, GP preprocessing and covariance factorization |
| `study_results/v6p2_mc_injection_20260923/inputs/v6p1/scripts/limit_solver.py` | `OneSignalProfile`, signed fits and nuisance profiling |
| `study_results/v6p2_mc_injection_20260923/protocol.json` | Exact previous procedure and limitations; previous signal generation was exact-N, not the requested Poisson-rate experiment |
| `study_results/v6p3_injection_design_20260923/source/report.tex` | Statistical definitions and rationale for independent-pilot common yields |

Pin SHA-256 hashes of the inputs and copied/adapted code in the new package. Copy the required portable dependencies into the new study or resolve them read-only; inspect modules' directory-relative input loading. Do not edit globals in the published v6.2 files or run its unchanged launcher as the new study. Its `--pilot` flag is a small-run/debug route, not the statistical pilot/evaluation split requested here.

The verified SHA-256 of `inputs/null_2021.npz` within v6.2 is `306a915b9ba6230aafbe058c94438c6af11f85b60d68f5aae4ebbfec4d4a9424`. Resolve any input-identity discrepancy before freezing the new protocol.

Known scientific runtime:

```text
/Applications/Xcode.app/Contents/Developer/usr/bin/python3
```

It has NumPy, SciPy, pandas and Matplotlib. Existing report builds use `/opt/homebrew/bin/tectonic`. Validate imports and input identities before numerical work. Use the supplied observed spectrum only to verify the input release; do not use evaluation results or fresh observed-data tuning to redefine the source truth, mask or templates.

## 3. Construct templates in full signal-count units

Let `w_i(shape,m)` denote the probability for a selected signal candidate to enter analysis bin `i`. Add below-support and above-support categories so the **complete** probability vector sums to one. Retain probabilities as fractions of the full template when fitting a window. The amplitude `A` counts full-selected candidates; it is not a count restricted to the fit window, generated resonances, or an epsilon-squared parameter.

**Gaussian:** use bin-integrated Gaussian probabilities at pole mass `m`, with the archived nominal 2021 resolution:

\[
w_i^{G}(m)=\Phi\!\left(\frac{e_{i+1}-m}{\sigma_m(m)}\right)
          -\Phi\!\left(\frac{e_i-m}{\sigma_m(m)}\right).
\]

Use consistent GeV units for edges, mass and resolution. Obtain `sigma_m` from the pinned coefficient array; the inherited polynomial is `0.00184825 - 0.001375*m + 0.085875*m**2`, with `m` and the result in GeV. Include the Gaussian CDF tails outside the analysis support.

**MC:** use the pinned native histogram CDF at the exact generated mass. Rebin by differences of `MC.cumulative(m, edges)` using the inherited uniform-within-source-bin convention. Keep native tails and core offset, underflow/overflow conventions, and the full selected normalization. Do not interpolate between masses or apply additional smearing or shifts.

**Implementation trap:** inherited `C.signal`/`continuous_signal` includes a physics-yield conversion and support normalization; `MC.distribution` also normalizes on support. They are not drop-in unit-count probability templates for this study. Construct the probabilities directly as above. Check category nonnegativity, sum-to-one closure, and expected support/window/training fractions before any fits. Tiny roundoff repair, if necessary, must be bounded and recorded; do not renormalize a substantial missing tail away.

Use the same template probabilities for generation and extraction within each primary shape case. Native MC versus Gaussian differences can reflect core offset, width, tails and window acceptance together; do not attribute all differences to tails or GP absorption without additional controls.

## 4. Pilot: choose and freeze the common expected yields

1. Draw 100 full-support spectra `B_pilot[j,i] ~ Poisson(b_i)` from the pinned mean `b_i`. Use the same pilot spectra at all masses.
2. For each mass and each pilot spectrum, fit once with the Gaussian template and once with the MC template. Use the same pole-centered mask and GP prescription that evaluation will use.
3. Save signed fitted yields, returned yield errors and fit diagnostics. Define

   \[
   s_0(m)=\frac{1}{100}\sum_{j=1}^{100}\widehat\sigma_{A,G,j}^{\rm pilot}(m),
   \qquad A(m,z)=z\,s_0(m).
   \]

   This averages the **returned uncertainty on signal yield**. It is not the standard deviation of pilot fitted yields, the standard error of the mean, a mass resolution, or a similarly named Asimov error.
4. Save both shapes' 100 individual errors, means, sample SDs, CVs, the MC/Gaussian mean-error ratio, and the Gaussian pilot mean's standard error `SD(sigma_G)/sqrt(100)`.
5. Write and hash the complete `pilot_reference.json` or equivalent table, including all masses, shapes, masks, input/code hashes, cohort IDs and the frozen yields. Freeze it **before evaluation fits begin**. Do not tune it after seeing evaluation performance.

Require 100 valid Gaussian pilot errors per mass before freezing that mass's scale. Retry numerical failures on the same saved counts using the predeclared procedure; unresolved errors make the scale incomplete. Never silently average fewer successful fits or draw replacement toys. Record unresolved MC pilot diagnostics separately; they must not change the Gaussian nominal scale.

Once frozen, `A(m,z)` is the known expected truth in evaluation. Do not add the pilot mean's standard error to the pull denominator, randomly redraw `s0` per evaluation toy, or refreeze it during a bootstrap.

## 5. Evaluation: 100 independent backgrounds with Poisson signal

Draw a new cohort `B_eval[t,i] ~ Poisson(b_i)`, `t=0,...,99`, independent of the pilot. Reuse each of these full-support backgrounds across masses, shapes and levels. For each `(mass, toy, shape, z>0)`, draw independently

\[
S_{tzi}\sim\operatorname{Poisson}(A(m,z)\,w_i),\qquad
n_{tzi}=B_{{\rm eval},ti}+S_{tzi}.
\]

Apply this to the complete signal category vector, including both outside-support categories. Equivalently, draw `N_signal ~ Poisson(A)` and then `Multinomial(N_signal, full_probabilities)`. For `z=0`, the signal vector is exactly zero.

- Keep `A` as a floating-point expected yield; do not replace it by `round(A)`.
- Do not rescale the realized signal histogram to have total `A`, inject an exact total, or Poisson-fluctuate the already fluctuated background again.
- Add signal in all analysis bins, including GP training sidebands. Cutting off tails before retraining would change the question.
- Save expected yield separately from realized full/support/window/training/outside-support signal counts. In adapting v6.2, replace its `N` semantics: truth-fixed fits and pulls use **`A_expected`**, while realized outside-support count is `draw[0]+draw[-1]`, not `A_expected - sum(signal_in_support)`.
- Save one evaluation null spectrum per toy and fit it once per mass/template. Reuse that fitted baseline for paired recovery; do not regenerate null backgrounds separately for each shape or level.

Use reproducible, distinct RNG namespaces. A concrete default is NumPy `SeedSequence` with master seed `63220260924` and integer keys:

```text
pilot background:     [master, 1, 0, toy, 0, 0]
evaluation background:[master, 2, 0, toy, 0, 0]
evaluation signal:   [master, 3, mass_MeV, toy, shape_id, level_id]
bootstrap:           [master, 4, 0, replicate, 0, 0]
```

Use `shape_id=1` for Gaussian and `2` for MC, and `level_id=1,2,3` for `z=1,3,5`. Background keys deliberately omit the actual mass because spectra are shared across masses. Smoke tests must have a separate namespace and never enter the 100-toy cohorts. Do not use Python's process-dependent `hash()` or execution-order-dependent global RNG state.

## 6. Repeat the declared GP extraction procedure

For each pilot, null and injected spectrum:

1. Construct the fixed pole-centered extraction mask and train only on its complement within the unchanged full support. Retain at least three exterior bins on each side. Never use extraction-window counts for GP training.
2. Hold the archived mass-dependent kernel parameters fixed, but recompute the count-dependent log targets/noise, GP conditional mean and correlated covariance using that spectrum's sidebands. Cache geometry only. The inherited positive-count preprocessing uses `log(n)` and `alpha=1/n`; zero-count handling uses target `0` and `alpha=1`. Retain and record this convention consistently.
3. With covariance factor `L`, fit the inherited model

   \[
   \lambda_i=\widehat b_i+(L\theta)_i+A\,w_i,\qquad
   -\log\mathcal L=\sum_{i\in W}(\lambda_i-n_i\log\lambda_i)+\tfrac12\theta^T\theta.
   \]

   Keep `Ahat` signed and enforce positive Poisson means. Retain correlated GP uncertainty and the existing covariance-mode diagnostics; do not replace it by independent per-bin errors.
4. Return the signed free fit and observed profile-Hessian yield uncertainty. For truth containment, refit nuisance coordinates at fixed **expected** `A(m,z)`, using the same toy's GP mean/covariance constraint. Do not retrain the GP again inside that likelihood profile.

Call this **GP conditioning/retraining with archived kernel parameters**, not full hyperparameter optimization and not a frozen-background extraction. A hyperparameter-reoptimized production-procedure validation would require its own consistently specified pilot, null and injection runs. Do not mix the two procedures in one comparison. Do not add an extra random GP nuisance draw to the generating spectrum; the primary ensemble has one fixed background truth plus counting fluctuations.

## 7. Required diagnostics and uncertainty reporting

For each `(mass, shape, z)` keep all 100 attempted toy IDs and compute:

| Quantity | Definition / interpretation |
|---|---|
| Absolute bias | `mean(Ahat - A_expected)`, with uncertainty from toy scatter |
| Pull | `(Ahat - A_expected)/sigma_postfit`; report mean and sample SD |
| Raw recovery, `z>0` | `Ahat/A_expected` |
| Paired response, `z>0` | `(Ahat_z - Ahat_0)/A_expected`, matching the same background and extraction template |
| Null offset | Signed fitted-yield and pull means/widths at `z=0` |
| Error response | Returned `sigma_postfit` and its ratio to the frozen reference; distinguish both from empirical fitted-yield scatter |
| Nominal containment | `q_true = 2*(profile_nll_at_A_expected - free_min_nll)` compared with `1` and `3.841459` |

These thresholds define nominal 68.27% and 95% **signed profile-likelihood sets**. They are not a physical nonnegative-signal upper-limit construction. Profile containment can be calculated without tracing every interval endpoint. Unit pull width and nominal containment are reference values to test, not assumptions or fit acceptance criteria.

Use each fit's returned error in pulls and expected `A`, not realized `N_signal`, as the truth. Subtract the null only in the paired-response diagnostic, not in raw yields, primary pulls or likelihood intervals. Report the average of per-toy recovery ratios explicitly; do not silently substitute a ratio of averages. At zero injection MC and Gaussian templates can give different fits, but changing common-yield versus per-toy injection scaling cannot repair either template's null bias.

Report standard errors of the toy-sample means and, when needed, use a fixed-seed block bootstrap that resamples whole evaluation toy IDs, preserving all masses, levels, shapes and null partners. Keep the frozen pilot scale fixed. For MC–Gaussian differences, use paired differences; count only complete pairs and disclose missing pairs. Do not treat 4 levels or 10 masses as independent extra toys.

With 100 toys, indicative one-standard-error precision is about 0.10 for a unit-width normal pull mean, 0.071 for its width, and 4.7/2.2 percentage points for containment near 68%/95%. Report actual finite-sample intervals, including 95% Clopper–Pearson intervals on containment counts. These precision estimates are not pass/fail thresholds. Input `z` is a reference-strength label, not achieved local or global significance.

## 8. Fit failures, validation and checkpointing

Predeclare a small bounded numerical retry policy on identical saved counts, with recorded alternative optimizer starts/tolerances and selection by convergence and objective value. Retain all attempts. Never select/reject by pull, fitted signal, recovery, or agreement with truth; never regenerate a difficult toy. Preserve inherited gradient, covariance, positive-mean and finite-error checks, and investigate rather than silently loosening failed thresholds.

Report `attempted`, `fit_valid`, `profile_valid`, and failure reasons separately. Report valid-fit moments and `contains/profile_valid` alongside `profile_valid/100`. Successful-fit containment is conditional on numerical success. If `k` intervals contain truth and `f` containment decisions remain unresolved, also report all-attempt accounting bounds `[k/100, (k+f)/100]`; these bounds are not confidence intervals. Do not report 100 selected successes as 100 attempted experiments.

Before the full run, verify on a tiny separate smoke-test cohort:

- Identical input edges/source identity, nonnegative integer toy counts, valid full-template probabilities and saved fractions.
- Pilot/evaluation RNG separation, intended background reuse, independent signal streams, and deterministic replay from keys.
- Injection amplitude fixed across evaluation toy IDs and shapes at each mass/level; identical generation/extraction templates for each primary case.
- GP training excludes the window and includes injected exterior tails; mean/covariance change with counts despite fixed kernel parameters.
- Signed likelihood fit and truth-fixed profile have valid means, convergence, finite positive errors, and consistent likelihood ordering. Retain the inherited numerical tolerance: require `q_true >= -2e-6` before clipping tiny negative values to zero; larger violations require investigation.
- A small known-background control checks count normalization and extraction mechanics. It is a diagnostic with truth supplied, not the primary GP result. Add clean-sideband controls only if needed to investigate a deficit, keeping them separately labeled and using saved paired spectra.
- Resume produces the same frozen scale, counts and completed rows; checkpoint reuse checks input, code, configuration and reference-table hashes.

Use at most four local workers, each with one numerical-library thread. Save both cohorts before fitting; checkpoint pilots, the frozen reference and evaluation chunks. Cache kernel geometry and reuse a given background's GP prediction across its two zero-signal template fits. Use a real wall-clock watchdog for each bounded launch (30 minutes is the inherited default), preserve completed checkpoints on interruption, and resume without changing seeds or scientific settings. Do not silently increase the toy count or launch remote compute.

## 9. Work count and required outputs

For each mass, fitting both pilot templates gives **200 pilot free fits**. Evaluation gives **200 null free fits plus 600 positive-injection free fits**. That is **1,000 free fits per mass, 10,000 over the ten masses**, despite only 200 unique background spectra. Evaluating nominal containment adds **800 truth-fixed nuisance profiles per mass, 8,000 total**. Retries, smoke tests and separately labeled controls are additional. The GP training prediction can be shared where the spectrum and mask are identical.

Deliver in the new study directory:

- `protocol.json`: all fixed choices, seeds, cohorts, template conventions, masks, GP settings, pilot rule, numerical acceptance/retries and limitations.
- Pinned input/code hashes; saved background cohorts, template category vectors, signal draws or a verified replay record, and completed checkpoints.
- Pilot rows and the frozen reference table with both shapes' errors and the shared expected yields.
- Evaluation rows with mass, shape, level, toy/cohort/seed IDs, `A_expected`, realized counts by region, `Ahat`, `sigma_postfit`, null partner, likelihoods/containment, numerical status and diagnostics, and background/injected-spectrum/template hashes.
- Summaries by mass/shape/level, paired MC–Gaussian comparisons, explicit failure ledgers and finite-sample uncertainty.
- Readable figures: reference errors and MC/Gaussian ratio; raw and paired recovery; absolute bias; pull means/widths; nominal containment with binomial intervals; and representative full-spectrum/window/residual displays showing the injected tails and GP response. Use consistent shape colors and label every sigma, yield convention and reference line.
- A concise standalone LaTeX/PDF report with mathematical definitions and plain-language conclusions, `README.md`, a reproducible launcher, rendered-page QA and `MANIFEST.sha256`. Mirror final deliverables under `output/pdf/` using the new study name.

The report must distinguish the new Poisson-rate ensemble from v6.2's exact-N multinomial ensemble. Do not attribute differences between their results solely to one code change: injection levels, counts, pairing and templates/settings must actually match for a controlled comparison.

Conclusions are conditional on one pinned background source, fixed empirical MC templates and this fit prescription. Supplied MC selection equivalence and signal-daughter association remain unvalidated; finite-MC, detector-response and source-estimation uncertainties are not propagated. The study can measure conditional fixed-yield bias and nominal-set containment with stated uncertainty. It does not establish physical-background adequacy, unconditional coverage, calibrated discovery significance, or an exclusion.

## 10. Reading and completion checks

The local v6.3 report supplies the detailed design derivation. The frequentist fixed-parameter simulation principle is also described in the [PDG Statistics review, Section 40.2.6.1](https://pdg.lbl.gov/2026/reviews/rpp2026-rev-statistics.pdf).

Before delivering results, confirm that the pilot scale was frozen before evaluation; all planned 100 toy IDs are accounted for in every cell; expected and realized signal counts are separate; MC and Gaussian use the same primary expected yields and masks; failures and limitations remain visible; and the package can reproduce the results without modifying prior studies. Report completed versus incomplete cells explicitly. No slide edits or revisions to the existing v6.3 explanatory report are part of this handoff.

Instruction-document review completed on 24 September 2026: the HPS-GPR agent checked inputs, normalization, masks, GP implementation and numerical tolerances; the statistics agent checked sampling, pilot independence, pairing, diagnostics, failure accounting and interpretation. The requested corrections are incorporated. This review does not substitute for validating the future implementation and results.
