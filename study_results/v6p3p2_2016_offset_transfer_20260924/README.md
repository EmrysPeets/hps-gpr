# HPS-GPR v6.3.2: 2016 offset and response transfer

The numerical run is complete: **2,000/2,000 pilot, 36,000/36,000 calibration and 16,000/16,000 evaluation free fits are valid**, together with all 16,000 truth-fixed profiles and 8,000 full-exposure native 90% profile-CLs limits. All planned toy IDs are retained; the failure ledger is empty. Numerical validity is distinct from scientific closure.

The study uses Gaussian signal generation and extraction at pole masses 42, 44, 60, 66, 76, 90, 92, 117, 160 and 178 MeV. Nominal and archived stress generators are compared at **controlled exposure factors 0.1 and 1.0**. The original nine-page 2021 report is preserved in `inputs/parent_2021_report.pdf`; the separately built seven-page appendix starts at page 10. Final packaging joins the two PDFs without editing the original pages.

## Findings and scope

An automatic 10%-to-full offset transfer is not justified. The calibration difference `delta_full - 10*delta_low` is +5,977 candidates at nominal 42 MeV (pointwise paired-bootstrap 95% interval [3,105, 8,872]) and -7,279 at stress 76 MeV ([-10,911, -3,903]). The exposure laws are approximations requiring a persistent fractional residual and Poisson error scaling; the count-dependent GP and nonlinear likelihood need not obey them exactly.

Across all full-exposure mass/strength cells, the transferred 10% offset leaves mean-pull ranges of -1.243 to +1.162 (nominal) and -1.013 to +1.587 (stress). Each source's own frozen affine correction reduces those ranges to -0.224 to +0.100 and -0.221 to +0.255, respectively. A constant offset subtraction cannot restore signal response: it cancels exactly from the paired-response diagnostic. Calibration learned from nominal toys does not transfer reliably to the stress generator.

Frozen same-source finite-rank acceptance at `z=5` ranges across masses from 81–95/100 (nominal) and 78–98/100 (stress). These are cell counts, not pooled estimates or proof of exactly 90% frozen-table coverage. Native raw profile-CLs containment at `z=5` spans 73–100/100 and 0–100/100, respectively. Exact cell denominators and 95% Clopper–Pearson intervals accompany the tables.

One observed 10% fitted yield or residual is not an ensemble bias; it includes counting noise and possible model mismatch. Actual 2016 production correction requires qualified background sources, selection-equivalent historical input and independent full-exposure calibration with offset, response and width uncertainty propagated. This appendix changes no production limits.

## Scientific design

The independent pilot has 100 nominal-source backgrounds per exposure and fixes `s0(m,L)`, the mean returned Gaussian yield error. Independent calibration and evaluation cohorts each have 100 backgrounds per source/exposure. Calibration uses `z=0,...,8`; evaluation uses `z=0,1,3,5`; expected full-selected yield is `A=z*s0`. The same nominal pilot scale is used for both generators.

Within each cohort/source, the low-exposure background is Poisson(0.1*b); an independent Poisson(0.9*b) increment yields the full-exposure spectrum. Signal counts are likewise paired by independent nonnegative increments between exposure-specific expected yields. Realized counts are never multiplied by ten. Backgrounds are reused across masses and strengths; nominal/stress sources and the three cohorts have independent streams. Gaussian CDF bin probabilities include outside-support losses, without support/window renormalization. Signal is injected in training sidebands as well as the extraction window.

Archived mass-specific kernel parameters and the pole-centered `±2.25*sigma_m` masks remain fixed. Every toy recomputes count-dependent GP preprocessing, mean and covariance. This is GP conditioning without hyperparameter optimization. The likelihood retains signed yields and profiles the correlated GP constraint, with positive Poisson means. Pulls use expected yield and the observed profile-Hessian yield error; they are not signed profile-likelihood roots or calibrated significances.

Calibration fixes `delta=mean(Ahat_0)`, `R=mean(Ahat_3-Ahat_0)/(3*s0)` and `k0=SD((Ahat_0-delta)/sigma_0)`. Evaluation reports raw, transferred-offset, direct-offset and affine estimates. Primary affine pulls hold calibration fixed. Their separate uncertainty diagnostic includes the calibration covariance and null-width scale; no corrected estimator rescales a native likelihood limit. The bootstrap uses 2,000 whole-toy replicates, preserves paired exposures/masses/levels within each source, and separates calibration/evaluation streams. The master seed is `63420160924`; exact namespaces are in `protocol.json`.

## Finite calibration and limits

The lower-tail rank probability uses raw signed yield `T=Ahat`: `p_A=(1+#100 calibration T_A <= evaluation T)/101`. Reject a grid value if `p_A <= 0.10`. For exchangeable draws, rejection is at most `10/101` marginally over calibration and the new experiment; inclusive ties are conservative. That marginal bound is distinct from the actual acceptance conditional on the saved table, assessed with the independent evaluation cohort and Clopper–Pearson intervals.

For continuous statistics the population tail mass at the tenth order statistic has the Beta(10,91) reference distribution. The associated central 95% acceptance range is approximately 0.836–0.951 before observing the table. This describes finite-calibration variation; it is not the held-out binomial confidence interval.

The accepted `z=0,...,8` set is saved without interpolation or monotonic repair. Its largest accepted node gives a reported upper endpoint. Empty sets are assigned the physical reporting convention `U=0` and flagged, so upper-envelope containment at zero differs from truth-grid acceptance. Holes remain explicit. Acceptance of `z=8` is right censoring: the reported endpoint is a lower endpoint, not a finite measured upper limit. Truth-grid acceptance and upper-envelope coverage remain separate columns.

The estimator-ordering toy-CLs diagnostic uses `min(1,p_A/p_0)` on this same coarse grid. It is not the native profile-CLs construction. Native 90% CLs limits are computed only at full exposure from the original raw likelihood and inherited bounded-q asymptotic tails, using the toy GP mean as null Asimov spectrum. They are not offset- or response-rescaled.

At full-exposure stress 76 MeV, `z=5`, using nominal calibration gives 0/100 rank-Neyman acceptance and 100 empty sets, but 100/100 toy-CLs acceptance and 100 right-censored endpoints. The shared floor `p_A=p_0=1/101` makes the ratio one. This is finite-MC saturation, not successful transfer or a rescue of the native limit (also 0/100 containment). The same issue occurs at 42 MeV. With each source's own calibration at 76 MeV, `z=5`, rank acceptance is 98/100 for stress and 87/100 for nominal, illustrating frozen-table variability.

## Source qualifications

The 0.1 branch is a controlled scaling of a fixed full-exposure generating mean, not a replay of the historical 2016 subset. That subset had support-count ratio 0.1022016; exact luminosity fraction, identical selection and event overlap were unverified. The archived stress histogram retains a failed broad-component source-fit flag and is a sensitivity control rather than a qualified physical background model. The historical 2016 support/optimizer-qualification exception also remains: successful fixed-kernel toys do not validate the original kernel optimization. See `provenance/stress_source_manifest.json`, `provenance/historical10_manifest.json` and `protocol.json`.

Source-estimation, selection and detector/systematic uncertainty are not sampled. Conditional closure under either fixed generator does not establish unconditional coverage, unknown-background adequacy, discovery/global significance or experimental exclusion. The coarse grid supplies no off-grid coverage guarantee.

## Reproduce or resume

From this package directory, use the validated scientific Python for fitting/reporting and the bundled Python for PDF packaging:

```bash
STUDY_PYTHON=/Applications/Xcode.app/Contents/Developer/usr/bin/python3
PDF_PYTHON=/Users/emryspeets/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
"$STUDY_PYTHON" scripts/launch.py --workers 4
"$STUDY_PYTHON" scripts/analyze.py
"$STUDY_PYTHON" scripts/validate.py
"$STUDY_PYTHON" scripts/make_appendix.py --build
"$PDF_PYTHON" scripts/package.py merge
```

The launcher uses a 1,800-second watchdog, at most four local workers, one numerical thread each, a duplicate-run lock and validated atomic checkpoints. Re-running resumes the same saved counts and settings. Independent pilot and calibration tables are frozen before later stages. Do not change the frozen protocol or numerical dependencies while reusing checkpoints. No remote compute is used.

Analysis and reporting regenerate summaries and the appendix without scientific generation or fitting. Validation checks the saved study before final reporting and packaging. Reporting verifies the analysis source hashes and requires complete attempted-toy accounting. PDF compilation uses `/opt/homebrew/bin/tectonic --only-cached`; override its path using `--tectonic` where needed. Packaging uses pypdf to join the unchanged original report and appendix. Final semantic/provenance and rendered-page QA is recorded separately under `qa/`.

After the QA gates pass, `"$PDF_PYTHON" scripts/package.py release --destination <output_directory>` creates the release package. Use the approved final output directory in place of the placeholder.

| Location | Content |
|---|---|
| `protocol.json` | Frozen scope, normalization, seeds, masks, fit gates and resource limits |
| `inputs/cohorts.npz`, `inputs/templates.npz` | Saved independent cohorts and full Gaussian probabilities |
| `pilot_reference.json`, `calibration_freeze.json` | Frozen scale and calibration identity |
| `results/pilot_rows.csv`, `calibration_rows.csv`, `evaluation_rows.csv` | All attempted numerical rows |
| `results/asimov_rows.csv` | 40 deterministic mean-spectrum fits, not additional toys |
| `results/calibration_parameters.csv` | Offset, response, width, covariance and uncertainty |
| `results/scaling_comparisons.csv` | Paired exposure-scaling residuals and intervals |
| `results/evaluation_diagnostics.csv` | Raw, offset and affine held-out diagnostics |
| `results/calibration_quantiles.csv` | Rank thresholds, finite-calibration Beta references |
| `results/rank_grid_rows.csv`, `limit_rows.csv` | Rank probabilities and complete accepted-set decisions |
| `results/limit_summary.csv` | Truth acceptance, upper-envelope containment, CP intervals and censoring |
| `results/nominal_to_stress_comparisons.csv` | Paired calibration-transfer contrasts |
| `results/failure_ledger.csv`, `summary.json` | Numerical failures, completeness and source hashes |
| `figures/`, `source/appendix.tex`, `pdf/appendix.pdf` | Reproducible figures and seven-page appendix |
| `inputs/parent_2021_report.pdf` | Unchanged original nine-page 2021 report |
| `provenance/`, `qa/` | Input identities, source qualifications and validation evidence |
