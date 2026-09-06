# 2021 10% background-profile comparison

Completed comparison of all 201 observed mass hypotheses, 50-250 MeV, based on
the exact v4.9.12 inputs and the v4.9.12.5 peak-dip methodology. No new toys,
unblinding, production edits, or changes to the concurrent task's files.

The four-page report is
`output/pdf/background_profile_comparison_20260905/background_profile_comparison_2021.pdf`
from the repository root. Individual figures are available in PDF and PNG under
`figures/`. The main figure is `observed_limits_2021_comparison.png`.

## Findings

- The alternative profiles the positive latent log-GP background directly in
  the Poisson likelihood, with exactly the same sidebands, kernel coordinates,
  fit windows, resolution and frozen yield-to-coupling conversion.
- Direct log-GP versus released Gaussian profiling changes observed upper
  limits by at most **1.189%**. A numerical control using the same stable solver
  for both models limits the background-distribution change to **0.232%**.
- Re-solving the released Gaussian model changes its limits by at most
  **1.139%** (86 MeV). This is a numerical diagnostic, not a replacement for the
  released curve. Independent optimizers support the new numerical solution.
- Fixed GP mean / released upper limit has median **0.607**, with range
  **0.137-1.373**. Fixing the background changes both uncertainty and the fitted
  signal, so it does not lower every observed limit. The fixed curve treats an
  estimated background as known and is only a conditional reference.
- The profiling-distribution change leaves the peak-dip structure essentially
  intact. The v4.9.12.5 moving-mask response remains a relevant modeling issue.
- Improved fit displays show profiled curves only inside their actual fit
  domain. Every residual uses the same GP-mean baseline; the plots do not join
  profiled and unprofiled subtractions at a window boundary. These displays use
  the original released fitted components, with the new log-GP fit overlaid.

## What was checked

All 201 native prediction hashes exactly match the released ledger. The ROOT
file SHA-256 matches its frozen value, and all 422 archived integer counts
match the native histogram. Native mass coordinates are retained to avoid
round-trip changes in GP arithmetic. The attested archived runtime is used.

All 603 new upper limits (log-GP, fixed, and Gaussian numerical control) pass
positivity, likelihood-nesting, CLs-root and monotonicity checks. Analytic
gradients and Hessians pass finite differences at four display masses.
Independent BFGS fits from two starts at seven masses agree within 1e-7 NLL.
An independent one-bin fixed-background limit check agrees within 1e-6 events.
See `derived/validation.json`, `numerical_checks.json`, and
`profile_diagnostics.json`. Numerical comparisons and all source hashes are
retained, including differences from the released optimizer.

`derived/observed_limits.csv` contains electron-channel values.
`derived/observed_limits_with_display_correction.csv` also contains the
expanded note's dimuon branching correction, applied to every displayed curve.
Neither file contains expected bands or calibrated global significance.

## Reproduction

Run from the repository root, using one low-priority process per command:

```bash
nice -n 10 python3 -B study_results/background_profile_comparison_20260905/run_comparison.py
nice -n 10 python3 -B study_results/background_profile_comparison_20260905/validate_comparison.py
nice -n 10 python3 -B study_results/background_profile_comparison_20260905/make_report.py
```

The scan takes about seven seconds on this machine; all numerical libraries
are limited to one thread. The fixed design and references are in `PROTOCOL.md`.

## Remaining scientific work

This requested profiling comparison is complete. The following are proposals
for a new study, not required unfinished work in this deliverable:

1. Compare with a simultaneous signal-plus-positive-exponential-polynomial
   Poisson model following the published HPS approach, or a small predeclared
   discrete-profile candidate set. That would change the background shape
   assumptions, which the present comparison deliberately holds constant.
2. Qualify windows and model flexibility on independent or held-out background
   and injected-signal controls, including nearby positive features and the
   moving training mask. Do not select using favorable observed limits.
3. Evaluate coverage for the complete inference procedure with retraining and
   the stated selection rules before adopting a replacement analysis.

The literature supports the statistical ingredients; it does not establish
unconditional coverage of the frozen, partially unblinded configuration.

## Coordination

The active task `Commit studies and start bands` confirmed authoritative input
paths and separate output ownership. It worked in
`study_results/v4p9p12_targeted_tail_refinement_20260905/`. This comparison stayed
in its own directory and matching PDF output directory; no parent validation
files or existing note artifacts were rewritten.
