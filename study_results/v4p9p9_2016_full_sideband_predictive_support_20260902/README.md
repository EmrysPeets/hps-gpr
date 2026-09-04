# HPS-GPR v4.9.9: stopped control-only 2016 support qualification

Canonical state: `stopped_development_absolute_or_technical_failure`.

There is no selected v4.9.9 2016 support, no full-2016 confirmation, no
support-specific observed 2016 scan, and no v4.9.9 combined limit.

## Question and frozen design

This separately versioned study tested whether the 2016-full lower GP support
could move slightly from the later reviewed 2016-10% prescription of
30--210 MeV.  Eligible edges were 29--33 MeV at fixed 210 MeV; 34 MeV was an
ineligible geometry control.  The protocol, its separated-control amendment,
and its absolute-adequacy amendment were all hashed before any fit or score.

The selector used only blocked controls below and above the 39--180 MeV search:

- low support selector: 35.25--38.75 MeV, trained only on the candidate's
  lower extension and other low blocks;
- high model check: 181--210 MeV, trained only on other high blocks.

Exactly zero search-region centers entered any training or scoring mask.  No
signal amplitude, pull, p-value, limit, epsilon-squared value, toy, or expected
band was computed by the selector.

## Development outcome

The development phase made 420 deterministic optimizer attempts and retained
140 selected anchor/block cells.  Every covariance and exact mask was logged.

The low-only comparison would have advanced support 29:

| Lower edge (MeV) | Technical | Absolute guard | NLPD improvement vs 30 | SE | Improvement/SE |
|---:|:---:|:---:|---:|---:|---:|
| 29 | pass | pass | 0.0101812 | 0.00760005 | 1.33962 |
| 30 | pass | pass | 0 | 0 | reference |
| 31 | fail | pass | -0.106474 | 0.0553864 | -1.92238 |
| 32 | pass | pass | -0.118272 | 0.0613256 | -1.92859 |
| 33 | pass | pass | -0.199070 | 0.0568911 | -3.49915 |
| 34 control | pass | pass | -0.216636 | 0.0799530 | -2.70954 |

For support 29 the mean low-control Mahalanobis statistic per bin was 0.6800,
the worst anchor/block value was 1.9983, the largest marginal standardized
residual was 2.2781, the paired Poisson-deviance direction was favorable, and
the NLPD improvement remained positive under every single-low-block deletion.

The candidate-independent high-only check nevertheless occupied the upper
length-scale bound in all 20 anchor/block cells.  Its absolute prediction was
not grossly poor (mean Mahalanobis per bin 1.1299, maximum 1.3302, maximum
marginal standardized residual 3.2779), but the predeclared technical gate
required zero bound contacts.  That failure is terminal.

No lowest-score fallback or post-result amendment is permitted.  In
particular, the attractive low-only support-29 diagnostic is not a frozen
production support.

## Exact artifacts

- `STUDY_PROTOCOL.md`, `PROTOCOL_AMENDMENT_PRE_EXECUTION.md`, and
  `PROTOCOL_AMENDMENT2_PRE_EXECUTION.md`: prospective rules.
- `FROZEN_PROTOCOL_SHA256`: immutable protocol chain.
- `derived/development/optimizer_attempts.csv`: 420 deterministic attempts.
- `derived/development/selected_predictive_scores.csv`: 140 selected-cell
  scores, covariance checks, and exact mask hashes.
- `derived/development/support_summary.csv`: low-control candidate comparison.
- `derived/development/high_control_summary.json`: terminal high-control result.
- `derived/development/phase1_decision.json`: machine-readable stop.
- `qa/final_validation.json`: independent hash, mask, and gate reconstruction.

## Reproduction

From the repository root:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 study_results/v4p9p9_2016_full_sideband_predictive_support_20260902/run_blocked_cv.py \
  --stage development

python3 study_results/v4p9p9_2016_full_sideband_predictive_support_20260902/analyze_selection.py phase1
python3 study_results/v4p9p9_2016_full_sideband_predictive_support_20260902/validate_release.py
```

The full-2016 confirmation and all downstream production commands are absent
by design.  A future differently designed study must be separately versioned
and frozen prospectively; it may not rewrite this result.

## Interpretation boundary

This is a failed prospective, out-of-search, control-region support
qualification.  It is not an observed-data limit or p-value, a coverage test,
an expected sensitivity, an exclusion, or a global-significance calibration.
