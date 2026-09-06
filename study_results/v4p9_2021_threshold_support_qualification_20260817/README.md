# HPS-GPR v4.9 2021 threshold-support qualification

This bundle starts the version 4.9 analysis note from the validated v4.8.3
release and evaluates a one-factor 2021 support change while preserving the
v4.2 inference settings and v4.5 matched-refit toy semantics.

## Outcome

The 30 MeV lower support edge produces a large paired improvement at 65 MeV,
but does not justify a production freeze.

| scenario | mean pull, 30--300 | mean pull, 40--300 | paired difference | two-sided 90% interval |
|---|---:|---:|---:|---:|
| 2021 1% source x10 | +0.973 | +2.783 | -1.811 | [-2.025, -1.597] |
| 2021 native 10% | -0.133 | +1.152 | -1.285 | [-1.484, -1.086] |

The native-10% 65 MeV cell clears the predeclared screen under 30--300 MeV
support; the 1% x10 cell does not.  The 30--300 lane also flags 55 and 60 MeV
in both scenarios and 70 MeV in 1% x10.  The bundle therefore records a
material support effect and a failed freeze decision, not lack of bias.

## Truth model

Every pure `fSigPowExpQ` and `fGenGammaThresh` restricted source fit is
rejected before extraction.  The executed form is explicitly called an
`fSigPowExpQ-anchored residual stress truth`:

```text
local 30--80 MeV logistic x exp(Chebyshev degree 6)
        -- C2 blend over 75--85 MeV -->
identified fSigPowExpQ anchor through 300 MeV
```

The local source-only deviance/ndf values are 1.010 and 1.140.  The native-10%
85--300 MeV tail remains poor (6.220), so the generator is conditional and not
globally qualified.  This limitation is logically prior to the toy pulls.

## Ensemble

- 25 independent Poisson backgrounds per source family; 50 total.
- Masses 55, 60, 65, and 70 MeV.
- Matched-reference strengths `z=0,1,3,5`.
- Primary GP support 30--300 MeV.
- Same-background paired control with GP support 40--300 MeV.
- 800 accepted states per support lane, zero exclusions.
- No expected bands, limit toys, observed limits, or exclusions.

## Reproduce

From this directory:

```bash
python3 build_fsig_anchor_truth.py validate

V4P9_SUPPORT=030_300 python3 run_threshold_closure.py preflight
V4P9_SUPPORT=040_300 python3 run_threshold_closure.py preflight

# The following return already_complete unless --force is deliberately used.
V4P9_SUPPORT=030_300 python3 run_threshold_closure.py run \
  --toy-start 0 --toy-stop 25 --workers 2
V4P9_SUPPORT=040_300 python3 run_threshold_closure.py run \
  --toy-start 0 --toy-stop 25 --workers 2

V4P9_SUPPORT=030_300 python3 run_threshold_closure.py collect
V4P9_SUPPORT=040_300 python3 run_threshold_closure.py collect
python3 analyze_results.py
python3 make_figures.py
```

Build and validate the note/release after Tectonic is available:

```bash
cd note/build_source
/opt/homebrew/bin/tectonic -C --keep-logs main.tex
cd ../..
python3 build_release_manifest.py
python3 validate_release.py
```

## Navigation

- `STATUS.md`: result and freeze disposition.
- `COMPLETED.md`: completed-work ledger.
- `REMAINING_STEPS.md`: exact continuation needed before a freeze.
- `study_spec.json`: frozen protocol and hashes.
- `derived/analysis/analysis_summary.json`: machine-readable statistical summary.
- `figures/v4p9_qualification/`: validated PDF/PNG figure suite.
- `note/`: v4.9 LaTeX source, rendered PDF, and visual-QA record.
- `runs/`: per-background raw, accepted, optimizer-attempt, and exclusion ledgers.

All claims in this bundle are conditional source-conditioned diagnostics.  The
25-background cells are not frequentist coverage, observed-data bias, expected
bands, limits, exclusions, or scan-wise significance calibration.
