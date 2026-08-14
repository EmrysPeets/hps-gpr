# v4.8.3 residual-structured 2021 functional-form study

This directory contains the requested continuation of the v4.8 functional-form
study.  It was developed in an isolated worktree from commit
`e2c930f3f879742b2846e3fca1ee1b7e8d99ecc6`; the user's dirty primary checkout
was not used as a build source or modified during production.

## Scope

Two post-exploratory source means were frozen and tested:

- `knot_spline`: the frozen v4.8 rigid mean multiplied by a positive natural
  log-spline residual correction.  Five blocked source-only folds and the
  one-standard-error rule select fixed nodes at 105 and 180 MeV over the
  predeclared three-node comparator.
- `regional_blend`: a positive, C2 log-space blend of fixed low-, middle-, and
  high-mass forms across 85--125 and 165--215 MeV overlaps.

The native-10% transfer freedoms are restricted and frozen by
`MODEL_PROTOCOL.json`.  Knot locations, overlap boundaries, and source shapes
are never selected with GPR pulls or recovery.

Both models fail the support-wide source-qualification gates and the separate
fake-gap, injection-refit, and tangent-space signal-influence gates.  They are
therefore **requested conditional stress truths only**, not qualified
generators and not demonstrations that the functional families cannot learn a
narrow signal.

## Pseudoexperiment inventory

Each model has exactly 20 closure backgrounds in five reported lanes:

1. native 1%;
2. 1% x 10;
3. 1% x 100;
4. native 10%; and
5. 10% x 10.

The extraction lattice is five masses (65, 90, 120, 180, and 210 MeV) by four
injections (0, 1, 3, and 5 reference sigma).  Both model collections contain
2,000 raw and 2,000 accepted states with zero exclusions.  Lanes are nested
within each source family; the two model streams are deliberately distinct, so
the model comparison is descriptive and unpaired.

The K2 model has three of 25 exploratory material zero-signal flags, all at
65 MeV.  The regional blend has sixteen, with sign-changing structure aligned
with its two overlap zones.  The post-hoc matched-background paired-response
medians are about 0.980 and 0.971 overall, but rare accepted branch changes and
the independent source-influence failures prohibit a signal-safety claim.

## Length-ceiling disposition

The original pull-blind pilot rejected factor 20 and used the protocol's
factor-25 fallback.  That fallback is insufficient for K2/native 1%: all nine
pilot states and 345/400 closure states contact factor 25.  The corresponding
closure row is upper-bound-censored.

After that defect was discovered, a frozen post-closure addendum used eight
fresh, background-only K2/native-1% toys.  Factor 35 failed at 120 MeV.  Factor
50 passed against a factor-75 sentinel in both the three-toy selection and
five-toy confirmation sets.  This is only a targeted background-optimizer
ceiling qualification.  It does not rerun or uncensor the factor-25 closure,
qualify a common ceiling for all ten model/lane combinations, or alter the
production v4.2 factor-15 card.

The pilots and remediation use one worker.  The closure driver is capped at
two workers, and the production runs used single-thread BLAS settings.

## Principal artifacts

- Protocol: `MODEL_PROTOCOL.json`
- Source fits and influence: `derived/source_fit_and_influence.json`
- Original pilot: `derived/residual_length_pilot/`
- Model collections: `derived/residual_closure/{knot_spline,regional_blend}/`
- Cross-model tables: `derived/residual_closure/`
- Targeted ceiling remediation:
  `derived/residual_length_ceiling_remediation/`
- Figures: `figures/`
- Full 221-page note:
  `note/HPS_GPR_Analysis_Note_v4p8p3_2021_residual_truths_20260814.pdf`
- Section 5 source:
  `note/source_overlays/sections/subsection_v4p8p3_residual_truths.tex`

The original comparison manifest remains immutable and describes the
factor-25 closure.  The ceiling-remediation manifest is separate so the later
technical check cannot be mistaken for a closure rerun.

## Validation

Read-only component checks are:

```bash
python3 fit_residual_models.py validate
python3 run_residual_length_pilot.py validate
python3 run_residual_closure.py --model knot_spline preflight
python3 run_residual_closure.py --model regional_blend preflight
python3 run_residual_length_ceiling_remediation.py validate
python3 validate_release.py
```

The pilot disposition stores a canonicalized-JSON SHA-256 for its scan
contract.  This intentionally differs from the SHA-256 of the indented JSON
file bytes; `validate_release.py` checks both conventions explicitly.
