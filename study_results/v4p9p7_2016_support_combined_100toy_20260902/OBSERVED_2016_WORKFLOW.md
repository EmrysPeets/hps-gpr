# Post-freeze full-2016 observed workflow

This workflow is intentionally inert until
`derived/analysis/support_freeze_decision.json` reports
`status: support_edge_frozen`, all initial/continuation/full-100 gates true, and
`observed_scan_authorized: true`.  The decision must bind the live frozen
protocol and study specification.  A passing independent confirmation audit
at `audit/confirmation_freeze_audit.json` must also recompute the phase-1 and
full-100 inventories, bind that exact canonical freeze, and retain the explicit
broad-tail conditional-stress-only waiver.  No bypass, alternative freeze
path, or provisional support value is accepted.

The observed scan uses the exact integer-MeV grid 39--180 MeV (142 rows), the
selected `data_range_2016 = [selected_low/1000, 0.210]`, the frozen 2016 k12
upper length-scale factor, 12 optimizer restarts, signed extraction, and 90%
asymptotic q-tilde CLs.  It does not run expected bands or combined inference.
The 65 MeV holdout is not used to choose the support edge; it enters only after
the support is frozen, as one ordinary point on the authorized observed grid.

This is not a strictly full-data-blind support study.  There is no event-level
disjointness evidence for `source_2016_10pct.root`, so it is described as the
pre-existing 2016 10% development sample/subset, not as an independent sample.
It supplies partial observed-shape information to the source-conditioned truth.
Full-100% histogram values entered truth construction only through the scalar
26--210 MeV normalization; no support-specific full-100% fit, local p0, or
upper limit was used to rank support edges before the freeze.
`SCIENTIFIC_SCOPE_CLARIFICATION.md` is the SHA-pinned authoritative statement
of this boundary and the broad-tail conditional-stress-only waiver.

## Static checks before the freeze

Run from this study directory:

```bash
python3 build_observed_2016_card.py preflight
python3 run_observed_2016_cli.py preflight
python3 review_observed_2016.py preflight
python3 validate_observed_2016.py preflight
python3 validate_observed_2016.py blocked-state
```

The four preflights must report `production_blocked_no_provisional_edge` and
`observed_data_evaluated: false`; `blocked-state` must report `pass` only when
the confirmation authorization, canonical freeze, generated card, and all
observed/review products are absent.  The mutating modes `build`, `primary`,
`plan`, `repeat`, `finalize`, and `validate` validate the SHA-pinned terminal
denial and then fail closed, even if a freeze file is later fabricated.

As of the completed phase-1 independent audit, no provisional support edge
passes the frozen practical gate.  Therefore no phase-2 continuation,
confirmation authorization, observed card, or observed scan is permitted in
this release.  The post-freeze sequence below is a dormant execution contract,
not authorization to bypass that outcome.

## Dormant post-freeze implementation scaffold — do not execute

There is no authorized post-freeze sequence in v4.9.7.  The commands below
document the prepared implementation contract only; the terminal phase-1
denial makes the first card-build command fail.  Enabling these steps would
require a separately versioned scientific protocol, not a retune or override
inside this release.

```bash
python3 audit/independent_freeze_audit.py static \
  --accept-broad-tail-fit-status-for-conditional-stress-truth-only
python3 audit/independent_freeze_audit.py phase1 \
  --accept-broad-tail-fit-status-for-conditional-stress-truth-only
python3 audit/independent_freeze_audit.py confirmation \
  --accept-broad-tail-fit-status-for-conditional-stress-truth-only
python3 build_observed_2016_card.py build
python3 run_observed_2016_cli.py preflight
nice -n 10 python3 run_observed_2016_cli.py primary
python3 review_observed_2016.py plan
```

Inspect `observed_scan/final_2016/optimizer_repair_plan.json`.  For every mass
listed in `repeat_masses_MeV`, run exactly three separate process invocations
with the unchanged frozen card:

```bash
nice -n 10 python3 run_observed_2016_cli.py repeat --mass-mev MASS --repeat-index 1
nice -n 10 python3 run_observed_2016_cli.py repeat --mass-mev MASS --repeat-index 2
nice -n 10 python3 run_observed_2016_cli.py repeat --mass-mev MASS --repeat-index 3
```

Then finalize and validate:

```bash
python3 review_observed_2016.py finalize
python3 validate_observed_2016.py validate
```

The reviewed output is
`observed_scan/final_2016/results_single_reviewed.csv`.  It contains exactly
142 rows and the combination assembler fields `dataset`, `mass_GeV`,
`const_opt`, `ls_opt`, `lml`, `interpolated`, `branch_multiplicity`,
`selected_source`, `row_source`, `review_status`,
`selected_support_low_MeV`, and `support_high_MeV`.

After validation, obtain the selected low edge from the freeze decision and
pass it explicitly to the existing assembler:

```bash
python3 assemble_reviewed_state_ledger.py \
  --reviewed-2016-csv observed_scan/final_2016/results_single_reviewed.csv \
  --support-freeze-json derived/analysis/support_freeze_decision.json \
  --support-2016-low-mev SELECTED_LOW \
  --support-2016-high-mev 210 \
  --output-csv derived/combined/reviewed_state_ledger.csv \
  --provenance-out derived/combined/reviewed_state_ledger_provenance.json
```

## Review rule

A primary row is repeated when it is non-finite, extraction- or
covariance-invalid, at a kernel bound, exactly at both optimizer initial
coordinates, lacks density support, or records an optimizer warning.  The
primary row plus three unchanged-card repeats form the candidate inventory.
A repaired candidate must be finite, covariance-valid, away from bounds, and
away from the exact-start signature.  Among branches reproduced at least
twice, the candidate with maximum GP log marginal likelihood is selected.
Branch matching uses only LML per training bin, length scale, kernel constant,
and covariance-derived `sigma_A`.  Amplitude, upper limit, epsilon-squared,
local p-value, and agreement with the primary result never rank a branch.
Interpolation is prohibited.

## Dependencies

- Python 3.9 or newer.
- Repository dependencies from `requirements.txt`: NumPy, pandas, SciPy,
  scikit-learn, matplotlib, uproot, hist, Click, PyYAML, joblib, and
  threadpoolctl.
- The repository-local `gp` package, whose Python files are SHA-256 checked.
- The frozen `runtime_overlay/hps_gpr` package entrypoint plus instrumented
  `gpr.py` and `io.py`; all fallback HPS-GPR modules are SHA-256 checked.
- The SHA-pinned full-2016 ROOT file and frozen v4.2 card already bundled under
  `inputs/`.
- The SHA-pinned independent auditor and static-truth audit.  The observed card
  builder and every scan/review/validation mode require the passing
  `static -> phase1 -> confirmation` audit chain and bind its file hashes.
- The SHA-pinned `SCIENTIFIC_SCOPE_CLARIFICATION.md`; both the independent
  audit chain and the post-freeze observed contract must bind it exactly.
