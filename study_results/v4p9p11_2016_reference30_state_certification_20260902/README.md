# v4.9.11 reference-card adequacy/state certification: terminal stop

This release retains the pre-existing reviewed 2016 support 30--210 MeV and
upper length factor 12; it does not claim that v4.9.11 selected or confirmed
that support against an alternative.

The support-30-only full-data low-control check passed every frozen technical
and absolute-adequacy gate.  Its canonical first execution used zero centers
at or above 38.75 MeV.  Mean Mahalanobis per bin was 0.6731, the worst
anchor/block value was 1.0412, and the largest absolute marginal standardized
residual was 1.6213.  These are pragmatic adequacy diagnostics, not calibrated
goodness-of-fit p-values or high-side validation.

The production-state qualification then applied a uniform, signal-blind
fit-state protocol to all 142 integer-MeV masses.  All 426 seeded fits ran, but
only 49 states passed every predeclared warning-free reproduction,
stationarity/polish, bounds, prediction, covariance, and two-sideband gate.
Ninety-three states remain unresolved.  The canonical decision is therefore
`stopped_unresolved_state` with `combination_authorized=false`.

No signal amplitude, p-value, Z value, or limit was computed in this release.
The 142-row CSV is a terminal audit ledger, not a qualified input to a combined
result.  No downstream release may consume it as a successful 2016 state
ledger.

The prospective code split is documented in
`PRE_ARCHIVE_CODE_SPLIT_AMENDMENT.md`.  The canonical control chain is the
first decision plus `run_control_frozen.py`; later timestamp-bearing control
decisions are explicitly noncanonical.  The archive/robust chain is
`run_downstream_certification.py`.  The old monolithic filename is a fail-fast
compatibility pointer only.

Canonical outcome and validation:

- `derived/state_certification_decision.json`;
- `derived/observed_2016_gp_states_reviewed.csv` (audit-only);
- `derived/robust_repeats/optimizer_attempts.csv`;
- `TERMINAL_STATE_SHA256`;
- `qa/final_validation.json` (35/35 checks pass);
- `FINAL_VALIDATION_SHA256`.

The terminal result does not authorize inference.  Any remediation must be a
separate prospectively frozen, uniform numerical-optimization study using the
same support/card and no signal or inference metric.
