# Optimizer audit and repair policy

## Scope

This is a hyperparameter-support and optimizer-closure pilot.  It does not
construct expected limit bands and it does not establish coverage, an
exclusion, or a discovery calibration.  The upper-bound choice must be based
on optimizer stability, boundary occupancy, and signal-injection behavior,
not on whether an observed limit or p-value becomes more favorable.

`audit_scan_optimization.py` reads only scan attempts with a valid
`_SUCCESS.json` marker and a matching recorded SHA-256.  It compares exact
common truth-model, scenario, background-toy, and mass rows.  It never
interpolates a missing row and it never launches a fit.

## Nested-likelihood closure

For a fixed toy and mass, changing the normalized length-scale upper factor
from \(f_a\) to \(f_b>f_a\) enlarges the feasible parameter domain without
removing the \(f_a\) solution.  The global log marginal likelihood (LML)
therefore cannot decrease.  The audit applies

\[
  \delta{\rm LML}={\rm LML}(f_b)-{\rm LML}(f_a)
\]

to every available factor pair from 6, 9, 12, 15, 20, and 25, with the
numerical tolerance

\[
  \max\!\left(10^{-4},
  10^{-6}\max(|{\rm LML}(f_a)|,|{\rm LML}(f_b)|,1)\right).
\]

The classifications are:

- `higher_factor_optimizer_miss`: the larger domain gives a lower LML beyond
  tolerance.  The smaller-factor optimum is a recorded feasible warm start
  for a repair at the larger factor.
- `lower_factor_optimizer_miss`: the larger-factor solution gives a higher
  LML and its recorded length scale lies inside the smaller domain.  That
  recorded solution is a feasible warm start for a repair at the smaller
  factor.
- `allowed_domain_gain`: the larger-factor solution improves the LML and lies
  outside the smaller domain.  This is the expected signature of a genuinely
  restrictive smaller bound.
- `consistent_lml_plateau`: the two LML values agree within tolerance.

The audit separately flags an exact initialization state when both the kernel
constant and length scale remain at their configured initial values.  Such a
selected row is not accepted as an optimized result even when it happens to
preserve nested LML ordering.  Independent salted repeats that return to the
same state are recorded as
`reproduced_but_not_validated_stationary_state`; reproduction is not silently
promoted to a pass.  The state is resolved only when a recorded non-initial
branch improves the LML beyond the stated numerical tolerance.  A feasible
warm-start repair may itself remain at its deliberately changed initial
coordinates; it resolves the original frozen-card state only if its actual
LML is higher beyond tolerance.  The audit records that distinction explicitly
instead of relabeling the warm optimum as the original initialization state.

## Repair procedure

Repairs begin only after the nominal production outputs are frozen.

1. Use `scan_optimizer_repair_manifest.csv` to identify exact target rows.  Do
   not rerun an entire curve and silently replace it.
2. At the same mass, toy, truth model, factor, training geometry, and immutable
   fit-code commit, initialize the target fit from the recorded feasible
   source optimum in both length scale and kernel constant.  Retain the
   nominal 12-restart setting.
3. For every exact initialization lock, also run independently salted
   optimizer repeats.  A forced repeat with the identical seed is not an
   independent repair.
4. Record every new row and its configuration, seed, and output hash.  Select
   the maximum finite LML only among actual fits for the same exact target
   row.
5. Rerun the audit.  Any surviving nested-dominance failure or initialization
   lock fails the optimizer gate.  A missing point remains missing; it is
   never interpolated.

The repair executor freezes the audit-summary and repair-manifest hashes in
its plan metadata.  Do not rerun the audit while that plan is executing:
intentional input drift stops newly starting children fail-closed.  If an
audit update is required between batches, run the audit once with no launcher
active and then use `repair_scan_optimization.py refreeze-plan`.  That command
keeps the JSONL plan, optimizer seeds, and successful fit directories
byte-for-byte unchanged, archives the previous metadata, and updates only the
source-hash anchors.  It launches no fits.

## Boundary occupancy

The tables report two descriptive thresholds:

- at bound: \(\ell_{\rm opt}/\ell_{\rm hi}\geq0.999\);
- near bound: \(\ell_{\rm opt}/\ell_{\rm hi}\geq0.98\).

Mass rows from the same toy are correlated.  Row-level fractions are useful
maps of where truncation occurs, but the independent ensemble unit is the
toy.  Accordingly, the table also counts toys with at least one bound or
near-bound row.  With only ten toys, even zero occupied toys is pilot evidence,
not a precise upper bound on the occupancy probability.

For the requested 2021 100% projection, the provisional candidate is the
smallest factor for which, after optimizer repair:

1. no fit is at or near the boundary in either analytic truth lane for both
   100%-projection scenarios (native 1% times 100 and native 10% times 10);
2. the LML and optimized length scale plateau against the next larger
   predeclared factors;
3. signal-injection bias and uncertainty are stable against the larger
   factors.

Native 1%, native 1% times 10, and native 10% remain essential diagnostics,
including the requested 1%-times-10 versus native-10% comparison, but a
boundary occupied only in the much lower-statistics native-1% sample does not
by itself set the 100%-projection bound.  If one common card is intended for
all exposures, the stricter version of the gate must instead be satisfied
across all five scenarios.

If factors 15, 20, and 25 share an interior likelihood plateau, lifting the
bound above 15 has no empirical benefit in this pilot.  If a larger factor
selects a longer scale and the injected-signal uncertainty increases or the
recovered amplitude is pulled downward, the GP is gaining signal-absorption
freedom and sensitivity is reduced.  Conversely, stable signal pulls and
uncertainties with an interior optimum indicate that the looser bound is
inactive rather than intrinsically harmful.

## Scientific limitations

- Ten toys and eleven masses at 20 MeV spacing are a targeted pilot, not a
  coverage study or a full-mass validation.
- `fGenGammaThresh` and `fSigPowExpQ` are smooth analytic stress truths.  They
  do not span all detector mismodeling or narrow background structures.
- The native 2021 10% source contains about 11.296 times the support yield of
  the native 1% source, so 1%-times-10 versus native 10% also tests source
  shape and normalization differences; it is not a pure luminosity identity.
- The sigma-scaled injection lane tests standardized recovery.  A fixed-yield
  or fixed-physics-coupling lane is still needed before claiming an absolute
  sensitivity change.
- Candidate-bound selection must be frozen before inspecting whether observed
  limits or asymptotic p-values improve.
