# Bounded sampling refinement design

Recommendation only; no new fits or frozen-file edits. Freeze this design and
the selected checkpoint hashes after first-pass production finishes.

## Diagnosis and fixed scientific procedure

The saved 2016 checkpoints show two different problems. At 41 MeV the
stress/fixed endpoint is 9.6875 reference sigma, between proposal nodes 8 and
12, with S+B tail ESS 1.45 and weight normalization 31.8 ± 31.6. At 45 MeV
ESS is 3.23. These need proposal overlap, not merely more draws at 8 and 12.
At 42–44 MeV, CLs at the actually sampled node 12 is respectively 0.933,
1.019 and 0.500, with healthy tail ESS. These require bracket extension.

Keep exactly the two truths, native supports/masks, frozen kernel choices,
per-toy posterior retraining, count-dependent alpha, signal tails, shared
coupling conversion, bounded statistic, and 90% CLs target. Proposal design
changes neither a truth nor the statistical ordering. No rescaling of Z,
endpoint, or likelihood is part of this repair.

## Selection, frozen before new draws

Build a deterministic selection ledger from first-pass `result.json` files;
never read validation outcomes to select or tune points. Eligible components
must be right-censored or fail an existing MC precision gate. Prioritize:

1. Any right-censored component, since it prevents a finite envelope.
2. A finite MC-limited component currently controlling a method's envelope.
3. A finite MC-limited component whose MC interval overlaps the other truth's
   endpoint interval. An unusable interval has unknown overlap, not proven
   irrelevance; retain that qualification and place it after finite overlaps.

Within a priority use the poorest available tail ESS, then the original scope
order and ascending mass. Select at most 24 coordinates. Record every eligible
but deferred coordinate and its reason. Do not prioritize validation failures,
small limits, p-values, apparent signals, or favorable improvements. Other
components may remain explicitly MC-limited; the existing collector's rule
requiring both truths to be resolved must not be relaxed.

## Proposal mesh and bracket

Use dimensionless strength `a=A/sigma_reference`. For each truth let `t_i` be
its full-spectrum mean and `g_i=sigma_reference*ctx.signal_i`, including the
training-region signal tails and all concatenated constituents. Define

\[
 I_{\rm full}(a)=\sum_i\frac{g_i^2}{t_i+a g_i},\qquad
 \tau(a)=I_{\rm full}(a)^{-1/2}.
\]

This is a proposal-overlap scale, not the profiled estimator uncertainty.
Within refinement windows place unshifted proposal centers with
`a_next=min(window_end,a+0.75*tau(a))`. Since the signal is nonnegative,
information decreases with a; this bounds each step's integrated Fisher
distance by 0.75. Include window endpoints and candidate centers exactly.
Retain every original proposal node, particularly 0, 2 and 5 for validation.
At each center retain the unchanged unshifted and two influence-shifted
proposals from `Context.proposals`; preserve its positivity failure checks.

Use saved influence metadata only as a placement hint. For method m define
`s=sd/sigma_reference` and
`h=max(0,(Ahat_observed-bias)/sigma_reference)+2*s`.
For a finite selected endpoint u, densify the interval spanning u and h,
expanded on each side by `max(0.5,2*s,4*tau(h))`, truncated at zero.
Take the union of windows for selected methods within the same truth bank.
Bad first-pass ESS makes its small reported MC interval untrustworthy;
do not let that interval alone determine the window width.

For a censored component with previous ceiling U, set the first new ceiling
to `min(64,max(1.5*U,U+4*tau(U),h+3*s))`. Densify continuously from
`max(0,min(U,h-2*s))` to that ceiling. This places the 42–44 MeV problem's
new ceiling above 12 without claiming an analytically predicted limit.
If the new ceiling is still accepted, it remains censored unless the bounded
second attempt below obtains an actual CLs crossing.

Distinguish **proposal centers** from **inversion scan strengths**. Keep the
original scan nodes and add the window boundaries, u/h candidates, and new
ceiling; use the frozen last-accepted-bracket inversion and its bisections.
Do not evaluate every dense proposal center as another full-bank likelihood
scan: that needlessly approaches quadratic work. Nevertheless, every inversion
or slope evaluation near the candidate endpoint must lie inside a covered
window. A crossing outside it triggers the second attempt or stays MC-limited.
Retain all evaluated nodes, the monotonicity check, and all root/slope traces.

## Counts, independence and stopping

Use 512 draws per proposal in each refined truth bank. Both background methods
use that same full-spectrum toy bank. Cap a truth bank at 64 distinct strength
centers (192 proposals); do not silently coarsen a mesh to meet the cap.
Regenerate any unrefined truth bank with its original nodes, 256 draws and
original seed, verifying its saved whole-array hash. It is needed to evaluate
the two-truth validation envelope; raw first-pass banks were not stored.

Allow only one further attempt for a selected component still censored or
failing MC gates: 1,024 draws per proposal, a fresh independent seed, the same
mesh rule around the newly located endpoint, and a ceiling at most
`min(64,2*previous_ceiling)`. Stop if the 64-center cap is exceeded or a bracket
is still absent. Keep the latest completed attempt regardless of whether its
limit is numerically larger or smaller, and retain earlier attempts separately.

Freeze each attempt's proposal table before generating its fresh draws.
Use a separate refinement seed namespace including attempt and proposal hash.
Exact Poisson weights and stratified variances remain those in `Bank`.
Do not naively combine first-pass draws with a mixture chosen after examining
those draws: fresh refined banks avoid that adaptive reweighting issue.

Retain all original ESS, endpoint-error, bracket, normalization and numerical
gates. Also report a separate sampling-readiness check that each root/slope
target's normalization SE is at most 0.05; the original “within 5 SE” test
alone accepts a useless normalization such as 31.8 ± 31.6. A new readiness
failure remains qualified, even if the frozen gate alone passes.

One numerical process, after first-pass completion. Bound the stage by 24
coordinates, 1.5 million generated calibration spectra including regenerated
banks, and a 60-minute scheduling budget. Check proposed counts and estimated
cost before starting a coordinate; stop scheduling when a cap would be exceeded.
Use that coordinate's recorded first-pass time, scaled conservatively for new
bank size and scan count. The 23 completed 2016 points available for this review
had a median of 19.3 s, but 41 MeV took
200.5 s: a median-only cost estimate is unsafe. Preserve the user's 50% weekly
availability floor. Deferred or interrupted work stays explicitly unfinished.

## Numerical checks, validation, and derivative provenance

Call the frozen approximation/scalar checks. In the derivative additionally
check exact versus approximate predictions and scalar/batch r/q at the new
maximum strength and the new root region, using deterministic independent
audit toys from the unshifted and shifted proposals. Retain the same 0.001
exact/approximate prediction/r/q gates and the existing tighter scalar/batch
tolerances; original q checks through 12 do not by
themselves certify a new 20–64-strength range. Rejecting the approximation
restores the exact backend for the coordinate; a genuine exact fit failure
halts it and remains recorded. Never discard failed toys.

Re-evaluate the unchanged validation test with both selected truth banks and
500 direct-Poisson spectra per truth/strength. Reusing the original validation
seeds gives the same paired holdout spectra, independent of both sampling
stages because selection uses calibration MC diagnostics only. State that
these are rescored holdouts, not additional independent validation evidence;
do not add their counts to the original counts. Preserve all flags, and run
the final family adjustment on the selected final coordinate versions.

Implement only a new `run_sampling_refinement.py`, new protocol/selection
ledger, and a separate `refined_v1/` output tree. Import the frozen Context,
Bank, likelihood, and inversion functions without editing or monkey-patching
the six frozen files. Record the original inference/input hashes separately
from the derivative driver/design hashes, proposal/scan tables, per-truth
counts and seeds, parent checkpoint SHA, and every attempt. Compatibility
must explicitly verify identical inference inputs and statistic definitions;
do not falsely label the complete sampling contracts identical. Final
collection replaces whole coordinates with their declared latest versions,
never pools endpoints, p-values, or validation counts.
