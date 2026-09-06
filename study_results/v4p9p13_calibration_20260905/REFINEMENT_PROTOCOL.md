# Independent Poisson-mixture sampling refinement, version 1

The statistical analysis remains the frozen v4.9.13 conditional 90% CLs
procedure. This derivative changes proposal placement and Monte Carlo counts,
with a separately audited candidate for numerical GP acceleration.
`run_sampling_refinement.py` imports the unchanged Context, Bank,
observed-q construction, inversion and validation functions. It does not edit
or monkey-patch the six first-pass files. The two generating truths, frozen
reviewed kernels, per-toy GP posterior refit and alpha, full signal tails,
bounded likelihood ratio, native supports, 2016 exception and shared coupling
remain unchanged. The all-three envelope still has exactly two joint truths.

## Execution and scheduling

Do not invoke numerical execution until the main production process has
finished. The driver requires all 456 original completed coordinates and all
47 original source/input hashes to match. It does not inspect host memory or
process tables. The coordinator is responsible for waiting for the first-pass
process to exit and maintaining the user's 50% weekly availability floor.

Typical commands, from the repository root, after first-pass completion:

```sh
python3 -B study_results/v4p9p13_calibration_20260905/run_sampling_refinement.py --plan-only
python3 -B study_results/v4p9p13_calibration_20260905/run_sampling_refinement.py
python3 -B study_results/v4p9p13_calibration_20260905/run_sampling_refinement.py --batch-index 2
```

`--scope 2016 --masses 41-45,74 --output <new-study-subdirectory>` permits an
explicit bounded subset. `--plan-only` reconstructs native observed contexts
and freezes meshes/counts but generates no calibration, audit or validation
toys. It is not a zero-computation command. Execution then resumes the same
output with identical arguments apart from `--plan-only`.
Relative `--output` paths are resolved inside this study. `--previous-input`
paths use the calling working directory unless absolute.

The default output is `refined_v1/attempt1_batch001/`. Up to 24 coordinates,
60 minutes of default scheduling time and 1.5 million newly generated calibration
spectra form one scheduling slice. Time/count estimates defer the next point
before draws; ongoing points finish or record a failure. Resuming starts a
new bounded scheduling slice; already completed coordinates are not rerun.
Other batches and deferred coordinates remain work. These bounds are not an
overall stopping condition or permission to discard unresolved endpoints.
Use later batch indices or explicit new output/subset selections to continue.
An explicit `--max-minutes` above 60 is permitted only with `--batch-size 1`.
This prevents a slow but necessary coordinate from being deferred forever by
its conservative first-pass runtime estimate. The override is frozen in the
selection record; it does not change the memory, numerical or statistical gates.

## Selection and frozen plans

Only censoring and existing Monte Carlo gates determine eligibility. Validation
outcomes, p-values, apparent signals and favorable limits are never selection
inputs. Priority is right-censoring; a limited component controlling its
method's envelope; overlapping finite MC intervals; unknown overlap; then
other qualified components. Within priority use smallest available tail ESS,
original scope order, and ascending mass. All eligible and deferred entries
remain in `selection.json`, with original/source checkpoint SHA-256 values.

Selection is frozen before native reconstruction. Each `point_plan.json`
freezes geometry and counts before numerical audit toys; `proposal_plan.json`
freezes proposal hashes, labels and generating metadata before calibration
draws. Existing records must match byte-for-byte on resume. A different plan
requires a separate output tree.

## Mesh, counts and inversion

For full-spectrum truth t and signal per reference sigma g, use

\[
 \tau(a)=\left[\sum_i g_i^2/(t_i+a g_i)\right]^{-1/2},\quad
 a_{j+1}=\min(a_{\rm end},a_j+0.75\tau(a_j)).
\]

Here `a=A/sigma_reference`, `g=sigma_reference*ctx.signal`, and all training
bins and constituent datasets are included. Nonnegative signal makes this a
conservative bound of 0.75 on each step's integrated Fisher distance. This
scale guides sampling and is not an estimator error or significance rescaling.
Retain every original proposal node. Each new node has the same unshifted and
two influence-shifted proposals as the first pass; positivity is mandatory.

For saved influence bias beta and propagated SD s in reference-sigma units,
use the placement hint `h=max(0,(Ahat_observed-beta)/sigma_reference)+2*s`.
A finite candidate u receives the interval spanning u and h, padded by
`max(0.5,2*s,4*tau(h))`. A censored ceiling U is extended to
`min(64,max(1.5*U,U+4*tau(U),h+3*s))`, with a dense bridge from
`max(0,min(U,h-2*s))`. Merge overlapping windows. Hints never set an endpoint;
the exact toy CLs must cross 0.10. Cap the union at 96 centers per truth.
Exceeding the cap visibly defers the coordinate; never silently coarsen it.

Dense proposal nodes are separate from inversion scan nodes. The scan retains
the original guards and adds candidate centers/window boundaries and ceiling.
The unchanged inversion evaluates the last accepted bracket and bisections,
retaining all traces and monotonicity checks. Candidate root/slope evaluations
outside covered proposal windows cannot be declared sampling-ready.

Refined truth banks use 512 draws per proposal. An optional second attempt
uses 1,024 and fresh independently seeded banks: specify `--attempt 2` and
one or more `--previous-input` attempt-1 directories. Previously refined
truths stay refined. The ceiling can extend to twice its prior value, capped
at 64; there is no automatic third attempt. Continued unresolved work must
remain documented for further planning within the user's budget.

Unrefined truths regenerate their original 256-draw banks using the original
seed and verify both proposal and whole-bank hashes. Crucially, reconstruct
those original proposals under the reproduced first-pass backend before any
extended-range check can force an exact fallback. Keep the resulting proposal
law fixed. Refined seeds use a distinct namespace, attempt and proposal hash.
No adapted mixture is used to rebalance previously inspected calibration draws.
Exact Poisson weights and stratified errors remain the unchanged Bank methods.

## Numerical and memory gates

Reproduce the original approximation/backend first and freeze all science
proposal arrays under that backend. For coordinates whose original backend
fell back to exact Cholesky, try one refinement-only candidate: the same joint
eigenfeature GP at relative kernel cutoff 1e-15, with nuisance covariance
cutoff 1e-7 instead of 1e-5. The unchanged retraining code still requires at
most 12 retained nuisance modes per constituent. A construction, rank or
approximate-fit failure rejects the candidate and restores exact Cholesky.
No numerical tolerance is relaxed, and this remains an approximation.

The candidate replays all 18 original audit draws: nine proposal means at
strengths 0, 2 and 5 for each of the two truths, generated under the exact
backend with the original `numeric-audit` RNG stream. It tests r and q at
strengths 2, 5 and 12, recording the proposal and full-count hashes. The science
proposal law frozen earlier is never regenerated under a candidate backend.

Run additional independent
audit toys at each new candidate region and the highest generated strength,
covering unshifted and both shifted proposals for both truths. Compare exact
and approximate count means/covariances and r/q through the new strengths,
using the existing 0.001 approximation gates. Retain scalar/batch tolerances
of 2e-5 in r and 1e-4 in q. Actual scalar/batch disagreement or an exact fit
failure is fatal and retained. Only an approximation failure triggers exact
backend restoration. Scalar-reference exceptions are also fatal. No failed
toy is discarded. Neither audit family is used as a statistical calibration
sample, and no candidate is selected from validation results.

`approximation_candidate_audit` (schema version 1) explicitly identifies the
final backend, nuisance cutoff and active candidate. Each candidate lists all
its `numerical_checks` indices, required stages and required check counts.
The original-cut candidate requires 18 `original_frozen_audit` checks; the
stricter candidate requires 18 `original_replay` checks. Both additionally
require exactly three `extended` checks per distinct candidate/ceiling center
per truth. Acceptance requires every required check, with every discrepancy
below 0.001. Rejected original and candidate rows remain in the ledger even
when a later candidate passes; they are not active-fit failures. Exact fallback
is explicitly identified by `active_candidate_id=exact_cached_cholesky`.
Every actual scalar check must pass regardless of the final backend.

The candidate IDs are `eigenfeature_rtol1e-15_nuisance1e-5` and
`eigenfeature_rtol1e-15_nuisance1e-7`. Collection must verify the final backend
and cutoff against the active accepted candidate, the complete index sets
against the numerical ledger, and each required stage count. It must not
ignore inactive rejected checks or accept an approximation merely because a
metadata flag is true. Numerical policy
`original_proposals_then_nuisance1e-7_replay_v1` is frozen with the sampling plan.

The point plan reports conservative peak-array estimates for both backends.
These include both retained banks, full count arrays, GP factor/padding copies,
profile Hessian workspaces, mixture-density matrices and validation work.
The default explicit guard is 4 GiB, configurable through `--max-memory-gib`
and frozen in selection/point metadata. After extended checks choose the actual
backend's estimate and write `memory_check.json` before any calibration bank
generation. An excess writes a visible deferred record; it never reduces
counts or mesh density. This is a conservative array bound, not a host-memory
measurement. No `ps`, `sysctl`, escalation or automatic bound override is used.

Original resolution gates remain archived as `frozen_mc_status`. A separate
sampling-readiness check also requires covered root/bracket/slope locations
and normalization SE at most 0.05 with mean within the original 5-SE/5%
criterion. Failed readiness leaves status `limited_mc`; right-censoring remains
right-censoring. The old broad normalization test alone cannot certify overlap.

## Validation, contracts and output

The unchanged validation function rescores the same 500 direct-Poisson
holdout spectra per truth and strength (0, 2, 5), paired across both methods.
They remain independent of the sampling stages because selection excludes
validation outcomes. They are reused holdouts, not 500 additional independent
observations; never pool first- and second-pass validation counts. Preserve
all flags and apply final family adjustment to the selected coordinate versions.

`contract.json` retains the original 47-entry `hashes` map exactly. Additional
driver/protocol/selection/runtime hashes live in separately audited
`sampling_hashes`. `sampling_refinement` has type
`independent_poisson_mixture_refinement`, version 1, the baseline contract SHA,
policy, attempt and selection identity. Per-point metadata repeats baseline
and source checkpoint identities and frozen plan hashes. Legacy `ntoy` and
`ntoys_per_proposal` are null; `ntoys_per_proposal_by_truth` and every
truth/method result give the actual integer counts. The count must equal
`provenance[truth].n / len(provenance[truth].meta.labels)`. `nvalidation` stays
500. Compatibility means identical inference plus an explicitly audited
sampling derivative, not falsely identical complete sampling contracts.

Each coordinate retains result JSON, proposal/scan plans, readiness checks,
numerical/scalar ledgers, approximation fallbacks, memory decision, validation
CSV/toy ledger, and any failure/deferred record. Final collection selects the
declared latest whole-coordinate version regardless of whether its limit rose
or fell. Never pool limits, p-values or toy counts across attempts.
